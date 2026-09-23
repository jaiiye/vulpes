#!/usr/bin/env bash
#
# Scheduled data collection for the Hyperliquid agent.
#
# Two independent jobs run under one lock:
#
#   1. record_snapshots.py     whale positions. The public API has no
#                              historical positions endpoint, so recording
#                              forward is the ONLY route to ever backtesting
#                              the smart-money factor.
#   2. run_bot.py --cycles 1   one dry-run decision cycle, appended to the
#                              journal as a live signal record.
#
# Each invocation is a fresh process that resumes any open position from the
# persisted state file, so the accumulated journal reads as one continuous
# session rather than a series of unrelated snapshots.
#
# Concurrency: the lock makes an overlapping invocation exit immediately.
# Without it, two runs would both pull the 37 MB leaderboard and both append
# to the same JSONL and state file, corrupting the position record.
#
# Usage:
#   scripts/collect.sh              one collection pass
#   scripts/collect.sh --summary    show accumulated coverage and exit
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

PYTHON="${PYTHON:-/usr/bin/python3}"
SYMBOLS="${SYMBOLS:-BTC,ETH}"
CONFIG="${CONFIG:-bots/fox_btc.yaml}"

LOG_DIR="$ROOT/logs"
RUN_LOG="$LOG_DIR/collect.log"
LOCK_FILE="$LOG_DIR/.collect.lock"
JOURNAL="$LOG_DIR/journal.jsonl"
SNAP_FILE="$ROOT/snapshots/whale_positions.jsonl"
MAX_LOG_BYTES="${MAX_LOG_BYTES:-5242880}"   # rotate at 5 MB

mkdir -p "$LOG_DIR"

log() {
    printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >>"$RUN_LOG"
}

# --------------------------------------------------------------------------
# Summary mode: report what has accumulated, write nothing.
# --------------------------------------------------------------------------
if [[ "${1:-}" == "--summary" ]]; then
    echo "=== whale snapshots ==="
    "$PYTHON" record_snapshots.py --summary

    echo
    echo "=== decision journal ==="
    if [[ -s "$JOURNAL" ]]; then
        "$PYTHON" - "$JOURNAL" <<'PY'
import collections, json, sys, time

kinds = collections.Counter()
first = last = None
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        continue
    kinds[rec.get("kind", "?")] += 1
    ts = rec.get("ts")
    if ts:
        first = ts if first is None else min(first, ts)
        last = ts if last is None else max(last, ts)

total = sum(kinds.values())
print(f"  {total} events")
if first and last:
    print(f"  span {(last - first) / 86400:.1f} days")
for kind, n in kinds.most_common():
    print(f"    {kind:22s} {n}")
PY
    else
        echo "  no journal yet"
    fi

    echo
    echo "=== signal quality ==="
    # Exits non-zero when there is nothing scoreable yet, which is the normal
    # state early on; it must not abort the rest of the summary.
    "$PYTHON" analyze_signals.py --journal "$JOURNAL" || true

    exit 0
fi

# --------------------------------------------------------------------------
# Rotate the run log before appending.
# --------------------------------------------------------------------------
if [[ -f "$RUN_LOG" ]]; then
    size=$(stat -c%s "$RUN_LOG" 2>/dev/null || echo 0)
    if (( size > MAX_LOG_BYTES )); then
        mv -f "$RUN_LOG" "$RUN_LOG.1"
    fi
fi

# --------------------------------------------------------------------------
# Exclusive, non-blocking lock. A slow run must not stack up behind itself.
# --------------------------------------------------------------------------
exec 9>"$LOCK_FILE" || exit 1
if ! flock -n 9; then
    log "SKIP: a previous collection run is still in progress"
    exit 0
fi

export PYTHONUNBUFFERED=1
started=$(date +%s)
failures=0
log "=== start (symbols=$SYMBOLS config=$CONFIG) ==="

# --------------------------------------------------------------------------
# 1. Whale snapshots. Failure here must not stop the decision cycle.
# --------------------------------------------------------------------------
out=$("$PYTHON" record_snapshots.py --symbols "$SYMBOLS" 2>&1)
rc=$?
if (( rc == 0 )); then
    log "snapshots OK: $out"
else
    failures=$((failures + 1))
    log "snapshots FAILED (rc=$rc): $(printf '%s' "$out" | tail -5 | tr '\n' '|')"
fi

# --------------------------------------------------------------------------
# 2. One dry-run decision cycle.
#
# The console output is reduced to one summary line per pass, but trades are
# read from the JOURNAL rather than scraped out of that output. The old filter
# (`grep -E 'factors:|cycle'`) silently dropped `ENTER`, `CLOSED` and every
# rejection reason - a background run could open and close a position and the
# log would not mention it. Reading the journal fixes that for a structural
# reason rather than a better pattern: the wording of a console line is prose
# that gets reworded, and a trade falling out of the log because a sentence
# changed is the one failure this log cannot afford.
# --------------------------------------------------------------------------
journal_lines_before=$(if [[ -s "$JOURNAL" ]]; then wc -l <"$JOURNAL"; else echo 0; fi)

# `--journal` is passed explicitly rather than relying on the bot's default:
# this pass reads the journal back to report the trades it produced, so the two
# have to be the same file even if that default ever changes.
out=$("$PYTHON" run_bot.py --config "$CONFIG" --journal "$JOURNAL" --cycles 1 2>&1)
rc=$?
if (( rc == 0 )); then
    # `tail -1` each, not `tail -3` on a combined pattern: the cycle summary is
    # printed twice (once through the logger, once at the end of main), and the
    # combined pattern recorded both, so every pass read as duplicated.
    factors=$(printf '%s' "$out" | grep -E 'factors:' | tail -1)
    cycles=$(printf '%s' "$out" | grep -E 'cycles \|' | tail -1)
    # The start-up check and anything it complained about. Without this the
    # filters above drop it, and it is the one line that says whether this
    # account can place a trade at all - a run that cannot trade looks exactly
    # like a run that found no signal.
    flight=$(printf '%s' "$out" | grep -iE 'preflight' | head -1 | sed 's/^.*\] *//')
    trouble=$(printf '%s' "$out" | grep -iE 'PREFLIGHT FAILED|preflight warning' | tail -1 | sed 's/^.*\] *//')
    log "cycle OK: ${factors:-no factor line} | ${cycles:-no summary}"
    if [[ -n "$flight" || -n "$trouble" ]]; then
        log "preflight: ${flight:-not reported}${trouble:+ | $trouble}"
    fi
else
    failures=$((failures + 1))
    log "cycle FAILED (rc=$rc): $(printf '%s' "$out" | tail -5 | tr '\n' '|')"
fi

# --------------------------------------------------------------------------
# 3. Anything this pass actually did, one line per event.
#
# `position_opened` / `position_closed` are the trade-level events and exist in
# both modes; `close_simulated` adds the cost breakdown that only a simulated
# close has. The failure kinds are included because they are rare and are the
# ones worth waking up for.
# --------------------------------------------------------------------------
trades=$("$PYTHON" - "$JOURNAL" "$journal_lines_before" <<'PY'
import json
import sys

path, before = sys.argv[1], int(sys.argv[2])
try:
    with open(path, encoding="utf-8") as handle:
        lines = handle.read().splitlines()
except FileNotFoundError:
    raise SystemExit(0)


def signed(value) -> str:
    # "n/a", not "$?" - that sequence already means "last exit code" to anyone
    # reading a shell log, and it would be read as a value rather than as an
    # absence.
    return "n/a" if value is None else f"${value:+.4f}"


def number(value) -> str:
    """Six significant figures, which is what these fields carry.

    Not `str(value)`: a stop of `84075.18462613407` is correct and unreadable,
    and this log is read by eye.
    """
    try:
        return f"{float(value):,.6g}"
    except (TypeError, ValueError):
        return "?"


def field(rec: dict, key: str) -> str:
    return str(rec.get(key, "?"))


for line in lines[before:]:
    try:
        rec = json.loads(line)
    except json.JSONDecodeError:
        continue
    kind = rec.get("kind")
    if kind == "position_opened":
        pos = rec.get("position") or {}
        print(
            f"OPEN {field(pos, 'side')} {number(pos.get('size'))} "
            f"{field(pos, 'symbol')} @ {number(pos.get('entry_price'))} "
            f"notional {number(pos.get('notional'))} stop {number(pos.get('stop_price'))} "
            f"score {number(rec.get('signal_score'))}"
        )
    elif kind == "position_closed":
        print(
            f"CLOSE {field(rec, 'side')} {field(rec, 'symbol')} {signed(rec.get('pnl'))} "
            f"({field(rec, 'reason')}) running {signed(rec.get('realised_pnl_total'))}"
        )
    elif kind == "close_simulated":
        # `$?` on an event written before the fee fields existed: unknown, and
        # reported as unknown rather than as zero.
        print(
            f"COST entry {signed(rec.get('entry_fee_usd'))} "
            f"+ exit {signed(rec.get('fee_usd'))} -> net {signed(rec.get('pnl_net'))} "
            f"(gross {signed(rec.get('pnl'))}, {field(rec, 'fee_bps')}bp/leg)"
        )
    elif kind in ("order_unfilled", "unwound_unprotected"):
        print(f"{kind.upper()}: {rec.get('note') or rec.get('symbol')}")
    elif kind in ("critical", "error"):
        # Truncated: these are transport errors and their text runs long.
        print(f"{kind.upper()}: {str(rec.get('message'))[:160]}")
PY
)
if [[ -n "$trades" ]]; then
    while IFS= read -r line; do
        log "trade: $line"
    done <<<"$trades"
fi

elapsed=$(( $(date +%s) - started ))
log "=== done in ${elapsed}s (failures=$failures) ==="

# Non-zero exit so a monitoring hook can see failures even though cron mail is
# disabled; the log file remains the source of truth.
exit "$failures"
