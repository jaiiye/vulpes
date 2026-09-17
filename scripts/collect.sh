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
# --------------------------------------------------------------------------
out=$("$PYTHON" run_bot.py --config "$CONFIG" --cycles 1 2>&1)
rc=$?
if (( rc == 0 )); then
    # The last few lines carry the signal verdict and the cycle summary.
    summary=$(printf '%s' "$out" | grep -E 'factors:|cycle' | tail -3 | tr '\n' '|')
    log "cycle OK: ${summary:-no signal}"
else
    failures=$((failures + 1))
    log "cycle FAILED (rc=$rc): $(printf '%s' "$out" | tail -5 | tr '\n' '|')"
fi

elapsed=$(( $(date +%s) - started ))
log "=== done in ${elapsed}s (failures=$failures) ==="

# Non-zero exit so a monitoring hook can see failures even though cron mail is
# disabled; the log file remains the source of truth.
exit "$failures"
