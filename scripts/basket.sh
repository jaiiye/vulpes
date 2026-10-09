#!/usr/bin/env bash
#
# Scheduled basket rebalance. One invocation = one period.
#
# The period length comes from the strategy, not from here: hold=18 on 4h bars
# is 72h, so the cron entry fires every 3 days. See basket.cron for why that is
# a fixed wall-clock time and what that costs.
#
# Three things this wrapper is for, none of which run_basket.py does:
#
#   1. It is not silent on failure. A traceback from a cron job goes to mail,
#      and MAILTO="" has already discarded that mail. Here every pass leaves a
#      line in logs/basket.log whether it worked or not.
#   2. It never stacks up behind itself: flock -n, same as collect.sh.
#   3. It records whether this period was dry-run or live. run_basket.py prints
#      the mode, but a background run nobody read has no mode, and "did this
#      period place orders" is the first question asked after the fact.
#
# Why the full console output is kept in basket_last.out instead of filtering
# it down to a few grep patterns (the way collect.sh reads its journal): the
# basket writes no journal events, so the console text is the ONLY record of a
# period. Its wording is prose that gets reworded, and a line dropping out of
# the log because a sentence changed is the one failure this log cannot afford.
# The grep patterns below are a convenience on top of that file, not the record.
#
# Usage:
#   scripts/basket.sh              one period (dry-run unless BASKET_LIVE=1)
#   scripts/basket.sh --summary    show the persisted basket, write nothing
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

PYTHON="${PYTHON:-/usr/bin/python3}"
LOG_DIR="$ROOT/logs"
RUN_LOG="$LOG_DIR/basket.log"
LAST_OUT="$LOG_DIR/basket_last.out"
LOCK_FILE="$LOG_DIR/.basket.lock"
STATE="${BASKET_STATE:-logs/basket_state.json}"
JOURNAL="${BASKET_JOURNAL:-logs/basket_journal.jsonl}"
MAX_LOG_BYTES="${MAX_LOG_BYTES:-1048576}"   # 1 MB. One period per pass, so slow.

mkdir -p "$LOG_DIR"

log() {
    printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >>"$RUN_LOG"
}

# --------------------------------------------------------------------------
# Summary mode: report the persisted basket, write nothing.
#
# The field names are not assumed. Only `basket_ts_ms` is known to exist
# (run_basket.py keys its duplicate-period guard on it); everything else is
# reported as its type so a renamed field shows up as a changed type rather
# than as a missing value.
# --------------------------------------------------------------------------
if [[ "${1:-}" == "--summary" ]]; then
    echo "=== persisted basket ($STATE) ==="
    "$PYTHON" - "$STATE" <<'PY'
import json
import sys
import time

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        state = json.load(handle)
except FileNotFoundError:
    print("  no basket state yet")
    raise SystemExit(0)

ts = state.get("basket_ts_ms")
if ts:
    age_h = (time.time() * 1000 - ts) / 3_600_000
    built = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts / 1000))
    print(f"  built {built} ({age_h:.1f}h ago)")
else:
    print("  no basket_ts_ms -- no period has completed yet")

for key in sorted(state):
    value = state[key]
    if isinstance(value, list):
        print(f"  {key}: {len(value)} items")
    elif isinstance(value, dict):
        print(f"  {key}: dict[{len(value)}]")
    elif key != "basket_ts_ms":
        print(f"  {key}: {value}")
PY
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
# Exclusive, non-blocking lock. A slow period must not stack up behind itself.
# --------------------------------------------------------------------------
exec 9>"$LOCK_FILE" || exit 1
if ! flock -n 9; then
    log "SKIP: a previous basket run is still in progress"
    exit 0
fi

export PYTHONUNBUFFERED=1
started=$(date +%s)

# --------------------------------------------------------------------------
# Real orders require BASKET_LIVE=1.
#
# run_basket.py already defaults to dry-run; passing --live here would invert
# that for every scheduled run, and a scheduled run is by definition the one
# nobody is watching. The mode is printed by the script and logged below, so a
# period that quietly stayed in dry-run is visible after the fact.
# --------------------------------------------------------------------------
live_flag=""
if [[ "${BASKET_LIVE:-0}" == "1" ]]; then
    live_flag="--live"
fi

# Dry-run has no account to read equity from, so it takes a notional. 200k is
# the figure the backtests are quoted against.
export DRY_RUN_EQUITY_USD="${DRY_RUN_EQUITY_USD:-200000}"

out=$("$PYTHON" run_basket.py --state "$STATE" --journal "$JOURNAL" $live_flag 2>&1)
rc=$?
printf '%s\n' "$out" >"$LAST_OUT"

# --------------------------------------------------------------------------
# One line per pass, plus one ALERT line per thing worth waking up for.
#
# The mode header is read back rather than inferred from $live_flag: it is the
# script's own statement of what it did, and it is the one line that must not
# be reconstructed from our own variables.
# --------------------------------------------------------------------------
header=$(printf '%s' "$out" | grep -E '^=== 篮子' | head -1 | tr -s ' ')

if (( rc == 0 )); then
    log "period OK: ${header:-no header line}"
else
    log "period FAILED (rc=$rc): $(printf '%s' "$out" | tail -5 | tr '\n' '|')"
fi

for pattern in '^权益' '^池 ' '^计划:' '^毛敞口' '^执行 ' '^已保存'; do
    line=$(printf '%s' "$out" | grep -E "$pattern" | tail -1)
    [[ -n "$line" ]] && log "  $line"
done

# These four are the ones that matter. "无篮子" is not an error (a thin pool is
# a normal outcome) but it is the difference between "held" and "did nothing",
# so it is logged at the same level as the real alarms.
for pattern in '风控拒绝' '持久化失败' '腿不平衡' '^无篮子'; do
    line=$(printf '%s' "$out" | grep -E "$pattern" | tail -1)
    [[ -n "$line" ]] && log "ALERT: $line"
done

elapsed=$(( $(date +%s) - started ))
log "=== done in ${elapsed}s (rc=$rc) ==="

# Propagate the failure so a monitoring hook can see it even though cron mail
# is off. The log file remains the source of truth.
exit "$rc"
