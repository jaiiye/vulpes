#!/usr/bin/env python3
"""Scan the mean-reversion exit rule, and check the ranking across windows.

Why the ranking has to be checked across windows
------------------------------------------------
Run on a single 170-day window, this scan picks a clear winner: `exit_level=35`
scores 77-81% above the matched-exposure random 75th percentile, against about
53-56% for the default of 50. That looks like a large, free improvement, and the
number is reproducible - `--days 170` will print it.

Repeated on six non-overlapping windows it is the *worst* of the candidates -
four windows of six, sign-test P = 0.69, with individual windows at 11% and 14%.
The default (50) and its looser neighbours (60, 70) are above chance in all six.

(The single-window figure moves between about 77% and 90% depending on exactly
which 170 days are taken - the archive is 415 days, so "the last 170" is not the
same slice as the one in RESEARCH.md. The direction never changes: the tight
exit looks strong on any one window and fails across them.)

So the scan's answer is "no improvement", and the reason it exists is that the
single-window answer was confidently wrong. Both numbers are printed, in that
order, so the failure is visible rather than summarised away.

The metric is the matched-exposure percentile, not the return
------------------------------------------------------------
`mean_reversion`'s raw return carries the market's direction - measured across
six windows it ranges from -993% to +395% for a rule whose *timing* is
consistent. Ranking exits by raw return would mostly rank them by how long they
stay long, which is the exit's definition rather than its quality. The
percentile compares against random entries with the same count and the same
holding-length distribution, so the market factor cancels.

Usage
-----
    python scan_exits.py                      # scan and cross-window check
    python scan_exits.py --exits 35 50 70     # specific thresholds
    python scan_exits.py --windows 1          # the single-window view only
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone

from backtest import combine as cb
from backtest import universe as U
from backtest.cross_section import Panel, liquid_split, restrict
from backtest.mean_reversion import MeanReversionConfig, simulate_mean_reversion
from backtest.trend_gate import percentile_of, random_control

DEFAULT_EXITS = (35, 40, 45, 50, 55, 60, 70, 80)
FEE_BPS = 7.2          # measured, $1k order size (see probe_spreads.py)
CONTROL_RUNS = 20      # verified against 100 and 200; the answers agree
ENTRY_RSI = 30.0

OHLC_SQL = """
SELECT coin, epoch_ms(time_bucket(INTERVAL '1 hour', timestamp)) AS t,
       arg_min(open, timestamp)::DOUBLE AS o, max(high)::DOUBLE AS h,
       min(low)::DOUBLE AS l, arg_max(close, timestamp)::DOUBLE AS c,
       SUM(volume)::DOUBLE AS v
FROM read_parquet('data/canonical/candles/*.parquet')
{where}
GROUP BY 1,2 ORDER BY 2,1;
"""


def load() -> tuple[dict, int, int]:
    """(rows by coin, first bar ms, bar count) for the whole archive."""
    proc = subprocess.run(["duckdb", "-json"], input=OHLC_SQL.format(where=""),
                          capture_output=True, text=True, timeout=2400)
    by: dict[str, list[dict]] = {}
    for r in json.loads(proc.stdout or "[]"):
        by.setdefault(r["coin"], []).append(r)
    by = {s: by[s] for s in U.tradeable_names(by)}
    t0 = min(v[0]["t"] for v in by.values())
    t1 = max(v[-1]["t"] for v in by.values())
    return by, t0, int((t1 - t0) // 3_600_000) + 1


def illiquid_of(by: dict, index: dict, t0: int, lo: int, hi: int) -> list[str]:
    """The lower-liquidity half of one window, selected from that window alone.

    Per-window selection matters: the archive grows from 252 coins a day to 890,
    so a pool fixed on one window would be measuring a different universe in the
    other windows.
    """
    close = {s: [(d[i]["c"] if i in d else None) for i in range(lo, hi)]
             for s, d in index.items()}
    vol = {s: [(d[i]["v"] if i in d else None) for i in range(lo, hi)]
           for s, d in index.items()}
    panel = Panel(times=[t0 + i * 3_600_000 for i in range(lo, hi)],
                  symbols=sorted(index), close=close, volume=vol)
    panel = restrict(panel, set(U.filter_by_coverage(panel, 0.95)))
    if len(panel.symbols) < 20:
        return []
    _, illiquid = liquid_split(panel, 0.5)
    return sorted(illiquid)


def measure(symbols: list[str], index: dict, lo: int, hi: int,
            exit_level: float) -> tuple[int, int]:
    """(symbols above the control's 75th percentile, symbols measured)."""
    cfg = MeanReversionConfig(rule="rsi", fee_bps=FEE_BPS,
                              exit_level=exit_level)
    above = trials = 0
    for s in symbols:
        d = index[s]
        rows = [d[i] for i in range(lo, hi) if i in d]
        if len(rows) < (hi - lo) * 0.95:
            continue
        c = [x["c"] for x in rows]
        o = [x["o"] for x in rows]
        h = [x["h"] for x in rows]
        low = [x["l"] for x in rows]
        r = simulate_mean_reversion(c, o, h, low, cfg)
        holds = [t.bars for t in r.trades]
        if not holds:
            continue
        rnd = random_control(c, o, len(r.trades), holds, runs=CONTROL_RUNS,
                             seed=1000, fee_bps=FEE_BPS)
        if percentile_of(r.net_return_pct, rnd) >= 75:
            above += 1
        trials += 1
    return above, trials


def label(t0: int, lo: int, hi: int) -> str:
    f = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%y-%m-%d")
    return f"{f(t0 + lo * 3_600_000)}~{f(t0 + (hi - 1) * 3_600_000)[3:]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exits", type=float, nargs="+", default=list(DEFAULT_EXITS))
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--fee-bps", type=float, default=FEE_BPS)
    ap.add_argument("--days", type=int, default=0,
                    help="restrict to the last N days as a single window; "
                         "--days 170 reproduces the scan that picked exit=35")
    args = ap.parse_args()

    by, t0, nbars = load()
    index = {s: {int((r["t"] - t0) // 3_600_000): r for r in rows}
             for s, rows in by.items()}
    print(f"archive: {len(by)} perp symbols, {nbars} 1h bars "
          f"({nbars / 24:.0f} days), fee {args.fee_bps} bp/leg")
    print()

    if args.days:
        # One window, taken from the end - the same convention as `--as-of` in
        # `run_backtest.py`, so this reproduces the published scan rather than
        # approximating it.
        n = min(args.days * 24, nbars)
        spans = [(nbars - n, nbars)]
    elif args.windows == 1:
        spans = [(0, nbars)]
    else:
        spans = cb.equal_windows(nbars, args.windows)

    per_exit: dict[float, list[float]] = {e: [] for e in args.exits}
    header = "".join(f"{'exit=' + format(e, 'g'):>13s}" for e in args.exits)
    print(f"  {'window':>14s}{header}")
    for lo, hi in spans:
        symbols = illiquid_of(by, index, t0, lo, hi)
        if not symbols:
            continue
        cells = []
        for e in args.exits:
            above, trials = measure(symbols, index, lo, hi, e)
            frac = above / trials if trials else 0.0
            per_exit[e].append(frac)
            cells.append(f"{above:>4d}/{trials:<3d}({frac:>4.0%})")
        print(f"  {label(t0, lo, hi):>14s}" + "".join(f"{c:>13s}" for c in cells))

    if len(spans) > 1:
        print()
        print("  Across windows - this is the part that decides:")
        print(f"  {'exit':>6s} {'median':>8s} {'range':>20s} {'>25%':>8s} {'sign test':>10s}")
        ranked = []
        for e in args.exits:
            v = per_exit[e]
            if not v:
                continue
            k = sum(1 for x in v if x > 0.25)
            p = cb.sign_test(k, len(v))
            ranked.append((cb.sign_test(k, len(v)), e))
            print(f"  {e:>6g} {statistics.median(v):>7.1%} "
                  f"[{min(v):>7.1%}, {max(v):>7.1%}] {k:>4d}/{len(v):<4d} {p:>10.4f}")
        ranked.sort()
        print()
        best_p = ranked[0][0]
        winners = [e for p, e in ranked if p == best_p]
        print(f"  Most consistent: exit={', '.join(format(e, 'g') for e in winners)} "
              f"(P={best_p:.4f})")
        print("  The default is 50; only adopt another value if it is at least as")
        print("  consistent across windows, not if it merely wins on one window.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
