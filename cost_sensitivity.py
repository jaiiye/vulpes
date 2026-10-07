#!/usr/bin/env python3
"""Where the mean-reversion edge crosses zero, at the cost actually paid.

Why this exists
---------------
`scan_exits.py` charges a flat `FEE_BPS = 7.2`, annotated "measured, $1k order
size (see probe_spreads.py)". That number is the *sum of two different things*:

    taker fee 3.5 bp  +  $1k order walking the book 3.67 bp  =  7.2 bp/leg

and both halves are now measurable directly rather than assumed:

* **The fee**, from 1.6e8 real fills (`fee / (price * size)`, USDC-denominated
  in every row): taker median **3.2 bp**, maker median **-0.1 bp** (a rebate).
  3.5 was close; it is also not what the number should be if the bot posts.
* **The walk**, from the 282-day orderbook archive: at **$100** the book does
  not move at all. BTC/ETH/SOL cost 0.07 / 0.24 / 0.07 bp one-way, and the same
  books priced at $1k and $10k give the same number to three decimals - the
  order is four orders of magnitude below the depth, so `round_trip == quoted
  spread` and the one-way cost is half of it.

So 7.2 bp describes an order size this bot does not trade. This script re-runs
the *same rule on the same pool with the same windows* with the cost as the
only free parameter, and reports where the net return crosses zero. The answer
is a fee threshold, which is the form the question actually takes: "is the edge
worth more than the round trip costs?"

What it does not tell you
-------------------------
* It inherits `scan_exits.py`'s pool, which is the lower half by liquidity, not
  the Q1-Q4 quintiles quoted in RESEARCH.md 24. Those quintiles came from a
  one-off script that is not in the repository (`liquidity_quintiles` is only
  referenced by its own tests), so the two pools are close but not provably
  identical. The *slope* - which is what this measures - does not depend on
  which of the two is used.
* A crossing point is a statement about the sample, not about the future. It is
  reported with the median and the mean because RESEARCH.md 24 shows they
  differ (2.5 vs ~4 bp), and a conclusion that flips between them is not one.
* Maker pricing is not free. Posting earns the rebate but pays adverse
  selection, which no column in `fills` records. The maker row below is a
  *floor* on the cost, not a cost.

Usage
-----
    python cost_sensitivity.py                      # the default fee ladder
    python cost_sensitivity.py --fees 1.5 3.2 5.0 7.2
    python cost_sensitivity.py --windows 1          # one window, fast check
"""

from __future__ import annotations

import argparse
import statistics
import sys

from backtest import combine as cb
from backtest import universe as U
from backtest.cross_section import Panel, restrict
from backtest.mean_reversion import MeanReversionConfig, simulate_mean_reversion
from scan_exits import load, illiquid_of

#: The ladder. 7.2 is what the repository has been using; 3.2 is the measured
#: taker fee alone; 5.0 is the fee plus a $100 walk priced off the probe table
#: (3.04 bp quoted spread / 2 + 3.2); 1.4 is the maker side.
DEFAULT_FEES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.2, 5.0, 7.2)

#: The repository's annotation for where the edge is worth collecting from.
BREAKEVEN_RANGE = (2.5, 4.0)
DEFAULT_EXIT = 50.0

#: RESEARCH.md 24 reports on "Q1-Q4" while `scan_exits.py` uses the lower half.
#: They are not the same pool - 80% of the universe against 50% - so the ladder
#: is runnable on either and `--pool` picks which. The default is the quintile
#: form because that is the one the published numbers use.
POOLS = ("q14", "half")


def _panel(index: dict, t0: int, lo: int, hi: int) -> Panel:
    """The 1h panel for one window, built from the shared index."""
    close = {s: [(d[i]["c"] if i in d else None) for i in range(lo, hi)]
             for s, d in index.items()}
    vol = {s: [(d[i]["v"] if i in d else None) for i in range(lo, hi)]
           for s, d in index.items()}
    return Panel(times=[t0 + i * 3_600_000 for i in range(lo, hi)],
                 symbols=sorted(index), close=close, volume=vol)


def quintiles_of(by: dict, index: dict, t0: int, lo: int, hi: int,
                 buckets: int = 5, keep: int = 4) -> list[str]:
    """The `keep` least-liquid quintiles of one window, merged.

    Q1-Q4 rather than all five because Q5 is the tier the effect is *absent*
    from (RESEARCH.md 23), so including it would dilute the measurement with
    symbols the rule is not claimed to work on.
    """
    panel = _panel(index, t0, lo, hi)
    panel = restrict(panel, set(U.filter_by_coverage(panel, 0.95)))
    if len(panel.symbols) < 20:
        return []
    tiers = U.liquidity_quintiles(panel, buckets)
    return sorted({s for tier in tiers[:keep] for s in tier})


def collect(by: dict, index: dict, t0: int, nbars: int, windows: int,
            kind: str = "q14") -> list[tuple[int, int, list[str]]]:
    """The pool for each window, resolved once and reused across fees.

    Resolving per fee would repeat `liquid_split`, which is the expensive half,
    and - worse - would allow a different pool at each fee, so the ladder would
    no longer be one measurement with one free parameter.
    """
    if kind not in POOLS:
        raise ValueError(f"unknown pool {kind!r}; expected one of {POOLS}")
    spans = cb.equal_windows(nbars, windows)
    pools = []
    for lo, hi in spans:
        symbols = (illiquid_of(by, index, t0, lo, hi) if kind == "half"
                   else quintiles_of(by, index, t0, lo, hi))
        if symbols:
            pools.append((lo, hi, symbols))
    return pools


def measure(pools: list[tuple[int, int, list[str]]], index: dict,
            fee_bps: float, exit_level: float) -> dict:
    """Net return per symbol for one fee, under both pooling conventions.

    Both are reported because the repository uses both and they disagree.
    `median` / `mean` pool every symbol across the six windows, which weights
    a window by how many symbols qualified in it - this is the convention
    behind RESEARCH.md 24's "median net return" table, and it reproduces that
    table exactly. `cw_median` / `cw_mean` take a median per window first and
    then summarise the six, weighting windows equally - the convention behind
    that section's "net return by cost" table.

    They differ by about 0.7pp at 7.2 bp, which moves the crossing point by
    roughly 2 bp. Reporting one alone would hide the choice rather than make
    it, and the choice is what decides whether 5 bp is above or below break-
    even.
    """
    cfg = MeanReversionConfig(rule="rsi", fee_bps=fee_bps,
                              exit_level=exit_level)
    values: list[float] = []
    per_window: list[float] = []
    trades = 0
    for lo, hi, symbols in pools:
        window: list[float] = []
        for s in symbols:
            d = index[s]
            rows = [d[i] for i in range(lo, hi) if i in d]
            if len(rows) < (hi - lo) * 0.95:
                continue
            r = simulate_mean_reversion(
                [x["c"] for x in rows], [x["o"] for x in rows],
                [x["h"] for x in rows], [x["l"] for x in rows], cfg)
            if not r.trades:
                continue
            window.append(r.net_return_pct)
            trades += len(r.trades)
        if window:
            per_window.append(statistics.median(window))
        values.extend(window)
    if not values:
        nan = float("nan")
        return {"n": 0, "fee": fee_bps, "median": nan, "mean": nan,
                "cw_median": nan, "cw_mean": nan, "positive": nan,
                "trades": 0}
    return {
        "fee": fee_bps,
        "n": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "cw_median": statistics.median(per_window),
        "cw_mean": statistics.fmean(per_window),
        "positive": sum(1 for v in values if v > 0) / len(values),
        "trades": trades,
    }


def crossing(rows: list[dict], key: str) -> float | None:
    """The fee at which `key` crosses zero, linearly interpolated.

    A bracket is required: extrapolating a line through one negative and no
    positive point would invent a threshold the data never reached.

    A point sitting exactly on zero counts as the answer rather than as a
    bracket with a degenerate one - `(x < 0)` is False for both signs, so a
    zero would otherwise be read as "no sign change" and skipped.
    """
    for a, b in zip(rows, rows[1:]):
        if a[key] == 0:
            return a["fee"]
        if b[key] == 0:
            return b["fee"]
        if (a[key] < 0) != (b[key] < 0):
            span = b[key] - a[key]
            if span == 0:
                continue
            return a["fee"] + (0 - a[key]) * (b["fee"] - a["fee"]) / span
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fees", type=float, nargs="+", default=list(DEFAULT_FEES))
    ap.add_argument("--windows", type=int, default=6)
    ap.add_argument("--exit-level", type=float, default=DEFAULT_EXIT)
    ap.add_argument("--pool", choices=POOLS, default="q14",
                    help="Q1-Q4 (the pool RESEARCH.md 24 reports on) or the "
                         "lower half (the pool scan_exits.py scans)")
    args = ap.parse_args()

    by, t0, nbars = load()
    index = {s: {int((r["t"] - t0) // 3_600_000): r for r in rows}
             for s, rows in by.items()}

    pools = collect(by, index, t0, nbars, args.windows, args.pool)
    if not pools:
        print("no usable pool; nothing measured", file=sys.stderr)
        return 1
    sizes = sorted({len(s) for _, _, s in pools})
    label = "Q1-Q4 by liquidity" if args.pool == "q14" else "lower half by liquidity"
    print(f"archive: {len(by)} perp symbols, {nbars} 1h bars "
          f"({nbars / 24:.0f} days)")
    print(f"pool: {label}, {args.windows} window(s), "
          f"{sizes[0]}-{sizes[-1]} symbols each, exit_level={args.exit_level:g}")
    print()
    print(f"  {'fee/leg':>8s} {'symbols':>8s} {'trades':>8s} "
          f"{'pooled':>18s} {'cross-window':>18s} {'>0':>7s}")
    print(f"  {'':>8s} {'':>8s} {'':>8s} "
          f"{'median':>9s} {'mean':>8s} {'median':>9s} {'mean':>8s}")
    rows = []
    for fee in args.fees:
        r = measure(pools, index, fee, args.exit_level)
        rows.append(r)
        print(f"  {fee:>8.2f} {r['n']:>8d} {r['trades']:>8d} "
              f"{r['median']:>8.2f}% {r['mean']:>7.2f}% "
              f"{r['cw_median']:>8.2f}% {r['cw_mean']:>7.2f}% "
              f"{r['positive']:>6.1%}")

    print()
    labels = (("median", "pooled median"),
              ("mean", "pooled mean"),
              ("cw_median", "cross-window median"),
              ("cw_mean", "cross-window mean"))
    for key, text in labels:
        x = crossing(rows, key)
        if x is None:
            print(f"  {text:<22s} no zero crossing inside the ladder")
            continue
        lo, hi = BREAKEVEN_RANGE
        verdict = ("above the researched 2.5-4 bp band"
                   if x > hi else
                   "inside the researched 2.5-4 bp band"
                   if x >= lo else
                   "below the researched 2.5-4 bp band")
        print(f"  {text:<22s} crosses zero at {x:>5.2f} bp/leg  ({verdict})")
    print()
    print("  Costs to compare against, measured rather than assumed:")
    print("    3.2  bp  taker fee alone            (1.6e8 real fills)")
    print("   -0.1  bp  maker fee alone            (a rebate)")
    print("    5.0  bp  taker fee + a $100 walk    (3.63/2 bp, Q1-Q4 touch)")
    print("    1.7  bp  maker fee + a $100 walk    (floor: adverse selection)")
    print("    7.2  bp  what the repository charges ($1k walk + assumed 3.5 fee)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
