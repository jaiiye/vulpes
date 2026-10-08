#!/usr/bin/env python3
"""Does the cross-sectional reversal pay for its own turnover?

Why this exists
---------------
It is the one effect in this archive that is still standing. The directional
timing effect is real but worth 2.54-5.46 bp/leg against a cost of 2.1-7.2 bp
(RESEARCH 26, 27) - it does not clear. The cross-sectional reversal is the
other candidate: IC -0.067, t = -4.4 to -6.1, and unlike everything else it has
not gone away under repeated testing.

But an IC is not a return. README says so plainly: *"我上面没有算成本 —— 这是结论里
最大的空白"*. The basket engine charges a fee, yet the study that found the
effect never ran it with one, so nobody knows whether a basket that rebalances
every day earns back what it spends. That is the first of the three gates the
effect has to pass, and it is the only one of the three that this archive can
settle.

So: same signal, same pool, same windows, **cost as the only free parameter**.
The answer is a fee threshold in bp/leg - the form the question actually takes.

How the cost is charged here, and why it is not the directional number
---------------------------------------------------------------------
A period costs `turnover * 2 legs * fee_bps`. The basket pays only for the names
it *replaces*, so a fee that looks survivable here is not the same number as in
`cost_sensitivity.py`, which pays on every position opened and closed. Both are
"bp/leg"; they are not interchangeable, and mixing them would understate the
basket by roughly a factor of `1 / turnover`.

What it cannot tell you
-----------------------
* **Survivorship is bracketed, not fixed.** The pool is "whoever is trading at
  row `REF_ROW`" with no coverage requirement - a coverage filter *is* the bias,
  because a delisted coin cannot have full coverage and a reversal basket's long
  leg is the leg that would have held it - and a name that dies mid-period is
  booked at zero (`late_exit_factor = 0.0`). That is the conservative side of
  the bracket. Coins that list *after* `REF_ROW` are still absent, and that one
  points the other way.
* **The third gate is architecture.** This needs a long-short basket; the live
  bot is single-position directional. Passing the cost gate buys the right to
  consider that rewrite, nothing more.
* The pool is the less liquid half, where the effect was found. It is also the
  half with the widest spreads and, per RESEARCH 27, the largest adverse
  selection (1-3 bp/leg on thin coins, against 0.11 bp on BTC). The ladder below
  is what decides whether that matters.

Usage
-----
    python cross_section_cost.py                      # 24h signal, 24h hold, 6 windows
    python cross_section_cost.py --fees 0 2.29 5.3 7.2
    python cross_section_cost.py --windows 3 --runs 8  # a fast check
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest import combine as cb  # noqa: E402
from backtest import universe as U  # noqa: E402
from backtest.cross_section import (  # noqa: E402
    CrossSectionBacktester,
    CrossSectionConfig,
    Panel,
    liquid_split,
    load_archive_panel,
    negate,
    past_return,
    random_signal,
    restrict,
    slice_rows,
)
from cost_sensitivity import crossing  # noqa: E402

#: One row per measured cost, not per assumption. 2.29 / 2.55 are the maker
#: legs after adverse selection (RESEARCH 27); 3.2 is the measured taker fee
#: from 1.6e8 fills; 4.2 and 5.3 are the walk plus the optimistic and
#: self-paid fee tiers; 7.2 is what the repository has been charging.
DEFAULT_FEES = (0.0, 2.29, 2.55, 3.2, 4.2, 5.3, 7.2)

#: 4h bars. 6 rows = 24h, which is the horizon the IC table reports on.
DEFAULT_LOOKBACK = 6
DEFAULT_HOLD = 6
DEFAULT_WINDOWS = 6
DEFAULT_RUNS = 20
DEFAULT_INTERVAL = 240
DEFAULT_ARCHIVE = "data/canonical/candles"

#: (one-way cost of walking the book in bp, books that could fill / sampled).
#: Measured on live L2, illiquid half, n=40 names (`probe_spreads.py --sizes`
#: 100 250 500 1000 2000 5000 10000 --fee-bps 3.2, 2026-10-08), fee excluded.
#:
#: This is what turns size into the free parameter and cost into a measurement
#: instead of an assumption. Note the last row: at $10k only 32 of 40 books
#: could fill at all, so that cost is already a median over the deep names -
#: the true cost of trading the illiquid half at $10k is worse than 15.22 bp,
#: and part of the pool simply cannot be traded there.
WALK_BP = {
    100: (2.95, (40, 40)),
    250: (3.59, (40, 40)),
    500: (4.28, (40, 40)),
    1_000: (5.23, (40, 40)),
    2_000: (6.09, (40, 40)),
    5_000: (10.44, (39, 40)),
    10_000: (15.22, (32, 40)),
}

#: Median taker fee, measured over 1.6e8 real fills (RESEARCH 26) - not the
#: worst tier the repository used to assume.
TAKER_FEE_BP = 3.2

#: A basket taking more than this share of a name's daily dollar volume is
#: moving the price it is trying to harvest. 5% is the conventional ceiling.
PARTICIPATION = 0.05

#: The pool is "every symbol trading at row `REF_ROW`", which is how the
#: three-gate study in README/RESEARCH 18 defines it: 171 symbols, no coverage
#: requirement, no lookahead. It deliberately does NOT use the 95%-coverage
#: filter that `scan_exits` and `cost_sensitivity` use, because that filter is
#: exactly the survivorship bias the second gate is about - a coin that is
#: delisted cannot have 95% coverage, and a reversal basket's long leg is the
#: leg that would have held it.
REF_ROW = 60


def archive_coins(archive: str = DEFAULT_ARCHIVE) -> list[str]:
    """Every perp in the archive. Spot pairs are `@N`-prefixed and excluded."""
    import json
    import subprocess

    sql = (
        "SELECT DISTINCT coin FROM read_parquet("
        f"'{archive}/*.parquet') AS x("
        "coin, timestamp, open, high, low, close, volume, filename) "
        "WHERE coin NOT LIKE '@%' ORDER BY 1;"
    )
    proc = subprocess.run(["duckdb", "-json"], input=sql,
                          capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise SystemExit(f"listing coins failed: {(proc.stderr or '').strip()[:300]}")
    return [str(r["coin"]) for r in json.loads(proc.stdout or "[]")]


def window_panels(panel: Panel, windows: int) -> list[Panel]:
    """Disjoint windows, each restricted to its own illiquid half.

    The split is recomputed per window rather than taken once, because a
    symbol's liquidity tier is not a property of the symbol - a coin that is
    thin in one window may be busy in the next, and freezing one split would
    make the basket hold names the effect was never claimed to work on.
    """
    # The pool is fixed once, on the whole panel, before any window is cut.
    # It used to be a 95%-coverage filter applied per window, which made the
    # pool a function of how many windows the run was split into: at 12 windows
    # a symbol needed 393 of 414 bars, at 24 it needed 98 of 207, so the two
    # runs held different universes and reported gross returns a factor of two
    # apart (35 bp against 70 bp per period) for the same signal. A measurement
    # whose subject changes with its own resolution cannot be compared across
    # resolutions. It also imported a survivorship bias the study it is
    # checking had taken care to avoid.
    #
    # Liquidity, by contrast, *is* recomputed per window - a symbol's tier is a
    # property of the window, not of the symbol.
    row = min(REF_ROW, len(panel) - 1)
    panel = restrict(panel, set(panel.present(row)))
    out: list[Panel] = []
    for lo, hi in cb.equal_windows(len(panel), windows):
        sub = slice_rows(panel, lo, hi)
        try:
            _, illiquid = liquid_split(sub)
        except ValueError:
            continue
        if len(illiquid) < 20:
            continue
        out.append(restrict(sub, illiquid))
    return out


def measure(panels: list[Panel], lookback: int, hold: int,
            fee_bps: float) -> dict:
    """One fee across every window: the basket's net return in each.

   Three aggregations, because RESEARCH 24's lesson is that they disagree and
    the disagreement moved the answer by a factor of two. `cw_*` take one
    number per window and then summarise the windows; `period_median` looks at
    a typical rebalancing period instead, which is the unit the cost is charged
    in and so the one that makes the crossing interpretable.
    """
    cfg = CrossSectionConfig(fee_bps=fee_bps)
    per_window: list[float] = []
    period_net: list[float] = []
    turnover: list[float] = []
    period_gross: list[float] = []
    for p in panels:
        res = CrossSectionBacktester(
            p, negate(past_return(lookback)), hold, cfg
        ).run()
        if not res.periods:
            continue
        per_window.append(res.net_return_pct)
        period_net.extend(x.net_return for x in res.periods)
        period_gross.extend(x.long_return - x.short_return for x in res.periods)
        turnover.extend(x.turnover for x in res.periods)
    if not per_window:
        nan = float("nan")
        return {"fee": fee_bps, "n_windows": 0, "n_periods": 0,
                "per_window": [], "cw_median": nan, "cw_mean": nan,
                "period_median": nan, "turnover": nan, "gross_bp": nan}
    return {
        "fee": fee_bps,
        "per_window": per_window,
        "n_windows": len(per_window),
        "n_periods": len(period_net),
        "cw_median": statistics.median(per_window),
        "cw_mean": statistics.fmean(per_window),
        "period_median": statistics.median(period_net) * 100.0,
        "turnover": statistics.fmean(turnover),
        "gross_bp": statistics.fmean(period_gross) * 10_000.0,
        "_gross": sorted(period_gross),
        "_net": sorted(period_net),
    }


def spread(vals: list[float]) -> str:
    """min / p05 / p25 / median / p75 / p95 / max, as bp.

    Reported because the mean and the median of these periods disagree by two
    orders of magnitude on the current archive, and a single summary would pick
    one of them and hide that something is wrong with the other.
    """
    if not vals:
        return "no periods"
    def q(f: float) -> float:
        i = min(len(vals) - 1, max(0, int(f * (len(vals) - 1))))
        return vals[i] * 10_000.0
    return "  ".join(f"{q(f):>9.1f}" for f in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0))


def random_control(panels: list[Panel], hold: int, fee_bps: float,
                  runs: int, seed: int) -> list[float]:
    """Net return per window from ranking at random, same cost, same hold.

    The control the repository accepts as the only real evidence: a long-short
    basket in a falling market earns from its short leg no matter what the
    signal says, so a net return means nothing until it is placed against this.
    """
    cfg = CrossSectionConfig(fee_bps=fee_bps)
    out: list[float] = []
    for i, p in enumerate(panels):
        vals = []
        for r in range(runs):
            bt = CrossSectionBacktester(p, random_signal(seed + 1000 * i + r),
                                        hold, cfg)
            res = bt.run()
            if res.periods:
                vals.append(res.net_return_pct)
        out.append(statistics.median(vals) if vals else float("nan"))
    return out


def capacity_rows(panels: list[Panel], lookback: int, hold: int,
                  interval: int, per_leg: int) -> tuple[list[dict], float]:
    """Where size, not cost, is what stops this basket.

    Two independent limits, reported side by side because the question is
    which one binds first:

    * **cost** - the measured walk for that notional (`WALK_BP`) plus the
      measured fee, fed through the same basket as the fee ladder;
    * **participation** - the order as a share of a name's daily dollar
      volume. Past a few percent of it the basket is moving the price it is
      trying to harvest, and no amount of gross return pays for that, because
      the walk cost measured on a resting book no longer describes the fill.
    """
    bars_per_day = 24 * 60.0 / interval
    daily: list[float] = []
    for p in panels:
        # Median dollar volume per bar: a listing spike must not decide how
        # much a name can absorb.
        for v in U.liquidity_medians(p).values():
            daily.append(v * bars_per_day)
    med_daily = statistics.median(daily)

    rows = []
    for size, (walk, (ok, tot)) in sorted(WALK_BP.items()):
        fee = walk + TAKER_FEE_BP
        r = measure(panels, lookback, hold, fee)
        r["size"] = size
        r["walk"] = walk
        r["fee_total"] = fee
        r["participation"] = size / med_daily
        r["notional"] = size * per_leg * 2
        r["fillable"] = f"{ok}/{tot}"
        rows.append(r)
    return rows, med_daily


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fees", type=float, nargs="+", default=list(DEFAULT_FEES))
    ap.add_argument("--windows", type=int, default=DEFAULT_WINDOWS)
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    ap.add_argument("--hold", type=int, default=DEFAULT_HOLD)
    ap.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                    help="random baskets per window for the control")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    ap.add_argument("--archive", default=DEFAULT_ARCHIVE)
    ap.add_argument("--diagnose", action="store_true",
                    help="print the per-period distribution instead of a summary")
    ap.add_argument("--capacity", action="store_true",
                    help="sweep order size: measured walk cost and share of "
                         "daily volume, against the net return they leave")
    args = ap.parse_args()

    coins = archive_coins(args.archive)
    if not coins:
        print("no coins in the archive", file=sys.stderr)
        return 1
    panel = load_archive_panel(coins, interval_minutes=args.interval)
    panels = window_panels(panel, args.windows)
    if not panels:
        print("no usable window; nothing measured", file=sys.stderr)
        return 1

    print(f"archive: {len(panel.symbols)} perp symbols, {len(panel)} "
          f"{args.interval}m bars ({len(panel) * args.interval / 60 / 24:.0f} days)")
    sizes = sorted(len(p.symbols) for p in panels)
    legs = sorted(max(1, int(len(p.symbols) * CrossSectionConfig().quantile))
                  for p in panels)
    print(f"signal: negated {args.lookback * args.interval / 60:.0f}h return, "
          f"hold {args.hold * args.interval / 60:.0f}h, "
          f"{len(panels)} windows, illiquid half of each, "
          f"dead names worth {CrossSectionConfig().late_exit_factor:g}")
    print(f"pool: {sizes[0]}-{sizes[-1]} illiquid symbols per window, "
          f"{legs[0]}-{legs[-1]} per leg; turnover is charged on the names "
          f"a period replaces")
    if args.diagnose:
        print("  per-period gross, bp   "
              f"{'min':>9s} {'p05':>9s} {'p25':>9s} {'med':>9s} "
              f"{'p75':>9s} {'p95':>9s} {'max':>9s}")
        for fee in args.fees:
            r = measure(panels, args.lookback, args.hold, fee)
            print(f"  fee {fee:>6.2f}  gross  {spread(r['_gross'])}")
            print(f"  {'':>11s}  net    {spread(r['_net'])}")
        return 0

    if args.capacity:
        rows, med_daily = capacity_rows(panels, args.lookback, args.hold,
                                       args.interval, legs[-1])
        print()
        print(f"  median daily dollar volume per name  ${med_daily:,.0f}")
        print(f"  gross notional at {legs[-1]} names per leg, "
              f"both legs: {2 * legs[-1]} orders")
        print()
        print(f"  {'per name':>9s} {'walk':>6s} {'tot':>6s} {'cw med':>8s} "
              f"{'period med':>11s} {'of daily':>9s} {'notional':>11s} "
              f" {'books':>7s}")
        for r in rows:
            over = " <- over 5%" if r["participation"] > PARTICIPATION else ""
            print(f"  ${int(r['size']):>8,} {r['walk']:>6.2f} "
                  f"{r['fee_total']:>6.2f} {r['cw_median']:>8.2f} "
                  f"{r['period_median']:>11.3f} "
                  f"{r['participation'] * 100:>8.2f}% "
                  f"${r['notional']:>10,.0f}  {r['fillable']:>7s}{over}")
        cap = PARTICIPATION * med_daily
        print()
        print(f"  participation hits {PARTICIPATION:.0%} at "
              f"${cap:,.0f} per name -> ${cap * 2 * legs[-1]:,.0f} notional")
        print("  net return is still positive at every size measured here, so "
              "what stops this")
        print("  basket is how much the names can absorb, not what they cost "
              "to trade")
        return 0

    print()
    print(f"  {'fee/leg':>8s} {'per-window net %':^38s} "
          f"{'cw med':>8s} {'cw mean':>8s} {'period med':>11s} "
          f"{'turn':>6s} {'gross bp':>9s}")
    rows = []
    for fee in args.fees:
        r = measure(panels, args.lookback, args.hold, fee)
        rows.append(r)
        wins = "  ".join(f"{v:+.1f}" for v in r["per_window"]) \
            if "per_window" in r else ""
        print(f"  {fee:>8.2f} {wins:>38s} "
              f"{r['cw_median']:>8.2f} {r['cw_mean']:>8.2f} "
              f"{r['period_median']:>11.3f} "
              f"{r['turnover']:>6.2f} {r['gross_bp']:>9.2f}")

    print()
    for key, text in (("cw_median", "cross-window median"),
                      ("cw_mean", "cross-window mean"),
                      ("period_median", "median period")):
        x = crossing(rows, key)
        if x is None:
            print(f"  {text:<20s} no zero crossing inside the ladder")
            continue
        print(f"  {text:<20s} crosses zero at {x:>6.2f} bp/leg")

    print()
    print("  against a random ranking, same cost, same hold:")
    for fee in args.fees:
        ctrl = random_control(panels, args.hold, fee, args.runs, args.seed)
        strat = next(r for r in rows if r["fee"] == fee)
        beats = sum(1 for w, c in zip(strat.get("per_window", []), ctrl)
                    if w > c)
        n = min(len(strat.get("per_window", [])), len(ctrl))
        p = cb.sign_test(beats, n) if n else float("nan")
        print(f"    {fee:>6.2f} bp  beats the control in {beats}/{n} windows "
              f"(P = {p:.3f})   control median "
              f"{statistics.median(ctrl):+.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
