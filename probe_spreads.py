#!/usr/bin/env python3
"""Sample live L2 books to calibrate what an order actually costs.

Why this exists
---------------
Every backtest in this repo charges a flat 3.5 bp per leg. That number is the
exchange's taker fee and nothing else - it omits the spread you cross, and it
cannot say what happens when an order is large enough to walk the book. Both
omissions point the same way, so a result that only survives under 3.5 bp is
not evidence of anything.

The price-based alternative was tried and is not good enough. Roll's estimator
(`backtest.microstructure.roll_spread`) attributes *all* negative lag-1
covariance to the bid-ask bounce, and this archive has as much negative
covariance at lag 24 as at lag 1, so Roll read 30-38 bp where the live book
shows about 3 bp. Quoting that as a cost overstated it by an order of magnitude
- in the pessimistic direction, which is just as misleading as the optimistic
one when it is used to reject a finding.

So: ask the exchange.

What it cannot tell you
-----------------------
The books are sampled **now**, not during the backtest window. For this archive
that is a gap of months. Spreads move with volatility and with the venue's own
liquidity, so treat the output as a calibration of *magnitude* - "a few bp, not
tens of bp" - rather than as the cost on any particular historical bar. A
per-bar spread series would need a historical book feed, which is not in the
archive.

Usage
-----
    python probe_spreads.py                    # the researched pool, both halves
    python probe_spreads.py --sizes 1000 10000 # specific order sizes
    python probe_spreads.py --top 40           # a shorter sample
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time

from agent.market_data import HyperliquidMarket, MarketDataError
from backtest import microstructure as ms
from backtest import universe as U
from backtest.cross_section import load_archive_panel, liquid_split

#: The window every research result in RESEARCH.md is measured on.
START_MS = 1_772_064_000_000
END_MS = 1_789_689_600_000
BARS = 4_896
COVERAGE = 0.95

POOL_SQL = """
SELECT coin, COUNT(DISTINCT epoch_ms(time_bucket(INTERVAL '1 hour', timestamp))) AS hrs
FROM read_parquet('data/canonical/candles/*.parquet')
WHERE timestamp >= to_timestamp({start}) AND timestamp < to_timestamp({end})
GROUP BY coin HAVING hrs >= {bars} * {cov}
ORDER BY coin;
"""


def archive_pool() -> tuple[set[str], set[str]]:
    """The liquid and illiquid halves of the researched universe."""
    sql = POOL_SQL.format(start=START_MS // 1000, end=END_MS // 1000,
                          bars=BARS, cov=COVERAGE)
    proc = subprocess.run(["duckdb", "-json"], input=sql,
                          capture_output=True, text=True, timeout=900)
    coins = U.tradeable_names([r["coin"] for r in json.loads(proc.stdout or "[]")])
    panel = load_archive_panel(coins, interval_minutes=60,
                               start_ms=START_MS, end_ms=END_MS)
    from backtest.cross_section import restrict
    panel = restrict(panel, set(U.filter_by_coverage(panel, COVERAGE)))
    return liquid_split(panel, 0.5)


def sample(market: HyperliquidMarket, symbols: list[str], sizes: list[float],
           depth: int, pause: float) -> list[dict]:
    live = set(market.asset_names())
    out: list[dict] = []
    for sym in symbols:
        if sym not in live:
            continue
        try:
            book = market.order_book(sym, depth=depth)
        except (MarketDataError, OSError, ValueError):
            continue
        levels = book.get("levels") or []
        if len(levels) < 2 or not levels[0] or not levels[1]:
            continue
        bids = [(float(x["px"]), float(x["sz"])) for x in levels[0]]
        asks = [(float(x["px"]), float(x["sz"])) for x in levels[1]]
        mid = (bids[0][0] + asks[0][0]) / 2.0
        row = {
            "symbol": sym,
            "mid": mid,
            "top_bps": ms.top_of_book_spread_bps(bids, asks),
            "depth_50bp": ms.depth_within_bps(asks, mid, 50.0),
            "depth_100bp": ms.depth_within_bps(asks, mid, 100.0),
        }
        for size in sizes:
            # Stored as the round trip; halved when reported, because a
            # strategy pays one side on entry and one on exit and quoting the
            # round trip as "the cost" would double it against the fee.
            row[f"rt_{int(size)}"] = ms.round_trip_cost_bps(bids, asks, size)
        out.append(row)
        time.sleep(pause)
    return out


def median(values) -> float:
    kept = [v for v in values if v is not None]
    return statistics.median(kept) if kept else float("nan")


def report(label: str, rows: list[dict], sizes: list[float], fee_bps: float) -> None:
    if not rows:
        print(f"  {label}: no books retrieved")
        return
    print(f"  {label}  (n={len(rows)})")
    print(f"    quoted spread at touch   {median([r['top_bps'] for r in rows]):>8.2f} bp")
    for size in sizes:
        key = f"rt_{int(size)}"
        fillable = sum(1 for r in rows if r[key] is not None)
        half = median([r[key] for r in rows]) / 2.0
        # The median over fillable books is not comparable to a row where
        # everything filled: it is taken over whichever names happen to be deep
        # enough, which are the cheap ones. Reporting it bare would make the
        # largest size look like the cheapest - an earlier version of this
        # script did exactly that ($100k at 0.55 bp, off 3 surviving names).
        share = fillable / len(rows)
        if not fillable:
            flag = "  <- nothing could fill this size"
        elif share >= 0.9:
            flag = ""
        elif share < 0.5:
            flag = "  <- biased: median is over the deep names only"
        else:
            flag = "  <- partial coverage"
        shown = f"{half:>8.2f}" if fillable else "     n/a"
        print(f"    ${int(size):<7,} one-way exec  {shown} bp"
              f"   + fee {fee_bps:.1f} = "
              f"{(f'{half + fee_bps:>6.2f}' if fillable else '   n/a')} bp/leg"
              f"   (fillable {fillable}/{len(rows)}){flag}")
    print(f"    depth within 50 bp       ${median([r['depth_50bp'] for r in rows]):>10,.0f}")
    print(f"    depth within 100 bp      ${median([r['depth_100bp'] for r in rows]):>10,.0f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=float, nargs="+", default=[1000.0, 10_000.0, 100_000.0],
                    help="order notionals to price, in USD")
    ap.add_argument("--depth", type=int, default=20, help="book levels to request")
    ap.add_argument("--top", type=int, default=0, help="limit to the first N symbols per half")
    ap.add_argument("--pause", type=float, default=0.02, help="seconds between requests")
    ap.add_argument("--fee-bps", type=float, default=3.5, help="taker fee per leg")
    args = ap.parse_args()

    liquid, illiquid = archive_pool()
    market = HyperliquidMarket(testnet=False)

    for label, symbols in (("liquid half", liquid), ("illiquid half", illiquid)):
        ordered = sorted(symbols)
        if args.top:
            ordered = ordered[:args.top]
        rows = sample(market, ordered, args.sizes, args.depth, args.pause)
        report(label, rows, args.sizes, args.fee_bps)
        print()

    print("  Note: books are sampled now, not during the backtest window. Use the")
    print("  magnitude, not the exact number - see this script's docstring.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
