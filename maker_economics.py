#!/usr/bin/env python3
"""What a resting order actually costs: adverse selection, and whether it fills.

Why this exists
---------------
RESEARCH.md 26 ends on a number it cannot defend. It prices a maker leg at
**1.7 bp** - half-spread walk (1.82) minus the maker rebate (-0.099) - and then
says so itself: that is a *lower bound*, because it assumes the quote fills and
that filling is free. `fills` records no resting orders, so nothing in the
archive states either the fill rate or what happens after a fill. 1.7 bp is the
whole case for "the maker convention works", and it was never measured.

Measuring it needs a reference price that is **not** the fill price. A maker buy
prints at the bid, so "future price minus fill price" mechanically credits the
captured half-spread: run that way the archive reports maker fills as *favourable*
by +0.5 bp (data/markout7.log), which is the half-spread, not an edge, and would
be double-counted if it were also claimed as spread capture. The reference has to
be the mid at the instant of the fill.

No book in the archive can provide it: `canonical/orderbook` is 1-minute and
covers BTC/ETH/SOL only, far too coarse for a markout. So the mid is rebuilt from
the fills themselves - maker buys print at the bid and maker sells at the ask, so

    mid_t = (last maker-buy price + last maker-sell price) / 2

on 1-second buckets, both legs required to be inside `LAG` seconds and to cross
cleanly (`ask > bid`). Reported quantity, in bp:

    buy  : (mid_{t+W} - mid_t) / mid_t
    sell : -(mid_{t+W} - mid_t) / mid_t

Negative is adverse selection. The two sides carry the market drift with opposite
signs, so **(buy + sell) / 2 is drift-neutral** - it is the adverse selection on
its own, with no view on where the market went.

Fill rate is the other half and cannot be read off either: with no order-level
data, post a buy at `mid * (1 - d)` and ask whether the mid ever trades down to
it within T seconds. That is a function of how far from mid you quote and how
long you wait, which is exactly the trade a market maker is making.

What it cannot tell you
-----------------------
Queue position. `fills` says nothing about where in the book a quote sat, so
every fill rate here is an **upper bound**: in reality a quote that the price
touches may still not fill ahead of the orders already resting there.

Usage
-----
    python maker_economics.py                       # adverse selection, one day
    python maker_economics.py --by-coin             # is it a thin-coin artefact
    python maker_economics.py --days 6              # cross-window + sign test
    python maker_economics.py --fill-rate           # the other half
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.combine import sign_test  # noqa: E402

DEFAULT_FILLS = "data/fills_wide/fills/date={day}.parquet"
DEFAULT_WINDOWS = (1, 5, 30, 120, 600)
DEFAULT_LAG = 5
DEFAULT_TS = (10, 30, 60, 300)
DEFAULT_DELTAS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
# Spread across the archive so a cross-window run is six independent samples.
DEFAULT_DAYS = (
    "2025-08-15",
    "2025-10-20",
    "2026-01-15",
    "2026-03-10",
    "2026-06-15",
    "2026-09-10",
)
MAKER_REBATE_BP = 0.099


def mid_cte(day: str, lag: int, fills: str = DEFAULT_FILLS) -> str:
    """The synthetic mid for one day, as a CTE ending in `midts(coin, s, mid)`.

    `lag` caps how stale the two legs may be when they are paired. It is a
    density knob, not a freshness one - a tighter cap leaves fewer seconds with a
    mid at all, so the mid every fill is later matched against is older. Tighten
    it far enough and the estimator degenerates (at `lag=0` it reads 0.000 on
    every horizon), which is why the default is 5 s and why `lag` is reported
    alongside every number.
    """
    return f"""raw AS (
  SELECT coin, epoch_ms(timestamp) AS ms, side,
         CAST(size AS DOUBLE) AS sz, CAST(price AS DOUBLE) AS px
  FROM read_parquet('{fills.format(day=day)}')
  WHERE NOT crossed AND sz > 0 AND px > 0
),
sec AS (
  SELECT coin, ms // 1000 AS s, side,
         sum(sz * px) / sum(sz) AS vwap, sum(sz) AS q
  FROM raw GROUP BY 1, 2, 3
),
bkt AS (SELECT coin, s, side, sum(vwap * q) / sum(q) AS px FROM sec GROUP BY 1, 2, 3),
bb AS (SELECT coin, s, px AS bid FROM bkt WHERE side = 'buy'),
aa AS (SELECT coin, s, px AS ask FROM bkt WHERE side = 'sell'),
m1 AS (
  SELECT bb.coin, bb.s, (bb.bid + x.ask) / 2 AS mid FROM bb
  ASOF JOIN aa x ON bb.coin = x.coin AND bb.s >= x.s
  WHERE x.ask > bb.bid AND (bb.s - x.s) <= {lag}
),
m2 AS (
  SELECT aa.coin, aa.s, (y.bid + aa.ask) / 2 AS mid FROM aa
  ASOF JOIN bb y ON aa.coin = y.coin AND aa.s >= y.s
  WHERE y.bid < aa.ask AND (aa.s - y.s) <= {lag}
),
midts AS (
  SELECT coin, s, avg(mid) AS mid
  FROM (SELECT * FROM m1 UNION ALL SELECT * FROM m2) GROUP BY 1, 2
)"""


def adverse_sql(
    day: str,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    lag: int = DEFAULT_LAG,
    by_coin: bool = False,
    fills: str = DEFAULT_FILLS,
) -> str:
    """One row per side (or per coin) of drift-neutral-able markout, in bp."""
    key = "fills.coin AS coin, fills.side AS side" if by_coin else "fills.side AS side"
    ctes = [
        f"j{w} AS (SELECT f.id, x.mid AS pn FROM fills f "
        f"ASOF JOIN midts x ON f.coin = x.coin AND (f.s + {w}) <= x.s "
        f"WHERE (x.s - f.s) <= {w} + 60)"
        for w in windows
    ]
    sel = [key, "count(*) AS n"] + [
        f"1e4 * avg(CASE WHEN fills.side = 'buy' THEN 1 ELSE -1 END "
        f"* (j{w}.pn - fills.m0) / fills.m0) AS mo{w}"
        for w in windows
    ]
    return (
        "WITH "
        + mid_cte(day, lag, fills)
        + ",\n"
        + """fills AS (
  SELECT row_number() OVER () AS id, bkt.coin, bkt.s, bkt.side, z.mid AS m0
  FROM bkt ASOF JOIN midts z ON bkt.coin = z.coin AND bkt.s >= z.s
  WHERE (bkt.s - z.s) <= """
        + str(lag)
        + "\n),\n"
        + ",\n".join(ctes)
        + "\nSELECT "
        + ", ".join(sel)
        + "\nFROM fills "
        + " ".join(f"JOIN j{w} ON fills.id = j{w}.id " for w in windows)
        + "\nGROUP BY "
        + ("1, 2" if by_coin else "1")
        + " ORDER BY "
        + ("3 DESC, 1" if by_coin else "1")
        + ";"
    )


def fillrate_sql(
    day: str,
    ts: tuple[int, ...] = DEFAULT_TS,
    deltas: tuple[float, ...] = DEFAULT_DELTAS,
    lag: int = DEFAULT_LAG,
    fills: str = DEFAULT_FILLS,
) -> str:
    """Fraction of postings at `mid * (1 - d)` that the mid reaches within T."""
    def tag(d: float) -> str:
        return str(d).replace(".", "p")

    frame = ",\n    ".join(
        f"min(mid) OVER (PARTITION BY coin ORDER BY s "
        f"ROWS BETWEEN 1 FOLLOWING AND {t} FOLLOWING) AS lo{t},\n    "
        f"max(mid) OVER (PARTITION BY coin ORDER BY s "
        f"ROWS BETWEEN 1 FOLLOWING AND {t} FOLLOWING) AS hi{t}"
        for t in ts
    )
    aggs = ",\n  ".join(
        f"avg(CASE WHEN lo{t} <= mid * (1 - {d / 1e4}) THEN 1.0 ELSE 0.0 END)"
        f" AS b{t}_{tag(d)},\n  "
        f"avg(CASE WHEN hi{t} >= mid * (1 + {d / 1e4}) THEN 1.0 ELSE 0.0 END)"
        f" AS s{t}_{tag(d)}"
        for t in ts
        for d in deltas
    )
    return (
        "WITH "
        + mid_cte(day, lag, fills)
        + ",\ngrid AS (\n"
        + """  SELECT g.coin, g.s, z.mid
  FROM (SELECT c.coin, u.s FROM (SELECT DISTINCT coin FROM midts) c,
        (SELECT unnest(generate_series(
            (SELECT min(s) FROM midts), (SELECT max(s) FROM midts))) AS s) u) g
  ASOF JOIN midts z ON g.coin = z.coin AND g.s >= z.s
  WHERE (g.s - z.s) <= 60
),
w AS (
  SELECT coin, s, mid,
    """
        + frame
        + """
  FROM grid
)
SELECT count(*) AS n,
  """
        + aggs
        + """
FROM w WHERE s % 30 = 0 AND mid IS NOT NULL;"""
    )


def run_sql(sql: str, timeout: int = 7200) -> list[dict]:
    """Run one statement through the duckdb CLI. Decimal columns come back as
    strings, so every consumer has to `float()` what it reads."""
    try:
        proc = subprocess.run(
            ["duckdb", "-json"], input=sql, capture_output=True, text=True, timeout=timeout
        )
    except FileNotFoundError as exc:
        raise SystemExit("duckdb was not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(f"duckdb timed out after {timeout}s") from exc
    if proc.returncode:
        raise SystemExit(f"duckdb failed:\n{proc.stderr[:800]}")
    try:
        return json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"unparseable duckdb output: {proc.stdout[:200]}") from exc


def drift_neutral(rows: list[dict], windows: tuple[int, ...]) -> dict[int, float]:
    """(buy + sell) / 2 per horizon - the drift-cancelling combination.

    Both sides see the same market drift with opposite signs, so their sum
    leaves the adverse selection and their difference leaves the drift. Averaging
    them without the sign convention would report the drift as if it were
    selection.
    """
    by = {r["side"]: {w: float(r[f"mo{w}"]) for w in windows} for r in rows}
    if "buy" not in by or "sell" not in by:
        return {}
    return {w: (by["buy"][w] + by["sell"][w]) / 2 for w in windows}


def median(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def net_maker_cost_bp(adverse_bp: float) -> float:
    """Cost of a maker leg once adverse selection is charged, in bp.

    RESEARCH.md 26 priced it at `1.82 - 0.099`. Adverse selection is reported
    **negative** by convention, so it enters with its sign flipped: adding it as
    written would subtract a cost from the bill.
    """
    return 1.82 - adverse_bp - MAKER_REBATE_BP


def report_adverse(rows: list[dict], windows: tuple[int, ...], label: str) -> None:
    print(f"\nadverse selection, {label} (bp; negative = adverse)")
    print(f"{'side':>6s} {'n':>10s} " + "  ".join(f"{w:>7d}s" for w in windows))
    for r in rows:
        print(
            f"{r['side']:>6s} {int(r['n']):>10,d} "
            + "  ".join(f"{float(r[f'mo{w}']):>8.3f}" for w in windows)
        )
    dn = drift_neutral(rows, windows)
    if not dn:
        print("  (one side missing - drift-neutral combination not available)")
        return
    by = {r["side"]: r for r in rows}
    print(
        f"{'NEUTRAL':>6s} {'':>10s} "
        + "  ".join(f"{dn[w]:>8.3f}" for w in windows)
        + "   <- (buy+sell)/2, drift cancelled"
    )
    print(
        f"{'drift':>6s} {'':>10s} "
        + "  ".join(
            f"{(float(by['buy'][f'mo{w}']) - float(by['sell'][f'mo{w}'])) / 2:>8.3f}"
            for w in windows
        )
        + "   <- (buy-sell)/2, the drift itself"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--day", default="2026-03-10")
    ap.add_argument("--days", type=int, default=0,
                    help="run this many spread-out days and sign-test them")
    ap.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    ap.add_argument("--lag", type=int, default=DEFAULT_LAG)
    ap.add_argument("--by-coin", action="store_true")
    ap.add_argument("--fill-rate", action="store_true")
    ap.add_argument("--ts", type=int, nargs="+", default=list(DEFAULT_TS))
    ap.add_argument("--deltas", type=float, nargs="+", default=list(DEFAULT_DELTAS))
    ap.add_argument("--fills", default=DEFAULT_FILLS)
    args = ap.parse_args(argv)

    windows = tuple(args.windows)

    if args.fill_rate:
        rows = run_sql(fillrate_sql(args.day, tuple(args.ts), tuple(args.deltas),
                                    args.lag, args.fills))
        x = rows[0]
        print(f"fill rate of a resting order, {args.day}  "
              f"(n = {int(x['n']):,} postings, queue position not modelled)")

        def tag(d: float) -> str:
            return str(d).replace(".", "p")

        for side, prefix in (("buy", "b"), ("sell", "s")):
            print(f"  {side} side, quoted {prefix}elow/above mid by:")
            for t in args.ts:
                line = f"    T={t:>4d}s  "
                for d in args.deltas:
                    line += f"{d:>5g}bp: {float(x[f'{prefix}{t}_{tag(d)}']) * 100:5.1f}%   "
                print(line)
        return 0

    if args.by_coin:
        rows = run_sql(adverse_sql(args.day, windows, args.lag, True, args.fills))
        print(f"drift-neutral adverse selection by coin, {args.day}, lag<={args.lag}s "
              f"(bp; negative = adverse)")
        print(f"{'coin':>8s} {'n':>9s} " + "  ".join(f"{w:>7d}s" for w in windows))
        for r in rows[:15]:
            print(
                f"{r['coin']:>8s} {int(r['n']):>9,d} "
                + "  ".join(f"{float(r[f'mo{w}']):>8.3f}" for w in windows)
            )
        col = [float(r[f"mo{windows[-1]}"]) for r in rows]
        neg = sum(1 for v in col if v < 0)
        print(f"\n{len(rows)} coins: median {median(col):.3f} bp at {windows[-1]}s, "
              f"{neg}/{len(rows)} negative")
        return 0

    if args.days > 1:
        daylist = list(DEFAULT_DAYS[: args.days])
        print("drift-neutral adverse selection per window (bp; negative = adverse)")
        print(f"{'day':>12s} " + "  ".join(f"{w:>7d}s" for w in windows))
        per: dict[int, list[float]] = {w: [] for w in windows}
        for d in daylist:
            rows = run_sql(adverse_sql(d, windows, args.lag, False, args.fills))
            dn = drift_neutral(rows, windows)
            if not dn:
                print(f"{d:>12s}  (incomplete)")
                continue
            for w in windows:
                per[w].append(dn[w])
            print(f"{d:>12s} " + "  ".join(f"{dn[w]:>8.3f}" for w in windows))
        print()
        for w in windows:
            vals = per[w]
            pos = sum(1 for v in vals if v > 0)
            print(
                f"  {w:>4d}s  median {median(vals):>7.3f} bp   "
                f"{pos}/{len(vals)} positive   "
                f"sign test P = {sign_test(pos, len(vals)):.4f}   "
                f"net maker leg {net_maker_cost_bp(median(vals)):.2f} bp"
            )
        return 0

    rows = run_sql(adverse_sql(args.day, windows, args.lag, False, args.fills))
    report_adverse(rows, windows, f"{args.day}, lag<={args.lag}s")
    dn = drift_neutral(rows, windows)
    if dn:
        print(f"\nmaker leg net of the {MAKER_REBATE_BP} bp rebate:")
        for w in windows:
            print(f"  {w:>4d}s horizon: {net_maker_cost_bp(dn[w]):.2f} bp/leg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
