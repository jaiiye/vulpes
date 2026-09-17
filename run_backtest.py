#!/usr/bin/env python3
"""Backtest CLI.

    python run_backtest.py                          # 90 days, BTC, with benchmarks
    python run_backtest.py --days 180 --symbol ETH
    python run_backtest.py --random-runs 50
    python run_backtest.py --no-benchmark           # faster, strategy only

Read the limitation banner in the output before drawing any conclusion from a
result: the highest-weighted factor cannot be replayed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.config import ConfigError, load_config  # noqa: E402
from backtest.benchmarks import (  # noqa: E402
    buy_and_hold,
    random_entry_runs,
    result_return_pct,
)
from backtest.data import BacktestDataError, HistoricalLoader  # noqa: E402
from backtest.engine import Backtester  # noqa: E402
from backtest.metrics import compute_metrics  # noqa: E402

DEFAULT_CONFIG = "bots/fox_btc.yaml"

BANNER = """
================================================================================
 BACKTEST SCOPE
 This replays the live factor code, discipline gates, sizing and exits against
 real historical candles, funding and fees.

 NOT INCLUDED: the smart money factor (40% of the live weight). The public
 Hyperliquid API exposes no historical whale positions, so it cannot be
 replayed, and the alignment gate is disabled for the run (otherwise it would
 block every trade). Treat results as testing the remaining ~60% of the model,
 not the deployed strategy.
================================================================================
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Backtest the Fox agent on historical Hyperliquid data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--symbol", default=None, help="override the config symbol")
    p.add_argument("--days", type=int, default=90, help="traded window")
    p.add_argument(
        "--warmup-days",
        type=int,
        default=30,
        help="extra history fetched so indicators are warm at the first bar",
    )
    p.add_argument("--equity", type=float, default=1000.0, help="starting equity")
    p.add_argument(
        "--fee-bps", type=float, default=3.5, help="taker fee in bps, applied both legs"
    )
    p.add_argument(
        "--random-runs",
        type=int,
        default=20,
        help="random-entry simulations for the significance test",
    )
    p.add_argument("--no-benchmark", action="store_true", help="skip benchmarks")
    p.add_argument("--cache-dir", default="backtest_cache")
    p.add_argument("--quiet", action="store_true", help="suppress progress output")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.symbol:
        config.symbol = args.symbol.upper()
    symbol = config.symbol

    print(BANNER)
    print(f"config      : {config.summary()}")
    print(f"window      : {args.days} days (+{args.warmup_days} warmup)")
    print(f"initial     : ${args.equity:,.2f}   taker fee {args.fee_bps}bp/leg")
    print()

    intervals = tuple(
        {config.indicators.entry_timeframe, config.indicators.trend_timeframe}
    )

    def progress(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    loader = HistoricalLoader(cache_dir=args.cache_dir)

    # BTC is always loaded: the macro filter reads its higher timeframe trend.
    symbols = {symbol}
    if symbol != "BTC":
        symbols.add("BTC")

    datasets = {}
    for sym in sorted(symbols):
        progress(f"loading {sym}...")
        try:
            datasets[sym] = loader.load(
                sym,
                intervals=intervals,
                days=args.days,
                warmup_days=args.warmup_days,
                progress=progress,
            )
        except BacktestDataError as exc:
            print(f"data error: {exc}", file=sys.stderr)
            return 1
        for warning in datasets[sym].warnings:
            print(f"  NOTE: {warning}")
    print()

    # Surface data-coverage caveats alongside the strategy caveats.
    data_warnings = [
        f"{sym}: {w}" for sym, ds in datasets.items() for w in ds.warnings
    ]
    dataset_warnings = list(data_warnings)

    # --- Strategy -----------------------------------------------------
    progress("running strategy...")
    started = time.time()
    backtester = Backtester(
        config,
        datasets,
        initial_equity=args.equity,
        taker_fee_bps=args.fee_bps,
    )
    result = backtester.run(progress=progress)
    metrics = compute_metrics(
        result,
        bar_interval_hours=_interval_hours(config.indicators.entry_timeframe),
    )

    print(f"=== STRATEGY ({result.symbol}) ===")
    print(metrics.report())
    print(f"  runtime       {time.time() - started:.1f}s")
    print()

    period_days = (result.end_ms - result.start_ms) / 86_400_000
    print(f"  period        {period_days:.0f} days, "
          f"{len(result.equity_curve)} bars")
    if result.blocked:
        print("  gate rejections:")
        for reason, count in sorted(result.blocked.items(), key=lambda kv: -kv[1]):
            print(f"    {count:5d}  {reason}")
    print()

    # --- Warnings ------------------------------------------------------
    all_warnings = result.warnings + dataset_warnings
    if all_warnings:
        print("=== CAVEATS ===")
        for warning in all_warnings:
            print(f"  - {warning}")
        print()

    # --- Benchmarks ----------------------------------------------------
    if not args.no_benchmark:
        print("=== BENCHMARKS ===")
        bh = buy_and_hold(datasets[symbol], config.indicators.entry_timeframe)
        print(f"  buy & hold    {bh:+.2f}%")

        progress(f"running {args.random_runs} random-entry simulations...")
        bench = random_entry_runs(
            config,
            datasets,
            initial_equity=args.equity,
            taker_fee_bps=args.fee_bps,
            runs=args.random_runs,
        )
        print(f"  {bench.summary()}")

        pct = bench.percentile_of(metrics.total_return_pct)
        print()
        print(f"  strategy {metrics.total_return_pct:+.2f}% sits at the "
              f"{pct:.0f}th percentile of random entries")

        if pct < 75:
            print(
                "  VERDICT: the strategy does not beat random entries with any "
                "confidence. The direction call carries no demonstrable edge."
            )
        elif pct < 90:
            print(
                "  VERDICT: weak evidence at best. Re-run with more random "
                "samples and a longer window before trusting this."
            )
        else:
            print(
                "  VERDICT: beats random entries in this sample, but the smart "
                "money factor is still untested and the sample may be small."
            )

        if metrics.trades < 30:
            print(
                f"  NOTE: only {metrics.trades} trades, so the percentile above "
                "rests on very few observations."
            )
        print()

    return 0


def _interval_hours(interval: str) -> float:
    from agent.market_data import TIMEFRAME_MS

    return TIMEFRAME_MS.get(interval, 3_600_000) / 3_600_000


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130)
