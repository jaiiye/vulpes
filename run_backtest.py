#!/usr/bin/env python3
"""Backtest CLI.

    python run_backtest.py                          # 90 days, BTC, with benchmarks
    python run_backtest.py --days 170 --symbol ETH
    python run_backtest.py --days 170 --symbols BTC,ETH,SOL   # pooled sample
    python run_backtest.py --random-runs 50
    python run_backtest.py --no-benchmark           # faster, strategy only

Read the limitation banner in the output before drawing any conclusion from a
result: the highest-weighted factor cannot be replayed.

Use `--symbols` to pool several markets into one sample. Sample size is the
binding constraint on every conclusion this tool can produce: the default
90-day single-market run yields about 14 trades, which the tool's own metrics
flag as statistically meaningless. Three markets over 170 days yield roughly
118, which is the smallest sample worth drawing an inference from here.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.config import ConfigError, load_config  # noqa: E402
from backtest.benchmarks import (  # noqa: E402
    buy_and_hold,
    random_entry_runs,
    result_return_pct,
)
from backtest.data import (  # noqa: E402
    DEFAULT_CANDLES_ARCHIVE,
    BacktestDataError,
    HistoricalLoader,
)
from backtest.engine import Backtester  # noqa: E402
from backtest.metrics import MIN_MEANINGFUL_TRADES, Metrics, compute_metrics  # noqa: E402
from backtest.leaderboard import LeaderboardError, reconstruct  # noqa: E402
from backtest.position_history import (  # noqa: E402
    DEFAULT_MAX_AGE_MS as SNAPSHOT_MAX_AGE_MS,
    DEFAULT_STORE as DEFAULT_POSITION_STORE,
    PositionHistoryError,
    load_leaderboard_history,
    load_position_history,
)

DEFAULT_CONFIG = "bots/fox_btc.yaml"

BANNER = """
================================================================================
BACKTEST SCOPE
This replays the live factor code, discipline gates, sizing and exits against
real historical candles, funding and fees.

{scope}
================================================================================
"""

#: Scope text for a run with no whale data. The factor is absent rather than
#: approximated, so the run exercises the rest of the model and says so.
SCOPE_NO_WHALES = """NOT INCLUDED: the smart money factor (40% of the live weight). The public
Hyperliquid API exposes no historical whale positions, so it cannot be
replayed, and the alignment gate is disabled for the run (otherwise it would
block every trade). Treat results as testing the remaining ~60% of the model,
not the deployed strategy."""

#: Scope text for a run backed by the Reservoir snapshots. The factor *is*
#: replayed here, so the old blanket "not included" was simply false - the same
#: class of untrue claim as the engine warning that used to announce a
#: confidence term was dropped while using it. This banner is printed before
#: any loading happens, so it states the intent and leaves the wallet basis to
#: the WHALE HISTORY block that follows.
SCOPE_WHALES = """INCLUDED: the smart money factor (40% of the live weight), replayed from the
Reservoir position snapshots with the alignment gate on, as live. Note the
wallet set is reconstructed, not read from the production leaderboard - the
WHALE HISTORY block below states the basis it actually used."""

#: Scope text for a leaderboard-backed run: the factor *and* the selection rule
#: are the production ones, so the remaining gaps are narrower and named.
SCOPE_LEADERBOARD = """INCLUDED: the smart money factor (40% of the live weight) with the alignment
gate on, and its wallet set rebuilt from the fills archive by trailing
realised PnL - the production selection rule. Remaining gaps: the archive
lags live by up to a day, and the account-value floor is omitted because that
column is not in the research projection."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Backtest the Fox agent on historical Hyperliquid data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--symbol", default=None, help="override the config symbol")
    p.add_argument(
        "--symbols",
        default=None,
        help="comma-separated markets to run and pool, e.g. BTC,ETH,SOL. "
        "Use this when a single market does not produce enough trades for the "
        "statistics to mean anything.",
    )
    p.add_argument(
        "--whale-snapshots",
        nargs="?",
        const=DEFAULT_POSITION_STORE,
        default=None,
        help="run the 40%% smart money factor from the Reservoir position "
        "snapshots in this directory (default: %(const)s). Off unless given. "
        "The wallet set is approximated by size and persistence rather than "
        "leaderboard profitability, so this tests the factor, not the exact "
        "production wallet set.",
    )
    p.add_argument(
        "--whale-leaderboard",
        action="store_true",
        help="with --whale-snapshots, choose wallets by a leaderboard "
        "reconstructed from the fills archive (realised PnL over "
        "day/week/month) instead of by position size. This is the production "
        "selection rule; the set then changes daily rather than being fixed.",
    )
    p.add_argument(
        "--whale-max-fills-per-day",
        type=float,
        default=None,
        help="exclude wallets averaging more fills than this per day from the "
        "reconstructed ranking. Production applies no such ceiling; it exists "
        "to test whether the factor is only reading market-maker inventory. "
        "Of the wallets entering the PnL top 60, 53%% trade >1000 times a day "
        "and 86%% more than 200.",
    )
    p.add_argument(
        "--entry-timeframe",
        default=None,
        help="override the config's entry timeframe, e.g. 15m. The trend "
        "timeframe is left alone, so this changes how often a signal is "
        "evaluated without changing what it is compared against. Needs "
        "--candles-archive for anything faster than 1h: the API caps at ~5000 "
        "bars, which is 52 days at 15m.",
    )
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
    p.add_argument(
        "--candles-archive",
        nargs="?",
        const=DEFAULT_CANDLES_ARCHIVE,
        default=None,
        help="build candles from the local one-second archive in this "
        "directory (default: %(const)s) instead of the public API. Off unless "
        "given. The API caps at ~5000 bars, which is 208 days at 1h and 52 at "
        "15m, so it cannot supply a second window or a faster timeframe; the "
        "archive holds 414 days at every interval. Funding still comes from "
        "the API - the archive has no funding history.",
    )
    p.add_argument(
        "--as-of",
        default=None,
        metavar="YYYY-MM-DD",
        help="pin the window to end at the close of this UTC day. Without it "
        "the window ends at the current time, so two runs on different days "
        "describe different periods and their numbers are not comparable.",
    )
    p.add_argument("--quiet", action="store_true", help="suppress progress output")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.entry_timeframe:
        # An override rather than a config copy: a duplicated config drifts away
        # from the real one the moment either is edited, and then the comparison
        # being reported is not the comparison that was run.
        from agent.market_data import TIMEFRAME_MS as _TF

        if args.entry_timeframe not in _TF:
            print(
                f"bad --entry-timeframe {args.entry_timeframe!r}; known: "
                f"{', '.join(_TF)}",
                file=sys.stderr,
            )
            return 2
        config = replace(
            config,
            indicators=replace(
                config.indicators, entry_timeframe=args.entry_timeframe
            ),
        )

    try:
        requested = _resolve_symbols(args, config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    as_of_ms = _as_of_ms(args.as_of)
    if args.as_of and as_of_ms is None:
        print(
            f"bad --as-of value {args.as_of!r}: expected YYYY-MM-DD",
            file=sys.stderr,
        )
        return 2

    if not args.whale_snapshots:
        scope = SCOPE_NO_WHALES
    elif args.whale_leaderboard:
        scope = SCOPE_LEADERBOARD
    else:
        scope = SCOPE_WHALES
    print(BANNER.format(scope=scope))
    print(f"config      : {config.summary()}")
    print(f"symbols     : {', '.join(requested)}")
    print(f"window      : {args.days} days (+{args.warmup_days} warmup)")
    if as_of_ms is not None:
        print(f"pinned to   : end of {args.as_of} UTC (reproducible)")
    else:
        print(
            "pinned to   : NOTHING - the window ends at the current time, so "
            "this run is not comparable with one made later"
        )
    print(f"initial     : ${args.equity:,.2f}   taker fee {args.fee_bps}bp/leg")
    print()

    intervals = tuple(
        {config.indicators.entry_timeframe, config.indicators.trend_timeframe}
    )

    def progress(msg: str) -> None:
        if not args.quiet:
            print(msg, flush=True)

    try:
        loader = HistoricalLoader(
            cache_dir=args.cache_dir,
            candles_archive=args.candles_archive,
        )
    except BacktestDataError as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return 1

    # BTC is always loaded: the macro filter reads its higher timeframe trend.
    to_load = sorted(set(requested) | {"BTC"})

    datasets = {}
    for sym in to_load:
        progress(f"loading {sym}...")
        try:
            datasets[sym] = loader.load(
                sym,
                intervals=intervals,
                days=args.days,
                warmup_days=args.warmup_days,
                progress=progress,
                end_ms=as_of_ms,
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

    # --- Whale history (optional) --------------------------------------
    whale_histories: dict[str, Any] = {}
    if args.whale_snapshots:
        print("=== WHALE HISTORY ===")
        board = None
        if args.whale_leaderboard:
            progress("reconstructing the leaderboard from fills...")
            try:
                board = reconstruct(
                    max_fills_per_day=args.whale_max_fills_per_day
                )
            except LeaderboardError as exc:
                print(f"data error: {exc}", file=sys.stderr)
                return 1
            print("  leaderboard:")
            for line in board.summary().splitlines():
                print(f"  {line}")
            for note in board.notes:
                print(f"    ! {note}")

        for sym in requested:
            dataset = datasets.get(sym)
            # Select wallets from data strictly before the traded window. Using
            # the window itself would pick the wallets that survived it, which
            # is the answer the backtest is supposed to be measuring.
            window_start_ms = (
                dataset.start_ms
                if dataset is not None
                else int(time.time() * 1000) - args.days * 86_400_000
            )
            try:
                if board is not None:
                    progress(f"loading positions for {sym} from the ranking...")
                    history = load_leaderboard_history(
                        board, market=sym, store=args.whale_snapshots
                    )
                else:
                    progress(
                        f"selecting whale wallets for {sym} from pre-window data..."
                    )
                    history = load_position_history(
                        store=args.whale_snapshots,
                        market=sym,
                        select_before_ms=window_start_ms,
                    )
            except PositionHistoryError as exc:
                print(f"data error: {exc}", file=sys.stderr)
                return 1

            whale_histories[sym] = history.series
            print(f"  {sym}:")
            for line in history.summary().splitlines():
                print(f"  {line}")
            for note in history.notes:
                print(f"    ! {note}")

            # The alignment gate blocks every trade while the factor has no
            # data, which is correct live behaviour and quietly misleading in a
            # backtest: a window the reconstruction does not cover reads as
            # "the strategy barely trades" rather than "the data stops here".
            if dataset is not None:
                covered = _whale_coverage_days(
                    history.series, dataset.start_ms, dataset.end_ms
                )
                window_days = (dataset.end_ms - dataset.start_ms) / 86_400_000
                missing = window_days - covered
                if missing > 5:
                    print(
                        f"    !! WHALE DATA COVERS {covered:.0f} OF "
                        f"{window_days:.0f} DAYS IN THIS WINDOW: the alignment "
                        "gate blocks every trade outside that span, so trade "
                        "counts below reflect missing data as well as the "
                        "strategy."
                    )
        print()

    # --- Strategy -----------------------------------------------------
    started = time.time()
    runs: list[tuple[str, Any, Metrics, Any]] = []

    for sym in requested:
        # A config per market. `Backtester` reads `config.symbol` when it is
        # constructed and `Synthesizer` falls back to it, so sharing one
        # mutable config would leak the previous market's symbol into the run.
        run_cfg = replace(config, symbol=sym)

        progress(f"running strategy on {sym}...")
        result = Backtester(
            run_cfg,
            datasets,
            initial_equity=args.equity,
            taker_fee_bps=args.fee_bps,
            whale_history=whale_histories.get(sym),
            # Daily snapshots, so the staleness window is measured in days.
            # The engine's default suits hourly funding records and would leave
            # the whale book empty for all but one bar in twenty-four.
            whale_max_age_ms=SNAPSHOT_MAX_AGE_MS if args.whale_snapshots else None,
        ).run(progress=progress)
        metrics = compute_metrics(
            result,
            bar_interval_hours=_interval_hours(run_cfg.indicators.entry_timeframe),
        )

        bench = None
        if not args.no_benchmark:
            progress(
                f"running {args.random_runs} random-entry simulations on {sym}..."
            )
            bench = random_entry_runs(
                run_cfg,
                datasets,
                initial_equity=args.equity,
                taker_fee_bps=args.fee_bps,
                runs=args.random_runs,
            )

        runs.append((sym, result, metrics, bench))

    runtime = time.time() - started
    print()

    if len(runs) == 1:
        _report_single_market(runs[0], datasets, config, dataset_warnings, runtime)
    else:
        _report_pooled(runs, datasets, config, dataset_warnings, runtime)

    return 0


def _whale_coverage_days(history: dict, start_ms: int, end_ms: int) -> float:
    """Days inside the traded window on which any whale position exists.

    Measured as the overlap with the window, not as the history's own span. The
    first version of this used the span - earliest point to latest - and so
    reported full coverage for a 169-day window whose whale data stopped on day
    69, because the history also contained earlier days outside the window. The
    warning it was written to raise therefore never fired, which is the exact
    failure mode it exists to prevent.

    Counting the window's own days also makes the archive's 52-day gap visible
    instead of being averaged away inside a span.
    """
    days = set()
    for series in history.values():
        for ts, _ in series.points:
            if start_ms <= ts <= end_ms:
                days.add(ts)
    return float(len(days))


def _print_verdict(percentile: float) -> None:
    if percentile < 75:
        print(
            "  VERDICT: the strategy does not beat random entries with any "
            "confidence. The direction call carries no demonstrable edge."
        )
    elif percentile < 90:
        print(
            "  VERDICT: weak evidence at best. Re-run with more random "
            "samples and a longer window before trusting this."
        )
    else:
        print(
            "  VERDICT: beats random entries in this sample, but the smart "
            "money factor is still untested and the sample may be small."
        )


def _report_single_market(run, datasets, config, dataset_warnings, runtime) -> None:
    """Full detail for one market. Unchanged from the original output."""
    sym, result, metrics, bench = run

    print(f"=== STRATEGY ({result.symbol}) ===")
    print(metrics.report())
    print(f"  runtime       {runtime:.1f}s")
    print()

    period_days = (result.end_ms - result.start_ms) / 86_400_000
    print(f"  period        {period_days:.0f} days, {len(result.equity_curve)} bars")
    if result.blocked:
        print("  gate rejections:")
        for reason, count in sorted(result.blocked.items(), key=lambda kv: -kv[1]):
            print(f"    {count:5d}  {reason}")
    print()

    all_warnings = result.warnings + dataset_warnings
    if all_warnings:
        print("=== CAVEATS ===")
        for warning in all_warnings:
            print(f"  - {warning}")
        print()

    if bench is None:
        return

    print("=== BENCHMARKS ===")
    print(f"  buy & hold    {buy_and_hold(datasets[sym], config.indicators.entry_timeframe):+.2f}%")
    print(f"  {bench.summary()}")

    percentile = bench.percentile_of(metrics.total_return_pct)
    print()
    print(
        f"  strategy {metrics.total_return_pct:+.2f}% sits at the "
        f"{percentile:.0f}th percentile of random entries"
    )
    _print_verdict(percentile)

    if metrics.trades < MIN_MEANINGFUL_TRADES:
        print(
            f"  NOTE: only {metrics.trades} trades, so the percentile above "
            "rests on very few observations."
        )
    print()


def _report_pooled(runs, datasets, config, dataset_warnings, runtime) -> None:
    """Compact per-market lines plus a pooled sample.

    Pooling is the entire point of `--symbols`. Any single market here yields
    too few trades for the distributional statistics to carry weight, and a
    conclusion drawn from one market is indistinguishable from luck.
    """
    print("=== PER MARKET ===")
    print(
        f"  {'market':7s}{'trades':>7s}{'return':>10s}"
        f"{'PF':>7s}{'win%':>7s}{'pctile':>8s}"
    )
    percentiles: list[float] = []
    for sym, _result, metrics, bench in runs:
        if bench is None:
            pctile = "n/a"
        else:
            value = bench.percentile_of(metrics.total_return_pct)
            percentiles.append(value)
            pctile = f"{value:.0f}"
        pf = (
            f"{metrics.profit_factor:.2f}"
            if math.isfinite(metrics.profit_factor)
            else "inf"
        )
        print(
            f"  {sym:7s}{metrics.trades:7d}{metrics.total_return_pct:+9.2f}%"
            f"{pf:>7s}{metrics.win_rate_pct:7.1f}{pctile:>8s}"
        )
    print()

    trades = sum(m.trades for _, _, m, _ in runs)
    wins = sum(m.wins for _, _, m, _ in runs)
    losses = sum(m.losses for _, _, m, _ in runs)
    # `avg_loss` is stored negative, so negate it back to a magnitude.
    gross_profit = sum(m.avg_win * m.wins for _, _, m, _ in runs)
    gross_loss = sum(-m.avg_loss * m.losses for _, _, m, _ in runs)
    total_pnl = sum(m.expectancy * m.trades for _, _, m, _ in runs)

    print("=== POOLED ===")
    print(f"  markets       {len(runs)}")
    print(f"  trades        {trades} ({wins}W / {losses}L)")
    if trades:
        print(f"  win rate      {wins / trades * 100:.1f}%")
        print(f"  expectancy    ${total_pnl / trades:+.2f}/trade  ({total_pnl:+.2f} total)")
    if gross_loss > 0:
        print(f"  profit factor {gross_profit / gross_loss:.2f}")
    print(f"  runtime       {runtime:.1f}s")

    if trades < MIN_MEANINGFUL_TRADES:
        print(
            f"  NOTE: {trades} pooled trades is below the {MIN_MEANINGFUL_TRADES} "
            "needed for these statistics to mean anything. Widen the window or "
            "add markets."
        )
    print()

    # The percentiles above are only meaningful next to the distribution they
    # refer to. Without it "pctile 0" reads as a mild miss when it actually
    # means every one of the random runs did better - which is how this section
    # was first misread. Print the distribution and the buy-and-hold line here
    # too, not only in the single-market report.
    have_bench = any(b is not None for _, _, _, b in runs)
    if have_bench:
        print("=== BENCHMARKS ===")
        for sym, _result, metrics, bench in runs:
            if bench is None:
                continue
            try:
                hold = buy_and_hold(
                    datasets[sym], config.indicators.entry_timeframe
                )
                hold_txt = f"  buy & hold {hold:+.2f}%"
            except (KeyError, TypeError):
                hold_txt = "  buy & hold n/a"
            print(
                f"  {sym:6s} strategy {metrics.total_return_pct:+.2f}%"
                f"   {hold_txt}"
            )
            print(f"         {bench.summary()}")
        print()

    if percentiles and all(p < 75 for p in percentiles):
        print(
            "  VERDICT: no market beat random entries, so the pooled result is "
            "not rescuing one weak market."
        )
        print()
    elif percentiles:
        print(
            f"  VERDICT: random-entry percentiles were "
            f"{', '.join(f'{p:.0f}' for p in percentiles)}; at least one market "
            "cleared 75, so inspect it individually before concluding anything."
        )
        print()

    if dataset_warnings:
        print("=== DATA CAVEATS ===")
        for warning in dataset_warnings:
            print(f"  - {warning}")
        print()

    # Strategy caveats are identical across markets, so they print once.
    shared = runs[0][1].warnings
    if shared:
        print("=== STRATEGY CAVEATS (identical for every market) ===")
        for warning in shared:
            print(f"  - {warning}")
        print()


def _resolve_symbols(args, config) -> list[str]:
    """Which markets to run, in order.

    Raises ValueError rather than silently preferring one option, because a
    command that says both `--symbol ETH` and `--symbols BTC,SOL` should fail
    loudly instead of running something the operator did not ask for.
    """
    if args.symbol and args.symbols:
        raise ValueError("use either --symbol or --symbols, not both")

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        symbols = [config.symbol.upper()]

    if not symbols:
        raise ValueError("no symbols requested")
    return symbols


def _as_of_ms(value: str | None) -> int | None:
    """Milliseconds at the close of a `YYYY-MM-DD` UTC day, or None.

    Returns None for a missing value (meaning "use the current time") and also
    for an unparseable one. The caller distinguishes the two by checking
    whether the flag was supplied at all, so a typo is an error rather than a
    silently unpinned run.
    """
    if not value:
        return None
    try:
        day = dt.datetime.strptime(value.strip(), "%Y-%m-%d")
    except (ValueError, AttributeError):
        return None
    return int(calendar.timegm(day.timetuple()) + 86_399) * 1000


def _interval_hours(interval: str) -> float:
    from agent.market_data import TIMEFRAME_MS

    return TIMEFRAME_MS.get(interval, 3_600_000) / 3_600_000


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130)
