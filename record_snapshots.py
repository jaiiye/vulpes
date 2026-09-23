#!/usr/bin/env python3
"""Record whale position snapshots for future backtesting.

Run this on a schedule (cron, systemd timer) alongside the agent. The smart
money factor cannot be backtested until enough history accumulates, and the
only way to get that history is to record it going forward.

    python record_snapshots.py
    python record_snapshots.py --symbols BTC,ETH,SOL
    python record_snapshots.py --summary          # show coverage so far
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.config import ConfigError, load_config  # noqa: E402
from agent.snapshots import (  # noqa: E402
    DEFAULT_MARKET_SNAPSHOT_PATH,
    DEFAULT_SNAPSHOT_PATH,
    MARKET_SNAPSHOT_INTERVAL_SECONDS,
    MarketSnapshotRecorder,
    SnapshotRecorder,
    coverage_summary,
    load_market_records,
    load_records,
    market_coverage_summary,
)
from run_bot import DEFAULT_CONFIG, load_dotenv  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Record whale position snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--symbols", default="BTC,ETH", help="comma-separated symbols to record"
    )
    p.add_argument("--path", default=DEFAULT_SNAPSHOT_PATH, help="JSONL output path")
    p.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="agent config whose smart_money settings define the wallet set",
    )
    p.add_argument(
        "--summary", action="store_true", help="report coverage and exit without recording"
    )
    p.add_argument(
        "--hyperfeed",
        action="store_true",
        help="force the curated Hyperfeed (the config's use_hyperfeed also enables it)",
    )
    p.add_argument(
        "--market-path",
        default=DEFAULT_MARKET_SNAPSHOT_PATH,
        help="JSONL output path for aggregate positioning",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "record a market snapshot even when the newest is younger than an "
            f"hour - {MARKET_SNAPSHOT_INTERVAL_SECONDS / 3600:.0f}h is the native "
            "cadence of funding and of Binance's long-short series"
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(Path(".env"))

    if args.summary:
        print(f"whale snapshots: {args.path}")
        print(coverage_summary(load_records(args.path)))
        print()
        print(f"market snapshots: {args.market_path}")
        print(market_coverage_summary(load_market_records(args.market_path)))
        return 0

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    if not symbols:
        print("no symbols given", file=sys.stderr)
        return 2

    # The recorder must select wallets exactly as the agent does, or the
    # recorded history describes a different strategy than the one being run.
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    started = time.time()
    recorder = SnapshotRecorder(
        path=args.path, use_hyperfeed=args.hyperfeed, config=config
    )
    written = recorder.record(symbols)

    # The market side skips itself whenever the newest record is under an hour
    # old, so running this every 15 minutes wastes three calls in four and
    # nothing else.
    market = MarketSnapshotRecorder(path=args.market_path)
    market_written = market.record(symbols, force=args.force)

    print(
        f"recorded {written}/{len(symbols)} whale snapshots and "
        f"{1 if market_written else 0} market snapshot in "
        f"{time.time() - started:.1f}s"
    )
    for error in recorder.errors:
        print(f"  whale error: {error}", file=sys.stderr)
    for error in market.errors:
        print(f"  market error: {error}", file=sys.stderr)

    # Either one landing means this pass did something. The exit code becomes
    # the cron's failure count, and a pass that recorded the market while the
    # whale source failed is not a pass that failed.
    return 0 if (written or market_written) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130)
