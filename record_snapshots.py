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
    DEFAULT_SNAPSHOT_PATH,
    SnapshotRecorder,
    coverage_summary,
    load_records,
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(Path(".env"))

    if args.summary:
        records = load_records(args.path)
        print(f"snapshot file: {args.path}")
        print(coverage_summary(records))
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

    print(
        f"recorded {written}/{len(symbols)} snapshots for "
        f"{', '.join(symbols)} in {time.time() - started:.1f}s -> {args.path}"
    )
    for error in recorder.errors:
        print(f"  error: {error}", file=sys.stderr)

    return 0 if written else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130)
