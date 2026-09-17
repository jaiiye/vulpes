#!/usr/bin/env python3
"""Entry point for the Fox-style Hyperliquid agent.

Usage:
    python run_bot.py                          # run the default config
    python run_bot.py --config bots/fox_btc.yaml
    python run_bot.py --validate               # check config, no network loop
    python run_bot.py --once                   # single cycle, then exit
    python run_bot.py --cycles 5 --interval 30 # bounded run
    python run_bot.py --show-signal            # print factor breakdown, no trading
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.bot import FoxAgent  # noqa: E402
from agent.config import ConfigError, load_config  # noqa: E402
from agent.market_data import MarketDataError  # noqa: E402
from agent.state import DEFAULT_STATE_PATH, StateStore  # noqa: E402
from agent.synthesizer import Synthesizer  # noqa: E402

DEFAULT_CONFIG = "bots/fox_btc.yaml"


def load_dotenv(path: Path) -> None:
    """Minimal .env loader so no extra dependency is needed."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fox-style multi-factor Hyperliquid trading agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", default=DEFAULT_CONFIG, help="path to the YAML config")
    p.add_argument(
        "--validate", action="store_true", help="validate the config and exit"
    )
    p.add_argument("--once", action="store_true", help="run a single cycle and exit")
    p.add_argument("--cycles", type=int, default=None, help="number of cycles to run")
    p.add_argument(
        "--interval", type=int, default=None, help="seconds between cycles (overrides config)"
    )
    p.add_argument(
        "--show-signal",
        action="store_true",
        help="print the factor breakdown without trading",
    )
    p.add_argument("--journal", default="logs/journal.jsonl", help="journal output path")
    p.add_argument(
        "--state",
        default=DEFAULT_STATE_PATH,
        help="path to the persisted state file (guardrails + open position)",
    )
    p.add_argument(
        "--show-state",
        action="store_true",
        help="print the persisted state and exit",
    )
    p.add_argument(
        "--clear-halt",
        action="store_true",
        help=(
            "clear a persisted halt and exit. Only do this after resolving the "
            "cause; it discards the drawdown/critical halt record."
        ),
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="enable live trading (requires --i-understand-the-risk)",
    )
    p.add_argument(
        "--i-understand-the-risk",
        action="store_true",
        help="required alongside --live to send real orders",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv(Path(".env"))

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    # Live trading needs two explicit flags. Dry run is the default.
    if args.live:
        if not args.i_understand_the_risk:
            print(
                "refusing --live without --i-understand-the-risk.\n"
                "This agent sends real orders to Hyperliquid. Test with the default\n"
                "dry-run mode first.",
                file=sys.stderr,
            )
            return 2
        if not os.getenv("HYPERLIQUID_PRIVATE_KEY"):
            print(
                "HYPERLIQUID_PRIVATE_KEY is not set. Create an API wallet at\n"
                "https://app.hyperliquid.xyz/API and export its key. Never use the\n"
                "main wallet key.",
                file=sys.stderr,
            )
            return 2
        config.execution.dry_run = False
    else:
        config.execution.dry_run = True

    if args.interval is not None:
        config.execution.poll_interval_seconds = max(5, args.interval)

    print(config.summary())
    print()

    if args.validate:
        print("config is valid")
        return 0

    if args.show_signal:
        return show_signal(config)

    store = StateStore(args.state)

    if args.show_state:
        state = store.load()
        print(state.describe() if state else f"no state at {args.state}")
        return 0

    if args.clear_halt:
        state = store.load()
        if state is None:
            print(f"no state at {args.state}; nothing to clear")
            return 0
        if not state.agent_halted:
            print("agent is not halted; nothing to clear")
            return 0
        print(f"clearing halt: {state.agent_halt_reason}")
        state.agent_halted = False
        state.agent_halt_reason = ""
        store.save(state)
        print("halt cleared. The drawdown peak is retained, so the same drawdown "
              "will halt again if equity is still below the limit.")
        return 0

    agent = FoxAgent(config, journal_path=args.journal, state_path=args.state)
    max_cycles = 1 if args.once else args.cycles
    stats = agent.run(max_cycles=max_cycles)

    print()
    print(stats.summary())
    if stats.errors:
        print(f"{len(stats.errors)} error(s) during the run; see the journal for detail.")
    return 0


def show_signal(config) -> int:
    """Print the factor breakdown for a symbol without trading."""
    synthesizer = Synthesizer(config)
    symbol = config.symbol

    print(f"computing factors for {symbol} (no orders will be sent)")
    print()

    try:
        factors = synthesizer.compute_factors(symbol)
    except MarketDataError as exc:
        print(f"market data error: {exc}", file=sys.stderr)
        return 1

    for name, score in factors.items():
        weight = getattr(config.weights, name if name != "smart_money" else "smart_money", 0.0)
        print(f"--- {name} (weight {weight:.0%}) ---")
        print(f"  score      : {score.score:.1f}  ({score.direction})")
        print(f"  confidence : {score.confidence:.0%}")
        for reason in score.reasons:
            print(f"  - {reason}")
        if score.details:
            print(f"  details    : {json.dumps(score.details, default=str)}")
        print()

    score, confidence, notes = synthesizer.blend(factors, config.weights)
    d = config.discipline
    if score >= d.long_threshold:
        action = "LONG"
    elif score <= d.short_threshold:
        action = "SHORT"
    else:
        action = "NEUTRAL"

    print("--- synthesis ---")
    print(f"  blended score : {score:.1f}")
    print(f"  confidence    : {confidence:.0%}")
    print(f"  thresholds    : long >= {d.long_threshold:g}, short <= {d.short_threshold:g}")
    print(f"  signal        : {action}")
    for note in notes:
        print(f"  - {note}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130)
