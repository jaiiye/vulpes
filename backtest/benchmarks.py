"""Benchmarks that give a backtest result something to be compared against.

A positive return alone proves nothing: any long-biased strategy looks clever
in a bull market. These benchmarks answer the two questions that actually
matter:

  1. Did it beat simply holding the asset?
  2. Did it beat random entries using the SAME gates, sizing, exits, fees and
     funding?

The second is the important one. If random directions produce a similar
distribution of outcomes, the signal carries no directional information and
the strategy has no edge regardless of how the headline number looks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from agent.factors.base import FactorScore
from agent.synthesizer import LONG, SHORT, Signal

from .data import HistoricalData
from .engine import Backtester, BacktestResult


class RandomEntryBacktester(Backtester):
    """Identical pipeline, random direction.

    Every discipline gate, the sizing model, the exit logic, fees and funding
    are unchanged. Only the long/short choice is noise.
    """

    def __init__(self, *args, seed: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.rng = random.Random(seed)

    def _generate_signal(self) -> Signal:
        action = LONG if self.rng.random() < 0.5 else SHORT
        score = 70.0 if action == LONG else 30.0
        return Signal(
            symbol=self.symbol,
            action=action,
            score=score,
            confidence=0.9,
            factors={
                # Zero confidence, so the blend redistributes weight exactly
                # as it does in the real run.
                "smart_money": FactorScore("smart_money", 50.0, 0.0),
                "technical": FactorScore("technical", score, 0.9),
                "market": FactorScore("market", score, 0.9),
            },
            reasons=["random entry benchmark"],
        )


@dataclass
class BenchmarkResult:
    """Outcome of one or more comparison runs."""

    name: str
    returns_pct: list[float]

    @property
    def mean_pct(self) -> float:
        return sum(self.returns_pct) / len(self.returns_pct) if self.returns_pct else 0.0

    def percentile_of(self, value: float) -> float:
        """What percentile `value` falls at within this distribution."""
        if not self.returns_pct:
            return 0.0
        below = sum(1 for r in self.returns_pct if r < value)
        return below / len(self.returns_pct) * 100.0

    def summary(self) -> str:
        if not self.returns_pct:
            return f"{self.name}: no runs"
        ordered = sorted(self.returns_pct)
        n = len(ordered)
        p5 = ordered[int(n * 0.05)]
        p50 = ordered[n // 2]
        p95 = ordered[min(n - 1, int(n * 0.95))]
        return (
            f"{self.name}: n={n} mean {self.mean_pct:+.2f}% "
            f"p5 {p5:+.2f}% p50 {p50:+.2f}% p95 {p95:+.2f}%"
        )


def buy_and_hold(dataset: HistoricalData, interval: str = "1h") -> float:
    """Return from holding the asset across the traded window, in percent."""
    series = dataset.candles.get(interval)
    if series is None or len(series) < 2:
        return 0.0

    times = series.times
    start_idx = 0
    for i, t in enumerate(times):
        if t >= dataset.start_ms:
            start_idx = i
            break

    # Mirrors the engine's convention: enter at the open of the first traded
    # bar, exit at the last close in the window.
    entry = series.opens[start_idx]
    exit_price = series.closes[-1]
    if entry <= 0:
        return 0.0
    return (exit_price - entry) / entry * 100.0


def random_entry_runs(
    config,
    datasets: dict[str, HistoricalData],
    initial_equity: float = 1000.0,
    taker_fee_bps: float = 3.5,
    runs: int = 20,
    seed: int = 42,
) -> BenchmarkResult:
    """Run the pipeline N times with random directions."""
    returns: list[float] = []
    for i in range(runs):
        bt = RandomEntryBacktester(
            config,
            datasets,
            initial_equity=initial_equity,
            taker_fee_bps=taker_fee_bps,
            seed=seed + i,
        )
        result = bt.run()
        returns.append(result_return_pct(result))
    return BenchmarkResult(name=f"random entry x{runs}", returns_pct=returns)


def result_return_pct(result: BacktestResult) -> float:
    if result.initial_equity <= 0:
        return 0.0
    return (
        (result.final_equity - result.initial_equity) / result.initial_equity * 100.0
    )
