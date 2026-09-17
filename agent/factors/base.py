"""Shared types for score factors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FactorScore:
    """One factor's opinion about a symbol.

    score is a 0-100 directional score where 50 is neutral:
      > 50 leans long, < 50 leans short.
    confidence is 0-1 and scales this factor's contribution to the blend.
    A confidence of 0 means "no data", and the weight is redistributed.
    """

    name: str
    score: float = 50.0
    confidence: float = 0.0
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def clamp(self) -> "FactorScore":
        self.score = max(0.0, min(100.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))
        return self

    @property
    def direction(self) -> str:
        if self.score > 50:
            return "long"
        if self.score < 50:
            return "short"
        return "neutral"


def lerp_score(value: float, low: float, high: float) -> float:
    """Map `value` in [low, high] onto a 0-100 score."""
    if high == low:
        return 50.0
    ratio = (value - low) / (high - low)
    return max(0.0, min(100.0, ratio * 100.0))


def band_score(
    value: float, bearish_at: float, bullish_at: float
) -> float:
    """Score a metric that is bearish when low and bullish when high."""
    return lerp_score(value, bearish_at, bullish_at)
