"""Multi-factor signal synthesizer.

Blends the three factors into a single 0-100 score and a discrete signal.

Design note on confidence handling: a factor that reports zero confidence
(for example, smart money data unavailable) does *not* contribute a neutral
50. Instead its weight is redistributed across the factors that do have data.
That keeps the scale honest, but it also means the resulting signal is built
on fewer inputs, which is exactly why the discipline layer separately requires
smart money alignment rather than trusting the blend alone.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import BotConfig
from .factors.base import FactorScore
from .factors.market_factor import MarketFactor
from .factors.smart_money import SmartMoneySnapshot, smart_money_factor_from_config
from .factors.technical import TechnicalFactor
from .market_data import HyperliquidMarket

LONG = "long"
SHORT = "short"
NEUTRAL = "neutral"


@dataclass
class Signal:
    """A synthesised trading signal."""

    symbol: str
    action: str = NEUTRAL
    score: float = 50.0
    confidence: float = 0.0
    factors: dict[str, FactorScore] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    blocked_by: str | None = None

    @property
    def is_actionable(self) -> bool:
        return self.action in (LONG, SHORT) and self.blocked_by is None

    def describe(self) -> str:
        head = (
            f"{self.symbol} -> {self.action.upper()} "
            f"score={self.score:.1f} conf={self.confidence:.0%}"
        )
        detail = " | ".join(f"{k}={v.score:.0f}({v.confidence:.0%})" for k, v in self.factors.items())
        return f"{head}\n  factors: {detail}"


class Synthesizer:
    """Combines factor scores into discrete signals."""

    def __init__(self, config: BotConfig, market: HyperliquidMarket | None = None) -> None:
        self.cfg = config
        self.market = market or HyperliquidMarket(
            testnet=config.execution.testnet, base_url=config.execution.base_url
        )
        self.smart_money = smart_money_factor_from_config(config, self.market)
        self.technical = TechnicalFactor(self.market, config.indicators)
        self.market_factor = MarketFactor(self.market)

    # ------------------------------------------------------------------
    def compute_factors(self, symbol: str) -> dict[str, FactorScore]:
        """Evaluate all three factors for `symbol`."""
        snapshot: SmartMoneySnapshot | None = None
        try:
            snapshot = self.smart_money.snapshot(symbol)
        except Exception as exc:  # noqa: BLE001 - degrade rather than crash the loop
            snapshot = SmartMoneySnapshot(symbol=symbol.upper(), errors=[str(exc)])

        return {
            "smart_money": self.smart_money.evaluate(symbol, snapshot),
            "technical": self.technical.evaluate(symbol),
            "market": self.market_factor.evaluate(symbol),
        }

    @staticmethod
    def blend(factors: dict[str, FactorScore], weights) -> tuple[float, float, list[str]]:
        """Confidence-weighted blend of factor scores.

        Returns (score, confidence, notes).
        """
        notes: list[str] = []
        weight_map = {
            "smart_money": weights.smart_money,
            "technical": weights.technical,
            "market": weights.market,
        }

        num = 0.0
        den = 0.0
        for name, score in factors.items():
            w = weight_map.get(name, 0.0)
            if w <= 0:
                continue
            effective = w * score.confidence
            if effective <= 0:
                notes.append(f"{name} contributed nothing (no data)")
                continue
            num += score.score * effective
            den += effective

        if den <= 0:
            return 50.0, 0.0, notes + ["no factor produced usable data"]

        blended = num / den
        # Confidence is the share of configured weight that actually had data.
        coverage = den / sum(weight_map.values()) if sum(weight_map.values()) else 0.0
        agreement = _agreement(factors)

        confidence = 0.55 * coverage + 0.45 * agreement
        if coverage < 1.0:
            notes.append(f"factor coverage {coverage:.0%} (some inputs missing)")
        notes.append(f"factor agreement {agreement:.0%}")
        return blended, max(0.0, min(1.0, confidence)), notes

    # ------------------------------------------------------------------
    def generate(self, symbol: str | None = None) -> Signal:
        """Produce a raw signal, before discipline filtering."""
        name = (symbol or self.cfg.symbol).upper()
        factors = self.compute_factors(name)
        score, confidence, notes = self.blend(factors, self.cfg.weights)

        d = self.cfg.discipline
        if score >= d.long_threshold:
            action = LONG
        elif score <= d.short_threshold:
            action = SHORT
        else:
            action = NEUTRAL

        reasons = list(notes)
        for fname, fscore in factors.items():
            for r in fscore.reasons[:4]:
                reasons.append(f"[{fname}] {r}")

        if action != NEUTRAL and d.min_score_gap > 0:
            anchor = d.long_threshold if action == LONG else d.short_threshold
            if abs(score - anchor) < d.min_score_gap:
                reasons.append(
                    f"score {score:.1f} within {d.min_score_gap:g} of threshold "
                    f"{anchor:g}: treated as neutral"
                )
                action = NEUTRAL

        return Signal(
            symbol=name,
            action=action,
            score=score,
            confidence=confidence,
            factors=factors,
            reasons=reasons,
        )


def _agreement(factors: dict[str, FactorScore]) -> float:
    """How much the factors agree on direction, 0-1.

    Factors with no confidence are ignored. Complete unanimity scores 1.0,
    a perfect 50/50 split scores 0.0.
    """
    directional: list[float] = []
    for score in factors.values():
        if score.confidence <= 0:
            continue
        directional.append(score.score - 50.0)

    if len(directional) < 2:
        return 0.4  # single-source signals are structurally low agreement

    longs = sum(1 for v in directional if v > 0)
    shorts = sum(1 for v in directional if v < 0)
    decided = longs + shorts
    if decided == 0:
        return 0.0
    return abs(longs - shorts) / decided


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


class Journal:
    """Append-only JSONL log of signals and decisions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, kind: str, payload: dict[str, Any]) -> None:
        record = {"ts": time.time(), "kind": kind, **payload}
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            # Journaling must never take down the trading loop.
            pass

    def signal(self, signal: Signal, extra: dict[str, Any] | None = None) -> None:
        self.write(
            "signal",
            {
                "symbol": signal.symbol,
                "action": signal.action,
                "score": round(signal.score, 3),
                "confidence": round(signal.confidence, 4),
                "blocked_by": signal.blocked_by,
                "factors": {
                    k: {
                        "score": round(v.score, 2),
                        "confidence": round(v.confidence, 3),
                        "details": v.details,
                    }
                    for k, v in signal.factors.items()
                },
                **(extra or {}),
            },
        )

    def event(self, kind: str, **payload: Any) -> None:
        self.write(kind, payload)
