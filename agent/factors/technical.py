"""Technical factor.

Blends Supertrend, RSI, ADX and EMA structure into one 0-100 score.

The 22-agent experiment is explicit that pure TA loses money on its own
(Viper: -18%). This factor is therefore designed to be *informative but
subordinate*: it can veto via `confidence`, and `require_smart_money_alignment`
in the discipline layer can make it non-authoritative.
"""

from __future__ import annotations

from ..indicators import CandleSeries
from ..market_data import HyperliquidMarket, safe_candles
from .base import FactorScore

# Supertrend is the primary trend read, so it carries the most weight.
SUPERTREND_WEIGHT = 0.40
EMA_WEIGHT = 0.25
RSI_WEIGHT = 0.20
ADX_WEIGHT = 0.15


class TechnicalFactor:
    """Multi-timeframe technical scoring."""

    def __init__(self, market: HyperliquidMarket, indicators_cfg) -> None:
        self.market = market
        self.cfg = indicators_cfg

    def evaluate(self, symbol: str) -> FactorScore:
        #: Which venue the candles come from. Signals must read mainnet whatever
        #: `execution.testnet` says; recorded so a violation shows up in the
        #: journal instead of staying silent.
        #:
        #: Computed before any early return on purpose. "Insufficient history"
        #: and "no candles at all" are the two cases where someone actually
        #: asks which venue was read - and they were the two that recorded
        #: nothing, because every early return skipped the `details` dict that
        #: was built further down.
        data_network = "testnet" if getattr(self.market, "testnet", False) else "mainnet"

        entry = safe_candles(
            self.market,
            symbol,
            self.cfg.entry_timeframe,
            self.cfg.lookback_candles,
        )
        if entry is None or len(entry) < max(self.cfg.ema_slow, self.cfg.adx_period * 2) + 5:
            return FactorScore(
                name="technical",
                score=50.0,
                confidence=0.0,
                reasons=[
                    f"insufficient {self.cfg.entry_timeframe} candle history "
                    f"({0 if entry is None else len(entry)} bars)"
                ],
                details={"data_network": data_network},
            )

        trend = safe_candles(
            self.market, symbol, self.cfg.trend_timeframe, self.cfg.lookback_candles
        )

        components: list[tuple[str, float, float]] = []  # (label, score, weight)
        reasons: list[str] = []
        details: dict[str, object] = {"data_network": data_network}

        # --- Supertrend (entry timeframe) ------------------------------
        st_line, st_dir = entry.supertrend(
            self.cfg.supertrend_period, self.cfg.supertrend_multiplier
        )
        st_now = st_dir[-1]
        if st_now is not None:
            dist_pct = 0.0
            if st_line[-1] and entry.last_price:
                dist_pct = (entry.last_price - st_line[-1]) / entry.last_price * 100
            # Direction gives the base score; distance from the line adds conviction.
            base = 70.0 if st_now == "buy" else 30.0
            tilt = max(-15.0, min(15.0, dist_pct * 3.0))
            st_score = max(0.0, min(100.0, base + tilt if st_now == "buy" else base - tilt))
            components.append(("supertrend", st_score, SUPERTREND_WEIGHT))
            details["supertrend"] = st_now
            details["supertrend_dist_pct"] = round(dist_pct, 3)
            reasons.append(
                f"Supertrend {self.cfg.entry_timeframe} = {st_now} ({dist_pct:+.2f}% from line)"
            )

        # --- EMA structure --------------------------------------------
        ema_fast = entry.ema(self.cfg.ema_fast)[-1]
        ema_slow = entry.ema(self.cfg.ema_slow)[-1]
        if ema_fast is not None and ema_slow is not None and ema_slow > 0:
            spread_pct = (ema_fast - ema_slow) / ema_slow * 100
            # +-1.5% EMA spread saturates the score.
            ema_score = 50.0 + max(-50.0, min(50.0, spread_pct * 33.3))
            components.append(("ema", ema_score, EMA_WEIGHT))
            details["ema_spread_pct"] = round(spread_pct, 3)
            reasons.append(
                f"EMA{self.cfg.ema_fast}/{self.cfg.ema_slow} spread {spread_pct:+.2f}%"
            )

        # --- RSI ------------------------------------------------------
        rsi_val = entry.rsi(self.cfg.rsi_period)[-1]
        if rsi_val is not None:
            # 50 is neutral; trend-friendly zone is 45-70 rather than extremes.
            rsi_score = 50.0 + (rsi_val - 50.0) * 1.2
            components.append(("rsi", rsi_score, RSI_WEIGHT))
            details["rsi"] = round(rsi_val, 2)
            reasons.append(f"RSI({self.cfg.rsi_period}) = {rsi_val:.1f}")

        # --- ADX (trend strength, gates conviction) --------------------
        adx_val = entry.adx(self.cfg.adx_period)[-1]
        trend_strength = 0.5
        if adx_val is not None:
            details["adx"] = round(adx_val, 2)
            # ADX 15 = directionless, ADX 40 = strong trend.
            trend_strength = max(0.0, min(1.0, (adx_val - 15.0) / 25.0))
            reasons.append(f"ADX = {adx_val:.1f} (trend strength {trend_strength:.0%})")

        # --- Higher timeframe gate ------------------------------------
        htf_dir = None
        if trend is not None and len(trend) > self.cfg.supertrend_period + 2:
            _, htf_dirs = trend.supertrend(
                self.cfg.supertrend_period, self.cfg.supertrend_multiplier
            )
            htf_dir = htf_dirs[-1]
            details["htf_supertrend"] = htf_dir
            if htf_dir is not None:
                reasons.append(
                    f"Supertrend {self.cfg.trend_timeframe} = {htf_dir} (higher timeframe)"
                )

        if not components:
            return FactorScore(
                name="technical",
                score=50.0,
                confidence=0.0,
                reasons=reasons + ["no technical component could be computed"],
                details=details,
            )

        total_w = sum(w for _, _, w in components)
        raw = sum(score * w for _, score, w in components) / total_w

        # Confidence: how strong the trend is, and whether the higher
        # timeframe agrees with the entry timeframe.
        confidence = 0.35 + 0.4 * trend_strength
        if htf_dir is not None and st_now is not None:
            if htf_dir != st_now:
                confidence *= 0.6
                reasons.append("higher timeframe conflict: conviction reduced")
            else:
                confidence = min(1.0, confidence * 1.15)

        return FactorScore(
            name="technical",
            score=raw,
            confidence=max(0.1, min(1.0, confidence)),
            reasons=reasons,
            details=details,
        ).clamp()
