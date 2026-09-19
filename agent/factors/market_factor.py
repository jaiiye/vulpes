"""Market structure factor.

Reads funding rate, 24h price change and open interest to judge crowding.

Hyperliquid settles funding *hourly*, which makes this a much faster signal
than on an 8-hour CEX schedule. Two reads are combined:

1. Crowding: extreme funding means the side paying is crowded, which is a
   contrarian signal over a multi-hour horizon.
2. Momentum: the 24h price change.

These pull in opposite directions by design, so the factor reports the net
and lowers confidence when they conflict.

Open interest is read and recorded on every cycle but is deliberately NOT part
of the score. The hypothesis - a concurrent rise in open interest and price
means fresh leveraged money is chasing the move - has never been evaluated
against this system's own data, and it cannot be evaluated in the backtest at
all, because `backtest/market.py` has no historical open interest and reports
zero. Recording the growth now is what makes that evaluation possible later;
scoring it now would be a guess dressed up as a signal.

An earlier version of this docstring described the open interest read as an
implemented "regime" input, while `oi_growth` was computed and then discarded
into a log string. The computation was real, the claim was not.
"""

from __future__ import annotations

from ..market_data import HyperliquidMarket, MarketDataError
from .base import FactorScore

# Hourly funding that is worth reacting to, in basis points.
FUNDING_NOTABLE_BPS = 0.5
FUNDING_EXTREME_BPS = 3.0


class MarketFactor:
    """Funding / OI / momentum based crowding score."""

    def __init__(self, market: HyperliquidMarket) -> None:
        self.market = market
        self._prev_oi: dict[str, float] = {}

    def evaluate(self, symbol: str) -> FactorScore:
        name = symbol.upper()
        try:
            ctx = self.market.asset_context(name)
        except MarketDataError as exc:
            return FactorScore(
                name="market",
                score=50.0,
                confidence=0.0,
                reasons=[f"market context unavailable: {exc}"],
            )

        reasons: list[str] = []
        details: dict[str, object] = {}

        funding_bps = ctx.funding * 10_000
        oi_notional = ctx.open_interest_notional
        details["funding_bps"] = round(funding_bps, 4)
        details["funding_apr_pct"] = round(ctx.funding_apr_pct, 2)
        details["open_interest_base"] = round(ctx.open_interest, 4)
        details["open_interest_usd"] = round(oi_notional, 2)
        details["day_change_pct"] = round(ctx.day_change_pct, 2)
        details["testnet"] = self.market.testnet

        # --- Crowding read (contrarian) --------------------------------
        # Positive funding = longs pay = longs crowded = lean short.
        if abs(funding_bps) < FUNDING_NOTABLE_BPS:
            crowding_score = 50.0
            reasons.append(
                f"funding {funding_bps:+.3f}bp/h is benign "
                f"(~{ctx.funding_apr_pct:+.1f}% APR)"
            )
        else:
            magnitude = min(
                1.0,
                (abs(funding_bps) - FUNDING_NOTABLE_BPS)
                / (FUNDING_EXTREME_BPS - FUNDING_NOTABLE_BPS),
            )
            direction = -1.0 if funding_bps > 0 else 1.0
            crowding_score = 50.0 + direction * magnitude * 22.0
            side = "longs" if funding_bps > 0 else "shorts"
            reasons.append(
                f"funding {funding_bps:+.3f}bp/h -> {side} crowded, "
                f"contrarian lean {direction:+.0f} (APR {ctx.funding_apr_pct:+.1f}%)"
            )

        # --- Momentum read ---------------------------------------------
        change = ctx.day_change_pct
        momentum_score = 50.0 + max(-50.0, min(50.0, change * 1.5))

        # --- Open interest: recorded, not scored ------------------------
        # Recorded in `details` so the journal accumulates a machine-readable
        # series. It is intentionally absent from `score` and `confidence`:
        # the hypothesis is unevaluated, and it cannot be backtested at all,
        # because historical open interest does not exist.
        #
        # `_prev_oi` lives in this process, so `oi_growth_pct` is only
        # meaningful when one process evaluates repeatedly. The scheduled
        # deployment runs `run_bot.py --cycles 1` every 15 minutes, so every
        # evaluate() sees an empty `_prev_oi` and the growth reads exactly 0.0
        # in production. That is not a signal of "no change"; it is the
        # absence of a previous reading. Derive growth offline from
        # consecutive `open_interest_usd` records, which is the durable series.
        prev = self._prev_oi.get(name)
        oi_growth_pct = 0.0
        if prev and prev > 0:
            oi_growth_pct = (oi_notional - prev) / prev * 100
        self._prev_oi[name] = oi_notional
        details["oi_growth_pct"] = round(oi_growth_pct, 4)

        reasons.append(f"24h change {change:+.2f}%, open interest ${oi_notional / 1e6:,.1f}M")
        if oi_growth_pct:
            reasons.append(f"OI change since last tick {oi_growth_pct:+.2f}%")

        # Blend: crowding dominates when funding is extreme, otherwise price momentum.
        crowding_weight = min(0.65, 0.30 + abs(funding_bps) / FUNDING_EXTREME_BPS * 0.35)
        score = crowding_score * crowding_weight + momentum_score * (1 - crowding_weight)

        # Confidence from funding significance. Open interest deliberately does
        # not feed confidence: absolute OI is venue-wide context, not a
        # per-symbol signal, and testnet OI is a small fraction of mainnet.
        funding_conf = min(1.0, abs(funding_bps) / FUNDING_EXTREME_BPS)
        confidence = 0.45 + 0.55 * funding_conf

        # Conflict detection: crowding says short while momentum says long (or vice versa).
        if (crowding_score - 50) * (momentum_score - 50) < 0:
            gap = abs(crowding_score - momentum_score)
            if gap > 20:
                confidence *= 0.7
                reasons.append(
                    "crowding and momentum disagree: confidence reduced"
                )

        return FactorScore(
            name="market",
            score=score,
            confidence=max(0.05, min(1.0, confidence)),
            reasons=reasons,
            details=details,
        ).clamp()
