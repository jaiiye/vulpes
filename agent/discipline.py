"""Fox-style trade discipline.

This module exists because the 22-agent experiment showed that *selectivity*,
not signal quality, separated the winners from the losers:

  - Fox and Ghost Fox used the same scanner. Fox chose which signals to take
    and returned +23%; Ghost Fox took more of them and trailed by 56 points.
  - Agents with >400 trades all lost heavily. Agents with <120 trades all won.
  - Mean reversion and pure TA bled out; smart-money-aligned selection won.
  - Agents that "self-adjusted" after losses loosened entries, raised leverage
    and removed protection, accelerating the drawdown every time.

Every rule below is a hard-coded guardrail in the spirit of that last point:
the agent cannot tune its way past them at runtime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import BotConfig
from .indicators import sma
from .market_data import HyperliquidMarket, safe_candles
from .synthesizer import LONG, SHORT, Signal


def _field(data, key: str, default):
    """Read `key` from a mapping or an object, falling back to `default`."""
    if isinstance(data, dict):
        return data.get(key, default)
    return getattr(data, key, default)


# ---------------------------------------------------------------------------
# Block-reason bucketing
# ---------------------------------------------------------------------------
# The messages emitted by the gates below carry variable content:
#
#   "cooldown active, 239 min remaining (cooldown is 240 min)"
#   "halted for another 187 min: 3 consecutive losses"
#   "smart money is not long (score 45.2) - signal rejected"
#
# Anything that counts or groups them has to collapse them first, or nearly
# every occurrence becomes its own value and the distribution disappears.
#
# This lives here, beside the code that emits the messages, so a new gate is
# declared in one place. It used to be duplicated in the backtest and in the
# journal analyser - neither of which owned the vocabulary.
#
# Order matters: the more specific keyword must win, because a smart-money
# rejection also mentions a score.
BLOCK_REASON_BUCKETS = (
    "cooldown",
    "daily cap",
    "confidence",
    "smart money",
    "macro filter",
    "halted",
    "already holding",
    "threshold",
)


def bucket_block_reason(reason: str | None) -> str:
    """Collapse a gate message into a stable bucket.

    Unrecognised wording is returned (lower-cased, truncated) rather than merged
    into a catch-all, so a newly added gate shows up as its own value instead of
    vanishing into "other".
    """
    if not reason:
        return "(none)"
    text = reason.lower()
    for key in BLOCK_REASON_BUCKETS:
        if key in text:
            return key
    return text[:60]


@dataclass
class DisciplineState:
    """Mutable state tracked across loop iterations.

    This state is persisted across restarts. Resetting it would silently
    disable the daily signal cap, the cooldown and the loss-streak halt, which
    together are the only part of this agent backed by evidence.
    """

    last_trade_ts: dict[str, float] = field(default_factory=dict)
    trades_today: dict[str, int] = field(default_factory=dict)
    day_anchor: float = field(default_factory=time.time)
    consecutive_losses: int = 0
    halted_until: float = 0.0
    halt_reason: str = ""

    def roll_day_if_needed(self, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if now < self.day_anchor:
            # The clock moved backwards. That happens when the system clock is
            # corrected, and always in a backtest replaying past dates. Without
            # re-anchoring, `now - day_anchor` stays negative forever and the
            # daily cap latches on permanently after the first few trades.
            self.day_anchor = now
            return
        if now - self.day_anchor >= 86_400:
            self.day_anchor = now
            self.trades_today.clear()

    def to_payload(self) -> dict:
        return {
            "last_trade_ts": dict(self.last_trade_ts),
            "trades_today": dict(self.trades_today),
            "day_anchor": self.day_anchor,
            "consecutive_losses": self.consecutive_losses,
            "halted_until": self.halted_until,
            "halt_reason": self.halt_reason,
        }

    @classmethod
    def from_payload(cls, data) -> "DisciplineState":
        """Rebuild from either a mapping or an object with matching attributes.

        Accepting both matters: the bot holds an AgentState dataclass, and an
        earlier version of this method only handled dicts. The resulting
        AttributeError was swallowed by the fallback, which silently reset the
        guardrails - exactly the bug this persistence exists to prevent.
        """
        state = cls()
        try:
            state.last_trade_ts = {
                str(k): float(v)
                for k, v in _field(data, "last_trade_ts", {}).items()
            }
            state.trades_today = {
                str(k): int(float(v))
                for k, v in _field(data, "trades_today", {}).items()
            }
            state.day_anchor = float(_field(data, "day_anchor", 0.0) or 0.0)
            state.consecutive_losses = int(
                float(_field(data, "consecutive_losses", 0) or 0)
            )
            state.halted_until = float(_field(data, "halted_until", 0.0) or 0.0)
            state.halt_reason = str(_field(data, "halt_reason", "") or "")
        except (TypeError, ValueError, AttributeError):
            return cls()
        if state.day_anchor <= 0:
            # A zero anchor would make roll_day_if_needed fire immediately and
            # wipe today's counts, so re-anchor to now.
            state.day_anchor = time.time()
        return state


class Discipline:
    """Applies selectivity guardrails to raw signals."""

    def __init__(
        self,
        config: BotConfig,
        market: HyperliquidMarket | None = None,
        now_fn=None,
    ) -> None:
        self.cfg = config
        self.d = config.discipline
        self.market = market
        # Injectable clock. The backtest drives this from the simulated
        # timestamp so the cooldown and daily cap replay exactly as they would
        # in live trading, instead of being reimplemented.
        self.now = now_fn or time.time
        self.state = DisciplineState()
        # Anchor the rolling day window to the injected clock, never wall time.
        # A backtest whose simulated time sits in the past would otherwise see
        # `now - day_anchor` stay negative, so the window never rolls and the
        # daily cap latches on permanently after the first few trades.
        self.state.day_anchor = self.now()
        self._btc_trend_cache: tuple[float, str | None] = (0.0, None)

    # ------------------------------------------------------------------
    # Individual gates
    # ------------------------------------------------------------------
    def daily_cap_reached(self, symbol: str) -> bool:
        self.state.roll_day_if_needed(self.now())
        return self.state.trades_today.get(symbol, 0) >= self.d.max_signals_per_day

    def in_cooldown(self, symbol: str) -> bool:
        last = self.state.last_trade_ts.get(symbol)
        if last is None:
            return False
        return (self.now() - last) < self.d.cooldown_minutes * 60

    def cooldown_remaining_minutes(self, symbol: str) -> float:
        last = self.state.last_trade_ts.get(symbol)
        if last is None:
            return 0.0
        remaining = self.d.cooldown_minutes * 60 - (self.now() - last)
        return max(0.0, remaining / 60.0)

    def btc_trend_direction(self) -> str | None:
        """BTC higher-timeframe Supertrend direction, cached for 5 minutes."""
        if self.market is None or not self.d.btc_trend_filter:
            return None

        cached_at = self._btc_trend_cache[0]
        if self.now() - cached_at < 300:
            return self._btc_trend_cache[1]

        series = safe_candles(
            self.market,
            "BTC",
            self.d.btc_trend_timeframe,
            self.cfg.indicators.lookback_candles,
        )
        direction = None
        if series is not None and len(series) > self.cfg.indicators.supertrend_period + 2:
            _, dirs = series.supertrend(
                self.cfg.indicators.supertrend_period,
                self.cfg.indicators.supertrend_multiplier,
            )
            direction = dirs[-1]

        self._btc_trend_cache = (self.now(), direction)
        return direction

    def check_macro_filter(self, signal: Signal) -> str | None:
        """Block counter-trend entries.

        The experiment's fix for Mamba: ban dip-buying while BTC 4h is down.
        That single rule would have avoided 14 of its 28 losing trades.
        """
        if not self.d.btc_trend_filter:
            return None

        btc = self.btc_trend_direction()
        if btc is None:
            return None

        if signal.action == LONG and btc == "sell":
            return (
                f"macro filter: BTC {self.d.btc_trend_timeframe} Supertrend is down, "
                "long entries blocked"
            )
        if signal.action == SHORT and btc == "buy":
            return (
                f"macro filter: BTC {self.d.btc_trend_timeframe} Supertrend is up, "
                "short entries blocked"
            )
        return None

    def check_trend_strength(self, signal: Signal) -> str | None:
        """Require ADX(14) above a floor before taking a directional view.

        ADX measures trend *strength* and carries no direction, so this is a
        regime gate rather than a directional one: it stands aside while the
        market chops, which is where an EMA/RSI reading is least meaningful.

        Fails CLOSED when the candles cannot be read. The opposite choice - let
        the trade through when the reading is unavailable - is how a gate stops
        running without anyone noticing, and this codebase has been bitten by
        that three times already. A closed gate shows up as a rejection reason
        in the counts; an open one shows up nowhere.
        """
        floor = self.d.min_adx
        if floor <= 0:
            return None
        series = safe_candles(
            self.market,
            signal.symbol,
            self.cfg.indicators.entry_timeframe,
            self.cfg.indicators.lookback_candles,
        )
        if series is None:
            return (
                f"ADX filter on but {signal.symbol} candles are unavailable, so "
                "trend strength cannot be confirmed"
            )
        period = self.cfg.indicators.adx_period
        if len(series) < period + 2:
            return (
                f"ADX filter on but only {len(series)} bars are available "
                f"(needs {period + 2})"
            )
        value = series.adx(period)[-1]
        if value is None:
            return "ADX filter on but ADX has no reading yet"
        if value < floor:
            return (
                f"ADX {value:.1f} below the {floor:g} floor: no trend to trade"
            )
        return None

    def check_volume(self, signal: Signal) -> str | None:
        """Require the entry bar's volume to clear a moving average of volume.

        Volume is the only input in this strategy not derived from price, so it
        is the only one that can corroborate a move rather than restate it.

        Fails CLOSED, for the same reason as `check_trend_strength`.
        """
        if not self.d.volume_confirm:
            return None
        series = safe_candles(
            self.market,
            signal.symbol,
            self.cfg.indicators.entry_timeframe,
            self.cfg.indicators.lookback_candles,
        )
        if series is None or not series.volumes:
            return (
                f"volume confirmation on but {signal.symbol} volume is "
                "unavailable"
            )
        period = int(self.d.volume_ma_period)
        if period < 2 or len(series.volumes) < period + 1:
            return (
                f"volume confirmation on but only {len(series.volumes)} bars "
                f"are available (needs {period + 1})"
            )
        average = sma(series.volumes, period)[-1]
        if average is None or average <= 0:
            return "volume confirmation on but the average is not computable"
        latest = series.volumes[-1]
        need = average * self.d.volume_multiplier
        if latest < need:
            return (
                f"volume {latest:,.0f} below {self.d.volume_multiplier:g}x the "
                f"{period}-bar average {average:,.0f}: move not confirmed"
            )
        return None

    def check_smart_money_alignment(self, signal: Signal) -> str | None:
        """Require the smart money factor to agree with the trade direction.

        Bison's rule: if the whale read points the other way, do not trade at
        all. This is the single highest-value gate in the whole agent.
        """
        if not self.d.require_smart_money_alignment:
            return None

        sm = signal.factors.get("smart_money")
        if sm is None:
            return "smart money factor missing entirely"
        if sm.confidence <= 0:
            return (
                "smart money alignment required but the factor has no data "
                f"({'; '.join(sm.reasons[:1]) or 'unknown reason'})"
            )

        if signal.action == LONG and sm.score <= 50:
            return f"smart money is not long (score {sm.score:.1f}) - signal rejected"
        if signal.action == SHORT and sm.score >= 50:
            return f"smart money is not short (score {sm.score:.1f}) - signal rejected"
        return None

    def check_loss_streak(self) -> str | None:
        """Halt after repeated losses instead of loosening rules.

        Dire Wolf lost 27% by doing the opposite: after losses it opened five
        parallel 25x positions. The correct response is to stop, not to
        increase size.
        """
        if self.state.halted_until > self.now():
            remaining = (self.state.halted_until - self.now()) / 60.0
            return f"halted for another {remaining:.0f} min: {self.state.halt_reason}"
        return None

    def check_confidence(self, signal: Signal) -> str | None:
        min_conf = self.cfg.min_confidence
        if signal.confidence < min_conf:
            return (
                f"confidence {signal.confidence:.0%} below minimum "
                f"{min_conf:.0%} - too many inputs missing"
            )
        return None

    def check_position_slots(self, open_positions: int) -> str | None:
        if open_positions >= self.cfg.risk.max_open_positions:
            return (
                f"already holding {open_positions} position(s), "
                f"limit is {self.cfg.risk.max_open_positions}"
            )
        return None

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    def evaluate(
        self, signal: Signal, open_positions: int = 0, is_reversal: bool = False
    ) -> Signal:
        """Apply every gate, recording the first one that blocks."""
        if signal.action == "neutral":
            signal.blocked_by = "score did not cross a threshold"
            return signal

        checks: list[str | None] = [
            self.check_loss_streak(),
            self.check_confidence(signal),
            self.check_smart_money_alignment(signal),
            self.check_macro_filter(signal),
            # Entry-quality gates. Deliberately before the cap/cooldown block and
            # outside the reversal exemption: a reversal is still an entry, and
            # both of these describe the bar being entered on rather than how
            # recently the symbol was traded.
            self.check_trend_strength(signal),
            self.check_volume(signal),
        ]

        # Cap and cooldown protect against churn; a reversal that closes an
        # opposite position is exempt from the count, but not from the cooldown.
        if not is_reversal:
            if self.daily_cap_reached(signal.symbol):
                checks.append(
                    f"daily cap reached ({self.d.max_signals_per_day} signals for "
                    f"{signal.symbol}) - trading frequency is the dominant loss driver"
                )
            if self.in_cooldown(signal.symbol):
                checks.append(
                    f"cooldown active, {self.cooldown_remaining_minutes(signal.symbol):.0f} min "
                    f"remaining (cooldown is {self.d.cooldown_minutes} min)"
                )
            checks.append(self.check_position_slots(open_positions))

        for reason in checks:
            if reason:
                signal.blocked_by = reason
                signal.action = "neutral"
                return signal

        return signal

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------
    def record_trade(self, symbol: str) -> None:
        self.state.roll_day_if_needed(self.now())
        self.state.last_trade_ts[symbol] = self.now()
        self.state.trades_today[symbol] = self.state.trades_today.get(symbol, 0) + 1

    def record_outcome(self, pnl: float, max_consecutive_losses: int = 3, halt_hours: float = 12.0) -> None:
        """Update the loss streak and halt if it gets too long."""
        if pnl < 0:
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= max_consecutive_losses:
                self.state.halted_until = self.now() + halt_hours * 3600
                self.state.halt_reason = (
                    f"{self.state.consecutive_losses} consecutive losses"
                )
        elif pnl > 0:
            self.state.consecutive_losses = 0

    def export_payload(self) -> dict:
        """Serialisable copy of the guardrail state, for persistence."""
        return self.state.to_payload()

    def restore_payload(self, data: dict) -> None:
        """Replace the guardrail state, typically from a previous run."""
        self.state = DisciplineState.from_payload(data or {})

    def snapshot(self) -> dict[str, object]:
        self.state.roll_day_if_needed(self.now())
        return {
            "trades_today": dict(self.state.trades_today),
            "last_trade_age_min": {
                sym: round((self.now() - ts) / 60, 1)
                for sym, ts in self.state.last_trade_ts.items()
            },
            "consecutive_losses": self.state.consecutive_losses,
            "halted_until": self.state.halted_until,
            "halt_reason": self.state.halt_reason,
            "btc_trend": self.btc_trend_direction(),
        }
