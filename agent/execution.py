"""Order execution and position management.

Two modes:

  dry_run = True   Simulation. Orders are journalled, never sent. This is the
                   default and the only mode the loop will use without a key.
  dry_run = False  Live. Requires HYPERLIQUID_PRIVATE_KEY (an API wallet key,
                   never the main wallet key).

The live path uses the official `hyperliquid-python-sdk`, imported lazily so
that simulation and backtesting work with zero third-party dependencies.

Sizing follows the risk model rather than a fixed notional: the stop distance
(in ATR terms) determines the position size, so a volatile symbol automatically
gets a smaller position. That is the difference between "2% risk per trade"
being a real constraint and being decoration.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from .config import BotConfig
from .http_util import HttpError, request_json
from .indicators import IndicatorError
from .market_data import HyperliquidMarket, MarketDataError, safe_candles
from .synthesizer import LONG, SHORT, Journal, Signal

# Last-resort stop distance when ATR cannot be computed, as a fraction of price.
FALLBACK_STOP_FRACTION = 0.01

# Taker fee per leg, matching Hyperliquid's base tier (0.035%).
TAKER_FEE_RATE = 0.00035

# Below this notional, round-trip fees start to dominate one risk unit and the
# sizing notes say so. From the "below a few thousand dollars notional, fees
# and slippage dominate" rule.
SMALL_NOTIONAL_USD = 1_000.0


class ExecutionError(RuntimeError):
    """A recoverable execution problem. The loop logs it and continues."""


class CriticalExecutionError(ExecutionError):
    """A failure that must stop the agent immediately.

    Raised when the agent may be holding a position it cannot manage, for
    example an unprotected position that also failed to close. Continuing to
    trade in that state risks compounding an already unsafe exposure.
    """


@dataclass
class Position:
    """A tracked open position."""

    symbol: str
    side: str            # "long" | "short"
    size: float          # absolute coin size
    entry_price: float
    notional: float      # absolute USD notional at entry
    leverage: int
    stop_price: float | None = None
    take_profit_price: float | None = None
    opened_ts: float = field(default_factory=time.time)
    entry_score: float = 0.0
    entry_reasons: list[str] = field(default_factory=list)
    is_dry_run: bool = True
    # The risk budget this position was sized against, so PnL can be reported
    # in R multiples.
    risk_usd: float = 0.0
    #: How far behind the best price the trailing stop sits, fixed at entry.
    #: None when no trail is configured.
    trailing_distance: float | None = None
    #: How far price must move in favour before the trail may replace the
    #: initial stop, in price units. 0 means from the first bar.
    trailing_activation: float = 0.0
    #: Best price seen since entry - highest for a long, lowest for a short.
    #: 0.0 means "not yet initialised"; it is seeded from `entry_price` on the
    #: first ratchet so a position opened at an extreme is not immediately
    #: treated as having given back a move it never made.
    best_price: float = 0.0

    @property
    def signed_size(self) -> float:
        return self.size if self.side == "long" else -self.size

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.side == "long":
            return (mark_price - self.entry_price) * self.size
        return (self.entry_price - mark_price) * self.size

    def ratchet_stop(self, high: float, low: float) -> float | None:
        """Move the stop toward entry, behind the best price reached.

        Returns the stop after ratcheting, or None when no trail is configured.

        Only ever moves toward entry. A trail that loosens is worse than no
        trail at all: it would let a position give back more than the distance
        it was supposed to protect, while still looking like protection.

        Call this AFTER the bar's stop check, not before. Ratcheting on this
        bar's high and then testing the result against this bar's low assumes
        the high came first, and on a down bar it did not - which turns the
        trail into a same-bar exit at a price the market never offered.
        """
        if not self.trailing_distance or self.trailing_distance <= 0:
            return None
        if self.best_price <= 0:
            self.best_price = self.entry_price

        if self.side == "long":
            self.best_price = max(self.best_price, high)
            # Defer until the trade has actually moved. Without this the trail
            # replaces a wider initial stop on the first bar, and the initial
            # stop never binds at all.
            if self.best_price - self.entry_price < self.trailing_activation:
                return self.stop_price
            candidate = self.best_price - self.trailing_distance
            if self.stop_price is None or candidate > self.stop_price:
                self.stop_price = candidate
        else:
            self.best_price = min(self.best_price, low)
            if self.entry_price - self.best_price < self.trailing_activation:
                return self.stop_price
            candidate = self.best_price + self.trailing_distance
            if self.stop_price is None or candidate < self.stop_price:
                self.stop_price = candidate
        return self.stop_price

    def pnl_pct(self, mark_price: float) -> float:
        if self.entry_price <= 0:
            return 0.0
        if self.side == "long":
            return (mark_price - self.entry_price) / self.entry_price * 100
        return (self.entry_price - mark_price) / self.entry_price * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "size": self.size,
            "entry_price": self.entry_price,
            "notional": self.notional,
            "leverage": self.leverage,
            "stop_price": self.stop_price,
            "take_profit_price": self.take_profit_price,
            "opened_ts": self.opened_ts,
            "entry_score": self.entry_score,
            "entry_reasons": list(self.entry_reasons),
            "is_dry_run": self.is_dry_run,
            # The trail's own state, not just the stop it has produced so far.
            # Persisting the stop alone would look fine after a restart and
            # quietly stop trailing: the position would appear protected while
            # the stop stood still for the rest of its life.
            "trailing_distance": self.trailing_distance,
            "best_price": self.best_price,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Position":
        """Rebuild a Position from persisted state.

        Tolerant on purpose: a partially written or hand-edited state file
        should degrade to a usable Position rather than crash the startup
        reconciliation.
        """

        def f(key: str, default: float = 0.0) -> float:
            try:
                value = data.get(key)
                return default if value is None else float(value)
            except (TypeError, ValueError):
                return default

        side = str(data.get("side", "")).lower()
        if side not in (LONG, SHORT):
            raise ExecutionError(f"persisted position has invalid side {side!r}")
        symbol = str(data.get("symbol") or "")
        if not symbol:
            raise ExecutionError("persisted position has no symbol")

        stop = data.get("stop_price")
        take = data.get("take_profit_price")
        trail = data.get("trailing_distance")
        # A trail with no best price would ratchet from zero on the next tick and
        # snap the stop to nonsense, so `best_price` falls back to the entry
        # rather than to 0.0 - the same seeding the open path uses.
        best = f("best_price") if data.get("best_price") else f("entry_price")
        return cls(
            symbol=symbol.upper(),
            side=side,
            size=f("size"),
            entry_price=f("entry_price"),
            notional=f("notional"),
            leverage=int(f("leverage", 1)),
            stop_price=None if stop is None else f("stop_price"),
            take_profit_price=None if take is None else f("take_profit_price"),
            trailing_distance=None if trail is None else f("trailing_distance"),
            trailing_activation=f("trailing_activation"),
            best_price=best,
            opened_ts=f("opened_ts"),
            entry_score=f("entry_score"),
            entry_reasons=list(data.get("entry_reasons") or []),
            is_dry_run=bool(data.get("is_dry_run", True)),
            risk_usd=f("risk_usd"),
        )


def is_reversal_signal(position: Position | None, signal: Signal) -> bool:
    """True when `signal` is a directional call against the open `position`.

    Shared by the live cycle and the backtest so the two cannot disagree about
    what counts as a reversal.

    This was previously spelled out inline in both places. That is exactly how
    the backtest came to never reverse at all: it did not evaluate signals while
    holding a position, so the condition was unreachable there. One definition
    means one behaviour.
    """
    if position is None:
        return False
    if signal.action not in (LONG, SHORT):
        return False
    return position.side != signal.action


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


@dataclass
class SizingResult:
    size: float
    notional: float
    stop_price: float | None
    take_profit_price: float | None
    atr: float
    risk_usd: float
    #: Distance the trailing stop keeps from the best price reached, in price
    #: units. None when no trail is configured. Carried rather than recomputed
    #: per bar because it is fixed at entry: re-deriving it from a later ATR
    #: would let the trail widen, which is the one thing a trail must not do.
    trailing_distance: float | None = None
    #: How far price must move in favour before the trail may take over, in
    #: price units. Fixed at entry for the same reason as `trailing_distance`.
    trailing_activation: float = 0.0
    notes: list[str] = field(default_factory=list)


@dataclass
class RestingOrderOutcome:
    """What actually happened to a limit order that did not fill immediately.

    `cancelled` is deliberately separate from `filled_size`: an order can be
    partially filled and then cancelled, and an order that could NOT be
    cancelled is a materially different (and more dangerous) situation than one
    that was.
    """

    filled_size: float = 0.0
    avg_price: float = 0.0
    cancelled: bool = False
    oid: Any = None


def compute_size(
    market: HyperliquidMarket,
    config: BotConfig,
    symbol: str,
    side: str,
    price: float,
    equity: float,
) -> SizingResult:
    """Size a position from the risk budget and ATR stop distance."""
    if price <= 0:
        raise ExecutionError(f"invalid price {price} for {symbol}")
    if equity <= 0:
        raise ExecutionError("account equity is zero; cannot size a position")

    notes: list[str] = []
    r = config.risk

    # ATR from the entry timeframe gives a volatility-aware stop distance.
    atr_value = price * FALLBACK_STOP_FRACTION
    series = safe_candles(
        market, symbol, config.indicators.entry_timeframe, config.indicators.lookback_candles
    )
    if series is None:
        notes.append(
            f"candle fetch failed; falling back to "
            f"{FALLBACK_STOP_FRACTION:.0%} of price as the ATR estimate"
        )
    else:
        try:
            atr_series = series.atr(config.indicators.atr_period)
            if atr_series and atr_series[-1]:
                atr_value = atr_series[-1]
            else:
                notes.append(
                    f"ATR series too short; falling back to "
                    f"{FALLBACK_STOP_FRACTION:.0%} of price"
                )
        except IndicatorError:
            notes.append(
                f"ATR unavailable, using {FALLBACK_STOP_FRACTION:.0%} of price as "
                "stop distance"
            )

    # Stop distance, in order of precedence:
    #   1. an explicit ATR multiple, which scales with volatility
    #   2. a fixed percentage of price
    #   3. 1.5 ATR as the risk unit when no protective order is placed at all
    #
    # ATR first because a fixed percentage is wide on a quiet day and inside the
    # noise on a violent one, and the same instrument is both within a month.
    if r.stop_loss_enabled and r.stop_loss_atr_multiple > 0:
        stop_distance = atr_value * r.stop_loss_atr_multiple
        notes.append(
            f"stop is {r.stop_loss_atr_multiple:g} x ATR(14) = {stop_distance:.4g} "
            f"({stop_distance / price:.2%} of price)"
        )
    elif r.stop_loss_enabled:
        stop_distance = price * r.stop_loss_pct / 100.0
    else:
        stop_distance = atr_value * 1.5
        notes.append(
            f"stop_loss disabled: using {1.5:g} x ATR ({stop_distance:.4g}) as the "
            "risk unit; no protective order will be placed"
        )

    if stop_distance <= 0:
        stop_distance = price * FALLBACK_STOP_FRACTION
        notes.append(
            f"stop distance degenerate, fell back to "
            f"{FALLBACK_STOP_FRACTION:.0%} of price"
        )

    risk_usd = equity * r.risk_per_trade_pct / 100.0
    size_by_risk = risk_usd / stop_distance

    # Cap by the allocation budget. This is a *notional* budget expressed as a
    # share of equity; leverage caps the margin required, not the notional, so
    # applying it here would silently multiply exposure by the leverage factor.
    allocation_usd = equity * r.account_allocation_pct / 100.0
    if r.max_position_usd > 0:
        allocation_usd = min(allocation_usd, r.max_position_usd)
    size_by_alloc = allocation_usd / price

    size = min(size_by_risk, size_by_alloc)
    limiting = "risk" if size_by_risk <= size_by_alloc else "allocation"
    notional = size * price

    if limiting == "risk":
        notes.append(
            f"sized by risk budget: ${risk_usd:.2f} risk over {stop_distance:.4g} "
            f"stop distance -> {size:.6g} {symbol}"
        )
    else:
        notes.append(
            f"sized by allocation cap: {r.account_allocation_pct:g}% of ${equity:,.2f} "
            f"= ${allocation_usd:,.2f} notional -> {size:.6g} {symbol} "
            f"(margin ~${allocation_usd / max(r.leverage, 1):,.2f} at {r.leverage}x)"
        )

    # Warn when fees start to dominate, per the "below a few thousand dollars
    # notional, fees and slippage dominate" rule.
    if notional < SMALL_NOTIONAL_USD:
        round_trip_fee = notional * TAKER_FEE_RATE * 2
        notes.append(
            f"notional ${notional:,.2f} is small: at {TAKER_FEE_RATE:.3%} taker x2 the "
            f"round trip costs ~${round_trip_fee:.3f}, a large share of one risk unit "
            f"(${risk_usd:.2f})"
        )

    implied_leverage = notional / equity if equity else 0.0
    if implied_leverage > r.leverage:
        notes.append(
            f"implied leverage {implied_leverage:.2f}x exceeds configured "
            f"{r.leverage}x; position reduced"
        )
        size = (equity * r.leverage) / price
        notional = size * price

    stop_price = None
    take_profit_price = None
    trailing_distance = None
    trailing_activation = 0.0
    if r.stop_loss_enabled:
        stop_price = price - stop_distance if side == LONG else price + stop_distance
        # NOTE on the two multiples: the trail does NOT start where the stop is
        # unless the two multiples are equal. A 1.5 ATR trail against a 2 ATR
        # stop sits CLOSER to entry, so it takes over on the first ratchet and
        # the 2 ATR stop never binds at all - the effective stop becomes 1.5
        # ATR from entry, tightening on every new high. Measured on 170 days x
        # 3 markets: 326 of 328 exits became stop-outs, the trade count tripled
        # (the single position slot frees on every exit), and profit factor fell
        # from 1.21 to 0.86. Set `trailing_activation_atr_multiple` to keep the
        # initial stop until the trade has actually moved.
        if r.trailing_stop_atr_multiple > 0:
            trailing_distance = atr_value * r.trailing_stop_atr_multiple
            activation = r.trailing_activation_atr_multiple
            trailing_activation = atr_value * activation
            notes.append(
                f"trailing stop {r.trailing_stop_atr_multiple:g} x ATR(14) = "
                f"{trailing_distance:.4g}, ratcheting toward entry only"
                + (
                    f"; inactive until price moves {activation:g} x ATR in "
                    "favour, so the initial stop holds until then"
                    if activation > 0
                    else "; active from the first bar"
                )
            )
    if r.take_profit_enabled:
        tp_distance = price * r.take_profit_pct / 100.0
        take_profit_price = price + tp_distance if side == LONG else price - tp_distance

    # Symmetric by construction for both sides: reward and risk are distances
    # from entry, so the sign of the position does not change the ratio. The
    # earlier LONG-only guard simply omitted the note for shorts.
    if stop_price is not None and take_profit_price is not None:
        reward = abs(take_profit_price - price)
        risk = abs(price - stop_price)
        if risk > 0:
            notes.append(f"reward:risk = {reward / risk:.2f}")

    return SizingResult(
        size=size,
        notional=notional,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
        atr=atr_value,
        risk_usd=risk_usd,
        trailing_distance=trailing_distance,
        trailing_activation=trailing_activation,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------


class Broker:
    """Order placement and position management."""

    # Hyperliquid requires at least 5 significant figures on size.
    SIZE_DECIMALS = 6

    # How long to wait for a resting limit order before cancelling it.
    RESTING_FILL_TIMEOUT = 20.0
    RESTING_POLL_INTERVAL = 1.0

    def __init__(
        self,
        config: BotConfig,
        market: HyperliquidMarket,
        journal: Journal,
        on_unprotected=None,
    ) -> None:
        self.cfg = config
        self.market = market
        self.journal = journal
        self.dry_run = config.execution.dry_run
        self._exchange: Any = None
        self._info: Any = None
        self._sz_decimals: dict[str, int] = {}
        self._base_url: str | None = None
        self._account_address: str | None = None
        # Most recent raw order response, used to recover the order id so a
        # resting order can be polled and cancelled.
        self._last_result: Any = None
        # Callback invoked when a position could not be closed. The bot uses
        # this to persist the exposure and stop trading.
        self.on_unprotected = on_unprotected

    # ------------------------------------------------------------------
    # Live SDK plumbing
    # ------------------------------------------------------------------
    def _ensure_sdk(self) -> None:
        if self.dry_run or self._exchange is not None:
            return

        try:
            import eth_account  # noqa: F401
            from hyperliquid.exchange import Exchange
            from hyperliquid.info import Info
            from hyperliquid.utils import constants
        except ImportError as exc:
            raise ExecutionError(
                "live trading requires the official SDK. Install with:\n"
                "  pip install hyperliquid-python-sdk\n"
                f"(import failed: {exc})"
            ) from exc

        key = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
        if not key:
            raise ExecutionError("HYPERLIQUID_PRIVATE_KEY is not set")

        base_url = constants.TESTNET_API_URL if self.cfg.execution.testnet else constants.MAINNET_API_URL
        wallet = eth_account.Account.from_key(key)

        account_address = os.getenv("HYPERLIQUID_ACCOUNT_ADDRESS") or wallet.address
        self._exchange = Exchange(wallet, base_url, account_address=account_address)
        self._info = Info(base_url, skip_ws=True)
        # Retained for the order-status query, which is a plain /info call.
        self._base_url = base_url
        self._account_address = account_address

        # Cache per-symbol size decimals so orders are not rejected.
        try:
            meta = self._info.meta()
            for idx, asset in enumerate(meta.get("universe", [])):
                self._sz_decimals[str(asset.get("name", "")).upper()] = int(
                    asset.get("szDecimals", 6)
                )
        except Exception as exc:  # noqa: BLE001
            self.journal.event("warn", message=f"could not cache szDecimals: {exc}")

    def _round_size(self, symbol: str, size: float) -> float:
        decimals = self._sz_decimals.get(symbol.upper(), self.SIZE_DECIMALS)
        factor = 10 ** decimals
        return round(size * factor) / factor

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------
    def account_equity(self, fallback: float = 1000.0) -> float:
        """Account value in USD. Falls back to `fallback` in dry-run mode."""
        if self.dry_run:
            return float(os.getenv("DRY_RUN_EQUITY_USD", fallback))

        self._ensure_sdk()
        address = getattr(self._exchange, "account_address", None) or getattr(
            self._exchange, "wallet", None
        )
        try:
            state = self._info.user_state(address)
            return float(state["marginSummary"]["accountValue"])
        except Exception as exc:  # noqa: BLE001
            raise ExecutionError(f"could not read account equity: {exc}") from exc

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def open_position(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        leverage: int,
        stop_price: float | None = None,
        take_profit_price: float | None = None,
        entry_score: float = 0.0,
        entry_reasons: list[str] | None = None,
        trailing_distance: float | None = None,
        trailing_activation: float = 0.0,
    ) -> Position:
        """Open a position, reducing any opposite exposure first."""
        size = self._round_size(symbol, size)
        if size <= 0:
            raise ExecutionError(f"computed size for {symbol} rounds to zero")

        is_buy = side == LONG
        notional = size * price

        if self.dry_run:
            self.journal.event(
                "order_simulated",
                symbol=symbol,
                side=side,
                size=size,
                price=price,
                notional=round(notional, 2),
                leverage=leverage,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                note="dry_run active: no order was sent",
            )
        else:
            self._ensure_sdk()
            try:
                # Update leverage before sizing the order.
                self._exchange.update_leverage(leverage, symbol, is_cross=True)
            except Exception as exc:  # noqa: BLE001
                self.journal.event(
                    "warn", message=f"update_leverage failed for {symbol}: {exc}"
                )

            order_type = self.cfg.execution.order_type
            if order_type == "limit":
                offset = self.cfg.execution.limit_offset_bps / 10_000.0
                limit_px = price * (1 + offset) if is_buy else price * (1 - offset)
                result = self._exchange.order(
                    symbol, is_buy, size, limit_px, {"limit": {"tif": "Gtc"}}
                )
            else:
                slippage = self.cfg.execution.slippage_bps / 10_000.0
                result = self._exchange.market_open(symbol, is_buy, size, price, slippage)

            # Stash the raw response BEFORE anything else touches it. The
            # resting-order path recovers the order id from here in order to
            # poll and cancel it; without this assignment every resting limit
            # order was silently abandoned on the book.
            self._last_result = result
            self.journal.event(
                "order_submitted",
                symbol=symbol,
                side=side,
                requested_size=size,
                result=result,
            )

            # A limit order that rests unfilled is NOT a position. Treating it
            # as one produced a phantom position: the agent would track, stop
            # and later "close" exposure it never had.
            state, filled_size, avg_px = self._parse_order_result(result)
            if state == "error":
                raise ExecutionError(f"order rejected: {avg_px!r}")

            if state == "resting":
                resolved = self._resolve_resting_order(symbol, size)
                filled_size, avg_px = resolved.filled_size, resolved.avg_price
                if filled_size <= 0:
                    # Nothing was opened. `cancelled` records whether the stray
                    # order was actually removed from the book; if it was not,
                    # the caller must not be told the order is gone.
                    note = (
                        "order rested and was cancelled; no position opened"
                        if resolved.cancelled
                        else "order rested; cancellation could NOT be confirmed"
                    )
                    self.journal.event(
                        "order_unfilled",
                        symbol=symbol,
                        side=side,
                        requested_size=size,
                        cancelled=resolved.cancelled,
                        note=note,
                    )
                    if not resolved.cancelled:
                        # An order we cannot cancel may still fill later and
                        # create exposure we are not tracking, so stop rather
                        # than continue trading blind.
                        raise CriticalExecutionError(
                            f"limit order for {symbol} is live on the exchange but "
                            "could not be cancelled; refusing to continue with "
                            "unknown pending exposure"
                        )
                    raise ExecutionError(
                        f"limit order for {symbol} did not fill within "
                        f"{self.RESTING_FILL_TIMEOUT}s and was cancelled"
                    )

            # Use the exchange's actual fill, not our requested price.
            requested_size = size
            size = self._round_size(symbol, filled_size or size)
            price = avg_px or price
            notional = size * price

            self.journal.event(
                "order_filled",
                symbol=symbol,
                side=side,
                requested_size=requested_size,
                filled_size=size,
                avg_price=price,
            )

            # Protective orders are attached after the fill, because the size
            # is only known then. A failure here must never leave the position
            # open, so it is unwound immediately.
            try:
                self._attach_protection(
                    symbol, side, size, stop_price, take_profit_price
                )
            except ExecutionError as exc:
                raise self._unwind_unprotected(symbol, side, size, price, exc) from exc

        return Position(
            symbol=symbol.upper(),
            side=side,
            size=size,
            entry_price=price,
            notional=notional,
            leverage=leverage,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            entry_score=entry_score,
            entry_reasons=list(entry_reasons or []),
            is_dry_run=self.dry_run,
            trailing_distance=trailing_distance,
            trailing_activation=trailing_activation,
            # Seeded from the fill so the first ratchet measures the move from
            # where we actually got in, not from zero.
            best_price=price,
        )

    # ------------------------------------------------------------------
    # Fill verification
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_order_result(result: Any) -> tuple[str, float, float]:
        """Classify an order response as ('filled'|'resting'|'error', size, price).

        Hyperliquid returns a per-order status object which is one of:
            {"filled": {"totalSz": "1.0", "avgPx": "100.0", "oid": N}}
            {"resting": {"oid": N}}
            {"error": "..."}
        """
        statuses = (
            (result or {}).get("response", {}).get("data", {}).get("statuses") or []
        )
        if not statuses or not isinstance(statuses[0], dict):
            # Missing shape: treat as resting so the caller waits and verifies
            # rather than assuming a fill.
            return "resting", 0.0, 0.0

        first = statuses[0]
        if "error" in first:
            return "error", 0.0, 0.0
        if "filled" in first:
            info = first["filled"] or {}
            try:
                filled_size = float(info.get("totalSz", 0) or 0)
            except (TypeError, ValueError):
                filled_size = 0.0
            try:
                avg_px = float(info.get("avgPx", 0) or 0)
            except (TypeError, ValueError):
                avg_px = 0.0
            return "filled", filled_size, avg_px
        return "resting", 0.0, 0.0

    def _query_order_status(self, oid: Any) -> dict[str, Any] | None:
        """Fetch the current status of an order by id."""
        if self._info is None or not self._account_address:
            return None
        try:
            return request_json(
                f"{self._base_url}/info",
                payload={
                    "type": "orderStatus",
                    "user": self._account_address,
                    "oid": oid,
                },
                timeout=10,
                retries=1,
            )
        except HttpError:
            return None

    @staticmethod
    def _parse_order_status(payload: Any) -> tuple[str, float, float]:
        """Read (state, filled_size, avg_price) from an `orderStatus` response.

        Hyperliquid wraps the order under `order`, e.g.
            {"status": "order", "order": {"status": "filled",
                                          "filledSize": "1.0", "avgPx": "100.0"}}
        """
        order = payload.get("order") if isinstance(payload, dict) else None
        if not isinstance(order, dict):
            return "", 0.0, 0.0
        state = str(order.get("status", "")).lower()
        try:
            filled = float(order.get("filledSize", 0) or 0)
        except (TypeError, ValueError):
            filled = 0.0
        try:
            avg_px = float(order.get("avgPx", 0) or 0)
        except (TypeError, ValueError):
            avg_px = 0.0
        return state, filled, avg_px

    def _resolve_resting_order(self, symbol: str, size: float) -> "RestingOrderOutcome":
        """Wait briefly for a resting order to fill, then cancel the remainder.

        Returns an outcome describing what actually happened. `cancelled` is
        True only when removal was observed or confirmed, because an order that
        is still live can fill later and create exposure the agent is not
        tracking. Reporting an unverified cancellation was the defect that let
        stray orders accumulate on the book.
        """
        oid = self._last_order_id()
        if oid is None:
            self.journal.event(
                "error",
                message=(
                    f"resting order for {symbol} returned no order id; it can be "
                    "neither polled nor cancelled"
                ),
            )
            return RestingOrderOutcome(cancelled=False)

        filled_size = 0.0
        avg_px = 0.0
        cancelled = False
        rejected = False
        full_fill = False
        deadline = time.time() + self.RESTING_FILL_TIMEOUT

        while time.time() < deadline:
            time.sleep(self.RESTING_POLL_INTERVAL)
            state, observed_filled, observed_px = self._parse_order_status(
                self._query_order_status(oid)
            )
            if observed_filled > 0:
                filled_size, avg_px = observed_filled, observed_px

            if state in ("canceled", "cancelled"):
                cancelled = True
                break
            if state == "rejected":
                rejected = True
                break
            if state == "filled" or (size > 0 and filled_size >= size):
                full_fill = True
                break
        else:
            # Loop exhausted without a terminal state, so this is the timeout
            # path: cancel the remainder and re-read once so a partial fill
            # during the wait is captured rather than discarded.
            api_cancelled = self._cancel_order(symbol, oid)
            state, final_filled, final_px = self._parse_order_status(
                self._query_order_status(oid)
            )
            if final_filled > 0:
                filled_size, avg_px = final_filled, final_px

            # Prefer the venue's own state over "the cancel call did not
            # raise": an order can still be open after a call that returned
            # without error.
            if state in ("canceled", "cancelled"):
                cancelled = True
            elif state == "":
                cancelled = api_cancelled
            else:
                cancelled = False

        # Report a partial fill regardless of how the remainder ended; it is
        # materially different from either a full fill or no fill.
        if 0 < filled_size < size and not full_fill and not rejected:
            self.journal.event(
                "warn",
                message=(
                    f"{symbol} limit order partially filled {filled_size}/{size}"
                    + (" before the remainder was cancelled" if cancelled else "")
                ),
            )

        return RestingOrderOutcome(
            filled_size=filled_size, avg_price=avg_px, cancelled=cancelled, oid=oid
        )

    def _last_order_id(self) -> Any:
        """Extract the order id from the most recent submission."""
        result = self._last_result
        statuses = (
            (result or {}).get("response", {}).get("data", {}).get("statuses") or []
        )
        if statuses and isinstance(statuses[0], dict):
            for key in ("resting", "filled"):
                node = statuses[0].get(key)
                if isinstance(node, dict) and node.get("oid") is not None:
                    return node["oid"]
        return None

    def _cancel_order(self, symbol: str, oid: Any) -> bool:
        """Request cancellation of one order.

        Returns True when the SDK call did not raise. That is a *request*, not
        a confirmed removal: the caller re-reads the venue state to decide
        whether the order is genuinely gone.

        The SDK signature is `cancel(name, oid)`. An earlier version also tried
        `cancel(symbol, (oid, symbol))`, which passed a tuple where an order id
        was expected and could never have worked.
        """
        exchange = self._exchange
        if exchange is None:
            return False
        try:
            exchange.cancel(symbol, oid)
        except Exception as exc:  # noqa: BLE001 - the caller verifies state
            self.journal.event(
                "error", message=f"cancel request failed for {symbol} oid={oid}: {exc}"
            )
            return False
        self.journal.event("order_cancel_requested", symbol=symbol, oid=oid)
        return True

    # ------------------------------------------------------------------
    # Unwind
    # ------------------------------------------------------------------
    def _unwind_unprotected(
        self, symbol: str, side: str, size: float, price: float, cause: Exception
    ) -> ExecutionError:
        """Close a position whose protective orders could not be attached.

        Returns the exception the caller should raise: a recoverable
        ExecutionError if the exposure was closed, or a CriticalExecutionError
        if it could not be, in which case the agent must stop.
        """
        self.journal.event(
            "critical",
            message=(
                f"protection failed for {symbol} ({cause}); attempting immediate unwind"
            ),
        )
        try:
            result = self._exchange.market_close(symbol)
            self.journal.event(
                "unwound_unprotected",
                symbol=symbol,
                side=side,
                size=size,
                result=result,
            )
            return ExecutionError(
                f"protection failed for {symbol} ({cause}); position was closed "
                "immediately"
            )
        except Exception as close_exc:  # noqa: BLE001 - nothing left to try
            exposed = Position(
                symbol=symbol.upper(),
                side=side,
                size=size,
                entry_price=price,
                notional=size * price,
                leverage=self.cfg.risk.leverage,
                is_dry_run=False,
            )
            if self.on_unprotected is not None:
                try:
                    self.on_unprotected(exposed)
                except Exception as exc:  # noqa: BLE001 - never mask the real error
                    self.journal.event(
                        "error", message=f"on_unprotected handler failed: {exc}"
                    )
            return CriticalExecutionError(
                f"UNPROTECTED POSITION: {symbol} {side} {size} is open and its "
                f"protective orders failed ({cause}); the closing attempt also "
                f"failed ({close_exc}). Manual intervention required."
            )

    def _attach_protection(
        self,
        symbol: str,
        side: str,
        size: float,
        stop_price: float | None,
        take_profit_price: float | None,
    ) -> None:
        """Place reduce-only trigger orders. Live mode only."""
        if self._exchange is None:
            return

        close_is_buy = side != LONG  # closing a long means selling

        for label, trigger in (("stop", stop_price), ("take_profit", take_profit_price)):
            if trigger is None:
                continue
            try:
                self._exchange.order(
                    symbol,
                    close_is_buy,
                    size,
                    trigger,
                    {"trigger": {"triggerPx": trigger, "isMarket": True, "tpsl": "sl" if label == "stop" else "tp"}},
                    reduce_only=True,
                )
                self.journal.event(
                    "protection_placed", symbol=symbol, kind=label, trigger_px=trigger
                )
            except Exception as exc:  # noqa: BLE001
                # Loud, and the caller decides whether to unwind.
                self.journal.event(
                    "error",
                    message=(
                        f"FAILED to attach {label} for {symbol} at {trigger}: {exc}. "
                        "Position is unprotected."
                    ),
                )
                raise ExecutionError(
                    f"could not attach {label} order for {symbol}: {exc}"
                ) from exc

    def close_position(self, position: Position, price: float, reason: str) -> float:
        """Close a position and return the realised PnL.

        The PnL is computed from `price` (the mark price the caller observed)
        rather than from the venue's fill, because the SDK's `market_close`
        does not surface a fill price. In live mode this makes the tracked PnL
        a close approximation, not an exact accounting figure; reconciliation
        against the venue is what eventually settles it.
        """
        pnl = position.unrealized_pnl(price)

        if position.is_dry_run or self.dry_run:
            self.journal.event(
                "close_simulated",
                symbol=position.symbol,
                side=position.side,
                size=position.size,
                entry_price=position.entry_price,
                exit_price=price,
                pnl=round(pnl, 4),
                reason=reason,
                note="dry_run active: no order was sent",
            )
        else:
            self._ensure_sdk()
            try:
                result = self._exchange.market_close(position.symbol)
            except Exception as exc:  # noqa: BLE001
                raise ExecutionError(f"market_close failed for {position.symbol}: {exc}") from exc
            self.journal.event(
                "close_placed",
                symbol=position.symbol,
                reason=reason,
                result=result,
            )

        return pnl

    # ------------------------------------------------------------------
    def check_exits(self, position: Position, mark_price: float) -> str | None:
        """Return an exit reason if the position should be closed now."""
        if position.stop_price is not None:
            if position.side == LONG and mark_price <= position.stop_price:
                return f"stop loss hit at {mark_price:.4g} (stop {position.stop_price:.4g})"
            if position.side == SHORT and mark_price >= position.stop_price:
                return f"stop loss hit at {mark_price:.4g} (stop {position.stop_price:.4g})"

        if position.take_profit_price is not None:
            if position.side == LONG and mark_price >= position.take_profit_price:
                return (
                    f"take profit hit at {mark_price:.4g} "
                    f"(target {position.take_profit_price:.4g})"
                )
            if position.side == SHORT and mark_price <= position.take_profit_price:
                return (
                    f"take profit hit at {mark_price:.4g} "
                    f"(target {position.take_profit_price:.4g})"
                )

        # Fallback exits when no explicit protective orders are configured.
        # Without these, a disabled stop-loss would mean an unbounded loss.
        if position.stop_price is None:
            pnl_pct = position.pnl_pct(mark_price)
            hard_stop = self.cfg.safety.hard_stop_pct
            if pnl_pct <= -abs(hard_stop):
                return f"safety stop: adverse move {pnl_pct:.2f}% exceeds {hard_stop:g}%"

        return None

    def live_positions(self) -> list[Position]:
        """Every open position on the exchange, as the venue reports it.

        Raises ExecutionError on a read failure rather than returning an empty
        list: a failed read must never be indistinguishable from "flat",
        because that is exactly how an orphaned position goes unnoticed.

        Returns [] in dry-run mode, where there is no exchange truth. Callers
        must not use this for reconciliation in that mode.
        """
        if self.dry_run:
            return []

        self._ensure_sdk()
        try:
            address = self._account_address or getattr(
                self._exchange, "account_address", None
            )
            state = self._info.user_state(address)
        except Exception as exc:  # noqa: BLE001
            raise ExecutionError(
                f"could not read live positions for reconciliation: {exc}"
            ) from exc

        out: list[Position] = []
        for ap in state.get("assetPositions", []) or []:
            pos = ap.get("position") if isinstance(ap, dict) else None
            if not isinstance(pos, dict):
                continue
            coin = str(pos.get("coin", "")).upper()
            if not coin:
                continue
            try:
                szi = float(pos.get("szi", 0) or 0)
            except (TypeError, ValueError):
                continue
            if szi == 0:
                continue
            try:
                entry = float(pos.get("entryPx", 0) or 0)
            except (TypeError, ValueError):
                entry = 0.0

            out.append(
                Position(
                    symbol=coin,
                    side=LONG if szi > 0 else SHORT,
                    size=abs(szi),
                    entry_price=entry,
                    notional=abs(szi) * entry,
                    leverage=self.cfg.risk.leverage,
                    is_dry_run=False,
                )
            )
        return out

    def reconcile(self, symbol: str) -> Position | None:
        """The live position for one symbol, or None if flat."""
        name = symbol.upper()
        for position in self.live_positions():
            if position.symbol == name:
                return position
        return None

    def has_protective_orders(self, symbol: str) -> bool:
        """Whether the venue holds any trigger orders for `symbol`.

        Used after adopting a reconciled position to decide whether it is
        already protected, instead of assuming it is.
        """
        if self.dry_run or self._info is None:
            return False
        address = self._account_address or getattr(
            self._exchange, "account_address", None
        )
        try:
            orders = self._info.frontend_open_orders(address)
        except Exception:  # noqa: BLE001
            return False
        for order in orders or []:
            if not isinstance(order, dict):
                continue
            if str(order.get("coin", "")).upper() != symbol.upper():
                continue
            if order.get("isTrigger") or order.get("triggerPx") or order.get("isPositionTpsl"):
                return True
        return False
