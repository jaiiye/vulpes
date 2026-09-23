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
from .market_data import (
    MAINNET_URL,
    TESTNET_URL,
    HyperliquidMarket,
    MarketDataError,
    safe_candles,
)
from .synthesizer import LONG, SHORT, Journal, Signal

# Last-resort stop distance when ATR cannot be computed, as a fraction of price.
FALLBACK_STOP_FRACTION = 0.01

# Taker fee per leg, matching Hyperliquid's base tier (0.035%).
TAKER_FEE_RATE = 0.00035

# Below this notional, round-trip fees start to dominate one risk unit and the
# sizing notes say so. From the "below a few thousand dollars notional, fees
# and slippage dominate" rule.
SMALL_NOTIONAL_USD = 1_000.0

# The venue's minimum order value. Hyperliquid refuses anything below it with
# `Order must have minimum value of $10. asset=N`, and the rule is the number
# in that message.
#
# Measured on testnet, because it is narrower than "orders need $10". The
# exemption is for *trigger* orders, not for `reduce_only`:
#     entry             $  9.54   refused
#     entry             $ 10.40   accepted
#     reduce-only stop  $  9.03   accepted   <- trigger, exempt
#     reduce-only close $  5.21   refused    <- market, NOT exempt
#     reduce-only close $ 10.41   accepted
# The venue's size quantum on BTC is 0.00001 = $0.87 at $87k, wider than the
# remaining uncertainty about where between 9.54 and 10.40 the line sits.
#
# Where it is enforced, and why, is in `Broker._reject_small_order` and
# `Broker._market_close_confirmed`.
MIN_ORDER_NOTIONAL_USD = 10.0


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
    #: Cost charged on the ENTRY leg of a simulated fill, in USD. Carried
    #: rather than recomputed at exit for the same reason as
    #: `trailing_distance`: the rate in force when the position was opened is
    #: the one the exit has to pair with, and re-deriving it from a config read
    #: later would silently re-price a trade that is already open. Stays 0.0 on
    #: a real fill, where the venue's own fee is the truth.
    entry_fee_usd: float = 0.0
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
            # Persisted, not recomputed: see the field's own note. Without this
            # an open simulated position would come back from a restart with 0
            # and its exit would report a round trip that was charged one leg.
            "entry_fee_usd": self.entry_fee_usd,
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
            # 0.0 for a state file written before this field existed, which is
            # the honest reading: that entry was never charged a simulated fee,
            # so its exit must not pretend one was collected.
            entry_fee_usd=f("entry_fee_usd"),
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


@dataclass
class CloseResult:
    """What a close realised, and what it cost.

    Two figures rather than one, because they are used for two different things
    and only one of them is charged. `pnl` is GROSS, and is what the run stats
    and `discipline.record_outcome` read. `pnl_net` is what the trade actually
    left behind, and is what a human should be shown. Moving the first to the
    second would change when the guardrails trip on an unchanged trade
    sequence - a separate decision with its own break with historical logs, so
    the two are kept apart here rather than conflated.

    The fees are `None` on a real fill, NOT 0.0: the venue charged a real fee
    and this code never asked what it was, so 0.0 would report a free close.
    That is the same distinction `universe.correlation_or_none` draws between
    "there is no relation" and "it cannot be measured" - and 0.0 here would be
    the optimistic direction, which is the one that misleads.
    """

    pnl: float
    entry_fee_usd: float | None = None
    exit_fee_usd: float | None = None

    @property
    def costs_known(self) -> bool:
        return self.exit_fee_usd is not None

    @property
    def cost_usd(self) -> float:
        """Total simulated cost. Only meaningful when `costs_known`."""
        return (self.entry_fee_usd or 0.0) + (self.exit_fee_usd or 0.0)

    @property
    def pnl_net(self) -> float | None:
        """Net of simulated fees, or None when they were never measured.

        Deliberately not falling back to `pnl`: a caller that printed the gross
        figure as if it were net would be making exactly the claim this field
        exists to prevent.
        """
        if not self.costs_known:
            return None
        return self.pnl - self.cost_usd


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
    #: Attempts at building the SDK clients before giving up on this cycle.
    #:
    #: The SDK does its own HTTP with no retry of its own, and `Info.__init__`
    #: fetches spot metadata over the network - so this is the one client
    #: construction in the process that can fail for a reason unrelated to
    #: whether the request was right.
    SDK_INIT_ATTEMPTS = 3
    SDK_INIT_BACKOFF_SECONDS = 2.0

    def _build_clients(self, exchange_cls, info_cls, wallet, base_url, account_address):
        """Build both clients, retrying the transient failures.

        Raises `ExecutionError`, never the SDK's own exception type, because
        that is the only type the callers' handlers know. `Exchange` and `Info`
        are built in one expression so a failure in the second cannot leave the
        first assigned - see `_ensure_sdk` for why that mattered.
        """
        last_error: Exception | None = None
        for attempt in range(1, self.SDK_INIT_ATTEMPTS + 1):
            try:
                return (
                    exchange_cls(wallet, base_url, account_address=account_address),
                    info_cls(base_url, skip_ws=True),
                )
            except Exception as exc:  # noqa: BLE001 - re-raised as ExecutionError
                last_error = exc
                if attempt < self.SDK_INIT_ATTEMPTS:
                    time.sleep(self.SDK_INIT_BACKOFF_SECONDS * attempt)
        raise ExecutionError(
            f"could not initialise the Hyperliquid clients after "
            f"{self.SDK_INIT_ATTEMPTS} attempts: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def _ensure_sdk(self) -> None:
        """Build the exchange and info clients on first use, or raise.

        Two properties here are load-bearing, and both were missing.

        **Failures are `ExecutionError`, not whatever the SDK raised.** Every
        caller already handles `ExecutionError` and degrades honestly: the
        signal layer skips, and startup reconciliation refuses to trade while
        the true position is unknown. A raw `SSLError` out of `Info.__init__`
        went through all of those handlers untouched - it escaped
        `live_positions`, escaped `reconcile_startup`'s `except ExecutionError`,
        and then escaped `run()`'s `except CriticalExecutionError` *before* the
        loop that IS built to survive cycle errors had started. Measured live:
        one transient `UNEXPECTED_EOF_WHILE_READING` from the testnet endpoint
        killed the process outright, seconds after start.

        **The pair is assigned together, after both exist.** The guard at the
        top tests `_exchange`, so a half-built pair (`_exchange` assigned,
        `Info` having raised) was never rebuilt: every later call would fail on
        `_info is None` until the process restarted.
        """
        if self.dry_run or self._exchange is not None:
            return

        try:
            import eth_account  # noqa: F401
            from hyperliquid.exchange import Exchange
            from hyperliquid.info import Info
        except ImportError as exc:
            raise ExecutionError(
                "live trading requires the official SDK. Install with:\n"
                "  pip install hyperliquid-python-sdk\n"
                f"(import failed: {exc})"
            ) from exc

        key = os.getenv("HYPERLIQUID_PRIVATE_KEY", "")
        if not key:
            raise ExecutionError("HYPERLIQUID_PRIVATE_KEY is not set")

        base_url = self.routing_base_url
        wallet = eth_account.Account.from_key(key)
        account_address = os.getenv("HYPERLIQUID_ACCOUNT_ADDRESS") or wallet.address

        exchange, info = self._build_clients(
            Exchange, Info, wallet, base_url, account_address
        )
        self._exchange = exchange
        self._info = info
        # Retained for the order-status query, which is a plain /info call.
        self._base_url = base_url
        self._account_address = account_address

        # Cache per-symbol size decimals so orders are not rejected.
        try:
            self._cache_sz_decimals(self._info.meta())
        except Exception as exc:  # noqa: BLE001
            self.journal.event("warn", message=f"could not cache szDecimals: {exc}")

    @property
    def routing_base_url(self) -> str:
        """Where orders go, derived in one place.

        The SDK build and the metadata read both need it, and two derivations
        that disagreed would send orders to one venue while reading another's
        instrument list - which is the shape of the mistake that put a
        6-decimal size on a 5-decimal venue.
        """
        override = self.cfg.execution.base_url
        if override:
            return override.rstrip("/")
        return TESTNET_URL if self.cfg.execution.testnet else MAINNET_URL

    def _cache_sz_decimals(self, meta: dict[str, Any]) -> None:
        """Record each asset's `szDecimals` from a `meta` payload."""
        for asset in meta.get("universe", []):
            self._sz_decimals[str(asset.get("name", "")).upper()] = int(
                asset.get("szDecimals", self.SIZE_DECIMALS)
            )

    def cache_venue_metadata(self) -> None:
        """Populate the size-decimal cache from the venue's public endpoint.

        Deliberately not through the SDK. `_ensure_sdk` cannot run in a dry run
        - there are no credentials, and none are needed to read an instrument
        list - yet `_round_size` falls back to `SIZE_DECIMALS` whenever this
        cache is empty, and that fallback is what sent `0.000121` BTC, six
        decimals, to a venue that accepts five. A check whose entire purpose is
        keeping that off the wire has to be able to establish the fact in the
        mode people actually run.

        Raises `ExecutionError` when the venue cannot be read at all, which is
        deliberate: a process that cannot read the instrument list does not
        know what size to send, and the alternative is learning it from a
        rejected order later. Transient failures are the retry budget's job,
        not something to paper over by guessing.
        """
        try:
            meta = request_json(
                f"{self.routing_base_url}/info",
                payload={"type": "meta"},
                timeout=15,
                retries=3,
            )
        except HttpError as exc:
            raise ExecutionError(
                f"could not read the venue's instrument list from "
                f"{self.routing_base_url}: {exc}"
            ) from exc
        self._cache_sz_decimals(meta)

    def size_decimals(self, symbol: str) -> int | None:
        """`szDecimals` for `symbol`, or None when the cache does not have it.

        None rather than `SIZE_DECIMALS`: "this venue's granularity is unknown"
        and "it is six" are different facts, and only the first should stop a
        start-up.
        """
        return self._sz_decimals.get(symbol.upper())

    def _round_size(self, symbol: str, size: float) -> float:
        decimals = self._sz_decimals.get(symbol.upper(), self.SIZE_DECIMALS)
        factor = 10 ** decimals
        return round(size * factor) / factor

    def _round_price(self, symbol: str, price: float) -> float:
        """Round `price` to what the venue will accept.

        Perp prices carry at most 5 significant figures and at most
        `6 - szDecimals` decimals. **`Exchange.order()` does not apply this** -
        only the SDK's own `market_open`/`market_close` do, inside
        `_slippage_price`. So a limit price or a trigger price computed here
        reached the venue raw and came back as `Order has invalid price.`

        Not a theoretical concern: the first live testnet order this project
        ever placed was rejected for exactly this, at
        `price * (1 + limit_offset_bps/10000)` = `86677.81725` - ten significant
        figures against a limit of five. No test could have caught it, because
        the suite's fake exchange accepts any float; only the venue knows its
        own formatting rules, which is why this needed a real order to surface.

        The expression mirrors the SDK's `_slippage_price` on purpose rather
        than reimplementing the rule: two roundings that disagreed would put
        the bot's limit orders and the SDK's market orders at different prices
        for the same intent.
        """
        if price <= 0:
            raise ExecutionError(f"invalid price {price} for {symbol}")
        decimals = self._sz_decimals.get(symbol.upper(), self.SIZE_DECIMALS)
        # `max(0, ...)`: a negative decimal count would round to tens, which is
        # never what a price needs. The SDK has no such guard, but a venue
        # `szDecimals` above 6 would mean the metadata is wrong, not the price.
        return round(float(f"{price:.5g}"), max(0, 6 - decimals))

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
            perp_value = float(state["marginSummary"]["accountValue"])
        except Exception as exc:  # noqa: BLE001
            raise ExecutionError(f"could not read account equity: {exc}") from exc

        # A unified account keeps its collateral in the SPOT balance and reports
        # `accountValue` as 0 whenever it is flat. Reading the perp summary alone
        # therefore told a funded account it had nothing, and `compute_size`
        # refused every trade with "account equity is zero; cannot size a
        # position" - which is exactly what happened on the first live testnet
        # run, while the account could and did trade.
        #
        # Measured on that account, flat and then with a position:
        #     flat       accountValue 0.0000   spot 998.8869
        #     in a trade accountValue 5.2015   spot 998.8774  (spot barely moves)
        #     closed     accountValue 0.0000   spot 998.8600  (the loss came
        #                                                      out of spot)
        # So for this account type the spot balance IS the collateral, and the
        # realised PnL of every round trip lands in it. Unrealised PnL is
        # deliberately not added: its composition in `accountValue` is not
        # something I could establish from measurement, and a position size
        # built on a figure I cannot explain is worse than one that is slightly
        # conservative. Sizing happens while flat, where the two coincide.
        if self.is_unified_account(address):
            return self._spot_usdc(address)
        return perp_value

    def is_unified_account(self, address: str | None = None) -> bool:
        """Whether spot and perp share one balance on this account.

        `userAbstraction` answers this directly. The alternative was inferring
        it from the shape of the balances, and a guess here becomes a wrong
        position size. Unreadable answers False, i.e. keep the previous
        behaviour rather than invent a number.
        """
        address = address or self._account_address
        if not address:
            return False
        try:
            mode = self._info.post(
                "/info", {"type": "userAbstraction", "user": address}
            )
        except Exception:  # noqa: BLE001 - an old venue may not have the endpoint
            return False
        return str(mode).strip().strip('"') == "unifiedAccount"

    def _spot_usdc(self, address: str) -> float:
        """USDC held in the spot balance, which is where a unified account's
        collateral actually sits."""
        state = self._info.spot_user_state(address)
        for balance in state.get("balances") or []:
            if str(balance.get("coin", "")).upper() == "USDC":
                return float(balance.get("total", 0) or 0)
        return 0.0

    def _simulated_fee(self, notional: float) -> float:
        """Cost to charge on one leg of a simulated fill.

        Per leg, on that leg's OWN notional: the two legs are not the same size.
        A position that ran from 78,044.5 to 85,144.5 pays the fee on $99.98 and
        on $109.07, not twice on the entry - which is also why this cannot be
        folded into a single round-trip number at entry time.

        Called only on the simulated path. A real fill is not charged here: the
        venue charges the real fee, and the raw order result is what records it.
        """
        return notional * self.cfg.execution.fee_bps / 10_000.0

    def _reject_small_order(self, symbol: str, size: float, order_price: float) -> None:
        """Refuse to send an entry the venue would refuse for being too small.

        Checked here, before anything is sent, rather than left to the venue's
        rejection. The rejection arrives as an opaque string -
        `order rejected: 'Order must have minimum value of $10. asset=3'` -
        that gets journalled as a generic entry failure, so a small account
        logs `ENTER ...` and then `ORDER FAILED` on every signal without ever
        being told that the cause is its size. A round trip to learn something
        already known is also just waste.

        Deliberately NOT applied to the protective orders. They are exempt from
        the minimum (measured: a $9.03 reduce-only stop was accepted where a
        $5.21 reduce-only market close was refused), so requiring them to clear
        it would refuse entries that work.

        `size` must already be rounded and `order_price` the price the venue
        will see: the rule is on the order's value, so both halves matter.
        """
        value = size * order_price
        if value >= MIN_ORDER_NOTIONAL_USD:
            return
        raise ExecutionError(
            f"{symbol}: entry of {size:.6g} at {order_price:,.6g} is worth "
            f"${value:,.2f}, below the venue's ${MIN_ORDER_NOTIONAL_USD:,.0f} "
            "minimum order value; the order was not sent. Raise "
            "account_allocation_pct, the risk budget, or the account size to "
            "trade this symbol."
        )

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
        # The SDK comes first because `_round_size` needs `szDecimals`, which
        # `_ensure_sdk` caches from the venue's own metadata. Rounding before
        # that used the fallback of 6 decimals, so the FIRST live order after
        # startup could carry a size the venue cannot represent - measured,
        # `0.000121` BTC against `szDecimals = 5` came back `Order has invalid
        # size.`, and the message the agent logged was `order rejected: 0.0`.
        # A dry run needs no SDK and sends nothing, so it keeps the fallback.
        if not self.dry_run:
            self._ensure_sdk()

        size = self._round_size(symbol, size)
        if size <= 0:
            raise ExecutionError(f"computed size for {symbol} rounds to zero")

        is_buy = side == LONG
        notional = size * price

        # Checked for dry runs too. The constraint belongs to the venue, so a
        # simulation that skipped it would report an entry the live path
        # refuses - which is the one thing a dry run must not do.
        #
        # Valued at the order's own price, not `price`: a limit order carries
        # the offset and a market order the SDK's slippage. The offset is exact
        # here; the slippage is not modelled and moves the value by
        # `slippage_bps`, a few bp against the venue's $0.87 size quantum.
        offset = (
            self.cfg.execution.limit_offset_bps / 10_000.0
            if self.cfg.execution.order_type == "limit"
            else 0.0
        )
        self._reject_small_order(
            symbol, size, price * (1 + offset) if is_buy else price * (1 - offset)
        )

        #: The simulated entry's cost. Stays 0.0 on a real fill, where the venue
        #: charges its own fee and the raw order result records it.
        entry_fee = 0.0

        if self.dry_run:
            entry_fee = self._simulated_fee(notional)
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
                fee_bps=self.cfg.execution.fee_bps,
                fee_usd=round(entry_fee, 4),
                note=(
                    "dry_run active: no order was sent. fee_usd is the "
                    "configured estimate (execution.fee_bps), not a venue "
                    "charge, and it is recorded rather than deducted - "
                    "realised_pnl stays gross, by decision"
                ),
            )
        else:
            # `_ensure_sdk` already ran, above the rounding: it is idempotent.
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
                # Rounded, or the venue refuses it: `Exchange.order()` applies
                # no formatting rule of its own. See `_round_price`.
                limit_px = self._round_price(
                    symbol, price * (1 + offset) if is_buy else price * (1 - offset)
                )
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
                raise ExecutionError(f"order rejected: {self._order_error(result)}")

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
            entry_fee_usd=entry_fee,
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
    def _order_error(result: Any) -> str:
        """The venue's own words for a refused order.

        `_parse_order_result` collapses the error case to `("error", 0.0, 0.0)`,
        so the text was dropped on the floor and `open_position` reported
        `order rejected: 0.0` - naming neither the rule nor the field that
        violated it. The refusal text is the only part that says what to fix:
        `Order has invalid size.` and `Order must have minimum value of $10.`
        call for completely different changes, and the log could not tell them
        apart.
        """
        statuses = (
            (result or {}).get("response", {}).get("data", {}).get("statuses") or []
        )
        if statuses and isinstance(statuses[0], dict) and "error" in statuses[0]:
            return str(statuses[0]["error"])
        return repr(result)

    @staticmethod
    def _parse_order_result(result: Any) -> tuple[str, float, float]:
        """Classify an order response as ('filled'|'resting'|'error', size, price).

        Hyperliquid returns a per-order status object which is one of:
            {"filled": {"totalSz": "1.0", "avgPx": "100.0", "oid": N}}
            {"resting": {"oid": N}}
            {"error": "..."}

        The error case discards the message here; `_order_error` recovers it.
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
            # `_market_close_confirmed`, because a refused close returns a
            # status rather than raising, and treating that as a successful
            # unwind would hand back a *recoverable* error for a position that
            # is still open and still unprotected - skipping `on_unprotected`,
            # which exists to demand manual intervention.
            result = self._market_close_confirmed(symbol)
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
            # The trigger is used BOTH as the order price and as `triggerPx`, and
            # the venue validates both - so rounding it once here covers both.
            trigger = self._round_price(symbol, trigger)

            # The venue call is in its own `try` so the result check below cannot
            # be swallowed and re-wrapped by the transport handler.
            try:
                result = self._exchange.order(
                    symbol,
                    close_is_buy,
                    size,
                    trigger,
                    {"trigger": {"triggerPx": trigger, "isMarket": True, "tpsl": "sl" if label == "stop" else "tp"}},
                    reduce_only=True,
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

            # `order()` does NOT raise when the venue rejects an order - it
            # returns a status whose first element is `{"error": ...}`. The entry
            # path checks that via `_parse_order_result`; this path did not, so a
            # rejected stop would have been journaled as `protection_placed` and
            # the caller told the position was protected.
            #
            # That was hidden by the second bug this fixes: the journal call used
            # to pass `kind=label`, which collides with
            # `Journal.event(self, kind, **payload)` and raised before anything
            # was written. Renaming the field alone would have replaced a loud
            # TypeError with a silent success, which is strictly worse.
            state, _, _ = self._parse_order_result(result)
            if state == "error":
                note = (
                    f"the venue rejected the {label} for {symbol} at {trigger}: "
                    f"{result}"
                )
                self.journal.event(
                    "error", message=f"{note}. Position is unprotected."
                )
                raise ExecutionError(note)

            # Journaled outside the `try`: if this write fails the order IS on
            # the venue, so the caller must not be told the position is
            # unprotected and unwind a position that is in fact protected.
            # (`protection`, not `kind` - see above.)
            self.journal.event(
                "protection_placed",
                symbol=symbol,
                protection=label,
                trigger_px=trigger,
                result=result,
            )

    def _market_close_confirmed(self, symbol: str) -> Any:
        """`market_close`, with an unconfirmed close turned into an exception.

        `Exchange.market_close()` does NOT raise when the venue refuses the
        order - it returns the same `{"error": ...}` status `order()` does,
        which `_parse_order_result` already handles on the entry path. Both
        callers of this treated any returned value as a close, so a refused
        close was indistinguishable from a filled one:

        - `close_position` journalled `close_placed` and told its caller the
          position was gone. `bot.close_position` then added a fabricated PnL
          to the run stats, fed it to `discipline.record_outcome`, and cleared
          `self.position` - the agent walked away from exposure it still had,
          with an invented result in its guardrails.
        - `_unwind_unprotected` journalled `unwound_unprotected` and returned a
          *recoverable* error, so the agent carried on while the position sat
          open and unprotected, and `on_unprotected` - the path that demands
          manual intervention - was never called.

        A market order has no legitimate resting state, so this requires a fill
        rather than merely the absence of an error. Anything else leaves the
        position tracked for reconciliation, which is what both callers already
        do safely with a raise.

        Measured way for it to be refused: the minimum applies to a reduce-only
        *market* order - a $5.21 partial close of a $10.41 position came back
        `Order must have minimum value of $10` with the position unchanged -
        while a reduce-only *trigger* order is exempt. A position whose value
        falls below $10 therefore cannot be closed this way; the stop resting
        on the venue, which is exempt, is what closes it.
        """
        result = self._exchange.market_close(symbol)
        state, filled, _ = self._parse_order_result(result)
        if state != "filled":
            raise ExecutionError(
                f"market_close for {symbol} is not confirmed as filled (venue "
                f"returned {state!r}, filled {filled}): {result}"
            )
        return result

    def close_position(
        self, position: Position, price: float, reason: str
    ) -> CloseResult:
        """Close a position and report what it realised and what it cost.

        The PnL is computed from `price` (the mark price the caller observed)
        rather than from the venue's fill, because the SDK's `market_close`
        does not surface a fill price. In live mode this makes the tracked PnL
        a close approximation, not an exact accounting figure; reconciliation
        against the venue is what eventually settles it.

        Raises rather than returning a PnL when the close is not confirmed as
        filled. The caller is told the truth about whether the position is
        gone, and a refused close is precisely the case it must not mistake for
        one - see `_market_close_confirmed` for what that mistake cost.

        The returned `pnl` is GROSS of simulated fees, and deliberately so: the
        run stats and `discipline.record_outcome` read it, and moving them to
        net would change when the guardrails trip on an unchanged trade
        sequence. The net figure comes back alongside it in `CloseResult` and is
        journalled, so it can be read without changing anything that acts on it.
        """
        pnl = position.unrealized_pnl(price)

        if position.is_dry_run or self.dry_run:
            # Both legs, each on its own notional. The entry leg was charged
            # when the position was opened and carried here (see
            # `Position.entry_fee_usd`); the exit leg is charged now.
            exit_fee = self._simulated_fee(position.size * price)
            result = CloseResult(
                pnl=pnl,
                entry_fee_usd=position.entry_fee_usd,
                exit_fee_usd=exit_fee,
            )
            self.journal.event(
                "close_simulated",
                symbol=position.symbol,
                side=position.side,
                size=position.size,
                entry_price=position.entry_price,
                exit_price=price,
                pnl=round(pnl, 4),
                fee_bps=self.cfg.execution.fee_bps,
                fee_usd=round(exit_fee, 4),
                entry_fee_usd=round(position.entry_fee_usd, 4),
                pnl_net=round(result.pnl_net, 4),
                reason=reason,
                note=(
                    "dry_run active: no order was sent. fee_usd is the exit "
                    "leg's configured estimate, entry_fee_usd the entry leg's; "
                    "pnl_net = pnl - both"
                ),
            )
            return result

        # Not `result`: that name holds the `CloseResult` above, and the venue's
        # raw response is a different thing entirely.
        self._ensure_sdk()
        try:
            placed = self._market_close_confirmed(position.symbol)
        except ExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ExecutionError(f"market_close failed for {position.symbol}: {exc}") from exc
        self.journal.event(
            "close_placed",
            symbol=position.symbol,
            reason=reason,
            result=placed,
        )
        # Fees are None rather than 0.0: the venue charged its own and this code
        # never asked what they were, so any figure here would be invented.
        return CloseResult(pnl=pnl)

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
