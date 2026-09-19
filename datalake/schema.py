"""Canonical records for the research data store.

Every record the backtest reads passes through these types, whatever produced
it: the Hydromancer S3 archive for history, or the Hyperliquid API going
forward. Two sources with different field names and different conventions is
precisely the shape of problem that left the 40%-weight smart money factor
unbacktestable, so the mapping happens once, here, and is tested against both.

What this module deliberately is not: it does not read parquet, does not talk
to S3, and imports nothing beyond the standard library. Storage format is a
separate decision, and this is the layer that must not change when it does.

A note on `notional`. The archive documents `size` as signed (positive long,
negative short) but says nothing about whether `notional` carries a sign. That
ambiguity has to be resolved at the boundary, because `mark_price` and
`unrealized_pnl` are both derived from it and the derivations silently invert
if the sign is wrong. Every adapter therefore routes notional through
`signed_notional`, so the stored record is unambiguous regardless of what the
source did. When a convention has no single owner, it drifts - which is the
same failure that let an unsigned notional reach the factor's long/short ratio
undetected.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from agent.factors.smart_money import signed_notional

# --- Provenance ------------------------------------------------------------
# Stamped on every record. A backtest that mixes archived history with
# forward-collected data needs to be able to tell them apart, because the two
# do not have the same coverage or the same fidelity.
SOURCE_HYDROMANCER = "hydromancer"
SOURCE_API = "api"


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------
# Sources disagree about types: the archive emits decimals as strings, the API
# mixes floats and strings, and a hand-edited file can contain anything.
# Non-finite values are rejected for the same reason `agent/state.py` rejects
# them - `nan` compares false against everything, so it would break every
# downstream comparison silently instead of raising.


def as_float(value: Any, default: float | None = None) -> float | None:
    """Coerce to a finite float, or `default` if that is not possible."""
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def as_required_float(value: Any, field_name: str) -> float:
    """Coerce a value the record cannot exist without."""
    number = as_float(value)
    if number is None:
        raise ValueError(f"{field_name} is required and must be a finite number")
    return number


def as_int(value: Any, default: int | None = None) -> int | None:
    number = as_float(value)
    return default if number is None else int(number)


def as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class Candle:
    """One OHLCV bar.

    `volume_quote` and `trade_count` are optional because the API omits them
    and the archive provides them; keeping the columns means a forward-collected
    row and an archived row can sit in the same file without a schema change.
    """

    symbol: str
    interval: str
    time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str
    volume_quote: float | None = None
    trade_count: int | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Candle":
        return cls(
            symbol=as_str(row.get("symbol")),
            interval=as_str(row.get("interval")),
            time_ms=int(as_required_float(row.get("time_ms"), "time_ms")),
            open=as_required_float(row.get("open"), "open"),
            high=as_required_float(row.get("high"), "high"),
            low=as_required_float(row.get("low"), "low"),
            close=as_required_float(row.get("close"), "close"),
            volume=as_required_float(row.get("volume"), "volume"),
            source=as_str(row.get("source")),
            volume_quote=as_float(row.get("volume_quote")),
            trade_count=as_int(row.get("trade_count")),
        )


@dataclass
class FundingRate:
    """One settled hourly funding rate."""

    symbol: str
    time_ms: int
    rate: float
    source: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "FundingRate":
        return cls(
            symbol=as_str(row.get("symbol")),
            time_ms=int(as_required_float(row.get("time_ms"), "time_ms")),
            rate=as_required_float(row.get("rate"), "rate"),
            source=as_str(row.get("source")),
        )


@dataclass
class PositionSnapshot:
    """One wallet's open position in one market, at one instant.

    Distinct from `agent.factors.smart_money.WalletPosition`, which is the
    handful of fields the factor consumes. This is the storage record and keeps
    every column the sources provide, including ones nothing reads yet, so a
    future indicator is not blocked by a decision made today.

    Two fields the factor needs are absent from the archive and are derived
    below rather than stored. Deriving them for both sources is deliberate: a
    stored value from the API and a derived one for history would not be
    comparable, and comparing the two stretches is the entire point.
    """

    time_ms: int
    user: str
    market: str
    size: float
    notional: float
    entry_price: float
    source: str
    liquidation_price: float | None = None
    leverage: float = 0.0
    leverage_type: str = ""
    funding_pnl: float = 0.0
    account_value: float = 0.0
    account_mode: str = ""

    @property
    def mark_price(self) -> float:
        """Implied mark price.

        `notional` and `size` are both signed, so the signs cancel and this is
        positive for both directions. Returns 0.0 for a flat position, which
        has no meaningful mark.
        """
        if self.size == 0:
            return 0.0
        return self.notional / self.size

    @property
    def unrealized_pnl(self) -> float:
        """Derived, because the archive does not carry it.

        `notional - size * entry_price` is algebraically `(mark - entry) * size`,
        written so that one expression covers both directions: for a short,
        both `notional` and `size` are negative, so the subtraction still adds.

        Leaving this at zero is what made the backtest's `in_profit_pct`, and
        with it the factor's confidence, disagree with live by a measurable
        margin.
        """
        return self.notional - self.size * self.entry_price

    @property
    def side(self) -> str:
        if self.size > 0:
            return "long"
        if self.size < 0:
            return "short"
        return "flat"

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "PositionSnapshot":
        return cls(
            time_ms=int(as_required_float(row.get("time_ms"), "time_ms")),
            user=as_str(row.get("user")),
            market=as_str(row.get("market")),
            size=as_required_float(row.get("size"), "size"),
            notional=as_required_float(row.get("notional"), "notional"),
            entry_price=as_required_float(row.get("entry_price"), "entry_price"),
            source=as_str(row.get("source")),
            liquidation_price=as_float(row.get("liquidation_price")),
            leverage=as_float(row.get("leverage"), 0.0) or 0.0,
            leverage_type=as_str(row.get("leverage_type")),
            funding_pnl=as_float(row.get("funding_pnl"), 0.0) or 0.0,
            account_value=as_float(row.get("account_value"), 0.0) or 0.0,
            account_mode=as_str(row.get("account_mode")),
        )


@dataclass
class AccountValueSnapshot:
    """One wallet's account-level aggregate, per dex, per day.

    This is what the `min_account_value` filter reads, and what makes a wallet
    ranking comparable at a point in time.
    """

    time_ms: int
    user: str
    dex: str
    account_value: float
    source: str
    collateral_token: str = ""
    total_long_notional: float = 0.0
    total_short_notional: float = 0.0
    account_mode: str = ""

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "AccountValueSnapshot":
        return cls(
            time_ms=int(as_required_float(row.get("time_ms"), "time_ms")),
            user=as_str(row.get("user")),
            dex=as_str(row.get("dex")),
            account_value=as_required_float(
                row.get("account_value"), "account_value"
            ),
            source=as_str(row.get("source")),
            collateral_token=as_str(row.get("collateral_token")),
            total_long_notional=as_float(row.get("total_long_notional"), 0.0) or 0.0,
            total_short_notional=as_float(row.get("total_short_notional"), 0.0) or 0.0,
            account_mode=as_str(row.get("account_mode")),
        )


@dataclass
class LeaderboardRow:
    """A wallet's trailing performance over one ranking window.

    Reconstructed rather than archived. Neither source publishes a historical
    leaderboard, and the smart money factor's first step is selecting wallets
    by exactly this ranking, so without a reconstruction the factor's positions
    would be read from a wallet set that never existed.

    `reconstructed` is not decoration: a rebuilt ranking is an approximation of
    the live one, and a backtest that silently treats the two as equivalent is
    claiming more fidelity than it has.
    """

    time_ms: int
    user: str
    window: str
    pnl: float
    source: str
    roi: float = 0.0
    volume: float = 0.0
    account_value: float = 0.0
    reconstructed: bool = True

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "LeaderboardRow":
        return cls(
            time_ms=int(as_required_float(row.get("time_ms"), "time_ms")),
            user=as_str(row.get("user")),
            window=as_str(row.get("window")),
            pnl=as_required_float(row.get("pnl"), "pnl"),
            source=as_str(row.get("source")),
            roi=as_float(row.get("roi"), 0.0) or 0.0,
            volume=as_float(row.get("volume"), 0.0) or 0.0,
            account_value=as_float(row.get("account_value"), 0.0) or 0.0,
            reconstructed=bool(row.get("reconstructed", True)),
        )


@dataclass
class OrderBookLevel:
    """One price level of an L2 book."""

    price: float
    size: float
    order_count: int = 0

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "OrderBookLevel":
        return cls(
            price=as_required_float(row.get("price"), "price"),
            size=as_required_float(row.get("size"), "size"),
            order_count=as_int(row.get("order_count"), 0) or 0,
        )


@dataclass
class OrderBookSnapshot:
    """A full L2 book at one instant, up to 20 levels per side.

    Levels are stored outermost-last for bids (best bid first) and best-ask
    first for asks, matching both sources. The properties below are what the
    backtest actually wants: the current backtest never models slippage, it
    only warns about a constant assumed spread. A real book makes that
    measurable.
    """

    time_ms: int
    symbol: str
    source: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    block_number: int | None = None

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> float | None:
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2.0

    @property
    def spread_bps(self) -> float | None:
        """Quoted spread in basis points, or None when one side is empty."""
        bid, ask = self.best_bid, self.best_ask
        if bid is None or ask is None:
            return None
        mid = (bid + ask) / 2.0
        if mid <= 0:
            return None
        return (ask - bid) / mid * 10_000.0

    def depth_notional(self, side: str, within_bps: float) -> float:
        """USD resting within `within_bps` of the mid, on one side.

        This is the number a slippage model needs: how much size could be taken
        before the price moved by that much.
        """
        mid = self.mid_price
        if mid is None or mid <= 0:
            return 0.0
        levels = self.bids if side == "bid" else self.asks
        limit = mid * within_bps / 10_000.0
        total = 0.0
        for level in levels:
            if side == "bid" and level.price < mid - limit:
                break
            if side == "ask" and level.price > mid + limit:
                break
            total += level.price * level.size
        return total

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "OrderBookSnapshot":
        return cls(
            time_ms=int(as_required_float(row.get("time_ms"), "time_ms")),
            symbol=as_str(row.get("symbol")),
            source=as_str(row.get("source")),
            bids=[OrderBookLevel.from_row(b) for b in row.get("bids") or []],
            asks=[OrderBookLevel.from_row(a) for a in row.get("asks") or []],
            block_number=as_int(row.get("block_number")),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Names are the on-disk dataset names, so the store layer and the schema cannot
# drift apart: a dataset that is not registered here cannot be written.
CANONICAL_DATASETS: dict[str, type] = {
    "candles": Candle,
    "funding": FundingRate,
    "positions": PositionSnapshot,
    "account_values": AccountValueSnapshot,
    "leaderboard": LeaderboardRow,
    "orderbook": OrderBookSnapshot,
}

# Field-name sets per record class, built once on first use.
_ROW_KEY_CACHE: dict[type, set[str]] = {}


def to_row(record: Any) -> dict[str, Any]:
    """Flatten a canonical record into a storage row.

    Derived properties are deliberately absent: they are recomputed on read so
    a fix to the derivation applies to history already written. Storing them
    would freeze today's formula into yesterday's data.
    """
    return asdict(record)


def dataset_for(dataset: str) -> type:
    """The record class for a dataset name, raising a useful error if unknown."""
    record_cls = CANONICAL_DATASETS.get(dataset)
    if record_cls is None:
        raise KeyError(
            f"unknown dataset {dataset!r}; known: {sorted(CANONICAL_DATASETS)}"
        )
    return record_cls


def _row_keys(record_cls: type) -> set[str]:
    """Field names the record accepts, cached because it is asked per row."""
    cached = _ROW_KEY_CACHE.get(record_cls)
    if cached is None:
        cached = {f.name for f in fields(record_cls)}
        _ROW_KEY_CACHE[record_cls] = cached
    return cached


def from_row(dataset: str, row: dict[str, Any]) -> Any:
    """Rebuild a canonical record from a storage row.

    Keys the record does not declare are dropped rather than forwarded. Rows
    are written by whichever source produced them and may carry columns this
    version does not know about yet; a reader that rejected them would be
    unable to read data written by a later version of itself.
    """
    record_cls = dataset_for(dataset)
    keys = _row_keys(record_cls)
    return record_cls.from_row({k: v for k, v in row.items() if k in keys})
