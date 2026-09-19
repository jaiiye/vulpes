"""Adapters from each source's own shape into the canonical records.

Both sources are adapted here and nowhere else. The alternative - letting the
backtest know which source a row came from - is how you end up with two
slightly different definitions of "notional" that only disagree on shorts.

Two asymmetries worth knowing about:

* The archive's snapshot tables carry **no timestamp column**. Time survives
  only in the partition path and the file name, so both adapters take the
  instant as an argument. There is no sensible default.
* The API's candle keys are single letters (`o`, `h`, `l`, `c`, `v`, `n`, `t`).
  Single-letter keys are easy to transpose and a transposed `h`/`l` pair
  produces bars that still look plausible, so `_validate_ohlc` rejects them
  rather than letting a silent mis-mapping into the store.
"""

from __future__ import annotations

from typing import Any

from agent.factors.smart_money import signed_notional

from .schema import (
    SOURCE_API,
    SOURCE_HYDROMANCER,
    AccountValueSnapshot,
    Candle,
    FundingRate,
    OrderBookLevel,
    OrderBookSnapshot,
    PositionSnapshot,
    as_float,
    as_int,
    as_str,
)


class NormalizeError(ValueError):
    """A source row could not be turned into a canonical record."""


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


def _validate_ohlc(
    symbol: str, interval: str, time_ms: int, o: float, h: float, l: float, c: float
) -> None:
    """Reject bars that cannot describe a real traded range.

    A transposed high/low yields a bar with h < l, and a column-order mistake
    can put volume into a price. Both would silently poison every indicator
    downstream - the backtest would still run and still print numbers.
    """
    if l > h:
        raise NormalizeError(
            f"{symbol} {interval} @ {time_ms}: low {l} exceeds high {h}"
        )
    for label, value in (("open", o), ("high", h), ("low", l), ("close", c)):
        if value <= 0:
            raise NormalizeError(
                f"{symbol} {interval} @ {time_ms}: {label} is {value}, "
                "which is not a price"
            )
    if not (l <= o <= h) or not (l <= c <= h):
        raise NormalizeError(
            f"{symbol} {interval} @ {time_ms}: open/close outside the {l}-{h} range"
        )


# ---------------------------------------------------------------------------
# Candles
# ---------------------------------------------------------------------------


def candle_from_api(row: dict[str, Any]) -> Candle:
    """A `candleSnapshot` element.

    Shape: {t, T, s, i, o, h, l, c, v, n}. `t` is the bar's open time in ms.
    The API has no quote-volume column, so that stays None rather than being
    invented from a price times a volume the API did not promise.
    """
    symbol = as_str(row.get("s") or row.get("symbol")).upper()
    interval = as_str(row.get("i") or row.get("interval"))
    time_ms = as_int(row.get("t", row.get("time")))
    if time_ms is None:
        raise NormalizeError(f"{symbol}: candle has no open time")

    o = as_float(row.get("o"))
    h = as_float(row.get("h"))
    l = as_float(row.get("l"))
    c = as_float(row.get("c"))
    if None in (o, h, l, c):
        raise NormalizeError(f"{symbol} {interval} @ {time_ms}: incomplete OHLC")

    volume = as_float(row.get("v"), 0.0) or 0.0
    _validate_ohlc(symbol, interval, time_ms, o, h, l, c)

    return Candle(
        symbol=symbol,
        interval=interval,
        time_ms=time_ms,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=volume,
        source=SOURCE_API,
        volume_quote=None,
        trade_count=as_int(row.get("n")),
    )


def candle_from_hydromancer(row: dict[str, Any]) -> Candle:
    """A row from `by_dex/<dex>/candles/1s/date=.../candles.parquet`.

    The archive stores one-second bars only, so the interval is not a choice
    the caller gets to make. Coarser intervals are produced by aggregating
    these, not by reading a different file.
    """
    symbol = as_str(row.get("coin") or row.get("base_symbol")).upper()
    time_ms = as_int(row.get("timestamp", row.get("time_ms")))
    if time_ms is None:
        raise NormalizeError(f"{symbol}: candle has no timestamp")

    o = as_float(row.get("open"))
    h = as_float(row.get("high"))
    l = as_float(row.get("low"))
    c = as_float(row.get("close"))
    if None in (o, h, l, c):
        raise NormalizeError(f"{symbol} 1s @ {time_ms}: incomplete OHLC")

    volume = as_float(row.get("volume"), 0.0) or 0.0
    _validate_ohlc(symbol, "1s", time_ms, o, h, l, c)

    return Candle(
        symbol=symbol,
        interval="1s",
        time_ms=time_ms,
        open=o,
        high=h,
        low=l,
        close=c,
        volume=volume,
        source=SOURCE_HYDROMANCER,
        volume_quote=as_float(row.get("volume_quote")),
        trade_count=as_int(row.get("trade_count")),
    )


# ---------------------------------------------------------------------------
# Funding
# ---------------------------------------------------------------------------


def funding_from_api(row: dict[str, Any]) -> FundingRate:
    """A `fundingHistory` element: {coin, fundingRate, premium, time}."""
    symbol = as_str(row.get("coin") or row.get("symbol")).upper()
    time_ms = as_int(row.get("time", row.get("time_ms")))
    rate = as_float(row.get("fundingRate", row.get("rate")))
    if time_ms is None or rate is None:
        raise NormalizeError(f"{symbol}: funding row missing time or rate")
    return FundingRate(
        symbol=symbol, time_ms=time_ms, rate=rate, source=SOURCE_API
    )


# ---------------------------------------------------------------------------
# Positions
# ---------------------------------------------------------------------------


def position_from_hydromancer(
    row: dict[str, Any], time_ms: int
) -> PositionSnapshot:
    """A row from the daily perp snapshot table.

    `time_ms` is not optional: the table has no timestamp column, so the caller
    must supply the partition instant. Defaulting it to "now" would stamp
    fifteen months of history with today's date.
    """
    user = as_str(row.get("user")).lower()
    market = as_str(row.get("market")).upper()
    size = as_float(row.get("size"))
    if size is None:
        raise NormalizeError(f"{market}: position for {user} has no size")
    if size == 0:
        raise NormalizeError(
            f"{market}: position for {user} has zero size; a flat position "
            "carries no information and should not be stored"
        )

    raw_notional = as_float(row.get("notional"), 0.0) or 0.0
    entry = as_float(row.get("entry_price"), 0.0) or 0.0

    return PositionSnapshot(
        time_ms=time_ms,
        user=user,
        market=market,
        size=size,
        # The archive does not document whether its notional is signed, and the
        # derivations below invert silently if it is not. Forcing the sign from
        # `size` removes the question for every reader downstream.
        notional=signed_notional(raw_notional, size),
        entry_price=entry,
        source=SOURCE_HYDROMANCER,
        liquidation_price=as_float(row.get("liquidation_price")),
        leverage=as_float(row.get("leverage"), 0.0) or 0.0,
        leverage_type=as_str(row.get("leverage_type")),
        funding_pnl=as_float(row.get("funding_pnl"), 0.0) or 0.0,
        account_value=as_float(row.get("account_value"), 0.0) or 0.0,
        account_mode=as_str(row.get("account_mode")),
    )


def position_from_api(
    user: str, row: dict[str, Any], time_ms: int
) -> PositionSnapshot:
    """One entry of `assetPositions[].position` from `clearinghouseState`.

    The API already reports `unrealizedPnl` and a signed `positionValue`, but
    neither is trusted here. The canonical record derives PnL from notional and
    entry price so that a forward-collected row and an archived row are
    comparable - storing the API's own figure would make the two stretches
    differ for reasons that have nothing to do with the market.
    """
    market = as_str(row.get("coin") or row.get("market")).upper()
    size = as_float(row.get("szi", row.get("size")))
    if size is None:
        raise NormalizeError(f"{market}: API position has no size")
    if size == 0:
        raise NormalizeError(
            f"{market}: API position for {user} has zero size; skipped"
        )

    raw_notional = as_float(row.get("positionValue"), 0.0) or 0.0
    entry = as_float(row.get("entryPx", row.get("entry_price")), 0.0) or 0.0

    leverage_block = row.get("leverage")
    leverage_value = 0.0
    leverage_type = ""
    if isinstance(leverage_block, dict):
        leverage_value = as_float(leverage_block.get("value"), 0.0) or 0.0
        leverage_type = as_str(leverage_block.get("type"))
    else:
        leverage_value = as_float(leverage_block, 0.0) or 0.0

    cum_funding = row.get("cumFunding")
    funding_pnl = 0.0
    if isinstance(cum_funding, dict):
        funding_pnl = as_float(cum_funding.get("allTime"), 0.0) or 0.0

    return PositionSnapshot(
        time_ms=time_ms,
        user=as_str(user).lower(),
        market=market,
        size=size,
        notional=signed_notional(raw_notional, size),
        entry_price=entry,
        source=SOURCE_API,
        liquidation_price=as_float(row.get("liquidationPx")),
        leverage=leverage_value,
        leverage_type=leverage_type,
        funding_pnl=funding_pnl,
        account_value=as_float(row.get("accountValue"), 0.0) or 0.0,
        account_mode=as_str(row.get("accountMode")),
    )


# ---------------------------------------------------------------------------
# Account values
# ---------------------------------------------------------------------------


def account_value_from_hydromancer(
    row: dict[str, Any], time_ms: int
) -> AccountValueSnapshot:
    """A row from `global/snapshots/account_values/date=...`.

    This is the table the `min_account_value` filter reads. Note that its
    `account_value` is denominated in the dex's collateral token, so rows from
    different dexes are not comparable without conversion.
    """
    return AccountValueSnapshot(
        time_ms=time_ms,
        user=as_str(row.get("user")).lower(),
        dex=as_str(row.get("dex")),
        account_value=as_float(row.get("account_value"), 0.0) or 0.0,
        source=SOURCE_HYDROMANCER,
        collateral_token=as_str(row.get("collateral_token")),
        total_long_notional=as_float(row.get("total_long_notional"), 0.0) or 0.0,
        total_short_notional=as_float(row.get("total_short_notional"), 0.0) or 0.0,
        account_mode=as_str(row.get("account_mode")),
    )


# ---------------------------------------------------------------------------
# Order book
# ---------------------------------------------------------------------------


def _levels_from_pairs(pairs: Any) -> list[OrderBookLevel]:
    """Convert either source's level encoding into canonical levels.

    The archive uses `{px, sz, n}` with decimal strings; the API uses the same
    keys with string values. Both are accepted so the two paths cannot drift.
    """
    levels: list[OrderBookLevel] = []
    for entry in pairs or []:
        if isinstance(entry, dict):
            price = as_float(entry.get("px", entry.get("price")))
            size = as_float(entry.get("sz", entry.get("size")))
            count = as_int(entry.get("n", entry.get("order_count")), 0) or 0
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            price = as_float(entry[0])
            size = as_float(entry[1])
            count = as_int(entry[2], 0) if len(entry) > 2 else 0
            count = count or 0
        else:
            continue
        if price is None or size is None:
            continue
        levels.append(OrderBookLevel(price=price, size=size, order_count=count))
    return levels


def orderbook_from_hydromancer(row: dict[str, Any]) -> OrderBookSnapshot:
    """A row from `by_dex/<dex>/orderbook/1m/perps/date=.../<coin>.parquet`.

    One row is one complete book, with both sides as lists of structs. Levels
    arrive best-first on both sides and fewer than 20 when the book is thin;
    the adapter preserves that rather than padding, because a padded level
    would be read as real resting size by a slippage model.
    """
    symbol = as_str(row.get("symbol") or row.get("coin")).upper()
    time_ms = as_int(row.get("block_time_ms", row.get("time_ms")))
    if time_ms is None:
        raise NormalizeError(f"{symbol}: order book has no timestamp")
    return OrderBookSnapshot(
        time_ms=time_ms,
        symbol=symbol,
        source=SOURCE_HYDROMANCER,
        bids=_levels_from_pairs(row.get("bids")),
        asks=_levels_from_pairs(row.get("asks")),
        block_number=as_int(row.get("block_number")),
    )


def orderbook_from_api(row: dict[str, Any]) -> OrderBookSnapshot:
    """An `l2Book` response: {coin, time, levels: [bids, asks]}."""
    symbol = as_str(row.get("coin") or row.get("symbol")).upper()
    time_ms = as_int(row.get("time", row.get("time_ms")))
    if time_ms is None:
        raise NormalizeError(f"{symbol}: order book has no timestamp")
    levels = row.get("levels") or [[], []]
    bids = _levels_from_pairs(levels[0] if len(levels) > 0 else [])
    asks = _levels_from_pairs(levels[1] if len(levels) > 1 else [])
    return OrderBookSnapshot(
        time_ms=time_ms,
        symbol=symbol,
        source=SOURCE_API,
        bids=bids,
        asks=asks,
    )
