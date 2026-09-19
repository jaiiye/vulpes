"""Canonical research data store.

One schema for every record the backtest reads, regardless of whether it came
from the Hydromancer S3 archive (history) or the Hyperliquid API (forward).

This package is intentionally dependency-free and I/O-free. It defines shapes
and the mappings into them; reading and writing parquet, S3 or anything else
belongs to a later layer, so the storage decision can change without touching
the schema.
"""

from .normalize import (
    NormalizeError,
    account_value_from_hydromancer,
    candle_from_api,
    candle_from_hydromancer,
    funding_from_api,
    orderbook_from_api,
    orderbook_from_hydromancer,
    position_from_api,
    position_from_hydromancer,
)
from .schema import (
    CANONICAL_DATASETS,
    SOURCE_API,
    SOURCE_HYDROMANCER,
    AccountValueSnapshot,
    Candle,
    FundingRate,
    LeaderboardRow,
    OrderBookLevel,
    OrderBookSnapshot,
    PositionSnapshot,
    as_float,
    as_int,
    as_str,
    dataset_for,
    from_row,
    to_row,
)

__all__ = [
    "CANONICAL_DATASETS",
    "SOURCE_API",
    "SOURCE_HYDROMANCER",
    "AccountValueSnapshot",
    "Candle",
    "FundingRate",
    "LeaderboardRow",
    "NormalizeError",
    "OrderBookLevel",
    "OrderBookSnapshot",
    "PositionSnapshot",
    "account_value_from_hydromancer",
    "as_float",
    "as_int",
    "as_str",
    "candle_from_api",
    "candle_from_hydromancer",
    "dataset_for",
    "from_row",
    "funding_from_api",
    "orderbook_from_api",
    "orderbook_from_hydromancer",
    "position_from_api",
    "position_from_hydromancer",
    "to_row",
]
