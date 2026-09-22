"""Hyperliquid market data access.

Read-only public endpoints, so this layer works with no credentials at all.
Doc: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .http_util import HttpError, request_json
from .indicators import CandleSeries, IndicatorError

MAINNET_URL = "https://api.hyperliquid.xyz"
TESTNET_URL = "https://api.hyperliquid-testnet.xyz"

# The leaderboard lives on a separate stats host, not the /info endpoint.
MAINNET_STATS_URL = "https://stats-data.hyperliquid.xyz/Mainnet"

# Hyperliquid expresses candle intervals as duration strings.
TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


class MarketDataError(RuntimeError):
    """Raised when market data cannot be retrieved or parsed."""


@dataclass
class AssetContext:
    """Per-asset context from the `metaAndAssetCtxs` endpoint.

    Note on units: `openInterest` from the API is a *base coin quantity*, not
    a USD value. Verified against mainnet: BTC reports ~37,400 which at
    ~$77k is ~$2.9B notional, the plausible figure. Multiply by mark price
    to get the notional.
    """

    index: int
    name: str
    mark_price: float = 0.0
    oracle_price: float = 0.0
    funding: float = 0.0
    open_interest: float = 0.0        # base coin units
    prev_day_price: float = 0.0
    day_volume: float = 0.0           # USD notional (dayNtlVlm)

    @property
    def funding_apr_pct(self) -> float:
        """Annualise the hourly funding rate into a percentage."""
        return self.funding * 24 * 365 * 100

    @property
    def open_interest_notional(self) -> float:
        """Open interest converted to USD."""
        price = self.mark_price or self.oracle_price
        return self.open_interest * price

    @property
    def day_change_pct(self) -> float:
        if self.prev_day_price <= 0:
            return 0.0
        return (self.mark_price - self.prev_day_price) / self.prev_day_price * 100


class HyperliquidMarket:
    """Thin read-only client for Hyperliquid public market endpoints."""

    def __init__(self, testnet: bool = True, base_url: str | None = None) -> None:
        self.base_url = (base_url or (TESTNET_URL if testnet else MAINNET_URL)).rstrip("/")
        self.testnet = testnet
        self._asset_map: dict[str, int] | None = None
        self._universe: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Low level
    # ------------------------------------------------------------------
    def info(self, payload: dict[str, Any]) -> Any:
        try:
            return request_json(f"{self.base_url}/info", payload=payload)
        except HttpError as exc:
            raise MarketDataError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Universe / metadata
    # ------------------------------------------------------------------
    def load_universe(self) -> list[dict[str, Any]]:
        """Fetch perp metadata once and cache the name -> index mapping."""
        if self._universe is not None:
            return self._universe

        data = self.info({"type": "meta"})
        if not isinstance(data, dict) or "universe" not in data:
            raise MarketDataError(f"unexpected meta response: {str(data)[:200]}")

        universe = data["universe"]
        self._universe = universe
        self._asset_map = {
            str(asset.get("name", "")).upper(): idx for idx, asset in enumerate(universe)
        }
        return universe

    def asset_index(self, symbol: str) -> int:
        self.load_universe()
        assert self._asset_map is not None
        key = symbol.upper()
        if key not in self._asset_map:
            raise MarketDataError(f"symbol `{key}` is not listed on Hyperliquid perps")
        return self._asset_map[key]

    def asset_names(self) -> list[str]:
        self.load_universe()
        assert self._asset_map is not None
        return [name for name, _ in sorted(self._asset_map.items(), key=lambda kv: kv[1])]

    # ------------------------------------------------------------------
    # Prices and context
    # ------------------------------------------------------------------
    def all_mids(self) -> dict[str, float]:
        data = self.info({"type": "allMids"})
        if not isinstance(data, dict):
            raise MarketDataError("unexpected allMids response")
        out: dict[str, float] = {}
        for name, price in data.items():
            try:
                out[str(name).upper()] = float(price)
            except (TypeError, ValueError):
                continue
        return out

    def mid_price(self, symbol: str) -> float:
        name = symbol.upper()
        mids = self.all_mids()
        if name not in mids:
            raise MarketDataError(f"no mid price for `{name}`")
        return mids[name]

    def asset_contexts(self) -> list[AssetContext]:
        """Fetch mark price, funding, OI and 24h price for every perp."""
        self.load_universe()
        data = self.info({"type": "metaAndAssetCtxs"})
        if not isinstance(data, list) or len(data) < 2:
            raise MarketDataError("unexpected metaAndAssetCtxs response")

        universe = data[0].get("universe", []) if isinstance(data[0], dict) else []
        ctxs = data[1] if isinstance(data[1], list) else []

        out: list[AssetContext] = []
        for idx, raw in enumerate(ctxs):
            if not isinstance(raw, dict):
                continue
            name = ""
            if idx < len(universe) and isinstance(universe[idx], dict):
                name = str(universe[idx].get("name", "")).upper()
            if not name:
                continue
            out.append(
                AssetContext(
                    index=idx,
                    name=name,
                    mark_price=_f(raw.get("markPx")),
                    oracle_price=_f(raw.get("oraclePx")),
                    funding=_f(raw.get("funding")),
                    open_interest=_f(raw.get("openInterest")),
                    prev_day_price=_f(raw.get("prevDayPx")),
                    day_volume=_f(raw.get("dayNtlVlm")),
                )
            )
        return out

    def asset_context(self, symbol: str) -> AssetContext:
        name = symbol.upper()
        for ctx in self.asset_contexts():
            if ctx.name == name:
                return ctx
        raise MarketDataError(f"no asset context for `{name}`")

    def order_book(self, symbol: str, depth: int = 5) -> dict[str, Any]:
        data = self.info({"type": "l2Book", "coin": symbol.upper()})
        if not isinstance(data, dict) or "levels" not in data:
            raise MarketDataError(f"unexpected l2Book response for {symbol}")
        data = dict(data)
        data["levels"] = [list(level)[:depth] for level in data["levels"]]
        return data

    def spread_bps(self, symbol: str) -> float:
        """Bid/ask spread in basis points, used as a liquidity guard."""
        book = self.order_book(symbol, depth=1)
        levels = book.get("levels") or []
        if len(levels) < 2 or not levels[0] or not levels[1]:
            return float("inf")
        best_bid = _f(levels[0][0].get("px"))
        best_ask = _f(levels[1][0].get("px"))
        if best_bid <= 0 or best_ask <= 0:
            return float("inf")
        mid = (best_bid + best_ask) / 2
        return (best_ask - best_bid) / mid * 10_000

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------
    def candles(
        self, symbol: str, interval: str = "1h", lookback: int = 300
    ) -> CandleSeries:
        """Fetch OHLCV candles ending at the current (incomplete) bar."""
        step = self._interval_ms(interval)
        # Add one extra bar so the currently-forming candle is included.
        end = int(time.time() * 1000)
        start = end - step * (lookback + 1)
        series = self.candles_range(symbol, interval, start, end)
        # Drop the still-forming candle for indicator stability.
        return _drop_incomplete(series) if len(series) > 2 else series

    def candles_range(
        self, symbol: str, interval: str, start_ms: int, end_ms: int
    ) -> CandleSeries:
        """Fetch OHLCV candles for an explicit window, inclusive of all bars.

        Unlike `candles()`, the final bar is kept: callers asking for an
        explicit window are working with closed bars, and the backtest depends
        on that distinction to avoid lookahead.
        """
        self._interval_ms(interval)

        data = self.info(
            {
                "type": "candleSnapshot",
                "req": {
                    "coin": symbol.upper(),
                    "interval": interval,
                    "startTime": int(start_ms),
                    "endTime": int(end_ms),
                },
            }
        )
        if not isinstance(data, list):
            raise MarketDataError(f"unexpected candleSnapshot response for {symbol}")

        opens: list[float] = []
        highs: list[float] = []
        lows: list[float] = []
        closes: list[float] = []
        volumes: list[float] = []
        times: list[int] = []

        for raw in data:
            if not isinstance(raw, dict):
                continue
            try:
                opens.append(float(raw["o"]))
                highs.append(float(raw["h"]))
                lows.append(float(raw["l"]))
                closes.append(float(raw["c"]))
                volumes.append(float(raw.get("v", 0) or 0))
                times.append(int(raw["t"]))
            except (KeyError, TypeError, ValueError):
                continue

        if not closes:
            raise MarketDataError(f"no candles returned for {symbol} {interval}")

        return CandleSeries(opens, highs, lows, closes, volumes, times)

    # ------------------------------------------------------------------
    # Funding
    # ------------------------------------------------------------------
    def funding_history(
        self, symbol: str, start_ms: int, end_ms: int
    ) -> list[tuple[int, float]]:
        """Hourly funding rates as (timestamp_ms, rate) pairs.

        Used by the backtest to charge the real carry cost of holding a
        position, rather than assuming funding is free.
        """
        data = self.info(
            {
                "type": "fundingHistory",
                "coin": symbol.upper(),
                "startTime": int(start_ms),
                "endTime": int(end_ms),
            }
        )
        if not isinstance(data, list):
            raise MarketDataError(f"unexpected fundingHistory response for {symbol}")

        out: list[tuple[int, float]] = []
        for raw in data:
            if not isinstance(raw, dict):
                continue
            try:
                out.append((int(raw["time"]), float(raw["fundingRate"])))
            except (KeyError, TypeError, ValueError):
                continue
        out.sort(key=lambda pair: pair[0])
        return out

    @staticmethod
    def _interval_ms(interval: str) -> int:
        if interval not in TIMEFRAME_MS:
            raise MarketDataError(
                f"unsupported interval `{interval}`; valid: {', '.join(TIMEFRAME_MS)}"
            )
        return TIMEFRAME_MS[interval]


def _drop_incomplete(series: CandleSeries) -> CandleSeries:
    """Return the series without its last (in-progress) candle."""
    return CandleSeries(
        series._opens[:-1],
        series._highs[:-1],
        series._lows[:-1],
        series._closes[:-1],
        series._volumes[:-1],
        series._times[:-1],
    )


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_candles(
    market: HyperliquidMarket, symbol: str, interval: str, lookback: int
) -> CandleSeries | None:
    """Fetch candles, returning None instead of raising on failure.

    Catches broadly on purpose: a data-layer surprise must degrade the signal,
    never take down the trading loop mid-cycle.
    """
    try:
        return market.candles(symbol, interval, lookback)
    except (MarketDataError, IndicatorError):
        return None
    except Exception:  # noqa: BLE001 - defensive: caller falls back to a default
        return None


def mainnet_data_market(
    market: HyperliquidMarket | None = None,
) -> HyperliquidMarket:
    """The client that signal-side code must read, given an execution client.

    **Signals always read mainnet.** `execution.testnet` chooses where *orders*
    go, never what the signal sees. Stated as a rule, this already existed for
    the whale factor - reading the leaderboard's mainnet addresses through the
    testnet API returned empty positions for all 25 sampled wallets, silently
    zeroing the heaviest weight in the blend.

    The other two factors did NOT follow it, and nothing caught that until it
    was measured. With `testnet: true` the `market` factor scored on testnet
    funding - 8.12 bp/h (711% APR) against 0.30 bp/h (26% APR) on mainnet -
    which moved its score from 37.0 to 51.9 and the blended confidence from
    58% to 85%. A testnet run was therefore not rehearsing the mainnet
    strategy; it was rehearsing a different one, on 60% of the factor weight.

    Returning `market` itself when it is already mainnet matters: mainnet runs
    and the backtest (whose market reports `testnet = False`) must not open a
    second client. Passing `None` yields a mainnet client for callers that have
    no execution client at all, such as the snapshot recorder.
    """
    if market is not None and not getattr(market, "testnet", False):
        return market
    return HyperliquidMarket(testnet=False)
