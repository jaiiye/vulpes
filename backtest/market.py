"""A market view that serves only past data.

`TechnicalFactor` and `MarketFactor` read the market through
`HyperliquidMarket`'s read-only interface. This class implements the same
interface over a historical dataset and a simulated clock, so the live factor
code runs unmodified in a backtest instead of being reimplemented.

The entire anti-lookahead property rests on one rule:

    `candles()` returns only bars that have fully CLOSED before the current
    simulated instant.

A bar timestamped T covers [T, T+interval). Standing at the open of the bar
starting at T (the moment a live agent would decide, since it only ever sees
closed bars), every bar with timestamp < T has closed. `bisect_left` gives
exactly that boundary, so no future bar can leak into an indicator.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from agent.indicators import CandleSeries
from agent.market_data import AssetContext, MarketDataError

from .data import HistoricalData

# Historical order books are not available, so the spread is an assumption.
# 2bp is typical for BTC on Hyperliquid; it only drives the advisory spread
# warning, never a PnL calculation, so a constant is honest here.
ASSUMED_SPREAD_BPS = 2.0

DAY_MS = 86_400_000


@dataclass
class Bar:
    """One OHLCV bar, used by the engine to fill and exit orders."""

    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class BacktestMarket:
    """Historical, clock-driven stand-in for `HyperliquidMarket`."""

    def __init__(
        self,
        datasets: dict[str, HistoricalData],
        entry_interval: str = "1h",
        spread_bps: float = ASSUMED_SPREAD_BPS,
    ) -> None:
        if not datasets:
            raise MarketDataError("BacktestMarket requires at least one dataset")
        self.datasets = {k.upper(): v for k, v in datasets.items()}
        self.entry_interval = entry_interval
        self.spread_bps_value = spread_bps
        self.testnet = False  # never used to select real endpoints here
        self._now_ms: int = 0

    # ------------------------------------------------------------------
    # Clock
    # ------------------------------------------------------------------
    def set_now(self, ts_ms: int) -> None:
        """Advance the simulated clock to a bar's opening timestamp."""
        self._now_ms = int(ts_ms)

    @property
    def now_ms(self) -> int:
        return self._now_ms

    # ------------------------------------------------------------------
    # Candle access (the anti-lookahead boundary)
    # ------------------------------------------------------------------
    def _dataset(self, symbol: str) -> HistoricalData:
        key = symbol.upper()
        if key not in self.datasets:
            raise MarketDataError(f"no backtest data loaded for {key}")
        return self.datasets[key]

    def _closed_count(self, symbol: str, interval: str) -> int:
        """How many bars of `interval` had closed by the simulated instant."""
        series = self._dataset(symbol).candles.get(interval)
        if series is None:
            raise MarketDataError(f"no {interval} series for {symbol}")
        return bisect.bisect_left(series.times, self._now_ms)

    def candles(
        self, symbol: str, interval: str = "1h", lookback: int = 300
    ) -> CandleSeries:
        """The last `lookback` CLOSED bars. Never includes the current bar."""
        series = self._dataset(symbol).candles.get(interval)
        if series is None:
            raise MarketDataError(f"no {interval} series for {symbol}")

        closed = self._closed_count(symbol, interval)
        start = max(0, closed - lookback)
        sliced = series.slice(start, closed)
        if len(sliced) == 0:
            raise MarketDataError(
                f"no closed {interval} bars for {symbol} at {self._now_ms}"
            )
        return sliced

    def bar_at(self, symbol: str, interval: str, ts_ms: int) -> Bar | None:
        """The bar starting exactly at `ts_ms`, for order filling."""
        series = self._dataset(symbol).candles.get(interval)
        if series is None:
            return None
        times = series.times
        idx = bisect.bisect_left(times, ts_ms)
        if idx >= len(times) or times[idx] != ts_ms:
            return None
        return Bar(
            time=times[idx],
            open=series.opens[idx],
            high=series.highs[idx],
            low=series.lows[idx],
            close=series.closes[idx],
            volume=series.volumes[idx],
        )

    def bar_times(self, symbol: str, interval: str) -> list[int]:
        series = self._dataset(symbol).candles.get(interval)
        return list(series.times) if series else []

    # ------------------------------------------------------------------
    # Price and context
    # ------------------------------------------------------------------
    def mid_price(self, symbol: str) -> float:
        """The opening price of the current bar.

        Standing at the open of bar T, that is the price actually available,
        so it is what an order would fill against.
        """
        series = self._dataset(symbol).candles.get(self.entry_interval)
        if series is None:
            raise MarketDataError(f"no {self.entry_interval} series for {symbol}")
        times = series.times
        idx = bisect.bisect_left(times, self._now_ms)
        if idx < len(times) and times[idx] == self._now_ms:
            return series.opens[idx]
        if idx == 0:
            raise MarketDataError(f"no price available for {symbol} at {self._now_ms}")
        return series.closes[idx - 1]

    def spread_bps(self, symbol: str) -> float:
        return self.spread_bps_value

    def asset_context(self, symbol: str) -> AssetContext:
        """Funding and price context as of the simulated instant.

        Open interest is reported as 0 because the API exposes no historical
        OI series. The market factor does not use OI for its confidence, so
        this degrades the reason string rather than the signal.
        """
        dataset = self._dataset(symbol)
        series = dataset.candles.get(self.entry_interval)
        if series is None:
            raise MarketDataError(f"no {self.entry_interval} series for {symbol}")

        closed = self._closed_count(symbol, self.entry_interval)
        if closed == 0:
            raise MarketDataError(f"no closed bars for {symbol} at {self._now_ms}")

        mark = series.closes[closed - 1]
        oracle = mark

        # 24h-ago close, on the entry timeframe, for a real day-change figure.
        bars_per_day = max(1, int(DAY_MS / 1000 / _interval_seconds(self.entry_interval)))
        prev_idx = max(0, closed - 1 - bars_per_day)
        prev_day = series.closes[prev_idx]

        funding = dataset.funding_at_ms(self._now_ms)

        return AssetContext(
            index=0,
            name=symbol.upper(),
            mark_price=mark,
            oracle_price=oracle,
            funding=funding,
            open_interest=0.0,
            prev_day_price=prev_day,
            day_volume=0.0,
        )

    def asset_contexts(self) -> list[AssetContext]:
        out: list[AssetContext] = []
        for symbol in self.datasets:
            try:
                out.append(self.asset_context(symbol))
            except MarketDataError:
                continue
        return out

    def order_book(self, symbol: str, depth: int = 5) -> dict:
        """No historical book exists; report a spread around the current price."""
        price = self.mid_price(symbol)
        half = price * self.spread_bps_value / 2 / 10_000
        return {
            "coin": symbol.upper(),
            "levels": [
                [{"px": str(price - half), "sz": "0"}],
                [{"px": str(price + half), "sz": "0"}],
            ],
        }


def _interval_seconds(interval: str) -> int:
    from agent.market_data import TIMEFRAME_MS

    return TIMEFRAME_MS.get(interval, 3_600_000) // 1000
