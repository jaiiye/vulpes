"""Historical market data for backtesting.

Fetches candles and funding rates from the public Hyperliquid API and caches
them on disk, because a multi-month backtest re-fetches the same windows
repeatedly and the API is rate limited.

What is available historically:
  * candles, any supported interval, months back     -> yes
  * hourly funding rates                             -> yes
  * open interest                                    -> no endpoint
  * whale / smart money positions                    -> no endpoint

The last two gaps shape what a backtest here can and cannot prove.
"""

from __future__ import annotations

import bisect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent.http_util import HttpError
from agent.indicators import CandleSeries, IndicatorError
from agent.market_data import TIMEFRAME_MS, HyperliquidMarket, MarketDataError

DEFAULT_CACHE_DIR = "backtest_cache"

# The API truncates large time-ranged responses, so ranges are requested in
# chunks. 30 days of hourly candles is 720 bars, comfortably under the limits
# observed for candleSnapshot; funding is chunked more tightly because it
# returned exactly the page size at 7 days.
CANDLE_CHUNK_DAYS = 30
FUNDING_CHUNK_DAYS = 10

RATE_LIMIT_SLEEP = 0.35  # seconds between chunk requests


class BacktestDataError(RuntimeError):
    """Raised when historical data cannot be assembled."""


@dataclass
class HistoricalData:
    """Everything the backtest needs for one symbol and window."""

    symbol: str
    start_ms: int
    end_ms: int
    candles: dict[str, CandleSeries] = field(default_factory=dict)
    funding: list[tuple[int, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Timestamp index derived from `funding`, built once by the loader. Without
    # it every `funding_at_ms` call rebuilt the list, making a per-bar lookup
    # O(n) despite the bisect below.
    _funding_times: list[int] = field(default_factory=list, repr=False)

    @property
    def available_start_ms(self) -> int:
        """Earliest timestamp actually present across the loaded candles."""
        starts = [s.times[0] for s in self.candles.values() if len(s)]
        return min(starts) if starts else self.start_ms

    def _funding_index(self) -> list[int]:
        """Cached funding timestamps, rebuilt if `funding` changed size.

        `funding` is only ever replaced wholesale (by the loader), never
        appended to, so a length check is sufficient to detect that the index
        is stale.
        """
        if len(self._funding_times) != len(self.funding):
            self._funding_times = [t for t, _ in self.funding]
        return self._funding_times

    def funding_at_ms(self, ts_ms: int) -> float:
        """The funding rate in effect at `ts_ms`.

        Funding is settled hourly, so the applicable rate is the most recent
        one at or before the timestamp. Returns 0.0 before the first record.
        """
        if not self.funding:
            return 0.0
        idx = bisect.bisect_right(self._funding_index(), ts_ms) - 1
        if idx < 0:
            return 0.0
        return self.funding[idx][1]

    def funding_between(self, start_ms: int, end_ms: int) -> list[tuple[int, float]]:
        """Funding records settled in (start_ms, end_ms]."""
        # Bounded by the index rather than scanning the whole history: this is
        # called once per closed trade, and the ranges are narrow.
        times = self._funding_index()
        lo = bisect.bisect_right(times, start_ms)
        hi = bisect.bisect_right(times, end_ms)
        return [(times[i], self.funding[i][1]) for i in range(lo, hi)]


class HistoricalLoader:
    """Fetches and caches historical candles and funding."""

    def __init__(
        self,
        market: HyperliquidMarket | None = None,
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
        testnet: bool = False,
    ) -> None:
        self.market = market or HyperliquidMarket(testnet=testnet)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _cache_path(self, kind: str, symbol: str, extra: str, start_ms: int, end_ms: int) -> Path | None:
        if self.cache_dir is None:
            return None
        # Bucket to calendar days so adjacent runs reuse the same file.
        start_day = start_ms // 86_400_000
        end_day = end_ms // 86_400_000
        name = f"{kind}_{symbol}_{extra}_{start_day}_{end_day}.json"
        return self.cache_dir / name

    def _read_cache(self, path: Path | None) -> dict | None:
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_cache(self, path: Path | None, payload: dict) -> None:
        if path is None:
            return
        try:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(path)
        except OSError:
            # A cache failure must never break a backtest.
            pass

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(
        self,
        symbol: str,
        intervals: tuple[str, ...] = ("1h", "4h"),
        days: int = 90,
        warmup_days: int = 30,
        progress=None,
    ) -> HistoricalData:
        """Load candles for each interval plus funding.

        `warmup_days` extends the fetch backwards so indicators have history at
        the first simulated bar; the extra bars are not traded.
        """
        symbol = symbol.upper()
        now_ms = int(time.time() * 1000)
        end_ms = now_ms
        start_ms = end_ms - days * 86_400_000
        fetch_start_ms = start_ms - warmup_days * 86_400_000

        data = HistoricalData(symbol=symbol, start_ms=start_ms, end_ms=end_ms)

        for interval in intervals:
            series = self._load_candles(
                symbol, interval, fetch_start_ms, end_ms, progress, data
            )
            data.candles[interval] = series
            if progress:
                progress(
                    f"  {interval}: {len(series)} bars "
                    f"({self._fmt_ms(series.times[0])} -> {self._fmt_ms(series.times[-1])})"
                )

        data.funding = self._load_funding(symbol, fetch_start_ms, end_ms, progress)
        # Build the lookup index once, so per-bar funding reads are O(log n).
        data._funding_times = [t for t, _ in data.funding]
        if progress:
            progress(f"  funding: {len(data.funding)} hourly records")

        # Clamp the traded window to what actually exists. candleSnapshot caps
        # at ~5000 bars, so a 1h request beyond ~208 days silently loses the
        # oldest data; trading a window with no bars would just produce nothing.
        entry = intervals[0]
        entry_series = data.candles.get(entry)
        if entry_series is not None and len(entry_series):
            earliest = entry_series.times[0]
            if earliest > data.start_ms:
                lost_days = (earliest - data.start_ms) / 86_400_000
                data.start_ms = earliest + warmup_days * 86_400_000
                # Never start beyond the data.
                if data.start_ms >= data.end_ms:
                    data.start_ms = earliest
                data.warnings.append(
                    f"{interval} history only reaches back to "
                    f"{self._fmt_ms(earliest)} ({lost_days:.0f} days of the "
                    "requested window are unavailable); the traded window was "
                    "shortened"
                )
                if progress:
                    progress(f"  {data.warnings[-1]}")

        return data

    def _load_candles(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        progress=None,
        data: HistoricalData | None = None,
    ) -> CandleSeries:
        path = self._cache_path("candles", symbol, interval, start_ms, end_ms)
        cached = self._read_cache(path)
        if cached:
            try:
                return CandleSeries.from_dict(cached)
            except IndicatorError:
                pass

        chunk_ms = CANDLE_CHUNK_DAYS * 86_400_000
        opens: list[float] = []
        highs: list[float] = []
        lows: list[float] = []
        closes: list[float] = []
        volumes: list[float] = []
        times: list[int] = []
        empty_chunks = 0

        for chunk_start, chunk_end in _chunks(start_ms, end_ms, chunk_ms):
            try:
                part = self.market.candles_range(
                    symbol, interval, chunk_start, chunk_end
                )
            except MarketDataError as exc:
                # "no candles returned" means the window predates available
                # history, which is expected once past the ~5000 bar cap.
                if "no candles" in str(exc).lower():
                    empty_chunks += 1
                    continue
                raise BacktestDataError(
                    f"failed to fetch {interval} candles for {symbol} "
                    f"({self._fmt_ms(chunk_start)} -> {self._fmt_ms(chunk_end)}): {exc}"
                ) from exc
            except HttpError as exc:
                raise BacktestDataError(
                    f"failed to fetch {interval} candles for {symbol} "
                    f"({self._fmt_ms(chunk_start)} -> {self._fmt_ms(chunk_end)}): {exc}"
                ) from exc
            opens.extend(part.opens)
            highs.extend(part.highs)
            lows.extend(part.lows)
            closes.extend(part.closes)
            volumes.extend(part.volumes)
            times.extend(part.times)
            time.sleep(RATE_LIMIT_SLEEP)

        if not closes:
            raise BacktestDataError(
                f"no {interval} candles returned for {symbol}: the requested "
                "window predates available history"
            )
        if empty_chunks and data is not None:
            data.warnings.append(
                f"{empty_chunks} of the requested {interval} windows predate "
                "available history and were skipped"
            )

        # Chunk boundaries can overlap by a bar; dedupe on timestamp.
        rows = sorted(
            zip(times, opens, highs, lows, closes, volumes), key=lambda r: r[0]
        )
        deduped: list[tuple] = []
        for row in rows:
            if deduped and row[0] == deduped[-1][0]:
                continue
            deduped.append(row)

        series = CandleSeries(
            [r[1] for r in deduped],
            [r[2] for r in deduped],
            [r[3] for r in deduped],
            [r[4] for r in deduped],
            [r[5] for r in deduped],
            [r[0] for r in deduped],
        )
        self._write_cache(path, series.to_dict())
        return series

    def _load_funding(
        self, symbol: str, start_ms: int, end_ms: int, progress=None
    ) -> list[tuple[int, float]]:
        path = self._cache_path("funding", symbol, "1h", start_ms, end_ms)
        cached = self._read_cache(path)
        if cached and isinstance(cached.get("funding"), list):
            return [(int(t), float(r)) for t, r in cached["funding"]]

        chunk_ms = FUNDING_CHUNK_DAYS * 86_400_000
        out: dict[int, float] = {}

        for chunk_start, chunk_end in _chunks(start_ms, end_ms, chunk_ms):
            try:
                part = self.market.funding_history(symbol, chunk_start, chunk_end)
            except (MarketDataError, HttpError) as exc:
                # Funding is a cost model, not a signal. Missing history should
                # degrade the result, not abort the run.
                if progress:
                    progress(f"  warning: funding chunk failed ({exc})")
                continue
            for ts, rate in part:
                out[ts] = rate
            time.sleep(RATE_LIMIT_SLEEP)

        funding = sorted(out.items())
        self._write_cache(path, {"funding": [[t, r] for t, r in funding]})
        return funding

    @staticmethod
    def _fmt_ms(ms: int) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime(ms / 1000))


def _chunks(start_ms: int, end_ms: int, chunk_ms: int):
    """Yield (start, end) windows covering [start_ms, end_ms]."""
    cursor = start_ms
    while cursor < end_ms:
        nxt = min(cursor + chunk_ms, end_ms)
        yield cursor, nxt
        cursor = nxt
