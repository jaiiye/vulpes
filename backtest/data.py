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
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent.http_util import HttpError
from agent.indicators import CandleSeries, IndicatorError
from agent.market_data import TIMEFRAME_MS, HyperliquidMarket, MarketDataError

DEFAULT_CACHE_DIR = "backtest_cache"

#: One-second bars from the Reservoir archive, partitioned by UTC day.
#: `candleSnapshot` on the public API caps at ~5000 bars, which is 208 days of
#: hourly candles and only 52 days of 15-minute ones - so the API cannot supply
#: a long enough window to validate a faster timeframe, and cannot supply a
#: second window at all once the first has used its history up. The archive
#: holds 414 days for every interval at one-second resolution.
DEFAULT_CANDLES_ARCHIVE = "data/canonical/candles"

# The API truncates large time-ranged responses, so ranges are requested in
# chunks. 30 days of hourly candles is 720 bars, comfortably under the limits
# observed for candleSnapshot; funding is chunked more tightly because it
# returned exactly the page size at 7 days.
CANDLE_CHUNK_DAYS = 30
FUNDING_CHUNK_DAYS = 10

RATE_LIMIT_SLEEP = 0.35  # seconds between chunk requests

#: Reconstructs interval bars from one-second bars, exactly.
#:
#: Deliberately a plain GROUP BY rather than a window function or a resample
#: helper: `arg_min(open)` is the first tick's open and `arg_max(close)` the
#: last tick's close, which is how the venue aggregates too. Verified against 8
#: hourly bars the API itself returned, spread across seven months - OHLC and
#: volume matched to the last digit, so this is a reconstruction, not an
#: approximation.
#:
#: `to_timestamp` (not `epoch_ms`) on the bounds because the column is
#: TIMESTAMPTZ: comparing it to a bare TIMESTAMP casts through the session time
#: zone and silently shifts the window. That mistake cost an hour of debugging
#: once already, and it produced plausible-looking bars rather than an error.
ARCHIVE_RESAMPLE_SQL = """
SELECT epoch_ms(time_bucket(INTERVAL '{bucket}', timestamp)) AS t,
       arg_min(open, timestamp)::DOUBLE AS o,
       max(high)::DOUBLE AS h,
       min(low)::DOUBLE AS l,
       arg_max(close, timestamp)::DOUBLE AS c,
       SUM(volume)::DOUBLE AS v,
       COUNT(*) AS ticks
FROM read_parquet({files})
WHERE coin = '{coin}'
  AND timestamp >= to_timestamp({start_s})
  AND timestamp <  to_timestamp({end_s})
GROUP BY 1
ORDER BY 1
"""


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
        candles_archive: str | Path | None = None,
    ) -> None:
        self.market = market or HyperliquidMarket(testnet=testnet)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        # When set, candles come from the local archive instead of the API.
        # Funding still comes from the API: the archive holds positions, fills
        # and candles, but no funding history. That is a real gap - a long
        # window pays funding for its whole length - and it is reported as a
        # warning by `load` rather than left for the reader to notice.
        self.candles_archive = Path(candles_archive) if candles_archive else None
        if self.candles_archive and not self.candles_archive.is_dir():
            raise BacktestDataError(
                f"no candle archive at {self.candles_archive}. Run: "
                "python sync_reservoir.py --datasets candles"
            )

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
        end_ms: int | None = None,
    ) -> HistoricalData:
        """Load candles for each interval plus funding.

        `warmup_days` extends the fetch backwards so indicators have history at
        the first simulated bar; the extra bars are not traded.

        `end_ms` pins the window. Left as None the window ends at the current
        time, which makes every run on a different day describe a different
        period - two runs of the "same" backtest are then not comparable, and a
        change in the numbers cannot be told apart from the clock having moved.
        Measured: a 90-day run at 06:14 traded from 2026-06-20 and the same
        command at 10:15 traded from 2026-06-21, for 14 trades versus 15. Pass
        an explicit timestamp for anything being compared or reported.
        """
        symbol = symbol.upper()
        end_ms = int(time.time() * 1000) if end_ms is None else int(end_ms)
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

        if self.candles_archive is not None:
            # Candles come from the archive; funding cannot - the archive holds
            # positions, fills and candles, but no funding history. Worth
            # stating only when the API actually falls short: the API reaches
            # back roughly as far as the archive, so most runs have full
            # coverage and a warning that always fires is a warning nobody reads.
            span_days = (end_ms - fetch_start_ms) / 86_400_000
            covered_days = len(data.funding) / 24
            if covered_days < span_days - 2:
                note = (
                    f"candles came from the archive, funding did not: the API "
                    f"supplied {len(data.funding)} hourly records, about "
                    f"{covered_days:.0f} of the {span_days:.0f} days fetched. "
                    "Days the API does not reach are charged zero funding, "
                    "which flatters a long position held through them."
                )
                data.warnings.append(note)
                if progress:
                    progress(f"  {note}")

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
        if self.candles_archive is not None:
            return self._load_candles_archive(
                symbol, interval, start_ms, end_ms, progress, data
            )

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

    def _load_candles_archive(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
        progress=None,
        data: HistoricalData | None = None,
    ) -> CandleSeries:
        """Rebuild interval bars from the one-second archive.

        Same contract as the API path - a `CandleSeries` of whole interval bars
        - so nothing downstream can tell which source it came from. That is the
        point: the window is the only thing that changes.
        """
        if interval not in TIMEFRAME_MS:
            raise BacktestDataError(f"unknown interval {interval!r}")

        # `arc` in the cache key so an archive-built series is never mistaken
        # for an API-fetched one. They should be identical, but if they ever
        # diverge, silently reusing the wrong file would hide it.
        path = self._cache_path("arc_candles", symbol, interval, start_ms, end_ms)
        cached = self._read_cache(path)
        if cached:
            try:
                return CandleSeries.from_dict(cached)
            except IndicatorError:
                pass

        files = self._archive_files(start_ms, end_ms)
        if not files:
            raise BacktestDataError(
                f"the candle archive at {self.candles_archive} holds no day "
                f"between {self._fmt_ms(start_ms)} and {self._fmt_ms(end_ms)}"
            )

        minutes = TIMEFRAME_MS[interval] // 60_000
        sql = ARCHIVE_RESAMPLE_SQL.format(
            bucket=f"{minutes} minutes",
            files="[" + ", ".join(f"'{p}'" for p in files) + "]",
            coin=symbol.upper(),
            start_s=start_ms // 1000,
            end_s=(end_ms // 1000) + 1,
        )
        rows = self._run_archive_sql(sql)

        if not rows:
            raise BacktestDataError(
                f"the candle archive holds no {interval} {symbol.upper()} bars "
                f"between {self._fmt_ms(start_ms)} and {self._fmt_ms(end_ms)}. "
                "The archive carries 260 coins, so this is more likely a symbol "
                "the venue does not list than an empty range."
            )

        series = CandleSeries(
            [float(r["o"]) for r in rows],
            [float(r["h"]) for r in rows],
            [float(r["l"]) for r in rows],
            [float(r["c"]) for r in rows],
            [float(r["v"]) for r in rows],
            [int(r["t"]) for r in rows],
        )
        self._write_cache(path, series.to_dict())

        if data is not None:
            # Compare against the requested span, not the span the files cover:
            # a partition can exist for a day whose seconds are mostly missing,
            # and that day is genuinely thin.
            expect = (end_ms - start_ms) // TIMEFRAME_MS[interval]
            if len(series) < expect * 0.98:
                data.warnings.append(
                    f"the archive produced {len(series)} {interval} bars for a "
                    f"window that spans {expect}; some seconds are missing from "
                    "the source, so bars near a gap may be thin"
                )
                if progress:
                    progress(f"  {data.warnings[-1]}")

        return series

    def _archive_files(self, start_ms: int, end_ms: int) -> list[str]:
        """Archive partitions covering [start_ms, end_ms], in order.

        Named explicitly rather than globbing the whole directory: DuckDB has no
        range syntax for globs, and handing it 414 files to open when 170 are
        needed makes the footer reads dominate the query. The day name is a UTC
        day, which is how the sync partitions it.
        """
        assert self.candles_archive is not None
        out: list[str] = []
        day = start_ms // 86_400_000
        last = end_ms // 86_400_000
        while day <= last:
            name = time.strftime("%Y-%m-%d", time.gmtime(day * 86_400))
            candidate = self.candles_archive / f"date={name}.parquet"
            if candidate.is_file():
                out.append(str(candidate))
            day += 1
        return out

    @staticmethod
    def _run_archive_sql(sql: str, timeout: int = 900) -> list[dict]:
        """Run a query over the archive and parse its JSON output.

        The statement goes in on **stdin**, not as a `-c` argument: a single
        argument is capped at 128 KB by the kernel (`MAX_ARG_STRLEN`) and the
        file list for a long window approaches that. Same reason as in
        `position_history._run_sql`, which is where this trips first.
        """
        try:
            proc = subprocess.run(
                ["duckdb", "-json"],
                input=sql,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise BacktestDataError(
                "duckdb was not found on PATH; it is required to read the "
                "candle archive"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise BacktestDataError(
                f"reading the candle archive timed out after {timeout}s"
            ) from exc

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip().splitlines()
            raise BacktestDataError(
                f"reading the candle archive failed: "
                f"{detail[-1] if detail else 'no output'}"
            )
        try:
            payload = json.loads(proc.stdout or "[]")
        except json.JSONDecodeError as exc:
            raise BacktestDataError(
                f"unparseable output from the candle archive: {proc.stdout[:200]}"
            ) from exc
        return payload if isinstance(payload, list) else []

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
