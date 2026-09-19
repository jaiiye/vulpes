"""Reconstructs historical whale positions from funding records.

The public API has no "positions at time T" endpoint, but `userFunding` returns
one record per funding settlement carrying `szi`, the position size at that
moment. Funding settles hourly, so paging that endpoint yields an hourly
position time series per wallet.

That turns the smart money factor, previously written off as un-backtestable,
into something that can be replayed across the full available history.

Two properties of the data shape the reconstruction:

  * Records exist only while a position is open, so a gap means either flat or
    genuinely no data. `position_at` therefore requires a record within
    `max_age_ms`, which is the same staleness discipline the live factor uses.
  * Responses are capped at 500 records and return the OLDEST page first, so
    collecting history requires advancing a cursor. Fetching without paging
    silently returns months-old data.
"""

from __future__ import annotations

import bisect
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from agent.http_util import HttpError, request_json

MAINNET_INFO_URL = "https://api.hyperliquid.xyz/info"

PAGE_SIZE = 500
MAX_PAGES = 40          # ~20,000 records, well beyond observed activity
REQUEST_TIMEOUT = 60
RATE_LIMIT_SLEEP = 0.2
DEFAULT_WORKERS = 8


@dataclass
class WalletPositionSeries:
    """A wallet's position in one coin over time.

    `points` holds (timestamp, size) and is the only series the funding-based
    collector produces. The Reservoir snapshots also carry an entry price, so
    `entries` is a parallel series populated only from that richer source.

    Notional is deliberately not carried: the snapshot's own notional is marked
    at the snapshot's price and is stale by up to a day. The factor marks at the
    bar's price instead, so only the entry price - which is a fixed historical
    fact - needs to survive from the archive.
    """

    wallet: str
    coin: str
    points: list[tuple[int, float]] = field(default_factory=list)
    funding_rates: list[tuple[int, float]] = field(default_factory=list)
    entries: list[tuple[int, float]] = field(default_factory=list)
    persistences: list[tuple[int, int]] = field(default_factory=list)
    error: str | None = None
    #: Timestamp index per series, keyed by `id(list)`. One entry per series
    #: rather than one shared slot: the snapshot builder calls `position_at`
    #: and `entry_at` for the same wallet in the same bar, and a single slot
    #: would have each call evict the other's index.
    _times_cache: dict[int, list[int]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __len__(self) -> int:
        return len(self.points)

    @property
    def span_ms(self) -> tuple[int, int] | None:
        if not self.points:
            return None
        return self.points[0][0], self.points[-1][0]

    @property
    def has_pnl(self) -> bool:
        """True when an entry price is available, so PnL can be computed."""
        return bool(self.entries)

    def _times_for(self, series: list[tuple[int, float]]) -> list[int]:
        """Cached timestamp list for a series, rebuilt only if it changed size.

        Without this the index is rebuilt on every lookup, which is O(n) per
        call and runs once per wallet per bar: 150 wallets against 4000 bars
        turns a 364-point series into hundreds of millions of operations.
        Series are only ever assigned wholesale, so a length check suffices.
        """
        key = id(series)
        times = self._times_cache.get(key)
        if times is None or len(times) != len(series):
            times = [p[0] for p in series]
            self._times_cache[key] = times
        return times

    def _at(self, series: list[tuple[int, float]], ts_ms: int,
            max_age_ms: int) -> float | None:
        """Most recent value at or before `ts_ms`, if fresh enough.

        Staleness is the central failure mode this guards against: the
        22-agent experiment's Scorpion v1 lost by treating months-old
        positions as fresh signals.
        """
        if not series:
            return None
        idx = bisect.bisect_right(self._times_for(series), ts_ms) - 1
        if idx < 0:
            return None
        record_ts, value = series[idx]
        if ts_ms - record_ts > max_age_ms:
            return None
        return value

    def position_at(self, ts_ms: int, max_age_ms: int) -> float | None:
        """The position size at `ts_ms`, or None if the data is too stale."""
        return self._at(self.points, ts_ms, max_age_ms)

    def entry_at(self, ts_ms: int, max_age_ms: int) -> float | None:
        """The average entry price at `ts_ms`, or None when unavailable."""
        return self._at(self.entries, ts_ms, max_age_ms)

    def persistence_at(self, ts_ms: int, max_age_ms: int) -> int | None:
        """How many leaderboard windows this wallet ranked in on that day.

        Per-day rather than fixed, because the live agent reads the ranking as
        it stands at the moment of the decision. A wallet can rank in all three
        windows today and none tomorrow, and a single value for the whole run
        would report whichever of those a caller happened to pick.
        """
        value = self._at(self.persistences, ts_ms, max_age_ms)
        return None if value is None else int(value)

    def to_payload(self) -> dict:
        return {
            "wallet": self.wallet,
            "coin": self.coin,
            "points": [[t, s] for t, s in self.points],
            "funding_rates": [[t, r] for t, r in self.funding_rates],
        }

    @classmethod
    def from_payload(cls, data: dict) -> "WalletPositionSeries":
        try:
            points = [(int(t), float(s)) for t, s in data.get("points") or []]
            rates = [(int(t), float(r)) for t, r in data.get("funding_rates") or []]
        except (TypeError, ValueError):
            points, rates = [], []
        return cls(
            wallet=str(data.get("wallet", "")),
            coin=str(data.get("coin", "")),
            points=points,
            funding_rates=rates,
        )


class WhaleHistoryCollector:
    """Fetches, caches and serves per-wallet position history."""

    def __init__(
        self,
        cache_dir: str | Path | None = "backtest_cache/whales",
        workers: int = DEFAULT_WORKERS,
        info_url: str = MAINNET_INFO_URL,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.workers = max(1, workers)
        self.info_url = info_url
        self.errors: list[str] = []

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    # One paging pass yields every coin a wallet ever funded, so the cache is
    # keyed by wallet alone. Keying by (wallet, coin) would triple the request
    # count when testing several symbols, for no additional information.
    def _all_cache_path(self, wallet: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{wallet.lower()}_all.json"

    def _legacy_cache_path(self, wallet: str, coin: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{wallet.lower()}_{coin.upper()}.json"

    def _load(self, wallet: str, coin: str) -> WalletPositionSeries | None:
        """Read one coin's series, preferring the combined cache."""
        combined = self._all_cache_path(wallet)
        if combined is not None and combined.exists():
            try:
                payload = json.loads(combined.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                entry = payload.get("coins", {}).get(coin.upper())
                if isinstance(entry, dict):
                    series = WalletPositionSeries.from_payload(
                        {"wallet": wallet, "coin": coin, **entry}
                    )
                    return series if series.points else None
                return None  # combined cache is authoritative for this wallet

        # Legacy per-coin cache from before the combined format existed.
        path = self._legacy_cache_path(wallet, coin)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        series = WalletPositionSeries.from_payload(payload)
        return series if series.points else None

    def _save_all(
        self, wallet: str, by_coin: dict[str, WalletPositionSeries]
    ) -> None:
        path = self._all_cache_path(wallet)
        if path is None:
            return
        try:
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "wallet": wallet.lower(),
                        "coins": {
                            coin: {
                                "points": [[t, s] for t, s in s.points],
                                "funding_rates": [[t, r] for t, r in s.funding_rates],
                            }
                            for coin, s in by_coin.items()
                        },
                    }
                ),
                encoding="utf-8",
            )
            tmp.replace(path)
        except OSError:
            pass

    def has_cache(self, wallet: str) -> bool:
        path = self._all_cache_path(wallet)
        return path is not None and path.exists()

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------
    def fetch_one(
        self, wallet: str, coin: str, start_ms: int, end_ms: int
    ) -> WalletPositionSeries:
        """Single-coin convenience wrapper over `fetch_all`."""
        by_coin = self.fetch_all(wallet, start_ms, end_ms)
        series = by_coin.get(coin.upper())
        if series is not None:
            return series
        return WalletPositionSeries(wallet=wallet.lower(), coin=coin.upper())

    def fetch_all(
        self, wallet: str, start_ms: int, end_ms: int
    ) -> dict[str, WalletPositionSeries]:
        """Page `userFunding` once and split the records by coin.

        Fetching per coin would repeat the same paginated walk for every
        symbol; the endpoint returns every coin's funding in one stream, so one
        pass serves all of them.
        """
        per_coin: dict[str, dict[int, float]] = {}
        per_coin_rates: dict[str, dict[int, float]] = {}
        error: str | None = None
        cursor = int(start_ms)

        for _ in range(MAX_PAGES):
            try:
                page = request_json(
                    self.info_url,
                    payload={
                        "type": "userFunding",
                        "user": wallet,
                        "startTime": cursor,
                        "endTime": int(end_ms),
                    },
                    timeout=REQUEST_TIMEOUT,
                    retries=2,
                )
            except HttpError as exc:
                error = str(exc)
                break

            if not isinstance(page, list) or not page:
                break

            newest = cursor
            for record in page:
                if not isinstance(record, dict):
                    continue
                ts = record.get("time")
                delta = record.get("delta")
                if ts is None or not isinstance(delta, dict):
                    continue
                ts = int(ts)
                newest = max(newest, ts)
                coin_name = str(delta.get("coin", "")).upper()
                if not coin_name:
                    continue
                try:
                    size = float(delta["szi"])
                except (KeyError, TypeError, ValueError):
                    continue
                per_coin.setdefault(coin_name, {})[ts] = size
                try:
                    rate = float(delta.get("fundingRate") or 0.0)
                except (TypeError, ValueError):
                    rate = 0.0
                per_coin_rates.setdefault(coin_name, {})[ts] = rate

            # The API returns the oldest page first, so advance past it.
            if newest <= cursor:
                break
            cursor = newest + 1
            if cursor >= end_ms:
                break
            time.sleep(RATE_LIMIT_SLEEP)

        out: dict[str, WalletPositionSeries] = {}
        for coin_name, points in per_coin.items():
            series = WalletPositionSeries(
                wallet=wallet.lower(),
                coin=coin_name,
                points=sorted(points.items()),
                funding_rates=sorted(per_coin_rates.get(coin_name, {}).items()),
                error=error,
            )
            out[coin_name] = series

        if out:
            self._save_all(wallet, out)
        return out

    def collect(
        self,
        wallets: list[str],
        coin: str,
        start_ms: int,
        end_ms: int,
        progress=None,
    ) -> dict[str, WalletPositionSeries]:
        """Fetch history for many wallets in parallel, using the cache."""
        out: dict[str, WalletPositionSeries] = {}
        self.errors = []
        to_fetch: list[str] = []

        for wallet in wallets:
            # A combined cache file means this wallet was already fetched. If
            # the coin is absent from it, the wallet genuinely has no recorded
            # positions in it, so refetching would only waste requests.
            if self.has_cache(wallet):
                cached = self._load(wallet, coin)
                if cached is not None:
                    out[wallet.lower()] = cached
                continue
            cached = self._load(wallet, coin)  # legacy per-coin cache
            if cached is not None:
                out[wallet.lower()] = cached
            else:
                to_fetch.append(wallet)

        if progress and out:
            progress(f"  {len(out)} wallets served from cache")
        if not to_fetch:
            return out

        if progress:
            progress(f"  fetching {len(to_fetch)} wallets ({self.workers} workers)...")

        done = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(self.fetch_all, w, start_ms, end_ms): w
                for w in to_fetch
            }
            for future in as_completed(futures):
                wallet = futures[future]
                done += 1
                try:
                    by_coin = future.result()
                except Exception as exc:  # noqa: BLE001 - isolate per-wallet failure
                    self.errors.append(f"{wallet[:12]}…: {exc}")
                    continue
                series = by_coin.get(coin.upper())
                if series is None:
                    continue
                if series.error:
                    self.errors.append(f"{wallet[:12]}…: {series.error}")
                if series.points:
                    out[wallet.lower()] = series
                if progress and done % 25 == 0:
                    progress(f"  {done}/{len(to_fetch)} fetched, {len(out)} usable")

        if progress:
            coverage = len(out) / len(wallets) * 100 if wallets else 0
            progress(
                f"  usable wallets: {len(out)}/{len(wallets)} ({coverage:.0f}%)"
            )
        return out


def series_stats(series_map: dict[str, WalletPositionSeries]) -> str:
    """Human-readable coverage summary."""
    if not series_map:
        return "  no wallet history available"

    lengths = [len(s) for s in series_map.values()]
    spans = [s.span_ms for s in series_map.values() if s.span_ms]
    lines = [
        f"  wallets with history: {len(series_map)}",
        f"  position points: {sum(lengths):,} "
        f"(median {sorted(lengths)[len(lengths) // 2]} per wallet)",
    ]
    if spans:
        lo = min(s[0] for s in spans)
        hi = max(s[1] for s in spans)
        lines.append(
            f"  coverage: {time.strftime('%Y-%m-%d', time.gmtime(lo / 1000))} -> "
            f"{time.strftime('%Y-%m-%d', time.gmtime(hi / 1000))}"
        )
    return "\n".join(lines)
