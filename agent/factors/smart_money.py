"""Smart money factor.

This is the load-bearing factor. In the 22-agent experiment every profitable
agent derived its edge from real-time smart money data, while the pure-TA and
mean-reversion agents lost 18-33%.

Two backends are supported:

1. `leaderboard` (default, zero credentials)
   Samples the Hyperliquid leaderboard and reads the live clearinghouse state
   of the top-ranked accounts, aggregating their positions in the target
   symbol. This is a genuine smart-money read, but the wallet set is
   leaderboard-derived rather than curated, so confidence is capped.

2. `hyperfeed` (optional, requires SENPI_API_KEY)
   Uses a curated whale feed. Higher quality wallet set. Enable with
   `use_hyperfeed: true` in the config once the key is exported.

Staleness is treated as a first-class hazard: Scorpion v1 lost by using
months-old position snapshots as fresh signals. Snapshots older than
`MAX_SNAPSHOT_AGE_SECONDS` collapse confidence to zero.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from ..config import BotConfig
from ..http_util import HttpError, request_json
from ..market_data import HyperliquidMarket, MarketDataError
from .base import FactorScore

MAX_SNAPSHOT_AGE_SECONDS = 900  # 15 minutes

# Reuse a snapshot for this long before refetching positions.
SNAPSHOT_REUSE_SECONDS = 60

# `quality_conf` is `0.5 + 0.5 * winner_ratio`, so it spans [0.5, 1.0].
# When the winner ratio cannot be known (the backtest), this midpoint is the
# neutral estimate - see the confidence block in `score_snapshot`.
NEUTRAL_QUALITY_CONF = 0.75

# ---------------------------------------------------------------------------
# Confidence model, used by `score_snapshot`
# ---------------------------------------------------------------------------
# Holder count at which the sample-size term saturates. Raised from 8 to 12
# when multi-window selection enlarged the candidate pool: 8 holders is no
# longer a strong sample at that point.
SAMPLE_SIZE_SATURATION = 12.0

# Component weights. They sum to 1.0.
WEIGHT_SAMPLE = 0.30
WEIGHT_CONSENSUS = 0.25
WEIGHT_QUALITY = 0.15
WEIGHT_CURATION = 0.15
WEIGHT_CONCENTRATION = 0.15

# Leaderboard-derived wallet sets are uncurated, so confidence is capped. The
# cap scales with curation quality rather than being pinned at a flat value.
LEADERBOARD_CONF_CAP_BASE = 0.75
LEADERBOARD_CONF_CAP_CURATION = 0.10

# Partial wallet-read failures degrade, but do not void, the signal.
WALLET_ERROR_CONFIDENCE_PENALTY = 0.9

# Wallet reads are independent HTTP calls, so they are worth parallelising:
# 25 sequential reads measured 58s and dominated the whole cycle.
MAX_WALLET_WORKERS = 12

# The leaderboard response is ~37 MB with no server-side limit, so the derived
# wallet list (a few hundred bytes) is cached on disk between processes.
#
# The TTL is deliberately much longer than the polling interval. The wallet
# *list* is a slow-moving set (top accounts by PnL across day/week/month); what
# actually needs to be fresh is their *positions*, and those are refetched every
# cycle regardless. A short TTL would force a repeat 37 MB download on almost
# every scheduled run: measured, a 15-minute cadence against a 900s TTL pulls
# ~2.7 GB/day. At 6h it drops to ~150 MB/day with no loss of signal freshness.
WALLET_CACHE_TTL_SECONDS = 6 * 3600
DEFAULT_WALLET_CACHE_PATH = "logs/smart_money_wallets.json"

# ---------------------------------------------------------------------------
# Wallet selection
# ---------------------------------------------------------------------------
# Ranking accounts by a single window is a luck-contaminated proxy. Measured
# on live data: the top-25 by "week" and the top-25 by "month" shared only 11%
# of their members (Jaccard 0.111), and the "day" set collapsed to 2 BTC
# holders with a single wallet at 79.9% of gross notional.
#
# Sampling several windows and unioning them fixes the coverage problem.
# Measured on BTC (holders / largest-position share of gross):
#
#   single window, top 25      10 holders   43.6%
#   union, hard-filtered >=2    3 holders   45.6%   <- WORSE
#   union, unfiltered         18 holders   27.8%   <- used
#
# So persistence is applied as a *weight*, not a hard filter: filtering out
# single-window wallets discards the directional traders who actually hold the
# symbol, leaving only large diversified accounts. Setting min_persistence > 1
# re-enables hard filtering if you want a smaller, stricter set.
DEFAULT_WINDOWS = ("day", "week", "month")
DEFAULT_TOP_PER_WINDOW = 60      # 3 x 60 unions to ~118 candidates
DEFAULT_MIN_PERSISTENCE = 1      # 1 = no hard filter, persistence only weights
DEFAULT_MAX_WALLETS = 150
MIN_ACCOUNT_VALUE = 10_000.0


@dataclass
class WalletPosition:
    """A single wallet's net position in one symbol."""

    wallet: str
    size: float = 0.0        # signed: positive = long
    notional: float = 0.0    # signed USD notional
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    leverage: float = 0.0
    account_value: float = 0.0
    # How many leaderboard windows this wallet ranked in. Used to weight its
    # opinion: a wallet that ranks across day, week and month is more likely to
    # have a repeatable edge than one that spiked in a single window.
    persistence: int = 1
    whitelisted: bool = False

    @property
    def side(self) -> str:
        if self.size > 0:
            return "long"
        if self.size < 0:
            return "short"
        return "flat"

    @property
    def weight(self) -> float:
        """Non-negative weight applied to this wallet's notional.

        Whitelisted wallets are treated as fully trusted. Otherwise the weight
        is the persistence count, floored at 1 so every wallet still counts.
        """
        if self.whitelisted:
            return float(max(self.persistence, 2))
        return float(max(self.persistence, 1))


def signed_notional(notional: float, size: float) -> float:
    """Give `notional` the sign of `size`, keeping the pair consistent.

    Both parsers used to repeat `abs(notional) if size > 0 else -abs(notional)`
    inline, so the convention had no single place to test - and none did, which
    is how a dropped sign went unnoticed.

    The convention matters because `size` carries the direction and `notional`
    must agree with it. `short_notional` sums `-p.notional` over positions with
    `size < 0`, which is a magnitude only while that notional is itself
    negative. Drop the sign and a short contributes a negative amount:
    `long_ratio_pct` then exceeds 100% and the factor reads a heavily shorted
    market as a long consensus.
    """
    magnitude = abs(float(notional))
    return magnitude if size > 0 else -magnitude


@dataclass
class SmartWallet:
    """A wallet selected as "smart money", with its selection evidence.

    `persistence` is the number of ranking windows the wallet appeared in.
    A wallet that ranks well across day, week and month is far more likely to
    have a repeatable edge than one that spiked in a single window.
    """

    address: str
    windows: list[str] = field(default_factory=list)
    persistence: int = 0
    account_value: float = 0.0
    window_pnl: dict[str, float] = field(default_factory=dict)
    whitelisted: bool = False

    def to_dict(self) -> dict:
        return {
            "address": self.address,
            "windows": self.windows,
            "persistence": self.persistence,
            "account_value": self.account_value,
            "window_pnl": self.window_pnl,
            "whitelisted": self.whitelisted,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SmartWallet":
        return cls(
            address=str(data.get("address", "")),
            windows=list(data.get("windows") or []),
            persistence=int(data.get("persistence", 0) or 0),
            account_value=float(data.get("account_value", 0) or 0),
            window_pnl=dict(data.get("window_pnl") or {}),
            whitelisted=bool(data.get("whitelisted", False)),
        )


@dataclass
class SmartMoneySnapshot:
    """Aggregated smart money view of one symbol."""

    symbol: str
    timestamp: float = field(default_factory=time.time)
    positions: list[WalletPosition] = field(default_factory=list)
    wallets_sampled: int = 0
    source: str = "leaderboard"
    errors: list[str] = field(default_factory=list)
    # Selection quality, used to calibrate confidence.
    avg_persistence: float = 0.0
    persistence_window_count: int = 1
    wallets_selected: int = 0
    # Whether per-wallet unrealised PnL is actually known. The backtest cannot
    # reconstruct it (no historical entry prices), and feeding 0.0 would be a
    # false claim that every whale is at break-even - which pins the quality
    # term to its floor and understates confidence by up to 0.075. Marking it
    # unavailable lets the scorer drop that term instead of guessing.
    pnl_available: bool = True

    @property
    def curation_score(self) -> float:
        """0-1 quality of the wallet set, from cross-window persistence.

        1.0 means every wallet ranked across all sampled windows; 0.0 means
        the set is a single window's snapshot (the old behaviour).
        """
        if self.persistence_window_count <= 1:
            return 0.0
        span = self.persistence_window_count - 1
        return max(0.0, min(1.0, (self.avg_persistence - 1.0) / span))

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    @property
    def longs(self) -> list[WalletPosition]:
        return [p for p in self.positions if p.size > 0]

    @property
    def shorts(self) -> list[WalletPosition]:
        return [p for p in self.positions if p.size < 0]

    @property
    def long_notional(self) -> float:
        return sum(p.notional for p in self.longs)

    @property
    def short_notional(self) -> float:
        return sum(-p.notional for p in self.shorts)

    @property
    def net_notional(self) -> float:
        return self.long_notional + self.short_notional

    @property
    def gross_notional(self) -> float:
        return self.long_notional - self.short_notional

    @property
    def long_ratio_pct(self) -> float:
        """Share of gross notional that is long, 0-100. Unweighted."""
        gross = self.long_notional + self.short_notional
        if gross <= 0:
            return 50.0
        return self.long_notional / gross * 100.0

    @property
    def weighted_long_ratio_pct(self) -> float:
        """Long share of gross notional, weighted by wallet persistence.

        Wallets that ranked across several leaderboard windows count for more
        than single-window wallets. Falls back to the unweighted ratio when no
        weighting information is available.
        """
        long_w = sum(p.weight * p.notional for p in self.longs)
        short_w = sum(p.weight * -p.notional for p in self.shorts)
        gross = long_w + short_w
        if gross <= 0:
            return self.long_ratio_pct
        return long_w / gross * 100.0

    @property
    def top1_share_pct(self) -> float:
        """Largest single position as a share of gross notional.

        A high value means the "consensus" is really one wallet's opinion.
        Measured at 79.9% under single-window selection.
        """
        gross = self.long_notional + self.short_notional
        if gross <= 0:
            return 0.0
        return max((abs(p.notional) for p in self.positions), default=0.0) / gross * 100.0

    @property
    def holder_avg_persistence(self) -> float:
        """Average persistence across wallets that actually hold the symbol."""
        if not self.positions:
            return 0.0
        return sum(p.persistence for p in self.positions) / len(self.positions)

    def weighted_long_ratio_for(self, min_persistence: int) -> float:
        """Long ratio restricted to wallets at or above a persistence floor."""
        subset = [p for p in self.positions if p.persistence >= min_persistence]
        if not subset:
            return 50.0
        long_w = sum(p.notional for p in subset if p.size > 0)
        short_w = sum(-p.notional for p in subset if p.size < 0)
        gross = long_w + short_w
        if gross <= 0:
            return 50.0
        return long_w / gross * 100.0

    @property
    def net_bias_pct(self) -> float:
        """Net directional bias as a share of gross notional, -100 to 100."""
        return self.long_ratio_pct - 50.0

    @property
    def wallet_count(self) -> int:
        return len(self.positions)

    @property
    def net_pnl_usd(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class LeaderboardWalletSource:
    """Derive the smart money set from the public leaderboard.

    The leaderboard is served from a separate stats host, NOT from the
    `/info` endpoint:

        https://stats-data.hyperliquid.xyz/Mainnet/leaderboard

    That endpoint returns every ranked account with a `windowPerformances`
    array covering day/week/month/allTime. Reading each selected account's live
    `clearinghouseState` from the public `/info` endpoint then yields a real
    smart money read with no API key.

    Selection strategy (see the constants block for the measurements that
    motivated it): rank by several windows, union the results, then keep the
    wallets that persist across windows. This replaces the earlier
    single-window top-N, which produced a tiny, unstable and highly
    concentrated sample.
    """

    STATS_URL = "https://stats-data.hyperliquid.xyz/Mainnet"
    STATS_URL_TESTNET = "https://stats-data.hyperliquid.xyz/Testnet"

    def __init__(
        self,
        market: HyperliquidMarket,
        cache_path: str | Path | None = DEFAULT_WALLET_CACHE_PATH,
        windows: tuple[str, ...] | list[str] = DEFAULT_WINDOWS,
        top_per_window: int = DEFAULT_TOP_PER_WINDOW,
        min_persistence: int = DEFAULT_MIN_PERSISTENCE,
        max_wallets: int = DEFAULT_MAX_WALLETS,
        min_account_value: float = MIN_ACCOUNT_VALUE,
        whitelist: list[str] | tuple[str, ...] = (),
        blacklist: list[str] | tuple[str, ...] = (),
    ) -> None:
        self.market = market
        self.windows = tuple(str(w) for w in windows) or DEFAULT_WINDOWS
        self.top_per_window = max(5, min(200, int(top_per_window)))
        self.min_persistence = max(1, int(min_persistence))
        self.max_wallets = max(0, int(max_wallets))
        self.min_account_value = float(min_account_value)

        self.whitelist = {str(a).lower() for a in whitelist if a}
        self.blacklist = {str(a).lower() for a in blacklist if a}

        self.cache_path = Path(cache_path) if cache_path else None
        self._wallet_cache: list[SmartWallet] = []
        self._cache_time = 0.0
        self.selection_notes: list[str] = []
        self._load_disk_cache()

    # ------------------------------------------------------------------
    # Disk cache
    # ------------------------------------------------------------------
    def _cache_signature(self) -> dict:
        """Fields that must match for a cached selection to be reusable."""
        return {
            "windows": list(self.windows),
            "top_per_window": self.top_per_window,
            "min_persistence": self.min_persistence,
            "max_wallets": self.max_wallets,
            "whitelist": sorted(self.whitelist),
            "blacklist": sorted(self.blacklist),
        }

    def _load_disk_cache(self) -> None:
        """Restore the selection from disk so restarts skip the 37 MB fetch."""
        if self.cache_path is None or not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        if payload.get("signature") != self._cache_signature():
            return
        age = time.time() - float(payload.get("cached_at", 0) or 0)
        if age > WALLET_CACHE_TTL_SECONDS:
            return

        wallets = payload.get("wallets")
        if isinstance(wallets, list) and wallets:
            restored = [
                SmartWallet.from_dict(w) for w in wallets if isinstance(w, dict)
            ]
            restored = [w for w in restored if w.address]
            if restored:
                self._wallet_cache = restored
                self._cache_time = float(payload["cached_at"] or 0)

    def _save_disk_cache(self, wallets: list[SmartWallet]) -> None:
        if self.cache_path is None:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(
                    {
                        "cached_at": self._cache_time,
                        "signature": self._cache_signature(),
                        "wallets": [w.to_dict() for w in wallets],
                    }
                ),
                encoding="utf-8",
            )
        except OSError:
            # A cache write failure must never break the trading loop.
            pass

    @property
    def stats_url(self) -> str:
        return self.STATS_URL_TESTNET if self.market.testnet else self.STATS_URL

    def leaderboard_rows(self) -> list[dict]:
        """Fetch the ranked account list from the stats host.

        The response is large (~37 MB / 45,000 accounts), so it needs a
        generous timeout and is cached aggressively by the caller.
        """
        try:
            data = request_json(
                f"{self.stats_url}/leaderboard", timeout=180, retries=2
            )
        except HttpError as exc:
            raise MarketDataError(f"leaderboard fetch failed: {exc}") from exc

        if isinstance(data, dict):
            rows = data.get("leaderboardRows")
        else:
            rows = data
        if not isinstance(rows, list):
            raise MarketDataError("unrecognised leaderboard response shape")
        return [r for r in rows if isinstance(r, dict)]

    @staticmethod
    def _window_pnl(row: dict, window: str) -> float:
        """Extract PnL for one window from the windowPerformances array."""
        perfs = row.get("windowPerformances")
        if isinstance(perfs, list):
            for entry in perfs:
                if (
                    isinstance(entry, list)
                    and len(entry) == 2
                    and entry[0] == window
                    and isinstance(entry[1], dict)
                ):
                    try:
                        return float(entry[1].get("pnl", 0) or 0)
                    except (TypeError, ValueError):
                        return 0.0
        return 0.0

    def select(self, max_age_seconds: int = WALLET_CACHE_TTL_SECONDS) -> list[SmartWallet]:
        """Choose the smart money set, ranked by cross-window persistence.

        Pipeline:
          1. Rank accounts by each window in `self.windows` and take the top
             `top_per_window` from each, skipping blacklisted / underfunded
             accounts.
          2. Union them, recording how many windows each wallet appeared in.
          3. Keep wallets with `persistence >= min_persistence`.
          4. Sort by persistence, then account size; cap at `max_wallets`.

        Whitelisted addresses are always included, marked as fully persistent,
        which makes manual curation possible without forking the code.
        """
        self.selection_notes = []

        if self._wallet_cache and time.time() - self._cache_time < max_age_seconds:
            return self._wallet_cache

        try:
            rows = self.leaderboard_rows()
        except MarketDataError:
            # Prefer a stale selection over none: positions are still real, and
            # the snapshot age check downstream handles staleness.
            self.selection_notes.append("leaderboard unavailable, reusing cached set")
            return self._wallet_cache

        # --- 1 & 2: per-window ranking, unioned with persistence counts ------
        appeared: dict[str, set[str]] = {}
        pnl_by_window: dict[str, dict[str, float]] = {}
        account_values: dict[str, float] = {}

        for window in self.windows:
            ranked = sorted(
                rows, key=lambda r: self._window_pnl(r, window), reverse=True
            )
            taken = 0
            for row in ranked:
                addr = str(row.get("ethAddress", "")).lower()
                if not addr or addr in self.blacklist:
                    continue
                try:
                    account_value = float(row.get("accountValue", 0) or 0)
                except (TypeError, ValueError):
                    continue
                # Skip accounts with nothing at stake; their positions are noise.
                if account_value < self.min_account_value:
                    continue

                appeared.setdefault(addr, set()).add(window)
                pnl_by_window.setdefault(addr, {})[window] = self._window_pnl(
                    row, window
                )
                account_values[addr] = account_value
                taken += 1
                if taken >= self.top_per_window:
                    break

        wallets = [
            SmartWallet(
                address=addr,
                windows=sorted(windows),
                persistence=len(windows),
                account_value=account_values.get(addr, 0.0),
                window_pnl=pnl_by_window.get(addr, {}),
            )
            for addr, windows in appeared.items()
        ]

        # --- Whitelist: always included, treated as maximally persistent -----
        known = {w.address for w in wallets}
        for addr in sorted(self.whitelist - self.blacklist):
            if addr in known:
                continue
            wallets.append(
                SmartWallet(
                    address=addr,
                    windows=list(self.windows),
                    persistence=len(self.windows),
                    whitelisted=True,
                )
            )

        if not wallets:
            self.selection_notes.append(
                f"no account in the {', '.join(self.windows)} top-"
                f"{self.top_per_window} met the ${self.min_account_value:,.0f} floor"
            )
            return []

        # --- 3: persistence filter, with a graceful fallback -----------------
        kept = [w for w in wallets if w.persistence >= self.min_persistence]
        if not kept:
            # Never silently return an empty set when data exists: fall back to
            # the union and say so loudly.
            lo = min(w.persistence for w in wallets)
            kept = wallets
            self.selection_notes.append(
                f"min_persistence={self.min_persistence} matched nothing "
                f"(best available persistence {lo}); using the full union of "
                f"{len(wallets)} wallets"
            )
        elif len(kept) < len(wallets):
            self.selection_notes.append(
                f"persistence filter kept {len(kept)}/{len(wallets)} wallets "
                f"(>= {self.min_persistence} of {len(self.windows)} windows)"
            )

        # --- 4: rank and cap ------------------------------------------------
        kept.sort(key=lambda w: (-w.persistence, -w.account_value))
        if self.max_wallets and len(kept) > self.max_wallets:
            self.selection_notes.append(
                f"capped at max_wallets={self.max_wallets} "
                f"from {len(kept)} candidates"
            )
            kept = kept[: self.max_wallets]

        self._wallet_cache = kept
        self._cache_time = time.time()
        self._save_disk_cache(kept)
        return kept

    def wallets(self, max_age_seconds: int = WALLET_CACHE_TTL_SECONDS) -> list[str]:
        """Selected wallet addresses. Thin wrapper over `select()`."""
        return [w.address for w in self.select(max_age_seconds)]

    def _wallet_positions(
        self, addr: str, symbol: str, persistence: int = 1, whitelisted: bool = False
    ) -> tuple[list[WalletPosition], list[str]]:
        """Read one wallet's position in `symbol`. Thread-safe."""
        out: list[WalletPosition] = []
        try:
            state = self.market.info({"type": "clearinghouseState", "user": addr})
        except MarketDataError as exc:
            return out, [f"{addr[:10]}…: {exc}"]
        if not isinstance(state, dict):
            return out, []

        try:
            account_value = float(
                state.get("marginSummary", {}).get("accountValue", 0) or 0
            )
        except (TypeError, ValueError):
            account_value = 0.0

        for ap in state.get("assetPositions", []) or []:
            pos = ap.get("position") if isinstance(ap, dict) else None
            if not isinstance(pos, dict):
                continue
            coin = str(pos.get("coin", "")).upper()
            if coin != symbol.upper():
                continue
            try:
                size = float(pos.get("szi", 0) or 0)
            except (TypeError, ValueError):
                continue
            if size == 0:
                continue

            try:
                entry = float(pos.get("entryPx", 0) or 0)
            except (TypeError, ValueError):
                entry = 0.0
            try:
                pnl = float(pos.get("unrealizedPnl", 0) or 0)
            except (TypeError, ValueError):
                pnl = 0.0

            value = pos.get("positionValue")
            try:
                notional = float(value) if value is not None else size * entry
            except (TypeError, ValueError):
                notional = size * entry

            leverage = 0.0
            lev = pos.get("leverage")
            if isinstance(lev, dict):
                try:
                    leverage = float(lev.get("value", 0) or 0)
                except (TypeError, ValueError):
                    leverage = 0.0

            out.append(
                WalletPosition(
                    wallet=addr,
                    size=size,
                    notional=abs(notional) if size > 0 else -abs(notional),
                    entry_price=entry,
                    unrealized_pnl=pnl,
                    leverage=leverage,
                    account_value=account_value,
                    persistence=persistence,
                    whitelisted=whitelisted,
                )
            )

        return out, []

    def positions(
        self,
        wallets: list[str] | list[SmartWallet],
        symbol: str,
    ) -> tuple[list[WalletPosition], list[str]]:
        """Read every wallet's live position in `symbol`, in parallel.

        Accepts either plain addresses or `SmartWallet` objects. When given
        `SmartWallet`s, each position carries the wallet's persistence so the
        score can weight repeat performers more heavily.

        The reads are independent, so this is bounded by the slowest single
        call rather than their sum. A failing wallet degrades the sample
        instead of aborting the read.
        """
        if not wallets:
            return [], []

        # Normalise to (address, persistence, whitelisted) triples so callers
        # can pass either representation.
        targets: list[tuple[str, int, bool]] = []
        for item in wallets:
            if isinstance(item, SmartWallet):
                targets.append((item.address, item.persistence, item.whitelisted))
            else:
                targets.append((str(item), 1, False))

        out: list[WalletPosition] = []
        errors: list[str] = []

        workers = min(MAX_WALLET_WORKERS, len(targets))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self._wallet_positions, addr, symbol, persistence, whitelisted
                ): addr
                for addr, persistence, whitelisted in targets
            }
            for future in as_completed(futures):
                addr = futures[future]
                try:
                    positions, errs = future.result()
                except Exception as exc:  # noqa: BLE001 - isolate per-wallet failures
                    errors.append(f"{addr[:10]}…: {exc}")
                    continue
                out.extend(positions)
                errors.extend(errs)

        return out, errors


class HyperfeedSource:
    """Optional curated whale feed.

    Enabled only when SENPI_API_KEY is present. Kept intentionally minimal:
    the response is expected to be a list of {wallet, coin, size, notional}
    records. If the shape does not match, the factor reports zero confidence
    rather than guessing.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (
            base_url or os.getenv("SENPI_BASE_URL", "https://api.senpi.ai")
        ).rstrip("/")
        self.api_key = os.getenv("SENPI_API_KEY", "")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def positions(self, symbol: str) -> tuple[list[WalletPosition], list[str]]:
        if not self.available:
            return [], ["SENPI_API_KEY not set"]

        try:
            payload = request_json(
                f"{self.base_url}/v1/smart-money/{symbol.upper()}",
                timeout=15,
                retries=2,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        except HttpError as exc:
            return [], [f"hyperfeed request failed: {exc}"]

        records = payload.get("positions") if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            return [], ["hyperfeed response shape unrecognised"]

        out: list[WalletPosition] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            try:
                size = float(rec.get("size", 0) or 0)
            except (TypeError, ValueError):
                continue
            if size == 0:
                continue
            try:
                notional = float(rec.get("notional", 0) or 0)
            except (TypeError, ValueError):
                notional = 0.0
            # Keep the address intact: truncating it collapses distinct wallets
            # into the same identifier, which breaks per-wallet aggregation and
            # any later reconciliation against the feed.
            wallet = str(rec.get("wallet") or rec.get("address") or "").lower()
            out.append(
                WalletPosition(
                    wallet=wallet or "unknown",
                    size=size,
                    notional=signed_notional(notional, size),
                )
            )
        return out, []


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_snapshot(
    snapshot: SmartMoneySnapshot,
    symbol: str | None = None,
    data_network: str = "mainnet",
) -> FactorScore:
    """Score a smart money snapshot. Pure: no I/O, no instance state.

    Both the live factor and the backtest call this, so their scoring,
    weighting and confidence maths cannot drift apart.

    Previously the backtest borrowed the method off a half-constructed
    instance (`SmartMoneyFactor.__new__`) and set the single attribute the
    method happened to read. It worked, but nothing would have caught it
    breaking: a new instance dependency would have surfaced as an
    AttributeError at runtime, in the backtest only.
    """
    snap = snapshot
    name = (symbol or snap.symbol).upper()

    reasons: list[str] = []
    details: dict[str, object] = {
        "source": snap.source,
        "data_network": data_network,
        "wallets_sampled": snap.wallets_sampled,
        "wallet_count": snap.wallet_count,
        "snapshot_age_seconds": round(snap.age_seconds, 1),
    }

    # --- Staleness gate ------------------------------------------------
    if snap.age_seconds > MAX_SNAPSHOT_AGE_SECONDS:
        return FactorScore(
            name="smart_money",
            score=50.0,
            confidence=0.0,
            reasons=[
                f"snapshot is {snap.age_seconds / 60:.1f} min old "
                f"(limit {MAX_SNAPSHOT_AGE_SECONDS / 60:.0f} min): refusing to trade "
                "on stale whale data"
            ],
            details=details,
        )

    if snap.wallet_count == 0:
        detail = f"; errors: {snap.errors[:2]}" if snap.errors else ""
        return FactorScore(
            name="smart_money",
            score=50.0,
            confidence=0.0,
            reasons=[
                f"no sampled wallet currently holds {name}"
                f" ({snap.wallets_sampled} wallets checked){detail}"
            ],
            details=details,
        )

    # --- Directional score ---------------------------------------------
    # Persistence-weighted: a wallet that ranked across all sampled windows
    # counts proportionally more than a single-window wallet.
    long_ratio = snap.weighted_long_ratio_pct
    raw_ratio = snap.long_ratio_pct
    top1 = snap.top1_share_pct
    details["long_ratio_pct"] = round(raw_ratio, 2)
    details["weighted_long_ratio_pct"] = round(long_ratio, 2)
    details["long_notional_usd"] = round(snap.long_notional, 2)
    details["short_notional_usd"] = round(snap.short_notional, 2)
    details["net_notional_usd"] = round(snap.net_notional, 2)
    details["net_pnl_usd"] = round(snap.net_pnl_usd, 2)
    details["top1_share_pct"] = round(top1, 2)
    details["holder_avg_persistence"] = round(snap.holder_avg_persistence, 2)

    # The ZoneIn reference formula: long_ratio >= 50 means long.
    # Centring at 50 and amplifying gives a continuous score.
    score = 50.0 + (long_ratio - 50.0) * 1.6
    score = max(0.0, min(100.0, score))

    reasons.append(
        f"{long_ratio:.1f}% of whale notional is long, persistence-weighted "
        f"({len(snap.longs)}L/${snap.long_notional / 1e6:.2f}M vs "
        f"{len(snap.shorts)}S/${snap.short_notional / 1e6:.2f}M; "
        f"raw {raw_ratio:.1f}%)"
    )

    # Concentration warning: a single wallet dominating the book means we
    # are reading one opinion, not a consensus.
    if top1 >= 60.0:
        reasons.append(
            f"warning: single largest position is {top1:.0f}% of gross - "
            "the 'consensus' is effectively one wallet"
        )

    # --- Conviction from wallet consensus ------------------------------
    # Unanimity is far more meaningful than a 51/49 split.
    total = snap.wallet_count
    dominant = max(len(snap.longs), len(snap.shorts))
    consensus = dominant / total if total else 0.0
    details["consensus_pct"] = round(consensus * 100, 1)

    # --- Wallet quality: are the whales actually in profit? ------------
    # Only meaningful when PnL is known. The backtest cannot reconstruct it,
    # and in that case the term is pinned to its own range midpoint rather
    # than to a fabricated observation. See the confidence block below.
    winners = [p for p in snap.positions if p.unrealized_pnl > 0]
    winner_ratio = len(winners) / total if total else 0.0
    details["in_profit_pct"] = round(winner_ratio * 100, 1)
    details["pnl_available"] = snap.pnl_available

    if snap.net_pnl_usd:
        reasons.append(
            f"aggregate unrealized PnL ${snap.net_pnl_usd / 1e3:+.1f}k, "
            f"{winner_ratio:.0%} of positions in profit"
        )

    # --- Confidence -----------------------------------------------------
    sample_conf = min(1.0, total / SAMPLE_SIZE_SATURATION)
    consensus_conf = max(0.0, (consensus - 0.5) * 2.0)  # 50% -> 0, 100% -> 1
    curation_conf = snap.curation_score

    # Concentration: when one wallet dominates the book, the effective
    # sample size is 1 regardless of how many holders there are. Single
    # window selection measured 79.9% here.
    concentration_conf = max(0.0, min(1.0, (80.0 - top1) / 60.0))

    if snap.pnl_available:
        quality_conf = 0.5 + 0.5 * winner_ratio
    else:
        # `quality_conf` ranges over [0.5, 1.0]. With winner_ratio unknown,
        # the midpoint is the neutral estimate: it puts the score at the
        # centre of the range the live agent would have produced.
        #
        # Dropping the term and renormalising the rest is NOT equivalent -
        # it redistributes weight onto unrelated factors and lands BELOW
        # the live range, i.e. biased in the opposite direction.
        quality_conf = NEUTRAL_QUALITY_CONF
        details["pnl_term"] = f"unknown; using midpoint {NEUTRAL_QUALITY_CONF}"
        reasons.append(
            "per-wallet PnL unavailable (not reconstructable); quality term "
            f"set to its neutral midpoint {NEUTRAL_QUALITY_CONF}"
        )

    confidence = (
        WEIGHT_SAMPLE * sample_conf
        + WEIGHT_CONSENSUS * consensus_conf
        + WEIGHT_QUALITY * quality_conf
        + WEIGHT_CURATION * curation_conf
        + WEIGHT_CONCENTRATION * concentration_conf
    )

    details["curation_score"] = round(curation_conf, 3)
    details["concentration_score"] = round(concentration_conf, 3)
    details["avg_persistence"] = round(snap.avg_persistence, 2)
    details["persistence_window_count"] = snap.persistence_window_count
    details["wallets_selected"] = snap.wallets_selected

    if snap.source == "leaderboard":
        # A set that survived the persistence filter is materially better
        # than a single-window snapshot, so the ceiling scales with
        # curation instead of being pinned at a flat 0.75.
        cap = LEADERBOARD_CONF_CAP_BASE + LEADERBOARD_CONF_CAP_CURATION * curation_conf
        if confidence > cap:
            confidence = cap
            reasons.append(f"leaderboard-derived set: confidence capped at {cap:.0%}")
    details["curated"] = snap.source == "hyperfeed"

    if snap.persistence_window_count > 1:
        reasons.append(
            f"wallet set screened across {snap.persistence_window_count} windows, "
            f"avg persistence {snap.avg_persistence:.1f} "
            f"(curation {curation_conf:.0%})"
        )

    if snap.errors:
        confidence *= WALLET_ERROR_CONFIDENCE_PENALTY
        reasons.append(f"{len(snap.errors)} wallet reads failed")

    return FactorScore(
        name="smart_money",
        score=score,
        confidence=max(0.0, min(1.0, confidence)),
        reasons=reasons,
        details=details,
    ).clamp()


# ---------------------------------------------------------------------------
# Factor
# ---------------------------------------------------------------------------


class SmartMoneyFactor:
    """Scores a symbol from aggregated whale positioning.

    Whale data is always read from MAINNET, regardless of which network the
    agent trades on. Testnet has no meaningful leaderboard and no whales, so
    querying testnet with mainnet addresses returns empty positions and
    silently zeroes the factor. The signal describes real market conditions;
    only order execution follows the configured network.
    """

    def __init__(
        self,
        market: HyperliquidMarket,
        use_hyperfeed: bool = False,
        data_market: HyperliquidMarket | None = None,
        cache_path: str | Path | None = DEFAULT_WALLET_CACHE_PATH,
        windows: tuple[str, ...] | list[str] = DEFAULT_WINDOWS,
        top_per_window: int = DEFAULT_TOP_PER_WINDOW,
        min_persistence: int = DEFAULT_MIN_PERSISTENCE,
        max_wallets: int = DEFAULT_MAX_WALLETS,
        min_account_value: float = MIN_ACCOUNT_VALUE,
        whitelist: list[str] | tuple[str, ...] = (),
        blacklist: list[str] | tuple[str, ...] = (),
    ) -> None:
        self.market = market
        self.use_hyperfeed = use_hyperfeed
        # Reuse the caller's client when it is already mainnet, so tests and
        # mainnet runs do not open a redundant client.
        self.data_market = data_market or (
            market if not market.testnet else HyperliquidMarket(testnet=False)
        )
        self.leaderboard = LeaderboardWalletSource(
            self.data_market,
            cache_path=cache_path,
            windows=windows,
            top_per_window=top_per_window,
            min_persistence=min_persistence,
            max_wallets=max_wallets,
            min_account_value=min_account_value,
            whitelist=whitelist,
            blacklist=blacklist,
        )
        self.hyperfeed = HyperfeedSource()
        self._snapshots: dict[str, SmartMoneySnapshot] = {}

    @property
    def uses_mainnet_data(self) -> bool:
        return not self.data_market.testnet

    # ------------------------------------------------------------------
    def peek(self, symbol: str) -> SmartMoneySnapshot | None:
        """Most recent cached snapshot, without a network call."""
        return self._snapshots.get(symbol.upper())

    def snapshot(self, symbol: str, force: bool = False) -> SmartMoneySnapshot:
        """Fetch (or reuse a fresh) smart money snapshot for `symbol`."""
        name = symbol.upper()
        cached = self._snapshots.get(name)
        if cached and not force and cached.age_seconds < SNAPSHOT_REUSE_SECONDS:
            return cached

        snap: SmartMoneySnapshot
        if self.use_hyperfeed and self.hyperfeed.available:
            positions, errors = self.hyperfeed.positions(name)
            snap = SmartMoneySnapshot(
                symbol=name,
                positions=positions,
                wallets_sampled=len({p.wallet for p in positions}),
                source="hyperfeed",
                errors=errors,
                # A curated feed is already screened; treat it as fully
                # persistent so confidence calibration does not double-penalise.
                avg_persistence=float(len(DEFAULT_WINDOWS)),
                persistence_window_count=len(DEFAULT_WINDOWS),
                wallets_selected=len({p.wallet for p in positions}),
            )
        else:
            selection = self.leaderboard.select()
            # Pass the SmartWallet objects so each position carries the
            # wallet's persistence for weighting.
            positions, errors = self.leaderboard.positions(selection, name)

            # Persistence of the wallets that actually hold the symbol is what
            # calibrates confidence; the candidate pool's average is not
            # informative when most candidates hold nothing.
            avg_persistence = (
                sum(p.persistence for p in positions) / len(positions)
                if positions
                else 0.0
            )
            snap = SmartMoneySnapshot(
                symbol=name,
                positions=positions,
                wallets_sampled=len(selection),
                source="leaderboard",
                errors=errors + self.leaderboard.selection_notes,
                avg_persistence=avg_persistence,
                persistence_window_count=len(self.leaderboard.windows),
                wallets_selected=len(selection),
            )

        self._snapshots[name] = snap
        return snap

    # ------------------------------------------------------------------
    def evaluate(self, symbol: str, snapshot: SmartMoneySnapshot | None = None) -> FactorScore:
        """Fetch a snapshot, then score it.

        All the scoring lives in the module-level `score_snapshot`, which is
        pure. This method only handles the I/O half.
        """
        name = symbol.upper()
        try:
            snap = snapshot or self.snapshot(name)
        except (MarketDataError, HttpError) as exc:
            return FactorScore(
                name="smart_money",
                score=50.0,
                confidence=0.0,
                reasons=[f"smart money data unavailable: {exc}"],
            )

        return score_snapshot(
            snap,
            name,
            data_network="mainnet" if self.uses_mainnet_data else "testnet",
        )


# ---------------------------------------------------------------------------
# Construction from config
# ---------------------------------------------------------------------------


def smart_money_factor_from_config(
    config: BotConfig,
    market: HyperliquidMarket,
    force_hyperfeed: bool = False,
) -> SmartMoneyFactor:
    """Build a factor wired to the config's wallet-selection settings.

    The SINGLE place that maps `SmartMoneySettings` onto the constructor. Both
    the live synthesizer and the snapshot recorder go through here.

    They used to assemble the argument list independently, and the recorder
    passed none of it - silently falling back to the module defaults. Those
    happened to equal the shipped config, so the divergence stayed invisible
    until `smart_money` was tuned, at which point the recorded history stopped
    describing the strategy being run, and the wallet cache signature stopped
    matching too.
    """
    sm = config.smart_money
    return SmartMoneyFactor(
        market,
        use_hyperfeed=force_hyperfeed or config.use_hyperfeed,
        windows=sm.windows,
        top_per_window=sm.top_per_window,
        min_persistence=sm.min_persistence,
        max_wallets=sm.max_wallets,
        min_account_value=sm.min_account_value,
        whitelist=sm.whitelist,
        blacklist=sm.blacklist,
    )
