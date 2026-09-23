"""Records two datasets forward, because neither has a history endpoint.

**Whale positions.** The public Hyperliquid API has no historical positions
endpoint, so the smart money factor (40% of the live weight) cannot be
backtested today. The only way that ever changes is to start recording now.

**Aggregate positioning.** Open interest, funding and premium for every perp,
plus the two series Hyperliquid does not publish at all - who is trading with
market orders, and how the accounts split long against short. Binance does
publish those, but caps its history endpoints at ~20 days, so the same argument
applies with even less room to wait.

The two are deliberately separate fields rather than one merged factor: wallets
and aggregates are different families, and the question worth asking later is
where they disagree. Merging them here would answer it in advance.

Records are append-only JSONL so a crash mid-write loses at most one line, and
each line is complete enough to reconstruct the factor offline.

Run it on a schedule alongside the agent, or from the CLI:

    python record_snapshots.py --symbols BTC,ETH
    python record_snapshots.py --summary
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import BotConfig
from .factors.smart_money import (
    SmartMoneyFactor,
    SmartMoneySnapshot,
    smart_money_factor_from_config,
)
from .http_util import request_json
from .market_data import HyperliquidMarket, mainnet_data_market

DEFAULT_SNAPSHOT_PATH = "snapshots/whale_positions.jsonl"
DEFAULT_SYMBOLS = ("BTC", "ETH")


@dataclass
class SnapshotRecord:
    """One timestamped observation of whale positioning."""

    timestamp: float
    symbol: str
    wallets_sampled: int
    wallet_count: int
    long_ratio_pct: float
    weighted_long_ratio_pct: float
    top1_share_pct: float
    holder_avg_persistence: float
    long_notional_usd: float
    short_notional_usd: float
    positions: list[dict]
    # The wallet-selection parameters in force when this was recorded. Without
    # them the snapshot cannot be reproduced or compared across config changes:
    # the same symbol, at the same instant, yields different numbers under
    # different selection settings.
    selection: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=str)

    @classmethod
    def from_snapshot(
        cls,
        snap: SmartMoneySnapshot,
        selection: dict | None = None,
    ) -> "SnapshotRecord":
        return cls(
            timestamp=snap.timestamp,
            symbol=snap.symbol,
            wallets_sampled=snap.wallets_sampled,
            wallet_count=snap.wallet_count,
            long_ratio_pct=round(snap.long_ratio_pct, 4),
            weighted_long_ratio_pct=round(snap.weighted_long_ratio_pct, 4),
            top1_share_pct=round(snap.top1_share_pct, 4),
            holder_avg_persistence=round(snap.holder_avg_persistence, 3),
            long_notional_usd=round(snap.long_notional, 2),
            short_notional_usd=round(snap.short_notional, 2),
            positions=[
                {
                    "wallet": p.wallet,
                    "size": p.size,
                    "notional": round(p.notional, 2),
                    "entry_price": p.entry_price,
                    "unrealized_pnl": round(p.unrealized_pnl, 2),
                    "persistence": p.persistence,
                    "whitelisted": p.whitelisted,
                }
                for p in snap.positions
            ],
            selection=dict(selection or {}),
        )


class SnapshotRecorder:
    """Appends whale snapshots to a JSONL file."""

    def __init__(
        self,
        path: str | Path | None = DEFAULT_SNAPSHOT_PATH,
        market: HyperliquidMarket | None = None,
        use_hyperfeed: bool = False,
        config: BotConfig | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        # Whale data is mainnet-only, same rule as the live factor - and now
        # the same *code*, so the two cannot drift apart.
        self.market = mainnet_data_market(market)

        # The recorder must select wallets exactly as the agent does, or the
        # recorded history describes a different strategy than the one being
        # run. Going through the shared factory is what guarantees that; see
        # `smart_money_factor_from_config` for why the two paths must not
        # assemble the arguments independently.
        if config is not None:
            self.factor = smart_money_factor_from_config(
                config, self.market, force_hyperfeed=use_hyperfeed
            )
        else:
            self.factor = SmartMoneyFactor(self.market, use_hyperfeed=use_hyperfeed)
        self.errors: list[str] = []

    @property
    def selection(self) -> dict:
        """Wallet-selection parameters in force, stored for provenance.

        Recorded on every snapshot so a later backtest can tell whether two
        stretches of history were even produced by the same selection rules.
        """
        lb = self.factor.leaderboard
        return {
            "windows": list(lb.windows),
            "top_per_window": lb.top_per_window,
            "min_persistence": lb.min_persistence,
            "max_wallets": lb.max_wallets,
            "min_account_value": lb.min_account_value,
            "whitelist": sorted(lb.whitelist),
            "blacklist": sorted(lb.blacklist),
            "use_hyperfeed": self.factor.use_hyperfeed,
        }

    def record(self, symbols: tuple[str, ...] | list[str] = DEFAULT_SYMBOLS) -> int:
        """Capture one snapshot per symbol. Returns the number written."""
        if self.path is None:
            return 0
        self.errors = []
        written = 0
        selection = self.selection

        for symbol in symbols:
            name = symbol.upper()
            try:
                snap = self.factor.snapshot(name, force=True)
            except Exception as exc:  # noqa: BLE001 - a collection run must not die
                self.errors.append(f"{name}: {exc}")
                continue

            record = SnapshotRecord.from_snapshot(snap, selection)
            if self._append(record):
                written += 1

        return written

    def _append(self, record: SnapshotRecord) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(record.to_json() + "\n")
            return True
        except OSError as exc:
            self.errors.append(f"write failed: {exc}")
            return False


def load_records(path: str | Path) -> list[SnapshotRecord]:
    """Read every recorded snapshot, skipping malformed lines."""
    path = Path(path)
    if not path.exists():
        return []

    out: list[SnapshotRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            out.append(
                SnapshotRecord(
                    timestamp=float(payload["timestamp"]),
                    symbol=str(payload["symbol"]),
                    wallets_sampled=int(payload.get("wallets_sampled", 0)),
                    wallet_count=int(payload.get("wallet_count", 0)),
                    long_ratio_pct=float(payload.get("long_ratio_pct", 50.0)),
                    weighted_long_ratio_pct=float(
                        payload.get("weighted_long_ratio_pct", 50.0)
                    ),
                    top1_share_pct=float(payload.get("top1_share_pct", 0.0)),
                    holder_avg_persistence=float(
                        payload.get("holder_avg_persistence", 0.0)
                    ),
                    long_notional_usd=float(payload.get("long_notional_usd", 0.0)),
                    short_notional_usd=float(payload.get("short_notional_usd", 0.0)),
                    positions=list(payload.get("positions") or []),
                    # Absent on records written before provenance was added.
                    selection=dict(payload.get("selection") or {}),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def coverage_summary(records: list[SnapshotRecord]) -> str:
    """How much history has accumulated, and whether it is usable yet."""
    if not records:
        return "no snapshots recorded yet"

    by_symbol: dict[str, list[float]] = {}
    for record in records:
        by_symbol.setdefault(record.symbol, []).append(record.timestamp)

    lines = []
    for symbol, stamps in sorted(by_symbol.items()):
        span_days = (max(stamps) - min(stamps)) / 86_400
        lines.append(
            f"  {symbol}: {len(stamps)} snapshots spanning {span_days:.1f} days"
        )

    total_days = (max(r.timestamp for r in records) - min(r.timestamp for r in records)) / 86_400
    if total_days < 30:
        lines.append(
            f"  NOT YET USABLE: {total_days:.1f} days of history. A smart money "
            "backtest needs months, not days."
        )
    return "\n".join(lines)


DEFAULT_MARKET_SNAPSHOT_PATH = "snapshots/market_positioning.jsonl"

#: Binance's public futures host. Everything read from it below is free and
#: needs no key - which is the point: these are the same aggregates a paid data
#: product resells, and the premium buys nothing but convenience.
BINANCE_FUTURES_URL = "https://fapi.binance.com"

#: The native cadence of everything in this file. Hyperliquid settles funding
#: hourly and Binance's taker/long-short series are `period=1h`, so sampling any
#: faster would only pad the file with repeats of the same observation.
MARKET_SNAPSHOT_INTERVAL_SECONDS = 3600.0
BINANCE_PERIOD = "1h"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class MarketSnapshotRecord:
    """One timestamped observation of *aggregate* positioning.

    The counterpart to `SnapshotRecord`, not a replacement. That one records what
    the selected whales hold; this one records what the market as a whole holds.
    They are different families - wallets versus aggregates - and the question
    worth asking later is where the two disagree, which cannot be asked at all
    until both have history.
    """

    timestamp: float
    #: Every Hyperliquid perp, from ONE `metaAndAssetCtxs` call.
    hyperliquid: dict[str, dict] = field(default_factory=dict)
    #: Binance futures, per symbol. A second venue's view, plus the two series
    #: Hyperliquid does not publish at all: who is buying with market orders, and
    #: how the *accounts* are split long against short.
    binance: dict[str, dict] = field(default_factory=dict)
    #: Per-source failures, kept inside the record rather than only on stderr. A
    #: snapshot that silently recorded half of what it meant to is worse than no
    #: snapshot at all, because the gap is invisible afterwards.
    errors: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, default=str)


class MarketSnapshotRecorder:
    """Appends aggregate-positioning snapshots to a JSONL file.

    Recording forward for the same reason `SnapshotRecorder` does, and here the
    reason is harder: Binance caps its history endpoints at 500 points - about
    20 days at 1h - and Hyperliquid publishes no OI or funding history at all.
    Nothing about these series can be measured until this has been running for a
    while, and there is no way to buy back the missing period.
    """

    def __init__(
        self,
        path: str | Path | None = DEFAULT_MARKET_SNAPSHOT_PATH,
        market: HyperliquidMarket | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        # Mainnet, same rule as every other signal read. The OI and funding of a
        # testnet market describe a market nobody trades.
        self.market = mainnet_data_market(market)
        self.errors: list[str] = []

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------
    def _fetch_hyperliquid(self) -> dict[str, dict]:
        """OI, funding and premium for every perp, in one call.

        Only the fields that cannot be recovered elsewhere. The mark price is
        deliberately not stored: the candle archive carries prices for all of
        these symbols, so OI in base units converts to USD at analysis time and
        nothing is lost by leaving it out of a file that grows forever.

        `premium` is stored as a ratio rather than as two prices for the same
        reason - and it is the one piece here that candles cannot reconstruct,
        because it is the perp's own mark against its oracle.
        """
        out: dict[str, dict] = {}
        for ctx in self.market.asset_contexts():
            if ctx.open_interest <= 0:
                # Delisted perps still come back from this endpoint with every
                # field at zero - measured on mainnet, 56 of 234, names like
                # MATIC and RNDR that have since been renamed - and a market that
                # does not exist has no positioning to record. A new listing
                # starts appearing here as soon as it has any open interest, so
                # leaving the empty rows out costs nothing but a quarter of a
                # file that grows forever.
                #
                # (An earlier version of this check tested for a leading "@",
                # on the assumption that spot pairs share this response. They do
                # not: this endpoint is perps only, and the candles archive is
                # the one that carries spot. The check excluded nothing.)
                continue
            premium = 0.0
            if ctx.oracle_price > 0:
                premium = (ctx.mark_price - ctx.oracle_price) / ctx.oracle_price
            out[ctx.name] = {
                "oi": round(ctx.open_interest, 6),
                "funding": round(ctx.funding, 10),
                "premium": round(premium, 8),
                "day_vol": round(ctx.day_volume, 4),
            }
        return out

    def _binance_get(self, path: str, pair: str, limit: int = 1) -> Any:
        # `payload=None` is what makes `request_json` issue a GET, and going
        # through it means the retry budget and the truncated-transfer guard
        # apply here too instead of being re-invented per caller.
        url = f"{BINANCE_FUTURES_URL}{path}?symbol={pair}"
        if limit:
            url += f"&period={BINANCE_PERIOD}&limit={limit}"
        return request_json(url, timeout=15, retries=3)

    def _binance_last(self, path: str, pair: str) -> dict:
        """The newest point of a `/futures/data/*` series, or raise.

        Raising rather than returning {} so the caller records *which* series
        failed: a symbol silently missing one of three is a hole nothing
        downstream can see.
        """
        body = self._binance_get(path, pair)
        if not isinstance(body, list) or not body:
            raise ValueError(f"{path} returned {str(body)[:80]!r}")
        return body[-1]

    def _fetch_binance(self, symbol: str) -> dict:
        """The series Hyperliquid does not publish, for one symbol.

        The pair is mapped as `<SYM>USDT`, which is right for the majors and
        wrong for names Binance lists under a multiplier (`1000PEPE`); that is
        why a failure is recorded per symbol instead of failing the pass.
        """
        pair = f"{symbol.upper()}USDT"
        out: dict = {}

        try:
            taker = self._binance_last("/futures/data/takerlongshortRatio", pair)
            out["taker"] = {
                "buy_vol": _num(taker.get("buyVol")),
                "sell_vol": _num(taker.get("sellVol")),
                "ratio": _num(taker.get("buySellRatio")),
            }
        except Exception as exc:  # noqa: BLE001 - one series failing is not fatal
            self.errors.append(f"{symbol} taker: {exc}")

        try:
            accounts = self._binance_last(
                "/futures/data/globalLongShortAccountRatio", pair
            )
            out["accounts"] = {
                "long_share": _num(accounts.get("longAccount")),
                "short_share": _num(accounts.get("shortAccount")),
                "ratio": _num(accounts.get("longShortRatio")),
            }
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"{symbol} accounts: {exc}")

        try:
            # No `period` on this one; `limit=0` is how `_binance_get` knows.
            premium = self._binance_get("/fapi/v1/premiumIndex", pair, limit=0)
            if not isinstance(premium, dict) or "markPrice" not in premium:
                raise ValueError(f"premiumIndex returned {str(premium)[:80]!r}")
            out["funding"] = {
                "funding_rate": _num(premium.get("lastFundingRate")),
                "mark_price": _num(premium.get("markPrice")),
                "index_price": _num(premium.get("indexPrice")),
            }
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"{symbol} funding: {exc}")

        return out

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def _last_timestamp(self) -> float | None:
        """Timestamp of the newest record, or None when there is no file.

        Read rather than tracked in a state file: the file *is* the state, and a
        second copy of it could disagree after a crash.
        """
        if self.path is None or not self.path.exists():
            return None
        last = ""
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last = line
        if not last:
            return None
        try:
            return float(json.loads(last)["timestamp"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None

    def record(
        self,
        symbols: tuple[str, ...] | list[str] = DEFAULT_SYMBOLS,
        force: bool = False,
    ) -> bool:
        """Capture one observation, or skip it when the last is still fresh.

        Returns whether a record was written. Skipping is the normal outcome
        when this is called from a 15-minute cron: the underlying series update
        hourly, so three of every four passes have nothing new to add.
        """
        if self.path is None:
            return False
        self.errors = []

        if not force:
            previous = self._last_timestamp()
            now = time.time()
            if previous is not None and now - previous < MARKET_SNAPSHOT_INTERVAL_SECONDS:
                return False

        record = MarketSnapshotRecord(timestamp=time.time())
        try:
            record.hyperliquid = self._fetch_hyperliquid()
        except Exception as exc:  # noqa: BLE001 - partial is better than nothing
            self.errors.append(f"hyperliquid: {exc}")
        for symbol in symbols:
            record.binance[symbol.upper()] = self._fetch_binance(symbol)

        record.errors = list(self.errors)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(record.to_json() + "\n")
            return True
        except OSError as exc:
            self.errors.append(f"write failed: {exc}")
            return False


def load_market_records(path: str | Path) -> list[MarketSnapshotRecord]:
    """Read every recorded market snapshot, skipping malformed lines."""
    path = Path(path)
    if not path.exists():
        return []

    out: list[MarketSnapshotRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            out.append(
                MarketSnapshotRecord(
                    timestamp=float(payload["timestamp"]),
                    hyperliquid=dict(payload.get("hyperliquid") or {}),
                    binance=dict(payload.get("binance") or {}),
                    errors=list(payload.get("errors") or []),
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return out


def market_coverage_summary(records: list[MarketSnapshotRecord]) -> str:
    """How much positioning history has accumulated, using the same bar as
    `coverage_summary`: months, not days."""
    if not records:
        return "no market snapshots recorded yet"

    span_days = (max(r.timestamp for r in records) - min(r.timestamp for r in records)) / 86_400
    assets = len(records[-1].hyperliquid)
    failed = sum(1 for r in records if r.errors)
    lines = [
        f"  {len(records)} snapshots spanning {span_days:.1f} days",
        f"  {assets} Hyperliquid perps in the newest one; "
        f"{len(records[-1].binance)} Binance symbols",
    ]
    if failed:
        lines.append(
            f"  {failed} snapshot(s) recorded with an error - a partial row is "
            "not the same as a complete one"
        )
    if span_days < 30:
        lines.append(
            f"  NOT YET USABLE: {span_days:.1f} days of history. Binance caps "
            "its own history endpoints at ~20 days, so this is the only route."
        )
    return "\n".join(lines)


def now_ts() -> float:
    return time.time()
