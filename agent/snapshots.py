"""Records whale position snapshots for future backtesting.

Why this exists: the public Hyperliquid API has no historical positions
endpoint, so the smart money factor (40% of the live weight, and the only
factor with evidence behind it) cannot be backtested today. The only way that
ever changes is to start recording now and accumulate a history.

Records are append-only JSONL so a crash mid-write loses at most one line, and
each line is complete enough to reconstruct the factor offline: the per-wallet
positions plus the selection metadata that produced them.

Run it on a schedule alongside the agent, or from the CLI:

    python record_snapshots.py --symbols BTC,ETH --state logs/agent_state.json
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import BotConfig
from .factors.smart_money import (
    SmartMoneyFactor,
    SmartMoneySnapshot,
    smart_money_factor_from_config,
)
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


def now_ts() -> float:
    return time.time()
