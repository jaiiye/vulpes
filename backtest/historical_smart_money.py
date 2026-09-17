"""The smart money factor, served from reconstructed whale history.

This is the piece that makes the 40% factor backtestable. It rebuilds a
`SmartMoneySnapshot` from reconstructed funding history and then hands it to
the LIVE `SmartMoneyFactor.evaluate`, so the scoring, weighting and confidence
math are byte-for-byte the production code rather than a parallel
implementation.

Coverage is the honest limitation: only ~26% of leaderboard accounts appear in
the funding endpoint at all, and only ~20 have records within the last month.
So a recent backtest runs on roughly the same order of sample as the live
factor's holder count, which is thin. Coverage is reported with every run.
"""

from __future__ import annotations

from .whale_history import WalletPositionSeries

from agent.factors.smart_money import (
    SmartMoneySnapshot,
    WalletPosition,
    score_snapshot,
)

# A position is treated as current only if a funding record exists this
# recently. Funding settles hourly, so 6h tolerates a few missed settlements
# while still refusing genuinely stale data.
DEFAULT_MAX_AGE_MS = 6 * 3_600_000


class HistoricalSmartMoney:
    """Drop-in replacement for `SmartMoneyFactor` for backtesting.

    Scoring is delegated to the shared, pure `score_snapshot`, so the backtest
    and the live agent run identical maths by construction rather than by
    convention.
    """

    def __init__(
        self,
        history: dict[str, WalletPositionSeries],
        price_lookup,
        persistence: dict[str, int] | None = None,
        max_age_ms: int = DEFAULT_MAX_AGE_MS,
        window_count: int = 3,
    ) -> None:
        self.history = history
        self.price_lookup = price_lookup  # callable(ts_ms) -> price
        self.persistence = persistence or {}
        self.max_age_ms = max_age_ms
        # Must match the live config's `smart_money.windows` length: it is the
        # denominator of `SmartMoneySnapshot.curation_score`, so a mismatch
        # miscalibrates confidence relative to the live agent.
        self.window_count = max(1, int(window_count))
        self._now_ms = 0
        self.coverage_notes: list[str] = []

    # ------------------------------------------------------------------
    def set_now(self, ts_ms: int) -> None:
        self._now_ms = int(ts_ms)

    # ------------------------------------------------------------------
    def build_snapshot(self, symbol: str) -> SmartMoneySnapshot:
        """Aggregate every wallet's reconstructed position at the clock time."""
        price = self.price_lookup(self._now_ms)
        positions: list[WalletPosition] = []

        for wallet, series in self.history.items():
            size = series.position_at(self._now_ms, self.max_age_ms)
            if size is None or size == 0:
                continue
            positions.append(
                WalletPosition(
                    wallet=wallet,
                    size=size,
                    # Signed notional, matching the live convention.
                    notional=size * price,
                    entry_price=0.0,
                    persistence=self.persistence.get(wallet.lower(), 1),
                )
            )

        selected = len(self.history)
        return SmartMoneySnapshot(
            symbol=symbol.upper(),
            positions=positions,
            wallets_sampled=selected,
            wallets_selected=selected,
            source="leaderboard",
            # Carried from the live config so curation_score is normalised the
            # same way the live agent normalises it.
            persistence_window_count=self.window_count,
            avg_persistence=(
                sum(p.persistence for p in positions) / len(positions)
                if positions
                else 0.0
            ),
            # Funding history carries no entry price, so unrealised PnL cannot
            # be reconstructed. Declaring that lets the scorer drop the quality
            # term instead of reading a fabricated 0.0 as "all break-even".
            pnl_available=False,
        )

    def snapshot(self, symbol: str, force: bool = False) -> SmartMoneySnapshot:
        return self.build_snapshot(symbol)

    def peek(self, symbol: str):
        return None

    def evaluate(self, symbol: str, snapshot=None):
        """Score via the shared pure scorer.

        `data_network="mainnet"` because reconstructed whale history always
        describes mainnet activity, whichever network the backtest simulates
        execution on - the same rule the live factor follows.
        """
        snap = snapshot or self.build_snapshot(symbol)
        return score_snapshot(snap, symbol, data_network="mainnet")


def coverage_report(
    history: dict[str, WalletPositionSeries], total_candidates: int
) -> str:
    """Describe how much of the intended wallet set was reconstructed."""
    if not history:
        return (
            "  NO WHALE HISTORY: the smart money factor was not included in "
            "this run."
        )
    pct = len(history) / total_candidates * 100 if total_candidates else 0.0
    points = sum(len(s) for s in history.values())
    lines = [
        f"  reconstructed wallets: {len(history)}/{total_candidates} "
        f"({pct:.0f}% of candidates)",
        f"  total position points: {points:,}",
    ]
    if pct < 40:
        lines.append(
            "  LOW COVERAGE: the smart money factor runs on a much smaller "
            "sample than production, so its contribution is under-powered."
        )
    return "\n".join(lines)
