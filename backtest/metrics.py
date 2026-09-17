"""Performance statistics for a backtest result.

Deliberately reports sample-size-sensitive numbers alongside the headline, and
refuses to call a result meaningful when the trade count is too small to
support it. A 3-trade backtest with a 100% win rate is not evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .engine import BacktestResult

# Below this many trades, distributional statistics are noise.
MIN_MEANINGFUL_TRADES = 30


@dataclass
class Metrics:
    initial_equity: float
    final_equity: float
    total_return_pct: float
    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    avg_win: float
    avg_loss: float
    expectancy: float
    expectancy_r: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe: float
    best_trade: float
    worst_trade: float
    avg_holding_hours: float
    gross_pnl: float
    fees_paid: float
    funding_paid: float
    exposure_pct: float
    is_meaningful: bool

    def report(self) -> str:
        lines = [
            f"  equity        ${self.initial_equity:,.2f} -> ${self.final_equity:,.2f} "
            f"({self.total_return_pct:+.2f}%)",
            f"  trades        {self.trades} ({self.wins}W / {self.losses}L, "
            f"win rate {self.win_rate_pct:.1f}%)",
            f"  avg win/loss  ${self.avg_win:+,.2f} / ${self.avg_loss:+,.2f}",
            f"  expectancy    ${self.expectancy:+.2f}/trade  "
            f"({self.expectancy_r:+.3f}R)",
            f"  profit factor {self.profit_factor:.2f}",
            f"  max drawdown  {self.max_drawdown_pct:.2f}%",
            f"  sharpe (ann)  {self.sharpe:.2f}",
            f"  best / worst  ${self.best_trade:+,.2f} / ${self.worst_trade:+,.2f}",
            f"  avg hold      {self.avg_holding_hours:.1f}h"
            f"   exposure {self.exposure_pct:.0f}%",
            f"  gross ${self.gross_pnl:+,.2f}  fees ${self.fees_paid:+,.2f}  "
            f"funding ${self.funding_paid:+,.2f}   (costs are positive)",
        ]
        if not self.is_meaningful:
            lines.append(
                f"  !! only {self.trades} trades; below {MIN_MEANINGFUL_TRADES} the "
                "distributional stats are not statistically meaningful"
            )
        return "\n".join(lines)


def _max_drawdown_pct(curve: list[tuple[int, float]]) -> float:
    peak = -math.inf
    worst = 0.0
    for _, equity in curve:
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak * 100.0
            worst = max(worst, dd)
    return worst


def _sharpe(curve: list[tuple[int, float]], bars_per_year: float) -> float:
    """Annualised Sharpe of per-bar equity returns, risk-free rate 0."""
    if len(curve) < 3:
        return 0.0
    returns: list[float] = []
    for i in range(1, len(curve)):
        prev = curve[i - 1][1]
        if prev <= 0:
            continue
        returns.append((curve[i][1] - prev) / prev)
    if len(returns) < 2:
        return 0.0

    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return mean / std * math.sqrt(bars_per_year)


def compute_metrics(
    result: BacktestResult, bar_interval_hours: float = 1.0
) -> Metrics:
    trades = result.trades
    n = len(trades)

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl <= 0]

    gross_profit = sum(t.net_pnl for t in wins)
    gross_loss = -sum(t.net_pnl for t in losses)

    avg_win = gross_profit / len(wins) if wins else 0.0
    avg_loss = -gross_loss / len(losses) if losses else 0.0

    expectancy = sum(t.net_pnl for t in trades) / n if n else 0.0
    expectancy_r = sum(t.r_multiple for t in trades) / n if n else 0.0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = math.inf if gross_profit > 0 else 0.0

    total_return = (
        (result.final_equity - result.initial_equity) / result.initial_equity * 100.0
        if result.initial_equity > 0
        else 0.0
    )

    bars_per_year = 365 * 24 / max(bar_interval_hours, 1e-9)
    total_bars = max(len(result.equity_curve), 1)
    holding_bars = sum(t.holding_hours for t in trades) / max(bar_interval_hours, 1e-9)
    exposure = min(100.0, holding_bars / total_bars * 100.0)

    return Metrics(
        initial_equity=result.initial_equity,
        final_equity=result.final_equity,
        total_return_pct=total_return,
        trades=n,
        wins=len(wins),
        losses=len(losses),
        win_rate_pct=len(wins) / n * 100.0 if n else 0.0,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        expectancy_r=expectancy_r,
        profit_factor=profit_factor,
        max_drawdown_pct=_max_drawdown_pct(result.equity_curve),
        sharpe=_sharpe(result.equity_curve, bars_per_year),
        best_trade=max((t.net_pnl for t in trades), default=0.0),
        worst_trade=min((t.net_pnl for t in trades), default=0.0),
        avg_holding_hours=(
            sum(t.holding_hours for t in trades) / n if n else 0.0
        ),
        gross_pnl=sum(t.gross_pnl for t in trades),
        fees_paid=result.fees_paid,
        funding_paid=result.funding_paid,
        exposure_pct=exposure,
        is_meaningful=n >= MIN_MEANINGFUL_TRADES,
    )
