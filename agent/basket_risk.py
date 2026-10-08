"""篮子级风控。

单币 bot 的风控是「这一笔亏多少」（止损、ATR、风险预算）。篮子的风控不一样，
因为篮子**没有单币止损** —— 每条腿的盈亏是靠另一条腿对冲掉的，给单个币挂止损
会把对冲结构拆掉。所以这里管的是组合层面的四个数：

  1. 毛敞口相对权益的比例（一次建仓不能把账户用光）
  2. 毛敞口的绝对上限
  3. 每腿的最大数量
  4. 单笔的最小名义（低于 venue 的最小 size 就别下）

还有一个不是限额而是前提：**拿不到权益就不要建仓**。按猜测的权益算出来的规模
不是规模。
"""

from __future__ import annotations

from agent.basket import BasketConfig
from agent.rebalance import RebalancePlan
from agent.synthesizer import LONG


def plan_gross_notional(plan: RebalancePlan) -> float:
    """建仓后的毛名义（新开的 + 持有不动的）。"""
    return sum(o.notional for o in plan.opens) + sum(
        p.notional for p in plan.holds
    )


def check_plan(
    plan: RebalancePlan, equity: float, cfg: BasketConfig | None = None
) -> str | None:
    """返回拒绝原因，None 表示通过。

    调用方必须看这个返回值：一期的篮子是 34 笔，建错一次的代价不是一笔止损，
    而是整个组合的结构。
    """
    cfg = cfg or BasketConfig()

    # 前提，不是限额：权益读不到时任何规模都是猜的。
    if equity is None or equity <= 0:
        return "equity unavailable; refusing to size a basket against nothing"

    gross = plan_gross_notional(plan)
    if gross <= 0:
        return "nothing to trade"

    if cfg.max_gross_notional > 0 and gross > cfg.max_gross_notional:
        return (
            f"gross ${gross:,.0f} exceeds cap ${cfg.max_gross_notional:,.0f}"
        )

    if cfg.max_gross_pct_of_equity > 0:
        cap = equity * cfg.max_gross_pct_of_equity / 100.0
        if gross > cap:
            return (
                f"gross ${gross:,.0f} exceeds "
                f"{cfg.max_gross_pct_of_equity:g}% of equity (${cap:,.0f})"
            )

    n_long = sum(1 for o in plan.opens if o.side == LONG) + sum(
        1 for p in plan.holds if p.side == LONG
    )
    n_short = (len(plan.opens) + len(plan.holds)) - n_long
    if cfg.max_legs_per_side > 0 and max(n_long, n_short) > cfg.max_legs_per_side:
        return (
            f"leg of {max(n_long, n_short)} exceeds "
            f"max {cfg.max_legs_per_side} per side"
        )

    too_small = [o.symbol for o in plan.opens if o.notional < cfg.min_leg_notional]
    if too_small:
        return (
            f"{len(too_small)} leg(s) below ${cfg.min_leg_notional:g}: "
            f"{', '.join(sorted(too_small)[:5])}"
        )

    return None
