"""篮子级风控的测试。

这里管的是组合层面的数，不是单笔止损 —— 篮子没有单币止损，因为每条腿的盈亏是
靠另一条腿对冲掉的，给单个币挂止损会把对冲结构拆掉。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.basket import Basket, BasketConfig  # noqa: E402
from agent.basket_risk import check_plan, plan_gross_notional  # noqa: E402
from agent.execution import Position  # noqa: E402
from agent.rebalance import plan_rebalance  # noqa: E402
from agent.synthesizer import LONG, SHORT  # noqa: E402


def target(longs, shorts, prices=None) -> Basket:
    names = set(longs) | set(shorts)
    return Basket(
        longs=set(longs),
        shorts=set(shorts),
        ts_ms=0,
        prices=prices or {s: 100.0 for s in names},
    )


def held(symbol: str, side: str, notional: float = 2000.0) -> Position:
    return Position(
        symbol=symbol,
        side=side,
        size=notional / 100.0,
        entry_price=100.0,
        notional=notional,
        leverage=1,
    )


class EquityTest(unittest.TestCase):
    def test_zero_equity_refuses(self):
        """拿不到权益就不要建仓 —— 按猜测的权益算出来的规模不是规模。"""
        plan = plan_rebalance({}, target(["L1", "L2"], ["S1", "S2"]), cfg=BasketConfig())
        self.assertIsNotNone(check_plan(plan, 0.0))
        self.assertIsNotNone(check_plan(plan, -1.0))


class GrossNotionalTest(unittest.TestCase):
    def test_absolute_cap(self):
        cfg = BasketConfig(per_coin_usd=2000.0, max_gross_notional=50_000.0)
        plan = plan_rebalance(
            {}, target([f"L{i}" for i in range(4)], [f"S{i}" for i in range(4)]), cfg=cfg
        )
        # 8 个币 x $2k
        self.assertAlmostEqual(plan_gross_notional(plan), 16_000.0)
        self.assertIsNone(check_plan(plan, 1e6, cfg))

        cfg2 = BasketConfig(per_coin_usd=2000.0, max_gross_notional=10_000.0)
        self.assertIsNotNone(check_plan(plan, 1e6, cfg2))

    def test_pct_of_equity_cap(self):
        cfg = BasketConfig(per_coin_usd=2000.0, max_gross_pct_of_equity=10.0)
        plan = plan_rebalance(
            {}, target([f"L{i}" for i in range(4)], [f"S{i}" for i in range(4)]), cfg=cfg
        )
        # 毛敞口 $16k：权益 $1M 时占 1.6% 通过，权益 $100k 时占 16% 拒绝。
        self.assertIsNone(check_plan(plan, 1_000_000.0, cfg))
        self.assertIsNotNone(check_plan(plan, 100_000.0, cfg))

    def test_holds_count_toward_gross(self):
        """持有不动的也是敞口，不能因为「这期没开仓」就不算。"""
        cur = {"L1": held("L1", LONG), "S1": held("S1", SHORT)}
        plan = plan_rebalance(cur, target(["L1", "L2"], ["S1", "S2"]), cfg=BasketConfig())
        # 2 个持有 + 2 个新开 = $8k
        self.assertAlmostEqual(plan_gross_notional(plan), 8_000.0)
        cfg = BasketConfig(max_gross_notional=4_000.0)
        self.assertIsNotNone(check_plan(plan, 1e6, cfg))


class LegCountTest(unittest.TestCase):
    def test_legs_per_side_cap(self):
        cfg = BasketConfig(max_legs_per_side=2)
        plan = plan_rebalance(
            {}, target([f"L{i}" for i in range(4)], [f"S{i}" for i in range(4)]), cfg=cfg
        )
        self.assertIsNotNone(check_plan(plan, 1e6, cfg))

        cfg_ok = BasketConfig(max_legs_per_side=5)
        self.assertIsNone(check_plan(plan, 1e6, cfg_ok))

    def test_holds_count_toward_legs(self):
        cur = {"L1": held("L1", LONG), "L2": held("L2", LONG)}
        plan = plan_rebalance(cur, target(["L1", "L2"], ["S1", "S2"]), cfg=BasketConfig())
        cfg = BasketConfig(max_legs_per_side=1)
        self.assertIsNotNone(check_plan(plan, 1e6, cfg))


class MinNotionalTest(unittest.TestCase):
    def test_tiny_leg_is_refused(self):
        """低于 venue 最小 size 的单不值得占一个腿的名额。"""
        cfg = BasketConfig(per_coin_usd=2000.0, min_leg_notional=1_500.0)
        # 日成交额极小 -> 被参与率削到 $1k，低于 min_leg_notional。
        plan = plan_rebalance(
            {},
            target(["THIN"], ["FAT"]),
            day_volumes={"THIN": 20_000.0, "FAT": 1e9},
            cfg=cfg,
        )
        self.assertIsNotNone(check_plan(plan, 1e6, cfg))

    def test_normal_size_passes(self):
        cfg = BasketConfig(per_coin_usd=2000.0, min_leg_notional=10.0)
        plan = plan_rebalance(
            {}, target(["L1", "L2"], ["S1", "S2"]), cfg=cfg
        )
        self.assertIsNone(check_plan(plan, 1e6, cfg))


class EmptyPlanTest(unittest.TestCase):
    def test_nothing_to_trade(self):
        self.assertIsNotNone(check_plan(type("P", (), {
            "opens": [], "holds": [],
        })(), 1e6))


if __name__ == "__main__":
    unittest.main()
