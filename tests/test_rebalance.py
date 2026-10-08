"""再平衡的测试。

重点是三条风险选择，不是「函数能跑」：先平后开、单笔失败不终止整轮、以及最后
把两腿削平。第三条尤其容易被漏 —— 不削的话剩下来的是净方向敞口，而市场方向是
这个策略没有 edge 的维度。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.basket import Basket, BasketConfig  # noqa: E402
from agent.execution import Position  # noqa: E402
from agent.rebalance import (  # noqa: E402
    RebalancePlan,
    execute_rebalance,
    leg_notional,
    plan_rebalance,
)
from agent.synthesizer import LONG, SHORT  # noqa: E402


def pos(symbol: str, side: str, price: float = 100.0, notional: float = 2000.0) -> Position:
    return Position(
        symbol=symbol,
        side=side,
        size=notional / price,
        entry_price=price,
        notional=notional,
        leverage=1,
    )


def basket(longs, shorts, prices=None, ts: int = 0) -> Basket:
    names = set(longs) | set(shorts)
    return Basket(
        longs=set(longs),
        shorts=set(shorts),
        ts_ms=ts,
        prices=prices or {s: 100.0 for s in names},
    )


class FakeBroker:
    """记录调用顺序，并可对指定 symbol 注入失败。"""

    def __init__(self, fail_opens=(), fail_closes=()):
        self.fail_opens = set(fail_opens)
        self.fail_closes = set(fail_closes)
        self.calls: list[tuple[str, str]] = []

    def open_position(self, symbol, side, size, price, leverage=1, **kw):
        self.calls.append(("open", symbol))
        if symbol in self.fail_opens:
            raise RuntimeError(f"rejected {symbol}")
        return Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=price,
            notional=size * price,
            leverage=leverage,
        )

    def close_position(self, position, price, reason):
        self.calls.append(("close", position.symbol))
        if position.symbol in self.fail_closes:
            raise RuntimeError(f"cannot close {position.symbol}")
        return None


# ---------------------------------------------------------------------------
class PlanTest(unittest.TestCase):
    def test_flat_to_full_basket(self):
        """空仓 -> 全部开仓，无平仓。"""
        cfg = BasketConfig(quantile=0.5, min_symbols=4)
        target = basket(["L1", "L2"], ["S1", "S2"])
        plan = plan_rebalance({}, target, cfg=cfg)
        self.assertEqual(len(plan.opens), 4)
        self.assertEqual(plan.closes, [])
        self.assertEqual(plan.holds, [])
        self.assertEqual(plan.turnover, 1.0)

    def test_unchanged_names_are_held_not_reopened(self):
        """方向没变的继续持有 —— 这正是换手率要省下的成本。"""
        cur = {"L1": pos("L1", LONG), "S1": pos("S1", SHORT)}
        target = basket(["L1", "L2"], ["S1", "S2"])
        plan = plan_rebalance(cur, target, cfg=BasketConfig())
        self.assertEqual({o.symbol for o in plan.opens}, {"L2", "S2"})
        self.assertEqual([p.symbol for p in plan.holds], ["L1", "S1"])
        self.assertEqual(plan.closes, [])
        # 4 个里有 2 个要开 -> 换手 50%
        self.assertAlmostEqual(plan.turnover, 0.5)

    def test_reversed_side_is_closed_then_reopened(self):
        """方向反了：先平后开，不能同时持两边。"""
        cur = {"X": pos("X", LONG)}
        target = basket(["L1"], ["X"])
        plan = plan_rebalance(cur, target, cfg=BasketConfig())
        self.assertEqual([p.symbol for p in plan.closes], ["X"])
        self.assertEqual([(o.symbol, o.side) for o in plan.opens],
                         [("L1", LONG), ("X", SHORT)])

    def test_dropped_names_are_closed(self):
        cur = {"OLD": pos("OLD", LONG), "L1": pos("L1", LONG)}
        target = basket(["L1", "L2"], ["S1", "S2"])
        plan = plan_rebalance(cur, target, cfg=BasketConfig())
        self.assertEqual([p.symbol for p in plan.closes], ["OLD"])

    def test_symbol_without_price_is_skipped(self):
        """拿不到价格就不下单 —— 宁可少一个币，也不按猜测的价格下错单。"""
        target = basket(["NOPRICE", "L1"], ["S1", "S2"])
        target.prices.pop("NOPRICE")
        plan = plan_rebalance({}, target, cfg=BasketConfig())
        self.assertNotIn("NOPRICE", {o.symbol for o in plan.opens})


class ParticipationTest(unittest.TestCase):
    def test_capped_by_daily_volume(self):
        cfg = BasketConfig(per_coin_usd=2000.0, max_participation=0.05)
        # 日成交额 $20k -> 上限 $1k，所以被削。
        notional, capped = leg_notional(100.0, 20_000.0, cfg)
        self.assertTrue(capped)
        self.assertAlmostEqual(notional, 1_000.0)

    def test_not_capped_when_volume_is_large(self):
        cfg = BasketConfig(per_coin_usd=2000.0, max_participation=0.05)
        notional, capped = leg_notional(100.0, 1_000_000.0, cfg)
        self.assertFalse(capped)
        self.assertAlmostEqual(notional, 2000.0)

    def test_missing_volume_does_not_cap(self):
        """没有成交额数据时不该缩到 0 —— 那会让整个篮子下不了单。"""
        cfg = BasketConfig(per_coin_usd=2000.0, max_participation=0.05)
        notional, capped = leg_notional(100.0, 0.0, cfg)
        self.assertFalse(capped)
        self.assertAlmostEqual(notional, 2000.0)

    def test_capped_order_is_flagged_in_plan(self):
        cfg = BasketConfig(per_coin_usd=2000.0, max_participation=0.05)
        target = basket(["THIN", "FAT"], ["S1", "S2"])
        plan = plan_rebalance(
            {}, target, day_volumes={"THIN": 20_000.0, "FAT": 1e9}, cfg=cfg
        )
        by_sym = {o.symbol: o for o in plan.opens}
        self.assertTrue(by_sym["THIN"].capped)
        self.assertFalse(by_sym["FAT"].capped)
        self.assertIn("capped", plan.describe())


class ExecuteTest(unittest.TestCase):
    def test_all_succeed(self):
        broker = FakeBroker()
        target = basket(["L1", "L2"], ["S1", "S2"])
        plan = plan_rebalance({}, target, cfg=BasketConfig())
        out = execute_rebalance(broker, plan)
        self.assertEqual(out.n_long, 2)
        self.assertEqual(out.n_short, 2)
        self.assertEqual(out.imbalance, 0)
        self.assertEqual(out.failed_opens, [])

    def test_closes_run_before_opens(self):
        """先平后开：平仓必须全部排在开仓之前。"""
        broker = FakeBroker()
        cur = {"OLD": pos("OLD", LONG)}
        target = basket(["L1", "L2"], ["S1", "S2"])
        plan = plan_rebalance(cur, target, cfg=BasketConfig())
        execute_rebalance(broker, plan)
        kinds = [k for k, _ in broker.calls]
        self.assertEqual(kinds.index("close"), 0)
        self.assertLess(kinds.index("close"), len(kinds) - 1)
        self.assertTrue(all(k == "close" for k in kinds[:1]))

    def test_single_open_failure_does_not_abort_the_round(self):
        """34 笔里挂几笔不该让这一期完全没有篮子。

        注意最终是 2 笔而不是 3 笔：L2 挂掉后是 1 long / 2 short，削平又去掉
        一个 short。这是两个机制叠加的正确结果，不是丢单。
        """
        broker = FakeBroker(fail_opens={"L2"})
        target = basket(["L1", "L2"], ["S1", "S2"])
        plan = plan_rebalance({}, target, cfg=BasketConfig())
        out = execute_rebalance(broker, plan)
        self.assertEqual(len(out.failed_opens), 1)
        self.assertEqual(out.failed_opens[0][0], "L2")
        # 其余三笔都尝试过了 —— 失败没有中断整轮。
        self.assertEqual(sum(1 for k, _ in broker.calls if k == "open"), 4)
        # 这一期仍然有篮子，而且是被削平过的。
        self.assertTrue(out.positions)
        self.assertEqual(out.imbalance, 0)

    def test_failed_close_keeps_the_position(self):
        """没平掉的敞口必须留在 positions 里，否则会漏掉它。"""
        broker = FakeBroker(fail_closes={"OLD"})
        cur = {"OLD": pos("OLD", LONG)}
        target = basket(["L1", "L2"], ["S1", "S2"])
        plan = plan_rebalance(cur, target, cfg=BasketConfig())
        out = execute_rebalance(broker, plan)
        self.assertEqual(len(out.failed_closes), 1)
        self.assertIn("OLD", out.positions)


class BalanceTest(unittest.TestCase):
    def test_imbalance_is_trimmed(self):
        """核心：腿不平衡必须被削平，不能带着净方向敞口过夜。"""
        broker = FakeBroker(fail_opens={"S2"})
        target = basket(["L1", "L2", "L3"], ["S1", "S2"])
        plan = plan_rebalance({}, target, cfg=BasketConfig())
        out = execute_rebalance(broker, plan)
        self.assertEqual(out.imbalance, 0)
        # 3 long 对 1 short -> 削掉 2 个 long。
        self.assertEqual(len(out.truncated), 2)
        self.assertEqual(out.n_long, 1)
        self.assertEqual(out.n_short, 1)

    def test_trims_the_most_recently_opened_first(self):
        """削最后开成的：先开成的那批是这一期信号最强的。"""
        broker = FakeBroker(fail_opens={"S2"})
        target = basket(["L1", "L2", "L3"], ["S1", "S2"])
        plan = plan_rebalance({}, target, cfg=BasketConfig())
        out = execute_rebalance(broker, plan)
        # opens 按 symbol 排序：L1, L2, L3, S1, S2。倒序削 long -> L3 先，再 L2。
        self.assertEqual(out.truncated, ["L3", "L2"])
        self.assertIn("L1", out.positions)

    def test_balanced_round_is_left_alone(self):
        broker = FakeBroker()
        target = basket(["L1", "L2"], ["S1", "S2"])
        plan = plan_rebalance({}, target, cfg=BasketConfig())
        out = execute_rebalance(broker, plan)
        self.assertEqual(out.truncated, [])
        self.assertEqual(len(out.positions), 4)

    def test_failed_trim_is_reported(self):
        """削平本身也可能失败，这时必须说出来，不能假装削过了。"""
        broker = FakeBroker(fail_opens={"S2"}, fail_closes={"L3", "L2"})
        target = basket(["L1", "L2", "L3"], ["S1", "S2"])
        plan = plan_rebalance({}, target, cfg=BasketConfig())
        events = []
        out = execute_rebalance(
            broker, plan, on_event=lambda k, m: events.append((k, m))
        )
        self.assertTrue(any(k == "truncate_failed" for k, _ in events))
        # 削不掉就是削不掉，imbalance 必须如实反映。
        self.assertNotEqual(out.imbalance, 0)


if __name__ == "__main__":
    unittest.main()
