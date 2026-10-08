"""篮子核心的测试。

重点不是「代码能跑」，而是两处口径一致，因为这两处一旦漂了，实测和回测的差异
就无法归因：

  1. 实盘的选币就是回测的选币（`rank_legs` 是同一段代码）。
  2. 实盘按 `day_volume` 分档就是回测按 4h 美元成交量中位数分档（Step 0 量的）。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.basket import (  # noqa: E402
    BasketConfig,
    build_basket,
    build_basket_live,
    illiquid_half,
    live_panel,
)
from agent.market_data import AssetContext  # noqa: E402
from backtest.cross_section import (  # noqa: E402
    CrossSectionBacktester,
    CrossSectionConfig,
    Panel,
    negate,
    past_return,
    rank_legs,
)


# ---------------------------------------------------------------------------
# 合成数据
# ---------------------------------------------------------------------------
def make_panel(prices: dict[str, list[float]], volumes: float = 1.0) -> Panel:
    """`prices` 的每个序列长度必须相同。"""
    n = len(next(iter(prices.values())))
    times = [i * 4 * 3600 * 1000 for i in range(n)]
    return Panel(
        times=times,
        symbols=sorted(prices),
        close={s: [float(x) for x in series] for s, series in prices.items()},
        volume={s: [volumes] * n for s in prices},
    )


def flat_then_move(start: float, end: float, n: int = 8) -> list[float]:
    """前 n-1 根不动，最后一根跳到 `end` — 让信号只看得到这一段。"""
    return [start] * (n - 1) + [end]


class _FakeSeries:
    def __init__(self, closes: list[float], times: list[int]):
        self.closes = list(closes)
        self.volumes = [1.0] * len(closes)
        self.times = list(times)


class FakeMarket:
    """只有篮子在用的三个方法。"""

    def __init__(self, day_vols: dict[str, float], candles: dict[str, _FakeSeries]):
        self._ctxs = [
            AssetContext(index=i, name=s, day_volume=v)
            for i, (s, v) in enumerate(day_vols.items())
        ]
        self._candles = candles

    def asset_contexts(self):
        return list(self._ctxs)

    def candles(self, symbol, interval="4h", lookback=8):
        if symbol not in self._candles:
            raise RuntimeError(f"boom {symbol}")
        return self._candles[symbol]


# ---------------------------------------------------------------------------
class RankLegsSharedTest(unittest.TestCase):
    """实盘与回测必须是同一段选币代码。"""

    def test_backtester_uses_the_shared_function(self):
        """`_rank` 只是转发到 `rank_legs`，没有第二份实现。"""
        import inspect

        src = inspect.getsource(CrossSectionBacktester._rank)
        self.assertIn("rank_legs", src)
        # 转发不该再出现排序/切片的痕迹，否则又是一份实现。
        for marker in ("scored.sort", "scored[:k]", "scored[-k:]"):
            self.assertNotIn(marker, src)

    def test_rank_legs_matches_backtester_on_same_panel(self):
        prices = {}
        n = 10
        for i in range(30):
            # 一半涨、一半跌，幅度随编号递增，保证排序稳定无并列。
            drift = 1.0 + (0.05 + i * 0.01) * (1 if i % 2 else -1)
            prices[f"C{i:02d}"] = [100.0 * x * (drift ** (x / n)) for x in range(n)]
        panel = make_panel(prices)
        signal = negate(past_return(6))

        cfg = CrossSectionConfig(quantile=0.2, min_symbols=20)
        bt = CrossSectionBacktester(panel, signal, hold_rows=6, config=cfg)
        # row 必须 >= lookback，否则 `past_return` 返回 None、无人可排。
        row = 7
        got = rank_legs(panel, row, signal, cfg.quantile, cfg.min_symbols)
        self.assertIsNotNone(got)
        self.assertEqual(bt._rank(row), got)


class BuildBasketTest(unittest.TestCase):
    def test_longs_are_the_losers(self):
        """反转语义：做多的是过去跌得最多的那一批，不是涨得最多的。"""
        prices = {
            "LOSER1": flat_then_move(100.0, 90.0),   # -10%
            "LOSER2": flat_then_move(100.0, 92.0),   # -8%
            "WINNER1": flat_then_move(100.0, 110.0),  # +10%
            "WINNER2": flat_then_move(100.0, 108.0),  # +8%
        }
        cfg = BasketConfig(lookback=6, quantile=0.5, min_symbols=4)
        basket = build_basket(make_panel(prices), cfg)
        self.assertIsNotNone(basket)
        self.assertEqual(basket.longs, {"LOSER1", "LOSER2"})
        self.assertEqual(basket.shorts, {"WINNER1", "WINNER2"})

    def test_returns_none_when_too_few_rankable_symbols(self):
        prices = {"A": flat_then_move(100.0, 90.0), "B": flat_then_move(100.0, 110.0)}
        cfg = BasketConfig(lookback=6, quantile=0.5, min_symbols=20)
        self.assertIsNone(build_basket(make_panel(prices), cfg))

    def test_prices_captured_at_the_entry_row(self):
        prices = {
            "L1": flat_then_move(100.0, 90.0),
            "L2": flat_then_move(100.0, 92.0),
            "W1": flat_then_move(100.0, 110.0),
            "W2": flat_then_move(100.0, 108.0),
        }
        cfg = BasketConfig(lookback=6, quantile=0.5, min_symbols=4)
        basket = build_basket(make_panel(prices), cfg)
        self.assertEqual(basket.prices["L1"], 90.0)
        self.assertEqual(basket.prices["W1"], 110.0)

    def test_ts_is_the_last_row(self):
        prices = {
            "L1": flat_then_move(100.0, 90.0),
            "L2": flat_then_move(100.0, 92.0),
            "W1": flat_then_move(100.0, 110.0),
            "W2": flat_then_move(100.0, 108.0),
        }
        panel = make_panel(prices)
        basket = build_basket(panel, BasketConfig(lookback=6, quantile=0.5, min_symbols=4))
        self.assertEqual(basket.ts_ms, panel.times[-1])


class IlliquidHalfTest(unittest.TestCase):
    def test_matches_liquid_split_cut(self):
        """分档边界必须与 `liquid_split` 逐字一致，否则两边不是同一个池。"""
        from backtest.cross_section import liquid_split

        # 12 个币，成交额拉开档次，中位数落在中间。
        vols = {f"S{i:02d}": 10.0 * (2 ** i) for i in range(12)}
        market = FakeMarket(vols, {})
        got = illiquid_half(market, 0.5)

        # 用同一批数构造 panel 交给回测的分档函数。
        names = sorted(vols)
        panel = make_panel({s: [1.0] * 4 for s in names})
        panel.volume = {s: [vols[s]] * 4 for s in names}
        _, illiq = liquid_split(panel, 0.5)
        self.assertEqual(got, illiq)

    def test_empty_when_no_volume(self):
        self.assertEqual(illiquid_half(FakeMarket({}, {}), 0.5), set())

    def test_ignores_zero_volume(self):
        """没有成交额的币不该进入池 —— 它既排不出流动性也下不了单。"""
        market = FakeMarket({"A": 10.0, "B": 20.0, "DEAD": 0.0}, {})
        self.assertEqual(illiquid_half(market, 0.5), {"A"})


class LivePanelTest(unittest.TestCase):
    def _market(self, n_sym: int = 4, broken: set[str] | None = None):
        broken = broken or set()
        times = [i * 4 * 3600 * 1000 for i in range(8)]
        candles = {
            f"S{i}": _FakeSeries([100.0] * 7 + [100.0 + i], times)
            for i in range(n_sym)
            if f"S{i}" not in broken
        }
        return FakeMarket({f"S{i}": 100.0 + i for i in range(n_sym)}, candles)

    def test_assembles_aligned_panel(self):
        panel = live_panel(self._market(), ["S0", "S1", "S2", "S3"], rows=8)
        self.assertEqual(len(panel), 8)
        self.assertEqual(len(panel.symbols), 4)
        # 每个币在最后一根都有价，否则信号算不出来。
        self.assertEqual(len(panel.present(7)), 4)

    def test_skips_symbol_that_fails(self):
        """单币拉不到不该带走整个池。"""
        skipped = []
        panel = live_panel(
            self._market(broken={"S1"}),
            ["S0", "S1", "S2", "S3"],
            rows=8,
            on_skip=lambda s, why: skipped.append(s),
        )
        self.assertEqual(skipped, ["S1"])
        self.assertEqual(panel.symbols, ["S0", "S2", "S3"])

    def test_raises_when_nothing_returned(self):
        with self.assertRaises(ValueError):
            live_panel(FakeMarket({}, {}), ["NOPE"], rows=8)


class BuildBasketLiveTest(unittest.TestCase):
    def test_end_to_end_on_fake_market(self):
        """选池 -> 拉线 -> 排名，全过程在假数据上跑通。

        池只取非流动半，所以成交额低的四个币进池，池里刻意涨跌混合
        （两个跌、两个涨），这样两条腿都不是空的。
        """
        times = [i * 4 * 3600 * 1000 for i in range(8)]
        moves = {0: 90.0, 1: 110.0, 2: 95.0, 3: 105.0}  # 跌 / 涨 / 跌 / 涨
        closes = {
            f"S{i}": _FakeSeries([100.0] * 7 + [moves.get(i, 100.0)], times)
            for i in range(8)
        }
        # 前四个成交额最低 -> 它们才是池；后四个被分档挡在外面。
        vols = {0: 10.0, 1: 11.0, 2: 12.0, 3: 13.0, 4: 9e3, 5: 9e3, 6: 9e3, 7: 9e3}
        market = FakeMarket({f"S{i}": v for i, v in vols.items()}, closes)

        cfg = BasketConfig(lookback=6, quantile=0.5, min_symbols=4)
        basket, pool_size = build_basket_live(market, cfg)
        self.assertEqual(pool_size, 4)
        self.assertIsNotNone(basket)
        # 反转：做多跌的、做空涨的。
        self.assertEqual(basket.longs, {"S0", "S2"})
        self.assertEqual(basket.shorts, {"S1", "S3"})
        self.assertEqual(basket.n_legs, 4)


if __name__ == "__main__":
    unittest.main()
