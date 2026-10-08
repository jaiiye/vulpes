"""篮子持久化的测试。

核心场景是重启：34 笔持仓必须活过进程重启，否则下一期会把它们当成新仓再开一遍，
敞口翻倍而日志上看起来只是「正常开仓」。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.basket_state import (  # noqa: E402
    BasketState,
    BasketStore,
    state_from_positions,
)
from agent.execution import Position  # noqa: E402
from agent.synthesizer import LONG, SHORT  # noqa: E402


def pos(symbol: str, side: str = LONG, price: float = 100.0) -> Position:
    return Position(
        symbol=symbol,
        side=side,
        size=20.0,
        entry_price=price,
        notional=2000.0,
        leverage=1,
    )


class RoundTripTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.store = BasketStore(self.dir / "basket_state.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_then_load(self):
        positions = {"L1": pos("L1", LONG), "S1": pos("S1", SHORT)}
        state = state_from_positions(positions, basket_ts_ms=1234)
        self.assertTrue(self.store.save(state))

        loaded = self.store.load()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.n_positions, 2)
        self.assertEqual(loaded.basket_ts_ms, 1234)
        self.assertEqual(loaded.longs, ["L1"])
        self.assertEqual(loaded.shorts, ["S1"])

    def test_restart_does_not_lose_positions(self):
        """重启场景：持仓必须原样回来，否则会重复开仓。"""
        positions = {f"L{i}": pos(f"L{i}", LONG) for i in range(3)}
        positions.update({f"S{i}": pos(f"S{i}", SHORT) for i in range(3)})
        self.store.save(state_from_positions(positions, basket_ts_ms=99))

        # 模拟重启：全新 store 指向同一个文件。
        restarted = BasketStore(self.dir / "basket_state.json")
        loaded = restarted.load()
        restored = loaded.positions_as_objects()
        self.assertEqual(set(restored), set(positions))
        self.assertEqual(sum(1 for p in restored.values() if p.side == LONG), 3)
        self.assertEqual(sum(1 for p in restored.values() if p.side == SHORT), 3)
        # 价格要回来，否则下一期无法算规模。
        self.assertEqual(restored["L1"].entry_price, 100.0)

    def test_missing_file_is_none(self):
        self.assertIsNone(self.store.load())

    def test_corrupt_json_is_none(self):
        self.store.path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(self.store.load())

    def test_version_mismatch_is_none(self):
        self.store.path.write_text(
            json.dumps({"version": 999, "positions": {}}), encoding="utf-8"
        )
        self.assertIsNone(self.store.load())

    def test_clear(self):
        self.store.save(state_from_positions({"L1": pos("L1")}, 1))
        self.store.clear()
        self.assertIsNone(self.store.load())


class CorruptionTest(unittest.TestCase):
    """手改过的文件必须退化成可用状态，而不是让启动崩掉。"""

    def test_bad_single_record_is_skipped(self):
        """一笔坏数据不该让整个篮子起不来。"""
        state = BasketState(positions={"L1": pos("L1").to_dict(), "BAD": "oops"})
        restored = state.positions_as_objects()
        self.assertEqual(set(restored), {"L1"})

    def test_non_dict_positions_coerced(self):
        state = BasketState()
        state.positions = "nonsense"  # 模拟写坏的文件
        store = BasketStore(Path("/tmp/definitely-not-used-basket-state.json"))
        # 直接测 load 的强制逻辑：构造一个 payload 走一遍。
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.json"
            p.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "positions": "nonsense",
                        "longs": "nope",
                        "shorts": None,
                    }
                ),
                encoding="utf-8",
            )
            loaded = BasketStore(p).load()
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.positions, {})
            self.assertEqual(loaded.longs, [])
            self.assertEqual(loaded.shorts, [])


if __name__ == "__main__":
    unittest.main()
