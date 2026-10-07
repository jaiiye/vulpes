"""The cost ladder's arithmetic - interpolated crossings and aggregation.

Not its data: `measure` is exercised through a stub simulator so the two
aggregation conventions can be pinned down with three symbols instead of the
archive.

`TestAggregation` exists because the repository uses both conventions and they
do not agree. RESEARCH.md 24 reports a break-even of 2.5-4 bp from its
cross-window table and a "median net return" of -0.39% at 7.2 bp from its
pooled one; at the same fee the pooled median of that section's own numbers
implies 5.46 bp. A factor of two in the number that decides whether the edge is
tradeable is not a detail to leave to whichever line of code ran last, so the
difference is asserted here directly.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cost_sensitivity as cs  # noqa: E402


class FakeGateResult:
    """Just enough of `GateResult` for `measure` to consume."""

    def __init__(self, value: float):
        self.net_return_pct = value
        self.trades = ["one trade"]


def index_of(spec: dict[str, float], bars: int = 10) -> dict:
    """`{symbol: {bar: row}}` shaped like `scan_exits`'s index.

    Every bar of a symbol carries the same close, which the stubbed simulator
    reads back as that symbol's return - so the spec is the answer.
    """
    return {s: {i: {"c": v, "o": v, "h": v, "l": v} for i in range(bars)}
            for s, v in spec.items()}


def stub(closes, opens, highs, lows, cfg):
    return FakeGateResult(closes[0])


class TestCrossing(unittest.TestCase):
    def test_interpolates_between_the_bracketing_points(self):
        rows = [{"fee": 1.0, "x": 1.0}, {"fee": 3.0, "x": -1.0}]
        self.assertAlmostEqual(cs.crossing(rows, "x"), 2.0)

    def test_a_point_sitting_exactly_on_zero_is_the_answer(self):
        """Both ends, because `(0 < 0)` is False and a zero is not a sign change."""
        self.assertEqual(cs.crossing([{"fee": 1.0, "x": 1.0},
                                      {"fee": 2.0, "x": 0.0}], "x"), 2.0)
        self.assertEqual(cs.crossing([{"fee": 1.0, "x": 0.0},
                                      {"fee": 2.0, "x": -1.0}], "x"), 1.0)

    def test_no_bracket_means_no_answer(self):
        rows = [{"fee": 1.0, "x": 1.0}, {"fee": 2.0, "x": 0.5}]
        self.assertIsNone(cs.crossing(rows, "x"))

    def test_a_single_point_cannot_cross(self):
        self.assertIsNone(cs.crossing([{"fee": 1.0, "x": -1.0}], "x"))

    def test_the_first_crossing_wins(self):
        rows = [{"fee": 0.0, "x": -1.0}, {"fee": 1.0, "x": 1.0},
                {"fee": 2.0, "x": -1.0}]
        self.assertAlmostEqual(cs.crossing(rows, "x"), 0.5)


class TestAggregation(unittest.TestCase):
    def test_cross_window_weights_windows_equally(self):
        """One +10 symbol in one window against three -1 symbols in another.

        Pooled, the three small symbols outvote the large one; cross-window,
        each window gets one vote. That is the whole difference between the
        two break-even figures.
        """
        index = index_of({"A": 10.0, "B": -1.0, "C": -1.0, "D": -1.0})
        pools = [(0, 10, ["A"]), (0, 10, ["B", "C", "D"])]

        with patch.object(cs, "simulate_mean_reversion", stub):
            r = cs.measure(pools, index, 7.2, 50.0)

        self.assertEqual(r["median"], -1.0)      # median of the four symbols
        self.assertEqual(r["cw_median"], 4.5)    # median of [10, -1]
        self.assertEqual(r["n"], 4)
        self.assertEqual(r["trades"], 4)
        self.assertEqual(r["positive"], 0.25)

    def test_a_symbol_short_of_coverage_is_skipped(self):
        index = {"A": {i: {"c": 1.0, "o": 1.0, "h": 1.0, "l": 1.0}
                       for i in range(5)}}
        with patch.object(cs, "simulate_mean_reversion", stub):
            r = cs.measure([(0, 10, ["A"])], index, 7.2, 50.0)
        self.assertEqual(r["n"], 0)
        self.assertEqual(r["trades"], 0)

    def test_a_symbol_with_no_trades_is_dropped(self):
        """Zero trades is not a zero return - it is no observation."""
        class NoTrades:
            net_return_pct = 0.0
            trades = []

        index = index_of({"A": 1.0, "B": 2.0})
        with patch.object(cs, "simulate_mean_reversion",
                          lambda *a, **k: NoTrades()):
            r = cs.measure([(0, 10, ["A", "B"])], index, 7.2, 50.0)
        self.assertEqual(r["n"], 0)


class TestPoolSelection(unittest.TestCase):
    def test_an_unknown_pool_is_rejected_before_any_work(self):
        with self.assertRaises(ValueError):
            cs.collect({}, {}, 0, 100, 1, "quintiles")

    def test_the_two_pools_are_the_ones_documented(self):
        self.assertEqual(cs.POOLS, ("q14", "half"))


if __name__ == "__main__":
    unittest.main()
