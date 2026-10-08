"""The maker-cost estimator's arithmetic and its SQL shape.

Not its data: every assertion here runs without duckdb. What is worth pinning is
the sign convention, because it is the whole difference between the two readings
of this archive. Measured against the *fill price* a maker fill looks favourable
by +0.5 bp; measured against the mid at the moment of the fill it is adverse by
-0.5 bp. The first number is the captured half-spread and the second is adverse
selection, and adding them to the same cost model would charge the spread twice.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import maker_economics as me  # noqa: E402


def rows(buy: float, sell: float, n: int = 10, w: int = 30) -> list[dict]:
    return [
        {"side": "buy", "n": n, f"mo{w}": buy},
        {"side": "sell", "n": n, f"mo{w}": sell},
    ]


class TestDriftNeutral(unittest.TestCase):
    """(buy + sell) / 2 has to cancel the drift and keep the selection."""

    def test_pure_drift_cancels_to_zero(self):
        """Buy +1.0, sell -1.0 is the market moving, nothing else."""
        self.assertAlmostEqual(me.drift_neutral(rows(1.0, -1.0), (30,))[30], 0.0)

    def test_selection_survives_the_drift(self):
        """Drift +1.0 with -0.5 of selection: buy 0.5, sell -1.5."""
        self.assertAlmostEqual(me.drift_neutral(rows(0.5, -1.5), (30,))[30], -0.5)

    def test_both_sides_adverse_is_not_double_counted(self):
        """Selection is per leg. Two legs of -0.5 read -0.5, not -1.0."""
        self.assertAlmostEqual(me.drift_neutral(rows(-0.5, -0.5), (30,))[30], -0.5)

    def test_a_missing_side_yields_nothing_rather_than_half_an_answer(self):
        self.assertEqual(me.drift_neutral([{"side": "buy", "n": 1, "mo30": 1.0}], (30,)), {})

    def test_every_window_is_returned(self):
        r = [
            {"side": "buy", "n": 1, "mo1": 1.0, "mo30": 2.0},
            {"side": "sell", "n": 1, "mo1": 3.0, "mo30": 4.0},
        ]
        self.assertEqual(me.drift_neutral(r, (1, 30)), {1: 2.0, 30: 3.0})


class TestNetMakerCost(unittest.TestCase):
    def test_reprices_the_26_lower_bound(self):
        """1.82 - 0.099 was 1.7 with selection uncharged; -0.5 makes it 2.22."""
        self.assertAlmostEqual(me.net_maker_cost_bp(0.0), 1.82 - 0.099)
        self.assertAlmostEqual(me.net_maker_cost_bp(-0.5), 2.221, places=3)

    def test_the_rebate_enters_once(self):
        """It is already inside the -0.099, and more selection must cost more."""
        self.assertAlmostEqual(
            me.net_maker_cost_bp(-0.5) - me.net_maker_cost_bp(0.0), 0.5
        )


class TestMedian(unittest.TestCase):
    def test_odd_and_even(self):
        self.assertEqual(me.median([3.0, 1.0, 2.0]), 2.0)
        self.assertAlmostEqual(me.median([4.0, 1.0, 2.0, 3.0]), 2.5)

    def test_empty(self):
        self.assertTrue(me.median([]) != me.median([]))  # NaN


class TestSqlShape(unittest.TestCase):
    def test_mid_is_built_from_both_sides_of_the_book(self):
        sql = me.mid_cte("2026-03-10", 5)
        self.assertIn("NOT crossed", sql)
        self.assertIn("ASOF JOIN", sql)
        self.assertIn("bb.bid + x.ask", sql)
        self.assertIn("y.bid + aa.ask", sql)
        self.assertIn("date=2026-03-10.parquet", sql)

    def test_the_lag_cap_is_applied_to_both_pairings(self):
        sql = me.mid_cte("2026-03-10", 1)
        self.assertEqual(sql.count("<= 1"), 2)

    def test_adverse_marks_out_against_mid_not_against_the_fill_price(self):
        """`m0` is the mid at the fill; the fill price itself is never the base."""
        sql = me.adverse_sql("2026-03-10", (1, 30))
        self.assertIn("j1", sql)
        self.assertIn("j30", sql)
        self.assertIn("(j30.pn - fills.m0) / fills.m0", sql)
        self.assertNotIn("fills.px", sql)

    def test_buy_and_sell_get_opposite_signs(self):
        sql = me.adverse_sql("2026-03-10", (30,))
        self.assertIn("WHEN fills.side = 'buy' THEN 1 ELSE -1 END", sql)

    def test_by_coin_groups_on_coin_as_well(self):
        self.assertIn("GROUP BY 1, 2", me.adverse_sql("2026-03-10", (30,), by_coin=True))
        self.assertIn("GROUP BY 1 ", me.adverse_sql("2026-03-10", (30,)))

    def test_fill_rate_grid_uses_the_actual_second_range(self):
        """The archive's seconds are absolute epoch, not 0..86399.

        A hard-coded 0..86399 silently produces an empty grid and a fill rate
        computed over nothing, which looks like a result.
        """
        sql = me.fillrate_sql("2026-03-10", (60,), (1.0,))
        self.assertIn("(SELECT min(s) FROM midts)", sql)
        self.assertIn("(SELECT max(s) FROM midts)", sql)
        self.assertNotIn("86399", sql)

    def test_delta_becomes_a_column_suffix(self):
        sql = me.fillrate_sql("2026-03-10", (60,), (0.5, 10.0))
        self.assertIn("b60_0p5", sql)
        self.assertIn("b60_10p0", sql)


class TestSignTest(unittest.TestCase):
    def test_six_of_six_is_the_floor_for_six_windows(self):
        """0.0312 is as good as six windows can get; it is not a strong result
        on its own, which is why the magnitudes matter alongside it."""
        self.assertAlmostEqual(me.sign_test(6, 6), 0.03125, places=5)
        self.assertAlmostEqual(me.sign_test(0, 6), 0.03125, places=5)


if __name__ == "__main__":
    unittest.main()
