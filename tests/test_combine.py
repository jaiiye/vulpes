"""Tests for the combination helpers.

The module exists to make one failure mode checkable, so the tests are mostly
about that failure mode: a blend whose full-sample number is the average of a
zero half and a strong half. The arithmetic helpers are pinned against
hand-computed values rather than stored ones.
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import combine as cb  # noqa: E402


class TestCorrelation(unittest.TestCase):
    def test_a_series_is_perfectly_correlated_with_itself(self):
        r = [0.01, -0.02, 0.03, 0.005, -0.01] * 20
        self.assertAlmostEqual(cb.correlation(r, r), 1.0, places=12)

    def test_a_negated_series_is_perfectly_anti_correlated(self):
        r = [0.01, -0.02, 0.03, 0.005, -0.01] * 20
        self.assertAlmostEqual(cb.correlation(r, [-x for x in r]), -1.0, places=12)

    def test_independent_series_are_near_zero(self):
        rng = random.Random(3)
        a = [rng.gauss(0, 0.01) for _ in range(2000)]
        b = [rng.gauss(0, 0.01) for _ in range(2000)]
        self.assertLess(abs(cb.correlation(a, b)), 0.1)

    def test_a_flat_series_yields_zero_not_an_error(self):
        """0.0 is the conservative reading for blending: it claims the
        diversification the data does not contradict. Raising would block a
        legitimate case (one leg that never traded)."""
        self.assertEqual(cb.correlation([0.0] * 50, [0.01, -0.01] * 25), 0.0)

    def test_misaligned_lengths_are_rejected(self):
        with self.assertRaises(cb.CombineError):
            cb.correlation([0.01] * 10, [0.01] * 9)


class TestSharpe(unittest.TestCase):
    def test_it_is_mean_over_stdev_scaled_by_root_n(self):
        r = [0.01, -0.005, 0.02, -0.01, 0.005] * 40
        expected = (sum(r) / len(r)) / math.sqrt(
            sum((x - sum(r) / len(r)) ** 2 for x in r) / len(r)
        ) * math.sqrt(len(r))
        self.assertAlmostEqual(cb.sharpe(r), expected, places=12)

    def test_a_flat_series_has_zero_sharpe(self):
        self.assertEqual(cb.sharpe([0.0] * 100), 0.0)

    def test_more_dispersion_at_the_same_mean_lowers_it(self):
        # Both series must have a non-zero mean: with a mean of exactly zero
        # every Sharpe is 0, so an earlier version of this test compared 0 to 0
        # and asserted one was greater.
        tight = [0.002, 0.000] * 100          # mean 0.001, stdev 0.001
        wide = [0.050, -0.048] * 100          # mean 0.001, stdev 0.049
        self.assertGreater(cb.sharpe(tight), cb.sharpe(wide))

    def test_too_few_periods_are_rejected(self):
        with self.assertRaises(cb.CombineError):
            cb.sharpe([0.01])


class TestRiskParityWeights(unittest.TestCase):
    def test_the_quieter_series_gets_the_larger_weight(self):
        quiet = [0.001, -0.001] * 100
        loud = [0.05, -0.05] * 100
        w = cb.risk_parity_weights([quiet, loud])
        self.assertGreater(w[0], w[1])
        self.assertAlmostEqual(sum(w), 1.0, places=12)

    def test_weights_are_inverse_volatility(self):
        quiet = [0.001, -0.001] * 100        # stdev 0.001
        loud = [0.004, -0.004] * 100         # stdev 0.004
        w = cb.risk_parity_weights([quiet, loud])
        self.assertAlmostEqual(w[0], 0.8, places=9)
        self.assertAlmostEqual(w[1], 0.2, places=9)

    def test_a_zero_volatility_component_is_refused(self):
        """Inverse volatility is infinite there, so any weight returned would be
        a choice the data did not make."""
        with self.assertRaises(cb.CombineError):
            cb.risk_parity_weights([[0.0] * 100, [0.01, -0.01] * 50])

    def test_equal_volatility_gives_equal_weights(self):
        # Both +-0.01, so both stdev 0.01. An earlier version paired +-0.01
        # with +-0.02 and called it "equal volatility" - the module returned
        # 0.667, correctly, and the test read that as the bug.
        a = [0.01, -0.01] * 100
        b = [0.01, -0.01] * 100
        w = cb.risk_parity_weights([a, b])
        self.assertAlmostEqual(w[0], 0.5, places=9)

    def test_double_the_volatility_takes_a_third_of_the_risk_budget(self):
        """Inverse volatility, not inverse variance: twice the stdev is half
        the weight, which is 1/3 of the total rather than 1/4."""
        a = [0.01, -0.01] * 100
        b = [0.02, -0.02] * 100
        w = cb.risk_parity_weights([a, b])
        self.assertAlmostEqual(w[0], 2 / 3, places=9)
        self.assertAlmostEqual(w[1], 1 / 3, places=9)


class TestBlend(unittest.TestCase):
    def test_it_is_a_weighted_sum_per_period(self):
        a = [0.01, 0.02, -0.01]
        b = [0.0, -0.01, 0.03]
        for got, want in zip(cb.blend([a, b], [0.5, 0.5]), [0.005, 0.005, 0.01]):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(cb.blend([a, b], [1.0, 0.0]), a)
        self.assertEqual(cb.blend([a, b], [0.0, 1.0]), b)

    def test_weights_must_sum_to_one(self):
        """Normalising silently would hide a caller's mistake about what they
        passed - a blend that is not fully invested is a different object."""
        with self.assertRaises(cb.CombineError):
            cb.blend([[0.01] * 10, [0.01] * 10], [0.5, 0.4])

    def test_the_wrong_number_of_weights_is_rejected(self):
        with self.assertRaises(cb.CombineError):
            cb.blend([[0.01] * 10, [0.01] * 10], [1.0])


class TestWeightSweep(unittest.TestCase):
    def test_it_finds_the_better_component(self):
        """`a` is a clean positive drift, `b` is the same drift drowned in
        noise, so the sweep should put nearly everything on `a`."""
        rng = random.Random(11)
        a = [0.001 + rng.gauss(0, 0.0005) for _ in range(400)]
        b = [0.001 + rng.gauss(0, 0.05) for _ in range(400)]
        w, s = cb.weight_sweep(a, b, steps=20)
        self.assertGreater(w, 0.9)
        self.assertGreater(s, cb.sharpe(b))

    def test_a_bad_step_count_is_rejected(self):
        with self.assertRaises(cb.CombineError):
            cb.weight_sweep([0.01] * 10, [0.01] * 10, steps=0)


class TestSignTest(unittest.TestCase):
    def test_all_windows_succeeding_is_the_smallest_tail(self):
        # 6 of 6: both tails are the same single outcome, p = 2/64.
        self.assertAlmostEqual(cb.sign_test(6, 6), 0.03125, places=9)

    def test_five_of_six(self):
        self.assertAlmostEqual(cb.sign_test(5, 6), 0.21875, places=9)

    def test_half_the_windows_is_not_evidence(self):
        """The bug this guards: doubling only the upper tail for k <= n/2
        returns more than 1.0. An earlier version reported P = 1.31 for 3 of 6,
        which is not a probability."""
        for k, n in ((3, 6), (2, 4), (1, 2)):
            with self.subTest(k=k, n=n):
                self.assertAlmostEqual(cb.sign_test(k, n), 1.0, places=9)

    def test_zero_successes_is_significant_not_certain(self):
        """The other half of the same bug, and the case an earlier version of
        this test got wrong: it listed (0, 5) among the "no evidence" cases.
        0 of 5 is a 1-in-32 outcome, so P = 0.0625, and a shortcut that only
        doubles the upper tail reports 1.0 there."""
        self.assertAlmostEqual(cb.sign_test(0, 5), 0.0625, places=9)
        self.assertAlmostEqual(cb.sign_test(0, 6), 0.03125, places=9)

    def test_it_is_symmetric(self):
        self.assertAlmostEqual(cb.sign_test(1, 6), cb.sign_test(5, 6), places=12)

    def test_impossible_inputs_are_rejected(self):
        for k, n in ((-1, 6), (7, 6), (1, 0), (1, -3)):
            with self.subTest(k=k, n=n):
                with self.assertRaises(cb.CombineError):
                    cb.sign_test(k, n)


class TestEqualWindows(unittest.TestCase):
    def test_windows_tile_the_range_without_overlapping(self):
        w = cb.equal_windows(600, 6)
        self.assertEqual(w[0], (0, 100))
        self.assertEqual(w[-1], (500, 600))
        for (a1, b1), (a2, b2) in zip(w, w[1:]):
            self.assertEqual(b1, a2)

    def test_all_windows_are_the_same_length(self):
        """Unequal windows would make a per-window comparison partly a
        comparison of window lengths."""
        sizes = {b - a for a, b in cb.equal_windows(617, 6)}
        self.assertEqual(len(sizes), 1)

    def test_a_remainder_is_dropped_not_distributed(self):
        w = cb.equal_windows(617, 6)
        self.assertEqual(len(w), 6)
        self.assertEqual(w[-1][1], 6 * (617 // 6))
        self.assertLess(w[-1][1], 617)

    def test_bad_arguments_are_rejected(self):
        for n, c in ((5, 6), (10, 0), (10, -1)):
            with self.subTest(n=n, count=c):
                with self.assertRaises(cb.CombineError):
                    cb.equal_windows(n, c)


class TestReplicates(unittest.TestCase):
    def test_it_reports_the_count_and_its_p_value_together(self):
        out = cb.replicates([0.46, 0.38, 0.48, 0.40, 0.59, 0.47], 0.25)
        self.assertEqual(out["count"], 6)
        self.assertEqual(out["trials"], 6)
        self.assertAlmostEqual(out["p_value"], 0.03125, places=9)
        self.assertAlmostEqual(out["median"], 0.465, places=9)
        self.assertEqual(out["range"], (0.38, 0.59))

    def test_a_measure_that_does_not_replicate(self):
        out = cb.replicates([0.3, 0.2, 0.4, 0.1, 0.5, 0.2], 0.25)
        self.assertEqual(out["count"], 3)
        self.assertAlmostEqual(out["p_value"], 1.0, places=9)

    def test_threshold_is_strict(self):
        """Boundary values do not count: "above 25%" must exclude exactly 25%,
        or a null result would report as a success."""
        out = cb.replicates([0.25] * 6, 0.25)
        self.assertEqual(out["count"], 0)


class TestHalvesAndStability(unittest.TestCase):
    def test_halves_are_contiguous_not_interleaved(self):
        """Contiguity is the point: a result carried by one stretch has to show
        up as one bad half, and interleaving would average it away."""
        r = list(range(10))
        first, second = cb.halves(r)
        self.assertEqual(first, [0, 1, 2, 3, 4])
        self.assertEqual(second, [5, 6, 7, 8, 9])

    def test_stability_reports_the_halves_alongside_the_full_sample(self):
        """The headline numbers must not be obtainable without the halves,
        because both times a headline failed in this repo it was quoted first."""
        rng = random.Random(21)
        a = [0.002 + rng.gauss(0, 0.005) for _ in range(200)]
        b = [0.001 + rng.gauss(0, 0.004) for _ in range(200)]
        out = cb.stability(a, b)
        for key in ("correlation", "sharpe_a", "sharpe_b", "sharpe_blend",
                    "risk_parity_weight_a", "halves_sharpe_a", "halves_sharpe_b",
                    "halves_sharpe_blend"):
            self.assertIn(key, out)
        self.assertEqual(len(out["halves_sharpe_blend"]), 2)

    def test_stability_exposes_a_result_carried_by_one_half(self):
        """The failure mode, constructed: the first half is noise and the second
        is a strong drift, so the full-sample Sharpe is a blend of the two and
        neither half resembles it."""
        rng = random.Random(31)
        n = 200
        signal = [rng.gauss(0, 0.005) for _ in range(n // 2)] + \
                 [0.004 + rng.gauss(0, 0.005) for _ in range(n // 2)]
        other = [rng.gauss(0, 0.005) for _ in range(n)]
        out = cb.stability(signal, other)
        first, second = out["halves_sharpe_a"]
        self.assertLess(abs(first), 0.6)
        self.assertGreater(second, 2.0)
        # The full-sample number sits between them, describing neither.
        self.assertGreater(out["sharpe_a"], first)
        self.assertLess(out["sharpe_a"], second)


if __name__ == "__main__":
    unittest.main()
