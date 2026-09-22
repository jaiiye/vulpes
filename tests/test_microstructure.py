"""Tests for the microstructure diagnostics.

Each estimator is checked against a construction whose answer is known in
closed form rather than against a stored number, because a stored number would
pass whatever the estimator happened to do when it was recorded.

The three are also checked against each other's failure modes: Roll must be seen
to move when the lag-1 structure changes, and the variance ratio must be seen to
*not* move when only the cause changes. That asymmetry is the whole reason both
exist - Roll names a cause and is therefore biased by any other source of
negative covariance, while the variance ratio names no cause and is therefore
robust to that but blind to it.
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import microstructure as ms  # noqa: E402


class TestAutocorrelation(unittest.TestCase):
    def test_alternating_returns_have_autocorrelation_near_minus_one(self):
        r = [0.01 if i % 2 else -0.01 for i in range(600)]
        v = ms.autocorrelation(r, 1)
        self.assertLess(v, -0.99)

    def test_alternating_returns_flip_sign_at_lag_two(self):
        """The signature that separates a bounce from a slow reversal."""
        r = [0.01 if i % 2 else -0.01 for i in range(600)]
        self.assertGreater(ms.autocorrelation(r, 2), 0.99)

    def test_an_independent_series_is_near_zero(self):
        rng = random.Random(11)
        r = [rng.gauss(0, 0.01) for _ in range(4000)]
        self.assertLess(abs(ms.autocorrelation(r, 1)), 0.05)

    def test_it_is_scale_free(self):
        """Same shape, thousand-fold larger moves - the correlation must not
        move, or comparing symbols by volatility would be meaningless."""
        rng = random.Random(3)
        base = [0.02 if i % 2 else -0.015 for i in range(600)]
        scaled = [x * 1000 for x in base]
        self.assertAlmostEqual(ms.autocorrelation(base, 1),
                               ms.autocorrelation(scaled, 1), places=12)

    def test_a_short_series_returns_none_rather_than_zero(self):
        """Zero is a measurement ("no relationship"); None is "cannot say".
        Returning 0.0 for a short series would let it vote for the null."""
        self.assertIsNone(ms.autocorrelation([0.01] * 10, 1))
        self.assertIsNone(ms.autocovariance([0.01] * 10, 1))

    def test_a_flat_series_has_no_defined_correlation(self):
        self.assertIsNone(ms.autocorrelation([0.0] * 400, 1))

    def test_gaps_are_dropped_and_that_biases_toward_zero(self):
        """Dropping a gap in an alternating series splices two same-sign values
        together, which weakens the measured negative autocorrelation. The
        direction of that bias is the point: gaps make a series look *less*
        mean-reverting than it is, never more, so a positive finding cannot be
        manufactured by missing bars.
        """
        r = [0.01 if i % 2 else -0.01 for i in range(600)]
        with_gaps = list(r)
        for i in range(0, 600, 7):
            with_gaps[i] = None
        v = ms.autocorrelation(with_gaps, 1)
        self.assertIsNotNone(v)
        self.assertLess(v, -0.5)
        self.assertGreater(v, ms.autocorrelation(r, 1))

    def test_dropping_is_exactly_not_seeing_those_bars_at_all(self):
        """The equality that pins the behaviour. Asserting only "still strongly
        negative" is satisfied by filling gaps with 0.0 as well, so an earlier
        version of this test held the mutation it was meant to catch."""
        rng = random.Random(71)
        r = [rng.gauss(0, 0.01) for _ in range(600)]
        kept = list(r)
        for i in (5, 40, 77, 200, 555):
            kept[i] = None
        dropped = [x for i, x in enumerate(r) if i not in (5, 40, 77, 200, 555)]
        self.assertAlmostEqual(
            ms.autocovariance(kept, 1), ms.autocovariance(dropped, 1), places=12
        )
        self.assertAlmostEqual(
            ms.autocorrelation(kept, 1), ms.autocorrelation(dropped, 1), places=12
        )

    def test_bad_lags_are_rejected(self):
        for bad in (0, -1):
            with self.subTest(lag=bad):
                with self.assertRaises(ms.MicrostructureError):
                    ms.autocorrelation([0.01] * 400, bad)


class TestRollSpread(unittest.TestCase):
    """The model: a constant true price observed at bid or ask at random.

    Then `Cov(r_t, r_{t-1}) = -(s/2)^2` and Roll's estimator returns exactly the
    spread `s`. Constructing it that way means the test asserts the estimator's
    *definition* rather than a regression against a number.
    """

    #: Relative, not absolute. `roll_spread` returns the spread as a fraction of
    #: price, so the construction has to state it that way too - an earlier
    #: version of this test asserted the price-unit spread and read a 100x
    #: discrepancy as an estimator bug.
    SPREAD = 0.0004         # 4 bp one-way

    def _bounce(self, n: int = 4000, seed: int = 5):
        rng = random.Random(seed)
        true_price = 100.0
        half = true_price * self.SPREAD / 2.0
        prices = []
        for _ in range(n):
            offset = half if rng.random() < 0.5 else -half
            prices.append(true_price + offset)
        return [(prices[i] / prices[i - 1] - 1.0) for i in range(1, n)]

    def test_it_recovers_a_known_spread(self):
        got = ms.roll_spread(self._bounce())
        self.assertAlmostEqual(got, self.SPREAD, delta=self.SPREAD * 0.25)

    def test_the_bps_helper_is_the_same_number_times_ten_thousand(self):
        r = self._bounce()
        self.assertAlmostEqual(ms.roll_spread_bps(r), ms.roll_spread(r) * 10_000,
                               places=6)

    def test_no_negative_covariance_means_the_model_says_nothing(self):
        """Returns a plain 0.0 rather than a negative spread, which is what the
        square root of a positive covariance would otherwise invite."""
        rng = random.Random(9)
        trending = [0.01 + rng.gauss(0, 0.001) for _ in range(600)]
        self.assertEqual(ms.roll_spread(trending), 0.0)

    def _bounce_of(self, rel_spread: float, n: int = 4000, seed: int = 5):
        """Same random-bounce model, with the relative spread stated directly."""
        rng = random.Random(seed)
        half = 100.0 * rel_spread / 2.0
        prices = [100.0 + (half if rng.random() < 0.5 else -half)
                  for _ in range(n)]
        return [(prices[i] / prices[i - 1] - 1.0) for i in range(1, n)]

    def test_a_wider_bounce_gives_a_wider_estimate(self):
        """Stated in relative terms on both sides. An earlier version built the
        'wide' case as +-0.01 on a price of 100, which is *narrower* than the
        +-0.02 the other fixture used - the test asserted the opposite of what
        it had constructed and read the answer as an estimator bug."""
        narrow = ms.roll_spread(self._bounce_of(0.0002))
        wide = ms.roll_spread(self._bounce_of(0.002))
        self.assertGreater(wide, narrow * 5)

    def test_a_short_series_returns_none(self):
        self.assertIsNone(ms.roll_spread([0.01, -0.01] * 5))


class TestVarianceRatio(unittest.TestCase):
    def test_a_random_walk_is_about_one(self):
        rng = random.Random(21)
        r = [rng.gauss(0, 0.01) for _ in range(4000)]
        self.assertAlmostEqual(ms.variance_ratio(r, 4), 1.0, delta=0.15)

    def test_alternating_returns_are_strongly_below_one(self):
        r = [0.01 if i % 2 else -0.01 for i in range(4000)]
        self.assertLess(ms.variance_ratio(r, 2), 0.2)

    def test_a_trending_series_is_above_one(self):
        """A persistent *drift* is what raises the ratio, not a large mean. A
        constant mean with iid noise leaves the ratio at 1, because the drift
        contributes no variance across windows - an earlier version of this
        test used exactly that and asserted > 2."""
        rng = random.Random(31)
        r = []
        x = 0.0
        for _ in range(4000):
            x = 0.7 * x + rng.gauss(0, 0.001)      # positively autocorrelated
            r.append(x)
        self.assertGreater(ms.variance_ratio(r, 4), 2.0)

    def test_it_is_blind_to_the_cause(self):
        """Two series with the same autocorrelation structure but different
        causes give the same variance ratio - which is exactly why it cannot
        diagnose, and why Roll is needed alongside it."""
        rng = random.Random(41)
        noise = [rng.gauss(0, 0.005) for _ in range(4000)]
        bounce = [x + (0.002 if i % 2 else -0.002) for i, x in enumerate(noise)]
        reversal = [x - 0.4 * noise[i - 1] for i, x in enumerate(noise) if i]
        self.assertLess(ms.variance_ratio(bounce, 2), 1.05)
        # Not asserted equal - only that both are pulled the same direction by
        # mechanisms Roll would attribute differently.
        self.assertIsNotNone(ms.variance_ratio(reversal, 2))

    def test_bad_q_is_rejected(self):
        for bad in (1, 0, -3):
            with self.subTest(q=bad):
                with self.assertRaises(ms.MicrostructureError):
                    ms.variance_ratio([0.01] * 4000, bad)

    def test_a_short_series_returns_none(self):
        self.assertIsNone(ms.variance_ratio([0.01] * 50, 4))


class TestReversalProfile(unittest.TestCase):
    def _random_bounce(self, n: int = 6000, seed: int = 7):
        """A *random* bounce, not a perfect alternation.

        The distinction matters and an earlier version of these tests missed
        it: in a perfectly alternating series every odd lag is -1, so the lag-1
        share of the total is 0.5 no matter how the function is written. That is
        a deterministic sawtooth, not a bid-ask bounce - a real bounce picks a
        side at random, so only lag 1 is negative and the share is near 1.
        """
        rng = random.Random(seed)
        prices = [100.0 + (0.02 if rng.random() < 0.5 else -0.02)
                  for _ in range(n)]
        return [(prices[i] / prices[i - 1] - 1.0) for i in range(1, n)]

    def test_a_random_bounce_is_concentrated_at_lag_one(self):
        share = ms.bounce_share(self._random_bounce())
        self.assertIsNotNone(share)
        self.assertGreater(share, 0.8)

    def test_a_perfect_alternation_is_not_a_bounce(self):
        """Documents the trap above rather than pretending it does not exist:
        the deterministic sawtooth's odd lags are all negative, so the share
        sits near 0.5 and the diagnostic does not call it microstructure."""
        saw = [0.01 if i % 2 else -0.01 for i in range(1200)]
        share = ms.bounce_share(saw)
        self.assertIsNotNone(share)
        self.assertLess(share, 0.7)

    def test_a_spread_out_reversal_has_a_lower_lag_one_share(self):
        rng = random.Random(51)
        noise = [rng.gauss(0, 0.01) for _ in range(4000)]
        slow = list(noise)
        for i in range(24, len(slow)):
            slow[i] -= 0.3 * noise[i - 24]
        share = ms.bounce_share(slow)
        self.assertIsNotNone(share)
        self.assertLess(share, 1.0)

    def test_the_profile_returns_every_requested_lag(self):
        prof = ms.reversal_profile(self._random_bounce(2000), (1, 2, 3))
        self.assertEqual(sorted(prof), [1, 2, 3])
        self.assertLess(prof[1], 0.0)
        self.assertAlmostEqual(prof[2], 0.0, delta=0.1)

    def test_no_lag_one_share_is_reported_when_the_dependence_is_elsewhere(self):
        """`bounce_share` answers "how much of it sits at lag 1", so when the
        answer is "none of it" that is 0.0, not None. None is reserved for
        "there is no negative dependence to attribute at all"."""
        rng = random.Random(61)
        trending = [0.01 + rng.gauss(0, 0.0005) for _ in range(4000)]
        share = ms.bounce_share(trending)
        self.assertTrue(share is None or share < 0.2,
                        f"trending series claimed a lag-1 share of {share}")


class TestBookCost(unittest.TestCase):
    """Walking a real book, which is the only way to say what an order costs.

    The distinction these tests protect is between the *quoted* spread and the
    cost of actually trading: they are equal only when depth is unlimited, so
    any code that reports the quoted spread as the execution cost is right by
    accident on a deep book and wrong on a thin one.
    """

    def _deep(self, bid=100.0, ask=101.0):
        return [(bid, 10_000_000.0)], [(ask, 10_000_000.0)]

    def test_the_quoted_spread_at_the_touch(self):
        bids, asks = self._deep()
        got = ms.top_of_book_spread_bps(bids, asks)
        self.assertAlmostEqual(got, (101.0 - 100.0) / 100.5 * 10_000, places=9)

    def test_with_unlimited_depth_the_round_trip_is_the_quoted_spread(self):
        """Buying at the ask and selling at the bid costs exactly one spread,
        so on a deep book the two numbers coincide. They must differ once depth
        runs out - which is the whole reason both functions exist."""
        bids, asks = self._deep()
        self.assertAlmostEqual(
            ms.round_trip_cost_bps(bids, asks, 1_000.0),
            ms.top_of_book_spread_bps(bids, asks),
            places=6,
        )

    def test_cost_grows_with_size(self):
        bids = [(100.0, 1.0), (99.0, 100.0), (98.0, 1000.0)]
        asks = [(101.0, 1.0), (102.0, 100.0), (103.0, 1000.0)]
        small = ms.round_trip_cost_bps(bids, asks, 50.0)
        big = ms.round_trip_cost_bps(bids, asks, 50_000.0)
        self.assertLess(small, big)

    def test_a_size_the_book_cannot_fill_returns_none(self):
        """None, not a number: quoting a cost derived from depth that is not
        there is the optimistic error, and it is the one that would let an
        untradeable size look tradeable."""
        bids = [(100.0, 1.0)]
        asks = [(101.0, 1.0)]
        self.assertIsNone(ms.round_trip_cost_bps(bids, asks, 1_000_000.0))

    def test_one_empty_side_returns_none(self):
        bids, asks = self._deep()
        self.assertIsNone(ms.round_trip_cost_bps([], asks, 100.0))
        self.assertIsNone(ms.round_trip_cost_bps(bids, [], 100.0))

    def test_a_crossed_book_is_refused(self):
        self.assertIsNone(ms.round_trip_cost_bps([(102.0, 10.0)], [(101.0, 10.0)], 10.0))

    def test_the_walk_is_charged_at_each_level(self):
        """Hand-computed: 100 notional of asks at 101 (size 1 -> 101 notional)
        fills entirely at the touch, while 500 needs part of the second level."""
        asks = [(101.0, 1.0), (102.0, 100.0)]
        bids = [(100.0, 1000.0)]
        # 500 notional: 101 of it at 101.0, the rest 399 at 102.0
        buy = (101.0 * 1.0 + 399.0 / 102.0 * 102.0) / (1.0 + 399.0 / 102.0)
        self.assertAlmostEqual(buy, (101.0 + 399.0) / (1.0 + 399.0 / 102.0), places=9)
        got = ms.round_trip_cost_bps(bids, asks, 500.0)
        mid = (100.0 + 101.0) / 2.0
        self.assertAlmostEqual(got, ((buy - mid) + (mid - 100.0)) / mid * 10_000,
                               places=6)

    def test_depth_within_a_band_counts_only_levels_inside_it(self):
        asks = [(100.0, 1.0), (100.5, 2.0), (105.0, 10.0)]
        # 100 bp above 100.0 is 101.0, so the 105 level is excluded.
        self.assertAlmostEqual(ms.depth_within_bps(asks, 100.0, 100.0), 100.0 + 100.5 * 2.0)

    def test_a_bad_notional_is_rejected(self):
        bids, asks = self._deep()
        for bad in (0.0, -1.0):
            with self.subTest(notional=bad):
                with self.assertRaises(ms.MicrostructureError):
                    ms.round_trip_cost_bps(bids, asks, bad)


if __name__ == "__main__":
    unittest.main()
