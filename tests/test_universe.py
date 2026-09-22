"""Tests for correlation-aware universe construction.

The module exists because three correlated majors were being counted as three
independent samples. Most of what can go wrong here is a *definition* going
wrong rather than arithmetic: the same panel yields 1.10 or 2.72 for "effective
sample" depending on which of two reciprocals is meant, and this project has
already drawn a wrong conclusion from an unlabelled statistic. So the tests
below pin the definitions, not just the numbers.
"""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import universe as U  # noqa: E402


class FakePanel:
    """Minimal stand-in: `universe` reads only times/symbols/price/volume."""

    def __init__(self, series: dict[str, list[float | None]]):
        self.symbols = sorted(series)
        self.times = list(range(len(next(iter(series.values())))))
        self._series = series
        self.volume: dict[str, list[float | None]] = {}
        self.close = {k: list(v) for k, v in series.items()}

    def price(self, symbol: str, i: int) -> float | None:
        v = self._series[symbol][i]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)


def ramp(values: list[float]) -> list[float]:
    return list(values)


class TestReturns(unittest.TestCase):
    def test_returns_are_the_row_to_row_change(self):
        panel = FakePanel({"A": [1.0, 2.0, 4.0]})
        r = U.returns_from(panel)["A"]
        self.assertIsNone(r[0])
        self.assertAlmostEqual(r[1], 1.0)
        self.assertAlmostEqual(r[2], 1.0)

    def test_a_gap_is_skipped_not_bridged(self):
        """A missing bar must not be carried across.

        Bridging would turn a halt into a zero return, and a run of zero
        returns drags every correlation computed from it toward zero exactly
        where the data is least trustworthy.
        """
        panel = FakePanel({"A": [1.0, None, 4.0, 5.0]})
        r = U.returns_from(panel)["A"]
        self.assertIsNone(r[1])
        self.assertIsNone(r[2])
        self.assertAlmostEqual(r[3], 0.25)

    def test_lookback_uses_the_row_lookback_back(self):
        panel = FakePanel({"A": [1.0, 2.0, 3.0, 9.0]})
        r = U.returns_from(panel, lookback=3)["A"]
        self.assertIsNone(r[2])
        self.assertAlmostEqual(r[3], 8.0)

    def test_bad_lookback_is_rejected(self):
        with self.assertRaises(U.UniverseError):
            U.returns_from(FakePanel({"A": [1.0]}), lookback=0)


class TestCorrelation(unittest.TestCase):
    def test_identical_series_correlate_at_one(self):
        xs = [float(i % 7) for i in range(60)]
        self.assertAlmostEqual(U.correlation_or_none(xs, xs), 1.0, places=9)

    def test_opposite_series_correlate_at_minus_one(self):
        xs = [float(i % 7) for i in range(60)]
        ys = [-v for v in xs]
        self.assertAlmostEqual(U.correlation_or_none(xs, ys), -1.0, places=9)

    def test_missing_rows_are_dropped_pairwise(self):
        """Not listwise: dropping a row because one series lacks it would throw
        away overlap the other pair can still use."""
        xs = [float(i % 7) if i % 3 else None for i in range(90)]
        ys = [float(i % 7) for i in range(90)]
        pairs = sum(1 for x, y in zip(xs, ys) if x is not None and y is not None)
        c = U.correlation_or_none(xs, ys, min_pairs=1)
        self.assertIsNotNone(c)
        self.assertAlmostEqual(c, 1.0, places=9)
        self.assertEqual(pairs, 60)

    def test_too_few_overlapping_rows_returns_none(self):
        xs = [1.0, 2.0, 3.0]
        ys = [3.0, 2.0, 1.0]
        self.assertIsNone(U.correlation_or_none(xs, ys, min_pairs=30))

    def test_a_flat_series_is_undefined_not_zero(self):
        """Returning 0.0 would read as "uncorrelated", i.e. as useful
        diversification, which is the opposite of what a halted price means."""
        xs = [1.0] * 60
        ys = [float(i) for i in range(60)]
        self.assertIsNone(U.correlation_or_none(xs, ys))

    def test_nan_is_treated_as_missing(self):
        xs = [float(i) for i in range(60)]
        ys = [float(i) if i != 5 else float("nan") for i in range(60)]
        c = U.correlation_or_none(xs, ys)
        self.assertIsNotNone(c)
        self.assertAlmostEqual(c, 1.0, places=9)


class TestPairwise(unittest.TestCase):
    def setUp(self):
        self.up = [1.0 + i * 0.01 for i in range(60)]
        self.down = [2.0 - i * 0.01 for i in range(60)]
        self.noise = [1.0 + ((i * 37) % 11) * 0.01 for i in range(60)]
        self.returns = {"A": self.up, "B": self.down, "C": self.noise}

    def test_keys_are_ordered_so_lookup_is_argument_order_independent(self):
        pairs = U.pairwise_correlations(self.returns)
        for key in pairs:
            self.assertLess(key[0], key[1])
        self.assertIn(("A", "B"), pairs)
        self.assertNotIn(("B", "A"), pairs)

    def test_every_pair_is_present(self):
        pairs = U.pairwise_correlations(self.returns)
        self.assertEqual(len(pairs), 3)

    def test_mean_pairwise_averages_the_pairs(self):
        pairs = U.pairwise_correlations(self.returns)
        expected = sum(pairs.values()) / len(pairs)
        self.assertAlmostEqual(
            U.mean_pairwise_correlation(self.returns), expected, places=12
        )

    def test_no_scoreable_pair_gives_none_not_zero(self):
        self.assertIsNone(U.mean_pairwise_correlation({"A": [1.0], "B": [2.0]}))


class TestSampleSizeDefinitions(unittest.TestCase):
    """The two corrections are reciprocals and must not be swapped.

    On three majors the average pair correlation is 0.859, so the gap is 2.7x;
    on forty liquid symbols it is 0.479, so the gap is 20x. Quoting the wrong
    one is the failure mode this class exists to prevent.
    """

    def test_the_two_are_reciprocal_up_to_k(self):
        for k in (2, 3, 40):
            for rho in (0.0, 0.25, 0.5, 0.859, 0.95):
                with self.subTest(k=k, rho=rho):
                    imp = U.sample_inflation(k, rho)
                    bets = U.effective_bets(k, rho)
                    self.assertAlmostEqual(bets * imp, float(k), places=9)

    def test_bets_is_k_when_uncorrelated(self):
        self.assertAlmostEqual(U.effective_bets(40, 0.0), 40.0)

    def test_bets_collapses_to_one_when_identical(self):
        self.assertAlmostEqual(U.effective_bets(40, 1.0), 1.0)

    def test_inflation_is_one_when_uncorrelated(self):
        self.assertAlmostEqual(U.sample_inflation(5, 0.0), 1.0)

    def test_effective_observations_divides_the_pooled_count(self):
        self.assertAlmostEqual(U.effective_observations(130, 3, 0.859), 130 / 2.72,
                               places=1)

    def test_a_single_symbol_is_untouched(self):
        self.assertAlmostEqual(U.sample_inflation(1, 0.9), 1.0)
        self.assertAlmostEqual(U.effective_bets(1, 0.9), 1.0)

    def test_out_of_range_inputs_are_rejected(self):
        with self.assertRaises(U.UniverseError):
            U.sample_inflation(3, 1.5)
        with self.assertRaises(U.UniverseError):
            U.effective_bets(0, 0.5)
        with self.assertRaises(U.UniverseError):
            U.effective_observations(-1, 3, 0.5)

    def test_strong_negative_correlation_does_not_produce_impossible_counts(self):
        """Below -1/(k-1) the denominator flips sign. Reporting a negative or
        infinite "number of bets" would be worse than reporting k."""
        self.assertLessEqual(U.effective_bets(4, -0.9), 4.0)
        self.assertGreater(U.effective_bets(4, -0.9), 0.0)


class TestClustering(unittest.TestCase):
    def _panel_returns(self, spec):
        """spec: {symbol: phase}; identical phase => identical series."""
        return {s: [math.sin((i + ph) * 0.3) for i in range(80)]
                for s, ph in spec.items()}

    def test_series_sharing_a_phase_land_in_one_group(self):
        rets = self._panel_returns({"A": 0.0, "B": 0.0, "C": 2.0})
        groups = U.cluster_by_correlation(rets, threshold=0.9)
        sizes = sorted(len(g) for g in groups)
        self.assertEqual(sizes, [1, 2])
        big = [g for g in groups if len(g) == 2][0]
        self.assertEqual(big, ["A", "B"])

    def test_grouping_is_transitive(self):
        """A~B and B~C must put all three together even if A-C is below the
        threshold; otherwise "one name per group" does not guarantee the
        picked names are pairwise below it."""
        rets = {
            "A": [float(i) for i in range(80)],
            "B": [float(i) + 0.1 * ((-1) ** i) for i in range(80)],
            # C tracks B closely but drifts away from A over the sample.
            "C": [float(i) + 0.1 * ((-1) ** i) + i * 0.5 for i in range(80)],
        }
        groups = U.cluster_by_correlation(rets, threshold=0.95)
        together = [g for g in groups if len(g) == 3]
        self.assertEqual(len(together), 1, f"groups={groups}")

    def test_symbols_scoring_against_nothing_are_kept_as_singletons(self):
        """They are the most diversifying names available, so dropping them
        would discard exactly what the caller wants."""
        good = [math.sin(i * 0.3) for i in range(80)]
        rets = {
            "A": good,
            "B": list(good),
            "Solo": [1.0 if i % 2 else -1.0 for i in range(80)],
        }
        groups = U.cluster_by_correlation(rets, threshold=0.99)
        singles = [g for g in groups if len(g) == 1]
        self.assertEqual(singles, [["Solo"]])

    def test_output_is_ordered_deterministically(self):
        rets = self._panel_returns({"A": 0.0, "B": 0.0, "C": 2.0, "D": 3.0})
        first = U.cluster_by_correlation(rets, threshold=0.9)
        second = U.cluster_by_correlation(
            dict(reversed(list(rets.items()))), threshold=0.9
        )
        self.assertEqual(first, second)
        # Largest group first.
        self.assertEqual([len(g) for g in first], sorted(
            (len(g) for g in first), reverse=True
        ))

    def test_threshold_out_of_range_is_rejected(self):
        with self.assertRaises(U.UniverseError):
            U.cluster_by_correlation({"A": [1.0] * 40}, threshold=2.0)


class TestSelection(unittest.TestCase):
    def setUp(self):
        base = [math.sin(i * 0.3) for i in range(80)]
        self.rets = {
            "BTC": list(base),
            "ETH": [v * 1.01 for v in base],           # same group as BTC
            "SOL": [v * 1.02 for v in base],           # same group as BTC
            "GOLD": [-v for v in base],                # own group
            "ODD": [1.0 if i % 3 else -1.0 for i in range(80)],  # own group
        }

    def test_at_most_one_symbol_per_correlation_group(self):
        picked = U.select_diversified(self.rets, threshold=0.9)
        self.assertEqual(len(picked), 3)
        self.assertIn("GOLD", picked)
        self.assertIn("ODD", picked)
        # Exactly one of the three near-duplicates.
        self.assertEqual(len({"BTC", "ETH", "SOL"} & set(picked)), 1)

    def test_liquidity_decides_within_a_group(self):
        picked = U.select_diversified(
            self.rets, threshold=0.9, liquidity={"BTC": 1.0, "ETH": 99.0, "SOL": 5.0}
        )
        self.assertIn("ETH", picked)
        self.assertNotIn("BTC", picked)

    def test_max_symbols_truncates_by_liquidity_not_alphabetically(self):
        picked = U.select_diversified(
            self.rets,
            max_symbols=2,
            threshold=0.9,
            liquidity={"BTC": 1.0, "ETH": 99.0, "SOL": 5.0, "GOLD": 50.0, "ODD": 2.0},
        )
        self.assertEqual(sorted(picked), ["ETH", "GOLD"])

    def test_result_is_sorted_and_repeatable(self):
        a = U.select_diversified(self.rets, threshold=0.9)
        b = U.select_diversified(
            dict(reversed(list(self.rets.items()))), threshold=0.9
        )
        self.assertEqual(a, b)
        self.assertEqual(a, sorted(a))

    def test_bad_max_symbols_is_rejected(self):
        with self.assertRaises(U.UniverseError):
            U.select_diversified(self.rets, max_symbols=0)


class TestReport(unittest.TestCase):
    def test_both_definitions_appear_so_neither_can_be_quoted_alone(self):
        base = [math.sin(i * 0.3) for i in range(80)]
        rets = {"A": base, "B": [v * 1.01 for v in base], "C": [-v for v in base]}
        text = U.correlation_report(rets, pooled_observations=130)
        self.assertIn("effective_bets", text)
        self.assertIn("sample_inflation", text)
        self.assertIn("pooled observations", text)
        # The definition must travel with the number.
        self.assertIn("k / (1 + (k-1)*rho)", text)
        self.assertIn("1 + (k-1)*rho", text)

    def test_it_says_so_when_nothing_can_be_scored(self):
        self.assertIn("no pair", U.correlation_report({"A": [1.0], "B": [2.0]}))

    def test_the_participation_ratio_caveat_is_printed(self):
        """Because a bare N_eff already caused one wrong conclusion here."""
        base = [math.sin(i * 0.3) for i in range(80)]
        text = U.correlation_report({"A": base, "B": [-v for v in base]})
        self.assertIn("participation ratio", text)


class TestCoverageFilter(unittest.TestCase):
    """Pooling across symbols requires a common window.

    A universe selected on liquidity alone contains recent listings: measured
    on one 35-symbol pool, four symbols covered between 8.9% and 74.5% of the
    window. Their trades belong to a different period, so including them mixes
    windows and the comparison stops being like-for-like.
    """

    def test_coverage_counts_usable_rows(self):
        panel = FakePanel({"A": [1.0] * 10, "B": [1.0] * 5 + [None] * 5})
        cov = U.coverage(panel)
        self.assertAlmostEqual(cov["A"], 1.0)
        self.assertAlmostEqual(cov["B"], 0.5)

    def test_filter_keeps_only_the_full_symbols(self):
        panel = FakePanel({"A": [1.0] * 10, "B": [1.0] * 5 + [None] * 5})
        self.assertEqual(U.filter_by_coverage(panel, 0.95), ["A"])

    def test_a_few_missing_hours_do_not_disqualify(self):
        """0.95 rather than 1.0: a handful of missing bars is a data gap, not a
        different window."""
        panel = FakePanel({"A": [1.0] * 10 + [None], "B": [None] * 10 + [1.0]})
        self.assertEqual(U.filter_by_coverage(panel, 0.9), ["A"])

    def test_bad_fraction_is_rejected(self):
        panel = FakePanel({"A": [1.0] * 5})
        for bad in (0.0, -0.1, 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises(U.UniverseError):
                    U.filter_by_coverage(panel, bad)


class TestLiquidityQuintiles(unittest.TestCase):
    """Tiers, not halves.

    On the researched universe the reversal effect is present in the four
    least-liquid tiers and absent in the most liquid one. A two-way split reports
    that as "present in the illiquid half" and hides where the boundary is.
    """

    @staticmethod
    def _panel(series, volume):
        """`FakePanel` starts with an empty volume dict, so it is attached here."""
        p = FakePanel(series)
        p.volume = volume
        return p

    def _tiers(self):
        # Dollar volume rises with the letter, so the expected order is known.
        series = {c: [100.0, 101.0, 100.5, 101.5, 100.0] for c in "ABCDE"}
        volume = {c: [float(i + 1) * 10.0] * 5 for i, c in enumerate("ABCDE")}
        return self._panel(series, volume)

    def test_the_least_liquid_tier_comes_first(self):
        tiers = U.liquidity_quintiles(self._tiers(), buckets=5)
        self.assertEqual(tiers[0], ["A"])
        self.assertEqual(tiers[-1], ["E"])

    def test_tiers_are_equal_sized_with_the_remainder_last(self):
        series = {f"S{i:02d}": [100.0] * 5 for i in range(11)}
        volume = {f"S{i:02d}": [float(i + 1) * 10.0] * 5 for i in range(11)}
        tiers = U.liquidity_quintiles(self._panel(series, volume), buckets=5)
        self.assertEqual([len(t) for t in tiers], [2, 2, 2, 2, 3])

    def test_every_symbol_appears_exactly_once(self):
        tiers = U.liquidity_quintiles(self._tiers(), buckets=5)
        flat = [s for t in tiers for s in t]
        self.assertEqual(sorted(flat), ["A", "B", "C", "D", "E"])

    def test_it_uses_the_median_not_the_mean(self):
        """A single listing spike must not move a symbol a whole tier."""
        series = {c: [100.0] * 5 for c in "AB"}
        volume = {"A": [1.0, 1.0, 1.0, 1.0, 1_000_000.0],      # spike on the last bar
                  "B": [10.0, 10.0, 10.0, 10.0, 10.0]}          # steady
        tiers = U.liquidity_quintiles(self._panel(series, volume), buckets=2)
        self.assertEqual(tiers[0], ["A"], "the spiky symbol ranked as the liquid one")

    def test_bad_bucket_counts_are_rejected(self):
        for bad in (0, 1, -3):
            with self.subTest(buckets=bad):
                with self.assertRaises(U.UniverseError):
                    U.liquidity_quintiles(self._tiers(), buckets=bad)

    def test_too_few_symbols_is_rejected(self):
        with self.assertRaises(U.UniverseError):
            U.liquidity_quintiles(self._tiers(), buckets=10)


class TestCorrelationHasOneConventionPerName(unittest.TestCase):
    """Two correlations exist and they disagree on purpose.

    `universe.correlation_or_none` returns None for a flat series ("undefined,
    not zero"); `combine.correlation` returns 0.0 ("a blend has to divide by
    something, and independent claims the least"). Both are deliberate. They
    were both called `correlation`, so an import picked the convention by
    accident - which is why this test pins that they still differ, and why at
    least one of them no longer answers to the bare name.
    """

    def test_universe_returns_none_for_a_flat_series(self):
        self.assertIsNone(U.correlation_or_none([0.0] * 50, [0.01, -0.01] * 25))

    def test_combine_returns_zero_for_the_same_input(self):
        from backtest.combine import correlation as combine_correlation
        self.assertEqual(combine_correlation([0.0] * 50, [0.01, -0.01] * 25), 0.0)

    def test_they_agree_where_both_are_defined(self):
        from backtest.combine import correlation as combine_correlation
        rng = random.Random(11)
        a = [rng.gauss(0, 1) for _ in range(200)]
        b = [rng.gauss(0, 1) for _ in range(200)]
        self.assertAlmostEqual(
            U.correlation_or_none(a, b, min_pairs=30),
            combine_correlation(a, b),
            places=9,
        )

    def test_the_name_says_which_convention(self):
        """A caller reading only the call site should be able to tell."""
        self.assertTrue(hasattr(U, "correlation_or_none"))
        self.assertFalse(
            hasattr(U, "correlation"),
            "the bare name is back; it collides with combine.correlation, "
            "which disagrees with this one on a flat series",
        )


class TestLiquidSplitSharesItsMedian(unittest.TestCase):
    """`liquid_split` and `liquidity_medians` must not drift apart.

    They were two implementations reading different accessors - `panel.close`
    versus `panel.price()`, which filters NaN - so a NaN would have split the
    same universe two ways. `liquid_split` now calls the shared one. This pins
    the agreement rather than the implementation.
    """

    @staticmethod
    def _panel():
        """One symbol carries a NaN price, which is the whole point.

        `universe.liquidity_medians` reads through `panel.price()`, which drops
        NaN; the version of this computation that used to live in
        `liquid_split` read `panel.close` and kept it. Without a NaN in the
        fixture the two are identical and a test cannot tell them apart - the
        first version of this test had no NaN and held the mutation that
        restored the duplicate.
        """
        series = {f"S{i:02d}": [100.0, 101.0, 100.5, 101.5] for i in range(10)}
        series["S03"] = [100.0, float("nan"), 100.5, 101.5]
        volume = {f"S{i:02d}": [float(i + 1) * 100.0] * 4 for i in range(10)}
        p = FakePanel(series)
        p.volume = volume
        return p

    def test_the_split_matches_the_medians_it_is_built_from(self):
        from backtest.cross_section import liquid_split
        panel = self._panel()
        for fraction in (0.2, 0.5, 0.8):
            with self.subTest(fraction=fraction):
                liquid, illiquid = liquid_split(panel, fraction)
                medians = U.liquidity_medians(panel)
                ordered = sorted(medians.values())
                cut = ordered[int(len(ordered) * (1.0 - fraction))]
                self.assertEqual(
                    liquid, {s for s, v in medians.items() if v >= cut}
                )
                self.assertEqual(illiquid, set(medians) - liquid)

    def test_the_two_halves_partition_the_universe(self):
        from backtest.cross_section import liquid_split
        liquid, illiquid = liquid_split(self._panel(), 0.5)
        self.assertFalse(liquid & illiquid)
        self.assertEqual(liquid | illiquid, set(U.liquidity_medians(self._panel())))


class TestPerpFilter(unittest.TestCase):
    """The archive interleaves spot pairs with perps.

    117 of 882 names on the sampled partition are `@<index>` spot pairs. They
    are not routeable by this system, their prices run to 2e-07, and four of
    them sit inside the top twenty by dollar volume - so a top-N cut on
    liquidity alone will pull some in.
    """

    def test_spot_pairs_are_recognised(self):
        self.assertFalse(U.is_perp_name("@142"))
        self.assertFalse(U.is_perp_name("@2"))
        self.assertTrue(U.is_perp_name("BTC"))
        self.assertTrue(U.is_perp_name("kPEPE"))

    def test_tradeable_names_drops_only_the_spot_pairs(self):
        names = ["BTC", "@142", "ETH", "@2", "kPEPE"]
        self.assertEqual(U.tradeable_names(names), ["BTC", "ETH", "kPEPE"])

    def test_the_thousand_x_perp_variants_are_kept(self):
        """`kPEPE` is a real perp (1000x notional), unlike `@N`. Dropping it
        would be over-filtering under the same prefix suspicion."""
        self.assertEqual(U.tradeable_names(["kPEPE", "kBONK", "kSHIB"]),
                         ["kBONK", "kPEPE", "kSHIB"])


class TestLiquidityMedians(unittest.TestCase):
    def test_median_is_used_so_one_spike_does_not_decide(self):
        panel = FakePanel({"A": [1.0, 1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0, 1.0]})
        panel.volume = {"A": [10.0, 20.0, 30.0, 1000.0], "B": [25.0, 25.0, 25.0, 25.0]}
        med = U.liquidity_medians(panel)
        self.assertAlmostEqual(med["A"], 25.0)
        self.assertAlmostEqual(med["B"], 25.0)

    def test_a_panel_without_volume_is_an_error_not_a_guess(self):
        with self.assertRaises(U.UniverseError):
            U.liquidity_medians(FakePanel({"A": [1.0, 2.0]}))


if __name__ == "__main__":
    unittest.main()
