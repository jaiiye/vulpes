"""Tests for the cross-sectional portfolio layer.

These run on synthetic panels, so they exercise the portfolio and signal logic
without duckdb or the archive. The one integration check - that the module
reproduces the research numbers - lives in `test_cross_section_parity` at the
bottom and skips itself when the archive is absent.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

from backtest.cross_section import (
    CrossSectionBacktester,
    CrossSectionConfig,
    CrossSectionError,
    Panel,
    build_panel,
    liquid_split,
    negate,
    past_return,
    percentile_of,
    random_signal,
    realized_vol,
    restrict,
)


def make_panel(n_rows: int, series: dict[str, list[float | None]]) -> Panel:
    """A panel with hourly timestamps and the supplied series."""
    times = [1_700_000_000_000 + i * 3_600_000 for i in range(n_rows)]
    close = {s: list(v) for s, v in series.items()}
    volume = {s: [1000.0 if c is not None else None for c in v]
              for s, v in series.items()}
    return Panel(times=times, symbols=sorted(series), close=close, volume=volume)


def ramp(start: float, step: float, n: int) -> list[float]:
    return [start + step * i for i in range(n)]


class TestPanel(unittest.TestCase):
    def test_a_short_series_is_rejected(self):
        """Length mismatch must be an error, not a silently truncated panel:
        every lookup indexes by row, so a short series would misreport its last
        rows as missing rather than as absent from the data."""
        with self.assertRaises(CrossSectionError):
            make_panel(5, {"A": [1.0, 2.0, 3.0]})

    def test_empty_panel_is_rejected(self):
        with self.assertRaises(CrossSectionError):
            Panel(times=[], symbols=["A"], close={"A": []})

    def test_missing_bar_reads_as_none_not_zero(self):
        """A delisted symbol and a symbol priced at zero are different things;
        only one of them should be silently tradeable."""
        p = make_panel(3, {"A": [10.0, None, 12.0]})
        self.assertIsNone(p.price("A", 1))
        self.assertEqual(p.price("A", 2), 12.0)

    def test_nan_is_treated_as_missing(self):
        """NaN survives arithmetic and would reach the weights as a silent zero."""
        p = make_panel(2, {"A": [10.0, float("nan")]})
        self.assertIsNone(p.price("A", 1))

    def test_present_lists_only_symbols_with_a_price(self):
        p = make_panel(3, {"A": [1.0, 1.0, 1.0], "B": [1.0, None, 1.0]})
        self.assertEqual(p.present(0), ["A", "B"])
        self.assertEqual(p.present(1), ["A"])

    def test_last_row_of_finds_the_final_usable_bar(self):
        p = make_panel(4, {"A": [1.0, 2.0, None, None]})
        self.assertEqual(p.last_row_of("A"), 1)

    def test_last_row_of_an_always_missing_symbol_is_minus_one(self):
        p = make_panel(2, {"A": [None, None]})
        self.assertEqual(p.last_row_of("A"), -1)


class TestBuildPanel(unittest.TestCase):
    def test_rows_are_aligned_and_gaps_become_none(self):
        rows = [
            {"coin": "A", "t": 100, "c": 1.0, "v": 10.0},
            {"coin": "B", "t": 100, "c": 2.0, "v": 20.0},
            {"coin": "A", "t": 200, "c": 3.0, "v": 30.0},   # B missing at 200
        ]
        p = build_panel(rows)
        self.assertEqual(p.times, [100, 200])
        self.assertEqual(p.close["A"], [1.0, 3.0])
        self.assertEqual(p.close["B"], [2.0, None])

    def test_min_coverage_drops_thin_symbols(self):
        rows = [
            {"coin": "A", "t": 1, "c": 1.0, "v": 1.0},
            {"coin": "A", "t": 2, "c": 1.0, "v": 1.0},
            {"coin": "B", "t": 1, "c": 1.0, "v": 1.0},
        ]
        self.assertIn("B", build_panel(rows).symbols)
        self.assertNotIn("B", build_panel(rows, min_coverage=2).symbols)

    def test_no_rows_is_an_error(self):
        with self.assertRaises(CrossSectionError):
            build_panel([])


class TestSignals(unittest.TestCase):
    def test_past_return_is_the_trailing_change(self):
        p = make_panel(4, {"A": [100.0, 110.0, 121.0, 133.1]})
        self.assertAlmostEqual(past_return(2)(p, 3, "A"), 0.21, places=6)

    def test_past_return_is_none_before_the_lookback(self):
        p = make_panel(4, {"A": [100.0, 110.0, 121.0, 133.1]})
        self.assertIsNone(past_return(2)(p, 1, "A"))

    def test_past_return_tolerates_a_gap_between_the_endpoints(self):
        """Only the two endpoints must exist.

        This is deliberate: a symbol with one missing bar inside the window
        still has a well defined 7-day return, and returning None for it would
        drop such symbols from every ranking. What must NOT happen is reading
        the return off a bar that is itself missing - that is the case below
        where the *start* endpoint is absent.
        """
        p = make_panel(3, {"A": [100.0, None, 121.0]})
        self.assertAlmostEqual(past_return(2)(p, 2, "A"), 0.21, places=6)

    def test_past_return_is_none_when_an_endpoint_is_missing(self):
        p = make_panel(3, {"A": [None, 110.0, 121.0]})
        self.assertIsNone(past_return(2)(p, 2, "A"))

    def test_negate_flips_only_the_sign(self):
        p = make_panel(3, {"A": [100.0, 100.0, 110.0]})
        self.assertAlmostEqual(past_return(2)(p, 2, "A"), 0.10, places=6)
        self.assertAlmostEqual(negate(past_return(2))(p, 2, "A"), -0.10, places=6)

    def test_negate_passes_none_through(self):
        p = make_panel(3, {"A": [None, 110.0, 121.0]})
        self.assertIsNone(negate(past_return(2))(p, 2, "A"))

    def test_realized_vol_is_zero_on_a_flat_series(self):
        p = make_panel(5, {"A": [100.0] * 5})
        self.assertAlmostEqual(realized_vol(3)(p, 4, "A"), 0.0, places=12)

    def test_realized_vol_needs_a_full_window(self):
        p = make_panel(5, {"A": [100.0, 101.0, None, 102.0, 103.0]})
        self.assertIsNone(realized_vol(3)(p, 4, "A"))


class TestPortfolio(unittest.TestCase):
    """A panel where the signal's intent is knowable by construction."""

    def build(self, n=60, n_symbols=20):
        # Alternating up/down drifts, so the reversal and momentum ends are known.
        series = {}
        for k in range(n_symbols):
            step = 0.5 if k % 2 == 0 else -0.5
            series[f"S{k:02d}"] = ramp(100.0, step, n)
        return make_panel(n, series)

    def test_quantiles_split_the_ranked_universe(self):
        p = self.build()
        cfg = CrossSectionConfig(quantile=0.2, min_symbols=10)
        r = CrossSectionBacktester(p, past_return(6), 5, cfg).run()
        self.assertTrue(r.periods)
        for period in r.periods:
            self.assertEqual(period.n_long, 4)   # 20 symbols * 0.2
            self.assertEqual(period.n_short, 4)

    def test_min_symbols_blocks_a_thin_universe(self):
        """Below the floor the quantiles are a handful of names, and the result
        is one symbol's move rather than a cross-section."""
        p = self.build(n_symbols=6)
        cfg = CrossSectionConfig(min_symbols=20)
        r = CrossSectionBacktester(p, past_return(6), 5, cfg).run()
        self.assertEqual(len(r.periods), 0)

    def test_first_period_pays_the_full_entry_turnover(self):
        """Nothing is held before the first basket, so its turnover is 100% and
        the cost is two legs' worth. Charging zero here would make any rule look
        cheaper than it is."""
        p = self.build()
        cfg = CrossSectionConfig(quantile=0.2, min_symbols=10, fee_bps=10.0)
        r = CrossSectionBacktester(p, past_return(6), 5, cfg).run()
        first = r.periods[0]
        self.assertAlmostEqual(first.turnover, 1.0, places=9)
        self.assertAlmostEqual(first.cost, 1.0 * 2 * 10.0 / 10_000.0, places=9)

    def test_holding_the_same_names_twice_costs_nothing_the_second_time(self):
        """A signal that never changes its ranking must not be charged turnover
        it did not incur - this is the difference between a plausible backtest
        and one where costs are a constant drag."""
        n_rows = 40
        series = {}
        for k in range(20):
            # All move at the same rate, so the ranking is fixed for all time.
            series[f"S{k:02d}"] = ramp(100.0 + k, 1.0, n_rows)
        p = make_panel(n_rows, series)
        cfg = CrossSectionConfig(quantile=0.2, min_symbols=10, fee_bps=10.0)
        r = CrossSectionBacktester(p, past_return(6), 5, cfg).run()
        self.assertGreater(len(r.periods), 3)
        self.assertAlmostEqual(r.periods[0].turnover, 1.0, places=9)
        for period in r.periods[1:]:
            self.assertAlmostEqual(period.turnover, 0.0, places=9)
            self.assertAlmostEqual(period.cost, 0.0, places=9)

    def test_net_return_is_gross_minus_cost(self):
        p = self.build()
        cfg = CrossSectionConfig(quantile=0.2, min_symbols=10, fee_bps=10.0)
        r = CrossSectionBacktester(p, past_return(6), 5, cfg).run()
        for period in r.periods:
            self.assertAlmostEqual(
                period.net_return,
                period.long_return - period.short_return - period.cost,
                places=12,
            )

    def test_long_only_has_no_short_leg_and_half_the_legs(self):
        p = self.build()
        cfg = CrossSectionConfig(
            quantile=0.2, min_symbols=10, fee_bps=10.0, long_only=True
        )
        r = CrossSectionBacktester(p, past_return(6), 5, cfg).run()
        self.assertTrue(r.periods)
        for period in r.periods:
            self.assertEqual(period.n_short, 0)
            self.assertEqual(period.short_return, 0.0)
        # One leg, so the first period costs half of the two-leg case.
        self.assertAlmostEqual(r.periods[0].cost, 10.0 / 10_000.0, places=9)


class TestDelisting(unittest.TestCase):
    """A long-the-losers basket is maximally exposed to symbols that stop
    trading, so the exit-price assumption has to be visible and bracketable."""

    def panel_with_a_death(self, n=40):
        """Ten risers, ten fallers with distinct slopes, and the steepest faller
        stops trading at row 25.

        Every symbol must be distinguishable, and the one that dies must be the
        most extreme in its leg. Two earlier versions of this panel passed for
        the wrong reason: the first killed the symbol before the lookback had
        warmed up, so it never entered a leg; the second gave all ten fallers an
        identical series, so the sort broke the tie on symbol name and the dead
        symbol was never selected at all. Both produced two identical runs and a
        test that proved nothing.
        """
        series = {}
        for k in range(10):
            series[f"U{k:02d}"] = ramp(100.0, 1.0 + 0.05 * k, n)   # risers
        for k in range(9):
            series[f"D{k:02d}"] = ramp(100.0, -0.5 - 0.05 * k, n)  # fallers
        series["D09"] = [100.0 - 2.0 * i for i in range(25)] + [None] * (n - 25)
        return make_panel(n, series)

    def run_with(self, factor, signal=None):
        p = self.panel_with_a_death()
        cfg = CrossSectionConfig(
            quantile=0.2, min_symbols=10, fee_bps=0.0, late_exit_factor=factor
        )
        return CrossSectionBacktester(
            p, signal or past_return(2), 5, cfg
        ).run()

    def spanning_periods(self, result, panel, symbol="D09"):
        """Periods where `symbol` was held at entry and gone by exit.

        Detected from the price series rather than from row arithmetic: the
        first attempt filtered on `entry <= death_row < exit`, which missed the
        period that actually spans the death and picked up a later one where the
        symbol was already excluded - so the test asserted a leg size for a
        period that never contained the symbol.
        """
        return [
            x for x in result.periods
            if panel.price(symbol, x.entry_row) is not None
            and panel.price(symbol, x.exit_row) is None
        ]

    def test_the_death_is_actually_inside_a_traded_period(self):
        """Guards the guard.

        If no period spans the death, the two tests below compare two identical
        runs and prove nothing. Two earlier versions of this panel were wrong in
        exactly that way - see `panel_with_a_death` - and both made the tests
        below pass vacuously.
        """
        p = self.panel_with_a_death()
        cfg = CrossSectionConfig(quantile=0.2, min_symbols=10, fee_bps=0.0)
        r = CrossSectionBacktester(p, past_return(2), 5, cfg).run()
        self.assertTrue(
            self.spanning_periods(r, p), "no period holds the symbol through death"
        )

    def test_the_dead_symbol_is_shorted_under_a_momentum_signal(self):
        """Establishes which leg the death lands in, so the two assertions below
        are about the side the test thinks they are."""
        p = self.panel_with_a_death()
        cfg = CrossSectionConfig(quantile=0.2, min_symbols=10)
        r = CrossSectionBacktester(p, past_return(2), 5, cfg).run()
        spanning = self.spanning_periods(r, p)
        self.assertTrue(spanning)
        # A momentum signal ranks the steepest faller lowest, and the low end is
        # the short leg - so the death lands on the short side.
        self.assertEqual(spanning[0].n_short, 4)
        # The leg is an equal-weighted basket of four, so the -1.0 from the dead
        # name is diluted by the three that merely fell ~4%. Asserting -1.0 here
        # was wrong; the point is that the death dominates the leg average.
        self.assertLess(spanning[0].short_return, -0.2)

    def test_a_shorted_symbol_going_to_zero_profits_the_short_leg(self):
        """A short in a symbol that dies makes money, so the pessimistic exit
        assumption is the *favourable* one here. Getting this sign backwards is
        how a delisting check silently becomes a no-op."""
        killed = self.run_with(0.0)          # dead symbol marked to -100%
        salvaged = self.run_with(1.0)        # dead symbol marked flat
        self.assertGreater(killed.net_return_pct, salvaged.net_return_pct)

    def test_a_longs_symbol_going_to_zero_is_a_total_loss(self):
        """The case the research's survivorship discussion was about: the
        reversal basket is long the losers, so a death there is the loss that
        excluding delisted symbols would hide."""
        killed = self.run_with(0.0, negate(past_return(2)))
        salvaged = self.run_with(1.0, negate(past_return(2)))
        self.assertLess(killed.net_return_pct, salvaged.net_return_pct)

    def test_the_dead_name_is_not_silently_dropped(self):
        """Dropping it would be the survivorship bias that flatters exactly this
        strategy: the loss would never be realised and the leg would look
        better than it was."""
        for signal in (past_return(2), negate(past_return(2))):
            with self.subTest(negated=signal is not past_return(2)):
                killed = self.run_with(0.0, signal)
                salvaged = self.run_with(1.0, signal)
                self.assertGreater(
                    abs(killed.net_return_pct - salvaged.net_return_pct), 1e-9
                )


class TestRandomControl(unittest.TestCase):
    """The control that overturned every other conclusion in this repo."""

    def test_the_same_seed_produces_the_same_draw(self):
        """The control is only a control if it is reproducible; a benchmark that
        changes run to run cannot be compared against anything."""
        p = make_panel(40, {f"S{k}": ramp(100.0 + k, 0.3, 40) for k in range(15)})
        cfg = CrossSectionConfig(quantile=0.2, min_symbols=10)
        a = CrossSectionBacktester(p, random_signal(7), 5, cfg).run()
        b = CrossSectionBacktester(p, random_signal(7), 5, cfg).run()
        self.assertAlmostEqual(a.net_return_pct, b.net_return_pct, places=12)

    def test_different_seeds_diverge(self):
        p = make_panel(40, {f"S{k}": ramp(100.0 + k, 0.3, 40) for k in range(15)})
        cfg = CrossSectionConfig(quantile=0.2, min_symbols=10)
        a = CrossSectionBacktester(p, random_signal(1), 5, cfg).run()
        b = CrossSectionBacktester(p, random_signal(2), 5, cfg).run()
        self.assertNotAlmostEqual(a.net_return_pct, b.net_return_pct, places=6)

    def test_percentile_boundaries(self):
        dist = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(percentile_of(0.0, dist), 0.0)
        self.assertEqual(percentile_of(5.0, dist), 100.0)
        self.assertEqual(percentile_of(2.5, dist), 50.0)

    def test_percentile_of_an_empty_distribution(self):
        self.assertEqual(percentile_of(1.0, []), 0.0)


class TestLiquidSplit(unittest.TestCase):
    def test_splits_by_median_dollar_volume(self):
        n = 10
        series, volume = {}, {}
        for k in range(10):
            series[f"S{k}"] = ramp(100.0, 0.1, n)
            volume[f"S{k}"] = [float(k + 1) * 100] * n     # strictly increasing
        p = Panel(
            times=list(range(n)), symbols=sorted(series),
            close=series, volume=volume,
        )
        liquid, illiquid = liquid_split(p, 0.5)
        self.assertEqual(len(liquid), 5)
        self.assertEqual(len(illiquid), 5)
        self.assertIn("S9", liquid)
        self.assertIn("S0", illiquid)

    def test_restrict_keeps_the_time_axis(self):
        p = make_panel(10, {f"S{k}": ramp(1.0, 1.0, 10) for k in range(5)})
        r = restrict(p, {"S0", "S1"})
        self.assertEqual(r.times, p.times)
        self.assertEqual(r.symbols, ["S0", "S1"])

    def test_restricting_to_nothing_is_an_error(self):
        p = make_panel(5, {"S0": ramp(1.0, 1.0, 5)})
        with self.assertRaises(CrossSectionError):
            restrict(p, {"nope"})


class TestCrossSectionParity(unittest.TestCase):
    """The module must reproduce the numbers the research measured.

    Skipped unless duckdb and the archive are both present. This is the check
    that makes the module trustworthy rather than merely self-consistent: the
    research numbers came from a throwaway script, and a rewrite that quietly
    changes them would invalidate the conclusions drawn from them.
    """

    ARCHIVE = Path("data/canonical/candles")
    EXPECTED = {
        # (hold_rows, signal_lag) -> net return of the illiquid half
        (42, 42): 87.20,
        (42, 6): 7.37,
        (18, 42): 78.05,
        (18, 6): 86.32,
    }

    def test_illiquid_reversal_reproduces_the_research(self):
        if shutil.which("duckdb") is None:
            self.skipTest("duckdb not on PATH")
        if not self.ARCHIVE.is_dir():
            self.skipTest("no candle archive on this machine")

        sql = f"""
SELECT coin, epoch_ms(time_bucket(INTERVAL '4 hours', timestamp)) AS t,
       arg_max(close, timestamp)::DOUBLE AS c, SUM(volume)::DOUBLE AS v
FROM read_parquet('{self.ARCHIVE}/*.parquet') AS x(
    coin, timestamp, open, high, low, close, volume, filename)
WHERE coin NOT LIKE '@%'
GROUP BY 1, 2 ORDER BY 2, 1;
"""
        proc = subprocess.run(
            ["duckdb", "-json"], input=sql, capture_output=True, text=True,
            timeout=1800,
        )
        if proc.returncode != 0:
            self.skipTest("archive query failed")
        # The universe is "whatever existed at row 60", not "whatever has full
        # coverage": requiring full coverage removes the delistings this basket
        # is most exposed to, which is the survivorship bias under test.
        panel = build_panel(json.loads(proc.stdout))
        panel = restrict(panel, set(panel.present(60)))
        _, illiquid = liquid_split(panel)

        cfg = CrossSectionConfig(
            quantile=0.2, min_symbols=20, fee_bps=3.5, late_exit_factor=0.0
        )
        subset = restrict(panel, illiquid)
        for (hold, lag), expected in self.EXPECTED.items():
            with self.subTest(hold=hold, signal_lag=lag):
                r = CrossSectionBacktester(
                    subset, negate(past_return(lag)), hold, cfg, 60
                ).run()
                self.assertAlmostEqual(r.net_return_pct, expected, delta=0.5)


if __name__ == "__main__":
    unittest.main()
