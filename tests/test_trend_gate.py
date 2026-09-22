"""Tests for the four-condition trend gate.

Most of these are ordinary arithmetic. One is not: `TestNoLookAhead` guards the
fill-timing convention, because the first version of the simulator filled exits
at the open of the same bar whose close produced the vote. That is one bar of
impossible foreknowledge, and it turned a rule that loses in seven of nine
configurations into one that cleared the 100th random percentile in all nine.
A regression there would be silent and would look like a discovery.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import trend_gate as tg  # noqa: E402


def ramp(n: int, start: float = 100.0, step: float = 1.0) -> list[float]:
    return [start + i * step for i in range(n)]


class TestVotes(unittest.TestCase):
    """Each condition, tested by moving only the parameter it owns.

    Data cannot isolate these: in any series that rises, momentum and the
    Bollinger condition are both satisfied, so dropping the last close hard
    enough to kill momentum usually kills the band condition too - and a test
    written that way would be asserting an accident of the sample rather than
    the wiring.

    Each of the four conditions reads exactly one config field. So the isolating
    move is to change *that field* and require the other three flags to be
    identical. A condition wired to the wrong comparison breaks that, while a
    bare 0-4 count would not show it.
    """

    N = 700  # long enough to clear the 480-bar volume warm-up

    def _flags(self, closes, volumes, **kw):
        detail = tg.vote_detail(closes, volumes, tg.TrendGateConfig(**kw))
        return detail[-1]

    def _closes_with_a_turn(self):
        """A long slide that has turned up over the last few bars.

        Gives the four conditions four different answers, which is what makes
        them separable: up over a short lookback, down over a long one, and
        below the long averages.
        """
        closes = [200.0 - 0.10 * i for i in range(self.N)]
        # A rise of 1.0 over five bars, against a 19-bar decline of 1.9, so the
        # 5-bar change is positive while the 24-bar change is negative. A
        # bigger bounce would make both positive and the case would prove
        # nothing.
        for k in range(1, 6):
            closes[-k] = closes[-6] + 0.2 * (6 - k)
        return closes

    def test_all_four_pass_on_a_clean_uptrend(self):
        closes = ramp(self.N, step=1.0)
        volumes = [100.0 + i for i in range(self.N)]
        f = self._flags(closes, volumes)
        self.assertTrue(f.momentum and f.volume and f.bollinger and f.ema)
        self.assertEqual(f.total, 4)

    def test_warm_up_bars_are_none_not_zero(self):
        """Zero would be indistinguishable from "all four are false", and the
        first 480 bars would then read as a strong bearish signal rather than
        as missing data."""
        closes = ramp(self.N)
        volumes = [100.0] * self.N
        v = tg.vote_series(closes, volumes, tg.TrendGateConfig())
        self.assertTrue(all(x is None for x in v[:479]))
        self.assertIsNotNone(v[479])

    def test_momentum_reads_its_own_lookback_and_nothing_else(self):
        closes = self._closes_with_a_turn()
        volumes = [100.0] * (self.N - 1) + [200.0]
        short = self._flags(closes, volumes, momentum_bars=5)
        long = self._flags(closes, volumes, momentum_bars=24)
        self.assertTrue(short.momentum)   # up over 5 bars
        self.assertFalse(long.momentum)   # down over 24
        # The other three must not have moved.
        self.assertEqual(short.volume, long.volume)
        self.assertEqual(short.bollinger, long.bollinger)
        self.assertEqual(short.ema, long.ema)

    def test_bollinger_reads_its_own_period_and_nothing_else(self):
        closes = self._closes_with_a_turn()
        volumes = [100.0] * (self.N - 1) + [200.0]
        short = self._flags(closes, volumes, bollinger_period=2)
        long = self._flags(closes, volumes, bollinger_period=200)
        self.assertTrue(short.bollinger)   # above the 2-bar mean
        self.assertFalse(long.bollinger)   # below the 200-bar mean
        self.assertEqual(short.momentum, long.momentum)
        self.assertEqual(short.volume, long.volume)
        self.assertEqual(short.ema, long.ema)

    def test_ema_reads_its_own_pair_and_nothing_else(self):
        closes = self._closes_with_a_turn()
        volumes = [100.0] * (self.N - 1) + [200.0]
        short = self._flags(closes, volumes, ema_fast=2, ema_slow=3)
        long = self._flags(closes, volumes, ema_fast=2, ema_slow=200)
        self.assertTrue(short.ema)         # fast above a 3-bar slow
        self.assertFalse(long.ema)         # fast below a 200-bar slow
        self.assertEqual(short.momentum, long.momentum)
        self.assertEqual(short.volume, long.volume)
        self.assertEqual(short.bollinger, long.bollinger)

    def test_volume_reads_only_volume(self):
        closes = ramp(self.N, step=1.0)
        mean = [100.0] * self.N
        below = list(mean)
        below[-1] = 50.0        # below its own trailing mean
        above = list(mean)
        above[-1] = 200.0
        f_below = self._flags(closes, below)
        f_above = self._flags(closes, above)
        self.assertFalse(f_below.volume)
        self.assertTrue(f_above.volume)
        # A volume change must not touch price-derived conditions.
        self.assertEqual(f_below.momentum, f_above.momentum)
        self.assertEqual(f_below.bollinger, f_above.bollinger)
        self.assertEqual(f_below.ema, f_above.ema)

    def test_volume_equal_to_its_mean_is_not_above_it(self):
        """The proposal says "> 1.0", so equality fails. Off-by-one on a strict
        comparison is invisible inside a 0-4 total."""
        closes = ramp(self.N, step=1.0)
        f = self._flags(closes, [100.0] * self.N)
        self.assertFalse(f.volume)

    def test_flat_momentum_is_not_positive(self):
        closes = ramp(self.N, step=1.0)
        closes[-1] = closes[-1 - 24]  # exactly flat over 24 bars
        f = self._flags(closes, [100.0] * (self.N - 1) + [200.0])
        self.assertFalse(f.momentum)

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(tg.TrendGateError):
            tg.vote_series([1.0] * 10, [1.0] * 9)

    def test_min_bull_votes_outside_the_range_is_rejected(self):
        for bad in (0, 5):
            with self.subTest(bad=bad):
                with self.assertRaises(tg.TrendGateError):
                    tg.TrendGateConfig(min_bull_votes=bad)

    def test_ema_fast_must_be_shorter_than_slow(self):
        with self.assertRaises(tg.TrendGateError):
            tg.TrendGateConfig(ema_fast=55, ema_slow=21)


class TestNoLookAhead(unittest.TestCase):
    """A fill must come from a price that is later than its decision.

    The vote at bar i consumes closes[i]; the first price genuinely after that
    information is opens[i + 1]. The assertion is on the *relationship between
    the signal bar and the fill bar*, not on price-versus-fill-bar: the first
    version of these tests checked `exit_price == opens[exit_bar]`, and moving
    the fill to the bar before its trigger moved both fields together, so the
    test passed while the bug was present.
    """

    N = 700

    def _series(self):
        closes = ramp(self.N, step=1.0)
        opens = [c - 0.5 for c in closes]
        # Rising volume: with a flat volume series the volume condition never
        # passes, so the maximum is 3 votes and a threshold of 4 never trades.
        volumes = [100.0 + i for i in range(self.N)]
        return closes, opens, volumes

    def _run(self, threshold=4, start=500):
        closes, opens, volumes = self._series()
        r = tg.simulate(closes, opens, volumes,
                        tg.TrendGateConfig(min_bull_votes=threshold),
                        start_bar=start)
        return r, closes, opens

    def test_every_fill_is_strictly_later_than_the_bar_that_decided_it(self):
        r, closes, opens = self._run()
        self.assertTrue(r.trades)
        for t in r.trades:
            with self.subTest(entry=t.entry_bar):
                self.assertGreater(
                    t.entry_bar, t.entry_signal_bar,
                    "entry filled on or before the bar whose close voted",
                )
                self.assertGreater(
                    t.exit_bar, t.exit_signal_bar,
                    "exit filled on or before the bar whose close voted",
                )

    def test_the_fill_price_is_an_open_on_the_fill_bar(self):
        """Price and bar must agree, and the price must be an open - the
        mark-out at the end of the sample is the one documented exception."""
        r, closes, opens = self._run()
        for i, t in enumerate(r.trades):
            with self.subTest(entry=t.entry_bar):
                self.assertAlmostEqual(t.entry_price, opens[t.entry_bar], places=9)
                final = i == len(r.trades) - 1 and t.exit_bar >= len(opens) - 1
                if final:
                    self.assertAlmostEqual(t.exit_price, closes[-1], places=9)
                else:
                    self.assertAlmostEqual(t.exit_price, opens[t.exit_bar], places=9)

    def test_the_signal_bar_is_recorded_and_is_not_the_fill_bar(self):
        """Guards the field itself: if a change stopped recording it, the
        checks above would compare -1 against a bar index and pass vacuously
        for an entry that never had a signal attributed to it."""
        r, _, _ = self._run()
        for t in r.trades:
            self.assertGreaterEqual(t.entry_signal_bar, 0)
            self.assertGreaterEqual(t.exit_signal_bar, 0)
            self.assertNotEqual(t.entry_signal_bar, t.entry_bar)
            self.assertNotEqual(t.exit_signal_bar, t.exit_bar)

    def _regime_series(self):
        """Up, then down, then up again - so the rule *exits* on its own.

        A monotonic ramp is not enough and that was a real gap: it produces a
        single trade that closes at the end-of-sample mark-out, which does not
        go through the exit path at all. A mutation that moved every ordinary
        exit one bar early changed nothing and was caught by nothing.
        """
        n = self.N
        closes = []
        price = 100.0
        for i in range(n):
            if i < 300:
                price += 1.0
            elif i < 500:
                price -= 1.2
            else:
                price += 1.0
            closes.append(price)
        opens = [c - 0.5 for c in closes]
        volumes = [100.0 + (i % 40) for i in range(n)]
        return closes, opens, volumes

    def test_ordinary_exits_are_also_delayed(self):
        closes, opens, volumes = self._regime_series()
        r = tg.simulate(closes, opens, volumes,
                        tg.TrendGateConfig(min_bull_votes=3), start_bar=500)
        # At least one trade must end before the sample does, otherwise this
        # test is exercising nothing but the mark-out again.
        ordinary = [t for t in r.trades if t.exit_bar < self.N - 1]
        self.assertTrue(ordinary, "no vote-driven exit occurred; series unusable")
        for t in ordinary:
            with self.subTest(exit=t.exit_bar):
                self.assertGreater(t.exit_bar, t.exit_signal_bar)


class TestSimulation(unittest.TestCase):
    N = 700

    def test_an_open_position_is_marked_out_at_the_end(self):
        """Leaving it open would report the trade count without the trade, and
        an unrealised gain is not a result."""
        closes = ramp(self.N, step=1.0)
        opens = [c - 0.5 for c in closes]
        r = tg.simulate(closes, opens, [100.0] * self.N,
                        tg.TrendGateConfig(min_bull_votes=1), start_bar=500)
        self.assertTrue(r.trades)
        self.assertLessEqual(r.trades[-1].exit_bar, self.N)

    def test_fees_are_charged_on_both_legs(self):
        """Same price in and out must still lose exactly two legs of costs."""
        closes = [100.0] * self.N
        opens = [100.0] * self.N
        volumes = [100.0] * self.N
        # A series where EMA is flat gives votes < 4 only if volume/momentum
        # fail; threshold 1 keeps it in the market so a round trip happens.
        r = tg.simulate(closes, opens, volumes, tg.TrendGateConfig(min_bull_votes=1),
                        start_bar=479)
        if r.trades:
            for t in r.trades:
                self.assertAlmostEqual(
                    t.net_return, (1.0 - 0.00035) ** 2 - 1.0, places=12
                )

    def test_exposure_is_a_fraction_of_the_traded_span(self):
        closes = ramp(self.N, step=1.0)
        opens = [c - 0.5 for c in closes]
        r = tg.simulate(closes, opens, [100.0] * self.N,
                        tg.TrendGateConfig(min_bull_votes=2), start_bar=500)
        self.assertGreater(r.exposure, 0.0)
        self.assertLessEqual(r.exposure, 1.0)
        self.assertEqual(r.bars_total, r.bars_total)  # set, not left at zero

    def test_a_bars_total_is_never_zero(self):
        """It is used as a divisor for exposure."""
        closes = ramp(self.N)
        r = tg.simulate(closes, [c - 0.5 for c in closes], [100.0] * self.N,
                        start_bar=self.N - 3)
        self.assertGreaterEqual(r.bars_total, 1)

    def test_no_tradable_range_is_an_error(self):
        closes = ramp(self.N)
        with self.assertRaises(tg.TrendGateError):
            tg.simulate(closes, closes, [100.0] * self.N, start_bar=self.N)

    def test_buy_and_hold_spans_the_traded_window_not_the_whole_series(self):
        """The reference has to start where trading starts.

        Measuring from bar 0 would credit the rule with the warm-up's rise,
        which is the sort of uncomparable-window slip the project's log records
        repeatedly. From bar 500 the ramp is 600 -> 799, about +33%.
        """
        closes = ramp(self.N, start=100.0, step=1.0)
        opens = [c - 0.5 for c in closes]
        start = 500
        r = tg.simulate(closes, opens, [100.0] * self.N,
                        tg.TrendGateConfig(min_bull_votes=4), start_bar=start)
        expected = (closes[self.N - 1] / opens[start] - 1.0) * 100.0
        self.assertAlmostEqual(r.buy_hold_return, expected, places=6)
        self.assertGreater(r.buy_hold_return, 30.0)


class TestRandomControl(unittest.TestCase):
    N = 700

    def test_it_returns_one_result_per_run(self):
        closes = ramp(self.N)
        opens = [c - 0.5 for c in closes]
        out = tg.random_control(closes, opens, 20, [5], runs=7, start_bar=0)
        self.assertEqual(len(out), 7)

    def test_the_same_seed_gives_the_same_distribution(self):
        closes = ramp(self.N)
        opens = [c - 0.5 for c in closes]
        a = tg.random_control(closes, opens, 20, [5], runs=5, seed=99, start_bar=0)
        b = tg.random_control(closes, opens, 20, [5], runs=5, seed=99, start_bar=0)
        self.assertEqual(a, b)

    def test_trades_never_overlap(self):
        """The real rule cannot hold two positions at once, so the control must
        not either - otherwise the control gets more exposure than the rule and
        the comparison stops being like-for-like.

        Asserted on the spans rather than on returns. A return threshold was
        tried first and caught nothing, because overlapping trades still
        produce a plausible-looking distribution.
        """
        for spans in tg.random_spans(40, [10], runs=30, seed=5,
                                     start_bar=0, last_bar=self.N - 1):
            cursor = -1
            for entry, exit_ in spans:
                with self.subTest(entry=entry):
                    self.assertGreater(entry, cursor, "span overlaps the previous one")
                    self.assertGreater(exit_, entry)
                cursor = exit_

    def test_spans_stay_inside_the_requested_range(self):
        last = self.N - 1
        for spans in tg.random_spans(40, [10], runs=10, seed=3,
                                     start_bar=100, last_bar=last):
            for entry, exit_ in spans:
                with self.subTest(entry=entry):
                    self.assertGreaterEqual(entry, 100)
                    self.assertLessEqual(exit_, last)

    def test_a_range_with_no_room_is_an_error(self):
        with self.assertRaises(tg.TrendGateError):
            tg.random_spans(5, [3], runs=1, start_bar=10, last_bar=10)

    def test_zero_trades_gives_an_empty_distribution(self):
        closes = ramp(self.N)
        self.assertEqual(tg.random_control(closes, closes, 0, [5], start_bar=0), [])

    def test_percentile_of_is_monotone(self):
        dist = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(tg.percentile_of(0.0, dist), 0.0)
        self.assertAlmostEqual(tg.percentile_of(5.0, dist), 100.0)
        self.assertAlmostEqual(tg.percentile_of(3.5, dist), 75.0)

    def test_an_empty_distribution_has_no_percentile(self):
        self.assertEqual(tg.percentile_of(1.0, []), 0.0)


class TestHysteresis(unittest.TestCase):
    """`exit_below` exists because of a measured cost, not a preference.

    Measured on BTC/ETH/SOL over 170 days: the symmetric rule round-trips 228
    times, and at 3.5bp a leg that is about 16 percentage points against a
    gross edge near 9. Turnover, not the signal, was the binding constraint -
    so the knob is tested like any other claim.
    """

    N = 700

    def _series(self):
        """A sawtooth that crosses the threshold repeatedly.

        A monotonic series cannot exercise hysteresis: it never leaves the
        regime, so both exit rules behave identically and the test would pass
        whether or not the setting was wired up at all.
        """
        closes = []
        price = 100.0
        for i in range(self.N):
            price += 0.6 if (i // 12) % 2 == 0 else -0.45
            closes.append(price)
        opens = [c - 0.2 for c in closes]
        volumes = [100.0 + (i % 30) for i in range(self.N)]
        return closes, opens, volumes

    def test_the_symmetric_default_is_unchanged(self):
        """`exit_below=None` must mean "exit below the entry threshold".

        It was the only behaviour before the knob existed, and every number
        reported for the rule was produced by it - so a silent change here
        would make the documented results unreproducible.
        """
        closes, opens, volumes = self._series()
        symmetric = tg.TrendGateConfig(min_bull_votes=3)
        explicit = tg.TrendGateConfig(min_bull_votes=3, exit_below=2)
        a = tg.simulate(closes, opens, volumes, symmetric, start_bar=500)
        b = tg.simulate(closes, opens, volumes, explicit, start_bar=500)
        self.assertEqual(a.net_return_pct, b.net_return_pct)
        self.assertEqual(len(a.trades), len(b.trades))

    def test_hysteresis_reduces_turnover_and_raises_the_average_hold(self):
        closes, opens, volumes = self._series()
        tight = tg.simulate(closes, opens, volumes,
                            tg.TrendGateConfig(min_bull_votes=3, exit_below=2),
                            start_bar=500)
        loose = tg.simulate(closes, opens, volumes,
                            tg.TrendGateConfig(min_bull_votes=3, exit_below=0),
                            start_bar=500)
        self.assertTrue(tight.trades and loose.trades)
        self.assertLess(len(loose.trades), len(tight.trades))
        self.assertGreater(loose.mean_bars, tight.mean_bars)

    def test_hysteresis_cannot_survive_an_absent_exit(self):
        """Equal thresholds would leave the rule with no reachable exit."""
        with self.assertRaises(tg.TrendGateError):
            tg.TrendGateConfig(min_bull_votes=3, exit_below=3)
        with self.assertRaises(tg.TrendGateError):
            tg.TrendGateConfig(min_bull_votes=2, exit_below=3)

    def test_exit_below_out_of_range_is_rejected(self):
        with self.assertRaises(tg.TrendGateError):
            tg.TrendGateConfig(min_bull_votes=3, exit_below=5)

    def test_exit_below_zero_is_allowed_and_means_all_four_bearish(self):
        """0 is a real setting: the position is held until every condition
        turns, which is the sticky version measured on the archive."""
        cfg = tg.TrendGateConfig(min_bull_votes=3, exit_below=0)
        self.assertEqual(cfg.closes_below, 0)


class TestCostSensitivity(unittest.TestCase):
    """Fees are large enough to be the whole story, so they are measured.

    228 round trips at 3.5bp a leg is ~16 percentage points over 170 days,
    against a gross edge of ~9. Any comparison that reports only the net number
    is reporting the fee schedule, not the signal.
    """

    N = 700

    def _series(self):
        closes = []
        price = 100.0
        for i in range(self.N):
            price += 0.6 if (i // 12) % 2 == 0 else -0.45
            closes.append(price)
        return closes, [c - 0.2 for c in closes], [100.0 + (i % 30) for i in range(self.N)]

    def test_zero_fees_is_gross_and_positive_fees_only_reduce_it(self):
        closes, opens, volumes = self._series()
        base = dict(min_bull_votes=3)
        gross = tg.simulate(closes, opens, volumes,
                            tg.TrendGateConfig(fee_bps=0.0, **base), start_bar=500)
        net = tg.simulate(closes, opens, volumes,
                          tg.TrendGateConfig(fee_bps=3.5, **base), start_bar=500)
        self.assertGreater(gross.net_return_pct, net.net_return_pct)

    def test_the_drag_is_about_two_legs_times_the_trade_count(self):
        """Pins the arithmetic, so a change to the cost model shows up as a
        number rather than as a slightly different backtest."""
        closes, opens, volumes = self._series()
        base = dict(min_bull_votes=3)
        gross = tg.simulate(closes, opens, volumes,
                            tg.TrendGateConfig(fee_bps=0.0, **base), start_bar=500)
        net = tg.simulate(closes, opens, volumes,
                          tg.TrendGateConfig(fee_bps=3.5, **base), start_bar=500)
        n = len(net.trades)
        self.assertGreater(n, 5)
        # Each round trip costs (1 - 0.00035)^2 - 1 in return terms, applied
        # multiplicatively; the simple approximation is 2 * 3.5bp * n.
        approx = n * 2 * 3.5 / 100.0
        self.assertAlmostEqual(
            (gross.net_return_pct - net.net_return_pct) / approx, 1.0, delta=0.75
        )


class TestFixedHold(unittest.TestCase):
    N = 700

    def test_the_cap_shortens_holding_relative_to_the_vote_exit(self):
        closes = ramp(self.N, step=1.0)
        opens = [c - 0.5 for c in closes]
        volumes = [100.0] * self.N
        base = tg.TrendGateConfig(min_bull_votes=1)
        free = tg.simulate(closes, opens, volumes, base, start_bar=500)
        capped = tg.fixed_hold_simulate(closes, opens, volumes, 3, base, start_bar=500)
        self.assertLess(capped.exposure, free.exposure)
        # The fill for the exit decision lands on the next bar, so a 3-bar hold
        # is 3 bars between fills, not 4.
        self.assertLessEqual(max(t.bars for t in capped.trades), 3)


class TestStopLoss(unittest.TestCase):
    """The stop is the one exit not driven by the rule's own flags.

    The fixture is a **flat** price series with a few bars whose *lows* pierce
    far below and whose closes never move. That shape is the whole design: it
    is the only construction in which "the stop watches the low" and "the stop
    watches the close" give different answers, so it can tell them apart. An
    earlier version of these tests used a falling series, where the close also
    breaches the level, and a mutation that switched the check to `closes`
    passed every test.

    `hold_ok` is left permanently true, so the rule never exits on its own and
    every exit in the result came from the stop.
    """

    N = 200
    ENTRY = 60
    PIERCED = (100, 110, 120)
    STOP = 0.05

    def _fixture(self):
        # Opens deliberately differ from closes. The stop level is anchored to
        # the *fill* (an open), so a fixture where opens == closes cannot tell
        # that apart from anchoring to the signal bar's close - and a mutation
        # doing exactly that passed an earlier version of these tests.
        closes = [100.0] * self.N
        opens = [99.0] * self.N
        highs = [100.1] * self.N
        lows = [99.9] * self.N
        entry = [None] * self.ENTRY + [True] * (self.N - self.ENTRY)
        hold = [None] * self.ENTRY + [True] * (self.N - self.ENTRY)
        return closes, opens, highs, lows, entry, hold

    def _pierced(self):
        _, _, _, lows, _, _ = self._fixture()
        for i in self.PIERCED:
            lows[i] = 90.0
        return lows

    def _run(self, **cfg_kw):
        closes, opens, highs, _, entry, hold = self._fixture()
        cfg = tg.TrendGateConfig(stop_loss_pct=self.STOP, **cfg_kw)
        return tg.simulate_flags(closes, opens, entry, hold, cfg,
                                 lows=self._pierced())

    # -- validation --------------------------------------------------------

    def test_it_needs_lows(self):
        closes, opens, _, _, entry, hold = self._fixture()
        with self.assertRaises(tg.TrendGateError):
            tg.simulate_flags(closes, opens, entry, hold,
                              tg.TrendGateConfig(stop_loss_pct=0.03))

    def test_a_mismatched_lows_length_is_rejected(self):
        closes, opens, _, _, entry, hold = self._fixture()
        with self.assertRaises(tg.TrendGateError):
            tg.simulate_flags(closes, opens, entry, hold,
                              tg.TrendGateConfig(stop_loss_pct=0.03),
                              lows=[100.0] * (self.N - 1))

    def test_bad_stop_values_are_rejected(self):
        for kw in ({"stop_loss_pct": -0.1}, {"stop_loss_pct": 1.0},
                   {"stop_cooldown_bars": -1},
                   {"stop_cooldown_bars": 5}):
            with self.subTest(kw=kw):
                with self.assertRaises(tg.TrendGateError):
                    tg.TrendGateConfig(**kw)

    # -- the off state -----------------------------------------------------

    def test_the_off_state_reproduces_the_unstopped_result(self):
        closes, opens, _, lows, entry, hold = self._fixture()
        without = tg.simulate_flags(closes, opens, entry, hold,
                                    tg.TrendGateConfig())
        explicit = tg.simulate_flags(closes, opens, entry, hold,
                                     tg.TrendGateConfig(stop_loss_pct=0.0),
                                     lows=lows)
        self.assertEqual(len(without.trades), len(explicit.trades))
        self.assertAlmostEqual(without.equity, explicit.equity, places=12)
        self.assertFalse(any(t.exit_reason == "stop" for t in explicit.trades))

    # -- the mechanism -----------------------------------------------------

    def test_a_stop_fires_on_the_low_even_when_the_close_never_moves(self):
        """The closes here are constant at 100 while the lows hit 90. A
        close-based stop would not fire at all, so this fails if the check ever
        moves off the low."""
        r = self._run()
        stops = [t for t in r.trades if t.exit_reason == "stop"]
        self.assertTrue(stops, "a pierced low did not trigger the stop")

    def test_the_stop_level_is_anchored_to_the_fill_price(self):
        """Anchoring to the signal bar's close instead would let the stop use a
        price from before the fill - look-ahead, and it moves the level."""
        r = self._run()
        for t in r.trades:
            if t.exit_reason != "stop":
                continue
            with self.subTest(entry=t.entry_bar):
                self.assertAlmostEqual(t.stop_level,
                                       t.entry_price * (1.0 - self.STOP), places=12)

    def test_a_stop_fills_exactly_one_bar_after_its_signal(self):
        """`>` would also pass if the exit were deferred by several bars, and
        deferring it is a plausible mistake - an earlier version of this
        assertion was too weak to notice."""
        r = self._run()
        stops = [t for t in r.trades if t.exit_reason == "stop"]
        self.assertTrue(stops)
        for t in stops:
            with self.subTest(signal=t.exit_signal_bar):
                self.assertEqual(t.exit_bar, t.exit_signal_bar + 1)

    def test_it_does_not_fill_at_the_stop_level_when_the_open_gaps(self):
        """The gap is the part a stop cannot protect against, so the simulator
        must not report the level as the fill."""
        closes, opens, highs, _, entry, hold = self._fixture()
        gapped = list(opens)
        for i in self.PIERCED:
            gapped[i + 1] = 90.0
        r = tg.simulate_flags(closes, gapped, entry, hold,
                              tg.TrendGateConfig(stop_loss_pct=self.STOP),
                              lows=self._pierced())
        stops = [t for t in r.trades if t.exit_reason == "stop"]
        self.assertTrue(stops)
        for t in stops:
            with self.subTest(exit=t.exit_bar):
                self.assertLess(t.exit_price, t.stop_level)

    def test_cooldown_actually_delays_re_entry(self):
        """`hold_ok` is permanently true, so without a cooldown the position is
        reopened on the next bar. Asserted on the *gap* rather than on the trade
        count: an earlier version used `assertLessEqual` on the count, which
        equality satisfies, so it held the mutation that removed the cooldown
        entirely.
        """
        cooled = self._run(stop_cooldown_bars=20)
        stops = [i for i, t in enumerate(cooled.trades)
                 if t.exit_reason == "stop"]
        self.assertTrue(stops)
        checked = 0
        for i in stops:
            if i + 1 >= len(cooled.trades):
                continue
            gap = cooled.trades[i + 1].entry_bar - cooled.trades[i].exit_bar
            with self.subTest(exit=cooled.trades[i].exit_bar):
                self.assertGreaterEqual(gap, 20 - 1)
            checked += 1
        self.assertTrue(checked, "no stop was followed by a re-entry to check")

    def test_a_cooldown_reduces_the_trade_count(self):
        """Strictly fewer, not fewer-or-equal."""
        base = self._run()
        cooled = self._run(stop_cooldown_bars=20)
        self.assertLess(len(cooled.trades), len(base.trades))

    def test_exit_reasons_are_always_recorded(self):
        for stop in (0.0, self.STOP):
            with self.subTest(stop=stop):
                if stop:
                    r = self._run()
                else:
                    closes, opens, _, lows, entry, hold = self._fixture()
                    r = tg.simulate_flags(closes, opens, entry, hold,
                                          tg.TrendGateConfig(), lows=lows)
                self.assertTrue(r.trades)
                for t in r.trades:
                    self.assertIn(t.exit_reason, ("hold", "cap", "stop", "markout"))


if __name__ == "__main__":
    unittest.main()
