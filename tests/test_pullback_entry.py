"""Tests for the pullback entry rule.

Each condition is tested by moving the parameter it owns, the same way
`test_trend_gate` does, because a combined boolean hides a mis-wired condition.
The three things the proposal left open - which EMA, cross-versus-level, and
whether the 3% band has a floor - are tested as *behaviours*, so a result table
cannot be produced without them being what they claim.

One structural fact drives most of the fixtures here and is asserted on its
own: on a plain linear ramp EMA(21) is never above SMA(20). A test built on a
ramp would therefore have measured the volume condition while claiming to
measure momentum.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.indicators import ema, sma  # noqa: E402
from backtest import pullback_entry as pb  # noqa: E402
from backtest import trend_gate as tg  # noqa: E402


def sawtooth(n: int = 700, up: float = 0.5, down: float = 0.4) -> list[float]:
    """A price series where EMA(21) sits above SMA(20) at the last bar.

    Picked by probe, not by argument. Two shapes were tried first and both
    failed, which is worth recording because the failures are informative:

    * a linear ramp puts EMA(21) *below* SMA(20) on every bar - the
      exponential weighting lags 10 bars against the simple average's 9.5, so
      the mid band is not a trend filter at all;
    * a step up followed by a plateau also puts EMA below, because the step has
      to land inside the last 20 bars or SMA(20) forgets it while EMA(21) does
      not.
    """
    out = []
    price = 100.0
    for i in range(n):
        price += up if (i // 15) % 2 == 0 else -down
        out.append(price)
    return out


def with_volume(closes: list[float], last: float | None = None) -> list[float]:
    """Flat volume, optionally spiked on the last bar.

    Flat means the volume condition FAILS (it needs strictly greater), which is
    what the negative tests want. Use `rising_volume` when it must pass.
    """
    out = [100.0] * len(closes)
    if last is not None:
        out[-1] = last
    return out


def rising_volume(n: int) -> list[float]:
    """Volume rising every bar, so it always exceeds its own trailing mean."""
    return [100.0 + i for i in range(n)]


class TestTheBandIsNotATrendFilter(unittest.TestCase):
    """Pins the fact the fixtures depend on.

    If a change made EMA(21) track SMA(20) on a ramp, every momentum test below
    would start measuring the band condition instead, and would keep passing.
    """

    def test_a_linear_ramp_never_puts_ema_above_the_mid(self):
        closes = [100.0 + i for i in range(700)]
        fast, mid = ema(closes, 21), sma(closes, 20)
        above = [
            i for i in range(700)
            if fast[i] is not None and mid[i] is not None and fast[i] > mid[i]
        ]
        self.assertEqual(above, [])

    def test_an_sawtooth_does(self):
        closes = sawtooth()
        fast, mid = ema(closes, 21), sma(closes, 20)
        self.assertIsNotNone(fast[-1])
        self.assertGreater(fast[-1], mid[-1])


class TestConfigValidation(unittest.TestCase):
    def test_a_band_with_no_room_is_rejected(self):
        with self.assertRaises(pb.PullbackConfigError):
            pb.PullbackConfig(min_momentum=0.05, max_momentum=0.03)

    def test_degenerate_periods_are_rejected(self):
        for kw in ({"ema_period": 1}, {"bollinger_period": 1},
                   {"volume_lookback": 0}):
            with self.subTest(kw=kw):
                with self.assertRaises(pb.PullbackConfigError):
                    pb.PullbackConfig(**kw)

    def test_a_negative_hold_cap_is_rejected(self):
        """`TrendGateError`, not `PullbackConfigError`.

        `hold_bars` lives on `ExecutionConfig` now and is validated there, so
        the error carries the origin of the check rather than being translated
        back into this module's type. A translation would be a compatibility
        shim around a relation that is uniform: one definition, one error.
        """
        with self.assertRaises(tg.TrendGateError):
            pb.PullbackConfig(hold_bars=-1)

    def test_a_zero_width_band_is_allowed(self):
        """0.0 == 0.0 is a real setting: "flat over 24h"."""
        cfg = pb.PullbackConfig(min_momentum=0.0, max_momentum=0.0)
        self.assertEqual(cfg.min_momentum, cfg.max_momentum)


class TestTheTwoCrossReadingsDifferInKind(unittest.TestCase):
    """EMA(21)-vs-SMA(20) and EMA(9)-vs-EMA(26) are not variations of each other.

    The first is an acceleration condition and the second is a trend condition,
    and the test that pins the difference is the linear ramp: EMA(21) sits below
    SMA(20) on every bar of one, while EMA(9) sits above EMA(26) throughout.
    That is why swapping them changes which side of the measured 24h reversal
    effect the rule bets on - and why the two were measured and reported apart
    rather than one replacing the other silently.
    """

    N = 700

    def test_ema9_is_above_ema26_throughout_a_linear_ramp(self):
        closes = [100.0 + i for i in range(self.N)]
        fast, slow = ema(closes, 9), ema(closes, 26)
        above = [
            i for i in range(self.N)
            if fast[i] is not None and slow[i] is not None and fast[i] > slow[i]
        ]
        self.assertGreater(len(above), self.N - 40)

    def test_ema21_is_never_above_sma20_on_the_same_ramp(self):
        closes = [100.0 + i for i in range(self.N)]
        fast, mid = ema(closes, 21), sma(closes, 20)
        above = [
            i for i in range(self.N)
            if fast[i] is not None and mid[i] is not None and fast[i] > mid[i]
        ]
        self.assertEqual(above, [])


class TestEmaPairConstructor(unittest.TestCase):
    def test_it_sets_the_documented_pair_and_band(self):
        cfg = pb.ema_pair()
        self.assertEqual(cfg.ema_period, 9)
        self.assertEqual(cfg.ema_slow_period, 26)
        self.assertEqual(cfg.cross_reference, "ema")
        self.assertEqual(cfg.min_momentum, 0.0)
        self.assertEqual(cfg.max_momentum, pb.MAX_MOMENTUM)

    def test_the_defaults_are_unchanged_so_earlier_numbers_reproduce(self):
        """Every result already reported used EMA(21) against the mid band;
        making the 9/26 pair the default would silently orphan them."""
        cfg = pb.PullbackConfig()
        self.assertEqual(cfg.ema_period, 21)
        self.assertEqual(cfg.cross_reference, "sma")
        self.assertIsNone(cfg.min_momentum)

    def test_a_fast_period_at_or_above_the_slow_one_is_rejected(self):
        for fast, slow in ((26, 26), (30, 26)):
            with self.subTest(fast=fast, slow=slow):
                with self.assertRaises(pb.PullbackConfigError):
                    pb.PullbackConfig(cross_reference="ema", ema_period=fast,
                                      ema_slow_period=slow)

    def test_an_unknown_reference_is_rejected(self):
        with self.assertRaises(pb.PullbackConfigError):
            pb.PullbackConfig(cross_reference="vwap")

    def test_the_volume_condition_can_be_switched_off(self):
        """A later proposal dropped it without saying so; choosing for the
        caller would make the results unattributable."""
        self.assertTrue(pb.PullbackConfig().require_volume)
        self.assertFalse(pb.ema_pair(require_volume=False).require_volume)


class TestVolumeCanBeDropped(unittest.TestCase):
    N = 700

    def _flags(self, **kw):
        closes = sawtooth(self.N)
        # Flat volume, so the condition fails whenever it is applied.
        return pb.entry_flags(closes, with_volume(closes),
                              pb.PullbackConfig(require_cross=False, **kw))[0]

    def test_flat_volume_blocks_the_entry_when_required(self):
        self.assertFalse(self._flags(require_volume=True)[-1])

    def test_the_same_bars_pass_once_it_is_not_required(self):
        self.assertTrue(self._flags(require_volume=False)[-1])


class TestImpulseAndPullbackContext(unittest.TestCase):
    """The chart-derived context: a rally, then a return to the averages.

    Disabled by default, so every number already reported is unchanged. The
    conditions are tested for being applied at all; the measurement that says
    they are too rare to evaluate is in RESEARCH.md section 2.13 and is not
    something a unit test can assert.
    """

    N = 700

    def _series(self):
        """Flat, then a sharp rally over the last 48 bars, ending at the high.

        A flat series will not do for the "condition off" case: on a flat one
        EMA(9) equals EMA(26), the band condition fails, and the test would be
        asserting something about the band rather than about the context.
        """
        closes = [100.0] * (self.N - 48) + [100.0 + 10.0 * k for k in range(48)]
        return closes, rising_volume(self.N)

    def _flags(self, **kw):
        """Built from `ema_pair`, and with `max_momentum` opened wide.

        Both adjustments are needed and both were found by the fixture failing:

        * the default reference is EMA(21)-against-SMA(20), and on the steady
          rise this fixture needs EMA(21) sits *below* SMA(20) - the structural
          fact recorded in `TestTheTwoCrossReadingsDifferInKind`. The condition
          never fires, so the test would have measured that instead;
        * the rally is sharp enough that its own 24h change breaches the 3% cap,
          which would mask the condition under test.

        Neutralising the other conditions by parameter rather than by reshaping
        the prices is the same isolation move used for the band's boundaries.
        """
        closes, volumes = self._series()
        kw.setdefault("max_momentum", 5.0)
        cfg = pb.ema_pair(require_cross=False, **kw)
        return pb.entry_flags(closes, volumes, cfg)[0]

    def test_disabled_by_default(self):
        cfg = pb.PullbackConfig()
        self.assertEqual(cfg.impulse_lookback, 0)
        self.assertEqual(cfg.min_impulse, 0.0)
        self.assertEqual(cfg.min_pullback, 0.0)
        self.assertIsNone(cfg.max_pullback)

    def test_the_rally_itself_passes_the_impulse_test(self):
        self.assertTrue(self._flags(impulse_lookback=48, min_impulse=0.05)[-1])

    def test_an_unreachable_impulse_requirement_blocks_it(self):
        # Parameter isolation: same bars, a bar the rally cannot clear.
        self.assertFalse(self._flags(impulse_lookback=48, min_impulse=10.0)[-1])

    def test_a_close_at_the_high_fails_a_required_pullback(self):
        """The point of the condition: do not enter at the top of the move."""
        self.assertTrue(self._flags(impulse_lookback=48, min_impulse=0.05,
                                    min_pullback=0.0)[-1])
        self.assertFalse(self._flags(impulse_lookback=48, min_impulse=0.05,
                                     min_pullback=0.01)[-1])

    def test_an_impossible_pullback_window_is_rejected(self):
        with self.assertRaises(pb.PullbackConfigError):
            pb.PullbackConfig(min_pullback=0.05, max_pullback=0.01)

    def test_negative_sizes_are_rejected(self):
        for kw in ({"impulse_lookback": -1}, {"min_impulse": -0.01},
                   {"min_pullback": -0.01}):
            with self.subTest(kw=kw):
                with self.assertRaises(pb.PullbackConfigError):
                    pb.PullbackConfig(**kw)

    def test_the_context_is_named_in_the_summary(self):
        cfg = pb.PullbackConfig(impulse_lookback=48, min_impulse=0.05,
                                min_pullback=0.01, max_pullback=0.05)
        text = pb.describe(cfg)
        self.assertIn("impulse", text)
        self.assertIn("pullback", text)


class TestMacdLines(unittest.TestCase):
    """MACD is composed here rather than reused, so it is tested here.

    `indicators.ema` takes plain floats; the MACD line carries a None prefix, so
    the signal line is an EMA over the *defined region*, mapped back. Getting
    that mapping wrong would shift the signal by the warm-up length and look
    like a working indicator.
    """

    N = 700

    def test_it_equals_the_difference_of_the_two_emas(self):
        closes = [100.0 + math.sin(i * 0.05) * 5 for i in range(self.N)]
        line, _ = pb.macd_lines(closes)
        ef, es = ema(closes, 12), ema(closes, 26)
        for i in (300, 500, self.N - 1):
            with self.subTest(i=i):
                self.assertAlmostEqual(line[i], ef[i] - es[i], places=9)

    def test_the_warmup_is_none_in_both_lines(self):
        closes = [100.0 + i * 0.1 for i in range(self.N)]
        line, sig = pb.macd_lines(closes)
        self.assertTrue(all(v is None for v in line[:24]))
        self.assertIsNotNone(line[25])
        # The signal lags the line further, by its own period.
        self.assertTrue(all(v is None for v in sig[:32]))
        self.assertIsNotNone(sig[34])

    def test_a_linear_trend_leaves_the_line_equal_to_its_signal(self):
        """Worth pinning: on a *steady* trend MACD is constant, so it equals its
        own signal to within floating-point noise. A test asserting "> on a
        rising series" would be asserting an accident of the fixture - the
        condition only separates when the trend accelerates or decelerates.
        """
        closing = [100.0 + i * 0.5 for i in range(self.N)]
        line, sig = pb.macd_lines(closing)
        self.assertAlmostEqual(line[-1], sig[-1], places=9)

    def test_an_accelerating_rise_puts_the_line_above_its_signal(self):
        closes = [100.0 + 0.001 * i * i for i in range(self.N)]
        line, sig = pb.macd_lines(closes)
        self.assertGreater(line[-1], sig[-1])

    def test_an_accelerating_fall_puts_it_below(self):
        closes = [700.0 - 0.001 * i * i for i in range(self.N)]
        line, sig = pb.macd_lines(closes)
        self.assertLess(line[-1], sig[-1])

    def test_bad_periods_are_rejected(self):
        with self.assertRaises(pb.PullbackConfigError):
            pb.macd_lines([1.0] * 50, fast=26, slow=12)


class TestCrossSubject(unittest.TestCase):
    """Whether the close or the moving average does the crossing.

    Both charts show the *close* reclaiming the mid band while EMA(9) is still
    below EMA(26). Encoding that with the EMA as the subject would be a
    different, later trade.
    """

    N = 700

    def _flags(self, **kw):
        closes = sawtooth(self.N)
        cfg = pb.PullbackConfig(require_cross=True, max_momentum=5.0, **kw)
        return pb.entry_flags(closes, rising_volume(self.N), cfg)[0]

    def test_the_two_subjects_produce_different_signals(self):
        """Compared as sets, not counts: equal counts with different members
        would pass a count comparison while the rule is a different rule."""
        by_price = {i for i, v in enumerate(self._flags(cross_subject="price")) if v}
        by_ema = {i for i, v in enumerate(self._flags(cross_subject="ema")) if v}
        self.assertTrue(by_price)
        self.assertTrue(by_ema)
        self.assertNotEqual(by_price, by_ema)

    def test_an_unknown_subject_is_rejected(self):
        with self.assertRaises(pb.PullbackConfigError):
            pb.PullbackConfig(cross_subject="vwap")

    def test_requiring_a_lagging_cross_can_only_remove_signals(self):
        """It asserts the golden cross has NOT happened, so the trade is taken
        before it - which can only make the rule stricter."""
        closes = sawtooth(self.N)
        volumes = rising_volume(self.N)
        off = pb.entry_flags(closes, volumes, pb.PullbackConfig(
            require_cross=True, max_momentum=5.0,
            require_lagging_cross=False))[0]
        on = pb.entry_flags(closes, volumes, pb.PullbackConfig(
            require_cross=True, max_momentum=5.0,
            require_lagging_cross=True))[0]
        fired_off = {i for i, v in enumerate(off) if v}
        fired_on = {i for i, v in enumerate(on) if v}
        self.assertTrue(fired_on <= fired_off, "the stricter rule fired extra bars")

    def test_requiring_macd_can_only_remove_signals(self):
        closes = sawtooth(self.N)
        volumes = rising_volume(self.N)
        off = pb.entry_flags(closes, volumes, pb.PullbackConfig(
            require_cross=True, max_momentum=5.0, require_macd=False))[0]
        on = pb.entry_flags(closes, volumes, pb.PullbackConfig(
            require_cross=True, max_momentum=5.0, require_macd=True))[0]
        fired_off = {i for i, v in enumerate(off) if v}
        fired_on = {i for i, v in enumerate(on) if v}
        self.assertTrue(fired_on <= fired_off)


class TestDescriptionNamesTheReference(unittest.TestCase):
    def test_the_two_references_render_differently(self):
        mid = pb.describe(pb.PullbackConfig())
        pair = pb.describe(pb.ema_pair())
        self.assertIn("EMA21 vs SMA20", mid)
        self.assertIn("EMA9 vs EMA26", pair)
        self.assertIn("no vol", pb.describe(pb.ema_pair(require_volume=False)))


class TestMomentumBandBoundaries(unittest.TestCase):
    """The 3% cap is the point of the rule, so its edges are pinned.

    The 24h change is imposed by moving the bar 24 back rather than the last
    close: the band and the EMAs are driven by the last 20 bars, and bar -25
    carries under 1% of EMA(21)'s weight, so they are left alone.
    """

    N = 700

    def _series(self):
        """A sawtooth, picked because the band condition holds at its last bar.

        Imposing a chosen 24h change on a synthetic series did not work: the
        band condition needs recent bars above the 20-bar mean, and every shape
        that achieves that drags the 24h change along with it. So the series is
        fixed and the threshold is set to the series' own measured change.
        """
        closes = []
        price = 100.0
        for i in range(self.N):
            price += 0.5 if (i // 15) % 2 == 0 else -0.4
            closes.append(price)
        return closes, with_volume(closes, last=1e9)

    def _change(self) -> float:
        closes, _ = self._series()
        return closes[-1] / closes[-25] - 1.0

    def _entry(self, **kw):
        closes, volumes = self._series()
        entry, hold = pb.entry_flags(
            closes, volumes, pb.PullbackConfig(require_cross=False, **kw)
        )
        # Guard the premise, so a change to the shape cannot quietly turn these
        # into tests of the volume condition instead of the band.
        self.assertTrue(hold[-1], "premise broken: price is not above the mid")
        return entry[-1]

    def test_the_upper_boundary_is_inclusive(self):
        """The proposal says "within 3%", so the value itself must pass.

        `<=` against `<` is invisible in a boolean and would bias the rule
        slightly tighter than specified.
        """
        c = self._change()
        self.assertTrue(self._entry(max_momentum=c))
        self.assertFalse(self._entry(max_momentum=c - 1e-9))

    def test_a_cap_below_the_move_rejects_it(self):
        # Subtracting, not halving: the fixture's 24h change is negative
        # (-0.43%), so c/2 is *above* c and would not be a tighter cap.
        c = self._change()
        self.assertLess(c, 0.0)  # the fixture is a falling 24h, as measured
        self.assertFalse(self._entry(max_momentum=c - 0.001))

    def test_no_floor_admits_a_fall(self):
        """The literal reading: "within 3%" caps the gain and says nothing
        about a drop, so a falling market still passes."""
        c = self._change()
        self.assertTrue(self._entry(max_momentum=c + 1.0, min_momentum=c - 1.0))

    def test_a_floor_excludes_a_fall(self):
        """The other reading, and the one that refuses a falling knife."""
        c = self._change()
        self.assertFalse(self._entry(max_momentum=c + 1.0, min_momentum=c + 1e-9))

    def test_the_floor_is_lower_bound_inclusive(self):
        c = self._change()
        self.assertTrue(self._entry(max_momentum=c + 1.0, min_momentum=c))


class TestCrossVersusLevel(unittest.TestCase):
    """The proposal says "crosses from below", which is an event.

    Not cosmetic: as an event the signal fires once per crossing, as a level it
    fires on every bar above the mid - measured on the archive, roughly twice
    as many entries.
    """

    N = 700

    def _closes(self):
        closes = sawtooth(self.N)
        return closes

    def test_the_event_fires_far_less_often_than_the_level(self):
        """Compared against each other rather than against a fixed count.

        An absolute threshold would be a fact about the fixture; the claim
        being tested is that the event is the rarer reading, which is what
        makes the ambiguity worth reporting rather than picking silently.
        """
        closes = self._closes()
        volumes = rising_volume(self.N)
        event, _ = pb.entry_flags(closes, volumes,
                                  pb.PullbackConfig(require_cross=True))
        level, _ = pb.entry_flags(closes, volumes,
                                  pb.PullbackConfig(require_cross=False))
        n_event = sum(1 for v in event if v)
        n_level = sum(1 for v in level if v)
        self.assertGreater(n_event, 0)
        self.assertGreater(n_level, n_event)

    def test_the_level_fires_on_many_bars(self):
        closes = self._closes()
        volumes = rising_volume(self.N)
        entry, _ = pb.entry_flags(closes, volumes,
                                  pb.PullbackConfig(require_cross=False))
        self.assertGreater(sum(1 for v in entry if v), 5)

    def test_an_already_high_ema_does_not_trigger_the_event(self):
        """The event needs the previous bar at or below the mid; a series that
        is above for its whole measurable span fires nothing."""
        # Steady acceleration from bar 0: EMA is above the mid from early on
        # and never crosses after that.
        closes = [100.0 + (i ** 2) * 0.01 for i in range(self.N)]
        volumes = with_volume(closes, last=1e9)
        entry, _ = pb.entry_flags(closes, volumes,
                                  pb.PullbackConfig(require_cross=True))
        warm = [v for v in entry if v is not None]
        self.assertTrue(warm)
        self.assertFalse(any(warm))


class TestVolumeCondition(unittest.TestCase):
    N = 700

    def _closes(self):
        return sawtooth(self.N)

    def test_volume_above_the_mean_is_required(self):
        closes = self._closes()
        cfg = pb.PullbackConfig(require_cross=False)
        quiet, _ = pb.entry_flags(closes, with_volume(closes, last=50.0), cfg)
        busy, _ = pb.entry_flags(closes, with_volume(closes, last=200.0), cfg)
        self.assertFalse(quiet[-1])
        self.assertTrue(busy[-1])

    def test_volume_equal_to_the_mean_is_not_above_it(self):
        closes = self._closes()
        entry, _ = pb.entry_flags(closes, with_volume(closes),
                                  pb.PullbackConfig(require_cross=False))
        self.assertFalse(entry[-1])


class TestWarmup(unittest.TestCase):
    N = 700

    def test_warmup_bars_are_none_not_false(self):
        """False would be indistinguishable from "the conditions are not met",
        and the first 480 bars would read as a rule that keeps declining to
        trade rather than as one that cannot be evaluated yet."""
        closes = sawtooth(self.N)
        volumes = with_volume(closes)
        entry, hold = pb.entry_flags(closes, volumes, pb.PullbackConfig())
        self.assertTrue(all(v is None for v in entry[:479]))
        self.assertIsNotNone(entry[479])
        self.assertTrue(all(v is None for v in hold[:479]))

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(pb.PullbackConfigError):
            pb.entry_flags([1.0] * 10, [1.0] * 9)

    def test_no_bars_is_rejected(self):
        with self.assertRaises(pb.PullbackConfigError):
            pb.entry_flags([], [])


class TestExitConditionActuallyBites(unittest.TestCase):
    """A mutation made `hold_ok` always True and no test noticed.

    That mutation means the position is never closed by the rule - only by the
    hold cap or the end of the sample - which is a different strategy. The
    tests below exist because the first set of them did not cover it.
    """

    N = 700

    def _run(self, hold_above_mid: bool):
        closes = sawtooth(self.N)
        opens = [c - 0.2 for c in closes]
        return pb.simulate_pullback(
            closes, opens, rising_volume(self.N),
            pb.PullbackConfig(require_cross=False, hold_above_mid=hold_above_mid),
            start_bar=500,
        )

    def test_holding_to_the_end_differs_from_exiting_on_the_band(self):
        exiting = self._run(hold_above_mid=True)
        holding = self._run(hold_above_mid=False)
        self.assertTrue(exiting.trades)
        # Exiting on the band must produce more, shorter trades; holding must
        # produce fewer, longer ones.
        self.assertGreater(len(exiting.trades), len(holding.trades))
        self.assertLess(exiting.mean_bars, holding.mean_bars)

    def test_the_exit_condition_is_the_mid_band_not_something_else(self):
        """When holding to the end is disabled the rule must still leave the
        market at some point before the sample ends."""
        r = self._run(hold_above_mid=True)
        self.assertTrue(
            any(t.exit_bar < self.N - 1 for t in r.trades),
            "no trade closed before the end of the sample",
        )


class TestSimulation(unittest.TestCase):
    N = 700

    def _series(self):
        closes = []
        price = 100.0
        for i in range(self.N):
            price += 0.5 if (i // 15) % 2 == 0 else -0.4
            closes.append(price)
        return closes, [c - 0.2 for c in closes], rising_volume(self.N)

    def test_exits_are_still_delayed_past_their_signal(self):
        """The shared simulator carries this invariant; reusing it is the
        reason this rule gets the guarantee for free."""
        closes, opens, volumes = self._series()
        r = pb.simulate_pullback(closes, opens, volumes,
                                 pb.PullbackConfig(require_cross=False),
                                 start_bar=500)
        self.assertTrue(r.trades)
        for t in r.trades:
            with self.subTest(entry=t.entry_bar):
                self.assertGreater(t.entry_bar, t.entry_signal_bar)
                self.assertGreater(t.exit_bar, t.exit_signal_bar)

    def test_a_hold_cap_shortens_positions(self):
        closes, opens, volumes = self._series()
        free = pb.simulate_pullback(closes, opens, volumes,
                                    pb.PullbackConfig(require_cross=False),
                                    start_bar=500)
        capped = pb.simulate_pullback(
            closes, opens, volumes,
            pb.PullbackConfig(require_cross=False, hold_bars=5), start_bar=500)
        self.assertLess(capped.exposure, free.exposure)

    def test_the_stop_actually_reaches_the_shared_simulator(self):
        """Pins the gap this rule used to have.

        `PullbackConfig` restated only three execution fields and copied them
        into a `TrendGateConfig`, so `stop_loss_pct` - implemented and tested in
        `simulate_flags` since RESEARCH.md section 2.16 - could not be set from
        this rule at all. The old version of this test asserted that a helper
        copied three fields, which is why it did not notice.

        Asserts a stop *fires*, not that a field is present: a config can carry
        a value all the way to the loop and still be ignored there.

        `hold_above_mid=False` removes the rule's own exit so the stop is the
        only thing that can close a position. Without that the rule exits on the
        first close back below its reference, which on this series happens
        before any decline reaches 1% - the first version of this test did that
        and read "no stop fired" as the stop being unreachable, when the stop
        had simply never been given a chance.
        """
        closes, opens, volumes = self._series()
        lows = [c - 0.2 for c in closes]
        plain = pb.simulate_pullback(
            closes, opens, volumes,
            pb.PullbackConfig(require_cross=False, hold_above_mid=False),
            start_bar=500, lows=lows)
        stopped = pb.simulate_pullback(
            closes, opens, volumes,
            pb.PullbackConfig(require_cross=False, hold_above_mid=False,
                              stop_loss_pct=0.01),
            start_bar=500, lows=lows)
        self.assertTrue(stopped.trades)
        self.assertTrue(
            any(t.exit_reason == "stop" for t in stopped.trades),
            "a 1% stop did not fire with the rule's own exit disabled",
        )
        self.assertFalse(any(t.exit_reason == "stop" for t in plain.trades))

    def test_a_stop_without_lows_is_refused(self):
        """The coupling is checked in one place rather than at every call site:
        `simulate_flags` rejects a stop it has no lows to evaluate, so a caller
        cannot get a silent no-op stop."""
        closes, opens, volumes = self._series()
        with self.assertRaises(tg.TrendGateError):
            pb.simulate_pullback(
                closes, opens, volumes,
                pb.PullbackConfig(require_cross=False, stop_loss_pct=0.01),
                start_bar=500)

    def test_fees_reduce_the_result(self):
        closes, opens, volumes = self._series()
        gross = pb.simulate_pullback(
            closes, opens, volumes,
            pb.PullbackConfig(require_cross=False, fee_bps=0.0), start_bar=500)
        net = pb.simulate_pullback(
            closes, opens, volumes,
            pb.PullbackConfig(require_cross=False, fee_bps=3.5), start_bar=500)
        self.assertGreater(gross.net_return_pct, net.net_return_pct)


class TestDescription(unittest.TestCase):
    def test_the_summary_states_which_reading_it_describes(self):
        cross = pb.describe(pb.PullbackConfig(require_cross=True))
        level = pb.describe(pb.PullbackConfig(require_cross=False))
        self.assertIn("cross", cross)
        self.assertIn("level", level)
        self.assertNotEqual(cross, level)

    def test_the_band_renders_with_and_without_a_floor(self):
        self.assertIn("<=", pb.describe(pb.PullbackConfig(min_momentum=None)))
        self.assertIn("..", pb.describe(pb.PullbackConfig(min_momentum=0.0)))


if __name__ == "__main__":
    unittest.main()
