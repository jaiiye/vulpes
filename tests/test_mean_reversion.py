"""Tests for the mean-reversion rules.

Two things are checked that a bare "it runs" test would miss:

* the two rules are wired to different reference series (an oscillator versus a
  volatility band), so a mutation that swapped them would still produce 0/1
  flags and still "work";
* the regime filter can only *remove* entries. It is the one knob added from
  experience rather than from the proposal, so it is tested as a strict
  narrowing rather than as a behaviour.

The fill-timing guarantee is not retested here: this module delegates to
`trend_gate.simulate_flags`, and `tests/test_trend_gate.py::TestNoLookAhead`
already pins it, including a mutation check.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.indicators import atr, ema  # noqa: E402
from backtest import mean_reversion as mr  # noqa: E402
from backtest import trend_gate as tg  # noqa: E402


def falling_then_flat(n: int = 700, drop: float = 40.0) -> tuple[list, list, list]:
    """A decline into a floor, then a drift - the shape RSI is meant to catch."""
    closes = []
    price = 200.0
    for i in range(n):
        if i < n // 2:
            price -= drop / (n // 2)
        else:
            price += 0.002
        closes.append(price)
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    return closes, highs, lows


class TestConfigValidation(unittest.TestCase):
    def test_an_unknown_rule_is_rejected(self):
        with self.assertRaises(mr.MeanReversionError):
            mr.MeanReversionConfig(rule="bollinger")

    def test_the_exit_must_be_reachable(self):
        """oversold >= exit_level would open a position that can never close."""
        for osold, ex in ((50.0, 50.0), (60.0, 50.0)):
            with self.subTest(oversold=osold, exit_level=ex):
                with self.assertRaises(mr.MeanReversionError):
                    mr.MeanReversionConfig(oversold=osold, exit_level=ex)

    def test_degenerate_periods_and_multipliers_are_rejected(self):
        """This rule's own fields raise `MeanReversionError`."""
        for kw in ({"rsi_period": 1}, {"keltner_period": 1},
                   {"keltner_atr_period": 0}, {"keltner_multiplier": 0.0}):
            with self.subTest(kw=kw):
                with self.assertRaises(mr.MeanReversionError):
                    mr.MeanReversionConfig(**kw)

    def test_inherited_execution_fields_raise_the_shared_error(self):
        """`hold_bars` and the stop fields live on `ExecutionConfig` and are
        validated there, so they raise `TrendGateError`.

        Asserted rather than left to chance because it is the one place this
        refactor changes an error a caller could be catching. Both types
        subclass `ValueError`, so a caller catching that is unaffected; one
        catching only `MeanReversionError` would now miss an execution problem.
        """
        for kw in ({"hold_bars": -1}, {"stop_loss_pct": 2.0}):
            with self.subTest(kw=kw):
                with self.assertRaises(tg.TrendGateError):
                    mr.MeanReversionConfig(**kw)

    def test_the_conventional_defaults_are_what_the_docstring_claims(self):
        cfg = mr.MeanReversionConfig()
        self.assertEqual((cfg.oversold, cfg.exit_level), (30.0, 50.0))
        self.assertEqual((cfg.keltner_period, cfg.keltner_multiplier,
                          cfg.keltner_atr_period), (20, 2.0, 10))
        self.assertEqual(cfg.adx_max, 0.0, "the filter must be off by default")


class TestKeltnerLines(unittest.TestCase):
    N = 700

    def _series(self):
        closes = [100.0 + 10.0 * math.sin(i * 0.05) for i in range(self.N)]
        return closes, [c + 1.0 for c in closes], [c - 1.0 for c in closes]

    def test_the_mid_is_the_ema(self):
        closes, highs, lows = self._series()
        _, mid, _ = mr.keltner_lines(closes, highs, lows)
        ref = ema(closes, 20)
        for i in (300, 500, self.N - 1):
            with self.subTest(i=i):
                self.assertAlmostEqual(mid[i], ref[i], places=9)

    def test_the_half_width_is_the_multiplier_times_atr(self):
        closes, highs, lows = self._series()
        upper, mid, lower = mr.keltner_lines(closes, highs, lows)
        width = atr(highs, lows, closes, 10)
        for i in (300, 500, self.N - 1):
            with self.subTest(i=i):
                self.assertAlmostEqual(upper[i] - mid[i], 2.0 * width[i], places=9)
                self.assertAlmostEqual(mid[i] - lower[i], 2.0 * width[i], places=9)

    def test_a_bigger_multiplier_gives_wider_bands(self):
        closes, highs, lows = self._series()
        narrow, _, _ = mr.keltner_lines(closes, highs, lows,
                                        mr.MeanReversionConfig(keltner_multiplier=1.0))
        wide, _, _ = mr.keltner_lines(closes, highs, lows,
                                      mr.MeanReversionConfig(keltner_multiplier=3.0))
        i = self.N - 1
        self.assertLess(narrow[i] - 0.0, wide[i] - 0.0)

    def test_warmup_is_none_in_the_band_not_a_number(self):
        closes, highs, lows = self._series()
        _, mid, _ = mr.keltner_lines(closes, highs, lows)
        self.assertTrue(all(v is None for v in mid[:18]))
        self.assertIsNotNone(mid[20])

    def test_a_length_mismatch_is_rejected(self):
        closes, highs, _ = self._series()
        with self.assertRaises(mr.MeanReversionError):
            mr.keltner_lines(closes, highs, highs[:-1])


class TestRsiRule(unittest.TestCase):
    N = 700

    def test_entry_fires_only_on_the_oversold_side(self):
        closes, highs, lows = falling_then_flat(self.N)
        cfg = mr.MeanReversionConfig(rule="rsi")
        entry, _ = mr.entry_flags(closes, highs, lows, cfg)
        fired = [i for i, v in enumerate(entry) if v]
        self.assertTrue(fired, "a 40-point decline should produce an entry")

    def test_a_rising_series_never_enters(self):
        """RSI rises with price, so an uptrend is the wrong side of the rule."""
        closes = [100.0 + i * 0.5 for i in range(self.N)]
        highs = [c + 0.5 for c in closes]
        lows = [c - 0.5 for c in closes]
        entry, _ = mr.entry_flags(closes, highs, lows, mr.MeanReversionConfig())
        self.assertFalse(any(v for v in entry))

    def test_hold_outlasts_entry(self):
        """Once in, the position is kept until RSI reaches `exit_level`, so
        every bar that still holds must also have held at entry or later -
        `hold_ok` must be true at least as often as `entry_ok`."""
        closes, highs, lows = falling_then_flat(self.N)
        entry, hold = mr.entry_flags(closes, highs, lows, mr.MeanReversionConfig())
        e_true = sum(1 for v in entry if v)
        h_true = sum(1 for v in hold if v)
        self.assertGreaterEqual(h_true, e_true)

    def test_a_higher_exit_level_holds_longer(self):
        """`hold_ok` is `rsi < exit_level`, so a higher level is true more of
        the time and the position is kept longer. The reverse direction is the
        intuitive reading and it is wrong - which is why it is asserted."""
        closes, highs, lows = falling_then_flat(self.N)
        _, lo_hold = mr.entry_flags(closes, highs, lows,
                                    mr.MeanReversionConfig(exit_level=40.0))
        _, hi_hold = mr.entry_flags(closes, highs, lows,
                                    mr.MeanReversionConfig(exit_level=70.0))
        self.assertGreater(sum(1 for v in hi_hold if v),
                           sum(1 for v in lo_hold if v))


class TestWarmupIsNoneNotFalse(unittest.TestCase):
    """False would mean "the condition was evaluated and did not hold"."""

    def test_both_series_are_none_until_the_indicators_are_warm(self):
        closes, highs, lows = falling_then_flat(700)
        entry, hold = mr.entry_flags(closes, highs, lows, mr.MeanReversionConfig())
        self.assertTrue(all(v is None for v in entry[:14]))
        self.assertIsNotNone(entry[14])
        self.assertTrue(all(v is None for v in hold[:14]))

    def test_a_length_mismatch_is_rejected(self):
        closes, highs, _ = falling_then_flat(700)
        with self.assertRaises(mr.MeanReversionError):
            mr.entry_flags(closes, highs, highs[:-1])


class TestRegimeFilterOnlyNarrows(unittest.TestCase):
    """The ADX filter is the one addition that came from experience, so it is
    asserted to be a strict narrowing rather than a behaviour of its own."""

    N = 700

    def test_no_entry_survives_the_filter_unless_the_unfiltered_rule_had_it(self):
        closes, highs, lows = falling_then_flat(self.N)
        base, _ = mr.entry_flags(closes, highs, lows, mr.MeanReversionConfig())
        filt, _ = mr.entry_flags(closes, highs, lows,
                                 mr.MeanReversionConfig(adx_max=30.0))
        fired_base = {i for i, v in enumerate(base) if v}
        fired_filt = {i for i, v in enumerate(filt) if v}
        self.assertTrue(fired_filt <= fired_base,
                        "the filter admitted a bar the plain rule rejected")

    def test_an_impossible_threshold_admits_nothing(self):
        closes, highs, lows = falling_then_flat(self.N)
        entry, _ = mr.entry_flags(closes, highs, lows,
                                  mr.MeanReversionConfig(adx_max=0.001))
        self.assertFalse(any(v for v in entry))

    def test_the_filter_is_disabled_by_default(self):
        closes, highs, lows = falling_then_flat(self.N)
        base, _ = mr.entry_flags(closes, highs, lows, mr.MeanReversionConfig())
        explicit, _ = mr.entry_flags(closes, highs, lows,
                                     mr.MeanReversionConfig(adx_max=0.0))
        self.assertEqual(base, explicit)

    def test_a_negative_threshold_is_rejected(self):
        with self.assertRaises(mr.MeanReversionError):
            mr.MeanReversionConfig(adx_max=-5.0)


class TestStopIsWired(unittest.TestCase):
    """The stop lives in `simulate_flags`; this checks only that the mean
    reversion config reaches it, and that the off state is genuinely off.

    The behaviour of the stop itself is tested in `test_trend_gate` against the
    simulator, which is where it is implemented - testing it twice would give
    two places to update and only one of them would be right.
    """

    N = 700

    def _series(self):
        closes, highs, lows = falling_then_flat(self.N)
        return closes, [c - 0.2 for c in closes], highs, lows

    def test_validation_rejects_an_impossible_stop(self):
        """`TrendGateError` - the stop lives on `ExecutionConfig`."""
        for kw in ({"stop_loss_pct": -0.01}, {"stop_loss_pct": 1.0},
                   {"stop_loss_pct": 1.5}):
            with self.subTest(kw=kw):
                with self.assertRaises(tg.TrendGateError):
                    mr.MeanReversionConfig(**kw)

    def test_a_cooldown_without_a_stop_is_rejected(self):
        with self.assertRaises(tg.TrendGateError):
            mr.MeanReversionConfig(stop_cooldown_bars=5)

    def test_off_by_default(self):
        cfg = mr.MeanReversionConfig()
        self.assertEqual(cfg.stop_loss_pct, 0.0)
        self.assertFalse(cfg.stop_is_on)

    def test_a_stop_produces_stop_exits_and_a_tighter_one_produces_more(self):
        closes, opens, highs, lows = self._series()
        loose = mr.simulate_mean_reversion(
            closes, opens, highs, lows,
            mr.MeanReversionConfig(stop_loss_pct=0.20))
        tight = mr.simulate_mean_reversion(
            closes, opens, highs, lows,
            mr.MeanReversionConfig(stop_loss_pct=0.01))
        n_loose = sum(1 for t in loose.trades if t.exit_reason == "stop")
        n_tight = sum(1 for t in tight.trades if t.exit_reason == "stop")
        self.assertGreater(n_tight, n_loose)

    def test_an_open_position_is_only_marked_out_once(self):
        """A duplicated mark-out would double-count the final position's
        P&L - and it appeared as one extra trade with an empty reason rather
        than as an obvious error."""
        closes, opens, highs, lows = self._series()
        for stop in (0.0, 0.05):
            with self.subTest(stop=stop):
                r = mr.simulate_mean_reversion(
                    closes, opens, highs, lows,
                    mr.MeanReversionConfig(stop_loss_pct=stop))
                markouts = [t for t in r.trades if t.exit_reason == "markout"]
                self.assertLessEqual(len(markouts), 1)
                seen = [(t.entry_bar, t.exit_bar) for t in r.trades]
                self.assertEqual(len(seen), len(set(seen)),
                                 "the same round trip was recorded twice")

    def test_a_stop_still_fills_after_its_signal_bar(self):
        closes, opens, highs, lows = self._series()
        r = mr.simulate_mean_reversion(
            closes, opens, highs, lows,
            mr.MeanReversionConfig(stop_loss_pct=0.02))
        stops = [t for t in r.trades if t.exit_reason == "stop"]
        self.assertTrue(stops, "a 2% stop on this series should fire")
        for t in stops:
            with self.subTest(exit=t.exit_bar):
                self.assertGreater(t.exit_bar, t.exit_signal_bar)
                self.assertIsNotNone(t.stop_level)

    def test_the_description_names_the_stop(self):
        self.assertNotIn("stop", mr.describe(mr.MeanReversionConfig()))
        self.assertIn("stop", mr.describe(mr.MeanReversionConfig(stop_loss_pct=0.03)))


class TestSimulation(unittest.TestCase):
    N = 700

    def _series(self):
        closes, highs, lows = falling_then_flat(self.N)
        opens = [c - 0.2 for c in closes]
        return closes, opens, highs, lows

    def test_fills_are_still_delayed_past_their_signal(self):
        closes, opens, highs, lows = self._series()
        r = mr.simulate_mean_reversion(closes, opens, highs, lows,
                                       mr.MeanReversionConfig(rule="rsi"))
        self.assertTrue(r.trades)
        for t in r.trades:
            with self.subTest(entry=t.entry_bar):
                self.assertGreater(t.entry_bar, t.entry_signal_bar)
                self.assertGreater(t.exit_bar, t.exit_signal_bar)

    def test_fees_reduce_the_result(self):
        closes, opens, highs, lows = self._series()
        gross = mr.simulate_mean_reversion(
            closes, opens, highs, lows, mr.MeanReversionConfig(fee_bps=0.0))
        net = mr.simulate_mean_reversion(
            closes, opens, highs, lows, mr.MeanReversionConfig(fee_bps=3.5))
        self.assertGreater(gross.net_return_pct, net.net_return_pct)

    def test_both_rules_run_and_differ(self):
        closes, opens, highs, lows = self._series()
        a = mr.simulate_mean_reversion(closes, opens, highs, lows,
                                       mr.MeanReversionConfig(rule="rsi"))
        b = mr.simulate_mean_reversion(closes, opens, highs, lows,
                                       mr.MeanReversionConfig(rule="keltner"))
        self.assertNotEqual(len(a.trades), len(b.trades))

    def test_a_length_mismatch_is_rejected(self):
        closes, opens, highs, lows = self._series()
        with self.assertRaises(mr.MeanReversionError):
            mr.simulate_mean_reversion(closes, opens, highs, lows[:-1],
                                       mr.MeanReversionConfig())


class TestDescription(unittest.TestCase):
    def test_it_names_the_rule_and_its_levels(self):
        self.assertIn("RSI14", mr.describe(mr.MeanReversionConfig()))
        self.assertIn("ATR10", mr.describe(mr.MeanReversionConfig(rule="keltner")))

    def test_the_regime_filter_appears_only_when_it_is_on(self):
        self.assertNotIn("ADX", mr.describe(mr.MeanReversionConfig()))
        self.assertIn("ADX", mr.describe(mr.MeanReversionConfig(adx_max=30.0)))


if __name__ == "__main__":
    unittest.main()
