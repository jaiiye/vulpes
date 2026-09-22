"""Tests for the shared execution config and its inheritance.

The refactor that introduced `ExecutionConfig` traded one problem for another,
and the new one is quieter. Before, each rule restated the execution fields and
copied them into a `TrendGateConfig`, so a rule could silently *lack* a field -
which is what happened to `pullback_entry`, where the stop was unreachable.
After, the fields exist everywhere by construction, but a subclass that forgets
`super().__post_init__()` silently loses all of their *validation* with no error
from Python, no failing import, and no visible difference in most results.

`TestSubclassContract` is the part that matters: it demonstrates the trap rather
than describing it, and it fails if the arrangement that catches it goes away.
"""

from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import mean_reversion as mr  # noqa: E402
from backtest import pullback_entry as pb  # noqa: E402
from backtest import trend_gate as tg  # noqa: E402

#: One constructor per rule config, so the shared assertions below run over all
#: of them. Adding a rule should mean adding a line here.
RULE_CONFIGS = {
    "TrendGateConfig": tg.TrendGateConfig,
    "MeanReversionConfig": mr.MeanReversionConfig,
    "PullbackConfig": pb.PullbackConfig,
}


class TestExecutionConfigItself(unittest.TestCase):
    def test_it_rejects_a_negative_hold_cap(self):
        with self.assertRaises(tg.TrendGateError):
            tg.ExecutionConfig(hold_bars=-1)

    def test_it_rejects_an_impossible_stop(self):
        for bad in (-0.01, 1.0, 1.5):
            with self.subTest(stop=bad):
                with self.assertRaises(tg.TrendGateError):
                    tg.ExecutionConfig(stop_loss_pct=bad)

    def test_it_rejects_a_negative_cooldown(self):
        with self.assertRaises(tg.TrendGateError):
            tg.ExecutionConfig(stop_loss_pct=0.02, stop_cooldown_bars=-1)

    def test_it_rejects_a_cooldown_with_no_stop_to_follow(self):
        with self.assertRaises(tg.TrendGateError):
            tg.ExecutionConfig(stop_cooldown_bars=5)

    def test_the_fields_are_exactly_the_ones_the_fill_loop_reads(self):
        """If `simulate_flags` grows a knob that is not here, the knob will be
        restated per rule again - which is the arrangement this replaced."""
        names = {f.name for f in dataclasses.fields(tg.ExecutionConfig)}
        self.assertEqual(
            names,
            {"fee_bps", "hold_bars", "funding_per_bar",
             "stop_loss_pct", "stop_cooldown_bars"},
        )


class TestEveryRuleSharesIt(unittest.TestCase):
    def test_every_rule_config_is_an_execution_config(self):
        """This is what lets each rule hand its own config to `simulate_flags`
        instead of copying fields into a `TrendGateConfig`."""
        for name, ctor in RULE_CONFIGS.items():
            with self.subTest(config=name):
                self.assertIsInstance(ctor(), tg.ExecutionConfig)

    def test_every_rule_inherits_the_validation(self):
        """One definition, one error: a bad execution field raises
        `TrendGateError` from every rule, not that rule's own type."""
        for name, ctor in RULE_CONFIGS.items():
            for kw in ({"hold_bars": -1}, {"stop_loss_pct": 2.0},
                       {"stop_cooldown_bars": 3}):
                with self.subTest(config=name, kw=kw):
                    with self.assertRaises(tg.TrendGateError):
                        ctor(**kw)

    def test_no_rule_restates_the_execution_fields(self):
        """A restated field shadows the inherited one and can carry a different
        default without any error - the failure the split removed."""
        inherited = {f.name for f in dataclasses.fields(tg.ExecutionConfig)}
        for name, ctor in RULE_CONFIGS.items():
            own = {f.name for f in dataclasses.fields(ctor)} - inherited
            with self.subTest(config=name):
                self.assertTrue(own.isdisjoint(inherited))

    def test_the_defaults_cannot_drift_between_rules(self):
        for name, ctor in RULE_CONFIGS.items():
            with self.subTest(config=name):
                cfg = ctor()
                base = tg.ExecutionConfig()
                for f in dataclasses.fields(tg.ExecutionConfig):
                    self.assertEqual(getattr(cfg, f.name), getattr(base, f.name))

    def test_each_rule_keeps_its_own_parameters(self):
        """The split must not have flattened the rules together: this rule's
        fields are still reachable and still validated as its own."""
        self.assertEqual(mr.MeanReversionConfig().rule, "rsi")
        self.assertEqual(pb.PullbackConfig().cross_reference, "sma")
        self.assertEqual(tg.TrendGateConfig().min_bull_votes, 3)
        with self.assertRaises(mr.MeanReversionError):
            mr.MeanReversionConfig(rule="bollinger")
        with self.assertRaises(tg.TrendGateError):
            tg.TrendGateConfig(min_bull_votes=9)


class TestSubclassContract(unittest.TestCase):
    """The one thing inheritance gets wrong silently.

    A subclass that forgets `super().__post_init__()` loses every execution
    check with no error at all. These tests make that visible, so the convention
    stated in `ExecutionConfig`'s docstring is enforced rather than trusted.
    """

    def test_a_subclass_that_skips_super_loses_validation(self):
        """Demonstrates the trap is real. If this ever fails, the failure mode
        has changed and the convention below needs re-reading."""

        @dataclasses.dataclass
        class Forgetful(tg.ExecutionConfig):
            def __post_init__(self) -> None:
                pass                      # deliberately no super() call

        Forgetful(hold_bars=-1)            # would raise if super() were called
        with self.assertRaises(tg.TrendGateError):
            tg.ExecutionConfig(hold_bars=-1)

    def test_every_shipped_subclass_calls_super(self):
        """The contract, checked on the real subclasses rather than a decoy.

        Checked by behaviour and not by reading source: a subclass that calls
        `super().__post_init__()` conditionally, or after its own checks, still
        fails this if the call is unreachable for a bad execution value.
        """
        for name, ctor in RULE_CONFIGS.items():
            if ctor is tg.ExecutionConfig:
                continue
            with self.subTest(config=name):
                with self.assertRaises(tg.TrendGateError):
                    ctor(hold_bars=-1)


class TestReplaceStillWorks(unittest.TestCase):
    """`dataclasses.replace` is how every measurement in this repository varies
    one parameter, so inheritance must not break it."""

    def test_replacing_an_inherited_field_keeps_the_rule_fields(self):
        cfg = mr.MeanReversionConfig(rule="keltner", oversold=25.0)
        changed = dataclasses.replace(cfg, fee_bps=7.2)
        self.assertEqual(changed.fee_bps, 7.2)
        self.assertEqual(changed.rule, "keltner")
        self.assertEqual(changed.oversold, 25.0)

    def test_replacing_a_rule_field_keeps_the_inherited_ones(self):
        cfg = mr.MeanReversionConfig(fee_bps=7.2, exit_level=60.0)
        changed = dataclasses.replace(cfg, exit_level=70.0)
        self.assertEqual(changed.exit_level, 70.0)
        self.assertEqual(changed.fee_bps, 7.2)

    def test_a_replacement_is_validated_like_a_construction(self):
        """`replace` re-runs `__post_init__`, so a bad replacement must raise
        from the same place a bad construction does."""
        with self.assertRaises(tg.TrendGateError):
            dataclasses.replace(mr.MeanReversionConfig(), stop_loss_pct=5.0)


if __name__ == "__main__":
    unittest.main()
