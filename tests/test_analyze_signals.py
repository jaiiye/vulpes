"""Tests for the forward signal-quality analyzer.

The analyzer makes a statistical claim ("does the signal predict?"), so a bug
in it would produce a confident wrong answer rather than an obvious crash.
These tests pin the arithmetic against hand-computable data.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_signals import (  # noqa: E402
    LONG,
    NEUTRAL,
    SHORT,
    SignalObservation,
    bucket_confidence,
    bucket_score,
    build_price_series,
    build_report,
    load_signals,
    price_at,
    score_horizon,
)


def write_journal(records: list[dict]) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for rec in records:
        handle.write(json.dumps(rec) + "\n")
    handle.close()
    return handle.name


def signal_record(ts, price, action=LONG, score=70.0, confidence=0.8, blocked=None):
    return {
        "kind": "signal",
        "ts": ts,
        "symbol": "BTC",
        "action": action,
        "score": score,
        "confidence": confidence,
        "price": price,
        "blocked_by": blocked,
    }


def rising_path(n: int, start_price: float = 100.0, step_s: float = 3600.0):
    """`n` hourly observations ending 1 hour ago, price rising by 1 each step.

    The last observation is placed an hour in the past so a +1h horizon always
    has somewhere to land.
    """
    now = time.time()
    base = now - 3600.0 - (n - 1) * step_s
    return [
        (base + i * step_s, start_price + i, LONG)
        for i in range(n)
    ]


class TestLoadSignals(unittest.TestCase):
    def test_reads_signal_events_only(self):
        path = write_journal(
            [
                {"kind": "start", "ts": 1},
                signal_record(100, 50.0),
                {"kind": "stop", "ts": 2},
                signal_record(200, 51.0),
            ]
        )
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        obs, warnings = load_signals(path)
        self.assertEqual(len(obs), 2)
        self.assertEqual(warnings, [])
        self.assertEqual(obs[0].price, 50.0)

    def test_skips_signals_without_a_usable_price(self):
        path = write_journal(
            [
                signal_record(100, 50.0),
                {"kind": "signal", "ts": 200, "price": None, "action": LONG},
                {"kind": "signal", "ts": 300, "action": LONG},  # no price key
                signal_record(400, 0.0),  # non-positive
            ]
        )
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        obs, _ = load_signals(path)
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].ts, 100)

    def test_malformed_lines_are_counted_not_raised(self):
        path = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        path.write("{ not json\n")
        path.write(json.dumps(signal_record(100, 50.0)) + "\n")
        path.close()
        self.addCleanup(lambda: Path(path.name).unlink(missing_ok=True))

        obs, warnings = load_signals(path.name)
        self.assertEqual(len(obs), 1)
        self.assertTrue(any("malformed" in w for w in warnings))

    def test_missing_file_is_reported(self):
        obs, warnings = load_signals("/nonexistent/journal.jsonl")
        self.assertEqual(obs, [])
        self.assertTrue(warnings)

    def test_action_is_normalised_to_lowercase(self):
        path = write_journal([signal_record(100, 50.0, action="LONG")])
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))
        obs, _ = load_signals(path)
        self.assertEqual(obs[0].action, LONG)


class TestPriceSeries(unittest.TestCase):
    def test_sorted_and_deduplicated(self):
        obs = [
            SignalObservation(3, "BTC", LONG, 70, 0.8, 103.0),
            SignalObservation(1, "BTC", LONG, 70, 0.8, 101.0),
            SignalObservation(2, "BTC", LONG, 70, 0.8, 102.0),
            SignalObservation(2, "BTC", LONG, 70, 0.8, 102.0),
        ]
        self.assertEqual(
            build_price_series(obs), [(1, 101.0), (2, 102.0), (3, 103.0)]
        )

    def test_zero_timestamps_are_dropped(self):
        obs = [SignalObservation(0, "BTC", LONG, 70, 0.8, 100.0)]
        self.assertEqual(build_price_series(obs), [])


class TestPriceAt(unittest.TestCase):
    def setUp(self):
        self.series = [(0.0, 100.0), (3600.0, 101.0), (7200.0, 102.0)]

    def test_exact_match(self):
        self.assertEqual(price_at(self.series, 3600.0, 1800), 101.0)

    def test_picks_the_nearest_within_the_gap(self):
        self.assertEqual(price_at(self.series, 3600.0 + 600, 1800), 101.0)
        self.assertEqual(price_at(self.series, 3600.0 - 600, 1800), 101.0)

    def test_rejects_when_the_nearest_point_is_too_far(self):
        """A stale price used as a forward return would fabricate edge."""
        self.assertIsNone(price_at(self.series, 5400.0, 600))

    def test_before_the_series(self):
        self.assertIsNone(price_at(self.series, -10000.0, 600))
        self.assertEqual(price_at(self.series, -100.0, 600), 100.0)

    def test_after_the_series(self):
        self.assertIsNone(price_at(self.series, 20000.0, 600))

    def test_empty_series(self):
        self.assertIsNone(price_at([], 0.0, 1800))


class TestScoreHorizon(unittest.TestCase):
    """Hand-computable: a monotonically rising price series with hourly marks.

    19 signals can be scored at +1h; the 20th has no forward price within the
    gap tolerance and is skipped.
    """

    def build(self, action):
        path = [
            (i * 3600.0, 100.0 + i, action) for i in range(20)
        ]
        now = time.time()
        base = now - 3600.0 - 19 * 3600.0
        obs = [
            SignalObservation(
                ts=base + i * 3600.0,
                symbol="BTC",
                action=action,
                score=70.0 if action == LONG else 30.0,
                confidence=0.8,
                price=100.0 + i,
            )
            for i in range(20)
        ]
        return obs, build_price_series(obs)

    def test_longs_all_hit_on_a_rising_series(self):
        obs, series = self.build(LONG)
        r = score_horizon(obs, series, 1.0, max_gap=1800)
        self.assertEqual(r.n, 19)
        self.assertEqual(r.hits, 19)
        self.assertEqual(r.skipped, 1)
        self.assertAlmostEqual(r.hit_rate, 1.0)

    def test_shorts_all_miss_on_a_rising_series(self):
        obs, series = self.build(SHORT)
        r = score_horizon(obs, series, 1.0, max_gap=1800)
        self.assertEqual(r.n, 19)
        self.assertEqual(r.hits, 0)
        self.assertAlmostEqual(r.hit_rate, 0.0)

    def test_neutral_signals_are_excluded_not_counted_as_misses(self):
        obs, series = self.build(NEUTRAL)
        r = score_horizon(obs, series, 1.0, max_gap=1800)
        self.assertEqual(r.n, 0)
        self.assertEqual(r.skipped, 0)

    def test_signals_without_forward_data_are_skipped(self):
        """The newest signal cannot be scored yet; counting it would bias."""
        obs, series = self.build(LONG)
        r = score_horizon(obs, series, 24.0, max_gap=1800)
        self.assertEqual(r.n, 0)
        self.assertEqual(r.skipped, 20)

    def test_subset_filter_is_applied(self):
        obs, series = self.build(LONG)
        r = score_horizon(
            obs, series, 1.0, max_gap=1800, subset=lambda o: o.score >= 70
        )
        self.assertEqual(r.n, 19)

        r_none = score_horizon(
            obs, series, 1.0, max_gap=1800, subset=lambda o: o.score >= 999
        )
        self.assertEqual(r_none.n, 0)

    def test_half_hits_gives_z_of_zero(self):
        """A coin-flip sample must not look significant."""
        now = time.time()
        base = now - 3600.0 - 39 * 3600.0
        obs = [
            SignalObservation(
                ts=base + i * 3600.0,
                symbol="BTC",
                action=LONG,
                score=70.0,
                confidence=0.8,
                # Alternating up/down so exactly half the longs hit.
                price=100.0 + (i % 2),
            )
            for i in range(40)
        ]
        series = build_price_series(obs)
        r = score_horizon(obs, series, 1.0, max_gap=1800)
        self.assertGreater(r.n, 0)
        self.assertLess(abs(r.z), 1.96)
        self.assertFalse(r.is_significant)

    def test_verdict_requires_a_minimum_sample(self):
        r = score_horizon([], [], 1.0)
        self.assertIn("样本不足", r.verdict())


class TestBuckets(unittest.TestCase):
    def test_score_buckets(self):
        self.assertEqual(bucket_score(85), ">=70")
        self.assertEqual(bucket_score(70), ">=70")
        self.assertEqual(bucket_score(65), "60-70")
        self.assertEqual(bucket_score(60), "60-70")
        self.assertEqual(bucket_score(50), "40-60")
        self.assertEqual(bucket_score(35), "30-40")
        self.assertEqual(bucket_score(10), "<30")

    def test_score_buckets_cover_the_whole_range(self):
        for score in (0, 15, 30, 40, 50, 60, 70, 99, 100):
            self.assertTrue(bucket_score(score))

    def test_confidence_buckets(self):
        self.assertEqual(bucket_confidence(0.9), ">=80%")
        self.assertEqual(bucket_confidence(0.8), ">=80%")
        self.assertEqual(bucket_confidence(0.7), "60-80%")
        self.assertEqual(bucket_confidence(0.5), "40-60%")
        self.assertEqual(bucket_confidence(0.1), "<40%")


class TestBuildReport(unittest.TestCase):
    def test_empty_journal_returns_a_clear_message(self):
        report, text = build_report("/nonexistent/journal.jsonl")
        self.assertEqual(report.total_signals, 0)
        self.assertIn("没有可分析的信号记录", text)

    def test_short_span_is_flagged_as_not_conclusive(self):
        path = write_journal(
            [signal_record(100 + i, 50.0 + i) for i in range(5)]
        )
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        report, text = build_report(path)
        self.assertEqual(report.total_signals, 5)
        self.assertIn("跨度不足 7 天", text)

    def test_blocked_signals_appear_as_a_control_group(self):
        path = write_journal(
            [
                signal_record(100, 50.0, blocked=None),
                signal_record(200, 51.0, blocked="cooldown active, 240 min remaining"),
            ]
        )
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        _, text = build_report(path)
        self.assertIn("被门禁拒绝的信号", text)
        self.assertIn("cooldown", text)
        self.assertIn("通过 1 条 / 拒绝 1 条", text)

    def test_report_survives_a_journal_of_only_neutral_signals(self):
        """The common early case: every signal rejected on threshold."""
        path = write_journal(
            [
                signal_record(100 + i, 50.0, action=NEUTRAL, score=50.0,
                              blocked="score did not cross a threshold")
                for i in range(10)
            ]
        )
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        report, text = build_report(path)
        self.assertEqual(report.total_signals, 10)
        # The bucket name, not the truncated raw text.
        self.assertIn("threshold", text)
        self.assertNotIn("did not cross a thresh\n", text)

    def test_no_directional_signals_skips_the_empty_tables(self):
        """A wall of n=0 tables would bury the gate distribution, which is the
        only useful number when nothing has fired yet."""
        path = write_journal(
            [
                signal_record(100 + i, 50.0, action=NEUTRAL, score=50.0,
                              blocked="score did not cross a threshold")
                for i in range(10)
            ]
        )
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        _, text = build_report(path)
        self.assertIn("可评分的方向性信号: 0", text)
        self.assertIn("暂无方向性信号", text)
        # The empty hit-rate table must not be printed.
        self.assertNotIn("样本不足 (n=0)", text)
        # But the gate distribution still must be.
        self.assertIn("拒绝原因分布", text)

    def test_directional_signals_do_print_the_tables(self):
        path = write_journal(
            [signal_record(100 + i, 50.0 + i, action=LONG) for i in range(5)]
        )
        self.addCleanup(lambda: Path(path).unlink(missing_ok=True))

        _, text = build_report(path)
        self.assertIn("可评分的方向性信号: 5", text)
        self.assertIn("方向命中率", text)
        self.assertNotIn("暂无方向性信号", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
