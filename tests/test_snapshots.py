"""Tests for the whale snapshot recorder.

This module had no coverage at all, which is how a real defect survived: the
recorder built its `SmartMoneyFactor` from bare defaults instead of the agent's
config. Those defaults happened to equal the shipped config, so nothing looked
wrong until `smart_money` was tuned - at which point the recorded history
silently stopped describing the strategy being run.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import load_config  # noqa: E402
from agent.factors.smart_money import (  # noqa: E402
    SmartMoneySnapshot,
    WalletPosition,
)
from agent.snapshots import (  # noqa: E402
    SnapshotRecord,
    SnapshotRecorder,
    coverage_summary,
    load_records,
)

WHALE = "0x" + "a" * 40
BANNED = "0x" + "b" * 40

CONFIG = f"""
name: t
symbol: BTC
smart_money:
  windows: [day, week]
  top_per_window: 17
  min_persistence: 2
  max_wallets: 33
  min_account_value: 25000
  whitelist: ["{WHALE}"]
  blacklist: ["{BANNED}"]
"""


def write_config(text: str = CONFIG) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    handle.write(text)
    handle.close()
    return handle.name


def make_snapshot(symbol: str = "BTC") -> SmartMoneySnapshot:
    return SmartMoneySnapshot(
        symbol=symbol,
        wallets_sampled=10,
        wallets_selected=10,
        source="leaderboard",
        persistence_window_count=3,
        avg_persistence=2.0,
        positions=[
            WalletPosition(
                wallet="0xabc",
                size=1.5,
                notional=150_000.0,
                entry_price=100.0,
                unrealized_pnl=250.0,
                leverage=2.0,
                account_value=50_000.0,
                persistence=2,
                whitelisted=True,
            ),
            WalletPosition(wallet="0xdef", size=-0.5, notional=-50_000.0),
        ],
    )


class TestRecorderFollowsConfig(unittest.TestCase):
    """The regression: the recorder must select wallets exactly as the agent."""

    def setUp(self):
        self.cfg_path = write_config()
        self.addCleanup(os.unlink, self.cfg_path)
        self.cfg = load_config(self.cfg_path)

    def test_config_values_drive_the_leaderboard(self):
        rec = SnapshotRecorder(path=None, config=self.cfg)
        lb = rec.factor.leaderboard

        self.assertEqual(lb.windows, ("day", "week"))
        self.assertEqual(lb.top_per_window, 17)
        self.assertEqual(lb.min_persistence, 2)
        self.assertEqual(lb.max_wallets, 33)
        self.assertAlmostEqual(lb.min_account_value, 25_000)

    def test_whitelist_and_blacklist_are_carried(self):
        rec = SnapshotRecorder(path=None, config=self.cfg)
        self.assertEqual(rec.factor.leaderboard.whitelist, {WHALE})
        self.assertEqual(rec.factor.leaderboard.blacklist, {BANNED})

    def test_changing_the_config_changes_the_recorder(self):
        """The defect was structural: no config path existed at all."""
        other = write_config(CONFIG.replace("top_per_window: 17", "top_per_window: 5"))
        self.addCleanup(os.unlink, other)

        rec = SnapshotRecorder(path=None, config=load_config(other))
        self.assertEqual(rec.factor.leaderboard.top_per_window, 5)

    def test_without_a_config_it_still_constructs(self):
        """Backwards compatible: bare construction keeps working."""
        rec = SnapshotRecorder(path=None)
        self.assertIsNotNone(rec.factor.leaderboard)

    def test_config_hyperfeed_flag_is_honoured(self):
        path = write_config(CONFIG + "\nuse_hyperfeed: true\n")
        self.addCleanup(os.unlink, path)
        rec = SnapshotRecorder(path=None, config=load_config(path))
        self.assertTrue(rec.factor.use_hyperfeed)


class TestSelectionProvenance(unittest.TestCase):
    """A snapshot is only interpretable alongside the rules that produced it."""

    def setUp(self):
        self.cfg_path = write_config()
        self.addCleanup(os.unlink, self.cfg_path)
        self.cfg = load_config(self.cfg_path)

    def test_selection_reports_every_relevant_parameter(self):
        sel = SnapshotRecorder(path=None, config=self.cfg).selection
        self.assertEqual(sel["top_per_window"], 17)
        self.assertEqual(sel["windows"], ["day", "week"])
        self.assertEqual(sel["min_persistence"], 2)
        self.assertEqual(sel["max_wallets"], 33)
        self.assertIn("min_account_value", sel)
        self.assertIn("use_hyperfeed", sel)

    def test_selection_is_json_serialisable(self):
        sel = SnapshotRecorder(path=None, config=self.cfg).selection
        self.assertIsInstance(json.dumps(sel), str)


class TestSnapshotRecord(unittest.TestCase):
    def test_positions_are_copied_across(self):
        rec = SnapshotRecord.from_snapshot(make_snapshot())
        self.assertEqual(len(rec.positions), 2)
        self.assertEqual(rec.positions[0]["wallet"], "0xabc")
        self.assertAlmostEqual(rec.positions[0]["notional"], 150_000.0)
        self.assertTrue(rec.positions[0]["whitelisted"])

    def test_selection_is_attached(self):
        rec = SnapshotRecord.from_snapshot(make_snapshot(), {"top_per_window": 7})
        self.assertEqual(rec.selection, {"top_per_window": 7})

    def test_selection_defaults_to_empty(self):
        self.assertEqual(SnapshotRecord.from_snapshot(make_snapshot()).selection, {})

    def test_selection_default_is_not_shared_between_instances(self):
        """A mutable default would leak state across records."""
        a = SnapshotRecord.from_snapshot(make_snapshot())
        b = SnapshotRecord.from_snapshot(make_snapshot())
        a.selection["x"] = 1
        self.assertEqual(b.selection, {})

    def test_json_round_trip_keeps_selection(self):
        rec = SnapshotRecord.from_snapshot(
            make_snapshot(), {"windows": ["day"], "top_per_window": 9}
        )
        payload = json.loads(rec.to_json())
        self.assertEqual(payload["selection"]["top_per_window"], 9)


class TestLoadRecords(unittest.TestCase):
    def write(self, records: list[dict]) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for rec in records:
            handle.write(json.dumps(rec) + "\n")
        handle.close()
        return handle.name

    def base(self, **overrides) -> dict:
        payload = {
            "timestamp": 1_700_000_000.0,
            "symbol": "BTC",
            "wallets_sampled": 5,
            "wallet_count": 2,
            "long_ratio_pct": 60.0,
            "weighted_long_ratio_pct": 61.0,
            "top1_share_pct": 30.0,
            "holder_avg_persistence": 1.5,
            "long_notional_usd": 1000.0,
            "short_notional_usd": 500.0,
            "positions": [],
        }
        payload.update(overrides)
        return payload

    def test_reads_selection_when_present(self):
        path = self.write([self.base(selection={"top_per_window": 42})])
        self.addCleanup(os.unlink, path)
        records = load_records(path)
        self.assertEqual(records[0].selection, {"top_per_window": 42})

    def test_tolerates_records_written_before_provenance_existed(self):
        """Old JSONL lines have no `selection` key and must still load."""
        path = self.write([self.base()])
        self.addCleanup(os.unlink, path)
        records = load_records(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].selection, {})

    def test_skips_malformed_lines(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        handle.write("{ not json\n")
        handle.write(json.dumps(self.base()) + "\n")
        handle.close()
        self.addCleanup(os.unlink, handle.name)

        self.assertEqual(len(load_records(handle.name)), 1)

    def test_missing_file_is_empty(self):
        self.assertEqual(load_records("/nonexistent/x.jsonl"), [])


class TestCoverageSummary(unittest.TestCase):
    def test_empty_is_reported(self):
        self.assertIn("no snapshots", coverage_summary([]))

    def test_short_history_is_flagged_unusable(self):
        rec = SnapshotRecord.from_snapshot(make_snapshot())
        rec.timestamp = 1_700_000_000.0
        other = SnapshotRecord.from_snapshot(make_snapshot())
        other.timestamp = 1_700_000_000.0 + 3600  # one hour later

        text = coverage_summary([rec, other])
        self.assertIn("NOT YET USABLE", text)

    def test_symbols_are_listed(self):
        a = SnapshotRecord.from_snapshot(make_snapshot("BTC"))
        b = SnapshotRecord.from_snapshot(make_snapshot("ETH"))
        text = coverage_summary([a, b])
        self.assertIn("BTC", text)
        self.assertIn("ETH", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
