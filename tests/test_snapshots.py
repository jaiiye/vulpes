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

import agent.snapshots as snapshots_module  # noqa: E402
from agent.config import load_config  # noqa: E402
from agent.factors.smart_money import (  # noqa: E402
    SmartMoneySnapshot,
    WalletPosition,
)
from agent.http_util import HttpError  # noqa: E402
from agent.market_data import AssetContext  # noqa: E402
from agent.snapshots import (  # noqa: E402
    MarketSnapshotRecorder,
    SnapshotRecord,
    SnapshotRecorder,
    coverage_summary,
    load_market_records,
    load_records,
    market_coverage_summary,
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


class FakeMarket:
    """Just enough market to answer `asset_contexts`."""

    def __init__(self, contexts):
        self._contexts = contexts

    def asset_contexts(self):
        return self._contexts


def context(name, **over) -> AssetContext:
    fields = dict(
        index=0,
        name=name,
        mark_price=100.0,
        oracle_price=100.0,
        funding=1e-5,
        open_interest=50.0,
        day_volume=1_000_000.0,
    )
    fields.update(over)
    return AssetContext(**fields)


class TestMarketRecorder(unittest.TestCase):
    """The aggregate-positioning recorder.

    Its whole reason for existing is that these series have no usable history -
    Binance caps its own at ~20 days and Hyperliquid publishes none at all - so
    what is worth pinning is what lands in a record, and what happens when half a
    collection fails.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "market.jsonl"

    def stub_binance(self, responses: dict) -> None:
        """Answer by URL fragment. A fragment missing from `responses` raises,
        which is how a per-series failure gets exercised."""
        original = snapshots_module.request_json

        def fake(url, **kwargs):
            for fragment, value in responses.items():
                if fragment in url:
                    return value
            raise HttpError(f"no stub for {url}")

        snapshots_module.request_json = fake
        self.addCleanup(lambda: setattr(snapshots_module, "request_json", original))

    def recorder(self, contexts, responses=None) -> MarketSnapshotRecorder:
        self.stub_binance(
            responses
            if responses is not None
            else {
                "takerlongshortRatio": [
                    {"buyVol": "60", "sellVol": "40", "buySellRatio": "1.5"}
                ],
                "globalLongShortAccountRatio": [
                    {
                        "longAccount": "0.7",
                        "shortAccount": "0.3",
                        "longShortRatio": "2.33",
                    }
                ],
                "premiumIndex": {
                    "markPrice": "101",
                    "indexPrice": "100",
                    "lastFundingRate": "0.0001",
                },
            }
        )
        return MarketSnapshotRecorder(path=self.path, market=FakeMarket(contexts))

    def last_record(self):
        return load_market_records(self.path)[-1]

    def test_every_perp_is_captured_from_one_call(self):
        recorder = self.recorder([context("BTC"), context("ETH")])

        self.assertTrue(recorder.record(("BTC",), force=True))

        self.assertEqual(sorted(self.last_record().hyperliquid), ["BTC", "ETH"])

    def test_perps_with_no_open_interest_are_left_out(self):
        """Delisted names come back with every field at zero - measured, 56 of
        234 on mainnet - and a market that does not exist has no positioning.

        The first version of this test asserted a leading-"@" filter, which was
        simply wrong about the venue: this endpoint is perps only. It passed, and
        the code excluded nothing, which is how the mistake survived a green
        suite until a real collection was counted.
        """
        recorder = self.recorder([context("BTC"), context("MATIC", open_interest=0.0)])

        recorder.record(("BTC",), force=True)

        self.assertEqual(list(self.last_record().hyperliquid), ["BTC"])

    def test_the_premium_is_the_mark_against_the_oracle(self):
        """The one field here the candle archive cannot reconstruct: it is the
        perp's own mark against its oracle, not a traded price."""
        recorder = self.recorder([context("BTC", mark_price=101.0, oracle_price=100.0)])

        recorder.record(("BTC",), force=True)

        self.assertAlmostEqual(
            self.last_record().hyperliquid["BTC"]["premium"], 0.01, places=8
        )

    def test_the_two_series_hyperliquid_does_not_publish_are_stored(self):
        recorder = self.recorder([context("BTC")])

        recorder.record(("BTC",), force=True)

        binance = self.last_record().binance["BTC"]
        self.assertEqual(binance["taker"]["buy_vol"], 60.0)
        self.assertEqual(binance["taker"]["ratio"], 1.5)
        self.assertEqual(binance["accounts"]["long_share"], 0.7)
        self.assertEqual(binance["funding"]["funding_rate"], 0.0001)

    def test_a_failed_series_is_recorded_rather_than_swallowed(self):
        """A symbol silently missing one of three series is a hole nothing
        downstream can see."""
        recorder = self.recorder([context("BTC")], responses={})

        recorder.record(("BTC",), force=True)

        record = self.last_record()
        self.assertEqual(record.binance["BTC"], {})
        joined = " ".join(record.errors)
        self.assertIn("taker", joined)
        self.assertIn("accounts", joined)
        self.assertIn("funding", joined)

    def test_a_failed_market_call_still_records_the_other_half(self):
        """Partial is worth more than nothing, provided it says so."""

        class Broken(FakeMarket):
            def asset_contexts(self):
                raise HttpError("venue unreachable")

        self.stub_binance(
            {
                "premiumIndex": {
                    "markPrice": "101",
                    "indexPrice": "100",
                    "lastFundingRate": "0.0001",
                }
            }
        )
        recorder = MarketSnapshotRecorder(
            path=self.path, market=Broken([context("BTC")])
        )

        self.assertTrue(recorder.record(("BTC",), force=True))

        record = self.last_record()
        self.assertEqual(record.hyperliquid, {})
        self.assertIn("hyperliquid", " ".join(record.errors))
        self.assertIn("BTC", record.binance)

    def test_a_second_call_within_the_hour_writes_nothing(self):
        """The series update hourly, so three passes in four have nothing new to
        add - and this file grows forever."""
        recorder = self.recorder([context("BTC")])
        recorder.record(("BTC",), force=True)

        self.assertFalse(recorder.record(("BTC",)))

    def test_force_records_anyway(self):
        recorder = self.recorder([context("BTC")])

        recorder.record(("BTC",), force=True)
        self.assertTrue(recorder.record(("BTC",), force=True))

        self.assertEqual(len(load_market_records(self.path)), 2)

    def test_the_interval_is_measured_against_the_file(self):
        """Read from the newest line, not from process state: there is no second
        copy of the clock to lose across a restart."""
        recorder = self.recorder([context("BTC")])
        recorder.record(("BTC",), force=True)
        payload = json.loads(self.path.read_text().splitlines()[-1])
        payload["timestamp"] -= 7200
        self.path.write_text(json.dumps(payload) + "\n")

        self.assertTrue(recorder.record(("BTC",)))

    def test_load_skips_malformed_lines(self):
        self.path.write_text(
            '{"timestamp": 1.0, "hyperliquid": {"BTC": {}}}\nnot json\n'
        )

        self.assertEqual(len(load_market_records(self.path)), 1)

    def test_a_missing_file_reads_as_empty(self):
        self.assertEqual(load_market_records(self.path), [])
        self.assertIn("no market snapshots", market_coverage_summary([]))

    def test_coverage_flags_a_short_history_and_partial_rows(self):
        recorder = self.recorder([context("BTC")], responses={})
        recorder.record(("BTC",), force=True)

        summary = market_coverage_summary(load_market_records(self.path))

        self.assertIn("NOT YET USABLE", summary)
        self.assertIn("recorded with an error", summary)
        self.assertIn("1 Hyperliquid perps", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
