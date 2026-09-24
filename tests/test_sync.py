"""Sync planning, cost arithmetic, and column-name validity.

`TestConfiguredColumnsExist` is the test worth having. Every column set in the
sync script is written by hand, a wrong name is not caught until a COPY runs
against a live bucket and fails, and by then the run has already spent money.
The expected schemas below were read from the actual archive files with
`parquet_metadata`, so the check is against reality rather than against the
script's own opinion.

That check has already earned its place: the fills set originally listed
`is_funding`, a column that does not exist in the archive.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sync_reservoir import (  # noqa: E402
    SYNC_TARGETS,
    TARGETS_BY_NAME,
    build_copy_sql,
    date_from_key,
    egress_cost_usd,
    human_gb,
    months_to_stay_free,
    to_gb,
)

# Read from the archive with `parquet_metadata` on a 2026-09-16 object. These
# are facts about the stored files, not about the documentation.
REAL_SCHEMAS: dict[str, set[str]] = {
    "candles": {
        "coin", "dex", "asset_class", "base_symbol", "quote_symbol",
        "timestamp", "open", "high", "low", "close", "volume",
        "volume_quote", "trade_count",
    },
    "positions": {
        "user", "market", "size", "notional", "entry_price",
        "liquidation_price", "leverage_type", "leverage", "funding_pnl",
        "account_value", "account_mode",
    },
    "account_values": {
        "user", "dex", "collateral_token", "account_value",
        "total_long_notional", "total_short_notional", "account_mode",
    },
    "orderbook": {"block_time_ms", "block_number", "bids", "asks"},
    "fills": {
        "coin", "dex", "asset_class", "base_symbol", "quote_symbol", "price",
        "size", "side", "timestamp", "direction", "realized_pnl", "tx_hash",
        "order_id", "trade_id", "fee", "fee_token", "address", "crossed",
        "start_position", "client_order_id", "builder", "builder_fee",
        "deployer_fee", "priority_gas", "twap_id", "is_liquidation",
        "liquidation_mark_px", "liquidation_method",
    },
}

GB = 1024**3


class TestConfiguredColumnsExist(unittest.TestCase):
    """A column name that is not in the file fails the COPY, after paying."""

    def test_every_dataset_has_a_known_schema(self):
        self.assertEqual(
            sorted(REAL_SCHEMAS), sorted(TARGETS_BY_NAME),
            "a dataset was added to the sync script without a recorded schema",
        )

    def test_research_columns_exist(self):
        for target in SYNC_TARGETS:
            schema = REAL_SCHEMAS[target.name]
            for column in target.research:
                with self.subTest(dataset=target.name, column=column):
                    self.assertIn(column, schema)

    def test_wide_columns_exist(self):
        for target in SYNC_TARGETS:
            schema = REAL_SCHEMAS[target.name]
            for column in target.wide:
                with self.subTest(dataset=target.name, column=column):
                    self.assertIn(column, schema)

    def test_wide_is_a_superset_of_research(self):
        """Otherwise `--columns wide` would silently drop a needed column."""
        for target in SYNC_TARGETS:
            with self.subTest(dataset=target.name):
                self.assertTrue(set(target.research) <= set(target.wide))

    def test_flow_columns_exist(self):
        for target in SYNC_TARGETS:
            schema = REAL_SCHEMAS[target.name]
            for column in target.flow:
                with self.subTest(dataset=target.name, column=column):
                    self.assertIn(column, schema)

    def test_flow_sits_between_research_and_wide(self):
        """`flow` is a middle tier: it adds what the position-flow work needs
        and nothing else. A column in `flow` but not in `wide` would mean the
        tiers are not ordered, and a column in research but not in flow would
        mean the position-flow set silently drops something already in use.

        Only datasets that define a tier are checked: a tier is optional, and
        the datasets that do not define one keep an empty tuple.
        """
        for target in SYNC_TARGETS:
            if not target.flow:
                continue
            with self.subTest(dataset=target.name):
                self.assertTrue(set(target.research) <= set(target.flow))
                self.assertTrue(set(target.flow) <= set(target.wide))

    def test_flow_is_meaningfully_cheaper_than_wide(self):
        """The whole reason the tier exists. `wide` keeps every non-identifier
        column; `flow` drops `start_position`, which alone is 11.4% of the
        archive, because it can be reconstructed from `size` + `direction`
        against the daily position snapshots."""
        from sync_reservoir import projected_fraction

        fills = TARGETS_BY_NAME["fills"]
        flow = projected_fraction("fills", fills.columns_for("flow"))
        wide = projected_fraction("fills", fills.columns_for("wide"))
        self.assertLess(flow, wide)
        self.assertNotIn("start_position", fills.columns_for("flow"))

    def test_a_dataset_without_a_flow_set_says_so(self):
        """No silent fallback: the caller asked for a tier that is not defined
        for this dataset, and the answer is not "use wide instead"."""
        with self.assertRaises(ValueError):
            TARGETS_BY_NAME["candles"].columns_for("flow")

    def test_unknown_column_mode_is_rejected(self):
        """No silent fallback to a default set: the cost difference is large."""
        for target in SYNC_TARGETS:
            with self.assertRaises(ValueError):
                target.columns_for("everything")


class TestCompletedFile(unittest.TestCase):
    """A stalled COPY leaves a zero-length file at the destination.

    Counting mere existence as "done" turns that into a permanently skipped
    day, and the hole is undetectable from the plan because the day is reported
    as already fetched. This happened: a hung duckdb created a 0-byte
    `date=2025-07-30.parquet` that the next run would have walked past.
    """

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_missing_file_is_not_complete(self):
        from sync_reservoir import completed_file

        self.assertFalse(completed_file(self.root / "nope.parquet"))

    def test_zero_length_file_is_not_complete(self):
        from sync_reservoir import completed_file

        path = self.root / "empty.parquet"
        path.write_bytes(b"")
        self.assertFalse(completed_file(path))

    def test_file_with_content_is_complete(self):
        from sync_reservoir import completed_file

        path = self.root / "ok.parquet"
        path.write_bytes(b"PAR1")
        self.assertTrue(completed_file(path))

    def test_truncated_day_is_refetched_not_skipped(self):
        """End to end through the planner, which is where it matters."""
        from sync_reservoir import TARGETS_BY_NAME, plan_dataset
        import sync_reservoir as module

        out = self.root / "positions"
        out.mkdir(parents=True)
        (out / "date=2026-09-15.parquet").write_bytes(b"")        # stalled
        (out / "date=2026-09-14.parquet").write_bytes(b"PAR1data")  # fine

        objects = [
            ("by_dex/hyperliquid/snapshots/perp/date=2026-09-14/a.parquet", 10),
            ("by_dex/hyperliquid/snapshots/perp/date=2026-09-15/b.parquet", 20),
        ]
        original = module.list_objects
        module.list_objects = lambda prefix, env=None: list(objects)
        try:
            plan = plan_dataset(
                TARGETS_BY_NAME["positions"], self.root, ("BTC",), None
            )
        finally:
            module.list_objects = original

        self.assertEqual([d.date for d in plan.days], ["2026-09-15"])
        self.assertEqual(plan.present, 1)


class TestProjectedFraction(unittest.TestCase):
    """The ratio the cost estimate and the spend cap both rest on.

    A silent default of 1.0 here is not a cosmetic bug: it overstated the
    archive by about 2.5x and would have made `--max-gb` refuse runs that were
    comfortably affordable.
    """

    def fraction(self, dataset, mode):
        from sync_reservoir import projected_fraction, TARGETS_BY_NAME

        return projected_fraction(dataset, TARGETS_BY_NAME[dataset].columns_for(mode))

    def test_fills_is_the_whole_reason_projection_matters(self):
        fraction = self.fraction("fills", "research")
        self.assertLess(fraction, 0.12)
        self.assertGreater(fraction, 0.10)

    def test_wide_mode_recovers_most_of_the_fill_volume(self):
        """Keeping the identifiers roughly quadruples the fills bill.

        The dropped columns are 57.1% of the file, so what remains is 42.9% -
        a ratio of 3.8, not the 5 that "57% is dropped" invites you to assume.
        """
        ratio = self.fraction("fills", "wide") / self.fraction("fills", "research")
        self.assertAlmostEqual(ratio, 3.82, places=1)
        self.assertGreater(ratio, 3.5)

    def test_candles_barely_compress(self):
        """Ten of thirteen columns are bulk numbers and the aggregation
        needs seven of them, so only `volume_quote` is truly skippable."""
        fraction = self.fraction("candles", "research")
        self.assertGreater(fraction, 0.70)
        self.assertLess(fraction, 0.73)

    def test_account_values_does_not_compress_at_all(self):
        """`user` alone is 91% of the file and cannot be dropped."""
        self.assertGreater(self.fraction("account_values", "research"), 0.97)

    def test_orderbook_keeps_every_column(self):
        """Its saving is by path - three markets out of ~178 - not by column."""
        self.assertEqual(self.fraction("orderbook", "wide"), 1.0)

    def test_unknown_dataset_defaults_to_no_saving(self):
        """Conservative: an unmeasured dataset must not promise a discount."""
        from sync_reservoir import projected_fraction

        self.assertEqual(projected_fraction("not_a_dataset", ("a",)), 1.0)

    def test_projection_actually_saves_across_the_archive(self):
        """Weighted by the real source sizes, not by intuition."""
        from sync_reservoir import (
            MEASURED_SOURCE_GB,
            TARGETS_BY_NAME,
            projected_fraction,
        )

        raw = sum(MEASURED_SOURCE_GB.values())
        projected = sum(
            gb * projected_fraction(
                name, TARGETS_BY_NAME[name].columns_for("research")
            )
            for name, gb in MEASURED_SOURCE_GB.items()
        )
        # 159.39 GB raw measured; projection should land it near 41 GB.
        self.assertAlmostEqual(raw, 159.39, places=2)
        self.assertLess(projected / raw, 0.30)
        self.assertLess(projected, 45.0)
        self.assertGreater(projected, 35.0)


class TestPlanBuilding(unittest.TestCase):
    """Planning, with the bucket stubbed.

    This path is otherwise only exercised against live AWS, which is how a
    missing dataclass default (`DatasetPlan.present`) survived until the first
    real dry run and then failed with a TypeError instead of a plan.
    """

    def setUp(self):
        import tempfile

        from sync_reservoir import TARGETS_BY_NAME

        self.target = TARGETS_BY_NAME["positions"]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Path(self.tmp.name)

    def plan(self, objects, limit=None, symbols=("BTC",)):
        from sync_reservoir import plan_dataset

        import sync_reservoir as module

        original = module.list_objects
        module.list_objects = lambda prefix, env=None: list(objects)
        try:
            return plan_dataset(self.target, self.store, tuple(symbols), limit)
        finally:
            module.list_objects = original

    def objects(self):
        return [
            ("by_dex/hyperliquid/snapshots/perp/date=2026-09-14/a.parquet", 100),
            ("by_dex/hyperliquid/snapshots/perp/date=2026-09-15/b.parquet", 200),
            ("by_dex/hyperliquid/snapshots/perp/date=2026-09-16/c.parquet", 300),
        ]

    def test_plans_every_missing_day(self):
        plan = self.plan(self.objects())
        self.assertEqual([d.date for d in plan.days],
                         ["2026-09-14", "2026-09-15", "2026-09-16"])
        self.assertEqual(plan.present, 0)
        self.assertEqual(plan.source_bytes, 600)

    def test_existing_days_are_skipped_not_refetched(self):
        """Resume is the whole point of a multi-hour backfill."""
        out = self.store / "positions"
        out.mkdir(parents=True)
        (out / "date=2026-09-15.parquet").write_bytes(b"x")
        plan = self.plan(self.objects())
        self.assertEqual([d.date for d in plan.days], ["2026-09-14", "2026-09-16"])
        self.assertEqual(plan.present, 1)

    def test_limit_caps_the_run(self):
        plan = self.plan(self.objects(), limit=2)
        self.assertEqual(len(plan.days), 2)

    def test_transfer_is_the_source_size_times_the_projection(self):
        plan = self.plan(self.objects())
        plan.fraction = 0.5
        self.assertEqual(plan.transfer_bytes, 300)

    def test_per_symbol_datasets_select_by_path(self):
        """The order book keeps one file per market, so symbols filter here."""
        from sync_reservoir import TARGETS_BY_NAME, plan_dataset
        import sync_reservoir as module

        objects = [
            ("by_dex/hyperliquid/orderbook/1m/perps/date=2026-09-16/BTC.parquet", 10),
            ("by_dex/hyperliquid/orderbook/1m/perps/date=2026-09-16/ETH.parquet", 20),
            ("by_dex/hyperliquid/orderbook/1m/perps/date=2026-09-16/DOGE.parquet", 40),
        ]
        original = module.list_objects
        module.list_objects = lambda prefix, env=None: list(objects)
        try:
            plan = plan_dataset(
                TARGETS_BY_NAME["orderbook"], self.store, ("BTC", "ETH"), None
            )
        finally:
            module.list_objects = original

        self.assertEqual(len(plan.days), 1)
        self.assertEqual(plan.source_bytes, 30)
        self.assertEqual(len(plan.days[0].keys), 2)

    def test_objects_without_a_partition_date_are_ignored(self):
        plan = self.plan([("by_dex/hyperliquid/snapshots/perp/README", 1)])
        self.assertEqual(plan.days, [])
        self.assertEqual(plan.source_bytes, 0)


class TestS3Path(unittest.TestCase):
    def test_builds_a_virtual_hosted_url(self):
        from sync_reservoir import BUCKET, s3_path

        self.assertEqual(
            s3_path("a/b.parquet"), f"s3://{BUCKET}/a/b.parquet"
        )


class TestCostArithmetic(unittest.TestCase):
    def test_free_allowance_is_per_month_and_does_not_carry(self):
        self.assertEqual(egress_cost_usd(80.0, months=1), 0.0)
        self.assertGreater(egress_cost_usd(180.0, months=1), 0.0)
        self.assertEqual(egress_cost_usd(180.0, months=2), 0.0)

    def test_months_to_stay_free_uses_ceiling(self):
        self.assertEqual(months_to_stay_free(63.58), 1)
        self.assertEqual(months_to_stay_free(100.0), 1)
        self.assertEqual(months_to_stay_free(100.1), 2)
        self.assertEqual(months_to_stay_free(214.6), 3)

    def test_measured_archive_fits_one_month_after_projection(self):
        """63.58 GB projected against 214.60 GB raw, from the real column sizes."""
        self.assertEqual(egress_cost_usd(63.58, months=1), 0.0)
        self.assertGreater(egress_cost_usd(214.60, months=1), 0.0)

    def test_human_gb_uses_binary_units(self):
        self.assertEqual(human_gb(GB), "1.00 GB")
        self.assertEqual(to_gb(0), 0.0)


class TestDateExtraction(unittest.TestCase):
    def test_pulls_the_partition_date(self):
        self.assertEqual(
            date_from_key(
                "by_dex/hyperliquid/candles/1s/date=2026-09-16/candles.parquet"
            ),
            "2026-09-16",
        )

    def test_handles_a_nested_name(self):
        self.assertEqual(
            date_from_key("global/snapshots/account_values/date=2025-07-29/x.parquet"),
            "2025-07-29",
        )

    def test_returns_empty_when_there_is_no_partition(self):
        self.assertEqual(date_from_key("some/other/key.parquet"), "")


class TestCopySql(unittest.TestCase):
    """The generated statement must project, and must not scan more than asked."""

    def target(self):
        return TARGETS_BY_NAME["candles"]

    def test_selects_only_the_requested_columns(self):
        sql = build_copy_sql(
            self.target(),
            ["s3://b/a.parquet"],
            ("coin", "timestamp", "close"),
            Path("out.parquet"),
            glob=False,
        )
        self.assertIn("SELECT coin, timestamp, close", sql)
        self.assertNotIn("volume_quote", sql)
        self.assertIn("FORMAT PARQUET", sql)

    def test_lists_multiple_files_explicitly(self):
        sql = build_copy_sql(
            self.target(),
            ["s3://b/a.parquet", "s3://b/b.parquet"],
            ("coin",),
            Path("out.parquet"),
            glob=False,
        )
        self.assertIn("['s3://b/a.parquet', 's3://b/b.parquet']", sql)

    def test_every_selected_market_is_read(self):
        """The regression, stated directly.

        This replaces `test_glob_form_reads_one_path`, which asserted that the
        per-market form reads a single path - and passed a wildcard pattern as
        that path, when the keys actually reaching this function come from a
        listing and are complete object paths. So the test encoded the wrong
        assumption and made a live bug invisible: over 282 days,
        `--symbols BTC,ETH,SOL` wrote BTC alone at 81 MB where three markets
        are 240 MB. The one-market case looked correct, which is why a
        single-symbol probe cannot find it.
        """
        sql = build_copy_sql(
            TARGETS_BY_NAME["orderbook"],
            ["s3://b/date=2026-09-22/BTC.parquet", "s3://b/date=2026-09-22/ETH.parquet"],
            ("block_time_ms", "bids", "asks"),
            Path("out.parquet"),
            glob=True,
        )
        self.assertIn("'s3://b/date=2026-09-22/BTC.parquet'", sql)
        self.assertIn("'s3://b/date=2026-09-22/ETH.parquet'", sql)

    def test_the_market_is_derived_from_the_file_name(self):
        """The source row has no market in it, only the path does.

        `datalake.orderbook_from_hydromancer` reads `row["symbol"]`: without
        this column every book normalises to an empty symbol and says nothing.
        """
        sql = build_copy_sql(
            TARGETS_BY_NAME["orderbook"],
            ["s3://b/date=2026-09-22/BTC.parquet"],
            ("block_time_ms", "bids", "asks"),
            Path("out.parquet"),
            glob=True,
        )
        self.assertIn("AS symbol", sql)
        self.assertIn("filename=true", sql)

    def test_a_projected_dataset_gets_no_derived_market(self):
        """`symbol` is a property of the file name, so it is added only where
        the file name is the only place the market appears."""
        sql = build_copy_sql(
            self.target(), ["s3://b/a.parquet"], ("coin", "close"),
            Path("out.parquet"), glob=False,
        )
        self.assertNotIn("AS symbol", sql)
        self.assertNotIn("filename=true", sql)

    def test_destination_is_quoted(self):
        sql = build_copy_sql(
            self.target(), ["s3://b/a.parquet"], ("coin",),
            Path("/tmp/dir with space/out.parquet"), glob=False,
        )
        self.assertIn("'/tmp/dir with space/out.parquet'", sql)


if __name__ == "__main__":
    unittest.main(verbosity=2)
