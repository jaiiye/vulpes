"""Cost and volume arithmetic in the Reservoir probe.

These are pure functions, so they are testable without touching AWS. That
matters: the whole point of the probe is to decide how much money to spend, and
an arithmetic error there is a planning error with a bill attached.

`TestPerDayVolume` exists because that exact error was made. Dividing a day's
total by its object count reads as "average file size", which is the right
number for nothing and understated the order book by a factor of 178, because
that dataset stores one small file per coin per day.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probe_reservoir import (  # noqa: E402
    FREE_EGRESS_GB_PER_MONTH,
    Dataset,
    egress_cost_usd,
    human_gb,
    months_needed_to_stay_free,
    projected_bytes,
    projected_fraction,
    request_cost_usd,
    to_gb,
)

GB = 1024**3


class TestUnitConversion(unittest.TestCase):
    def test_binary_gigabytes(self):
        """AWS meters S3 in binary GB, so 1 GB here must be 2**30 bytes."""
        self.assertEqual(to_gb(GB), 1.0)
        self.assertEqual(to_gb(0), 0.0)

    def test_human_readable(self):
        self.assertEqual(human_gb(2 * GB), "2.00 GB")


class TestEgressCost(unittest.TestCase):
    """The monthly allowance does not roll over, so timing is the whole cost."""

    def test_under_the_allowance_is_free(self):
        self.assertEqual(egress_cost_usd(50.0, months=1), 0.0)
        self.assertEqual(egress_cost_usd(FREE_EGRESS_GB_PER_MONTH, months=1), 0.0)

    def test_only_the_excess_is_billed(self):
        # 150 GB in one month: 50 GB billable.
        cost = egress_cost_usd(150.0, months=1)
        self.assertAlmostEqual(cost, 50.0 * 0.114, places=6)

    def test_spreading_over_months_can_make_it_free(self):
        self.assertGreater(egress_cost_usd(250.0, months=1), 0.0)
        self.assertEqual(egress_cost_usd(250.0, months=3), 0.0)

    def test_degenerate_inputs_are_zero_not_errors(self):
        self.assertEqual(egress_cost_usd(0.0, months=1), 0.0)
        self.assertEqual(egress_cost_usd(-5.0, months=1), 0.0)
        self.assertEqual(egress_cost_usd(500.0, months=0), 0.0)


class TestMonthsToStayFree(unittest.TestCase):
    def test_one_month_when_already_under_the_allowance(self):
        self.assertEqual(months_needed_to_stay_free(10.0), 1)
        self.assertEqual(months_needed_to_stay_free(100.0), 1)

    def test_adds_a_month_past_the_boundary(self):
        """Exactly 100 GB fits; a byte more does not."""
        self.assertEqual(months_needed_to_stay_free(100.1), 2)

    def test_even_multiples_do_not_waste_a_month(self):
        """200 GB over two months is 100 GB each, which is still free.

        Floor-plus-one returned 3 here, which is how the boundary test above
        caught a wasted month in a formula that looked obviously right.
        """
        self.assertEqual(months_needed_to_stay_free(200.0), 2)
        self.assertEqual(months_needed_to_stay_free(300.0), 3)
        self.assertEqual(months_needed_to_stay_free(250.0), 3)

    def test_zero_volume_needs_no_spreading(self):
        self.assertEqual(months_needed_to_stay_free(0.0), 1)


class TestRequestCost(unittest.TestCase):
    def test_requests_are_negligible(self):
        self.assertLess(request_cost_usd(50_000), 0.05)
        self.assertEqual(request_cost_usd(0), 0.0)


class TestPerDayVolume(unittest.TestCase):
    """The regression: a day's total must not be divided by its object count."""

    def test_single_object_day(self):
        ds = Dataset(name="candles", prefix="p", note="", days=414)
        ds.sample_bytes = 33 * 1024 * 1024
        ds.sample_objects = 1
        self.assertAlmostEqual(ds.per_day_bytes, 33 * 1024 * 1024)
        self.assertAlmostEqual(ds.total_bytes(), 33 * 1024 * 1024 * 414)

    def test_many_small_files_in_one_day(self):
        """178 per-coin files totalling 44 MB is a 44 MB day, not a 0.25 MB one.

        Treating it as the average file size understated the whole dataset by
        the object count, turning 12 GB into 0.07 GB.
        """
        ds = Dataset(name="orderbook", prefix="p", note="", per_coin=True, days=277)
        ds.sample_bytes = 44 * 1024 * 1024
        ds.sample_objects = 178
        self.assertAlmostEqual(ds.per_day_bytes, 44 * 1024 * 1024)
        total = ds.total_bytes()
        self.assertGreater(to_gb(total), 10.0)
        self.assertAlmostEqual(total, 44 * 1024 * 1024 * 277)

    def test_object_count_scales_with_coverage(self):
        ds = Dataset(name="orderbook", prefix="p", note="", days=277)
        ds.sample_objects = 178
        self.assertEqual(ds.total_objects(), 178 * 277)

    def test_unmeasured_dataset_reports_zero(self):
        ds = Dataset(name="x", prefix="p", note="")
        self.assertEqual(ds.per_day_bytes, 0.0)
        self.assertEqual(ds.total_bytes(), 0.0)
        self.assertEqual(ds.total_objects(), 0)


class TestPrioritySubsetCost(unittest.TestCase):
    """The three datasets we would actually take first."""

    def test_priority_subset_stays_inside_one_month_of_free_egress(self):
        """candles + positions + three-coin order book, from the measured sizes.

        If this ever exceeds the monthly allowance the plan changes, so it is
        worth failing loudly rather than discovering it on an invoice.
        """
        candles = Dataset(name="candles", prefix="p", note="", days=414)
        candles.sample_bytes = 33 * 1024 * 1024
        candles.sample_objects = 1

        positions = Dataset(name="positions", prefix="p", note="", days=364)
        positions.sample_bytes = 19 * 1024 * 1024
        positions.sample_objects = 1

        # Only three coins, out of the 178 files a day holds.
        orderbook = Dataset(
            name="orderbook", prefix="p", note="", per_coin=True, days=277
        )
        orderbook.sample_bytes = 44 * 1024 * 1024 * 3 / 178
        orderbook.sample_objects = 3

        total_gb = to_gb(sum(d.total_bytes() for d in (candles, positions, orderbook)))
        self.assertLess(total_gb, FREE_EGRESS_GB_PER_MONTH)
        self.assertEqual(egress_cost_usd(total_gb, months=1), 0.0)


# Column-level compressed sizes read from each file's own footer with
# `parquet_metadata`. These are measurements of one archive day, not estimates,
# and they are the basis for the projection decision - so they are pinned here
# rather than left in a report that will scroll away.

CANDLES_COLUMNS = {
    "volume_quote": 8_731_899,
    "volume": 5_515_570,
    "close": 3_704_413,
    "low": 3_694_532,
    "high": 3_682_079,
    "open": 3_680_466,
    "timestamp": 3_434_808,
    "trade_count": 741_571,
    "base_symbol": 10_513,
    "coin": 9_963,
    "quote_symbol": 2_139,
    "asset_class": 1_656,
    "dex": 1_585,
}
CANDLES_FILE_BYTES = 33_242_474

FILLS_COLUMNS = {
    "tx_hash": 109_397_747,
    "client_order_id": 70_514_540,
    "start_position": 47_830_412,
    "trade_id": 33_404_661,
    "fee": 33_256_328,
    "order_id": 24_887_420,
    "realized_pnl": 24_725_516,
    "size": 21_859_831,
    "price": 15_536_364,
    "address": 15_121_862,
    "timestamp": 7_233_159,
    "base_symbol": 3_346_514,
    "coin": 3_343_591,
    "builder_fee": 2_675_109,
    "priority_gas": 2_292_516,
    "direction": 1_943_791,
    "builder": 848_227,
    "twap_id": 669_567,
    "crossed": 581_655,
    "is_liquidation": 51_401,
    "side": 47_468,
    "liquidation_mark_px": 31_380,
    "liquidation_method": 13_978,
    "quote_symbol": 10_545,
    "deployer_fee": 9_123,
    "dex": 8_080,
    "fee_token": 8_031,
    "asset_class": 8_031,
}
FILLS_FILE_BYTES = 420_102_763

CANDLE_NEEDED = {"coin", "timestamp", "open", "high", "low", "close", "volume"}
# A leaderboard rebuild needs a wallet, a time and a realised PnL. Nothing else.
FILL_NEEDED = {"address", "timestamp", "realized_pnl"}


class TestColumnProjection(unittest.TestCase):
    """How much a column-projected read actually avoids."""

    def test_candle_projection_saves_about_a_quarter(self):
        """Less than it looks like it should.

        Ten of the thirteen columns are bulk numbers, and the hourly
        aggregation needs seven of them. Only `volume_quote` is genuinely
        skippable, and it happens to be the largest.
        """
        fraction = projected_fraction(CANDLES_COLUMNS, CANDLE_NEEDED)
        self.assertGreater(fraction, 0.70)
        self.assertLess(fraction, 0.73)
        self.assertEqual(projected_bytes(CANDLES_COLUMNS, CANDLE_NEEDED), 23_721_831)

    def test_dropping_volume_quote_is_most_of_the_candle_saving(self):
        with_quote = projected_fraction(
            CANDLES_COLUMNS, CANDLE_NEEDED | {"volume_quote"}
        )
        without = projected_fraction(CANDLES_COLUMNS, CANDLE_NEEDED)
        self.assertGreater(with_quote - without, 0.25)

    def test_fill_projection_saves_nearly_ninety_percent(self):
        """The decisive number: fills is 76% of the whole archive by volume.

        Two columns nobody needs for research - a transaction hash and a client
        order id - are 43% of the file on their own.
        """
        fraction = projected_fraction(FILLS_COLUMNS, FILL_NEEDED)
        self.assertLess(fraction, 0.12)
        self.assertGreater(fraction, 0.10)
        self.assertEqual(projected_bytes(FILLS_COLUMNS, FILL_NEEDED), 47_080_537)

    def test_the_two_useless_fill_columns_are_the_reason(self):
        overhead = FILLS_COLUMNS["tx_hash"] + FILLS_COLUMNS["client_order_id"]
        self.assertGreater(overhead / FILLS_FILE_BYTES, 0.42)

    def test_degenerate_inputs(self):
        self.assertEqual(projected_fraction({}, {"a"}), 0.0)
        self.assertEqual(projected_fraction({"a": 10}, set()), 0.0)
        self.assertEqual(projected_fraction({"a": 10}, {"missing"}), 0.0)


class TestProjectionAtArchiveScale(unittest.TestCase):
    """What the projection does to the bill."""

    def test_projected_archive_fits_inside_one_month_of_free_egress(self):
        """The point of the whole exercise.

        Unprojected, the archive is 215 GB and costs real money in one month.
        Projected, it is under the 100 GB allowance and costs nothing - which
        is what makes replaying the 40%-weight factor affordable at all.
        """
        totals = {
            "candles": (12.82, CANDLE_NEEDED, CANDLES_COLUMNS),
            "positions": (
                6.86,
                {"user", "market", "size", "notional", "entry_price"},
                {
                    "user": 10_177_569, "funding_pnl": 2_336_138,
                    "liquidation_price": 1_840_514, "notional": 1_700_751,
                    "entry_price": 1_583_455, "account_value": 1_286_787,
                    "size": 1_056_019, "leverage": 155_144,
                    "account_mode": 51_991, "leverage_type": 38_174,
                    "market": 2_024,
                },
            ),
            "account_values": (
                31.77,
                {"user", "dex", "account_value"},
                {
                    "user": 85_161_328, "account_value": 6_299_818,
                    "total_long_notional": 1_279_176,
                    "total_short_notional": 604_949, "account_mode": 317_259,
                    "dex": 2_839, "collateral_token": 2_736,
                },
            ),
            "fills": (163.15, FILL_NEEDED, FILLS_COLUMNS),
        }

        unprojected = sum(size for size, _, _ in totals.values())
        projected = sum(
            size * projected_fraction(columns, needed)
            for size, needed, columns in totals.values()
        )

        # These are the datasets with meaningful projection; a three-coin order
        # book adds about 0.2 GB either way.
        self.assertAlmostEqual(unprojected, 214.60, places=1)
        self.assertLess(projected, FREE_EGRESS_GB_PER_MONTH)
        self.assertGreater(projected, 55.0)
        self.assertLess(projected / unprojected, 0.32)

        # The 76%-of-volume dataset becomes a minor line item.
        fills_before = totals["fills"][0]
        fills_after = fills_before * projected_fraction(
            FILLS_COLUMNS, FILL_NEEDED
        )
        self.assertLess(fills_after, 20.0)
        self.assertLess(fills_after, fills_before / 8)

        self.assertEqual(egress_cost_usd(projected, months=1), 0.0)
        self.assertGreater(egress_cost_usd(unprojected, months=1), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
