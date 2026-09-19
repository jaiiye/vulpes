"""Canonical schema and source adapters.

The load-bearing tests here are the derivation ones. `unrealized_pnl` and
`mark_price` are not stored by the archive, so they are recomputed from
`notional` and `entry_price`, and a sign error in that reconstruction would not
raise - it would quietly report every short as being on the wrong side of the
market. The API does report its own PnL, which lets the tests check the
derivation against an independent figure instead of against itself.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datalake.normalize import (  # noqa: E402
    NormalizeError,
    account_value_from_hydromancer,
    candle_from_api,
    candle_from_hydromancer,
    funding_from_api,
    orderbook_from_api,
    orderbook_from_hydromancer,
    position_from_api,
    position_from_hydromancer,
)
from datalake.schema import (  # noqa: E402
    CANONICAL_DATASETS,
    SOURCE_API,
    SOURCE_HYDROMANCER,
    Candle,
    OrderBookSnapshot,
    PositionSnapshot,
    as_float,
    dataset_for,
    from_row,
    to_row,
)

DAY_MS = 86_400_000


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------


class TestPositionDerivations(unittest.TestCase):
    """`mark_price` and `unrealized_pnl` are reconstructed, not stored."""

    def test_long_in_profit(self):
        # 2 BTC at 100, now worth 110: notional 220, pnl +20.
        pos = PositionSnapshot(
            time_ms=0, user="0xa", market="BTC", size=2.0, notional=220.0,
            entry_price=100.0, source=SOURCE_API,
        )
        self.assertAlmostEqual(pos.mark_price, 110.0)
        self.assertAlmostEqual(pos.unrealized_pnl, 20.0)

    def test_long_at_a_loss(self):
        pos = PositionSnapshot(
            time_ms=0, user="0xa", market="BTC", size=2.0, notional=180.0,
            entry_price=100.0, source=SOURCE_API,
        )
        self.assertAlmostEqual(pos.mark_price, 90.0)
        self.assertAlmostEqual(pos.unrealized_pnl, -20.0)

    def test_short_in_profit(self):
        """Both `notional` and `size` are negative, so the signs must cancel.

        This is the case that breaks silently: a formula that only works for
        longs reports this short as losing 20 instead of winning 20.
        """
        pos = PositionSnapshot(
            time_ms=0, user="0xa", market="BTC", size=-2.0, notional=-180.0,
            entry_price=100.0, source=SOURCE_HYDROMANCER,
        )
        self.assertAlmostEqual(pos.mark_price, 90.0)
        self.assertAlmostEqual(pos.unrealized_pnl, 20.0)
        self.assertEqual(pos.side, "short")

    def test_short_at_a_loss(self):
        pos = PositionSnapshot(
            time_ms=0, user="0xa", market="BTC", size=-2.0, notional=-220.0,
            entry_price=100.0, source=SOURCE_HYDROMANCER,
        )
        self.assertAlmostEqual(pos.mark_price, 110.0)
        self.assertAlmostEqual(pos.unrealized_pnl, -20.0)

    def test_flat_position_has_no_mark(self):
        pos = PositionSnapshot(
            time_ms=0, user="0xa", market="BTC", size=0.0, notional=0.0,
            entry_price=100.0, source=SOURCE_API,
        )
        self.assertEqual(pos.mark_price, 0.0)
        self.assertEqual(pos.side, "flat")

    def test_derivation_agrees_with_the_exchange_reported_pnl(self):
        """Checked against an independent figure rather than against itself.

        The API reports `unrealizedPnl`; the canonical record ignores it and
        recomputes. If the two ever diverge, the recomputation is wrong and
        every historical row is wrong with it.
        """
        cases = [
            # (szi, entryPx, positionValue, reported pnl)
            (2.0, "100", "220", "20"),
            (2.0, "100", "180", "-20"),
            (-2.0, "100", "180", "20"),
            (-2.0, "100", "220", "-20"),
            (0.5, "30000", "16250", "1250"),
            (-1.25, "2000", "2750", "-250"),
        ]
        for szi, entry, value, reported in cases:
            with self.subTest(szi=szi, value=value):
                row = {
                    "coin": "BTC", "szi": szi, "entryPx": entry,
                    "positionValue": value, "unrealizedPnl": reported,
                }
                pos = position_from_api("0xAbC", row, time_ms=0)
                self.assertAlmostEqual(
                    pos.unrealized_pnl, float(reported), places=6
                )


# ---------------------------------------------------------------------------
# Sign conventions
# ---------------------------------------------------------------------------


class TestSignNormalisation(unittest.TestCase):
    """The archive does not document whether its notional is signed.

    Both possibilities must land on the same canonical record, because the
    alternative is a store where half the rows mean one thing and half mean
    another, decided by whichever source wrote them.
    """

    def test_unsigned_archive_notional_is_signed_from_size(self):
        row = {
            "user": "0xAbC", "market": "BTC", "size": -2.0,
            "notional": 180.0,          # magnitude only
            "entry_price": 100.0,
        }
        pos = position_from_hydromancer(row, time_ms=DAY_MS)
        self.assertEqual(pos.notional, -180.0)
        self.assertAlmostEqual(pos.mark_price, 90.0)
        self.assertAlmostEqual(pos.unrealized_pnl, 20.0)

    def test_already_signed_archive_notional_is_unchanged(self):
        row = {
            "user": "0xAbC", "market": "BTC", "size": -2.0,
            "notional": -180.0,         # already negative
            "entry_price": 100.0,
        }
        pos = position_from_hydromancer(row, time_ms=DAY_MS)
        self.assertEqual(pos.notional, -180.0)

    def test_both_encodings_produce_identical_records(self):
        base = {
            "user": "0xAbC", "market": "BTC", "size": -2.0, "entry_price": 100.0
        }
        unsigned = position_from_hydromancer(
            {**base, "notional": 180.0}, time_ms=DAY_MS
        )
        signed = position_from_hydromancer(
            {**base, "notional": -180.0}, time_ms=DAY_MS
        )
        self.assertEqual(unsigned, signed)

    def test_zero_size_position_is_rejected(self):
        """A flat row carries nothing and would pollute the positions table."""
        with self.assertRaises(NormalizeError):
            position_from_hydromancer(
                {"user": "0xa", "market": "BTC", "size": 0.0, "notional": 0.0},
                time_ms=DAY_MS,
            )


# ---------------------------------------------------------------------------
# Cross-source consistency
# ---------------------------------------------------------------------------


class TestCrossSourceConsistency(unittest.TestCase):
    """History and live data must be comparable.

    The 40%-weight factor was never backtested because its inputs only existed
    live. Now that history exists, the two stretches have to agree on what a
    record means, or the comparison is between conventions rather than markets.
    """

    def test_same_position_from_each_source_derives_the_same_values(self):
        api_row = {
            "coin": "ETH", "szi": -1.25, "entryPx": "2000",
            "positionValue": "2750", "unrealizedPnl": "-250",
            "liquidationPx": "2600",
            "leverage": {"type": "cross", "value": 10},
            "cumFunding": {"allTime": "-12.5"},
        }
        api_pos = position_from_api("0xAbC", api_row, time_ms=DAY_MS)

        archive_row = {
            "user": "0xAbC", "market": "ETH", "size": -1.25,
            "notional": 2750.0, "entry_price": 2000.0,
            "liquidation_price": 2600.0, "leverage": 10,
            "leverage_type": "cross", "funding_pnl": -12.5,
        }
        archive_pos = position_from_hydromancer(archive_row, time_ms=DAY_MS)

        self.assertEqual(api_pos.market, archive_pos.market)
        self.assertEqual(api_pos.notional, archive_pos.notional)
        self.assertAlmostEqual(api_pos.mark_price, archive_pos.mark_price)
        self.assertAlmostEqual(
            api_pos.unrealized_pnl, archive_pos.unrealized_pnl
        )
        self.assertEqual(api_pos.leverage, archive_pos.leverage)
        self.assertEqual(api_pos.leverage_type, archive_pos.leverage_type)
        # Only the provenance differs.
        self.assertNotEqual(api_pos.source, archive_pos.source)

    def test_user_addresses_are_lower_cased_from_both_sources(self):
        """Mixed case would split one wallet into two identities."""
        api_pos = position_from_api(
            "0xABCDEF", {"coin": "BTC", "szi": 1.0, "entryPx": "1",
                         "positionValue": "1"}, time_ms=0,
        )
        archive_pos = position_from_hydromancer(
            {"user": "0xABCDEF", "market": "BTC", "size": 1.0,
             "notional": 1.0, "entry_price": 1.0}, time_ms=0,
        )
        self.assertEqual(api_pos.user, "0xabcdef")
        self.assertEqual(api_pos.user, archive_pos.user)


# ---------------------------------------------------------------------------
# Candles
# ---------------------------------------------------------------------------


class TestCandleAdapters(unittest.TestCase):
    def test_api_single_letter_keys_map_to_the_right_columns(self):
        row = {"t": 1_700_000_000_000, "T": 1_700_003_600_000, "s": "btc",
               "i": "1h", "o": "100", "h": "110", "l": "95", "c": "105",
               "v": "12.5", "n": 42}
        candle = candle_from_api(row)
        self.assertEqual(candle.symbol, "BTC")
        self.assertEqual(candle.interval, "1h")
        self.assertEqual((candle.open, candle.high, candle.low, candle.close),
                         (100.0, 110.0, 95.0, 105.0))
        self.assertEqual(candle.volume, 12.5)
        self.assertEqual(candle.trade_count, 42)
        self.assertEqual(candle.source, SOURCE_API)
        # The API does not offer a quote-denominated volume.
        self.assertIsNone(candle.volume_quote)

    def test_archive_candle_is_always_one_second(self):
        """Coarser bars come from aggregating, not from a different file."""
        row = {"coin": "BTC", "timestamp": 1_700_000_000_000, "open": "100",
               "high": "101", "low": "99", "close": "100.5",
               "volume": "0.25", "volume_quote": "25.0", "trade_count": 3}
        candle = candle_from_hydromancer(row)
        self.assertEqual(candle.interval, "1s")
        self.assertEqual(candle.source, SOURCE_HYDROMANCER)
        self.assertEqual(candle.volume_quote, 25.0)

    def test_transposed_high_and_low_is_rejected(self):
        """The single-letter keys make this the likely mistake, and a bar with
        h and l swapped still looks like a plausible bar downstream."""
        row = {"t": 1, "s": "BTC", "i": "1h",
               "o": "100", "h": "95", "l": "110", "c": "105", "v": "1"}
        with self.assertRaises(NormalizeError):
            candle_from_api(row)

    def test_open_outside_the_bar_range_is_rejected(self):
        row = {"t": 1, "s": "BTC", "i": "1h",
               "o": "120", "h": "110", "l": "95", "c": "105", "v": "1"}
        with self.assertRaises(NormalizeError):
            candle_from_api(row)

    def test_non_positive_price_is_rejected(self):
        row = {"t": 1, "s": "BTC", "i": "1h",
               "o": "0", "h": "110", "l": "0", "c": "105", "v": "1"}
        with self.assertRaises(NormalizeError):
            candle_from_api(row)


# ---------------------------------------------------------------------------
# Funding, account values, order book
# ---------------------------------------------------------------------------


class TestOtherAdapters(unittest.TestCase):
    def test_funding_from_api(self):
        rate = funding_from_api(
            {"coin": "btc", "fundingRate": "0.0000125", "time": 1_700_000_000_000}
        )
        self.assertEqual(rate.symbol, "BTC")
        self.assertAlmostEqual(rate.rate, 0.0000125)

    def test_funding_without_a_rate_is_rejected(self):
        with self.assertRaises(NormalizeError):
            funding_from_api({"coin": "BTC", "time": 1})

    def test_account_value_from_archive(self):
        snap = account_value_from_hydromancer(
            {"user": "0xAbC", "dex": "hyperliquid", "collateral_token": "USDC",
             "account_value": 250_000.0, "total_long_notional": 500_000.0,
             "total_short_notional": 120_000.0},
            time_ms=DAY_MS,
        )
        self.assertEqual(snap.user, "0xabc")
        self.assertEqual(snap.account_value, 250_000.0)
        self.assertEqual(snap.total_short_notional, 120_000.0)

    def test_orderbook_from_archive_string_decimals(self):
        row = {
            "block_time_ms": 1_700_000_000_000,
            "block_number": 926030000,
            "bids": [{"px": "100.5", "sz": "2", "n": 3},
                     {"px": "100.0", "sz": "5", "n": 1}],
            "asks": [{"px": "101.0", "sz": "4", "n": 2}],
        }
        book = orderbook_from_hydromancer(row)
        self.assertEqual(book.best_bid, 100.5)
        self.assertEqual(book.best_ask, 101.0)
        self.assertEqual(book.block_number, 926030000)
        self.assertEqual(book.bids[1].order_count, 1)

    def test_orderbook_from_api_level_shape(self):
        row = {"coin": "ETH", "time": 1_700_000_000_000,
               "levels": [[{"px": "2000", "sz": "1.5", "n": 2}],
                          [{"px": "2001", "sz": "3", "n": 4}]]}
        book = orderbook_from_api(row)
        self.assertEqual(book.symbol, "ETH")
        self.assertEqual(book.best_bid, 2000.0)
        self.assertEqual(book.best_ask, 2001.0)

    def test_thin_book_is_not_padded(self):
        """Padding would be read as real resting size by a slippage model."""
        book = orderbook_from_hydromancer(
            {"block_time_ms": 1, "bids": [{"px": "100", "sz": "1"}],
             "asks": [{"px": "101", "sz": "1"}]}
        )
        self.assertEqual(len(book.bids), 1)
        self.assertEqual(len(book.asks), 1)


class TestOrderBookProperties(unittest.TestCase):
    def book(self):
        return orderbook_from_hydromancer({
            "block_time_ms": 0,
            "bids": [{"px": "99.99", "sz": "10"}, {"px": "99.90", "sz": "20"}],
            "asks": [{"px": "100.01", "sz": "5"}, {"px": "100.10", "sz": "30"}],
        })

    def test_mid_price(self):
        self.assertAlmostEqual(self.book().mid_price, 100.0)

    def test_spread_in_basis_points(self):
        # (100.01 - 99.99) / 100 = 0.0002 -> 2 bps
        self.assertAlmostEqual(self.book().spread_bps, 2.0, places=6)

    def test_depth_counts_only_levels_within_the_band(self):
        book = self.book()
        # 1 bp of 100 is 0.01, so the 99.99 bid qualifies and 99.90 does not.
        self.assertAlmostEqual(book.depth_notional("bid", 1.0), 99.99 * 10)
        # 10 bps is 0.10, so both bid levels qualify.
        self.assertAlmostEqual(
            book.depth_notional("bid", 10.0), 99.99 * 10 + 99.90 * 20
        )

    def test_empty_side_reports_nothing_rather_than_zero_spread(self):
        """A one-sided book has no spread; returning 0.0 would read as
        perfectly liquid."""
        book = orderbook_from_hydromancer(
            {"block_time_ms": 0, "bids": [{"px": "100", "sz": "1"}], "asks": []}
        )
        self.assertIsNone(book.spread_bps)
        self.assertIsNone(book.mid_price)
        self.assertEqual(book.depth_notional("ask", 10.0), 0.0)


# ---------------------------------------------------------------------------
# Coercion and serialisation
# ---------------------------------------------------------------------------


class TestCoercion(unittest.TestCase):
    def test_non_finite_values_are_rejected(self):
        """`nan` compares false against everything, so it would disable every
        guard that reads it instead of raising."""
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                self.assertIsNone(as_float(bad))
                self.assertEqual(as_float(bad, 0.0), 0.0)

    def test_booleans_are_not_numbers(self):
        """`True` is a valid float in Python and would become a price of 1.0."""
        self.assertIsNone(as_float(True))
        self.assertIsNone(as_float(False))

    def test_numeric_strings_are_accepted(self):
        """Both sources emit decimals as strings to preserve precision."""
        self.assertEqual(as_float("0.0000125"), 0.0000125)
        self.assertEqual(as_float("  42  "), 42.0)

    def test_garbage_falls_back(self):
        self.assertIsNone(as_float("abc"))
        self.assertIsNone(as_float(None))
        self.assertEqual(as_float("abc", 7.0), 7.0)


class TestRowRoundTrip(unittest.TestCase):
    def test_position_survives_a_round_trip(self):
        original = position_from_hydromancer(
            {"user": "0xAbC", "market": "BTC", "size": -2.0,
             "notional": 180.0, "entry_price": 100.0,
             "liquidation_price": 130.0, "leverage": 5,
             "leverage_type": "isolated", "funding_pnl": -3.5,
             "account_value": 50_000.0},
            time_ms=DAY_MS,
        )
        restored = from_row("positions", to_row(original))
        self.assertEqual(restored, original)
        self.assertAlmostEqual(restored.unrealized_pnl, original.unrealized_pnl)

    def test_derived_fields_are_not_written_to_storage(self):
        """Storing them would freeze today's formula into yesterday's rows."""
        row = to_row(position_from_hydromancer(
            {"user": "0xa", "market": "BTC", "size": 1.0,
             "notional": 100.0, "entry_price": 90.0},
            time_ms=0,
        ))
        self.assertNotIn("unrealized_pnl", row)
        self.assertNotIn("mark_price", row)
        self.assertNotIn("side", row)
        self.assertIn("notional", row)

    def test_unknown_columns_are_dropped_not_fatal(self):
        """A row carrying a column this version does not know must still load,
        otherwise a newer writer makes the store unreadable to the reader."""
        original = Candle(symbol="BTC", interval="1h", time_ms=1, open=1.0,
                          high=2.0, low=0.5, close=1.5, volume=3.0,
                          source=SOURCE_API)
        row = to_row(original)
        row["column_added_later"] = "whatever"
        self.assertEqual(from_row("candles", row), original)

    def test_nested_order_book_levels_round_trip(self):
        book = orderbook_from_hydromancer({
            "block_time_ms": 5, "bids": [{"px": "100", "sz": "2", "n": 3}],
            "asks": [{"px": "101", "sz": "4", "n": 1}],
        })
        restored = from_row("orderbook", to_row(book))
        self.assertEqual(restored, book)
        self.assertEqual(restored.spread_bps, book.spread_bps)

    def test_unknown_dataset_name_is_rejected(self):
        with self.assertRaises(KeyError):
            from_row("candlesticks", {})

    def test_every_registered_dataset_has_a_working_adapter_path(self):
        """Guards against registering a dataset nothing can populate."""
        self.assertEqual(
            sorted(CANONICAL_DATASETS),
            ["account_values", "candles", "funding", "leaderboard",
             "orderbook", "positions"],
        )
        for name in CANONICAL_DATASETS:
            self.assertIsNotNone(dataset_for(name))


if __name__ == "__main__":
    unittest.main(verbosity=2)
