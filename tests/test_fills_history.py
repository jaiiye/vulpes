"""The parts of `fills_history` that can be checked without the archive.

The SQL itself is exercised against the real store by whoever runs a backtest;
what is tested here is the arithmetic around it - window bounds and partition
selection - plus the two validation gates. Both gates matter for the same
reason: a wallet list is interpolated into the query text, so a value that
should have been rejected does not raise, it changes what the statement means.

`TestFileSelection` is not book-keeping. Reading the wrong partitions is silent
- the ranking still returns wallets, just the wrong ones - and the failure mode
would look like a finding.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import fills_history as fh  # noqa: E402


def ms(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=timezone.utc).timestamp() * 1000)


class TestWindowBounds(unittest.TestCase):
    def test_lower_bound_sits_days_back_and_is_exclusive(self):
        as_of = 100 * fh._DAY_MS
        self.assertEqual(fh._window_lower_ms(as_of, 7), as_of - 7 * fh._DAY_MS)
        self.assertEqual(fh._window_lower_ms(as_of, 1), as_of - fh._DAY_MS)

    def test_a_one_day_window_does_not_reach_two_days_back(self):
        as_of = 10 * fh._DAY_MS
        self.assertGreater(fh._window_lower_ms(as_of, 1), as_of - 2 * fh._DAY_MS)


class TestFileSelection(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _touch(self, *days: str) -> None:
        for day in days:
            (self.root / f"date={day}.parquet").write_bytes(b"")

    def test_only_existing_days_are_listed(self):
        self._touch("2026-03-08", "2026-03-09", "2026-03-10")
        picked = fh._files_for_window(self.root, ms(2026, 3, 10), 3)
        names = [Path(p.strip("'")).name for p in picked]
        self.assertEqual(
            names,
            ["date=2026-03-08.parquet", "date=2026-03-09.parquet",
             "date=2026-03-10.parquet"],
        )

    def test_a_missing_day_is_skipped_not_faked(self):
        # The archive really does have missing days; inventing a path for one
        # would make DuckDB fail on a file that was never written.
        self._touch("2026-03-08", "2026-03-10")
        picked = fh._files_for_window(self.root, ms(2026, 3, 10), 3)
        names = [Path(p.strip("'")).name for p in picked]
        self.assertEqual(
            names, ["date=2026-03-08.parquet", "date=2026-03-10.parquet"]
        )

    def test_a_window_with_nothing_on_disk_raises(self):
        with self.assertRaises(fh.FillsHistoryError):
            fh._files_for_window(self.root, ms(2026, 3, 10), 3)


class TestWalletValidation(unittest.TestCase):
    def test_an_address_that_is_not_hex_is_dropped(self):
        """The rejection is what keeps the `IN` list from changing meaning."""
        bad = ["0x' OR '1'='1", "not-an-address", "0XABCD"]
        with patch.object(fh, "_run_sql", return_value=[]) as run:
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "date=2026-03-10.parquet").write_bytes(b"")
                with self.assertRaises(fh.FillsHistoryError):
                    fh._rebuild_series(Path(tmp), "BTC", bad)
        self.assertFalse(run.called, "the query ran before validation")

    def test_no_parquet_files_raises_rather_than_querying_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(fh.FillsHistoryError):
                fh._rebuild_series(Path(tmp), "BTC", ["0x" + "a" * 40])


class TestMissingStore(unittest.TestCase):
    def test_selecting_from_an_absent_store_names_the_sync_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope"
            with self.assertRaises(fh.FillsHistoryError) as ctx:
                fh.select_wallets(store=missing, market="BTC",
                                  as_of_ms=ms(2026, 3, 10))
            self.assertIn("sync_reservoir", str(ctx.exception))


class TestCacheSignature(unittest.TestCase):
    """A cache that answers a different question is worse than a slow one.

    The archive is not immutable - two days were rewritten in place on
    2026-10-02 - and the rebuild only covers the wallets it is handed, so both
    the files and the wallet list have to be part of what a hit matches on.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _touch(self, name: str, body: bytes = b"x") -> None:
        (self.root / name).write_bytes(body)

    def test_a_different_wallet_list_is_a_different_signature(self):
        self._touch("date=2026-03-10.parquet")
        a = fh._signature(self.root, "BTC", ["0x" + "a" * 40])
        b = fh._signature(self.root, "BTC", ["0x" + "b" * 40])
        self.assertNotEqual(a, b)

    def test_the_symbol_is_part_of_the_signature(self):
        self._touch("date=2026-03-10.parquet")
        self.assertNotEqual(
            fh._signature(self.root, "BTC", []),
            fh._signature(self.root, "ETH", []),
        )

    def test_rewriting_a_file_in_place_changes_the_signature(self):
        """A count is not enough: the sync also rewrites days that already exist."""
        self._touch("date=2026-03-10.parquet", b"before")
        first = fh._signature(self.root, "BTC", [])
        self._touch("date=2026-03-10.parquet", b"after and longer")
        self.assertNotEqual(first, fh._signature(self.root, "BTC", []))


class TestCacheMiss(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name)
        self._root_tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._root_tmp.name)
        (self.root / "date=2026-03-10.parquet").write_bytes(b"x")

    def tearDown(self):
        self._tmp.cleanup()
        self._root_tmp.cleanup()

    def test_an_absent_cache_is_a_miss(self):
        sig = fh._signature(self.root, "BTC", [])
        self.assertIsNone(fh._read_cache(self.cache, "BTC", sig))

    def test_a_stale_signature_is_a_miss(self):
        parquet, meta = fh._cache_paths(self.cache, "BTC")
        self.cache.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(b"")
        meta.write_text('{"symbol": "BTC", "wallets": [], "files": []}')
        sig = fh._signature(self.root, "BTC", [])
        with patch.object(fh, "_run_sql", return_value=[{"wallet": "x"}]):
            self.assertIsNone(fh._read_cache(self.cache, "BTC", sig))

    def test_a_matching_signature_reads_through(self):
        parquet, meta = fh._cache_paths(self.cache, "BTC")
        self.cache.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(b"")
        sig = fh._signature(self.root, "BTC", [])
        meta.write_text(json.dumps(sig))
        with patch.object(fh, "_run_sql", return_value=[{"wallet": "x"}]):
            self.assertEqual(fh._read_cache(self.cache, "BTC", sig),
                             [{"wallet": "x"}])

    def test_an_unreadable_cache_is_a_miss_rather_than_a_failure(self):
        """The rebuild it was meant to skip is still available, so a corrupt
        cache must not turn a slow run into a failed one."""
        parquet, meta = fh._cache_paths(self.cache, "BTC")
        self.cache.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(b"")
        sig = fh._signature(self.root, "BTC", [])
        meta.write_text(json.dumps(sig))
        with patch.object(fh, "_run_sql",
                          side_effect=fh.FillsHistoryError("bad parquet")):
            self.assertIsNone(fh._read_cache(self.cache, "BTC", sig))

    def test_a_corrupt_signature_file_is_a_miss(self):
        parquet, meta = fh._cache_paths(self.cache, "BTC")
        self.cache.mkdir(parents=True, exist_ok=True)
        parquet.write_bytes(b"")
        meta.write_text("{not json")
        sig = fh._signature(self.root, "BTC", [])
        self.assertIsNone(fh._read_cache(self.cache, "BTC", sig))

    def test_cache_none_skips_the_cache_entirely(self):
        calls = []

        def fake(sql, timeout=1800):
            calls.append(sql)
            return []

        with patch.object(fh, "_run_sql", fake):
            fh._rebuild_series(self.root, "BTC", ["0x" + "a" * 40], None)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("COPY", calls[0])


if __name__ == "__main__":
    unittest.main()
