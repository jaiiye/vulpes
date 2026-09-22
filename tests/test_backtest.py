"""Backtest tests.

The most important test in this file is `TestNoLookahead`. A backtest that
leaks future bars can report anything, and a negative result from a buggy
engine is just as untrustworthy as a positive one from an overfitted one.
"""

from __future__ import annotations

import calendar
import json
import math
from dataclasses import replace  # noqa: E402
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import load_config  # noqa: E402
from agent.indicators import CandleSeries  # noqa: E402
from agent.market_data import MarketDataError  # noqa: E402
from backtest.benchmarks import (  # noqa: E402
    RandomEntryBacktester,
    buy_and_hold,
    result_return_pct,
)
from backtest.data import HistoricalData  # noqa: E402
from backtest.engine import Backtester  # noqa: E402
from backtest.market import BacktestMarket  # noqa: E402
from backtest.metrics import Metrics, compute_metrics  # noqa: E402

HOUR_MS = 3_600_000

CONFIG = """
name: t
symbol: BTC
risk_preset: conservative
indicators:
  entry_timeframe: 1h
  trend_timeframe: 4h
  ema_fast: 5
  ema_slow: 10
  adx_period: 5
  rsi_period: 5
  atr_period: 5
  supertrend_period: 5
  lookback_candles: 200
risk:
  leverage: 2
  account_allocation_pct: 10.0
  risk_per_trade_pct: 1.0
  stop_loss_enabled: true
  stop_loss_pct: 3.0
  take_profit_enabled: true
  take_profit_pct: 9.0
  max_drawdown_pct: 50.0
safety:
  hard_stop_pct: 5.0
execution:
  dry_run: true
  testnet: true
"""


def make_series(closes: list[float], start_ms: int = 1_700_000_000_000) -> CandleSeries:
    """Candles from a close path, with a 1% high/low band around each close."""
    n = len(closes)
    opens = [closes[0]] + closes[:-1]
    highs = [max(o, c) * 1.01 for o, c in zip(opens, closes)]
    lows = [min(o, c) * 0.99 for o, c in zip(opens, closes)]
    times = [start_ms + i * HOUR_MS for i in range(n)]
    return CandleSeries(opens, highs, lows, closes, [1000.0] * n, times)


def make_dataset(
    closes: list[float],
    symbol: str = "BTC",
    start_ms: int = 1_700_000_000_000,
    trend_closes: list[float] | None = None,
) -> HistoricalData:
    ds = HistoricalData(symbol=symbol, start_ms=start_ms, end_ms=start_ms + len(closes) * HOUR_MS)
    ds.candles["1h"] = make_series(closes, start_ms)
    ds.candles["4h"] = make_series(trend_closes or closes[::4] or closes, start_ms)
    return ds


def write_config(text: str) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    handle.write(text)
    handle.close()
    return handle.name


# ---------------------------------------------------------------------------
# The critical property
# ---------------------------------------------------------------------------


class TestNoLookahead(unittest.TestCase):
    """Factors must only ever see bars that closed before the decision."""

    def setUp(self):
        self.closes = [100.0 + i for i in range(50)]  # strictly rising
        self.dataset = make_dataset(self.closes)
        self.market = BacktestMarket({"BTC": self.dataset}, entry_interval="1h")

    def test_candles_excludes_the_current_bar(self):
        """Standing at bar 10's open, the last visible bar is bar 9."""
        series = self.dataset.candles["1h"]
        self.market.set_now(series._times[10])
        visible = self.market.candles("BTC", "1h", lookback=100)

        self.assertEqual(len(visible), 10)
        self.assertEqual(visible._times[-1], series._times[9])
        self.assertNotIn(series._times[10], visible._times)

    def test_candles_never_contains_future_timestamps(self):
        series = self.dataset.candles["1h"]
        for cursor in (5, 12, 25, 40):
            self.market.set_now(series._times[cursor])
            visible = self.market.candles("BTC", "1h", lookback=100)
            self.assertTrue(
                all(t < series._times[cursor] for t in visible._times),
                f"cursor {cursor} leaked a future bar",
            )

    def test_lookback_limits_the_window(self):
        series = self.dataset.candles["1h"]
        self.market.set_now(series._times[30])
        self.assertEqual(len(self.market.candles("BTC", "1h", 5)), 5)

    def test_no_closed_bars_raises_rather_than_returning_future_data(self):
        series = self.dataset.candles["1h"]
        self.market.set_now(series._times[0])  # nothing has closed yet
        with self.assertRaises(MarketDataError):
            self.market.candles("BTC", "1h", 100)

    def test_mid_price_is_the_current_bar_open(self):
        series = self.dataset.candles["1h"]
        self.market.set_now(series._times[7])
        self.assertAlmostEqual(self.market.mid_price("BTC"), series._opens[7])

    def test_asset_context_uses_only_closed_bars(self):
        series = self.dataset.candles["1h"]
        self.market.set_now(series._times[10])
        ctx = self.market.asset_context("BTC")
        # Mark price is the last CLOSED close, i.e. bar 9.
        self.assertAlmostEqual(ctx.mark_price, series._closes[9])

    def test_unknown_symbol_raises(self):
        self.market.set_now(self.dataset.candles["1h"]._times[5])
        with self.assertRaises(MarketDataError):
            self.market.candles("ETH", "1h", 10)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics(unittest.TestCase):
    def result(self, pnls: list[float], curve=None, equity=1000.0):
        from backtest.engine import BacktestResult, ExecutedTrade

        trades = []
        for i, pnl in enumerate(pnls):
            trades.append(
                ExecutedTrade(
                    symbol="BTC",
                    side="long",
                    entry_time=0,
                    entry_price=100.0,
                    size=1.0,
                    notional=100.0,
                    exit_time=HOUR_MS,
                    exit_price=100.0,
                    net_pnl=pnl,
                    risk_usd=10.0,
                )
            )
        final = equity + sum(pnls)
        return BacktestResult(
            symbol="BTC",
            start_ms=0,
            end_ms=len(curve or []) * HOUR_MS,
            initial_equity=equity,
            final_equity=final,
            trades=trades,
            equity_curve=curve or [(i * HOUR_MS, equity) for i in range(10)],
        )

    def test_win_rate_and_counts(self):
        m = compute_metrics(self.result([10, -5, 10, -5]))
        self.assertEqual(m.trades, 4)
        self.assertEqual(m.wins, 2)
        self.assertEqual(m.losses, 2)
        self.assertAlmostEqual(m.win_rate_pct, 50.0)

    def test_expectancy(self):
        m = compute_metrics(self.result([10, -5, 10, -5]))
        self.assertAlmostEqual(m.expectancy, 2.5)

    def test_expectancy_r_uses_risk_budget(self):
        m = compute_metrics(self.result([10, -10]))
        self.assertAlmostEqual(m.expectancy_r, 0.0)

    def test_profit_factor(self):
        m = compute_metrics(self.result([30, -10]))
        self.assertAlmostEqual(m.profit_factor, 3.0)

    def test_profit_factor_zero_when_no_wins(self):
        m = compute_metrics(self.result([-10, -10]))
        self.assertAlmostEqual(m.profit_factor, 0.0)

    def test_total_return(self):
        m = compute_metrics(self.result([100]))
        self.assertAlmostEqual(m.total_return_pct, 10.0)

    def test_max_drawdown(self):
        curve = [(0, 1000.0), (1, 1200.0), (2, 900.0), (3, 1100.0)]
        m = compute_metrics(self.result([0], curve=curve))
        # Peak 1200 -> trough 900 is a 25% drawdown.
        self.assertAlmostEqual(m.max_drawdown_pct, 25.0)

    def test_no_trades_is_not_meaningful(self):
        m = compute_metrics(self.result([]))
        self.assertEqual(m.trades, 0)
        self.assertFalse(m.is_meaningful)
        self.assertAlmostEqual(m.total_return_pct, 0.0)

    def test_small_sample_flagged(self):
        m = compute_metrics(self.result([1.0] * 5))
        self.assertFalse(m.is_meaningful)

    def test_large_sample_is_meaningful(self):
        m = compute_metrics(self.result([1.0, -0.5] * 20))
        self.assertTrue(m.is_meaningful)

    def test_sharpe_is_zero_for_flat_curve(self):
        curve = [(i, 1000.0) for i in range(10)]
        m = compute_metrics(self.result([], curve=curve))
        self.assertAlmostEqual(m.sharpe, 0.0)


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


class TestBuyAndHold(unittest.TestCase):
    def test_rising_market(self):
        ds = make_dataset([100.0, 110.0, 120.0, 130.0])
        # Starts at the first bar's open and ends at the last close.
        pct = buy_and_hold(ds, "1h")
        self.assertAlmostEqual(pct, (130.0 - 100.0) / 100.0 * 100.0, places=4)

    def test_falling_market(self):
        ds = make_dataset([100.0, 90.0, 80.0])
        self.assertLess(buy_and_hold(ds, "1h"), 0)

    def test_missing_interval_returns_zero(self):
        ds = HistoricalData(symbol="BTC", start_ms=0, end_ms=1)
        self.assertEqual(buy_and_hold(ds, "1h"), 0.0)


class TestResultReturn(unittest.TestCase):
    def test_percentage(self):
        from backtest.engine import BacktestResult

        r = BacktestResult(
            symbol="BTC",
            start_ms=0,
            end_ms=1,
            initial_equity=1000.0,
            final_equity=1100.0,
        )
        self.assertAlmostEqual(result_return_pct(r), 10.0)

    def test_zero_equity_guard(self):
        from backtest.engine import BacktestResult

        r = BacktestResult(
            symbol="BTC", start_ms=0, end_ms=1, initial_equity=0.0, final_equity=10.0
        )
        self.assertEqual(result_return_pct(r), 0.0)


# ---------------------------------------------------------------------------
# Engine end to end on synthetic data
# ---------------------------------------------------------------------------


class TestEngineOnSyntheticData(unittest.TestCase):
    def setUp(self):
        self.cfg_path = write_config(CONFIG)
        self.addCleanup(os.unlink, self.cfg_path)
        self.cfg = load_config(self.cfg_path)

    def oscillating(self, n=600) -> list[float]:
        """A mean-reverting path, which should still produce trades."""
        return [100.0 + 10 * math.sin(i / 12.0) for i in range(n)]

    def test_runs_and_produces_a_curve(self):
        ds = make_dataset(self.oscillating())
        bt = Backtester(self.cfg, {"BTC": ds}, initial_equity=1000.0)
        result = bt.run()
        self.assertGreater(len(result.equity_curve), 0)
        self.assertEqual(result.symbol, "BTC")

    def test_smart_money_factor_is_replaced(self):
        ds = make_dataset(self.oscillating())
        bt = Backtester(self.cfg, {"BTC": ds})
        score = bt.synth.smart_money.evaluate("BTC")
        self.assertEqual(score.confidence, 0.0)
        self.assertTrue(any("no historical whale" in r for r in score.reasons))

    def test_alignment_gate_disabled_with_a_warning(self):
        ds = make_dataset(self.oscillating())
        bt = Backtester(self.cfg, {"BTC": ds})
        self.assertFalse(bt.cfg.discipline.require_smart_money_alignment)
        self.assertTrue(
            any("alignment gate DISABLED" in w for w in bt.warnings), bt.warnings
        )

    def test_accounting_identity_holds(self):
        """final equity must equal initial plus every trade's net PnL."""
        ds = make_dataset(self.oscillating())
        bt = Backtester(self.cfg, {"BTC": ds}, initial_equity=1000.0)
        result = bt.run()
        expected = 1000.0 + sum(t.net_pnl for t in result.trades)
        self.assertAlmostEqual(result.final_equity, expected, places=6)

    def test_funding_is_charged(self):
        ds = make_dataset(self.oscillating())
        # 1% hourly funding is extreme, so any held position must show a cost.
        ds.funding = [
            (ds.start_ms + i * HOUR_MS, 0.0001) for i in range(600)
        ]
        bt = Backtester(self.cfg, {"BTC": ds})
        result = bt.run()
        if result.trades:
            self.assertNotEqual(result.funding_paid, 0.0)

    def test_flat_market_barely_trades(self):
        """Not exactly zero: on a perfectly flat series Supertrend's
        initialisation tie-breaks `close >= mid` to "buy", which can trip one
        entry. What matters is that a featureless market does not generate
        activity, so the signal is not simply noise-driven."""
        ds = make_dataset([100.0] * 400)
        bt = Backtester(self.cfg, {"BTC": ds})
        result = bt.run()
        self.assertLessEqual(len(result.trades), 2)

    def test_random_entry_uses_the_same_pipeline(self):
        ds = make_dataset(self.oscillating())
        bt = RandomEntryBacktester(self.cfg, {"BTC": ds}, seed=1)
        self.assertFalse(bt.cfg.discipline.require_smart_money_alignment)
        result = bt.run()
        # Random entries should trade far more often than a selective signal.
        self.assertGreater(len(result.equity_curve), 0)


class TestClockSeeding(unittest.TestCase):
    """The discipline day window must anchor to simulated time."""

    def test_day_anchor_uses_simulated_start(self):
        cfg_path = write_config(CONFIG)
        self.addCleanup(os.unlink, cfg_path)
        cfg = load_config(cfg_path)

        closes = [100.0 + i * 0.01 for i in range(300)]
        ds = make_dataset(closes)
        bt = Backtester(cfg, {"BTC": ds})

        # Anchored at the dataset start, not wall time.
        self.assertAlmostEqual(
            bt.discipline.state.day_anchor, ds.start_ms / 1000.0, places=3
        )

    def test_clock_moving_backwards_reanchors(self):
        """A backtest replaying the past must not latch the daily cap."""
        from agent.discipline import DisciplineState

        state = DisciplineState()
        state.day_anchor = 2_000_000_000.0  # "future"
        state.trades_today = {"BTC": 3}

        state.roll_day_if_needed(1_700_000_000.0)
        self.assertAlmostEqual(state.day_anchor, 1_700_000_000.0)
        # Counts survive: they belong to the replayed day.
        self.assertEqual(state.trades_today, {"BTC": 3})


class TestWhalePositionSeries(unittest.TestCase):
    """The staleness guard is what stops the Scorpion v1 failure mode:
    treating a months-old position as a fresh signal."""

    def series(self):
        from backtest.whale_history import WalletPositionSeries

        s = WalletPositionSeries(wallet="0xabc", coin="BTC")
        s.points = [(1_000_000, 5.0), (2_000_000, -3.0), (3_000_000, 0.0)]
        return s

    def test_returns_the_most_recent_prior_position(self):
        s = self.series()
        self.assertAlmostEqual(s.position_at(2_500_000, 10_000_000), -3.0)

    def test_exact_timestamp_matches(self):
        s = self.series()
        self.assertAlmostEqual(s.position_at(2_000_000, 1), -3.0)

    def test_stale_data_is_refused(self):
        s = self.series()
        # 2s after the record, with a 1s tolerance.
        self.assertIsNone(s.position_at(2_002_000, 1_000))

    def test_tolerance_boundary_is_inclusive(self):
        s = self.series()
        # Exactly at the tolerance, the position is still considered current.
        self.assertAlmostEqual(s.position_at(2_001_000, 1_000), -3.0)

    def test_before_the_first_record_returns_none(self):
        s = self.series()
        self.assertIsNone(s.position_at(500_000, 10_000_000))

    def test_empty_series_returns_none(self):
        from backtest.whale_history import WalletPositionSeries

        self.assertIsNone(WalletPositionSeries("0xabc", "BTC").position_at(1, 10**12))

    def test_flat_position_is_returned_as_flat_not_none(self):
        s = self.series()
        self.assertEqual(s.position_at(3_500_000, 10_000_000), 0.0)

    # ------------------------------------------------------------------
    # Entry prices. The Reservoir snapshots carry them; the funding-derived
    # series do not, and the difference decides whether the wallet-quality
    # confidence term can be computed at all.
    # ------------------------------------------------------------------

    def series_with_entries(self):
        from backtest.whale_history import WalletPositionSeries

        s = WalletPositionSeries(wallet="0xabc", coin="BTC")
        s.points = [(1_000_000, 2.0), (2_000_000, -3.0)]
        s.entries = [(1_000_000, 100.0), (2_000_000, 110.0)]
        return s

    def test_has_pnl_only_when_entries_exist(self):
        self.assertFalse(self.series().has_pnl)
        self.assertTrue(self.series_with_entries().has_pnl)

    def test_entry_at_reads_the_parallel_series(self):
        s = self.series_with_entries()
        self.assertAlmostEqual(s.entry_at(1_500_000, 10**12), 100.0)
        self.assertAlmostEqual(s.entry_at(2_500_000, 10**12), 110.0)

    def test_entry_lookups_honour_staleness_too(self):
        """An entry price is only usable if it belongs to the position it
        describes; carrying a stale one would pair today's size with last
        month's cost basis."""
        s = self.series_with_entries()
        self.assertIsNone(s.entry_at(2_500_000, 1_000))

    def test_entry_at_on_a_series_without_entries_returns_none(self):
        self.assertIsNone(self.series().entry_at(1_500_000, 10**12))

    def test_the_two_lookups_keep_separate_timestamp_indexes(self):
        """White-box on purpose: the failure this guards is silent slowness.

        The snapshot builder calls both lookups for the same wallet in the same
        bar. With a single shared cache slot each call evicts the other's index,
        so the O(n) rebuild happens on every lookup instead of once - 150
        wallets against 4000 bars turns a 364-point series into hundreds of
        millions of operations, and nothing about the *results* would look
        wrong.
        """
        s = self.series_with_entries()
        for ts in (1_500_000, 2_500_000, 1_800_000, 2_200_000):
            s.position_at(ts, 10**12)
            s.entry_at(ts, 10**12)
        self.assertEqual(len(s._times_cache), 2)

    def test_payload_round_trip(self):
        from backtest.whale_history import WalletPositionSeries

        s = self.series()
        s.funding_rates = [(1_000_000, 0.0001)]
        restored = WalletPositionSeries.from_payload(s.to_payload())
        self.assertEqual(restored.points, s.points)
        self.assertEqual(restored.funding_rates, s.funding_rates)


class TestFetchAllSplitsByCoin(unittest.TestCase):
    """One paginated walk must serve every coin: fetching per coin would
    repeat the same 10+ request sequence for each symbol."""

    def collector(self, pages):
        from backtest.whale_history import WhaleHistoryCollector

        c = WhaleHistoryCollector(cache_dir=None, workers=1)
        calls = {"n": 0}

        def fake(url, payload=None, timeout=None, retries=None):
            page = pages[calls["n"]] if calls["n"] < len(pages) else []
            calls["n"] += 1
            return page

        c._fake = fake
        import backtest.whale_history as mod

        self._orig = mod.request_json
        mod.request_json = fake
        self.addCleanup(lambda: setattr(mod, "request_json", self._orig))
        return c, calls

    def rec(self, ts, coin, szi, rate=0.0001):
        return {
            "time": ts,
            "delta": {"coin": coin, "szi": str(szi), "fundingRate": str(rate)},
        }

    def test_splits_records_across_coins(self):
        pages = [
            [
                self.rec(1000, "BTC", 1.5),
                self.rec(1000, "ETH", -2.0),
                self.rec(2000, "BTC", 1.6),
            ],
            [],
        ]
        c, _ = self.collector(pages)
        by_coin = c.fetch_all("0xabc", 0, 10**9)

        self.assertEqual(set(by_coin), {"BTC", "ETH"})
        self.assertEqual(by_coin["BTC"].points, [(1000, 1.5), (2000, 1.6)])
        self.assertEqual(by_coin["ETH"].points, [(1000, -2.0)])

    def test_pagination_advances_past_each_page(self):
        pages = [
            [self.rec(1000, "BTC", 1.0)],
            [self.rec(2000, "BTC", 2.0)],
            [],
        ]
        c, calls = self.collector(pages)
        series = c.fetch_all("0xabc", 0, 10**9)["BTC"]
        self.assertEqual(series.points, [(1000, 1.0), (2000, 2.0)])
        self.assertGreaterEqual(calls["n"], 3)

    def test_records_without_szi_are_skipped(self):
        pages = [
            [
                {"time": 1000, "delta": {"coin": "BTC"}},
                self.rec(2000, "BTC", 3.0),
            ],
            [],
        ]
        c, _ = self.collector(pages)
        self.assertEqual(c.fetch_all("0xabc", 0, 10**9)["BTC"].points, [(2000, 3.0)])

    def test_fetch_one_wrapper_returns_the_requested_coin(self):
        pages = [[self.rec(1000, "BTC", 1.5), self.rec(1000, "ETH", 2.5)], []]
        c, _ = self.collector(pages)
        btc = c.fetch_one("0xabc", "BTC", 0, 10**9)
        self.assertEqual(btc.coin, "BTC")
        self.assertEqual(btc.points, [(1000, 1.5)])


class TestHistoricalSmartMoney(unittest.TestCase):
    """Rebuilds a snapshot and delegates scoring to the LIVE factor."""

    def build(self, points_by_wallet, max_age_ms=6 * 3_600_000):
        from backtest.historical_smart_money import HistoricalSmartMoney
        from backtest.whale_history import WalletPositionSeries

        hist = {}
        for wallet, points in points_by_wallet.items():
            series = WalletPositionSeries(wallet=wallet, coin="BTC")
            series.points = points
            hist[wallet] = series

        advisor = HistoricalSmartMoney(
            hist, price_lookup=lambda ts: 100.0, max_age_ms=max_age_ms
        )
        advisor.set_now(1_000_000)
        return advisor

    def test_aggregates_positions_into_a_snapshot(self):
        advisor = self.build({"0xa": [(999_000, 2.0)], "0xb": [(999_000, -1.0)]})
        snap = advisor.build_snapshot("BTC")
        self.assertEqual(len(snap.positions), 2)
        self.assertEqual(snap.wallets_sampled, 2)

    def test_notional_is_signed_by_side(self):
        advisor = self.build({"0xa": [(999_000, 2.0)], "0xb": [(999_000, -1.0)]})
        snap = advisor.build_snapshot("BTC")
        by_wallet = {p.wallet: p.notional for p in snap.positions}
        self.assertAlmostEqual(by_wallet["0xa"], 200.0)
        self.assertAlmostEqual(by_wallet["0xb"], -100.0)

    def test_stale_wallets_are_excluded(self):
        # Clock is at 1_000_000ms. A record at timestamp 0 is 1_000_000ms old,
        # which exceeds a 60_000ms (one minute) tolerance.
        advisor = self.build(
            {"0xfresh": [(999_000, 1.0)], "0xstale": [(0, 5.0)]},
            max_age_ms=60_000,
        )
        snap = advisor.build_snapshot("BTC")
        self.assertEqual([p.wallet for p in snap.positions], ["0xfresh"])

    def test_flat_wallets_are_excluded(self):
        advisor = self.build({"0xa": [(999_000, 0.0)]})
        self.assertEqual(len(advisor.build_snapshot("BTC").positions), 0)

    def test_evaluate_returns_a_factor_score(self):
        advisor = self.build({"0xa": [(999_000, 2.0)], "0xb": [(999_000, 1.0)]})
        score = advisor.evaluate("BTC")
        self.assertEqual(score.name, "smart_money")
        self.assertGreater(score.score, 50)  # both long

    def test_evaluate_with_no_positions_has_zero_confidence(self):
        advisor = self.build({})
        score = advisor.evaluate("BTC")
        self.assertEqual(score.confidence, 0.0)
        self.assertEqual(score.score, 50.0)

    def test_uses_the_live_evaluate_implementation(self):
        """The whole point: scoring must be the production code, not a copy."""
        from agent.factors.smart_money import SmartMoneyFactor

        advisor = self.build({"0xa": [(999_000, 2.0)]})
        score = advisor.evaluate("BTC")
        # Fields only the live evaluate produces.
        self.assertIn("data_network", score.details)
        self.assertIn("weighted_long_ratio_pct", score.details)
        self.assertTrue(hasattr(SmartMoneyFactor, "evaluate"))


class TestCoverageReport(unittest.TestCase):
    def test_empty_history_is_flagged(self):
        from backtest.historical_smart_money import coverage_report

        text = coverage_report({}, 100)
        self.assertIn("NO WHALE HISTORY", text)

    def test_low_coverage_is_flagged(self):
        from backtest.historical_smart_money import coverage_report
        from backtest.whale_history import WalletPositionSeries

        hist = {"0xa": WalletPositionSeries("0xa", "BTC")}
        text = coverage_report(hist, 100)
        self.assertIn("LOW COVERAGE", text)


class TestStopFillIsNotOptimistic(unittest.TestCase):
    """A stop-market order cannot fill at the stop when the bar gapped past it.

    Assuming the exact stop price whenever it was merely touched flattered every
    stopped-out trade, which is the most dangerous direction for a backtest to
    be wrong in.
    """

    def _run_single_bar(
        self,
        bar: dict,
        side: str,
        stop: float,
        entry: float,
        take_profit: float | None = None,
    ):
        """Run one management step with a hand-built bar."""
        from agent.execution import Position
        from backtest.engine import Backtester
        from backtest.market import Bar

        cfg_path = write_config(CONFIG)
        self.addCleanup(os.unlink, cfg_path)
        cfg = load_config(cfg_path)

        closes = [entry] * 30
        bt = Backtester(cfg, {"BTC": make_dataset(closes)})
        bt.position = Position(
            symbol="BTC",
            side=side,
            size=1.0,
            entry_price=entry,
            notional=entry,
            leverage=2,
            stop_price=stop,
            take_profit_price=take_profit,
            is_dry_run=True,
            risk_usd=10.0,
        )
        bar_obj = Bar(
            time=bt.datasets["BTC"].start_ms,
            open=bar["open"],
            high=bar["high"],
            low=bar["low"],
            close=bar["close"],
            volume=1.0,
        )
        bt._manage(bar_obj, bt.datasets["BTC"].start_ms)
        return bt

    def test_long_stop_gapping_through_fills_at_the_open(self):
        # Stop at 97, but the bar OPENS at 90: no fill at 97 was possible.
        bt = self._run_single_bar(
            {"open": 90.0, "high": 101.0, "low": 89.0, "close": 95.0},
            side="long",
            stop=97.0,
            entry=100.0,
        )
        self.assertEqual(len(bt.trades), 1)
        self.assertAlmostEqual(bt.trades[0].exit_price, 90.0)

    def test_long_stop_touched_without_a_gap_still_fills_at_the_stop(self):
        bt = self._run_single_bar(
            {"open": 100.0, "high": 101.0, "low": 96.0, "close": 99.0},
            side="long",
            stop=97.0,
            entry=100.0,
        )
        self.assertAlmostEqual(bt.trades[0].exit_price, 97.0)

    def test_short_stop_gapping_through_fills_at_the_open(self):
        # Stop at 103, bar opens at 110: fill is 110, i.e. worse than the stop.
        bt = self._run_single_bar(
            {"open": 110.0, "high": 111.0, "low": 100.0, "close": 105.0},
            side="short",
            stop=103.0,
            entry=100.0,
        )
        self.assertEqual(len(bt.trades), 1)
        self.assertAlmostEqual(bt.trades[0].exit_price, 110.0)

    def test_take_profit_still_fills_at_the_target(self):
        """Gapping THROUGH a target is favourable, so the exact target is the
        conservative assumption and must stay unchanged."""
        bt = self._run_single_bar(
            {"open": 100.0, "high": 120.0, "low": 99.0, "close": 118.0},
            side="long",
            stop=97.0,
            entry=100.0,
            take_profit=109.0,
        )
        # The stop is not touched, so the target is the exit - and it fills at
        # 109 even though the bar traded up to 120.
        self.assertEqual(len(bt.trades), 1)
        self.assertAlmostEqual(bt.trades[0].exit_price, 109.0)


class TestFundingLookups(unittest.TestCase):
    """`funding_at_ms` used to rebuild its timestamp list on every call."""

    def make(self):
        n = 100
        return HistoricalData(
            symbol="BTC",
            start_ms=0,
            end_ms=n * HOUR_MS,
            funding=[(i * HOUR_MS, 0.0001 * (i + 1)) for i in range(n)],
        )

    def test_funding_at_ms_returns_the_most_recent_rate(self):
        data = self.make()
        # Between settlement 4 and 5, settlement 4 is the rate in effect.
        self.assertAlmostEqual(data.funding_at_ms(4 * HOUR_MS), 0.0005)
        self.assertAlmostEqual(data.funding_at_ms(4 * HOUR_MS + 1), 0.0005)

    def test_funding_at_ms_before_the_first_record_is_zero(self):
        data = self.make()
        self.assertEqual(data.funding_at_ms(-1), 0.0)

    def test_funding_at_ms_is_exact_on_a_settlement_boundary(self):
        data = self.make()
        self.assertAlmostEqual(data.funding_at_ms(10 * HOUR_MS), 0.0011)

    def test_index_is_rebuilt_when_funding_is_replaced(self):
        """The loader assigns `funding` after construction."""
        data = self.make()
        data.funding_at_ms(4 * HOUR_MS)  # warm the index
        data.funding = [(i * HOUR_MS, 7.0) for i in range(3)]
        data._funding_times = []
        self.assertAlmostEqual(data.funding_at_ms(2 * HOUR_MS), 7.0)

    def test_funding_between_is_bounded(self):
        data = self.make()
        got = data.funding_between(2 * HOUR_MS, 5 * HOUR_MS)
        # Half-open on the left, inclusive on the right: (2h, 5h].
        self.assertEqual([t for t, _ in got], [3 * HOUR_MS, 4 * HOUR_MS, 5 * HOUR_MS])

    def test_funding_between_empty_range(self):
        self.assertEqual(self.make().funding_between(10 * HOUR_MS, 10 * HOUR_MS), [])


class TestCandleSeriesPublicAccessors(unittest.TestCase):
    """Only closes/highs/lows were public, so the backtest reached into
    `_opens`, `_times` and `_volumes`."""

    def test_all_six_arrays_are_public(self):
        s = make_series([1.0, 2.0, 3.0])
        self.assertEqual(len(s.opens), 3)
        self.assertEqual(len(s.highs), 3)
        self.assertEqual(len(s.lows), 3)
        self.assertEqual(len(s.closes), 3)
        self.assertEqual(len(s.volumes), 3)
        self.assertEqual(len(s.times), 3)

    def test_accessors_match_the_constructor_arguments(self):
        s = CandleSeries([1.0], [2.0], [0.5], [1.5], [99.0], [123])
        self.assertEqual(s.opens, [1.0])
        self.assertEqual(s.highs, [2.0])
        self.assertEqual(s.lows, [0.5])
        self.assertEqual(s.closes, [1.5])
        self.assertEqual(s.volumes, [99.0])
        self.assertEqual(s.times, [123])

    def test_backtest_does_not_touch_private_fields(self):
        """Guard against a regression back to `series._times` style access."""
        import backtest.benchmarks as bench_mod
        import backtest.data as data_mod
        import backtest.market as market_mod

        for module in (data_mod, market_mod, bench_mod):
            source = Path(module.__file__).read_text(encoding="utf-8")
            # Match any `<name>._<private-array>` access on a candle series.
            for private in ("_times", "_opens", "_highs", "_lows", "_volumes"):
                for obj in ("series", "part", "entry_series"):
                    self.assertNotIn(
                        f"{obj}{private}",
                        source,
                        f"{module.__name__} still reads {obj}.{private}",
                    )


class TestBacktestConfidenceParity(unittest.TestCase):
    """The backtest cannot reconstruct per-wallet PnL. Feeding 0.0 was a false
    claim that every whale sat at break-even, pinning the quality term to its
    floor and shifting confidence by up to 0.075."""

    def scorer(self):
        from agent.factors.smart_money import SmartMoneyFactor

        f = SmartMoneyFactor.__new__(SmartMoneyFactor)
        f.data_market = type("M", (), {"testnet": False})()
        return f

    def snapshot(self, pnl, available):
        from agent.factors.smart_money import SmartMoneySnapshot, WalletPosition

        positions = [
            WalletPosition(wallet=f"L{i}", size=1.0, notional=1e6, unrealized_pnl=pnl)
            for i in range(3)
        ] + [
            WalletPosition(wallet=f"S{i}", size=-1.0, notional=-1e6, unrealized_pnl=pnl)
            for i in range(3)
        ]
        return SmartMoneySnapshot(
            symbol="BTC",
            wallets_sampled=6,
            wallets_selected=6,
            source="leaderboard",
            persistence_window_count=3,
            avg_persistence=2.0,
            positions=positions,
            pnl_available=available,
        )

    def test_unknown_pnl_lands_centred_in_the_live_range(self):
        scorer = self.scorer()
        all_up = scorer.evaluate("BTC", self.snapshot(5_000.0, True)).confidence
        all_down = scorer.evaluate("BTC", self.snapshot(-5_000.0, True)).confidence
        unknown = scorer.evaluate("BTC", self.snapshot(0.0, False)).confidence

        lo, hi = min(all_up, all_down), max(all_up, all_down)
        self.assertLessEqual(lo, unknown)
        self.assertLessEqual(unknown, hi)
        # Centred, not merely inside: the error is symmetric.
        self.assertAlmostEqual(unknown - lo, hi - unknown, places=6)

    def test_old_behaviour_would_have_been_biased_low(self):
        """Documents why `pnl_available=False` exists: reading 0.0 as a real
        observation sits at the pessimistic edge of the range."""
        scorer = self.scorer()
        as_if_known_zero = scorer.evaluate("BTC", self.snapshot(0.0, True)).confidence
        all_up = scorer.evaluate("BTC", self.snapshot(5_000.0, True)).confidence
        self.assertAlmostEqual(as_if_known_zero, all_up - 0.075, places=6)

    def test_details_report_pnl_availability(self):
        scorer = self.scorer()
        known = scorer.evaluate("BTC", self.snapshot(0.0, True))
        unknown = scorer.evaluate("BTC", self.snapshot(0.0, False))
        self.assertTrue(known.details["pnl_available"])
        self.assertFalse(unknown.details["pnl_available"])
        self.assertIn("pnl_term", unknown.details)

    def test_live_path_is_unchanged_by_default(self):
        """`pnl_available` defaults to True, so live scoring is untouched."""
        from agent.factors.smart_money import SmartMoneySnapshot

        self.assertTrue(SmartMoneySnapshot(symbol="BTC").pnl_available)


class TestWhaleHistoryWindowCount(unittest.TestCase):
    """`persistence_window_count` must come from the live config, not a
    hardcoded 3: it normalises curation_score and therefore confidence."""

    def build(self, window_count, persistence=2):
        from backtest.historical_smart_money import HistoricalSmartMoney
        from backtest.whale_history import WalletPositionSeries

        series = WalletPositionSeries("0xa", "BTC")
        series.points = [(1_700_000_000_000, 1.0)]
        sm = HistoricalSmartMoney(
            {"0xa": series},
            price_lookup=lambda _ts: 100.0,
            # persistence > 1, otherwise curation_score is identically zero and
            # the window count would be untestable.
            persistence={"0xa": persistence},
            window_count=window_count,
        )
        sm.set_now(1_700_000_000_000)
        return sm.build_snapshot("BTC")

    def test_window_count_is_carried_through(self):
        self.assertEqual(self.build(2).persistence_window_count, 2)
        self.assertEqual(self.build(5).persistence_window_count, 5)

    def test_snapshot_declares_pnl_unavailable(self):
        self.assertFalse(self.build(3).pnl_available)

    def test_curation_score_shifts_with_the_window_count(self):
        """Same persistence, different normalisation -> different curation."""
        two = self.build(2).curation_score
        four = self.build(4).curation_score
        self.assertNotAlmostEqual(two, four)


class TestSharedScorerIsPure(unittest.TestCase):
    """Scoring was borrowed off a half-built instance via
    `SmartMoneyFactor.__new__`, setting only the one attribute the method read.

    It worked, but nothing would have caught it breaking: a new instance
    dependency would have surfaced as an AttributeError at runtime, in the
    backtest only. It is now a module-level pure function that the live factor,
    the backtest and the tests all call.
    """

    def snapshot(self, **overrides):
        from agent.factors.smart_money import SmartMoneySnapshot, WalletPosition

        positions = [
            WalletPosition(wallet=f"L{i}", size=1.0, notional=1e6, unrealized_pnl=250.0)
            for i in range(7)
        ] + [
            WalletPosition(wallet=f"S{i}", size=-1.0, notional=-1e6, unrealized_pnl=-90.0)
            for i in range(2)
        ]
        fields = {
            "symbol": "BTC",
            "wallets_sampled": 9,
            "wallets_selected": 9,
            "source": "leaderboard",
            "persistence_window_count": 3,
            "avg_persistence": 2.0,
            "positions": positions,
        }
        fields.update(overrides)
        return SmartMoneySnapshot(**fields)

    def test_pure_function_needs_no_instance_at_all(self):
        from agent.factors.smart_money import score_snapshot

        score = score_snapshot(self.snapshot())
        self.assertEqual(score.name, "smart_money")
        self.assertTrue(0.0 <= score.score <= 100.0)
        self.assertTrue(0.0 <= score.confidence <= 1.0)
        self.assertEqual(score.details["data_network"], "mainnet")

    def test_data_network_is_a_parameter_not_instance_state(self):
        from agent.factors.smart_money import score_snapshot

        self.assertEqual(
            score_snapshot(self.snapshot(), data_network="testnet").details[
                "data_network"
            ],
            "testnet",
        )

    def test_live_evaluate_delegates_to_the_shared_scorer(self):
        """The live factor's output must be byte-identical to the pure call."""
        from agent.factors.smart_money import SmartMoneyFactor, score_snapshot
        from agent.market_data import HyperliquidMarket

        snap = self.snapshot()
        factor = SmartMoneyFactor(HyperliquidMarket(testnet=False), cache_path=None)
        live = factor.evaluate("BTC", snap)
        shared = score_snapshot(snap, "BTC", data_network="mainnet")

        self.assertEqual(live.score, shared.score)
        self.assertEqual(live.confidence, shared.confidence)
        self.assertEqual(live.reasons, shared.reasons)
        self.assertEqual(live.details, shared.details)

    def test_backtest_evaluate_delegates_to_the_same_scorer(self):
        from agent.factors.smart_money import score_snapshot
        from backtest.historical_smart_money import HistoricalSmartMoney
        from backtest.whale_history import WalletPositionSeries

        series = WalletPositionSeries("0xa", "BTC")
        series.points = [(1_700_000_000_000, 2.0)]
        sm = HistoricalSmartMoney(
            {"0xa": series},
            price_lookup=lambda _ts: 100.0,
            persistence={"0xa": 3},
            window_count=3,
        )
        sm.set_now(1_700_000_000_000)

        snap = sm.build_snapshot("BTC")
        from_backtest = sm.evaluate("BTC", snap)
        direct = score_snapshot(snap, "BTC", data_network="mainnet")

        self.assertEqual(from_backtest.score, direct.score)
        self.assertEqual(from_backtest.confidence, direct.confidence)
        self.assertEqual(from_backtest.details, direct.details)

    def test_backtest_no_longer_builds_a_bare_instance(self):
        from backtest.historical_smart_money import HistoricalSmartMoney

        sm = HistoricalSmartMoney({}, price_lookup=lambda _ts: 100.0)
        self.assertFalse(
            hasattr(sm, "_scorer"),
            "the half-constructed scorer instance should be gone",
        )

    def test_production_code_does_not_bypass_constructors(self):
        """Source-level guard: `__new__(` must not reappear in production."""
        import agent.factors.smart_money as sm_mod
        import backtest.historical_smart_money as hsm_mod

        for module in (sm_mod, hsm_mod):
            source = Path(module.__file__).read_text(encoding="utf-8")
            self.assertNotIn(
                "__new__(",
                source,
                f"{module.__name__} bypasses __init__ again",
            )


REVERSAL_CONFIG = """
name: t
symbol: BTC
discipline:
  long_threshold: 60.0
  short_threshold: 40.0
  max_signals_per_day: 3
  cooldown_minutes: 240
  require_smart_money_alignment: false
  btc_trend_filter: false
risk:
  leverage: 2
  account_allocation_pct: 10.0
  risk_per_trade_pct: 1.0
  stop_loss_enabled: false
  take_profit_enabled: false
  max_drawdown_pct: 50.0
safety:
  hard_stop_pct: 50.0
execution:
  dry_run: true
  testnet: true
"""


class TestReversalExits(unittest.TestCase):
    """The backtest used to evaluate signals only while flat.

    That meant a position could only ever exit via a stop or a target, while
    the live agent closes and reverses on the first opposite signal. The
    backtest's holding periods were therefore an artefact of the simulation.
    """

    def build(self, script, config_text=REVERSAL_CONFIG):
        """A Backtester driven by a scripted signal sequence.

        Missing entries fall through to a neutral signal, and stops/targets are
        disabled, so the only thing that can close a position is a reversal.
        """
        from agent.synthesizer import Signal

        cfg_path = write_config(config_text)
        self.addCleanup(os.unlink, cfg_path)
        cfg = load_config(cfg_path)

        # Flat prices: no stop, no target, no end-of-run PnL distortion.
        closes = [100.0] * 40
        bt = Backtester(cfg, {"BTC": make_dataset(closes)})

        neutral = Signal(symbol="BTC", action="neutral", score=50.0, confidence=0.5)
        seq = iter(script)
        bt._generate_signal = lambda: next(seq, neutral)
        return bt

    def test_opposite_signal_closes_the_open_position(self):
        from agent.synthesizer import Signal

        long_sig = Signal(symbol="BTC", action="long", score=70.0, confidence=0.9)
        short_sig = Signal(symbol="BTC", action="short", score=30.0, confidence=0.9)

        bt = self.build([long_sig, short_sig])
        bt.run()

        # First trade is the long, closed BY THE REVERSAL - not by a stop, a
        # target or the end of the run.
        self.assertGreaterEqual(len(bt.trades), 1)
        first = bt.trades[0]
        self.assertEqual(first.side, "long")
        self.assertIn("reversal to short", first.exit_reason)

    def test_reversal_opens_the_opposite_side(self):
        from agent.synthesizer import Signal

        long_sig = Signal(symbol="BTC", action="long", score=70.0, confidence=0.9)
        short_sig = Signal(symbol="BTC", action="short", score=30.0, confidence=0.9)

        bt = self.build([long_sig, short_sig])
        bt.run()

        # The short was opened and then closed at the end of the run.
        self.assertGreaterEqual(len(bt.trades), 2)
        self.assertEqual(bt.trades[1].side, "short")
        self.assertEqual(bt.trades[1].exit_reason, "end of backtest")

    def test_reversal_is_exempt_from_the_cooldown(self):
        """A long at bar 0 starts a 4h cooldown; the reversal at bar 1 must
        still be allowed, because the discipline layer exempts reversals."""
        from agent.synthesizer import Signal

        long_sig = Signal(symbol="BTC", action="long", score=70.0, confidence=0.9)
        short_sig = Signal(symbol="BTC", action="short", score=30.0, confidence=0.9)

        bt = self.build([long_sig, short_sig])
        bt.run()

        self.assertEqual(bt.blocked.get("cooldown", 0), 0)
        self.assertGreaterEqual(len(bt.trades), 2)

    def test_same_direction_signal_while_holding_is_blocked(self):
        """No duplicate entries: the position-slot gate must refuse it."""
        from agent.synthesizer import Signal

        long_sig = Signal(symbol="BTC", action="long", score=70.0, confidence=0.9)

        # Cooldown is zeroed so the gate order actually reaches the slot check.
        bt = self.build(
            [long_sig, long_sig, long_sig],
            config_text=REVERSAL_CONFIG.replace(
                "cooldown_minutes: 240", "cooldown_minutes: 0"
            ),
        )
        bt.run()

        self.assertGreaterEqual(bt.blocked.get("already holding", 0), 1)
        # Only the initial entry plus the end-of-run close.
        self.assertEqual(len(bt.trades), 1)

    def test_signals_are_evaluated_while_a_position_is_open(self):
        """The core regression: signals_seen must reflect every bar, not only
        the flat ones. Previously 537 of 4320 bars were evaluated."""
        from agent.synthesizer import Signal

        long_sig = Signal(symbol="BTC", action="long", score=70.0, confidence=0.9)
        bt = self.build([long_sig])  # one long, then neutral forever
        bt.run()

        expected = len(bt.market.bar_times("BTC", bt.cfg.indicators.entry_timeframe))
        self.assertEqual(bt.signals_seen, expected)

    def test_no_position_means_no_reversal(self):
        from agent.synthesizer import Signal

        short_sig = Signal(symbol="BTC", action="short", score=30.0, confidence=0.9)
        bt = self.build([short_sig])
        bt.run()

        # A short opened on a flat book is an entry, not a reversal.
        self.assertEqual(len(bt.trades), 1)
        self.assertEqual(bt.trades[0].exit_reason, "end of backtest")


class TestSnapshotBackedWhalePnl(unittest.TestCase):
    """`pnl_available` must reflect the data actually present.

    The engine used to report unconditionally that the wallet-quality
    confidence term was excluded. That is true of funding-derived history and
    false of the Reservoir snapshots, which carry entry prices - so every
    snapshot-backed run was told a term was missing when it was being used.
    """

    def build(self, wallets):
        from backtest.historical_smart_money import HistoricalSmartMoney
        from backtest.whale_history import WalletPositionSeries

        history = {}
        for key, (size, entry) in wallets.items():
            series = WalletPositionSeries(wallet=key, coin="BTC")
            series.points = [(1_000, size)]
            if entry is not None:
                series.entries = [(1_000, entry)]
            history[key] = series

        source = HistoricalSmartMoney(
            history,
            price_lookup=lambda _t: 200.0,
            window_count=3,
            max_age_ms=10**12,
        )
        source.set_now(1_500)
        return source.build_snapshot("BTC")

    def test_entry_prices_yield_real_pnl_and_the_quality_term(self):
        snap = self.build({"0xlong": (2.0, 100.0)})
        self.assertTrue(snap.pnl_available)
        position = snap.positions[0]
        self.assertAlmostEqual(position.entry_price, 100.0)
        self.assertAlmostEqual(position.notional, 400.0)
        self.assertAlmostEqual(position.unrealized_pnl, 200.0)

    def test_short_is_marked_the_other_way(self):
        snap = self.build({"0xshort": (-2.0, 100.0)})
        position = snap.positions[0]
        self.assertAlmostEqual(position.notional, -400.0)
        self.assertAlmostEqual(position.unrealized_pnl, -200.0)

    def test_notional_is_marked_at_the_bar_price_not_the_entry(self):
        """The snapshot's own notional is stale by up to a day; the entry
        price is a fixed historical fact and the only part worth keeping."""
        snap = self.build({"0xa": (3.0, 50.0)})
        self.assertAlmostEqual(snap.positions[0].notional, 600.0)

    def test_absent_entry_price_marks_pnl_unavailable(self):
        snap = self.build({"0xunknown": (2.0, None)})
        self.assertFalse(snap.pnl_available)
        self.assertEqual(snap.positions[0].unrealized_pnl, 0.0)

    def test_partial_coverage_is_unavailable_rather_than_a_mixture(self):
        """Half real and half guessed must not be reported as measured."""
        snap = self.build({"0xa": (1.0, 100.0), "0xb": (1.0, None)})
        self.assertFalse(snap.pnl_available)


class TestWhaleWarningAccuracy(unittest.TestCase):
    """The engine's warning must describe the data it actually received.

    It used to assert unconditionally that the wallet-quality confidence term
    was excluded. True of funding-derived history, false of the Reservoir
    snapshots - so every snapshot-backed run was told a term had been dropped
    while it was being used. A wrong warning is worse than none: it makes the
    output untrustworthy in a direction no reader can see.
    """

    def setUp(self):
        self.cfg_path = write_config(CONFIG)
        self.addCleanup(os.unlink, self.cfg_path)
        self.cfg = load_config(self.cfg_path)
        self.ds = make_dataset(
            [100.0 + 10 * math.sin(i / 12.0) for i in range(400)]
        )

    def series(self, wallet, with_entry):
        from backtest.whale_history import WalletPositionSeries

        item = WalletPositionSeries(wallet=wallet, coin="BTC")
        item.points = [(1_000, 1.0)]
        if with_entry:
            item.entries = [(1_000, 90.0)]
        return item

    def warnings_for(self, history):
        bt = Backtester(self.cfg, {"BTC": self.ds}, whale_history=history)
        return " ".join(bt.warnings)

    def test_entries_are_reported_as_enabling_the_quality_term(self):
        text = self.warnings_for({"0xa": self.series("0xa", True)})
        self.assertIn("wallet-quality confidence term comes from real", text)
        self.assertNotIn("excluded", text)

    def test_missing_entries_are_reported_as_excluding_it(self):
        text = self.warnings_for({"0xa": self.series("0xa", False)})
        self.assertIn("excluded", text)
        self.assertNotIn("comes from real", text)

    def test_partial_coverage_says_how_many_carry_an_entry(self):
        text = self.warnings_for(
            {"0xa": self.series("0xa", True), "0xb": self.series("0xb", False)}
        )
        self.assertIn("1 carry an entry price", text)

    def test_the_engine_makes_no_claim_about_how_wallets_were_chosen(self):
        """The engine cannot see the selection, so it must not describe it.

        An earlier version asserted "selected by position size and persistence,
        not by leaderboard profitability" unconditionally. That was true of the
        size-filtered loader and false of the reconstructed-ranking loader,
        which selects on profitability by construction - so adding the second
        source silently turned the warning into a lie.
        """
        for history in (
            {"0xa": self.series("0xa", True)},
            {"0xa": self.series("0xa", False)},
        ):
            with self.subTest(has_entries=history["0xa"].has_pnl):
                text = self.warnings_for(history)
                self.assertNotIn("position size and persistence", text)
                self.assertNotIn("not by leaderboard profitability", text)

    def test_each_loader_states_its_own_basis(self):
        """The claim moved to the place that knows it.

        `run_backtest` prints the loader's notes verbatim, so whichever loader
        ran is the one that describes the wallet set.
        """
        import inspect

        from backtest.position_history import (
            load_leaderboard_history,
            load_position_history,
        )

        self.assertIn(
            "approximated by position size and persistence",
            inspect.getsource(load_position_history),
        )
        self.assertIn(
            "realised-PnL", inspect.getsource(load_leaderboard_history)
        )


class TestLeaderboardReconstruction(unittest.TestCase):
    """The reconstructed ranking must match production's selection rule.

    Production ranks each window by its `pnl` figure, takes the top N from
    each, unions them and keeps the ones that persist. Every assertion here is
    about one of those steps, because a reconstruction that ranks by something
    else produces a wallet set that never existed.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Imported lazily so the module's duckdb dependency is only exercised
        # by the tests that need it.
        from backtest import leaderboard as module

        self.module = module

    def test_rank_sum_break_keeps_the_better_wallets(self):
        """A rank of 1 is best, so the sum must be ordered ascending.

        Written as a source assertion because the consequence only appears
        when the cap binds. Measured against the archive: with the cap at 100
        the descending order kept wallets averaging a rank-sum of 21,118
        against 11,545 for ascending - it was selecting the worse half.
        """
        source = Path(self.module.__file__).read_text(encoding="utf-8")
        self.assertIn("ORDER BY c.persistence DESC, {tiebreak} ASC", source)
        self.assertNotIn("{tiebreak} DESC", source)

    def test_cap_is_applied_after_the_tiebreak(self):
        source = Path(self.module.__file__).read_text(encoding="utf-8")
        self.assertIn("QUALIFY ROW_NUMBER() OVER", source)

    def test_signature_tracks_every_file_not_just_their_count(self):
        """The sync keeps adding days, so a coarser key would serve a ranking
        built from a different archive than the one on disk."""
        (self.root / "date=2026-01-01.parquet").write_bytes(b"a")
        first = self.module._signature(self.root, self.module.DEFAULT_WINDOWS, 60, 150, None, None)
        (self.root / "date=2026-01-02.parquet").write_bytes(b"b")
        second = self.module._signature(self.root, self.module.DEFAULT_WINDOWS, 60, 150, None, None)
        self.assertNotEqual(first, second)

    def test_signature_tracks_parameter_changes(self):
        (self.root / "date=2026-01-01.parquet").write_bytes(b"a")
        base = self.module._signature(self.root, self.module.DEFAULT_WINDOWS, 60, 150, None, None)
        other = self.module._signature(self.root, self.module.DEFAULT_WINDOWS, 30, 150, None, None)
        self.assertNotEqual(base, other)

    def test_signature_tracks_the_activity_ceiling(self):
        """A different ceiling is a different wallet set, so it must not
        reuse a cached board built without one."""
        (self.root / "date=2026-01-01.parquet").write_bytes(b"a")
        base = self.module._signature(
            self.root, self.module.DEFAULT_WINDOWS, 60, 150, None, None
        )
        capped = self.module._signature(
            self.root, self.module.DEFAULT_WINDOWS, 60, 150, None, None,
            max_fills_per_day=50,
        )
        self.assertNotEqual(base, capped)
        self.assertIsNone(base["max_fills_per_day"])
        self.assertEqual(capped["max_fills_per_day"], 50.0)

    def test_activity_ceiling_filters_on_a_trailing_average_not_one_day(self):
        """A single busy day must not disqualify an otherwise directional
        wallet, so the count is averaged over the longest ranking window.

        The first version of this asserted only that a window frame and a
        threshold exist - both of which survive replacing the trailing SUM with
        a single day's MAX, so a mutant doing exactly that went undetected. The
        averaging is the load-bearing part and is now asserted directly.
        """
        import inspect

        source = inspect.getsource(self.module.reconstruct)
        self.assertIn("SUM(fills) OVER (PARTITION BY address", source)
        self.assertIn("RANGE BETWEEN {window_days - 1} PRECEDING", source)
        self.assertIn("/ {window_days}.0 AS fills_per_day", source)
        self.assertIn("fills_per_day <= {float(max_fills_per_day)}", source)
        self.assertNotIn("MAX(fills)", source)

    def test_no_ceiling_means_no_filter_clause(self):
        """The default must reproduce production exactly, so the diagnostic
        cannot leak into a run that did not ask for it."""
        import inspect

        source = inspect.getsource(self.module.reconstruct)
        self.assertIn('activity_where = ""', source)
        self.assertIn('activity_frame = "0.0 AS fills_per_day"', source)

    def test_signature_tracks_a_rewritten_day(self):
        """A re-fetched day keeps its name, so a count-based key misses it.

        This is the case a surviving mutant exposed: the signature's
        size-and-mtime fields were untested, and only the file *list* was
        checked. A sync that overwrites one day - which is exactly what
        re-running after an interrupted fetch does - would then be served a
        ranking built from the previous contents of that day.
        """
        path = self.root / "date=2026-01-01.parquet"
        path.write_bytes(b"a")
        before = self.module._signature(
            self.root, self.module.DEFAULT_WINDOWS, 60, 150, None, None
        )
        self.assertIn([path.name, 1, int(path.stat().st_mtime)], before["files"])

        path.write_bytes(b"a much longer body")
        after = self.module._signature(
            self.root, self.module.DEFAULT_WINDOWS, 60, 150, None, None
        )
        self.assertEqual([f[0] for f in before["files"]], [f[0] for f in after["files"]])
        self.assertNotEqual(before, after)

    def real_signature(self, top=60):
        return self.module._signature(
            self.root, self.module.DEFAULT_WINDOWS, top, 150, None, None
        )

    def test_cache_round_trips_and_reuses(self):
        board = self.module.Leaderboard(
            by_day={"2026-01-01": {"0xaa": 2}}, windows=(("day", 1),)
        )
        cache = self.root / "board.json"
        signature = self.real_signature()
        self.module._save_cache(cache, board, signature)

        loaded = self.module._load_cache(cache, signature)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.by_day, {"2026-01-01": {"0xaa": 2}})
        self.assertEqual(loaded.top_per_window, 60)
        self.assertEqual(loaded.windows, (("day", 1),))

    def test_cache_is_not_reused_when_the_signature_moved(self):
        board = self.module.Leaderboard(by_day={"2026-01-01": {"0xaa": 2}})
        cache = self.root / "board.json"
        self.module._save_cache(cache, board, self.real_signature())
        # Same files, different selection parameters: a cached board from a
        # different rule describes a different wallet set.
        self.assertIsNone(self.module._load_cache(cache, self.real_signature(top=30)))

    def test_corrupt_cache_is_ignored_rather_than_raised(self):
        cache = self.root / "board.json"
        cache.write_text("{ not json")
        self.assertIsNone(self.module._load_cache(cache, self.real_signature()))


class TestWhaleCoverageWarning(unittest.TestCase):
    """Coverage is measured against the window, not against the history span.

    The first version used the span - earliest point to latest - and so
    reported full coverage for a 169-day window whose whale data stopped on day
    69, because the history also held earlier days outside the window. The
    warning never fired. The alignment gate blocks every trade once the factor
    has no data, so a window the reconstruction does not cover reads as "the
    strategy barely trades" instead of "the data stops here".
    """

    DAY = 86_400_000

    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from backtest.whale_history import WalletPositionSeries

        self.WalletPositionSeries = WalletPositionSeries

    def series_with_days(self, days):
        item = self.WalletPositionSeries(wallet="0xa", coin="BTC")
        item.points = [(d * self.DAY, 1.0) for d in days]
        return {"0xa": item}

    def test_days_outside_the_window_are_not_counted(self):
        from run_backtest import _whale_coverage_days

        # Points on days 0..9 outside the window, and 100..104 inside it.
        history = self.series_with_days(list(range(10)) + list(range(100, 105)))
        covered = _whale_coverage_days(history, 100 * self.DAY, 200 * self.DAY)
        self.assertEqual(covered, 5.0)

    def test_span_based_measurement_would_have_overstated(self):
        """The regression, stated as a contrast: the span is 104 days for a
        window that holds 5."""
        from run_backtest import _whale_coverage_days

        history = self.series_with_days(list(range(10)) + list(range(100, 105)))
        covered = _whale_coverage_days(history, 100 * self.DAY, 200 * self.DAY)
        span = (104 * self.DAY) / self.DAY
        self.assertLess(covered, span / 10)

    def test_no_points_in_window_reports_zero(self):
        from run_backtest import _whale_coverage_days

        history = self.series_with_days([0, 1, 2])
        self.assertEqual(
            _whale_coverage_days(history, 500 * self.DAY, 600 * self.DAY), 0.0
        )

    def test_a_gap_inside_the_window_is_visible(self):
        """The archive has no ABCI state for 52 days; a span-based count
        averaged that away, a day count does not."""
        from run_backtest import _whale_coverage_days

        history = self.series_with_days([100, 101, 102, 160, 161, 162])
        covered = _whale_coverage_days(history, 100 * self.DAY, 200 * self.DAY)
        self.assertEqual(covered, 6.0)


class TestSqlTransport(unittest.TestCase):
    """Statements are piped in, because a single argument cannot exceed 128 KB.

    The leaderboard loader interpolates an `IN` list of every selected wallet;
    at 3,214 wallets that is 142 KB, and `duckdb -c <sql>` fails with "Argument
    list too long" on a query that is otherwise perfectly valid.
    """

    def test_sql_runner_does_not_pass_the_statement_as_an_argument(self):
        from backtest import position_history

        source = Path(position_history.__file__).read_text(encoding="utf-8")
        self.assertIn('["duckdb", "-json"]', source)
        self.assertIn("input=sql", source)
        self.assertNotIn('"duckdb", "-json", "-c"', source)


class TestArchiveCandleLoader(unittest.TestCase):
    """The candle archive has to be a drop-in replacement for the API.

    Candle source is the binding constraint on every long backtest:
    `candleSnapshot` caps at ~5000 bars, which is 208 days at 1h and 52 at 15m,
    and once a window has used that up there is no second window to check it
    against. The archive holds 414 days at one-second resolution, so it is what
    makes an out-of-sample window - or any 15m test - possible at all. That only
    helps if the rebuilt bars are *the same bars*, which is what the parity test
    at the bottom measures.
    """

    DAY = 86_400_000

    def make_archive(self, days):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        for day in days:
            (Path(tmp) / f"date={day}.parquet").write_bytes(b"")
        return tmp

    def loader(self, archive):
        from backtest.data import HistoricalLoader

        return HistoricalLoader(cache_dir=None, candles_archive=archive)

    # ------------------------------------------------------------------
    # Partition selection
    # ------------------------------------------------------------------

    def test_exactly_the_requested_days_are_selected(self):
        archive = self.make_archive(
            ["2026-03-01", "2026-03-02", "2026-03-03", "2026-03-04"]
        )
        loader = self.loader(archive)
        start = calendar.timegm(time.strptime("2026-03-02", "%Y-%m-%d"))
        end = calendar.timegm(time.strptime("2026-03-04", "%Y-%m-%d"))
        files = loader._archive_files(start * 1000, end * 1000)
        names = [Path(f).name for f in files]
        # Both endpoints are inclusive: a window ending at midnight still needs
        # the bar that opens at that instant.
        self.assertEqual(
            names,
            ["date=2026-03-02.parquet", "date=2026-03-03.parquet",
             "date=2026-03-04.parquet"],
        )

    def test_days_the_archive_lacks_are_skipped_not_faked(self):
        """The archive has a 52-day gap. Missing days must drop out silently
        here and be reported as a coverage warning later - inventing an empty
        placeholder would turn a data gap into a bar of garbage."""
        archive = self.make_archive(["2026-03-01", "2026-03-05"])
        loader = self.loader(archive)
        start = calendar.timegm(time.strptime("2026-03-01", "%Y-%m-%d"))
        end = calendar.timegm(time.strptime("2026-03-05", "%Y-%m-%d"))
        names = [
            Path(f).name
            for f in loader._archive_files(start * 1000, end * 1000)
        ]
        self.assertEqual(
            names, ["date=2026-03-01.parquet", "date=2026-03-05.parquet"]
        )

    def test_a_missing_archive_directory_names_the_command_to_build_it(self):
        from backtest.data import BacktestDataError, HistoricalLoader

        with self.assertRaises(BacktestDataError) as ctx:
            HistoricalLoader(cache_dir=None, candles_archive="/nonexistent/x")
        self.assertIn("sync_reservoir", str(ctx.exception))

    def test_no_flag_means_the_api_path_is_used(self):
        from backtest.data import HistoricalLoader

        self.assertIsNone(HistoricalLoader(cache_dir=None).candles_archive)

    # ------------------------------------------------------------------
    # The query itself
    # ------------------------------------------------------------------

    def test_bounds_are_compared_as_timestamps_not_as_bare_epochs(self):
        """White-box, and it is the one mistake that produced plausible garbage.

        The column is TIMESTAMPTZ. Comparing it against `epoch_ms(...)` - which
        returns a bare TIMESTAMP - casts through the session time zone, so the
        window silently shifts by the machine's UTC offset and you get real bars
        from the wrong hours. It looks like wrong data, not like a bug. Measured
        once: `epoch_ms` gave o=64185 against the API's o=65494 for the same
        hour, an 8-hour slip on a UTC+8 box.
        """
        from backtest import data as data_mod

        sql = data_mod.ARCHIVE_RESAMPLE_SQL
        self.assertIn("to_timestamp(", sql)
        self.assertNotIn("epoch_ms(", sql.split("WHERE")[1])

    def test_the_resample_uses_venue_style_first_and_last_ticks(self):
        """`arg_min`/`arg_max` rather than an average or a midpoint: the open is
        the first tick's open and the close is the last tick's close, which is
        how the venue builds its own bars and why the rebuild is exact."""
        from backtest import data as data_mod

        sql = data_mod.ARCHIVE_RESAMPLE_SQL
        self.assertIn("arg_min(open, timestamp)", sql)
        self.assertIn("arg_max(close, timestamp)", sql)
        self.assertIn("max(high)", sql)
        self.assertIn("min(low)", sql)

    def test_archive_bars_are_cached_under_a_distinct_key(self):
        """Sharing the API cache key would hide a divergence between the two
        sources instead of exposing it - they are supposed to agree, and if they
        ever stop agreeing that is exactly what must not be papered over."""
        from backtest.data import HistoricalLoader

        loader = HistoricalLoader(cache_dir="/tmp/x", candles_archive="/tmp")
        api_key = loader._cache_path("candles", "BTC", "1h", 0, 1000)
        arc_key = loader._cache_path("arc_candles", "BTC", "1h", 0, 1000)
        self.assertNotEqual(api_key.name, arc_key.name)

    # ------------------------------------------------------------------
    # Parity, measured
    # ------------------------------------------------------------------

    def test_rebuilt_bars_equal_the_api_bars_where_both_exist(self):
        """The check that makes the archive usable: rebuild bars from the
        one-second archive and compare against bars the venue itself returned.

        Skipped unless the archive, a cached API response and duckdb are all
        present - this is an integration check against local data, not
        something CI can be expected to have. It is also the reason the archive
        path can be trusted for windows the API cannot reach at all.
        """
        from backtest.data import DEFAULT_CANDLES_ARCHIVE, HistoricalLoader

        if shutil.which("duckdb") is None:
            self.skipTest("duckdb not on PATH")
        if not Path(DEFAULT_CANDLES_ARCHIVE).is_dir():
            self.skipTest("no candle archive on this machine")

        cached = sorted(
            Path("backtest_cache").glob("candles_BTC_1h_*.json"),
            key=lambda p: -p.stat().st_size,
        )
        if not cached:
            self.skipTest("no cached API candles to compare against")

        payload = json.loads(cached[0].read_text(encoding="utf-8"))
        stamps = payload["t"]
        loader = self.loader(DEFAULT_CANDLES_ARCHIVE)

        compared = 0
        for index in (5, len(stamps) // 2, len(stamps) - 10):
            ts = stamps[index]
            series = loader._load_candles_archive(
                "BTC", "1h", ts, ts + 3_600_000, None, None
            )
            # The API bar carries the hour's own timestamp, so match on it.
            self.assertEqual(series.times[0], ts)
            self.assertAlmostEqual(series.opens[0], payload["o"][index], places=6)
            self.assertAlmostEqual(series.highs[0], payload["h"][index], places=6)
            self.assertAlmostEqual(series.lows[0], payload["l"][index], places=6)
            self.assertAlmostEqual(series.closes[0], payload["c"][index], places=6)
            self.assertAlmostEqual(
                series.volumes[0], payload["v"][index], places=6
            )
            compared += 1
        self.assertEqual(compared, 3)


class TestMacroFilterSilentlyOff(unittest.TestCase):
    """A run without BTC loaded loses the macro filter, silently.

    The filter asks the market for BTC bars whichever market is being traded,
    and `safe_candles` swallows the lookup failure. So the gate simply stops
    running: no exception, no zero-division, just a rejection count with no
    `macro filter` rows. Measured on a 170-day window, ETH saw 0 blocks instead
    of 101 and SOL 0 instead of 218, and SOL's return went from -0.23% to
    +1.88% depending only on whether BTC was in the dict.

    `run_backtest.py` loads BTC alongside any market for this reason, so the CLI
    is safe. This guards every other caller - the benchmarks module, a sweep
    script, anything assembling `datasets` by hand.
    """

    def setUp(self):
        self.cfg_path = write_config(CONFIG)
        self.addCleanup(os.unlink, self.cfg_path)
        self.cfg = load_config(self.cfg_path)

    def warnings(self, datasets):
        # Trading ETH, so the engine's own lookup of the traded market succeeds
        # and only the BTC lookup - the one the macro filter makes - is missing.
        bt = Backtester(replace(self.cfg, symbol="ETH"), datasets)
        return " ".join(bt.warnings)

    def dataset(self):
        return make_dataset([100.0 + 10 * math.sin(i / 12.0) for i in range(400)])

    def test_missing_btc_is_reported(self):
        text = self.warnings({"ETH": self.dataset()})
        self.assertIn("macro filter", text)
        self.assertIn("BTC was not loaded", text)

    def test_btc_present_stays_quiet(self):
        text = self.warnings(
            {"ETH": self.dataset(), "BTC": self.dataset()}
        )
        self.assertNotIn("macro filter", text)

    def test_the_lowercase_key_also_counts_as_loaded(self):
        """The check is on uppercase keys because `BacktestMarket` uppercases
        them when it stores the dict. A lowercase 'btc' that the market looks up
        successfully must not still warn."""
        text = self.warnings(
            {"ETH": self.dataset(), "btc": self.dataset()}
        )
        self.assertNotIn("BTC was not loaded", text)


class TestEntryTimeframeOverride(unittest.TestCase):
    """Changing the entry timeframe must not change anything else.

    The 15m experiment is only interpretable if the timeframe is the single
    difference: if the trend timeframe moved too, a worse result could be either
    effect and neither could be attributed. The override is a `replace` on the
    config rather than a copied config file, because a copy drifts from the real
    one and then the run being reported is not the run that was configured.
    """

    def setUp(self):
        self.cfg_path = write_config(CONFIG)
        self.addCleanup(os.unlink, self.cfg_path)
        self.cfg = load_config(self.cfg_path)

    def test_only_the_entry_timeframe_moves(self):
        from dataclasses import replace as _replace

        before = self.cfg.indicators
        after = _replace(before, entry_timeframe="15m")

        self.assertEqual(after.entry_timeframe, "15m")
        self.assertEqual(after.trend_timeframe, before.trend_timeframe)
        for field in (
            "supertrend_period",
            "supertrend_multiplier",
            "adx_period",
            "rsi_period",
            "ema_fast",
            "ema_slow",
            "lookback_candles",
        ):
            with self.subTest(field=field):
                self.assertEqual(getattr(after, field), getattr(before, field))

    def test_the_frequency_gates_do_not_scale_with_the_timeframe(self):
        """`max_signals_per_day` and `cooldown_minutes` are wall-clock limits.

        A cooldown of 240 minutes is 4 bars at 1h and 16 at 15m, so it binds
        *less* on the finer timeframe - which is why the flat trade count at 15m
        cannot be explained by these gates. Kept as a test because that is the
        intuitive explanation and it is the one that turned out to be wrong: the
        engine's own counters show `daily cap` never fired, and the flat count
        came from threshold rejections rising instead.
        """
        self.assertGreater(self.cfg.discipline.cooldown_minutes, 0)
        self.assertGreater(self.cfg.discipline.max_signals_per_day, 0)


class TestDailyStalenessWindow(unittest.TestCase):
    """Daily snapshots need a window measured in days, not hours.

    The engine default is sized for hourly funding records. Applied to daily
    snapshots it leaves the whale book empty for all but one bar in
    twenty-four, and does so silently: the factor simply reports no positions.
    Measured against the real archive, 12 of 48 bars were populated at six
    hours and 48 of 48 at twenty-six.
    """

    DAY = 86_400_000

    def history(self):
        from backtest.whale_history import WalletPositionSeries

        series = WalletPositionSeries(wallet="0xa", coin="BTC")
        # End-of-day marks, one per day.
        series.points = [(self.DAY - 1, 2.0), (2 * self.DAY - 1, 3.0)]
        return {"0xa": series}

    def holders_at(self, ts_ms, max_age_ms):
        from backtest.historical_smart_money import HistoricalSmartMoney

        source = HistoricalSmartMoney(
            self.history(),
            price_lookup=lambda _t: 100.0,
            window_count=3,
            max_age_ms=max_age_ms,
        )
        source.set_now(ts_ms)
        return source.build_snapshot("BTC").wallet_count

    def test_a_daily_window_sees_a_position_on_every_bar(self):
        for hour in range(24):
            with self.subTest(hour=hour):
                self.assertEqual(
                    self.holders_at(self.DAY + hour * 3_600_000, 26 * 3_600_000),
                    1,
                )

    def test_an_hourly_window_is_blind_for_three_quarters_of_the_day(self):
        """Six fresh hours out of twenty-four, so eighteen are blind.

        The arithmetic is deterministic, so this asserts the exact count rather
        than a loose bound: the point is the ratio, and a bound would pass for
        a window that is only slightly wrong.
        """
        blind = sum(
            1
            for hour in range(24)
            if self.holders_at(self.DAY + hour * 3_600_000, 6 * 3_600_000) == 0
        )
        self.assertEqual(blind, 18)


class TestEndOfDayTimestamp(unittest.TestCase):
    def test_lands_on_the_last_second_of_the_utc_day(self):
        import time as _time

        from backtest.position_history import end_of_day_ms

        ms = end_of_day_ms("2026-03-31")
        stamp = _time.strftime("%Y-%m-%d %H:%M:%S", _time.gmtime(ms / 1000))
        self.assertEqual(stamp, "2026-03-31 23:59:59")

    def test_handles_month_and_year_boundaries(self):
        import time as _time

        from backtest.position_history import end_of_day_ms

        self.assertEqual(
            _time.strftime("%Y-%m-%d", _time.gmtime(end_of_day_ms("2025-12-31") / 1000)),
            "2025-12-31",
        )
        self.assertEqual(
            _time.strftime("%Y-%m-%d", _time.gmtime(end_of_day_ms("2026-01-01") / 1000)),
            "2026-01-01",
        )


class TestSymbolSelection(unittest.TestCase):
    """`--symbols` exists because sample size is the binding constraint.

    A single 90-day market yields about 14 trades, which the metrics module
    itself flags as below the threshold for the distributional statistics to
    mean anything. Pooling markets is the cheap way to get a sample worth
    drawing an inference from.
    """

    class Args:
        def __init__(self, symbol=None, symbols=None):
            self.symbol = symbol
            self.symbols = symbols

    class Config:
        symbol = "BTC"

    def resolve(self, **kwargs):
        from run_backtest import _resolve_symbols

        return _resolve_symbols(self.Args(**kwargs), self.Config())

    def test_defaults_to_the_config_symbol(self):
        self.assertEqual(self.resolve(), ["BTC"])

    def test_symbol_overrides_the_config(self):
        self.assertEqual(self.resolve(symbol="eth"), ["ETH"])

    def test_symbols_are_split_and_normalised(self):
        self.assertEqual(
            self.resolve(symbols=" btc , eth ,sol "), ["BTC", "ETH", "SOL"]
        )

    def test_combining_both_options_is_an_error(self):
        """Silently preferring one would run something other than what was
        typed, which is how a mistyped command produces a believable result
        from the wrong market."""
        with self.assertRaises(ValueError):
            self.resolve(symbol="BTC", symbols="ETH")

    def test_blank_symbols_is_an_error(self):
        with self.assertRaises(ValueError):
            self.resolve(symbols=" , ")


if __name__ == "__main__":
    unittest.main(verbosity=2)
