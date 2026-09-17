"""Backtest tests.

The most important test in this file is `TestNoLookahead`. A backtest that
leaks future bars can report anything, and a negative result from a buggy
engine is just as untrustworthy as a positive one from an overfitted one.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
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
        from backtest.engine import BacktestResult, Trade

        trades = []
        for i, pnl in enumerate(pnls):
            trades.append(
                Trade(
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
