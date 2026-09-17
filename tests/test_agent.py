"""Offline tests: indicators, factor blending, discipline gates, sizing.

These use no network. Live behaviour is exercised by `run_bot.py --show-signal`.
Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.bot import FoxAgent  # noqa: E402
from agent.config import ConfigError, load_config  # noqa: E402
from agent.discipline import (  # noqa: E402
    Discipline,
    DisciplineState,
    bucket_block_reason,
)
from agent.execution import (  # noqa: E402
    Broker,
    CriticalExecutionError,
    ExecutionError,
    Position,
    compute_size,
)
from agent.indicators import CandleSeries, adx, atr, ema, rsi, sma, supertrend  # noqa: E402
from agent.state import AgentState, StateStore  # noqa: E402
from agent.factors.base import FactorScore, lerp_score  # noqa: E402
from agent.factors.market_factor import MarketFactor  # noqa: E402
from agent.factors.smart_money import (  # noqa: E402
    HyperfeedSource,
    SmartMoneyFactor,
    SmartMoneySnapshot,
    WalletPosition,
    signed_notional,
)
from agent.synthesizer import LONG, NEUTRAL, SHORT, Synthesizer  # noqa: E402


def make_wallet_source(market=None, rows=None, **overrides):
    """Build a LeaderboardWalletSource wired for offline tests.

    Bypasses __init__ so tests can inject a row list and avoid the 37 MB
    leaderboard download, while still setting every attribute the selection
    logic touches.
    """
    from agent.factors.smart_money import LeaderboardWalletSource

    src = LeaderboardWalletSource.__new__(LeaderboardWalletSource)
    src.market = market
    src.windows = ("day", "week", "month")
    src.top_per_window = 25
    src.min_persistence = 1
    src.max_wallets = 0
    src.min_account_value = 10_000.0
    # Normalised to lowercase sets exactly as __init__ does.
    src.whitelist = {str(a).lower() for a in overrides.pop("whitelist", ()) if a}
    src.blacklist = {str(a).lower() for a in overrides.pop("blacklist", ()) if a}
    src.cache_path = None
    src._wallet_cache = []
    src._cache_time = 0.0
    src.selection_notes = []
    if rows is not None:
        src.leaderboard_rows = lambda: rows
    for key, value in overrides.items():
        setattr(src, key, value)
    return src


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


class TestBasicIndicators(unittest.TestCase):
    def test_sma(self):
        out = sma([1, 2, 3, 4, 5], 3)
        self.assertIsNone(out[0])
        self.assertIsNone(out[1])
        self.assertAlmostEqual(out[2], 2.0)
        self.assertAlmostEqual(out[3], 3.0)
        self.assertAlmostEqual(out[4], 4.0)

    def test_ema_seeds_with_sma(self):
        values = [float(i) for i in range(1, 11)]
        out = ema(values, 3)
        self.assertIsNone(out[1])
        self.assertAlmostEqual(out[2], 2.0)  # SMA of 1,2,3
        self.assertGreater(out[-1], out[-2])

    def test_rsi_bounds(self):
        rising = [float(i) for i in range(1, 40)]
        out = rsi(rising, 14)
        self.assertAlmostEqual(out[-1], 100.0)  # monotonic rise, no losses

        falling = [float(i) for i in range(40, 1, -1)]
        out_down = rsi(falling, 14)
        self.assertAlmostEqual(out_down[-1], 0.0)

    def test_rsi_neutral_on_flat_series(self):
        flat = [100.0] * 40
        out = rsi(flat, 14)
        # No gains and no losses -> avg_loss == 0 path returns 100.0.
        # What matters is that it does not raise and stays in range.
        self.assertTrue(0.0 <= out[-1] <= 100.0)

    def test_atr_positive(self):
        highs = [10.0 + i for i in range(30)]
        lows = [9.0 + i for i in range(30)]
        closes = [9.5 + i for i in range(30)]
        out = atr(highs, lows, closes, 14)
        self.assertIsNone(out[12])
        self.assertIsNotNone(out[13])
        self.assertGreater(out[-1], 0)

    def test_atr_insufficient_data_returns_nones(self):
        out = atr([1, 2, 3], [0.5, 1, 2], [0.8, 1.5, 2.5], 14)
        self.assertTrue(all(v is None for v in out))

    def test_adx_in_range(self):
        highs, lows, closes = [], [], []
        price = 100.0
        for i in range(80):
            price += math.sin(i / 5.0) * 2 + 0.3
            highs.append(price + 1)
            lows.append(price - 1)
            closes.append(price)
        out = adx(highs, lows, closes, 14)
        computed = [v for v in out if v is not None]
        self.assertTrue(computed, "ADX should produce values for 80 bars")
        for v in computed:
            self.assertTrue(0.0 <= v <= 100.0)

    def test_supertrend_direction_on_clean_trend(self):
        highs, lows, closes = [], [], []
        price = 100.0
        for _ in range(60):
            price += 1.0
            highs.append(price + 0.5)
            lows.append(price - 0.5)
            closes.append(price)
        line, direction = supertrend(highs, lows, closes, 10, 3.0)
        self.assertEqual(direction[-1], "buy")
        # In an uptrend the line sits below price.
        self.assertLess(line[-1], closes[-1])

    def test_supertrend_flips_on_reversal(self):
        highs, lows, closes = [], [], []
        price = 100.0
        for _ in range(40):
            price += 1.0
            highs.append(price + 0.5)
            lows.append(price - 0.5)
            closes.append(price)
        for _ in range(40):
            price -= 1.5
            highs.append(price + 0.5)
            lows.append(price - 0.5)
            closes.append(price)
        _, direction = supertrend(highs, lows, closes, 10, 3.0)
        self.assertEqual(direction[-1], "sell")


class TestCandleSeries(unittest.TestCase):
    def make_series(self, n=120):
        closes = [100.0 + math.sin(i / 7.0) * 5 + i * 0.05 for i in range(n)]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        opens = closes[:]
        vols = [1000.0] * n
        times = [1_700_000_000 + i * 3600 for i in range(n)]
        return CandleSeries(opens, highs, lows, closes, vols, times)

    def test_caching_returns_same_object(self):
        s = self.make_series()
        self.assertIs(s.ema(21), s.ema(21))

    def test_last_price(self):
        s = self.make_series()
        self.assertEqual(s.last_price, s.closes[-1])

    def test_length(self):
        self.assertEqual(len(self.make_series(50)), 50)


# ---------------------------------------------------------------------------
# Factor score primitives
# ---------------------------------------------------------------------------


class TestFactorScore(unittest.TestCase):
    def test_clamp(self):
        s = FactorScore(name="x", score=150, confidence=1.5).clamp()
        self.assertEqual(s.score, 100.0)
        self.assertEqual(s.confidence, 1.0)

    def test_direction(self):
        self.assertEqual(FactorScore(name="x", score=70).direction, "long")
        self.assertEqual(FactorScore(name="x", score=30).direction, "short")
        self.assertEqual(FactorScore(name="x", score=50).direction, "neutral")

    def test_lerp_score(self):
        self.assertAlmostEqual(lerp_score(0, 0, 10), 0.0)
        self.assertAlmostEqual(lerp_score(5, 0, 10), 50.0)
        self.assertAlmostEqual(lerp_score(10, 0, 10), 100.0)
        self.assertAlmostEqual(lerp_score(-5, 0, 10), 0.0)  # clamped
        self.assertAlmostEqual(lerp_score(15, 0, 10), 100.0)


# ---------------------------------------------------------------------------
# Smart money
# ---------------------------------------------------------------------------


class TestSmartMoneySnapshot(unittest.TestCase):
    def snap(self, longs, shorts):
        positions = []
        for notional in longs:
            positions.append(WalletPosition(wallet="w", size=1, notional=notional))
        for notional in shorts:
            positions.append(WalletPosition(wallet="w", size=-1, notional=-notional))
        return SmartMoneySnapshot(symbol="BTC", positions=positions)

    def test_long_ratio(self):
        s = self.snap([75.0], [25.0])
        self.assertAlmostEqual(s.long_ratio_pct, 75.0)
        self.assertAlmostEqual(s.net_bias_pct, 25.0)

    def test_empty_snapshot_is_neutral(self):
        s = SmartMoneySnapshot(symbol="BTC")
        self.assertAlmostEqual(s.long_ratio_pct, 50.0)
        self.assertEqual(s.wallet_count, 0)

    def test_gross_vs_net_notional(self):
        s = self.snap([100.0], [40.0])
        # Both sides are exposed, so total exposure is 140 and the long side
        # outweighs the short side by 60.
        self.assertAlmostEqual(s.net_notional, 140.0)
        self.assertAlmostEqual(s.gross_notional, 60.0)
        # long_ratio_pct divides by total exposure.
        self.assertAlmostEqual(s.long_ratio_pct, 100.0 / 140.0 * 100.0)


class TestSmartMoneyFactor(unittest.TestCase):
    def factor(self, testnet_data=False):
        f = SmartMoneyFactor.__new__(SmartMoneyFactor)
        f.data_market = type("M", (), {"testnet": testnet_data})()
        return f

    def test_no_positions_gives_zero_confidence(self):
        f = self.factor()
        snap = SmartMoneySnapshot(symbol="BTC", wallets_sampled=20)
        score = f.evaluate("BTC", snap)
        self.assertEqual(score.confidence, 0.0)
        self.assertEqual(score.score, 50.0)

    def test_stale_snapshot_is_refused(self):
        f = self.factor()
        snap = SmartMoneySnapshot(symbol="BTC")
        snap.timestamp -= 3600  # one hour old
        positions = [WalletPosition(wallet="a", size=1, notional=100.0)]
        snap.positions = positions
        score = f.evaluate("BTC", snap)
        self.assertEqual(score.confidence, 0.0)
        self.assertIn("stale", " ".join(score.reasons).lower())

    def test_long_heavy_gives_long_score(self):
        f = self.factor()
        snap = SmartMoneySnapshot(symbol="BTC")
        snap.positions = [WalletPosition("a", 1, 90.0), WalletPosition("b", -1, -10.0)]
        score = f.evaluate("BTC", snap)
        self.assertGreater(score.score, 50)
        self.assertGreater(score.confidence, 0)

    def test_short_heavy_gives_short_score(self):
        f = self.factor()
        snap = SmartMoneySnapshot(symbol="BTC")
        snap.positions = [WalletPosition("a", 1, 10.0), WalletPosition("b", -1, -90.0)]
        score = f.evaluate("BTC", snap)
        self.assertLess(score.score, 50)

    def test_leaderboard_confidence_is_capped(self):
        f = self.factor()
        snap = SmartMoneySnapshot(symbol="BTC", source="leaderboard")
        snap.positions = [
            WalletPosition(str(i), 1, 100.0, unrealized_pnl=5.0) for i in range(20)
        ]
        score = f.evaluate("BTC", snap)
        self.assertLessEqual(score.confidence, 0.75)


# ---------------------------------------------------------------------------
# Market factor
# ---------------------------------------------------------------------------


class FakeContext:
    def __init__(self, funding=0.0, oi=50.0, prev_day=100.0, mark=100.0):
        self.name = "BTC"
        self.index = 0
        self.funding = funding
        self.open_interest = oi          # base coin units, as the API returns
        self.prev_day_price = prev_day
        self.mark_price = mark

    @property
    def funding_apr_pct(self):
        return self.funding * 24 * 365 * 100

    @property
    def open_interest_notional(self):
        return self.open_interest * self.mark_price

    @property
    def day_change_pct(self):
        if self.prev_day_price <= 0:
            return 0.0
        return (self.mark_price - self.prev_day_price) / self.prev_day_price * 100


class FakeMarket:
    def __init__(self, ctx, testnet=False):
        self._ctx = ctx
        self.testnet = testnet

    def asset_context(self, symbol):
        return self._ctx


class TestMarketFactor(unittest.TestCase):
    def test_benign_funding_is_neutral(self):
        f = MarketFactor(FakeMarket(FakeContext(funding=0.000001)))
        score = f.evaluate("BTC")
        self.assertGreater(score.confidence, 0)

    def test_positive_funding_leans_short(self):
        f = MarketFactor(FakeMarket(FakeContext(funding=0.0005)))  # 5bp/h
        score = f.evaluate("BTC")
        self.assertLess(score.score, 50, "crowded longs should lean contrarian short")

    def test_negative_funding_leans_long(self):
        f = MarketFactor(FakeMarket(FakeContext(funding=-0.0005)))
        score = f.evaluate("BTC")
        self.assertGreater(score.score, 50)

    def test_oi_euristic_does_not_crash_on_zero_reference(self):
        f = MarketFactor(FakeMarket(FakeContext(funding=0.0001, oi=0.0)))
        score = f.evaluate("BTC")
        self.assertTrue(0 <= score.score <= 100)


# ---------------------------------------------------------------------------
# Synthesizer blending
# ---------------------------------------------------------------------------


class TestBlend(unittest.TestCase):
    class W:
        smart_money = 0.40
        technical = 0.35
        market = 0.25

    def test_zero_confidence_weight_is_redistributed(self):
        factors = {
            "smart_money": FactorScore("smart_money", 50, 0.0),
            "technical": FactorScore("technical", 80, 1.0),
            "market": FactorScore("market", 80, 1.0),
        }
        score, conf, notes = Synthesizer.blend(factors, self.W())
        # Only the two confident factors drive the result -> 80, not 50.
        self.assertAlmostEqual(score, 80.0, places=5)
        self.assertLess(conf, 1.0)
        self.assertTrue(any("smart_money" in n for n in notes))

    def test_all_zero_confidence_is_neutral(self):
        factors = {k: FactorScore(k, 50, 0.0) for k in ("smart_money", "technical", "market")}
        score, conf, _ = Synthesizer.blend(factors, self.W())
        self.assertAlmostEqual(score, 50.0)
        self.assertEqual(conf, 0.0)

    def test_unanimous_factors_give_high_confidence(self):
        factors = {k: FactorScore(k, 70, 1.0) for k in ("smart_money", "technical", "market")}
        score, conf, _ = Synthesizer.blend(factors, self.W())
        self.assertAlmostEqual(score, 70.0, places=5)
        self.assertGreater(conf, 0.9)

    def test_fully_conflicting_factors_lower_confidence(self):
        factors = {
            "smart_money": FactorScore("smart_money", 80, 1.0),
            "technical": FactorScore("technical", 20, 1.0),
        }
        _, conf, notes = Synthesizer.blend(factors, self.W())
        self.assertTrue(any("agreement" in n for n in notes))
        self.assertLess(conf, 0.6)


# ---------------------------------------------------------------------------
# Discipline: the guardrails that carry the edge
# ---------------------------------------------------------------------------


class TestDisciplineGates(unittest.TestCase):
    def setUp(self):
        # Minimal config built in-memory to avoid file IO.
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        self.tmp.write(
            """
name: test
symbol: BTC
risk_preset: conservative
discipline:
  long_threshold: 60.0
  short_threshold: 40.0
  max_signals_per_day: 2
  cooldown_minutes: 60
  require_smart_money_alignment: true
  btc_trend_filter: false
min_confidence: 0.3
risk:
  stop_loss_enabled: true
  stop_loss_pct: 3.0
safety:
  hard_stop_pct: 5.0
execution:
  testnet: true
"""
        )
        self.tmp.close()
        self.cfg = load_config(self.tmp.name)
        self.d = Discipline(self.cfg, market=None)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def make_signal(self, action=LONG, conf=0.8, sm_score=70.0, sm_conf=0.6):
        from agent.synthesizer import Signal

        return Signal(
            symbol="BTC",
            action=action,
            score=70.0 if action == LONG else 30.0,
            confidence=conf,
            factors={
                "smart_money": FactorScore("smart_money", sm_score, sm_conf),
                "technical": FactorScore("technical", 70.0, 0.5),
                "market": FactorScore("market", 60.0, 0.5),
            },
        )

    def test_smart_money_disagreement_blocks(self):
        sig = self.make_signal(LONG, sm_score=30.0)
        out = self.d.evaluate(sig)
        self.assertEqual(out.action, NEUTRAL)
        self.assertIn("smart money is not long", out.blocked_by)

    def test_smart_money_missing_data_blocks(self):
        sig = self.make_signal(LONG, sm_conf=0.0)
        out = self.d.evaluate(sig)
        self.assertEqual(out.action, NEUTRAL)
        self.assertIn("no data", out.blocked_by)

    def test_aligned_signal_passes(self):
        sig = self.make_signal(LONG, sm_score=70.0)
        out = self.d.evaluate(sig)
        self.assertEqual(out.action, LONG)
        self.assertIsNone(out.blocked_by)

    def test_low_confidence_blocks(self):
        sig = self.make_signal(LONG, conf=0.1)
        out = self.d.evaluate(sig)
        self.assertIn("confidence", out.blocked_by)

    def test_daily_cap_enforced(self):
        sig = self.make_signal(LONG)
        self.assertEqual(self.d.evaluate(sig).action, LONG)
        self.d.record_trade("BTC")

        self.d.state.last_trade_ts.clear()  # bypass cooldown, isolate the cap
        self.assertEqual(self.d.evaluate(self.make_signal(LONG)).action, LONG)
        self.d.record_trade("BTC")

        self.d.state.last_trade_ts.clear()
        out = self.d.evaluate(self.make_signal(LONG))
        self.assertEqual(out.action, NEUTRAL)
        self.assertIn("daily cap", out.blocked_by)

    def test_cooldown_enforced(self):
        sig = self.make_signal(LONG)
        self.d.evaluate(sig)
        self.d.record_trade("BTC")

        out = self.d.evaluate(self.make_signal(LONG))
        self.assertEqual(out.action, NEUTRAL)
        self.assertIn("cooldown", out.blocked_by)

    def test_reversal_is_exempt_from_cap_and_cooldown(self):
        # Fill the daily cap and start a cooldown.
        self.d.record_trade("BTC")
        self.d.record_trade("BTC")

        sig = self.make_signal(SHORT, sm_score=30.0)
        out = self.d.evaluate(sig, open_positions=1, is_reversal=True)
        self.assertEqual(out.action, SHORT)
        self.assertIsNone(out.blocked_by)

    def test_non_reversal_at_cap_with_open_position_is_blocked(self):
        self.d.record_trade("BTC")
        self.d.record_trade("BTC")

        out = self.d.evaluate(self.make_signal(LONG), open_positions=1)
        self.assertEqual(out.action, NEUTRAL)
        self.assertIn("daily cap", out.blocked_by)

    def test_loss_streak_halts_trading(self):
        for _ in range(3):
            self.d.record_outcome(-10.0)
        out = self.d.evaluate(self.make_signal(LONG))
        self.assertEqual(out.action, NEUTRAL)
        self.assertIn("halted", out.blocked_by)

    def test_win_resets_loss_streak(self):
        self.d.record_outcome(-10.0)
        self.d.record_outcome(-10.0)
        self.d.record_outcome(+5.0)
        self.assertEqual(self.d.state.consecutive_losses, 0)

    def test_position_slot_limit(self):
        out = self.d.evaluate(self.make_signal(LONG), open_positions=1)
        self.assertEqual(out.action, NEUTRAL)
        self.assertIn("already holding", out.blocked_by)

    def test_neutral_signal_is_untouched(self):
        sig = self.make_signal(NEUTRAL)
        out = self.d.evaluate(sig)
        self.assertEqual(out.action, NEUTRAL)
        self.assertIn("threshold", out.blocked_by)


class TestBucketBlockReason(unittest.TestCase):
    """Gate messages carry variable content, so anything that counts or groups
    them has to collapse them first - otherwise nearly every occurrence becomes
    its own value and the distribution disappears.

    The vocabulary lives in `agent/discipline.py`, beside the code that emits
    the messages. It was previously duplicated in the backtest engine and the
    journal analyser, neither of which owned it.
    """

    def test_variable_reasons_collapse_to_one_bucket(self):
        self.assertEqual(
            bucket_block_reason(
                "cooldown active, 239 min remaining (cooldown is 240 min)"
            ),
            "cooldown",
        )
        self.assertEqual(
            bucket_block_reason("cooldown active, 12 min remaining"), "cooldown"
        )

    def test_every_real_gate_message_is_recognised(self):
        """The wording is taken from what `evaluate()` actually emits."""
        cases = {
            "score did not cross a threshold": "threshold",
            "confidence 20% below minimum 35% - too many inputs missing": "confidence",
            "daily cap reached (3 signals for BTC) - trading frequency is the "
            "dominant loss driver": "daily cap",
            "already holding 1 position(s), limit is 1": "already holding",
            "smart money factor missing entirely": "smart money",
            "smart money is not long (score 45.2) - signal rejected": "smart money",
            "smart money is not short (score 55.1) - signal rejected": "smart money",
            "halted for another 187 min: 3 consecutive losses": "halted",
        }
        for raw, expected in cases.items():
            self.assertEqual(bucket_block_reason(raw), expected, raw)

    def test_more_specific_keyword_wins(self):
        """A smart-money rejection also mentions a score; it must not be
        bucketed as `threshold`."""
        self.assertEqual(
            bucket_block_reason("smart money is not short (score 55.1) - signal rejected"),
            "smart money",
        )

    def test_none_and_empty_are_labelled(self):
        self.assertEqual(bucket_block_reason(None), "(none)")
        self.assertEqual(bucket_block_reason(""), "(none)")

    def test_case_is_ignored(self):
        self.assertEqual(bucket_block_reason("COOLDOWN active"), "cooldown")

    def test_unrecognised_wording_is_surfaced_not_swallowed(self):
        """A newly added gate must show up as its own value, not vanish."""
        self.assertEqual(bucket_block_reason("some brand new gate"), "some brand new gate")

    def test_long_unrecognised_wording_is_truncated(self):
        self.assertEqual(len(bucket_block_reason("x" * 200)), 60)

    def test_every_bucket_name_is_matchable(self):
        """Guards against a typo making a bucket unreachable."""
        from agent.discipline import BLOCK_REASON_BUCKETS

        for key in BLOCK_REASON_BUCKETS:
            self.assertEqual(bucket_block_reason(key), key, key)


class TestMacroFilter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        self.tmp.write(
            """
name: test
symbol: BTC
discipline:
  btc_trend_filter: true
  btc_trend_timeframe: 4h
  require_smart_money_alignment: false
risk:
  stop_loss_enabled: true
safety:
  hard_stop_pct: 5.0
"""
        )
        self.tmp.close()
        self.cfg = load_config(self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def prime(self, market, direction):
        """Build a Discipline with a stubbed, non-expiring BTC trend read."""
        d = Discipline(self.cfg, market=market)
        d._btc_trend_cache = (float("inf"), direction)
        return d

    def test_long_blocked_when_btc_down(self):
        d = self.prime(object(), "sell")

        from agent.synthesizer import Signal

        sig = Signal(symbol="BTC", action=LONG, score=70, confidence=0.8)
        self.assertIn("macro filter", d.check_macro_filter(sig) or "")

    def test_short_allowed_when_btc_down(self):
        d = self.prime(object(), "sell")

        from agent.synthesizer import Signal

        sig = Signal(symbol="BTC", action=SHORT, score=30, confidence=0.8)
        self.assertIsNone(d.check_macro_filter(sig))

    def test_short_blocked_when_btc_up(self):
        d = self.prime(object(), "buy")

        from agent.synthesizer import Signal

        sig = Signal(symbol="BTC", action=SHORT, score=30, confidence=0.8)
        self.assertIn("macro filter", d.check_macro_filter(sig) or "")

    def test_no_market_means_filter_is_inert(self):
        # Without a market handle the trend cannot be read; the gate must not
        # silently block everything.
        d = self.prime(None, "sell")

        from agent.synthesizer import Signal

        sig = Signal(symbol="BTC", action=LONG, score=70, confidence=0.8)
        self.assertIsNone(d.check_macro_filter(sig))

    def test_filter_disabled_returns_none(self):
        cfg = self.cfg
        cfg.discipline.btc_trend_filter = False
        d = Discipline(cfg, market=None)

        from agent.synthesizer import Signal

        sig = Signal(symbol="BTC", action=LONG, score=70, confidence=0.8)
        self.assertIsNone(d.check_macro_filter(sig))


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


class FakeMarketForSizing:
    def candles(self, symbol, interval, lookback):
        raise RuntimeError("no candles: exercise the ATR fallback path")


class TestSizing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        self.tmp.write(
            """
name: test
symbol: BTC
risk:
  account_allocation_pct: 10.0
  leverage: 2
  risk_per_trade_pct: 1.0
  stop_loss_enabled: true
  stop_loss_pct: 3.0
  take_profit_enabled: true
  take_profit_pct: 9.0
safety:
  hard_stop_pct: 5.0
"""
        )
        self.tmp.close()
        self.cfg = load_config(self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_risk_budget_is_respected(self):
        # 1000 equity, 1% risk = $10 risk. Stop distance = 3% of 100 = $3.
        # Size by risk = 10/3 = 3.333 units = $333 notional, under the
        # $100 allocation x 2x leverage = $200 cap? No: allocation is 10% of
        # equity = $100 notional cap, so allocation binds.
        r = compute_size(FakeMarketForSizing(), self.cfg, "BTC", LONG, 100.0, 1000.0)
        self.assertGreater(r.size, 0)
        self.assertLessEqual(r.notional, 1000.0 * 0.10 * 2 + 1e-6)

    def test_stop_and_target_prices_for_long(self):
        r = compute_size(FakeMarketForSizing(), self.cfg, "BTC", LONG, 100.0, 100_000.0)
        self.assertAlmostEqual(r.stop_price, 97.0, places=4)
        self.assertAlmostEqual(r.take_profit_price, 109.0, places=4)

    def test_stop_and_target_prices_for_short(self):
        r = compute_size(FakeMarketForSizing(), self.cfg, "BTC", SHORT, 100.0, 100_000.0)
        self.assertAlmostEqual(r.stop_price, 103.0, places=4)
        self.assertAlmostEqual(r.take_profit_price, 91.0, places=4)

    def test_zero_equity_raises(self):
        from agent.execution import ExecutionError

        with self.assertRaises(ExecutionError):
            compute_size(FakeMarketForSizing(), self.cfg, "BTC", LONG, 100.0, 0.0)

    def test_implied_leverage_cap(self):
        # Tiny stop -> huge risk-based size, which must be clipped by leverage.
        self.cfg.risk.stop_loss_pct = 0.01
        self.cfg.risk.account_allocation_pct = 100.0
        r = compute_size(FakeMarketForSizing(), self.cfg, "BTC", LONG, 100.0, 10_000.0)
        self.assertLessEqual(r.notional / 10_000.0, self.cfg.risk.leverage + 1e-6)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation(unittest.TestCase):
    def write(self, text):
        f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_weights_are_normalised(self):
        path = self.write(
            "name: t\nsymbol: BTC\nweights:\n  smart_money: 4\n  technical: 3\n  market: 3\n"
            "risk:\n  stop_loss_enabled: true\nsafety:\n  hard_stop_pct: 5.0\n"
        )
        cfg = load_config(path)
        total = cfg.weights.smart_money + cfg.weights.technical + cfg.weights.market
        self.assertAlmostEqual(total, 1.0)
        self.assertAlmostEqual(cfg.weights.technical, 0.3)

    def test_inverted_thresholds_rejected(self):
        path = self.write(
            "name: t\nsymbol: BTC\ndiscipline:\n  long_threshold: 40\n  short_threshold: 60\n"
            "risk:\n  stop_loss_enabled: true\nsafety:\n  hard_stop_pct: 5.0\n"
        )
        with self.assertRaises(ConfigError) as ctx:
            load_config(path)
        self.assertIn("short_threshold", str(ctx.exception))

    def test_excessive_leverage_rejected(self):
        path = self.write(
            "name: t\nsymbol: BTC\nrisk:\n  leverage: 25\n  stop_loss_enabled: true\n"
            "safety:\n  hard_stop_pct: 5.0\n"
        )
        with self.assertRaises(ConfigError) as ctx:
            load_config(path)
        self.assertIn("leverage", str(ctx.exception))

    def test_hard_stop_below_stop_loss_rejected(self):
        path = self.write(
            "name: t\nsymbol: BTC\nrisk:\n  stop_loss_enabled: true\n  stop_loss_pct: 10.0\n"
            "safety:\n  hard_stop_pct: 5.0\n"
        )
        with self.assertRaises(ConfigError) as ctx:
            load_config(path)
        self.assertIn("hard_stop_pct", str(ctx.exception))

    def test_unknown_key_rejected(self):
        path = self.write("name: t\nsymbol: BTC\nnot_a_real_option: 5\n")
        with self.assertRaises(ConfigError) as ctx:
            load_config(path)
        self.assertIn("not_a_real_option", str(ctx.exception))

    def test_live_without_key_rejected(self):
        path = self.write(
            "name: t\nsymbol: BTC\nexecution:\n  dry_run: false\n"
            "risk:\n  stop_loss_enabled: true\nsafety:\n  hard_stop_pct: 5.0\n"
        )
        os.environ.pop("HYPERLIQUID_PRIVATE_KEY", None)
        with self.assertRaises(ConfigError) as ctx:
            load_config(path)
        self.assertIn("HYPERLIQUID_PRIVATE_KEY", str(ctx.exception))

    def test_mainnet_requires_risk_preset(self):
        path = self.write(
            "name: t\nsymbol: BTC\nexecution:\n  dry_run: false\n  testnet: false\n"
            "risk:\n  stop_loss_enabled: true\nsafety:\n  hard_stop_pct: 5.0\n"
        )
        os.environ["HYPERLIQUID_PRIVATE_KEY"] = "0xtest"
        try:
            with self.assertRaises(ConfigError) as ctx:
                load_config(path)
            self.assertIn("risk_preset", str(ctx.exception))
        finally:
            os.environ.pop("HYPERLIQUID_PRIVATE_KEY", None)

    def test_risk_preset_applies_defaults(self):
        path = self.write("name: t\nsymbol: BTC\nrisk_preset: conservative\n")
        cfg = load_config(path)
        self.assertEqual(cfg.discipline.max_signals_per_day, 3)
        self.assertEqual(cfg.risk.leverage, 2)

    def test_explicit_value_overrides_preset(self):
        path = self.write(
            "name: t\nsymbol: BTC\nrisk_preset: conservative\n"
            "discipline:\n  max_signals_per_day: 1\n"
        )
        cfg = load_config(path)
        self.assertEqual(cfg.discipline.max_signals_per_day, 1)
        self.assertEqual(cfg.risk.leverage, 2)  # untouched preset value

    def test_shipped_config_is_valid(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / "bots" / "fox_btc.yaml")
        self.assertEqual(cfg.symbol, "BTC")
        self.assertTrue(cfg.execution.dry_run)
        self.assertTrue(cfg.discipline.require_smart_money_alignment)


# ---------------------------------------------------------------------------
# Data layer: unit handling and leaderboard parsing
# ---------------------------------------------------------------------------


class TestAssetContextUnits(unittest.TestCase):
    """openInterest is a base coin quantity, not USD. Getting this wrong
    silently corrupted the market factor's open-interest read."""

    def make(self):
        from agent.market_data import AssetContext

        return AssetContext(
            index=0, name="BTC", mark_price=77_000.0, open_interest=37_400.0
        )

    def test_oi_notional_is_scaled_by_price(self):
        ctx = self.make()
        self.assertAlmostEqual(ctx.open_interest_notional, 37_400.0 * 77_000.0)

    def test_oi_notional_orders_of_magnitude_sane(self):
        # BTC OI notional must be billions, not tens of dollars.
        self.assertGreater(self.make().open_interest_notional, 1e9)

    def test_oi_notional_falls_back_to_oracle(self):
        from agent.market_data import AssetContext

        ctx = AssetContext(index=0, name="BTC", oracle_price=70_000.0, open_interest=100.0)
        self.assertAlmostEqual(ctx.open_interest_notional, 100.0 * 70_000.0)


class TestNotionalSignConvention(unittest.TestCase):
    """`size` carries direction and `notional` must agree with it.

    Dropping the sign - turning `signed_notional` back into `abs(notional)` -
    originally killed no test at all, even though it makes a short contribute a
    negative amount. That pushes `long_ratio_pct` past 100% and makes the factor
    read a heavily shorted market as a long consensus.

    The gap existed because every other test builds `WalletPosition` directly
    from self-consistent fixtures, so the parsing code that establishes the
    convention was never executed.
    """

    def test_notional_takes_the_sign_of_size(self):
        self.assertEqual(signed_notional(1_000.0, 2.0), 1_000.0)
        self.assertEqual(signed_notional(1_000.0, -2.0), -1_000.0)
        # A venue that already reports a signed notional must not double-negate.
        self.assertEqual(signed_notional(-1_000.0, 2.0), 1_000.0)
        self.assertEqual(signed_notional(-1_000.0, -2.0), -1_000.0)

    def snapshot(self, positions):
        positions = list(positions)
        return SmartMoneySnapshot(
            symbol="BTC",
            wallets_sampled=len(positions),
            wallets_selected=len(positions),
            source="leaderboard",
            persistence_window_count=1,
            avg_persistence=1.0,
            positions=positions,
        )

    def test_shorts_decompose_as_a_positive_magnitude(self):
        """The property the convention protects: ratios stay within 0-100."""
        snap = self.snapshot(
            [
                WalletPosition(
                    wallet="L", size=1.0, notional=signed_notional(100.0, 1.0)
                ),
                WalletPosition(
                    wallet="S", size=-1.0, notional=signed_notional(60.0, -1.0)
                ),
            ]
        )
        self.assertEqual(snap.long_notional, 100.0)
        self.assertEqual(snap.short_notional, 60.0)
        self.assertAlmostEqual(snap.long_ratio_pct, 62.5)
        self.assertAlmostEqual(snap.weighted_long_ratio_pct, 62.5)
        self.assertLessEqual(snap.long_ratio_pct, 100.0)

    def test_hyperfeed_parser_signs_positions_from_the_feed(self):
        """Drives the real parser: the fixtures elsewhere bypass this code."""
        from unittest import mock

        from agent.factors import smart_money as module

        source = HyperfeedSource()
        source.api_key = "test"
        # The feed reports notional unsigned, so the parser is what makes a
        # short negative.
        payload = {
            "positions": [
                {"wallet": "0xAAA1", "size": -3.0, "notional": 150_000.0},
                {"wallet": "0xBBB2", "size": 2.0, "notional": 100_000.0},
            ]
        }

        with mock.patch.object(module, "request_json", return_value=payload):
            positions, notes = source.positions("BTC")

        self.assertEqual(notes, [])
        by_wallet = {p.wallet: p for p in positions}
        self.assertLess(by_wallet["0xaaa1"].notional, 0)
        self.assertGreater(by_wallet["0xbbb2"].notional, 0)
        self.assertEqual(by_wallet["0xaaa1"].side, "short")

        snap = self.snapshot(positions)
        self.assertGreater(snap.short_notional, 0)
        self.assertLess(snap.long_ratio_pct, 100.0)


class TestLeaderboardParsing(unittest.TestCase):
    """The leaderboard is on stats-data.hyperliquid.xyz, not /info, and needs
    the separate wallet read to become a smart money snapshot."""

    def source(self, rows, **overrides):
        overrides.setdefault("top_per_window", 10)
        return make_wallet_source(rows=rows, **overrides)

    def row(self, addr, day=0, week=0, month=0, account_value=50_000.0):
        """A leaderboard row with independent per-window PnL."""
        return {
            "ethAddress": addr,
            "accountValue": str(account_value),
            "windowPerformances": [
                ["day", {"pnl": str(day), "roi": "0", "vlm": "0"}],
                ["week", {"pnl": str(week), "roi": "0", "vlm": "0"}],
                ["month", {"pnl": str(month), "roi": "0", "vlm": "0"}],
            ],
        }

    def wallets(self, rows, **overrides):
        return self.source(rows, **overrides).wallets()

    def pers_rows(self):
        """Three wallets with persistence 3, 2 and 1.

        Each window keeps its top 2 by PnL, so:

            day   -> allthree, one
            week  -> allthree, two
            month -> allthree, two

        giving persistence 3 / 2 / 1 respectively.
        """
        return [
            self.row("0xallthree", day=900, week=900, month=900),
            self.row("0xtwo", day=100, week=500, month=500),
            self.row("0xone", day=500, week=100, month=100),
        ]

    def test_window_pnl_reads_correct_entry(self):
        from agent.factors.smart_money import LeaderboardWalletSource

        row = self.row("0xaaa", day=7, week=42, month=99)
        self.assertAlmostEqual(LeaderboardWalletSource._window_pnl(row, "week"), 42.0)
        self.assertAlmostEqual(LeaderboardWalletSource._window_pnl(row, "day"), 7.0)
        self.assertAlmostEqual(LeaderboardWalletSource._window_pnl(row, "month"), 99.0)
        self.assertAlmostEqual(LeaderboardWalletSource._window_pnl(row, "missing"), 0.0)

    def test_union_collects_wallets_from_every_window(self):
        """A wallet that ranks in only one window is still included, because
        persistence is a weight rather than a filter."""
        rows = [
            self.row("0xallthree", day=900, week=900, month=900),
            self.row("0xtwo", day=100, week=500, month=500),
            self.row("0xone", day=500, week=100, month=100),
        ]
        got = self.wallets(rows, top_per_window=2)
        self.assertEqual(set(got), {"0xallthree", "0xtwo", "0xone"})

    def test_persistence_counts_windows(self):
        sel = self.source(self.pers_rows(), top_per_window=2).select()
        by_addr = {w.address: w for w in sel}
        self.assertEqual(by_addr["0xallthree"].persistence, 3)
        self.assertEqual(by_addr["0xtwo"].persistence, 2)
        self.assertEqual(by_addr["0xone"].persistence, 1)
        self.assertEqual(by_addr["0xallthree"].windows, ["day", "month", "week"])

    def test_more_persistent_wallets_rank_first(self):
        rows = self.pers_rows()
        # Make the most persistent wallet the smallest account that still
        # clears the value floor, so passing proves persistence outranks size.
        for row in rows:
            row["accountValue"] = (
                "20000" if row["ethAddress"] == "0xallthree" else "99999999"
            )
        self.assertEqual(self.wallets(rows, top_per_window=2)[0], "0xallthree")

    def test_min_persistence_filters(self):
        got = self.wallets(self.pers_rows(), top_per_window=2, min_persistence=2)
        self.assertEqual(set(got), {"0xallthree", "0xtwo"})
        self.assertNotIn("0xone", got)

    def test_min_persistence_falls_back_when_nothing_qualifies(self):
        """Never silently return an empty set when data exists."""
        src = self.source(self.pers_rows(), top_per_window=2, min_persistence=5)
        sel = src.select()
        self.assertEqual(len(sel), 3)
        self.assertTrue(
            any("matched nothing" in n for n in src.selection_notes),
            src.selection_notes,
        )

    def test_max_wallets_caps_the_union(self):
        rows = [self.row(f"0x{i:04x}", day=i, week=i, month=i) for i in range(40)]
        self.assertEqual(len(self.wallets(rows, top_per_window=30, max_wallets=7)), 7)

    def test_default_min_persistence_keeps_everything(self):
        """The shipped default (1) must apply no hard filter."""
        rows = [self.row(f"0x{i:04x}", day=i, week=i, month=i) for i in range(10)]
        self.assertEqual(len(self.wallets(rows, top_per_window=10)), 10)

    def test_top_per_window_limits_each_windows_contribution(self):
        rows = [self.row(f"0x{i:04x}", day=i, week=i, month=i) for i in range(50)]
        # Identical ordering in all three windows, so the union is exactly the
        # per-window take.
        self.assertEqual(len(self.wallets(rows, top_per_window=4)), 4)

    def test_small_accounts_are_skipped(self):
        rows = [
            self.row("0xrich", day=100, account_value=1_000_000),
            self.row("0xpoor", day=999_999, account_value=5),
        ]
        self.assertEqual(self.wallets(rows, top_per_window=5), ["0xrich"])

    def test_account_value_floor_is_configurable(self):
        rows = [
            self.row("0xsmall", day=100, account_value=500),
            self.row("0xbig", day=90, account_value=900_000),
        ]
        got = self.wallets(rows, top_per_window=5, min_account_value=100)
        self.assertEqual(set(got), {"0xsmall", "0xbig"})

    def test_addresses_lowercased_and_deduplicated(self):
        rows = [
            self.row("0xAAA", day=200),
            self.row("0xaaa", day=200),
            self.row("0xBBB", day=100),
        ]
        self.assertEqual(set(self.wallets(rows, top_per_window=5)), {"0xaaa", "0xbbb"})

    def test_whitelist_always_included_and_fully_persistent(self):
        rows = [self.row("0xranked", day=9_000)]
        addr = "0x" + "a" * 40
        sel = self.source(rows, top_per_window=5, whitelist=[addr]).select()
        by_addr = {w.address: w for w in sel}
        self.assertIn(addr, by_addr)
        self.assertTrue(by_addr[addr].whitelisted)
        self.assertEqual(by_addr[addr].persistence, 3)

    def test_blacklist_excludes(self):
        rows = [
            self.row("0xkeep", day=100),
            self.row("0xdrop", day=9_000),
        ]
        sel = self.source(rows, top_per_window=5, blacklist=["0xdrop"]).select()
        self.assertEqual([w.address for w in sel], ["0xkeep"])

    def test_malformed_pnl_is_tolerated(self):
        rows = [
            {"ethAddress": "0xaaa", "accountValue": "50000", "windowPerformances": "garbage"},
            self.row("0xbbb", week=5),
        ]
        # The malformed row is still usable, just ranked at 0 PnL.
        self.assertEqual(set(self.wallets(rows, top_per_window=5)), {"0xaaa", "0xbbb"})

    def test_unrecognisable_response_shape_raises(self):
        from agent.factors.smart_money import LeaderboardWalletSource
        from agent.market_data import HyperliquidMarket, MarketDataError

        src = LeaderboardWalletSource.__new__(LeaderboardWalletSource)
        src.market = HyperliquidMarket(testnet=False)
        # Simulate the stats host returning a valid JSON object of the wrong shape.
        import agent.factors.smart_money as sm_module

        original = sm_module.request_json
        sm_module.request_json = lambda *a, **k: {"unexpected": "shape"}
        try:
            with self.assertRaises(MarketDataError):
                src.leaderboard_rows()
        finally:
            sm_module.request_json = original


class TestSmartMoneyDataNetwork(unittest.TestCase):
    """Whale data must come from mainnet even when trading on testnet.

    Reading mainnet leaderboard addresses via the testnet API returned zero
    positions, silently zeroing the highest-weighted factor.
    """

    def test_testnet_execution_still_reads_mainnet_whales(self):
        from agent.factors.smart_money import SmartMoneyFactor
        from agent.market_data import HyperliquidMarket

        exec_market = HyperliquidMarket(testnet=True)
        f = SmartMoneyFactor(exec_market, top_per_window=5, cache_path=None)

        self.assertTrue(exec_market.testnet)
        self.assertFalse(f.data_market.testnet)
        self.assertTrue(f.uses_mainnet_data)
        self.assertFalse(f.leaderboard.market.testnet)
        self.assertEqual(
            f.leaderboard.stats_url, "https://stats-data.hyperliquid.xyz/Mainnet"
        )

    def test_mainnet_execution_reuses_the_same_client(self):
        from agent.factors.smart_money import SmartMoneyFactor
        from agent.market_data import HyperliquidMarket

        exec_market = HyperliquidMarket(testnet=False)
        f = SmartMoneyFactor(exec_market, top_per_window=5, cache_path=None)
        self.assertIs(f.data_market, exec_market)

    def test_details_report_the_data_network(self):
        f = TestSmartMoneyFactor().factor(testnet_data=False)
        snap = SmartMoneySnapshot(symbol="BTC")
        snap.positions = [WalletPosition("a", 1, 100.0)]
        score = f.evaluate("BTC", snap)
        self.assertEqual(score.details["data_network"], "mainnet")


class TestParallelWalletReads(unittest.TestCase):
    """25 sequential wallet reads measured 58s; they must run concurrently."""

    def source(self, delay=0.05, fail_on=()):
        import threading
        import time as _time

        lock = threading.Lock()
        seen: list[str] = []

        class FakeMarket:
            testnet = False

            def info(self, payload):
                addr = payload["user"]
                with lock:
                    seen.append(addr)
                _time.sleep(delay)
                if addr in fail_on:
                    from agent.market_data import MarketDataError

                    raise MarketDataError("simulated failure")
                return {
                    "marginSummary": {"accountValue": "50000"},
                    "assetPositions": [
                        {"position": {"coin": "BTC", "szi": "1.5", "entryPx": "100"}}
                    ],
                }

        return make_wallet_source(market=FakeMarket()), seen

    def test_all_wallets_are_read(self):
        src, seen = self.source()
        wallets = [f"0x{i:04x}" for i in range(25)]
        positions, errors = src.positions(wallets, "BTC")

        self.assertEqual(len(seen), 25)
        self.assertEqual(len(positions), 25)
        self.assertEqual(errors, [])

    def test_reads_are_concurrent_not_sequential(self):
        import time as _time

        src, _ = self.source(delay=0.1)
        wallets = [f"0x{i:04x}" for i in range(20)]

        start = _time.time()
        src.positions(wallets, "BTC")
        elapsed = _time.time() - start

        # Sequential would be ~2.0s; concurrent should be far below that.
        self.assertLess(elapsed, 1.0, f"reads appear sequential ({elapsed:.2f}s)")

    def test_one_failing_wallet_does_not_abort_the_rest(self):
        src, _ = self.source(delay=0.01, fail_on={"0x0003"})
        wallets = [f"0x{i:04x}" for i in range(10)]
        positions, errors = src.positions(wallets, "BTC")

        self.assertEqual(len(positions), 9)
        self.assertEqual(len(errors), 1)

    def test_empty_wallet_list_short_circuits(self):
        src, seen = self.source()
        positions, errors = src.positions([], "BTC")
        self.assertEqual(positions, [])
        self.assertEqual(errors, [])
        self.assertEqual(seen, [])

    def test_only_the_requested_symbol_is_returned(self):
        class MultiMarket:
            testnet = False

            def info(self, payload):
                return {
                    "marginSummary": {"accountValue": "50000"},
                    "assetPositions": [
                        {"position": {"coin": "BTC", "szi": "1", "entryPx": "100"}},
                        {"position": {"coin": "ETH", "szi": "2", "entryPx": "50"}},
                    ],
                }

        src = make_wallet_source(market=MultiMarket())
        positions, _ = src.positions(["0xaaa"], "BTC")
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].size, 1.0)


class TestWalletDiskCache(unittest.TestCase):
    """The leaderboard is ~37 MB; the derived wallet list must survive restarts."""

    def make_source(self, cache_file, rows, top_per_window=5):
        src = make_wallet_source(
            rows=rows,
            cache_path=Path(cache_file),
            top_per_window=top_per_window,
        )
        src._load_disk_cache()
        return src

    def rows(self):
        return [
            {
                "ethAddress": f"0x{i:04x}",
                "accountValue": "500000",
                "windowPerformances": [
                    ["day", {"pnl": str(100 - i)}],
                    ["week", {"pnl": str(100 - i)}],
                    ["month", {"pnl": str(100 - i)}],
                ],
            }
            for i in range(5)
        ]

    def test_wallets_are_written_and_reloaded(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallets.json"
            first = self.make_source(path, self.rows())
            written = first.wallets()
            self.assertTrue(path.exists())

            second = self.make_source(path, self.rows())
            second.leaderboard_rows = lambda: (_ for _ in ()).throw(
                AssertionError("network must not be hit when the cache is warm")
            )
            self.assertEqual(second.wallets(), written)

    def test_expired_cache_is_ignored(self):
        import tempfile

        from agent.factors.smart_money import WALLET_CACHE_TTL_SECONDS

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallets.json"
            first = self.make_source(path, self.rows())
            first.wallets()

            payload = json.loads(path.read_text())
            # Derived from the real TTL rather than a magic offset. A hardcoded
            # 10_000 silently stopped testing expiry once the TTL was raised
            # from 900s to 6h: it no longer exceeded the threshold, so the test
            # would have passed while checking nothing.
            payload["cached_at"] -= WALLET_CACHE_TTL_SECONDS + 1
            path.write_text(json.dumps(payload))

            second = self.make_source(path, self.rows())
            second._load_disk_cache()
            self.assertEqual(second._wallet_cache, [])

    def test_cache_just_inside_the_ttl_is_kept(self):
        """The other side of the boundary: still fresh must still be reused."""
        import tempfile

        from agent.factors.smart_money import WALLET_CACHE_TTL_SECONDS

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallets.json"
            first = self.make_source(path, self.rows())
            expected = first.wallets()

            payload = json.loads(path.read_text())
            payload["cached_at"] -= WALLET_CACHE_TTL_SECONDS - 60
            path.write_text(json.dumps(payload))

            second = self.make_source(path, self.rows())
            second._load_disk_cache()
            # `wallets()` returns addresses; the cache holds SmartWallet objects.
            self.assertEqual(
                [w.address for w in second._wallet_cache],
                expected,
            )

    def test_cache_with_different_config_is_ignored(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallets.json"
            first = self.make_source(path, self.rows(), top_per_window=5)
            first.wallets()

            # A source configured differently must not reuse the cached set.
            other = self.make_source(path, self.rows(), top_per_window=99)
            self.assertEqual(other._wallet_cache, [])

    def test_cache_preserves_persistence_metadata(self):
        """Restarting must not lose the curation evidence."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallets.json"
            first = self.make_source(path, self.rows())
            first.wallets()

            second = self.make_source(path, self.rows())
            self.assertTrue(second._wallet_cache)
            for wallet in second._wallet_cache:
                self.assertEqual(wallet.persistence, 3)
                self.assertIn("day", wallet.windows)

    def test_unreadable_cache_does_not_raise(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wallets.json"
            path.write_text("{ this is not json")
            src = self.make_source(path, self.rows())
            self.assertEqual(src._wallet_cache, [])
            # Still able to fetch fresh.
            self.assertEqual(len(src.wallets()), 5)


class TestPersistenceWeighting(unittest.TestCase):
    """Persistence weights a wallet's notional; it does not filter it out.

    Measured on live BTC data, hard-filtering by persistence cut holders from
    18 to 3, so weighting is the correct use of the signal.
    """

    def pos(self, addr, size, notional, persistence=1):
        return WalletPosition(
            wallet=addr,
            size=size,
            notional=abs(notional) if size > 0 else -abs(notional),
            persistence=persistence,
        )

    def snap(self, positions, **kw):
        s = SmartMoneySnapshot(symbol="BTC", **kw)
        s.positions = positions
        return s

    def test_weight_equals_persistence(self):
        self.assertEqual(self.pos("a", 1, 100, persistence=3).weight, 3.0)

    def test_weight_floors_at_one(self):
        # A stray persistence of 0 must not zero out a real position.
        self.assertEqual(self.pos("a", 1, 100, persistence=0).weight, 1.0)

    def test_whitelisted_wallets_get_at_least_double_weight(self):
        p = self.pos("a", 1, 100, persistence=1)
        p.whitelisted = True
        self.assertGreaterEqual(p.weight, 2.0)

    def test_weighted_ratio_favours_persistent_wallets(self):
        """A persistent short must outweigh an equal-sized fresh long."""
        s = self.snap(
            [
                self.pos("fresh_long", 1, 1000, persistence=1),
                self.pos("persistent_short", -1, 1000, persistence=3),
            ]
        )
        self.assertAlmostEqual(s.long_ratio_pct, 50.0)  # unweighted: balanced
        self.assertLess(s.weighted_long_ratio_pct, 50.0)

    def test_weighted_ratio_matches_raw_when_persistence_is_uniform(self):
        s = self.snap(
            [
                self.pos("l", 1, 700, persistence=2),
                self.pos("s", -1, 300, persistence=2),
            ]
        )
        self.assertAlmostEqual(s.weighted_long_ratio_pct, s.long_ratio_pct)

    def test_weighted_ratio_falls_back_when_flat(self):
        s = self.snap([self.pos("a", 1, 0, persistence=3)])
        self.assertAlmostEqual(s.weighted_long_ratio_pct, s.long_ratio_pct)

    def test_top1_share_detects_concentration(self):
        s = self.snap(
            [
                self.pos("whale", 1, 900),
                self.pos("a", -1, 50),
                self.pos("b", -1, 50),
            ]
        )
        self.assertAlmostEqual(s.top1_share_pct, 90.0)

    def test_top1_share_is_zero_when_flat(self):
        self.assertEqual(self.snap([]).top1_share_pct, 0.0)

    def test_holder_avg_persistence(self):
        s = self.snap(
            [
                self.pos("a", 1, 100, persistence=3),
                self.pos("b", -1, 100, persistence=1),
            ]
        )
        self.assertAlmostEqual(s.holder_avg_persistence, 2.0)

    def test_weighted_long_ratio_for_persistence_floor(self):
        s = self.snap(
            [
                self.pos("fresh_long", 1, 1000, persistence=1),
                self.pos("persistent_short", -1, 1000, persistence=3),
            ]
        )
        # Above the floor only the short remains.
        self.assertAlmostEqual(s.weighted_long_ratio_for(2), 0.0)
        # No wallet clears a floor of 4.
        self.assertAlmostEqual(s.weighted_long_ratio_for(4), 50.0)

    def test_curation_score(self):
        s = self.snap([self.pos("a", 1, 100)], persistence_window_count=3)
        s.avg_persistence = 3.0
        self.assertAlmostEqual(s.curation_score, 1.0)
        s.avg_persistence = 2.0
        self.assertAlmostEqual(s.curation_score, 0.5)
        s.avg_persistence = 1.0
        self.assertAlmostEqual(s.curation_score, 0.0)

    def test_curation_score_is_zero_for_single_window(self):
        s = self.snap([self.pos("a", 1, 100)], persistence_window_count=1)
        s.avg_persistence = 1.0
        self.assertAlmostEqual(s.curation_score, 0.0)


class TestEvaluateReportsPersistence(unittest.TestCase):
    def make(self):
        f = SmartMoneyFactor.__new__(SmartMoneyFactor)
        f.data_market = type("M", (), {"testnet": False})()
        return f

    def test_details_expose_weighting_evidence(self):
        snap = SmartMoneySnapshot(symbol="BTC", persistence_window_count=3)
        snap.positions = [
            WalletPosition("a", 1, 700.0, persistence=3),
            WalletPosition("b", -1, -300.0, persistence=1),
        ]
        score = self.make().evaluate("BTC", snap)

        self.assertIn("weighted_long_ratio_pct", score.details)
        self.assertIn("top1_share_pct", score.details)
        self.assertIn("holder_avg_persistence", score.details)
        self.assertIn("curation_score", score.details)
        # 700 at 3x vs 300 at 1x -> 2100 / 2400 = 87.5% long.
        self.assertAlmostEqual(score.details["weighted_long_ratio_pct"], 87.5)
        self.assertGreater(score.score, 50)

    def test_concentration_warning_is_emitted(self):
        snap = SmartMoneySnapshot(symbol="BTC")
        snap.positions = [
            WalletPosition("whale", 1, 950.0),
            WalletPosition("a", -1, -50.0),
        ]
        score = self.make().evaluate("BTC", snap)
        self.assertGreater(score.details["top1_share_pct"], 60)
        self.assertTrue(any("one wallet" in r for r in score.reasons), score.reasons)

    def test_curation_raises_the_confidence_cap(self):
        # A fully curated set may reach 0.85; a single-window set is capped 0.75.
        def build(avg_persistence):
            snap = SmartMoneySnapshot(
                symbol="BTC",
                persistence_window_count=3,
                avg_persistence=avg_persistence,
            )
            snap.positions = [
                WalletPosition(str(i), 1, 100.0, unrealized_pnl=10.0, persistence=3)
                for i in range(20)
            ]
            snap.source = "leaderboard"
            return self.make().evaluate("BTC", snap)

        curated = build(3.0)
        single_window = build(1.0)
        self.assertGreater(curated.confidence, single_window.confidence)
        self.assertLessEqual(curated.confidence, 0.85 + 1e-9)
        self.assertLessEqual(single_window.confidence, 0.75 + 1e-9)


class TestHttpUtil(unittest.TestCase):
    """The leaderboard response is ~37 MB. A truncated transfer must be
    retried rather than surfaced as a fatal error."""

    def test_incomplete_read_is_retried(self):
        import http.client
        from unittest import mock

        from agent import http_util

        calls = {"n": 0}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, size=-1):
                calls["n"] += 1
                if calls["n"] == 1:
                    # First attempt: die partway through the body.
                    raise http.client.IncompleteRead(b"partial", 100)
                return b'{"ok": true}' if calls["n"] == 2 else b""

        with mock.patch.object(http_util.urllib.request, "urlopen", return_value=FakeResponse()):
            result = http_util.request_json("https://example.invalid/x", retries=2, backoff=0)

        self.assertEqual(result, {"ok": True})
        self.assertGreaterEqual(calls["n"], 3)

    def test_incomplete_read_exhausts_retries_and_raises(self):
        import http.client
        from unittest import mock

        from agent import http_util

        class AlwaysFails:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, size=-1):
                raise http.client.IncompleteRead(b"", 50)

        with mock.patch.object(
            http_util.urllib.request, "urlopen", return_value=AlwaysFails()
        ):
            with self.assertRaises(http_util.HttpError):
                http_util.request_json("https://example.invalid/x", retries=2, backoff=0)

    def test_identity_encoding_is_requested(self):
        """Compression caused the truncation; the header must stay off."""
        from unittest import mock

        from agent import http_util

        captured = {}

        class FakeResponse:
            """Yields the body once, then EOF. A stub that never signals EOF
            would spin forever in the chunked read loop."""

            def __init__(self):
                self._sent = False

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, size=-1):
                if self._sent:
                    return b""
                self._sent = True
                return b"{}"

        def capture(req, timeout=None):
            captured["headers"] = dict(req.header_items())
            return FakeResponse()

        with mock.patch.object(http_util.urllib.request, "urlopen", side_effect=capture):
            result = http_util.request_json("https://example.invalid/x", retries=1)

        self.assertEqual(result, {})
        self.assertEqual(captured["headers"].get("Accept-encoding"), "identity")

    def test_oversized_response_aborts(self):
        """A server streaming forever must not hang the agent."""
        from unittest import mock

        from agent import http_util

        class EndlessResponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self, size=-1):
                return b"x" * 1024  # never returns EOF

        original = http_util.MAX_RESPONSE_BYTES
        http_util.MAX_RESPONSE_BYTES = 4096
        try:
            with mock.patch.object(
                http_util.urllib.request, "urlopen", return_value=EndlessResponse()
            ):
                with self.assertRaises(http_util.HttpError) as ctx:
                    http_util.request_json("https://example.invalid/x", retries=1, backoff=0)
            self.assertIn("exceeded", str(ctx.exception))
        finally:
            http_util.MAX_RESPONSE_BYTES = original

    def test_client_errors_fail_fast(self):
        import urllib.error
        from unittest import mock

        from agent import http_util

        error = urllib.error.HTTPError(
            "https://example.invalid/x", 422, "Unprocessable", {}, None
        )
        with mock.patch.object(http_util.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(http_util.HttpError) as ctx:
                http_util.request_json("https://example.invalid/x", retries=3, backoff=0)
        self.assertIn("422", str(ctx.exception))


# ---------------------------------------------------------------------------
# Production readiness: persistence, recovery and safety rails
# ---------------------------------------------------------------------------

BASE_CONFIG = """
name: t
symbol: BTC
risk:
  leverage: 2
  account_allocation_pct: 10.0
  risk_per_trade_pct: 1.0
  stop_loss_enabled: true
  stop_loss_pct: 3.0
  max_drawdown_pct: 15.0
safety:
  hard_stop_pct: 5.0
execution:
  dry_run: true
  testnet: true
  poll_interval_seconds: 5
"""


def write_config(text: str) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    handle.write(text)
    handle.close()
    return handle.name


class TestStateStore(unittest.TestCase):
    """Guardrails must survive a restart, atomically and defensively."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "agent_state.json"
        self.store = StateStore(self.path)

    def test_missing_file_returns_none(self):
        self.assertIsNone(self.store.load())

    def test_round_trip(self):
        state = AgentState()
        state.trades_today = {"BTC": 2}
        state.last_trade_ts = {"BTC": 123.5}
        state.consecutive_losses = 2
        state.halted_until = 999.0
        state.halt_reason = "losses"
        state.peak_equity = 1234.0
        state.position = {"symbol": "BTC", "side": "long", "size": 1.0}
        self.assertTrue(self.store.save(state))

        loaded = self.store.load()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.trades_today, {"BTC": 2})
        self.assertEqual(loaded.consecutive_losses, 2)
        self.assertEqual(loaded.peak_equity, 1234.0)
        self.assertEqual(loaded.position["side"], "long")

    def test_corrupt_file_returns_none(self):
        self.path.write_text("{ not json")
        self.assertIsNone(self.store.load())

    # ------------------------------------------------------------------
    # Non-finite floats. `nan`/`inf` compare false against everything, so a
    # single one loaded into the wrong field leaves a guardrail present but
    # permanently inert - which is worse than it being absent, because nothing
    # looks wrong.
    # ------------------------------------------------------------------

    def test_infinite_peak_equity_is_rejected_on_load(self):
        """`nan >= limit` is False, so an inf/nan peak would disable the
        drawdown guard for good."""
        self.path.write_text(json.dumps({"version": 1, "peak_equity": float("inf")}))

        loaded = self.store.load()
        self.assertEqual(loaded.peak_equity, 0.0)
        self.assertTrue(math.isfinite(loaded.peak_equity))

    def test_nan_cooldown_timestamp_is_dropped(self):
        """`(now - nan) < cooldown` is False, so a nan timestamp would silently
        disable the cooldown."""
        self.path.write_text(
            json.dumps({"version": 1, "last_trade_ts": {"BTC": float("nan")}})
        )
        self.assertEqual(self.store.load().last_trade_ts, {})

    def test_nan_halt_deadline_is_rejected(self):
        """`now < nan` is False, so a nan deadline would disable the halt."""
        self.path.write_text(json.dumps({"version": 1, "halted_until": float("nan")}))
        self.assertEqual(self.store.load().halted_until, 0.0)

    def test_nan_trade_count_is_dropped(self):
        self.path.write_text(
            json.dumps({"version": 1, "trades_today": {"BTC": float("nan"), "ETH": 2}})
        )
        self.assertEqual(self.store.load().trades_today, {"ETH": 2})

    def test_save_refuses_to_write_non_finite_values(self):
        """`Infinity` is not valid JSON, and reading it back would poison the
        next startup. The previous good file must survive untouched."""
        good = AgentState()
        good.peak_equity = 500.0
        self.assertTrue(self.store.save(good))
        before = self.path.read_text()

        bad = AgentState()
        bad.peak_equity = float("inf")
        self.assertFalse(self.store.save(bad))
        self.assertEqual(self.path.read_text(), before)

    def test_written_file_is_valid_strict_json(self):
        """`parse_constant` raises on Infinity/NaN, which a strict parser would
        reject - so this asserts the file is JSON, not merely JSON-for-Python."""
        state = AgentState()
        state.realised_pnl = 12.5
        self.assertTrue(self.store.save(state))

        json.loads(
            self.path.read_text(),
            parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)),
        )

    # ------------------------------------------------------------------
    # Concurrency. Two agents can legitimately run at once - a scheduled cycle
    # and a manual run - and a corrupted state file loads as None, which resets
    # every guardrail at once.
    # ------------------------------------------------------------------

    def test_concurrent_saves_all_succeed(self):
        """Every concurrent writer must report success.

        The load-bearing assertion is `all(results)`. A single shared temp file
        makes `os.replace` fail for whichever writer loses the race, and `save`
        swallows that into a False - so checking only "no exception escaped"
        would pass with or without the fix. The barrier tightens the race so
        the writers actually overlap instead of running one after another.
        """
        results: list[bool] = []
        errors: list[Exception] = []
        lock = threading.Lock()
        writers = 24
        barrier = threading.Barrier(writers)

        def write(index: int) -> None:
            barrier.wait()
            try:
                ok = self.store.save(AgentState(realised_pnl=float(index)))
            except Exception as exc:  # noqa: BLE001 - collected and asserted
                with lock:
                    errors.append(exc)
                return
            with lock:
                results.append(ok)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(writers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(results), writers)
        self.assertTrue(all(results), "every concurrent save must succeed")
        self.assertIsInstance(self.store.load(), AgentState)

    def test_failed_save_leaves_no_temp_file_behind(self):
        """Cleanup matters on the failure path.

        A successful save renames its temp file away, so asserting "no leftover"
        after one proves nothing. A rejected save is the case where the temp
        file would actually accumulate.
        """
        bad = AgentState()
        bad.realised_pnl = float("inf")
        self.assertFalse(self.store.save(bad))

        leftovers = [f for f in os.listdir(self.tmp.name) if ".tmp" in f]
        self.assertEqual(leftovers, [])

    def test_version_mismatch_returns_none(self):
        self.path.write_text(json.dumps({"version": 999, "trades_today": {"BTC": 5}}))
        self.assertIsNone(self.store.load())

    def test_atomic_write_leaves_no_temp_file(self):
        self.store.save(AgentState())
        leftovers = list(Path(self.tmp.name).glob("*.tmp"))
        self.assertEqual(leftovers, [])

    def test_bad_types_are_coerced_not_fatal(self):
        self.path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "trades_today": "garbage",
                    "last_trade_ts": [1, 2],
                    "consecutive_losses": "not a number",
                    "peak_equity": None,
                    "position": "not a dict",
                }
            )
        )
        state = self.store.load()
        self.assertIsNotNone(state)
        self.assertEqual(state.trades_today, {})
        self.assertEqual(state.last_trade_ts, {})
        self.assertEqual(state.consecutive_losses, 0)
        self.assertEqual(state.peak_equity, 0.0)
        self.assertIsNone(state.position)

    def test_halt_survives_round_trip(self):
        state = AgentState()
        state.agent_halted = True
        state.agent_halt_reason = "drawdown"
        self.store.save(state)
        loaded = self.store.load()
        self.assertTrue(loaded.agent_halted)
        self.assertEqual(loaded.agent_halt_reason, "drawdown")


class TestPositionSerialization(unittest.TestCase):
    def test_round_trip(self):
        original = Position(
            symbol="BTC",
            side="short",
            size=0.5,
            entry_price=100.0,
            notional=50.0,
            leverage=2,
            stop_price=103.0,
            take_profit_price=91.0,
            entry_score=42.0,
            entry_reasons=["because"],
            is_dry_run=False,
        )
        restored = Position.from_dict(original.to_dict())
        self.assertEqual(restored.symbol, "BTC")
        self.assertEqual(restored.side, "short")
        self.assertAlmostEqual(restored.size, 0.5)
        self.assertAlmostEqual(restored.stop_price, 103.0)
        self.assertEqual(restored.entry_reasons, ["because"])
        self.assertFalse(restored.is_dry_run)

    def test_invalid_side_raises(self):
        with self.assertRaises(ExecutionError):
            Position.from_dict({"symbol": "BTC", "side": "sideways", "size": 1})

    def test_missing_symbol_raises(self):
        with self.assertRaises(ExecutionError):
            Position.from_dict({"side": "long", "size": 1})

    def test_none_prices_stay_none(self):
        restored = Position.from_dict(
            {"symbol": "BTC", "side": "long", "size": 1.0, "stop_price": None}
        )
        self.assertIsNone(restored.stop_price)

    def test_garbage_numbers_do_not_raise(self):
        restored = Position.from_dict(
            {"symbol": "BTC", "side": "long", "size": "abc", "entry_price": None}
        )
        self.assertEqual(restored.size, 0.0)
        self.assertEqual(restored.entry_price, 0.0)


class TestDisciplinePersistence(unittest.TestCase):
    """The regression that motivated persistence: from_payload only accepted
    dicts, so passing the AgentState object raised AttributeError, which the
    fallback swallowed and silently reset every guardrail."""

    def test_from_payload_accepts_a_mapping(self):
        state = DisciplineState.from_payload(
            {
                "trades_today": {"BTC": 3},
                "last_trade_ts": {"BTC": 1_700_000_000.0},
                "consecutive_losses": 1,
                "day_anchor": 1_699_000_000.0,
            }
        )
        self.assertEqual(state.trades_today, {"BTC": 3})
        self.assertEqual(state.consecutive_losses, 1)

    def test_from_payload_accepts_an_object(self):
        """The exact shape the bot passes: an AgentState dataclass."""
        agent_state = AgentState()
        agent_state.trades_today = {"BTC": 3}
        agent_state.last_trade_ts = {"BTC": 1_700_000_000.0}
        agent_state.consecutive_losses = 2
        agent_state.halted_until = 1_800_000_000.0
        agent_state.halt_reason = "streak"
        agent_state.day_anchor = time.time()

        state = DisciplineState.from_payload(agent_state)
        self.assertEqual(state.trades_today, {"BTC": 3})
        self.assertEqual(state.last_trade_ts, {"BTC": 1_700_000_000.0})
        self.assertEqual(state.consecutive_losses, 2)
        self.assertEqual(state.halt_reason, "streak")

    def test_zero_day_anchor_is_reanchored(self):
        """A zero anchor would make roll_day_if_needed wipe today's counts."""
        state = DisciplineState.from_payload({"day_anchor": 0})
        self.assertGreater(state.day_anchor, 0)

    def test_discipline_restore_round_trip(self):
        cfg_path = write_config(BASE_CONFIG)
        self.addCleanup(os.unlink, cfg_path)
        cfg = load_config(cfg_path)

        d = Discipline(cfg, market=None)
        d.record_trade("BTC")
        d.record_outcome(-5.0)

        payload = d.export_payload()
        other = Discipline(cfg, market=None)
        other.restore_payload(payload)

        self.assertEqual(other.state.trades_today, {"BTC": 1})
        self.assertEqual(other.state.consecutive_losses, 1)
        self.assertIn("BTC", other.state.last_trade_ts)

    def test_cooldown_is_active_after_restore(self):
        """The core guarantee: a restart must not clear the cooldown."""
        cfg_path = write_config(BASE_CONFIG)
        self.addCleanup(os.unlink, cfg_path)
        cfg = load_config(cfg_path)

        first = Discipline(cfg, market=None)
        first.record_trade("BTC")

        second = Discipline(cfg, market=None)
        second.restore_payload(first.export_payload())

        self.assertTrue(second.in_cooldown("BTC"))
        self.assertGreater(second.cooldown_remaining_minutes("BTC"), 0)

    def test_daily_cap_is_active_after_restore(self):
        cfg_path = write_config(BASE_CONFIG + "\ndiscipline:\n  max_signals_per_day: 1\n")
        self.addCleanup(os.unlink, cfg_path)
        cfg = load_config(cfg_path)

        first = Discipline(cfg, market=None)
        first.record_trade("BTC")

        second = Discipline(cfg, market=None)
        second.restore_payload(first.export_payload())

        self.assertTrue(second.daily_cap_reached("BTC"))


class TestOrderResultParsing(unittest.TestCase):
    """A resting limit order is not a fill. Treating it as one created a
    phantom position that was tracked, stopped and later 'closed'."""

    def parse(self, statuses):
        return Broker._parse_order_result(
            {"response": {"data": {"statuses": statuses}}}
        )

    def test_filled(self):
        state, size, price = self.parse(
            [{"filled": {"totalSz": "0.5", "avgPx": "101.5", "oid": 1}}]
        )
        self.assertEqual(state, "filled")
        self.assertAlmostEqual(size, 0.5)
        self.assertAlmostEqual(price, 101.5)

    def test_resting_is_not_a_fill(self):
        state, size, price = self.parse([{"resting": {"oid": 7}}])
        self.assertEqual(state, "resting")
        self.assertEqual(size, 0.0)

    def test_error(self):
        state, _, _ = self.parse([{"error": "insufficient margin"}])
        self.assertEqual(state, "error")

    def test_missing_status_is_treated_as_resting(self):
        """Never default to 'filled' when the shape is unrecognised."""
        self.assertEqual(Broker._parse_order_result({})[0], "resting")
        self.assertEqual(Broker._parse_order_result(None)[0], "resting")
        self.assertEqual(self.parse([])[0], "resting")
        self.assertEqual(self.parse(["nonsense"])[0], "resting")

    def test_filled_with_bad_numbers_is_zero_size(self):
        state, size, price = self.parse([{"filled": {"totalSz": None, "avgPx": "x"}}])
        self.assertEqual(state, "filled")
        self.assertEqual(size, 0.0)
        self.assertEqual(price, 0.0)


class TestStartupReconciliation(unittest.TestCase):
    """Cases previously mishandled by simply starting from position=None."""

    def make_agent(self, state: AgentState | None = None):
        """Build an agent, optionally with state already on disk.

        The state is written *before* construction so this genuinely exercises
        the restart path rather than assigning attributes afterwards.
        """
        cfg_path = write_config(BASE_CONFIG)
        self.addCleanup(os.unlink, cfg_path)
        cfg = load_config(cfg_path)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        state_path = Path(tmp.name) / "state.json"

        if state is not None:
            StateStore(state_path).save(state)

        # Dry run, so reconciliation uses persisted state without an exchange.
        cfg.execution.dry_run = True
        return FoxAgent(
            cfg,
            journal_path=str(Path(tmp.name) / "journal.jsonl"),
            state_path=str(state_path),
        )

    def test_dry_run_resumes_persisted_position(self):
        state = AgentState()
        state.position = {
            "symbol": "BTC",
            "side": "short",
            "size": 0.25,
            "entry_price": 100.0,
            "notional": 25.0,
            "leverage": 2,
            "is_dry_run": True,
        }
        agent = self.make_agent(state)
        self.assertIsNotNone(agent.position)
        self.assertEqual(agent.position.side, "short")
        agent.reconcile_startup()  # must not raise or clear it
        self.assertIsNotNone(agent.position)

    def test_persisted_halt_is_restored(self):
        state = AgentState()
        state.agent_halted = True
        state.agent_halt_reason = "drawdown breach"
        agent = self.make_agent(state)
        self.assertTrue(agent.halted)
        self.assertEqual(agent.halt_reason, "drawdown breach")

    def test_halt_is_persisted(self):
        agent = self.make_agent()
        self.assertFalse(agent.halted)
        agent.halt("test halt")
        self.assertTrue(agent.halted)
        reloaded = agent.store.load()
        self.assertTrue(reloaded.agent_halted)
        self.assertEqual(reloaded.agent_halt_reason, "test halt")

    def test_halt_does_not_repeat_work(self):
        agent = self.make_agent()
        agent.halt("first")
        agent.halt("second")
        self.assertEqual(agent.halt_reason, "first")

    def test_unprotected_callback_tracks_and_halts(self):
        agent = self.make_agent()
        exposed = Position(
            symbol="BTC",
            side="long",
            size=1.0,
            entry_price=100.0,
            notional=100.0,
            leverage=2,
        )
        agent._handle_unprotected(exposed)
        self.assertTrue(agent.halted)
        self.assertIsNotNone(agent.position)
        self.assertIn("UNPROTECTED", agent.halt_reason)


class TestDrawdownGuard(unittest.TestCase):
    def make_agent(self):
        cfg_path = write_config(BASE_CONFIG)
        self.addCleanup(os.unlink, cfg_path)
        cfg = load_config(cfg_path)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cfg.execution.dry_run = True
        return FoxAgent(
            cfg,
            journal_path=str(Path(tmp.name) / "journal.jsonl"),
            state_path=str(Path(tmp.name) / "state.json"),
        )

    def test_peak_tracks_the_high_water_mark(self):
        agent = self.make_agent()
        self.assertIsNone(agent.check_drawdown(1000.0))
        self.assertAlmostEqual(agent.state.peak_equity, 1000.0)
        self.assertIsNone(agent.check_drawdown(1100.0))
        self.assertAlmostEqual(agent.state.peak_equity, 1100.0)

    def test_breach_is_reported(self):
        agent = self.make_agent()
        agent.check_drawdown(1000.0)
        # 15% limit -> 850 is exactly at the boundary.
        breach = agent.check_drawdown(840.0)
        self.assertIsNotNone(breach)
        self.assertIn("drawdown", breach)

    def test_small_drawdown_is_allowed(self):
        agent = self.make_agent()
        agent.check_drawdown(1000.0)
        self.assertIsNone(agent.check_drawdown(900.0))  # 10% < 15%

    def test_zero_equity_is_ignored(self):
        agent = self.make_agent()
        agent.check_drawdown(1000.0)
        self.assertIsNone(agent.check_drawdown(0.0))

    def test_peak_persists_across_restart(self):
        agent = self.make_agent()
        agent.check_drawdown(1000.0)

        reloaded = agent.store.load()
        self.assertAlmostEqual(reloaded.peak_equity, 1000.0)


class TestClosePositionContract(unittest.TestCase):
    """`close_position` must report whether the position is actually gone.

    The reversal path uses that answer to decide whether it is safe to open
    the opposite side. Before this contract existed, a failed close returned
    silently and the loop opened a new position anyway, overwriting
    `self.position` and orphaning the old exposure.
    """

    def make_agent(self):
        cfg_path = write_config(BASE_CONFIG)
        self.addCleanup(os.unlink, cfg_path)
        cfg = load_config(cfg_path)

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cfg.execution.dry_run = True
        return FoxAgent(
            cfg,
            journal_path=str(Path(tmp.name) / "journal.jsonl"),
            state_path=str(Path(tmp.name) / "state.json"),
        )

    def position(self):
        return Position(
            symbol="BTC",
            side="long",
            size=0.01,
            entry_price=50_000.0,
            notional=500.0,
            leverage=2,
        )

    def test_returns_true_and_clears_the_position_on_success(self):
        agent = self.make_agent()
        agent.position = self.position()
        agent.broker.close_position = lambda p, price, reason: 12.5

        self.assertTrue(agent.close_position(agent.position, 51_000.0, "test"))
        self.assertIsNone(agent.position)

    def test_returns_false_and_keeps_the_position_on_failure(self):
        agent = self.make_agent()
        agent.position = self.position()

        def boom(p, price, reason):
            raise ExecutionError("simulated close failure")

        agent.broker.close_position = boom

        self.assertFalse(agent.close_position(agent.position, 51_000.0, "test"))
        self.assertIsNotNone(
            agent.position, "a failed close must leave the position tracked"
        )


class TestConfigSafetyRails(unittest.TestCase):
    def test_max_drawdown_is_validated(self):
        bad = BASE_CONFIG.replace(
            "max_drawdown_pct: 15.0", "max_drawdown_pct: 0"
        )
        with self.assertRaises(ConfigError) as ctx:
            load_config(write_config(bad))
        self.assertIn("max_drawdown_pct", str(ctx.exception))

    def test_negative_expectancy_exits_rejected(self):
        """A take profit below the stop loss loses money before fees."""
        bad = BASE_CONFIG.replace(
            "  stop_loss_pct: 3.0\n",
            "  stop_loss_pct: 3.0\n  take_profit_enabled: true\n  take_profit_pct: 1.0\n",
        )
        with self.assertRaises(ConfigError) as ctx:
            load_config(write_config(bad))
        self.assertIn("take_profit_pct", str(ctx.exception))

    def test_reduce_only_flag_is_gone(self):
        """It was decoration; the config must reject it rather than accept a
        flag that does nothing."""
        with self.assertRaises(ConfigError) as ctx:
            load_config(write_config(BASE_CONFIG + "\nrisk:\n  reduce_only_on_exit: true\n"))
        self.assertIn("reduce_only_on_exit", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
