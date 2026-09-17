"""Execution-layer tests, focused on the live-trading order paths.

These exist because the resting-order path had ZERO coverage, which let a
critical defect survive a 204-test suite: `self._last_result` was initialised
but never assigned, so `_last_order_id()` always returned None, so every
resting limit order was abandoned on the book while the agent journalled that
it had been cancelled. An orphan order can fill later and create exposure that
nothing is tracking.

The tests below pin the behaviour that matters: an order is only reported as
cancelled when the venue's own state says so.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.config import load_config  # noqa: E402
from agent.execution import (  # noqa: E402
    Broker,
    CriticalExecutionError,
    ExecutionError,
    Position,
    RestingOrderOutcome,
    compute_size,
)
from agent.synthesizer import Journal  # noqa: E402

CONFIG = """
name: t
symbol: BTC
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
  order_type: limit
  limit_offset_bps: 5.0
"""


class FakeExchange:
    """Scripted stand-in for the Hyperliquid Exchange client."""

    def __init__(self, order_result, cancel_raises=False):
        self.order_result = order_result
        self.cancel_raises = cancel_raises
        self.orders: list[tuple] = []
        self.cancels: list[tuple] = []
        self.close_calls = 0

    def order(self, symbol, is_buy, size, px, order_type, reduce_only=False):
        self.orders.append((symbol, is_buy, size, px, order_type, reduce_only))
        return self.order_result

    def market_open(self, symbol, is_buy, size, px, slippage):
        self.orders.append((symbol, is_buy, size, px, "market"))
        return self.order_result

    def market_close(self, symbol):
        self.close_calls += 1
        return {"status": "ok"}

    def cancel(self, name, oid):
        self.cancels.append((name, oid))
        if self.cancel_raises:
            raise RuntimeError("cancel rejected by venue")
        return {"status": "ok"}

    def update_leverage(self, *args, **kwargs):
        return None


def resting(oid):
    return {"response": {"data": {"statuses": [{"resting": {"oid": oid}}]}}}


def filled(size, px, oid=1):
    return {
        "response": {
            "data": {
                "statuses": [{"filled": {"totalSz": str(size), "avgPx": str(px), "oid": oid}}]
            }
        }
    }


def order_status(state, filled_size=0.0, avg_px=0.0):
    return {
        "status": "order",
        "order": {"status": state, "filledSize": str(filled_size), "avgPx": str(avg_px)},
    }


class ExecutionTestCase(unittest.TestCase):
    """Shared fixtures for building an offline Broker."""

    def setUp(self):
        self.cfg_path = self._write(CONFIG)
        self.addCleanup(lambda: Path(self.cfg_path).unlink(missing_ok=True))
        self.cfg = load_config(self.cfg_path)

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.journal = Journal(str(Path(self.tmp.name) / "journal.jsonl"))

    def _write(self, text: str) -> str:
        handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
        handle.write(text)
        handle.close()
        return handle.name

    def broker(self, exchange=None, on_unprotected=None):
        broker = Broker(self.cfg, None, self.journal, on_unprotected=on_unprotected)
        # The config validator refuses dry_run=False without a key, so the
        # live code path is enabled here instead of in the YAML.
        broker.dry_run = False
        broker._exchange = exchange
        # Keep the wait short so tests do not sleep for the production timeout.
        broker.RESTING_FILL_TIMEOUT = 0.05
        broker.RESTING_POLL_INTERVAL = 0.005
        return broker

    def events(self) -> list[dict]:
        import json

        path = Path(self.tmp.name) / "journal.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]


# ---------------------------------------------------------------------------
# The regression that motivated this file
# ---------------------------------------------------------------------------


class TestLastResultIsRecorded(ExecutionTestCase):
    """`open_position` MUST stash the raw response, or the oid is unrecoverable."""

    def test_open_position_records_the_raw_result(self):
        exchange = FakeExchange(filled(0.001, 50_000.0))
        broker = self.broker(exchange)

        broker.open_position("BTC", "long", 0.001, 50_000.0, leverage=2)

        self.assertIsNotNone(
            broker._last_result,
            "the raw order response must be retained for resting-order recovery",
        )

    def test_last_order_id_reads_the_resting_oid(self):
        broker = self.broker()
        broker._last_result = resting(4242)
        self.assertEqual(broker._last_order_id(), 4242)

    def test_last_order_id_reads_the_filled_oid(self):
        broker = self.broker()
        broker._last_result = filled(1.0, 100.0, oid=99)
        self.assertEqual(broker._last_order_id(), 99)

    def test_last_order_id_is_none_without_a_result(self):
        self.assertIsNone(self.broker()._last_order_id())


# ---------------------------------------------------------------------------
# Resting-order resolution
# ---------------------------------------------------------------------------


class TestResolveRestingOrder(ExecutionTestCase):
    def test_filled_during_the_wait(self):
        broker = self.broker()
        broker._last_result = resting(7)
        broker._query_order_status = lambda oid: order_status("filled", 0.5, 101.5)

        outcome = broker._resolve_resting_order("BTC", size=0.5)

        self.assertAlmostEqual(outcome.filled_size, 0.5)
        self.assertAlmostEqual(outcome.avg_price, 101.5)
        self.assertFalse(outcome.cancelled)

    def test_timeout_cancels_and_confirms_removal(self):
        """Reports stays open until a cancel is actually issued, so this
        exercises the real timeout -> cancel -> verify sequence."""
        exchange = FakeExchange(resting(7))
        broker = self.broker(exchange)
        broker._last_result = resting(7)
        broker._query_order_status = lambda oid: order_status(
            "canceled" if exchange.cancels else "open"
        )

        outcome = broker._resolve_resting_order("BTC", size=0.5)

        self.assertEqual(outcome.filled_size, 0.0)
        self.assertTrue(outcome.cancelled)
        self.assertEqual(exchange.cancels, [("BTC", 7)])

    def test_unconfirmable_cancellation_is_reported_honestly(self):
        """An order still open after a cancel attempt must NOT be called gone."""
        exchange = FakeExchange(filled(0.1, 100.0), cancel_raises=True)
        broker = self.broker(exchange)
        broker._last_result = resting(7)
        broker._query_order_status = lambda oid: order_status("open")

        outcome = broker._resolve_resting_order("BTC", size=0.5)

        self.assertFalse(outcome.cancelled)

    def test_missing_oid_never_claims_cancellation(self):
        """This is the exact defect: no oid meant 'return 0,0' and the caller
        announced a cancellation that had never been attempted."""
        exchange = FakeExchange(resting(1))
        broker = self.broker(exchange)
        broker._last_result = None  # what the old bug produced

        outcome = broker._resolve_resting_order("BTC", size=0.5)

        self.assertEqual(outcome, RestingOrderOutcome(cancelled=False))
        self.assertEqual(exchange.cancels, [], "nothing was cancelled")

    def test_partial_fill_is_captured(self):
        broker = self.broker()
        broker._last_result = resting(7)
        states = iter([order_status("open", 0.2, 100.0), order_status("canceled", 0.2, 100.0)])
        broker._query_order_status = lambda oid: next(states, order_status("canceled", 0.2, 100.0))

        outcome = broker._resolve_resting_order("BTC", size=0.5)

        self.assertAlmostEqual(outcome.filled_size, 0.2)
        self.assertTrue(outcome.cancelled)
        self.assertTrue(
            any("partially filled" in e.get("message", "") for e in self.events())
        )

    def test_rejected_order_is_not_cancelled(self):
        broker = self.broker(FakeExchange(resting(7)))
        broker._last_result = resting(7)
        broker._query_order_status = lambda oid: order_status("rejected")

        outcome = broker._resolve_resting_order("BTC", size=0.5)

        self.assertEqual(outcome.filled_size, 0.0)
        self.assertFalse(outcome.cancelled)


class TestOpenPositionWithRestingOrder(ExecutionTestCase):
    def test_uncancellable_order_halts_instead_of_claiming_success(self):
        """The critical case: an order we cannot cancel may still fill, so the
        agent must stop rather than continue believing it is flat."""
        exchange = FakeExchange(resting(7))
        broker = self.broker(exchange)
        broker._last_result = None  # unrecoverable oid
        broker._query_order_status = lambda oid: order_status("open")

        with self.assertRaises(CriticalExecutionError) as ctx:
            broker.open_position("BTC", "long", 0.001, 50_000.0, leverage=2)

        self.assertIn("could not be cancelled", str(ctx.exception))

    def test_confirmed_cancellation_is_a_recoverable_error(self):
        exchange = FakeExchange(resting(7))
        broker = self.broker(exchange)
        states = iter([order_status("open"), order_status("canceled")])
        broker._query_order_status = lambda oid: next(states, order_status("canceled"))

        with self.assertRaises(ExecutionError) as ctx:
            broker.open_position("BTC", "long", 0.001, 50_000.0, leverage=2)

        self.assertNotIsInstance(ctx.exception, CriticalExecutionError)
        self.assertIn("did not fill", str(ctx.exception))

    def test_unfilled_order_is_journalled_as_unfilled_not_cancelled(self):
        exchange = FakeExchange(resting(7))
        broker = self.broker(exchange)
        broker._last_result = None
        broker._query_order_status = lambda oid: order_status("open")

        with self.assertRaises(ExecutionError):
            broker.open_position("BTC", "long", 0.001, 50_000.0, leverage=2)

        unfilled = [e for e in self.events() if e["kind"] == "order_unfilled"]
        self.assertEqual(len(unfilled), 1)
        self.assertFalse(unfilled[0]["cancelled"])


# ---------------------------------------------------------------------------
# Fill reporting
# ---------------------------------------------------------------------------


class TestFillReporting(ExecutionTestCase):
    def test_requested_and_filled_size_are_both_recorded(self):
        """The journal previously logged the post-fill size under both keys,
        destroying the ability to audit partial fills."""
        exchange = FakeExchange(filled(0.0008, 50_100.0))
        broker = self.broker(exchange)

        broker.open_position("BTC", "long", 0.001, 50_000.0, leverage=2)

        events = [e for e in self.events() if e["kind"] == "order_filled"]
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0]["requested_size"], 0.001, places=6)
        self.assertAlmostEqual(events[0]["filled_size"], 0.0008, places=6)
        self.assertAlmostEqual(events[0]["avg_price"], 50_100.0)

    def test_exchange_fill_price_overrides_the_requested_price(self):
        exchange = FakeExchange(filled(0.001, 50_999.0))
        broker = self.broker(exchange)

        position = broker.open_position("BTC", "long", 0.001, 50_000.0, leverage=2)

        self.assertAlmostEqual(position.entry_price, 50_999.0)


class TestCancelOrderSignature(ExecutionTestCase):
    def test_cancel_uses_name_then_oid(self):
        exchange = FakeExchange(filled(1, 1))
        broker = self.broker(exchange)

        self.assertTrue(broker._cancel_order("BTC", 55))
        self.assertEqual(exchange.cancels, [("BTC", 55)])

    def test_cancel_failure_is_reported(self):
        exchange = FakeExchange(filled(1, 1), cancel_raises=True)
        broker = self.broker(exchange)
        self.assertFalse(broker._cancel_order("BTC", 55))

    def test_no_exchange_returns_false(self):
        broker = self.broker(None)
        self.assertFalse(broker._cancel_order("BTC", 55))


# ---------------------------------------------------------------------------
# Sizing notes
# ---------------------------------------------------------------------------


class FakeCandles:
    def __init__(self, atr_value):
        self._atr = atr_value

    def atr(self, period):
        return [self._atr]


class FakeMarket:
    def __init__(self, series):
        self._series = series
        self.testnet = False

    def candles(self, symbol, interval, lookback):
        if self._series is None:
            from agent.market_data import MarketDataError

            raise MarketDataError("simulated failure")
        return self._series


class TestSizingNotes(ExecutionTestCase):
    def test_allocation_note_does_not_claim_leverage_multiplies_notional(self):
        """The old note read '$100 at 2x', implying $200 of exposure, while the
        code capped notional at $100."""
        market = FakeMarket(FakeCandles(500.0))
        result = compute_size(market, self.cfg, "BTC", "long", 50_000.0, 1_000.0)

        note = next(n for n in result.notes if "allocation cap" in n)
        self.assertIn("notional", note)
        self.assertNotIn("at 2x ->", note)
        self.assertAlmostEqual(result.notional, 100.0, places=6)

    def test_silent_atr_fallback_is_now_reported(self):
        market = FakeMarket(None)  # candle fetch fails
        result = compute_size(market, self.cfg, "BTC", "long", 50_000.0, 1_000.0)
        self.assertTrue(any("candle fetch failed" in n for n in result.notes))

    def test_short_atr_series_is_reported(self):
        market = FakeMarket(FakeCandles(0.0))
        result = compute_size(market, self.cfg, "BTC", "long", 50_000.0, 1_000.0)
        self.assertTrue(any("ATR series too short" in n for n in result.notes))

    def test_small_notional_warning_uses_the_shared_fee_rate(self):
        market = FakeMarket(FakeCandles(500.0))
        result = compute_size(market, self.cfg, "BTC", "long", 50_000.0, 1_000.0)
        self.assertTrue(any("is small" in n for n in result.notes))

    def test_short_side_gets_a_reward_risk_note(self):
        market = FakeMarket(FakeCandles(500.0))
        result = compute_size(market, self.cfg, "BTC", "short", 50_000.0, 1_000.0)
        self.assertTrue(any("reward:risk" in n for n in result.notes), result.notes)

    def test_zero_equity_is_rejected(self):
        market = FakeMarket(FakeCandles(500.0))
        with self.assertRaises(ExecutionError):
            compute_size(market, self.cfg, "BTC", "long", 50_000.0, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
