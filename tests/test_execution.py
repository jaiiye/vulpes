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

from agent.config import ConfigError, load_config  # noqa: E402
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
        # The venue answers a market close with an order status like any other
        # order. `{"status": "ok"}` is a shape it never returns, and it became
        # load-bearing once `Broker._market_close_confirmed` started reading
        # this value: a close without a fill is not a close.
        return filled(0.001, 0.0, oid=99)

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


class TestAttachProtection(ExecutionTestCase):
    """The stop-loss path, which had no test at all - and it showed.

    Two defects lived here at once. The venue call's rejection was never
    checked, so a refused stop was journaled as `protection_placed` while the
    position had nothing protecting it; and the journal call passed `kind=`,
    which collides with `Journal.event(self, kind, **payload)` and raised a
    TypeError on *every* call. The second masked the first - the operator saw a
    message about the journal instead of the venue's actual reason.

    These run against the real `Journal`, not a stand-in. The collision only
    exists in the real signature, so a double would have gone on hiding it.
    """

    PLACED = {"response": {"data": {"statuses": [{"resting": {"oid": 11}}]}}}
    REJECTED = {
        "response": {"data": {"statuses": [{"error": "Order has invalid price."}]}}
    }

    def test_a_placed_order_is_journalled_without_raising(self):
        """The TypeError regression: no protective order was ever recorded as
        placed, because the bookkeeping raised after the venue accepted it."""
        ex = FakeExchange(self.PLACED)
        self.broker(exchange=ex)._attach_protection("BTC", "long", 0.001, 90_000.0, None)

        placed = [e for e in self.events() if e["kind"] == "protection_placed"]
        self.assertEqual(len(placed), 1)
        self.assertEqual(placed[0]["symbol"], "BTC")
        self.assertEqual(placed[0]["trigger_px"], 90_000.0)

    def test_which_protection_it_was_is_its_own_field(self):
        """`kind` is the event's own name. Reusing it for the payload is exactly
        what collided with the signature."""
        ex = FakeExchange(self.PLACED)
        self.broker(exchange=ex)._attach_protection("BTC", "long", 0.001, 90_000.0, 95_000.0)

        placed = [e for e in self.events() if e["kind"] == "protection_placed"]
        self.assertEqual([e["protection"] for e in placed], ["stop", "take_profit"])

    def test_the_order_is_sent_reduce_only_as_a_sell(self):
        """A protective order that could open a position is not a protective
        order - it is a second way to lose money."""
        ex = FakeExchange(self.PLACED)
        self.broker(exchange=ex)._attach_protection("BTC", "long", 0.001, 90_000.0, None)

        _, is_buy, _, _, order_type, reduce_only = ex.orders[0]
        self.assertFalse(is_buy, "closing a long means selling")
        self.assertTrue(reduce_only)
        self.assertIn("trigger", order_type)

    def test_a_rejected_order_is_not_reported_as_placed(self):
        """The venue returns a rejection as a normal response, not as an
        exception. Unchecked, the caller was told the position was protected
        while nothing was on the book."""
        ex = FakeExchange(self.REJECTED)
        with self.assertRaises(ExecutionError):
            self.broker(exchange=ex)._attach_protection("BTC", "long", 0.001, 90_000.0, None)

        kinds = [e["kind"] for e in self.events()]
        self.assertNotIn(
            "protection_placed", kinds, "a rejected order was recorded as placed"
        )
        errors = [e for e in self.events() if e["kind"] == "error"]
        self.assertTrue(errors, "a rejection must be journaled")
        self.assertIn("unprotected", errors[0]["message"])

    def test_the_rejection_reason_reaches_the_caller(self):
        """The earlier code replaced the venue's reason with a TypeError about
        the journal, which is what made this take a live order to find."""
        ex = FakeExchange(self.REJECTED)
        with self.assertRaises(ExecutionError) as ctx:
            self.broker(exchange=ex)._attach_protection("BTC", "long", 0.001, 90_000.0, 95_000.0)

        message = str(ctx.exception)
        self.assertIn("invalid price", message)
        self.assertIn("stop", message)
        self.assertNotIn("Journal.event", message)

    def test_a_journal_failure_does_not_claim_the_position_is_unprotected(self):
        """By the time the journal is written the order IS on the venue, so a
        bookkeeping error must not send the caller to unwind a position that is
        in fact protected."""
        ex = FakeExchange(self.PLACED)
        broker = self.broker(exchange=ex)
        broker.journal.event = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("disk full")
        )

        with self.assertRaises(RuntimeError):
            broker._attach_protection("BTC", "long", 0.001, 90_000.0, None)
        self.assertEqual(len(ex.orders), 1, "the order should still have been sent")


class PriceRuleFakeExchange(FakeExchange):
    """A fake that enforces the venue's price format.

    `FakeExchange` echoes whatever it is handed, which is precisely why a
    rounding bug shipped: a double cannot enforce a rule it was never taught,
    and this rule belongs to the venue. This one implements the documented perp
    rule - at most 5 significant figures and at most `6 - szDecimals` decimals -
    and rejects the way the venue does: as a normal response, not an exception.
    """

    def __init__(self, sz_decimals: int, **kw):
        super().__init__(**kw)
        self.sz_decimals = sz_decimals
        self.rejected_prices: list[float] = []

    def accepts(self, px: float) -> bool:
        if px <= 0:
            return False
        return float(px) == round(float(f"{px:.5g}"), max(0, 6 - self.sz_decimals))

    def order(self, symbol, is_buy, size, px, order_type, reduce_only=False):
        if not self.accepts(px):
            self.rejected_prices.append(px)
            return {
                "response": {"data": {"statuses": [{"error": "Order has invalid price."}]}}
            }
        return super().order(symbol, is_buy, size, px, order_type, reduce_only)


class TestPriceRounding(ExecutionTestCase):
    """Prices must be rounded or the venue refuses them.

    Found by placing a real testnet order - the first this project ever sent.
    `price * (1 + 5bp)` reached the venue as `86677.81725`, ten significant
    figures against a limit of five, and came back `Order has invalid price.`
    Every limit order and every protective order was being refused.

    No test in the suite could have caught it, because `FakeExchange` accepts
    any float. `PriceRuleFakeExchange` above is the fake that would have.
    """

    #: The exact prices the first live order was built from: the observed mid,
    #: and `mid * (1 + 5bp)` - the ten-significant-figure value the venue
    #: refused with `Order has invalid price.`
    LIVE_MID = 86_634.5
    LIVE_REJECTED = 86_677.81725

    def test_the_live_rejected_price_rounds_to_the_accepted_one(self):
        """Pinned to the number the venue actually accepted in the live test."""
        self.assertEqual(self.broker()._round_price("BTC", self.LIVE_REJECTED), 86678.0)

    def test_five_significant_figures_and_sz_decimals_both_hold(self):
        broker = self.broker()
        broker._sz_decimals["BTC"] = 5      # 6 - 5 = 1 decimal
        rounded = broker._round_price("BTC", 123456.789)
        self.assertEqual(rounded, 123460.0)

        broker._sz_decimals["TINY"] = 0     # 6 decimals
        self.assertEqual(broker._round_price("TINY", 0.01234567), 0.012346)

    def test_it_matches_the_sdk_s_market_order_rounding(self):
        """Both sides must agree, or a limit order and a market order for the
        same intent would sit at different prices."""
        broker = self.broker()
        for price in (86677.81725, 1234.56789, 0.0123456789, 99999.99):
            with self.subTest(price=price):
                self.assertEqual(
                    broker._round_price("BTC", price),
                    round(float(f"{price:.5g}"), max(0, 6 - broker.SIZE_DECIMALS)),
                )

    def test_a_non_positive_price_is_refused(self):
        for bad in (0.0, -1.0):
            with self.subTest(price=bad):
                with self.assertRaises(ExecutionError):
                    self.broker()._round_price("BTC", bad)

    def test_a_limit_order_reaches_the_venue_rounded(self):
        """End to end through `open_position`, reproducing the live case: the
        mid it was given produces `mid * (1 + 5bp)` = `86677.81725`, which is
        what the venue refused."""
        ex = PriceRuleFakeExchange(sz_decimals=5, order_result=filled(0.001, 86_678.0))
        broker = self.broker(exchange=ex)
        broker._sz_decimals["BTC"] = 5
        broker.open_position("BTC", "long", 0.001, self.LIVE_MID, 3)

        # Without the rounding this list holds 86677.81725 and the order is
        # refused, which is what happened on the live account.
        self.assertEqual(ex.rejected_prices, [], "the venue refused the limit price")
        self.assertEqual(ex.orders[0][3], 86_678.0)

    def test_a_trigger_price_reaches_the_venue_rounded(self):
        """The stop is sent twice - as the order price and as `triggerPx` - and
        the venue validates both."""
        ex = PriceRuleFakeExchange(sz_decimals=5, order_result=resting(11))
        broker = self.broker(exchange=ex)
        broker._sz_decimals["BTC"] = 5
        trigger = self.LIVE_REJECTED * 0.98
        broker._attach_protection("BTC", "long", 0.001, trigger, None)

        self.assertEqual(ex.rejected_prices, [], "the venue refused the trigger price")
        _, _, _, px, order_type, _ = ex.orders[0]
        self.assertEqual(px, round(float(f"{trigger:.5g}"), 1))
        self.assertIn("trigger", order_type)
        self.assertEqual(order_type["trigger"]["triggerPx"], px)


class FakeAccountInfo:
    """The three reads `account_equity` needs, scripted."""

    def __init__(self, perp_value=0.0, spot_usdc=0.0,
                 abstraction="unifiedAccount", abstraction_raises=False,
                 spot_coins=("USDC",)):
        self.perp_value = perp_value
        self.spot_usdc = spot_usdc
        self.abstraction = abstraction
        self.abstraction_raises = abstraction_raises
        self.spot_coins = spot_coins

    def user_state(self, address):
        return {
            "marginSummary": {"accountValue": str(self.perp_value)},
            "assetPositions": [],
        }

    def spot_user_state(self, address):
        return {
            "balances": [
                {"coin": c, "total": str(self.spot_usdc)} for c in self.spot_coins
            ]
        }

    def post(self, path, payload):
        if self.abstraction_raises:
            raise RuntimeError("endpoint not supported")
        return self.abstraction


class TestAccountEquity(ExecutionTestCase):
    """Equity has to be right: every position size is derived from it.

    Found by trying to trade a funded account. It reported $0.00 and refused
    every order with "account equity is zero; cannot size a position", while the
    account held 999 USDC and the venue accepted market orders - so the money
    was there and the reading was wrong.
    """

    def _broker(self, info):
        # A non-None `_exchange` is all `_ensure_sdk` checks, so the fake info
        # is used as-is and nothing touches the network.
        broker = self.broker(exchange=object())
        broker._info = info
        broker._account_address = "0xacct"
        return broker

    def test_a_unified_account_is_read_from_spot(self):
        info = FakeAccountInfo(perp_value=0.0, spot_usdc=998.8)
        self.assertAlmostEqual(self._broker(info).account_equity(), 998.8)

    def test_a_classic_account_is_still_read_from_the_perp_summary(self):
        info = FakeAccountInfo(perp_value=500.0, spot_usdc=998.8, abstraction="default")
        self.assertAlmostEqual(self._broker(info).account_equity(), 500.0)

    def test_a_unified_account_with_a_position_open_still_reads_spot(self):
        """`accountValue` turns non-zero while a position is open - it tracked
        margin, not the balance - so keying off the *value* instead of the
        account mode would size the next trade on about $5."""
        info = FakeAccountInfo(perp_value=5.20, spot_usdc=998.8)
        self.assertAlmostEqual(self._broker(info).account_equity(), 998.8)

    def test_an_unreadable_mode_keeps_the_previous_behaviour(self):
        """Inventing a number is worse than reporting what the perp summary
        said; an older venue may not have the endpoint at all."""
        info = FakeAccountInfo(perp_value=42.0, spot_usdc=998.8, abstraction_raises=True)
        self.assertAlmostEqual(self._broker(info).account_equity(), 42.0)

    def test_spot_holding_no_usdc_reports_zero_rather_than_a_guess(self):
        info = FakeAccountInfo(perp_value=0.0, spot_usdc=123.0, spot_coins=("HYPE",))
        self.assertAlmostEqual(self._broker(info).account_equity(), 0.0)

    def test_dry_run_is_unaffected(self):
        import os as _os

        broker = self._broker(FakeAccountInfo(perp_value=0.0, spot_usdc=998.8))
        broker.dry_run = True
        expected = float(_os.getenv("DRY_RUN_EQUITY_USD", 1000.0))
        self.assertAlmostEqual(broker.account_equity(), expected)

    def test_sizing_over_a_unified_account_now_produces_a_position(self):
        """The end that matters: `compute_size` refused this outright before,
        so the bot could not place a single order on this account type."""
        broker = self._broker(FakeAccountInfo(perp_value=0.0, spot_usdc=1000.0))
        result = compute_size(None, self.cfg, "BTC", "long", 50_000.0, broker.account_equity())
        self.assertGreater(result.notional, 0)

    def test_the_same_sizing_still_refuses_a_genuinely_empty_account(self):
        """The fix must not turn "no money" into a tradeable number."""
        broker = self._broker(FakeAccountInfo(perp_value=0.0, spot_usdc=0.0))
        with self.assertRaises(ExecutionError):
            compute_size(None, self.cfg, "BTC", "long", 50_000.0, broker.account_equity())


class SdkBlip(Exception):
    """Stands in for `requests.exceptions.SSLError` - the point is only that it
    is NOT an `ExecutionError`, because that is what the handlers know."""


class TestSdkInitFailureIsRecoverable(ExecutionTestCase):
    """A blip while lazily building the SDK must not kill the process.

    Found by running live. One transient SSL error out of `Info.__init__`
    escaped `_ensure_sdk`, then `live_positions`, then `reconcile_startup`'s
    `except ExecutionError`, then `run()`'s `except CriticalExecutionError` -
    killing the process seconds after start, *before* the loop that is built to
    survive cycle errors had begun.

    The SDK is not installed in the offline test environment, so these exercise
    `_build_clients` directly with injected classes rather than pretending to
    build a real client.
    """

    def _broker(self):
        broker = self.broker(exchange=None)
        broker.SDK_INIT_ATTEMPTS = 2
        broker.SDK_INIT_BACKOFF_SECONDS = 0
        return broker

    def test_a_persistent_failure_raises_execution_error_not_the_sdk_type(self):
        broker = self._broker()

        class Raises:
            def __init__(self, *a, **k):
                raise SdkBlip("UNEXPECTED_EOF_WHILE_READING")

        with self.assertRaises(ExecutionError):
            broker._build_clients(Raises, Raises, None, "u", "a")

    def test_it_is_retried_before_giving_up(self):
        """Counted per *attempt*, not per constructor.

        Both clients raising would give two calls per attempt, so a count that
        does not distinguish the two would pass with the retry removed - it did,
        until this was written to fail only the second client.
        """
        broker = self._broker()          # 2 attempts
        attempts = []

        class FailingInfo:
            def __init__(self, *a, **k):
                attempts.append(1)
                raise SdkBlip("blip")

        with self.assertRaises(ExecutionError):
            broker._build_clients(
                lambda *a, **k: "exchange", FailingInfo, None, "u", "a"
            )
        self.assertEqual(len(attempts), broker.SDK_INIT_ATTEMPTS)

    def test_a_later_attempt_can_succeed(self):
        """The failure is transient by nature, so the retry is the thing that
        turns a dead process into a settled one.

        Only the second client fails, which also pins that a failure there
        rebuilds the pair rather than keeping the first - it is counted per
        attempt, not per constructor.
        """
        broker = self._broker()
        broker.SDK_INIT_ATTEMPTS = 3
        seen = []

        class FlakyInfo:
            def __init__(self, *a, **k):
                seen.append(1)
                if len(seen) < 3:
                    raise SdkBlip("blip")

        exchange, info = broker._build_clients(
            lambda *a, **k: "exchange", FlakyInfo, None, "u", "a"
        )
        self.assertEqual(len(seen), 3, "should have taken three attempts")
        self.assertEqual(exchange, "exchange")
        self.assertIsNotNone(info)

    def test_a_failure_in_the_second_client_assigns_neither(self):
        """`_ensure_sdk`'s guard tests `_exchange`, so storing it before `Info`
        exists would make the pair permanently un-buildable - every later call
        would fail on `_info is None` until the process restarted."""
        broker = self._broker()
        broker.SDK_INIT_ATTEMPTS = 1

        class Raises:
            def __init__(self, *a, **k):
                raise SdkBlip("blip")

        with self.assertRaises(ExecutionError):
            broker._build_clients(lambda *a, **k: "exchange", Raises, None, "u", "a")
        self.assertIsNone(broker._exchange)
        self.assertIsNone(broker._info)

    def test_the_failure_message_names_the_attempts_and_the_cause(self):
        broker = self._broker()

        class Raises:
            def __init__(self, *a, **k):
                raise SdkBlip("UNEXPECTED_EOF_WHILE_READING")

        with self.assertRaises(ExecutionError) as ctx:
            broker._build_clients(Raises, Raises, None, "u", "a")
        message = str(ctx.exception)
        self.assertIn(str(broker.SDK_INIT_ATTEMPTS), message)
        self.assertIn("SdkBlip", message)
        self.assertIn("UNEXPECTED_EOF_WHILE_READING", message)

    # -- the invariant the callers depend on -------------------------------

    def test_live_positions_passes_the_build_failure_through_unchanged(self):
        """`live_positions` calls `_ensure_sdk` outside its own `try`, so it
        relies on that method raising only `ExecutionError`. This pins that."""
        broker = self._broker()
        broker._ensure_sdk = lambda: (_ for _ in ()).throw(
            ExecutionError("could not initialise the Hyperliquid clients")
        )
        with self.assertRaises(ExecutionError):
            broker.live_positions()


# ---------------------------------------------------------------------------
# The venue's minimum order value
# ---------------------------------------------------------------------------


class MinValueFakeExchange(FakeExchange):
    """A fake that enforces the venue's minimum order value.

    Same reasoning as `PriceRuleFakeExchange`: the rule belongs to the venue,
    and `FakeExchange` echoes whatever it is handed. This one refuses the way
    the venue does - as a normal response, not an exception - and only for the
    orders the venue refuses.

    The exemption is keyed on the order being a *trigger* order, which is what
    was measured, not on `reduce_only`, which is what it looks like:

        entry             $ 9.54   refused
        entry             $10.40   accepted
        reduce-only stop  $ 9.03   accepted   (trigger)
        reduce-only close $ 5.21   refused    (market)
    """

    def __init__(self, minimum: float = 10.0, **kw):
        super().__init__(**kw)
        self.minimum = minimum
        self.refused: list[float] = []

    @staticmethod
    def _is_trigger(order_type) -> bool:
        return isinstance(order_type, dict) and "trigger" in order_type

    def _refusal(self, value: float):
        if value >= self.minimum:
            return None
        self.refused.append(value)
        return {
            "response": {
                "data": {
                    "statuses": [
                        {
                            "error": f"Order must have minimum value of "
                            f"${self.minimum:g}. asset=3"
                        }
                    ]
                }
            }
        }

    def order(self, symbol, is_buy, size, px, order_type, reduce_only=False):
        if not self._is_trigger(order_type):
            refusal = self._refusal(size * px)
            if refusal is not None:
                return refusal
        return super().order(symbol, is_buy, size, px, order_type, reduce_only)


class RefusingCloseFakeExchange(FakeExchange):
    """A fake whose market close is refused, the way the venue refuses one.

    Measured: a $5.21 reduce-only partial close of a $10.41 position came back
    `Order must have minimum value of $10. asset=3`, and the position was
    unchanged - a refused close does not partially close.
    """

    def market_close(self, symbol):
        self.close_calls += 1
        return {
            "response": {
                "data": {
                    "statuses": [
                        {"error": "Order must have minimum value of $10. asset=3"}
                    ]
                }
            }
        }


class UnconfirmableCloseFakeExchange(FakeExchange):
    """A fake whose close comes back without a fill.

    The venue does not answer this way today. It is here because
    `_parse_order_result` maps any shape it does not recognise to "resting",
    and a market order cannot rest - so an unrecognised response means the
    close is unconfirmed, which must not be reported as done.
    """

    def market_close(self, symbol):
        self.close_calls += 1
        return {"status": "ok"}


class ExecutionValueTestCase(ExecutionTestCase):
    """Shared fixtures for the two classes below."""

    def broker_for(self, exchange, **kw):
        broker = self.broker(exchange=exchange, **kw)
        broker._sz_decimals["BTC"] = 5
        return broker

    @staticmethod
    def position(size=0.00012, entry=87_000.0):
        return Position(
            symbol="BTC",
            side="long",
            size=size,
            entry_price=entry,
            notional=size * entry,
            leverage=3,
            is_dry_run=False,
        )

    def journal_text(self) -> str:
        # The journal file is created lazily, so "nothing was written" and "no
        # file" are the same state.
        path = Path(self.tmp.name) / "journal.jsonl"
        return path.read_text() if path.exists() else ""


class TestMinimumOrderValue(ExecutionValueTestCase):
    """The $10 minimum is checked before the order is sent.

    Recorded from a live testnet run as a two-line footnote - "`0.0001 BTC` was
    refused (`Order must have minimum value of $10`)" - which understated it.
    For an account whose allocation budget lands below $10 that refusal is what
    happens on every signal, and nothing in the log says why: the cycle prints
    `ENTER ...` and then `ORDER FAILED: order rejected: 'Order must have minimum
    value of $10.'`, which reads like a broken integration rather than a small
    account.
    """

    def test_an_entry_below_the_minimum_is_refused_before_it_is_sent(self):
        """The property is that the venue never sees it. Asserting only
        `ExecutionError` would also pass with the check removed, because the
        venue's own refusal raises the same type - so the call count is part of
        the assertion."""
        ex = MinValueFakeExchange(order_result=filled(0.00012, 87_000.0))
        broker = self.broker_for(ex)

        with self.assertRaises(ExecutionError) as ctx:
            broker.open_position("BTC", "long", 0.00011, 87_000.0, 3)

        self.assertEqual(ex.orders, [], "no order should have reached the venue")
        self.assertIn("minimum order value", str(ctx.exception))

    def test_an_entry_at_or_above_the_minimum_is_sent(self):
        ex = MinValueFakeExchange(order_result=filled(0.00012, 87_000.0))
        broker = self.broker_for(ex)

        broker.open_position("BTC", "long", 0.00012, 87_000.0, 3)

        self.assertEqual(len(ex.orders), 1)
        self.assertEqual(ex.refused, [])

    def test_the_message_states_the_value_and_the_limit(self):
        broker = self.broker_for(MinValueFakeExchange(order_result=filled(1.0, 1.0)))

        with self.assertRaises(ExecutionError) as ctx:
            broker._reject_small_order("BTC", 1.0, 9.5)

        message = str(ctx.exception)
        self.assertIn("$9.50", message)
        self.assertIn("$10 minimum order value", message)

    def test_the_limit_is_inclusive(self):
        """The rule reads "minimum value of $10", so exactly $10 is allowed.

        The exact boundary could NOT be measured: the venue's size quantum on
        BTC is 0.00001 = $0.87 at $87k, wider than the gap between the largest
        refused case ($9.54) and the smallest accepted one ($10.40). So this
        follows the venue's own wording rather than a measurement, and is
        pinned here so the choice is visible rather than implied by a `<`.
        """
        broker = self.broker_for(MinValueFakeExchange(order_result=filled(1.0, 1.0)))

        broker._reject_small_order("BTC", 1.0, 10.0)  # exactly at it: allowed

        with self.assertRaises(ExecutionError):
            broker._reject_small_order("BTC", 1.0, 9.999)  # just under: refused

    def test_the_rounded_size_decides_when_the_two_straddle_the_limit(self):
        """The venue values the order it receives, not the one requested.

        `0.0001142` requests $10.01 of value, but with `szDecimals = 6` the
        venue can only receive `0.000114`, which is $9.99 - so this must be
        refused. A check on the requested size would send it and be refused
        anyway, which is the outcome the check exists to avoid.
        """
        ex = MinValueFakeExchange(order_result=filled(0.0001, 87_650.0))
        broker = self.broker(exchange=ex)
        broker._sz_decimals["BTC"] = 6

        with self.assertRaises(ExecutionError):
            broker.open_position("BTC", "long", 0.0001142, 87_650.0, 3)

        self.assertEqual(ex.orders, [])

    def test_a_dry_run_refuses_it_too(self):
        """A dry run that reported an entry the live path refuses would be
        wrong about the one thing it exists to rehearse."""
        broker = Broker(self.cfg, None, self.journal)
        self.assertTrue(broker.dry_run)

        with self.assertRaises(ExecutionError):
            broker.open_position("BTC", "long", 0.0001, 87_000.0, 3)

        self.assertNotIn("order_simulated", self.journal_text())

    def test_a_trigger_order_below_the_minimum_is_still_accepted(self):
        """$9.03 of stop on a $10.41 position is the live case that was
        accepted while a same-sized market close was refused. Requiring the
        protective orders to clear the minimum would refuse entries that work,
        so this pins the exemption."""
        ex = MinValueFakeExchange(order_result=resting(1))
        broker = self.broker_for(ex)

        broker._attach_protection("BTC", "long", 0.00012, 70_000.0, None)

        self.assertEqual(len(ex.orders), 1)
        self.assertEqual(ex.refused, [])


class TestCloseConfirmation(ExecutionValueTestCase):
    """A refused close must not be reported as a close.

    `Exchange.market_close()` returns a status instead of raising when the
    venue refuses, exactly as `order()` does. `_attach_protection` already
    documented and handled that rule; both close paths did not.
    """

    def test_a_confirmed_close_returns_the_pnl(self):
        ex = FakeExchange(order_result=filled(0.00012, 87_000.0))
        broker = self.broker_for(ex)

        pnl = broker.close_position(self.position(), 88_000.0, "test")

        self.assertAlmostEqual(pnl, (88_000.0 - 87_000.0) * 0.00012, places=6)
        self.assertIn("close_placed", self.journal_text())

    def test_a_refused_close_raises_instead_of_returning_a_pnl(self):
        """It used to return a PnL, so `bot.close_position` cleared
        `self.position`, added the invented PnL to the run stats and fed it to
        `discipline.record_outcome` - the agent walked away from exposure it
        still had, with a made-up result in its guardrails."""
        ex = RefusingCloseFakeExchange(order_result=filled(0.00012, 87_000.0))
        broker = self.broker_for(ex)

        with self.assertRaises(ExecutionError) as ctx:
            broker.close_position(self.position(), 86_000.0, "test")

        self.assertIn("not confirmed as filled", str(ctx.exception))
        self.assertNotIn("close_placed", self.journal_text())

    def test_a_close_without_a_fill_is_not_treated_as_one(self):
        ex = UnconfirmableCloseFakeExchange(order_result=filled(1.0, 1.0))
        broker = self.broker_for(ex)

        with self.assertRaises(ExecutionError):
            broker.close_position(self.position(), 86_000.0, "test")

        self.assertNotIn("close_placed", self.journal_text())

    def test_a_refused_unwind_stops_the_agent_and_reports_the_exposure(self):
        """`_unwind_unprotected` used to turn a refused close into a
        *recoverable* error: the agent kept running with a position open and
        unprotected, and `on_unprotected` - the channel that demands manual
        intervention - was never called."""
        reported = []
        ex = RefusingCloseFakeExchange(order_result=filled(0.00012, 87_000.0))
        broker = self.broker_for(ex, on_unprotected=reported.append)

        raised = broker._unwind_unprotected(
            "BTC", "long", 0.00012, 87_000.0, ExecutionError("protection failed")
        )

        self.assertIsInstance(raised, CriticalExecutionError)
        self.assertEqual(len(reported), 1)
        self.assertEqual(reported[0].symbol, "BTC")

    def test_a_successful_unwind_stays_recoverable(self):
        """Stopping everything is for an unconfirmed close, not for every
        unwind - otherwise an ordinary protection failure would halt the run."""
        ex = FakeExchange(order_result=filled(0.00012, 87_000.0))
        broker = self.broker_for(ex)

        raised = broker._unwind_unprotected(
            "BTC", "long", 0.00012, 87_000.0, ExecutionError("protection failed")
        )

        self.assertNotIsInstance(raised, CriticalExecutionError)
        self.assertIsInstance(raised, ExecutionError)
        self.assertIn("unwound_unprotected", self.journal_text())


class RefusingOrderFakeExchange(FakeExchange):
    """A fake whose entry is refused with whatever wording it is given."""

    def __init__(self, error: str, **kw):
        super().__init__(**kw)
        self.error = error

    def order(self, symbol, is_buy, size, px, order_type, reduce_only=False):
        self.orders.append((symbol, is_buy, size, px, order_type, reduce_only))
        return {"response": {"data": {"statuses": [{"error": self.error}]}}}


class TestRejectionIsDiagnosable(ExecutionValueTestCase):
    """A refused order must report the venue's own words.

    `_parse_order_result` returns `("error", 0.0, 0.0)` and the caller printed
    the *price* from that tuple, so every refusal surfaced as
    `order rejected: 0.0`. Found by trying to diagnose one: the live message
    named neither the rule nor the field, and the two rules it could have been
    call for different fixes.
    """

    def test_a_refused_entry_reports_the_venue_s_text(self):
        ex = RefusingOrderFakeExchange(
            "Order has invalid size.", order_result=filled(1.0, 1.0)
        )
        broker = self.broker_for(ex)

        with self.assertRaises(ExecutionError) as ctx:
            broker.open_position("BTC", "long", 0.00012, 87_000.0, 3)

        message = str(ctx.exception)
        self.assertIn("Order has invalid size.", message)
        self.assertNotIn("rejected: 0.0", message)

    def test_the_other_rule_is_still_distinguishable(self):
        """The two refusals must not collapse to the same log line."""
        ex = RefusingOrderFakeExchange(
            "Order must have minimum value of $10. asset=3",
            order_result=filled(1.0, 1.0),
        )
        broker = self.broker_for(ex)

        with self.assertRaises(ExecutionError) as ctx:
            broker.open_position("BTC", "long", 0.00012, 87_000.0, 3)

        self.assertIn("minimum value of $10", str(ctx.exception))

    def test_a_missing_error_text_falls_back_to_the_raw_response(self):
        """Never silently blank: if the shape is unexpected, show what came."""
        broker = self.broker_for(FakeExchange(order_result=filled(1.0, 1.0)))

        self.assertEqual(broker._order_error({"unexpected": "shape"}), "{'unexpected': 'shape'}")


class TestSizeIsRoundedWithTheVenueDecimals(ExecutionValueTestCase):
    """`szDecimals` must be known before the entry is rounded.

    `_round_size` rounds to `self._sz_decimals`, which `_ensure_sdk` fills from
    the venue's metadata - but the rounding ran first, so the *first* live order
    after startup used the fallback of 6 decimals. Measured on testnet:
    `0.000121` BTC against `szDecimals = 5` came back
    `Order has invalid size.`, and the agent logged `order rejected: 0.0`.

    The SDK is not installed in the offline test environment, so `_ensure_sdk`
    is stubbed - the same constraint the `_ensure_sdk` failure tests work under.
    What is pinned is the *order* of the two calls, which is the whole fix.
    """

    def test_the_sdk_is_initialised_before_the_size_is_rounded(self):
        ex = MinValueFakeExchange(order_result=filled(0.00012, 87_000.0))
        broker = self.broker(exchange=None)
        calls = []

        def fake_ensure_sdk():
            # Exactly what the real one does that matters here: cache
            # `szDecimals` from the venue's metadata.
            calls.append(1)
            broker._exchange = ex
            broker._sz_decimals["BTC"] = 5

        broker._ensure_sdk = fake_ensure_sdk
        self.assertEqual(broker._sz_decimals, {}, "must start uncached")

        broker.open_position("BTC", "long", 0.000121, 87_000.0, 3)

        self.assertEqual(calls, [1], "the SDK must be initialised")
        self.assertEqual(len(ex.orders), 1)
        self.assertEqual(
            ex.orders[0][2],
            0.00012,
            "6 decimals is what the venue refuses (Order has invalid size.); "
            "5 is what it takes",
        )

    def test_a_dry_run_never_initialises_the_sdk(self):
        """A dry run has no credentials, and this call moved in front of the
        branch that used to hold it - so it must not become unconditional."""
        broker = Broker(self.cfg, None, self.journal)
        broker._ensure_sdk = lambda: (_ for _ in ()).throw(
            AssertionError("a dry run must not build an exchange client")
        )

        broker.open_position("BTC", "long", 0.00012, 87_000.0, 3)

        self.assertIn("order_simulated", self.journal_text())


class TestSimulatedFees(ExecutionValueTestCase):
    """A simulated fill carries a cost, and only a simulated one.

    The paper record used to be gross, and the open question of this project is
    whether the edge survives the cost - so the record was structurally unable
    to answer it, in the optimistic direction. The fee is RECORDED, not
    deducted: `realised_pnl` and `discipline.record_outcome` still read the
    gross figure, which is a separate decision with its own consequences.
    """

    def dry_broker(self, config=None):
        broker = Broker(config or self.cfg, None, self.journal)
        self.assertTrue(broker.dry_run, "these tests are about the dry path")
        broker._sz_decimals["BTC"] = 5
        return broker

    def event(self, kind):
        return next(e for e in self.events() if e.get("kind") == kind)

    def test_a_simulated_entry_is_charged_the_configured_fee(self):
        broker = self.dry_broker()

        broker.open_position("BTC", "long", 0.00012, 87_000.0, 3)

        entry = self.event("order_simulated")
        self.assertAlmostEqual(entry["notional"], 10.44, places=2)
        self.assertEqual(entry["fee_bps"], self.cfg.execution.fee_bps)
        self.assertAlmostEqual(entry["fee_usd"], 10.44 * 7.2 / 10_000.0, places=4)

    def test_the_two_legs_are_charged_on_their_own_notional(self):
        """Not twice the entry. A position that moved pays the fee on two
        different notionals, which is also why the round trip cannot be priced
        once at entry."""
        broker = self.dry_broker()
        position = broker.open_position("BTC", "long", 0.00012, 87_000.0, 3)

        broker.close_position(position, 95_000.0, "test")

        closed = self.event("close_simulated")
        self.assertAlmostEqual(
            closed["entry_fee_usd"], 0.00012 * 87_000.0 * 7.2e-4, places=4
        )
        self.assertAlmostEqual(
            closed["fee_usd"], 0.00012 * 95_000.0 * 7.2e-4, places=4
        )
        self.assertGreater(
            closed["fee_usd"],
            closed["entry_fee_usd"],
            "the exit notional is larger, so its fee is too",
        )

    def test_the_net_figure_is_recorded_and_the_return_stays_gross(self):
        """The asymmetry is the point of this change: the journal can be read
        net, and nothing that ACTS on the number moved."""
        broker = self.dry_broker()
        position = broker.open_position("BTC", "long", 0.00012, 87_000.0, 3)

        returned = broker.close_position(position, 95_000.0, "test")

        closed = self.event("close_simulated")
        gross = (95_000.0 - 87_000.0) * 0.00012
        self.assertAlmostEqual(returned, gross, places=6)
        self.assertAlmostEqual(closed["pnl"], gross, places=4)
        self.assertAlmostEqual(
            closed["pnl_net"],
            gross - closed["entry_fee_usd"] - closed["fee_usd"],
            places=4,
        )
        self.assertLess(closed["pnl_net"], closed["pnl"])

    def test_a_real_fill_is_not_charged_a_simulated_fee(self):
        """The venue charges its own there and the raw result carries it, so an
        estimate on top would be a second, invented charge."""
        ex = FakeExchange(order_result=filled(0.00012, 87_000.0))
        broker = self.broker_for(ex)

        position = broker.open_position("BTC", "long", 0.00012, 87_000.0, 3)

        self.assertEqual(position.entry_fee_usd, 0.0)
        self.assertNotIn("fee_usd", self.event("order_submitted"))

    def test_a_zero_fee_leaves_the_net_equal_to_the_gross(self):
        """The whole thing has to be switchable off."""
        cfg = load_config(
            self._write(CONFIG.replace("  dry_run: true", "  fee_bps: 0\n  dry_run: true"))
        )
        broker = self.dry_broker(cfg)
        position = broker.open_position("BTC", "long", 0.00012, 87_000.0, 3)

        broker.close_position(position, 95_000.0, "test")

        closed = self.event("close_simulated")
        self.assertEqual(closed["fee_usd"], 0.0)
        self.assertEqual(closed["entry_fee_usd"], 0.0)
        self.assertEqual(closed["pnl_net"], closed["pnl"])

    def test_a_negative_fee_is_refused(self):
        """It would credit the paper record, which is the one direction this
        field must never move."""
        with self.assertRaises(ConfigError):
            load_config(
                self._write(
                    CONFIG.replace("  dry_run: true", "  fee_bps: -1.0\n  dry_run: true")
                )
            )

    def test_the_entry_fee_survives_a_state_round_trip(self):
        """A restart must not turn a charged entry into a free one - and the
        exit reads this value off the position, not off the config."""
        broker = self.dry_broker()

        position = broker.open_position("BTC", "long", 0.00012, 87_000.0, 3)
        restored = Position.from_dict(position.to_dict())

        self.assertGreater(restored.entry_fee_usd, 0.0)
        self.assertAlmostEqual(restored.entry_fee_usd, position.entry_fee_usd, places=6)

    def test_a_state_file_from_before_this_field_charges_nothing(self):
        """Honest degradation: that entry was never charged, so its exit must
        not report a fee it did not collect."""
        data = self.position().to_dict()
        data.pop("entry_fee_usd")

        self.assertEqual(Position.from_dict(data).entry_fee_usd, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
