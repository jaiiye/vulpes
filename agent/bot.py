"""The agent loop.

Pipeline per cycle:

    market data -> factor scores -> synthesised signal -> discipline gates
                                                       -> sizing -> execution
                                                       -> exit management

Every stage is journalled so a decision can be reconstructed afterwards,
including the ones that were rejected. The rejection log matters more than the
execution log: Fox beat Ghost Fox purely by rejecting more signals.

Four safety properties this loop must maintain, because their absence caused
real defects:

1. Never assume a position exists. The exchange is the source of truth, and it
   is reconciled before trading starts and after any execution failure.
2. Never leave an unprotected position. If protective orders cannot be
   attached, the position is closed immediately or the agent halts.
3. Never treat a resting order as a fill.
4. Never lose the guardrails. Daily cap, cooldown and loss streak are persisted
   across restarts, because resetting them on restart silently disables the
   only part of this agent with evidence behind it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import BotConfig
from .discipline import Discipline
from .execution import (
    Broker,
    CriticalExecutionError,
    ExecutionError,
    Position,
    is_reversal_signal,
    compute_size,
)
from .market_data import HyperliquidMarket, MarketDataError, mainnet_data_market
from .state import DEFAULT_STATE_PATH, AgentState, StateStore
from .synthesizer import LONG, SHORT, Journal, Signal, Synthesizer


@dataclass
class CycleStats:
    cycles: int = 0
    signals_seen: int = 0
    signals_taken: int = 0
    signals_rejected: int = 0
    exits: int = 0
    realised_pnl: float = 0.0
    consecutive_errors: int = 0
    peak_consecutive_errors: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        rejection_rate = (
            self.signals_rejected / self.signals_seen * 100 if self.signals_seen else 0.0
        )
        return (
            f"{self.cycles} cycles | {self.signals_seen} raw signals | "
            f"{self.signals_taken} taken | {self.signals_rejected} rejected "
            f"({rejection_rate:.0f}% selectivity) | {self.exits} exits | "
            f"realised PnL ${self.realised_pnl:+.2f}"
        )


class FoxAgent:
    """A minimal, disciplined Hyperliquid trading agent."""

    # Consecutive failing cycles before the agent stops itself. Without this a
    # persistent fault (bad key, delisted symbol) spins forever.
    MAX_CONSECUTIVE_ERRORS = 5

    def __init__(
        self,
        config: BotConfig,
        journal_path: str = "logs/journal.jsonl",
        state_path: str | None = DEFAULT_STATE_PATH,
    ) -> None:
        self.cfg = config
        self.journal = Journal(journal_path)
        self.store = StateStore(state_path)
        #: Order routing, account equity, mark price, reconciliation - the
        #: things that genuinely live on whichever venue the orders go to.
        self.market = HyperliquidMarket(
            testnet=config.execution.testnet, base_url=config.execution.base_url
        )
        #: Every decision input: the three factors, the gates, and the ATR
        #: behind the stop distance. Always mainnet, so `testnet: true`
        #: rehearses the mainnet strategy instead of a different one.
        #:
        #: On mainnet this is the *same object* as `self.market`, so nothing
        #: about a mainnet run or the backtest changes. The split only has an
        #: effect when the execution venue is testnet.
        self.data_market = mainnet_data_market(self.market)
        self.synthesizer = Synthesizer(config, self.market, data_market=self.data_market)
        self.discipline = Discipline(config, self.data_market)
        self.broker = Broker(
            config, self.market, self.journal, on_unprotected=self._handle_unprotected
        )
        self.stats = CycleStats()

        self.halted = False
        self.halt_reason = ""

        # Restore persisted state before anything can trade.
        self.state: AgentState = self.store.load() or AgentState()
        self.discipline.restore_payload(self.state)
        self.position = self._restore_position(self.state)
        self.stats.realised_pnl = self.state.realised_pnl
        if self.state.agent_halted:
            self.halted = True
            self.halt_reason = self.state.agent_halt_reason or "halted in a previous run"

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _restore_position(self, state: AgentState) -> Position | None:
        if not state.position:
            return None
        try:
            return Position.from_dict(state.position)
        except ExecutionError as exc:
            self.journal.event(
                "error", message=f"could not restore persisted position: {exc}"
            )
            return None

    def _persist(self) -> bool:
        """Write the guardrails and position to disk. Never raises."""
        ds = self.discipline.state
        self.state.last_trade_ts = dict(ds.last_trade_ts)
        self.state.trades_today = dict(ds.trades_today)
        self.state.day_anchor = ds.day_anchor
        self.state.consecutive_losses = ds.consecutive_losses
        self.state.halted_until = ds.halted_until
        self.state.halt_reason = ds.halt_reason
        self.state.position = self.position.to_dict() if self.position else None
        self.state.realised_pnl = self.stats.realised_pnl
        self.state.agent_halted = self.halted
        self.state.agent_halt_reason = self.halt_reason

        ok = self.store.save(self.state)
        if not ok:
            self.log(
                "WARNING: could not persist state; guardrails will not survive a restart"
            )
        return ok

    def halt(self, reason: str) -> None:
        """Stop opening new positions. Exit management keeps running."""
        if self.halted:
            return
        self.halted = True
        self.halt_reason = reason
        self.log(f"HALTED: {reason}")
        self.journal.event("halted", reason=reason)
        # Persist so a restart cannot be used to bypass the halt.
        self._persist()

    def _handle_unprotected(self, position: Position) -> None:
        """Called by the broker when a position could not be closed at all.

        Track it so exit management keeps trying, and stop opening anything new.
        """
        self.position = position
        self.halt(
            f"UNPROTECTED POSITION {position.side} {position.size} "
            f"{position.symbol} could not be closed; manual intervention required"
        )

    # ------------------------------------------------------------------
    def log(self, message: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {message}", flush=True)

    # ------------------------------------------------------------------
    # Startup reconciliation
    # ------------------------------------------------------------------
    def reconcile_startup(self) -> None:
        """Align tracked state with the exchange before any trading happens.

        Four cases, each of which was previously mishandled by simply starting
        with position=None:

          tracked flat  + exchange flat      -> nothing to do
          tracked flat  + exchange has pos   -> orphan: adopt it
          tracked has   + exchange flat      -> phantom: clear it
          both have, mismatched              -> trust the exchange, alert
        """
        symbol = self.cfg.symbol

        if self.cfg.execution.dry_run:
            if self.position is not None:
                self.log(
                    f"resumed tracked {self.position.side.upper()} "
                    f"{self.position.size:.6g} {symbol} from persisted state"
                )
            return

        try:
            live = self.broker.live_positions()
        except ExecutionError as exc:
            # Cannot verify -> refuse to trade rather than guess.
            raise CriticalExecutionError(
                f"startup reconciliation failed: {exc}. Refusing to trade without "
                "knowing the true position state."
            ) from exc

        live_by_symbol = {p.symbol: p for p in live}
        live_pos = live_by_symbol.get(symbol)

        if self.position is None and live_pos is None:
            self.log("reconciliation: flat on the exchange, nothing to recover")
            return

        if self.position is None and live_pos is not None:
            self.log(
                f"RECONCILED: adopted untracked position {live_pos.side.upper()} "
                f"{live_pos.size:.6g} {symbol} @ {live_pos.entry_price:.4g}"
            )
            self.position = live_pos
            self._warn_if_unprotected(symbol, adopted=True)
            self.journal.event("reconciled_orphan", position=live_pos.to_dict())
            self._persist()
            return

        if self.position is not None and live_pos is None:
            self.log(
                f"RECONCILED: tracked {self.position.side.upper()} "
                f"{self.position.size:.6g} {symbol} no longer exists on the "
                "exchange; clearing"
            )
            self.journal.event("reconciled_phantom", position=self.position.to_dict())
            self.position = None
            self._persist()
            return

        # Every other combination returned above, so both are present here.
        # An explicit guard rather than `assert`: assertions are stripped under
        # `-O`, and the code below would then dereference None.
        if live_pos is None or self.position is None:
            self.log("reconciliation: nothing to reconcile")
            return

        tolerance = max(self.position.size * 0.01, 1e-9)
        if (
            self.position.side != live_pos.side
            or abs(self.position.size - live_pos.size) > tolerance
        ):
            self.log(
                f"RECONCILED: tracked {self.position.side.upper()} "
                f"{self.position.size:.6g} disagrees with exchange "
                f"{live_pos.side.upper()} {live_pos.size:.6g}; adopting the exchange"
            )
            self.journal.event(
                "reconciled_mismatch",
                tracked=self.position.to_dict(),
                live=live_pos.to_dict(),
            )
        else:
            self.log(
                f"reconciliation: tracked position matches the exchange "
                f"({live_pos.side.upper()} {live_pos.size:.6g} {symbol})"
            )

        self.position = live_pos
        self._warn_if_unprotected(symbol)
        self._persist()

        # Positions in other symbols are outside this agent's remit.
        others = [p for p in live if p.symbol != symbol]
        if others:
            detail = ", ".join(f"{p.symbol} {p.side} {p.size:.6g}" for p in others)
            self.log(f"WARNING: unmanaged position(s) in other symbols: {detail}")
            self.journal.event("warn", message=f"unmanaged positions: {detail}")

    def _warn_if_unprotected(self, symbol: str, adopted: bool = False) -> None:
        """Check whether the venue actually holds protective orders."""
        try:
            protected = self.broker.has_protective_orders(symbol)
        except Exception:  # noqa: BLE001 - advisory check
            return
        if not protected:
            prefix = "adopted position" if adopted else "position"
            self.log(
                f"WARNING: {prefix} has no protective orders on the exchange - "
                "only the in-process exit check protects it"
            )
            self.journal.event(
                "warn", message=f"{prefix} for {symbol} has no protective orders"
            )

    # ------------------------------------------------------------------
    # Drawdown guard
    # ------------------------------------------------------------------
    def check_drawdown(self, equity: float) -> str | None:
        """Track peak equity and report a breach of the drawdown limit.

        The watermark is tagged with the basis it was measured on, because the
        two bases are not comparable: live reads the real account, while a dry
        run reads the constant `DRY_RUN_EQUITY_USD` - a number chosen for
        readable sizing, not a fact about the account. Sharing one field between
        them let a dry run halt a live account permanently; the mechanism and
        the measurement are in `AgentState.peak_equity_mode`.

        A dry run is still allowed to halt on drawdown - varying
        `DRY_RUN_EQUITY_USD` between runs is a way to rehearse this guard - it
        just cannot move the live watermark.
        """
        if equity <= 0:
            return None

        mode = "dry_run" if self.cfg.execution.dry_run else "live"
        if self.state.peak_equity_mode and self.state.peak_equity_mode != mode:
            # A watermark from the other basis says nothing about this one, so
            # start a fresh one. This makes the breaker less eager, which is why
            # it happens only on a *known* mismatch: an untagged peak is kept,
            # just below.
            self.state.peak_equity = equity
            self.state.peak_equity_mode = mode
            self._persist()
            return None
        if not self.state.peak_equity_mode:
            # Written before the tag existed. Keep the number - discarding a
            # real watermark would understate risk - and label it as this
            # basis, so the next run in the other mode resets it.
            self.state.peak_equity_mode = mode
            self._persist()

        peak = self.state.peak_equity
        if equity > peak:
            self.state.peak_equity = equity
            self._persist()
            return None

        if peak <= 0:
            self.state.peak_equity = equity
            self._persist()
            return None

        drawdown = (peak - equity) / peak * 100.0
        limit = self.cfg.risk.max_drawdown_pct
        if drawdown >= limit:
            return (
                f"equity drawdown {drawdown:.2f}% from peak ${peak:,.2f} exceeds "
                f"the {limit:g}% limit"
            )
        return None

    # ------------------------------------------------------------------
    def run(self, max_cycles: int | None = None) -> CycleStats:
        """Run the agent loop until interrupted or `max_cycles` is reached."""
        self.log(f"starting agent: {self.cfg.summary()}")
        if self.cfg.execution.dry_run:
            self.log("DRY RUN: no orders will be sent to Hyperliquid")
        else:
            self.log(
                f"LIVE MODE on {'testnet' if self.cfg.execution.testnet else 'MAINNET'}"
            )

        self.journal.event(
            "start",
            config=self.cfg.summary(),
            dry_run=self.cfg.execution.dry_run,
            restored_state=self.state.describe(),
        )
        if self.state.saved_at and self.state.position:
            self.log(f"restored state: {self.state.describe()}")

        try:
            self.reconcile_startup()
        except CriticalExecutionError as exc:
            self.halt(str(exc))
            self.log(self.stats.summary())
            return self.stats

        if self.halted:
            if self.position is None:
                self.log(
                    f"halting immediately: {self.halt_reason}\n"
                    "Delete the state file (logs/agent_state.json) or clear the "
                    "flag to resume trading."
                )
                self.journal.event("stop", reason="halted with no position to manage")
                return self.stats
            self.log(
                f"running in exit-only mode to manage the existing position: "
                f"{self.halt_reason}"
            )

        try:
            while max_cycles is None or self.stats.cycles < max_cycles:
                self.stats.cycles += 1
                try:
                    self.run_cycle()
                    self.stats.consecutive_errors = 0
                except CriticalExecutionError as exc:
                    self.stats.errors.append(str(exc))
                    self.halt(str(exc))
                    break
                except Exception as exc:  # noqa: BLE001 - the loop must survive
                    self.stats.consecutive_errors += 1
                    self.stats.peak_consecutive_errors = max(
                        self.stats.peak_consecutive_errors,
                        self.stats.consecutive_errors,
                    )
                    self.stats.errors.append(str(exc))
                    self.log(
                        f"cycle error "
                        f"({self.stats.consecutive_errors}/"
                        f"{self.MAX_CONSECUTIVE_ERRORS}, continuing): "
                        f"{type(exc).__name__}: {exc}"
                    )
                    self.journal.event(
                        "error",
                        message=str(exc),
                        error_type=type(exc).__name__,
                        consecutive=self.stats.consecutive_errors,
                    )
                    if self.stats.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                        self.halt(
                            f"{self.stats.consecutive_errors} consecutive cycle "
                            f"failures; last error: {exc}"
                        )
                        break

                if max_cycles is not None and self.stats.cycles >= max_cycles:
                    break
                time.sleep(self.cfg.execution.poll_interval_seconds)
        except KeyboardInterrupt:
            self.log("interrupted by user")
        finally:
            self.shutdown()

        return self.stats

    # ------------------------------------------------------------------
    def run_cycle(self) -> None:
        symbol = self.cfg.symbol

        # --- 1. Exit management runs first: protecting capital outranks entries.
        self.manage_open_position(symbol)

        # --- 2. Equity and drawdown, evaluated before any new risk is taken.
        equity = self.broker.account_equity()
        breach = self.check_drawdown(equity)
        if breach:
            self.halt(breach)

        # --- 3. Generate the raw signal.
        signal = self.synthesizer.generate(symbol)
        self.stats.signals_seen += 1

        price = self._mark_price(symbol)
        self.log(signal.describe())

        if self.halted:
            self.log(f"  NO NEW ENTRIES: {self.halt_reason}")
            self.journal.signal(signal, {"price": price, "halted": True})
            self.stats.signals_rejected += 1
            self.summarise_if_due()
            return

        # --- 4. Discipline gates.
        open_count = 1 if self.position else 0
        # One shared definition of "reversal", used by the backtest too.
        is_reversal = is_reversal_signal(self.position, signal)
        signal = self.discipline.evaluate(signal, open_count, is_reversal)

        extra = {"price": price}
        if signal.blocked_by:
            self.stats.signals_rejected += 1
            self.log(f"  REJECTED: {signal.blocked_by}")
            self.journal.signal(signal, extra)
            self.summarise_if_due()
            return

        self.journal.signal(signal, extra)

        if signal.action == "neutral":
            return

        # --- 5. Handle a reversal: close the old side first.
        if is_reversal and self.position is not None:
            if not self.close_position(
                self.position, price, f"reversal to {signal.action}"
            ):
                # The old side is still open. Opening the new side now would
                # overwrite `self.position` and orphan the existing exposure,
                # so abandon this cycle instead.
                self.log("  reversal aborted: could not close the existing position")
                self.journal.event(
                    "warn", message="reversal aborted: closing the old side failed"
                )
                return

        # --- 6. Size and execute.
        try:
            # `data_market`, not `self.market`: this reads candles for the ATR
            # that sets the stop distance, which is a decision input. Sizing on
            # testnet-candle volatility would size a stop the mainnet run would
            # never use. `price` is passed separately and does come from the
            # execution venue, because that is the price actually paid.
            sizing = compute_size(
                self.data_market, self.cfg, symbol, signal.action, price, equity
            )
        except ExecutionError as exc:
            self.log(f"  SIZING FAILED: {exc}")
            self.journal.event("error", message=f"sizing failed: {exc}")
            return

        for note in sizing.notes:
            self.log(f"  sizing: {note}")

        if sizing.size <= 0:
            self.log("  SIZING produced zero size; skipping")
            return

        self.log(
            f"  ENTER {signal.action.upper()} {sizing.size:.6g} {symbol} @ {price:.4g} "
            f"(notional ${sizing.notional:,.2f}, risk ${sizing.risk_usd:.2f})"
        )

        try:
            self.position = self.broker.open_position(
                symbol=symbol,
                side=signal.action,
                size=sizing.size,
                price=price,
                leverage=self.cfg.risk.leverage,
                stop_price=sizing.stop_price,
                take_profit_price=sizing.take_profit_price,
                entry_score=signal.score,
                entry_reasons=signal.reasons,
                trailing_distance=sizing.trailing_distance,
                trailing_activation=sizing.trailing_activation,
            )
        except CriticalExecutionError:
            # The broker already recorded the exposure via on_unprotected.
            raise
        except ExecutionError as exc:
            # Recoverable: no position was opened, or it was unwound.
            self.log(f"  ORDER FAILED: {exc}")
            self.journal.event("error", message=f"order failed: {exc}")
            # Do not trust tracked state after a failed entry.
            self.position = None
            self._persist()
            return

        # The order may have filled at a different size than requested, so the
        # discipline record is written only once the position is confirmed.
        self.discipline.record_trade(symbol)
        self.stats.signals_taken += 1
        self.journal.event(
            "position_opened",
            position=self.position.to_dict(),
            signal_score=signal.score,
        )
        self._persist()

        # --- 7. Spread guard: complain loudly if execution was expensive.
        self.check_liquidity(symbol)
        self.summarise_if_due()

    # ------------------------------------------------------------------
    def manage_open_position(self, symbol: str) -> None:
        """Check exits on the tracked position."""
        if self.position is None:
            return

        price = self._mark_price(symbol)
        reason = self.broker.check_exits(self.position, price)
        if reason:
            self.close_position(self.position, price, reason)
            return

        # Ratchet after the exit check, never before. Raising the stop on this
        # observation and then testing it against the same observation would
        # make the trail fire on the tick that raised it.
        #
        # Live has a mark price, not a bar, so the observed extreme is the point
        # itself. That makes the trail coarser than in a backtest, which sees
        # each bar's high and low - it will give back up to one poll interval of
        # move. Stated rather than hidden: the live trail is a slightly looser
        # version of the one that was measured.
        before = self.position.stop_price
        self.position.ratchet_stop(price, price)
        if self.position.stop_price != before:
            self.log(
                f"  TRAIL: stop moved {before:.4g} -> {self.position.stop_price:.4g} "
                f"(best {self.position.best_price:.4g})"
            )
            self._persist()

        pnl = self.position.unrealized_pnl(price)
        self.log(
            f"holding {self.position.side.upper()} {self.position.size:.6g} {symbol} "
            f"@ {self.position.entry_price:.4g}, mark {price:.4g}, "
            f"unrealised ${pnl:+.2f}"
        )

    def close_position(self, position: Position, price: float, reason: str) -> bool:
        """Close `position`. Returns True only when it is confirmed gone.

        On failure the position stays tracked so reconciliation can resolve it
        later, and the caller must not treat the account as flat.
        """
        try:
            pnl = self.broker.close_position(position, price, reason)
        except ExecutionError as exc:
            # The exchange may or may not have closed it. Do not assume either
            # way: leave the position tracked and let reconciliation resolve it.
            self.log(f"  CLOSE FAILED: {exc}")
            self.journal.event("error", message=f"close failed: {exc}")
            return False

        self.stats.realised_pnl += pnl
        self.stats.exits += 1
        self.discipline.record_outcome(pnl)
        self.log(f"  CLOSED {position.side.upper()} {position.symbol}: {reason} -> ${pnl:+.2f}")
        self.journal.event(
            "position_closed",
            symbol=position.symbol,
            side=position.side,
            pnl=round(pnl, 4),
            reason=reason,
            realised_pnl_total=round(self.stats.realised_pnl, 4),
        )
        self.position = None
        self._persist()
        return True

    # ------------------------------------------------------------------
    def check_liquidity(self, symbol: str) -> None:
        """Warn when the spread is wide enough to eat the edge."""
        try:
            spread = self.market.spread_bps(symbol)
        except Exception:  # noqa: BLE001 - advisory check only
            return
        limit = self.cfg.execution.max_spread_bps
        if spread > limit:
            self.log(
                f"  WARNING: spread {spread:.1f}bp exceeds limit {limit:g}bp; "
                "execution costs are eating into the edge"
            )
            self.journal.event(
                "warn", message=f"spread {spread:.1f}bp > {limit}bp for {symbol}"
            )

    def _mark_price(self, symbol: str) -> float:
        try:
            return self.market.mid_price(symbol)
        except MarketDataError as exc:
            raise MarketDataError(f"cannot price {symbol}: {exc}") from exc

    def summarise_if_due(self) -> None:
        if self.stats.cycles % 10 == 0:
            self.log(f"  -- {self.stats.summary()}")
            self.journal.event(
                "heartbeat",
                stats={
                    "cycles": self.stats.cycles,
                    "taken": self.stats.signals_taken,
                    "rejected": self.stats.signals_rejected,
                    "realised_pnl": round(self.stats.realised_pnl, 4),
                    "peak_equity": round(self.state.peak_equity, 2),
                },
                discipline=self.discipline.snapshot(),
            )

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Persist state and report. Positions are left open by design.

        Force-closing on shutdown would turn every redeploy into a market order,
        which is worse than leaving a stopped position for the next start to
        reconcile.
        """
        self.log("shutting down")
        self._persist()
        if self.position is not None:
            self.log(
                f"position left open: {self.position.side.upper()} "
                f"{self.position.size:.6g} {self.position.symbol} "
                f"@ {self.position.entry_price:.4g}"
            )
        if self.halted:
            self.log(f"agent is HALTED: {self.halt_reason}")
        self.log(self.stats.summary())
        self.journal.event(
            "stop",
            stats={
                "cycles": self.stats.cycles,
                "signals_seen": self.stats.signals_seen,
                "signals_taken": self.stats.signals_taken,
                "signals_rejected": self.stats.signals_rejected,
                "exits": self.stats.exits,
                "realised_pnl": round(self.stats.realised_pnl, 4),
                "peak_consecutive_errors": self.stats.peak_consecutive_errors,
            },
            position=self.position.to_dict() if self.position else None,
            halted=self.halted,
            halt_reason=self.halt_reason,
        )
