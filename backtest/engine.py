"""The replay engine.

Design principle: this file contains no strategy logic. It drives the clock,
supplies bars, and routes data through the SAME `Synthesizer`, `Discipline`
and `compute_size` objects the live agent uses. If the backtest reimplemented
the rules, the two could silently diverge and the result would mean nothing.

What is modelled faithfully:
  * factor computation, on closed bars only
  * discipline gates: cooldown, daily cap, loss streak, macro filter
  * ATR-based position sizing
  * taker fees on both legs
  * real hourly funding, charged for the holding period
  * intra-bar stop and target touches (the live agent polls every few minutes,
    so it would catch them too)

What is NOT modelled, and why it matters:
  * the smart money factor. The public API exposes no historical whale
    positions, so 40% of the live model simply cannot be replayed. This is
    reported prominently rather than quietly dropped.
  * historical open interest, for the same reason.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from agent.config import BotConfig
from agent.discipline import Discipline, bucket_block_reason
from agent.execution import (
    ExecutionError,
    Position,
    compute_size,
    is_reversal_signal,
)
from agent.factors.base import FactorScore
from agent.factors.smart_money import SmartMoneySnapshot
from agent.synthesizer import LONG, SHORT, Synthesizer

from .data import HistoricalData
from .historical_smart_money import HistoricalSmartMoney
from .market import BacktestMarket, Bar

# Taker fee in basis points. Hyperliquid charges 0.035% base tier. Modelling
# taker on both legs is the conservative choice.
DEFAULT_TAKER_FEE_BPS = 3.5


class Clock:
    """Mutable clock handed to the discipline layer."""

    def __init__(self, now: float | None = None) -> None:
        self._now = time.time() if now is None else now

    def set(self, seconds: float) -> None:
        self._now = seconds

    def __call__(self) -> float:
        return self._now


class NullSmartMoney:
    """Stands in for the whale factor when no history exists.

    Reporting zero confidence is the honest answer, not a neutral 50: the
    backtest genuinely has no information about whale positioning. The
    synthesizer then redistributes that weight across the factors that do
    have data, and the run is labelled accordingly.
    """

    def __init__(self, reason: str = "no historical whale data available") -> None:
        self.reason = reason

    def snapshot(self, symbol: str, force: bool = False) -> SmartMoneySnapshot:
        return SmartMoneySnapshot(symbol=symbol.upper(), wallets_sampled=0)

    def peek(self, symbol: str) -> None:
        return None

    def evaluate(self, symbol: str, snapshot=None) -> FactorScore:
        return FactorScore(
            name="smart_money",
            score=50.0,
            confidence=0.0,
            reasons=[self.reason],
        )


@dataclass
class Trade:
    """A completed round trip."""

    symbol: str
    side: str
    entry_time: int
    entry_price: float
    size: float
    notional: float
    exit_time: int = 0
    exit_price: float = 0.0
    gross_pnl: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    net_pnl: float = 0.0
    exit_reason: str = ""
    entry_score: float = 0.0
    risk_usd: float = 0.0

    @property
    def r_multiple(self) -> float:
        """PnL in units of the risk budget, when one was recorded."""
        if self.risk_usd <= 0:
            return 0.0
        return self.net_pnl / self.risk_usd

    @property
    def holding_hours(self) -> float:
        return (self.exit_time - self.entry_time) / 3_600_000


@dataclass
class BacktestResult:
    symbol: str
    start_ms: int
    end_ms: int
    initial_equity: float
    final_equity: float
    trades: list[Trade] = field(default_factory=list)
    blocked: dict[str, int] = field(default_factory=dict)
    signals_seen: int = 0
    signals_actionable: int = 0
    equity_curve: list[tuple[int, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fees_paid: float = 0.0
    funding_paid: float = 0.0


class Backtester:
    """Replays history through the live decision code."""

    def __init__(
        self,
        config: BotConfig,
        datasets: dict[str, HistoricalData],
        initial_equity: float = 1000.0,
        taker_fee_bps: float = DEFAULT_TAKER_FEE_BPS,
        include_smart_money: bool = False,
        whale_history: dict | None = None,
        wallet_persistence: dict[str, int] | None = None,
        whale_max_age_ms: int | None = None,
    ) -> None:
        self.cfg = config
        self.symbol = config.symbol.upper()
        self.datasets = datasets
        self.initial_equity = initial_equity
        self.fee_rate = taker_fee_bps / 10_000.0

        self.market = BacktestMarket(datasets, entry_interval=config.indicators.entry_timeframe)
        # Seed the clock from the data, not wall time. The discipline layer
        # anchors its rolling day window to this, and a wall-clock anchor would
        # sit in the future relative to a historical replay.
        self.clock = Clock(datasets[self.symbol].start_ms / 1000.0)

        # Initialised before anything can append to them.
        self.warnings: list[str] = []

        # The macro filter asks the market for BTC bars, whichever market is
        # being traded, and `safe_candles` swallows the lookup failure. So a run
        # whose datasets omit BTC loses the gate entirely and silently: the
        # filter blocks nothing, the rejection counts show no `macro filter`
        # rows, and the equity curve just looks different.
        #
        # Measured on a 170-day window: ETH saw 0 blocks instead of 101 and SOL
        # 0 instead of 218, and SOL's return went from -0.23% to +1.88% depending
        # only on whether BTC happened to be in the dict. The CLI loads BTC
        # alongside any market for exactly this reason; this warning is for
        # every other caller.
        if (
            config.discipline.btc_trend_filter
            and "BTC" not in {k.upper() for k in datasets}
        ):
            self.warnings.append(
                "BTC was not loaded, so the macro filter that blocks "
                "counter-trend entries against the BTC "
                f"{config.discipline.btc_trend_timeframe} Supertrend could not "
                "read it and did not run. Load BTC alongside the traded market "
                "or the gate is off."
            )

        # The live pipeline, unmodified.
        self.synth = Synthesizer(config, self.market)
        self.smart_money_source = None

        if whale_history:
            # Reconstructed whale positions let the 40% factor actually run,
            # scored by the live factor's own evaluate().
            # The staleness window belongs to the data source, not to the
            # engine. It defaults to the funding-derived value (hourly records,
            # so six hours tolerates a few missed settlements), but daily
            # snapshots need a window measured in days - at six hours the book
            # reads empty for all but one bar in twenty-four, and it does so
            # silently, with the factor simply reporting no positions.
            smart_money_kwargs: dict = {
                "price_lookup": lambda _ts: self.market.mid_price(self.symbol),
                "persistence": wallet_persistence or {},
                "window_count": len(config.smart_money.windows),
            }
            if whale_max_age_ms is not None:
                smart_money_kwargs["max_age_ms"] = int(whale_max_age_ms)
            self.smart_money_source = HistoricalSmartMoney(
                whale_history, **smart_money_kwargs
            )
            self.synth.smart_money = self.smart_money_source

            # Whether the wallet-quality confidence term can be computed depends
            # on the source, not on the fact that whale history exists. The
            # funding-derived series carry no entry price; the Reservoir
            # snapshots do. Reporting the funding case unconditionally told
            # every snapshot-backed run that a term was excluded when it was
            # not - a claim about the run that was simply false.
            with_entry = sum(1 for s in whale_history.values() if s.has_pnl)
            if with_entry == len(whale_history) and whale_history:
                self.warnings.append(
                    f"smart money factor INCLUDED from {len(whale_history)} "
                    "reconstructed wallets, each with an entry price, so the "
                    "wallet-quality confidence term comes from real unrealised "
                    "PnL - the same term the live agent uses."
                )
            elif with_entry:
                self.warnings.append(
                    f"smart money factor INCLUDED from {len(whale_history)} "
                    f"reconstructed wallets; {with_entry} carry an entry price. "
                    "A snapshot with any position from a wallet lacking one "
                    "reports PnL as unavailable and the quality term is dropped."
                )
            else:
                self.warnings.append(
                    f"smart money factor INCLUDED from {len(whale_history)} "
                    "reconstructed wallets, none with an entry price, so the "
                    "wallet-quality confidence term is excluded and the "
                    "remaining weights are renormalised."
                )

            # No claim is made here about how the wallet set was chosen. The
            # engine receives a dict of series and cannot tell whether it came
            # from a PnL ranking or a size filter - and an earlier version
            # asserted "selected by position size and persistence" regardless,
            # which became false the moment a leaderboard-backed source was
            # added. The loaders state their own basis, and the CLI prints it.
            #
            # What the engine *can* verify is whether entry prices are present,
            # so that is the only thing it reports.
        elif not include_smart_money:
            self.synth.smart_money = NullSmartMoney(self._smart_money_gap_reason())

        self.discipline = Discipline(config, self.market, now_fn=self.clock)

        self.equity = initial_equity
        self.position: Position | None = None
        self.trades: list[Trade] = []
        self.blocked: dict[str, int] = {}
        self.equity_curve: list[tuple[int, float]] = []
        self.signals_seen = 0
        self.signals_actionable = 0
        self.fees_paid = 0.0
        self.funding_paid = 0.0

        # Only neutralise the alignment gate when the factor genuinely has no
        # data. With reconstructed history the gate can stay on, which is the
        # whole point of collecting it.
        if not include_smart_money and not whale_history:
            self._apply_smart_money_mitigation()

    # ------------------------------------------------------------------
    def _smart_money_gap_reason(self) -> str:
        return (
            "BACKTEST LIMITATION: the public Hyperliquid API exposes no "
            "historical whale positions, so the smart money factor "
            "(40% of the live weight) cannot be replayed"
        )

    def _apply_smart_money_mitigation(self) -> None:
        """Disable the alignment gate, and say so loudly.

        `require_smart_money_alignment` blocks every trade when the factor has
        no data, which is correct live behaviour but would make a backtest
        produce zero trades. Turning it off is a real change to the strategy,
        so it is recorded as a warning rather than done silently.
        """
        if self.cfg.discipline.require_smart_money_alignment:
            self.cfg.discipline.require_smart_money_alignment = False
            self.warnings.append(
                "smart money alignment gate DISABLED for the backtest: without "
                "whale data it would block 100% of trades. The live agent keeps "
                "this gate on."
            )
        self.warnings.append(self._smart_money_gap_reason())

    # ------------------------------------------------------------------
    def run(self, progress=None) -> BacktestResult:
        times = self.market.bar_times(self.symbol, self.cfg.indicators.entry_timeframe)
        if not times:
            raise RuntimeError(f"no bars available for {self.symbol}")

        # Trade only inside the requested window; earlier bars are warmup for
        # indicators and must not generate orders.
        trade_start = self.datasets[self.symbol].start_ms
        started = False

        for ts in times:
            self.market.set_now(ts)

            if ts < trade_start:
                # Warmup: factors stay primed so the first traded bar has full
                # indicator history, but no order may be placed and the
                # discipline clock does not advance.
                continue

            if not started:
                started = True
                if progress:
                    progress(
                        f"  trading from {time.strftime('%Y-%m-%d', time.gmtime(ts / 1000))}"
                    )

            self.clock.set(ts / 1000)
            if self.smart_money_source is not None:
                self.smart_money_source.set_now(ts)

            bar = self.market.bar_at(
                self.symbol, self.cfg.indicators.entry_timeframe, ts
            )
            if bar is None:
                continue

            # Exits first: protecting capital outranks new entries, matching
            # the live loop's ordering.
            if self.position is not None:
                self._manage(bar, ts)

            # The signal is processed on EVERY bar, not only when flat. The
            # live cycle generates a signal every time regardless of whether a
            # position is open, because an opposite signal closes it. Gating
            # this on `position is None` meant a position could only ever exit
            # via a stop or a target, which is not what the live agent does.
            self._process_signal(bar, ts)

            self.equity_curve.append((ts, self._marked_equity(bar)))

        if self.position is not None:
            # Close at the final bar's close so the result is a realised figure.
            last = self.market.bar_at(
                self.symbol, self.cfg.indicators.entry_timeframe, times[-1]
            )
            if last is not None:
                self._close(last.time, last.close, "end of backtest", last)

        result = BacktestResult(
            symbol=self.symbol,
            start_ms=trade_start,
            end_ms=times[-1],
            initial_equity=self.initial_equity,
            final_equity=self.equity,
            trades=self.trades,
            blocked=self.blocked,
            signals_seen=self.signals_seen,
            signals_actionable=self.signals_actionable,
            equity_curve=self.equity_curve,
            warnings=self.warnings,
            fees_paid=self.fees_paid,
            funding_paid=self.funding_paid,
        )
        return result

    # ------------------------------------------------------------------
    def _generate_signal(self):
        """Signal source for entries.

        Extracted as a hook so the random-entry benchmark can substitute pure
        noise while keeping every other element of the pipeline (gates,
        sizing, exits, fees, funding) identical. That isolates the question
        that matters: is the direction call informative?
        """
        return self.synth.generate(self.symbol)

    def _process_signal(self, bar: Bar, ts: int) -> None:
        """Generate, gate and act on this bar's signal.

        Mirrors the live cycle's ordering exactly. The signal is produced on
        every bar regardless of whether a position is open, because an opposite
        signal closes the open one.

        This was previously gated on `position is None`, which meant a position
        could only ever exit via a stop or a target. The live agent would have
        reversed out of it on the first opposite signal, so the backtest's
        holding periods were an artefact of the simulation rather than of the
        strategy.
        """
        signal = self._generate_signal()
        self.signals_seen += 1

        open_position = self.position is not None
        # One shared definition of "reversal", used by the live cycle too. The
        # discipline layer exempts reversals from the daily cap and cooldown,
        # exactly as it does live - no rule to reimplement here.
        is_reversal = is_reversal_signal(self.position, signal)

        signal = self.discipline.evaluate(
            signal, 1 if open_position else 0, is_reversal
        )

        if signal.blocked_by:
            key = bucket_block_reason(signal.blocked_by)
            self.blocked[key] = self.blocked.get(key, 0) + 1
            return

        if signal.action not in (LONG, SHORT):
            self.blocked["neutral score"] = self.blocked.get("neutral score", 0) + 1
            return

        self.signals_actionable += 1

        price = bar.open  # decided on closed bars, filled at the next open

        # Close the old side BEFORE sizing the new one, matching the live
        # ordering: if sizing then fails the account ends up flat rather than
        # holding two exposures.
        if is_reversal and self.position is not None:
            self._close(ts, price, f"reversal to {signal.action}", bar)

        try:
            sizing = compute_size(
                self.market, self.cfg, self.symbol, signal.action, price, self.equity
            )
        except ExecutionError as exc:
            # A sizing *decision* that says no - a size that rounds to zero, a
            # price that cannot be sized against. That is a skip, not a fault.
            self.blocked[f"sizing failed: {exc}"] = (
                self.blocked.get(f"sizing failed: {exc}", 0) + 1
            )
            return
        # Anything else propagates. A blanket `except Exception` here once
        # turned a mismatched keyword argument into 343 counted "skips" and
        # zero trades, with no error anywhere: a bug presented itself as a
        # strategy that simply found nothing to do.

        if sizing.size <= 0:
            self.blocked["zero size"] = self.blocked.get("zero size", 0) + 1
            return

        side = LONG if signal.action == LONG else SHORT
        self.position = Position(
            symbol=self.symbol,
            side=side,
            size=sizing.size,
            entry_price=price,
            notional=sizing.notional,
            leverage=self.cfg.risk.leverage,
            stop_price=sizing.stop_price,
            take_profit_price=sizing.take_profit_price,
            opened_ts=ts / 1000,
            entry_score=signal.score,
            is_dry_run=True,
            risk_usd=sizing.risk_usd,
            trailing_distance=sizing.trailing_distance,
            trailing_activation=sizing.trailing_activation,
            best_price=price,
        )
        self.discipline.record_trade(self.symbol)

    # ------------------------------------------------------------------
    def _manage(self, bar: Bar, ts: int) -> None:
        """Check stops and targets against this bar's range."""
        pos = self.position
        if pos is None:
            return

        # When both the stop and the target sit inside one bar's range, the
        # order of the two touches is unknowable. Assuming the stop is hit
        # first is the pessimistic convention, and avoids flattering results.
        if pos.stop_price is not None:
            # A stop-market order fills at the stop, or WORSE when the bar
            # gapped past it: a bar that OPENS beyond the stop cannot fill at
            # the stop at all. Assuming the exact stop price whenever it was
            # merely touched flatters every stopped-out trade, so the fill is
            # clamped to the bar's open when the gap went against us.
            if pos.side == LONG and bar.low <= pos.stop_price:
                self._close(ts, min(pos.stop_price, bar.open), "stop loss", bar)
                return
            if pos.side == SHORT and bar.high >= pos.stop_price:
                self._close(ts, max(pos.stop_price, bar.open), "stop loss", bar)
                return

        if pos.take_profit_price is not None:
            if pos.side == LONG and bar.high >= pos.take_profit_price:
                self._close(ts, pos.take_profit_price, "take profit", bar)
                return
            if pos.side == SHORT and bar.low <= pos.take_profit_price:
                self._close(ts, pos.take_profit_price, "take profit", bar)
                return

        # Backstop that applies when no explicit stop was configured, mirroring
        # the live agent's fallback.
        if pos.stop_price is None:
            hard = self.cfg.safety.hard_stop_pct
            if pos.pnl_pct(bar.close) <= -abs(hard):
                self._close(ts, bar.close, "safety stop", bar)
                return

        # Ratchet the trailing stop LAST, and only if the position survived the
        # checks above.
        #
        # Order matters here. Raising the stop on this bar's high and then
        # testing it against this bar's low assumes the high came first - and on
        # a down bar it did not. That produces a same-bar exit at a price the
        # market never offered, which is the flattering direction. Trailing on
        # the extremes of a bar whose stop has already been tested makes the
        # trail take effect from the next bar, which is the honest reading of
        # what a resting order could have done.
        self.position.ratchet_stop(bar.high, bar.low)

    # ------------------------------------------------------------------
    def _close(self, ts: int, price: float, reason: str, bar: Bar) -> None:
        pos = self.position
        if pos is None:
            return

        size = pos.size
        entry_notional = pos.entry_price * size
        exit_notional = price * size

        gross = pos.unrealized_pnl(price)

        # Taker fee on both legs.
        fees = (entry_notional + exit_notional) * self.fee_rate
        self.fees_paid += fees

        # Real funding over the holding period. Positive funding means longs
        # pay shorts, so a long is charged and a short is credited.
        dataset = self.datasets[self.symbol]
        rate_sum = sum(
            rate
            for _, rate in dataset.funding_between(
                int(pos.opened_ts * 1000), ts
            )
        )
        funding = rate_sum * entry_notional * (1.0 if pos.side == LONG else -1.0)
        self.funding_paid += funding

        net = gross - fees - funding
        self.equity += net

        trade = Trade(
            symbol=self.symbol,
            side=pos.side,
            entry_time=int(pos.opened_ts * 1000),
            entry_price=pos.entry_price,
            size=size,
            notional=entry_notional,
            exit_time=ts,
            exit_price=price,
            gross_pnl=gross,
            fees=fees,
            funding=funding,
            net_pnl=net,
            exit_reason=reason,
            entry_score=pos.entry_score,
            risk_usd=pos.risk_usd,
        )
        self.trades.append(trade)
        self.discipline.record_outcome(net)
        self.position = None

    # ------------------------------------------------------------------
    def _marked_equity(self, bar: Bar) -> float:
        if self.position is None:
            return self.equity
        return self.equity + self.position.unrealized_pnl(bar.close)
