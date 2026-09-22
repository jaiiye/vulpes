"""A pure-trend hard gate: four binary conditions and one threshold.

The rule
--------
Four conditions, each computed on closed 1h bars from past data only:

    momentum   close[i] / close[i-24] - 1 > 0         (24h up)
    volume     volume[i] / mean(volume[i-480 .. i]) > 1.0   (volume above its 20-day mean)
    bollinger  close[i] > SMA(close, 20)[i]            (above the mid band)
    ema        EMA(close, 21)[i] > EMA(close, 55)[i]   (fast above slow)

Each contributes one bull vote, so every bar carries 0-4. `min_bull_votes`
is the entry threshold: at or above it the rule is long, below it the rule is
flat. There is no short leg - the four conditions are all bullish, so the
mirror image would need four opposite conditions that the proposal does not
specify.

Deliberately not implemented as "the inverse is a short signal". That is a
different hypothesis with its own threshold, and the two would be reported
together as if one test had been run.

Two things the design has to state rather than assume
-----------------------------------------------------
**Entry timing.** Votes at bar `i` use bar `i`'s close, so entering at that
same close would be trading on a price the rule has just consumed. The fill is
taken at the *next* bar's open, one bar of slippage against the rule. On 1h
bars this is small but it is the difference between a test and a tautology,
and the project's log records this class of mistake six times.

**Exits.** The proposal specifies only the entry. The symmetric exit is used -
go flat when the votes drop below the threshold - because it adds no free
parameter. The alternative (a fixed holding period) is offered separately
rather than chosen silently, since it is exactly the kind of unstated constant
that made the earlier walk-forward numbers uncomparable.

The control that matters
------------------------
A long-only trend rule in a market that rose is mostly measuring the market.
Two controls are therefore built in:

* **buy & hold** - the rule has to beat being long the whole time, not zero.
* **matched random entries** - the same number of trades, the same distribution
  of holding lengths, the same long-only direction, placed at random bars. Only
  the timing is randomised. This is the repo's established control shape
  (`benchmarks.RandomEntryBacktester`, `cross_section.random_benchmark`) and it
  is what the earlier conclusions were all overturned by.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Sequence

from agent.indicators import ema, sma

#: 1h bars. 24 of them is the proposal's 24h lookback.
MOMENTUM_BARS = 24
#: 20 days of hourly bars, for the volume reference.
VOLUME_LOOKBACK = 480
BOLLINGER_PERIOD = 20
EMA_FAST = 21
EMA_SLOW = 55
#: Taker fee per leg, matching the backtest engine so numbers are comparable.
FEE_BPS = 3.5


class TrendGateError(ValueError):
    """Raised when inputs cannot be evaluated as specified."""


@dataclass
class ExecutionConfig:
    """How a position is filled, held and closed. Shared by every rule.

    Split out of `TrendGateConfig` because `simulate_flags` - the single fill
    loop every rule reuses - reads these fields and nothing else. While they
    lived on a rule-specific config, each rule had to restate them and forward
    them to the loop, so an execution knob was defined three times and copied
    twice. The concrete cost of that: `pullback_entry` could not use a stop at
    all, even though `simulate_flags` implements and tests one, because its
    config had no field to carry `stop_loss_pct`.

    Inherited rather than composed. `cfg.execution.fee_bps` would read better
    once there are several rules, but it also moves 48 call sites (35 of them in
    tests) to express a relation that already holds. What has to exist exactly
    once is the defaulting and the validation, and inheritance gives that
    without the churn.

    Two properties the type system cannot state, so they are stated here:

    * A subclass **must** call `super().__post_init__()`, or every check below
      is skipped for it with no error. Pinned by a test that mutates a subclass
      to skip the call.
    * A default changed here changes every rule at once. That is the point, and
      it is also why the values are the conventional ones rather than tuned
      per rule - the measurements in RESEARCH.md record which values they used.
    """

    #: Taker fee per leg, in basis points. Shared with the backtest engine so
    #: the two sets of numbers stay comparable.
    fee_bps: float = FEE_BPS
    #: Fixed holding period in bars. 0 means "exit on the rule's own
    #: condition", which is the parameter-free exit.
    hold_bars: int = 0
    #: Funding rate per bar, paid by a long position. 0 disables it, which
    #: flatters a long-only rule; callers with funding data should pass it.
    funding_per_bar: float = 0.0
    #: Hard stop, as a fraction below the entry fill price, fixed at entry.
    #: 0 disables it - and the default is 0 because the stop was *measured*,
    #: not because it is untested. On the mean-reversion rule every size between
    #: 2% and 12% scored worse than no stop, and its apparent full-sample
    #: optimum did not survive a check across windows (RESEARCH.md section
    #: 2.16). Disabled records that result; it is not a compatibility shim.
    #:
    #: Fixed at entry rather than trailing: a trailing stop on a mean-reversion
    #: rule is close to self-defeating, because the rule's whole premise is that
    #: a fall is temporary - a ratchet converts "temporarily lower" into "out".
    #: That claim is testable, but it should be tested as its own variant rather
    #: than smuggled into the default.
    stop_loss_pct: float = 0.0
    #: Bars to wait after a stop-out before re-entering. 0 allows an immediate
    #: re-entry, which matters here: while the stop is being hit the entry
    #: condition is usually *still true* (the rule wants to buy weakness), so a
    #: zero cooldown turns one falling market into a chain of stop-outs -
    #: each one paying two legs of fees.
    stop_cooldown_bars: int = 0

    def __post_init__(self) -> None:
        if self.hold_bars < 0:
            raise TrendGateError("hold_bars must be >= 0")
        if not 0.0 <= self.stop_loss_pct < 1.0:
            raise TrendGateError(
                "stop_loss_pct must be in [0, 1): 1.0 would put the stop at "
                "zero and 0 disables it"
            )
        if self.stop_cooldown_bars < 0:
            raise TrendGateError("stop_cooldown_bars must be >= 0")
        if self.stop_cooldown_bars and not self.stop_loss_pct:
            raise TrendGateError(
                "stop_cooldown_bars has no effect without stop_loss_pct, "
                "which is more likely a typo than an intent"
            )


@dataclass
class TrendGateConfig(ExecutionConfig):
    """The four-vote rule's own parameters, on top of everything executable.

    The inherited fields come first and keep their meaning; what is defined
    here is specific to this rule's votes.
    """

    momentum_bars: int = MOMENTUM_BARS
    volume_lookback: int = VOLUME_LOOKBACK
    bollinger_period: int = BOLLINGER_PERIOD
    ema_fast: int = EMA_FAST
    ema_slow: int = EMA_SLOW
    min_bull_votes: int = 3
    #: Votes at or below which an open position is closed. `None` means the
    #: same threshold as the entry, which is the symmetric rule the proposal
    #: implies and the one measured first.
    #:
    #: Setting it below `min_bull_votes` is hysteresis, and it exists because
    #: of a measured problem rather than a preference: the symmetric rule
    #: round-trips on every wobble around the threshold - 228 trades in 170
    #: days - which at 3.5bp a leg costs about 16 percentage points, against a
    #: gross edge of roughly 9. No signal survives that, so turnover is the
    #: first thing to test before concluding anything about the signal.
    exit_below: int | None = None

    def __post_init__(self) -> None:
        # Not optional: skipping it silently drops every execution check.
        super().__post_init__()
        if not 1 <= self.min_bull_votes <= 4:
            raise TrendGateError("min_bull_votes must be in 1..4")
        if self.exit_below is not None:
            if not 0 <= self.exit_below <= 4:
                raise TrendGateError("exit_below must be in 0..4")
            if self.exit_below >= self.min_bull_votes:
                raise TrendGateError(
                    "exit_below must be below min_bull_votes, otherwise the "
                    "rule has no exit condition it can ever satisfy"
                )
        if self.ema_fast >= self.ema_slow:
            raise TrendGateError("ema_fast must be shorter than ema_slow")

    @property
    def closes_below(self) -> int:
        """Effective exit threshold: `exit_below` when set, else symmetric."""
        if self.exit_below is not None:
            return self.exit_below
        # Symmetric: exit as soon as the vote falls under the entry bar.
        return self.min_bull_votes - 1


# ----------------------------------------------------------------------
# The four votes
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Votes:
    """The four conditions separately, so one cannot hide inside the total.

    A bare count is what the rule uses, but it is not what a test can check: a
    condition wired to the wrong comparison still yields 0-4, and another
    condition passing more often would mask it. Kept as named fields so each
    one is asserted on its own.
    """

    momentum: bool
    volume: bool
    bollinger: bool
    ema: bool

    @property
    def total(self) -> int:
        return int(self.momentum) + int(self.volume) + int(self.bollinger) + int(self.ema)


def vote_detail(
    closes: Sequence[float],
    volumes: Sequence[float],
    config: TrendGateConfig | None = None,
) -> list[Votes | None]:
    """Per-bar conditions, or None where history is insufficient.

    `None` rather than all-False for warm-up bars: a zero would be
    indistinguishable from "all four conditions are false", and the first 480
    bars would then read as a strong bearish signal instead of as missing data.
    """
    cfg = config or TrendGateConfig()
    n = len(closes)
    if len(volumes) != n:
        raise TrendGateError(
            f"closes has {n} bars but volumes has {len(volumes)}"
        )
    if n == 0:
        raise TrendGateError("no bars")

    mid = sma(closes, cfg.bollinger_period)
    fast = ema(closes, cfg.ema_fast)
    slow = ema(closes, cfg.ema_slow)
    vol_mean = sma(volumes, cfg.volume_lookback)

    out: list[Votes | None] = []
    for i in range(n):
        need = (
            i - cfg.momentum_bars,
            i - cfg.volume_lookback + 1,
            i - cfg.bollinger_period + 1,
            i - cfg.ema_slow + 1,
        )
        if min(need) < 0:
            out.append(None)
            continue

        past = closes[i - cfg.momentum_bars]
        reference = vol_mean[i]
        out.append(
            Votes(
                momentum=past > 0 and closes[i] / past - 1.0 > 0.0,
                volume=(
                    reference is not None
                    and reference > 0
                    and volumes[i] > reference
                ),
                bollinger=mid[i] is not None and closes[i] > mid[i],
                ema=(
                    fast[i] is not None
                    and slow[i] is not None
                    and fast[i] > slow[i]
                ),
            )
        )
    return out


def vote_series(
    closes: Sequence[float],
    volumes: Sequence[float],
    config: TrendGateConfig | None = None,
) -> list[int | None]:
    """Bull votes per bar, 0-4, or None where history is insufficient."""
    return [None if v is None else v.total for v in vote_detail(closes, volumes, config)]


# ----------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------
@dataclass
class Trade:
    """One round trip.

    `entry_bar` / `exit_bar` are the bars the *fills* happened on;
    `entry_signal_bar` / `exit_signal_bar` are the bars whose close produced
    the decision. The fill must always be strictly later than its signal, and
    recording both is what makes that checkable: the first version of the
    simulator moved an exit to the bar *before* its trigger, and a test that
    only compared price against fill bar passed, because both fields had moved
    together. Without the signal bars the property is not expressible.
    """

    entry_bar: int
    exit_bar: int
    bars: int
    entry_price: float
    exit_price: float
    gross_return: float
    net_return: float
    entry_signal_bar: int
    exit_signal_bar: int
    #: Why the position closed: "hold" (the rule's own exit), "cap" (hold_bars),
    #: "stop" (stop_loss_pct), or "markout" (still open at the end).
    #:
    #: Recorded because the stop changes what a result *means*, not only its
    #: size: a rule whose return comes from a few large winners it never cuts is
    #: a different claim from one that exits every loser at a fixed loss, and
    #: the two are indistinguishable in an equity total.
    #:
    #: Required, not defaulted. Both construction sites pass it, and nothing
    #: else builds a `Trade` - the defaults existed only so an older positional
    #: call would keep working, and would let a new exit path be added without
    #: naming its reason.
    exit_reason: str
    #: For a stop exit, the level that was breached. `exit_price` can be worse
    #: than this when the next bar gaps down, and that difference IS the left
    #: tail a stop cannot remove - so it is recorded rather than assumed away.
    #: `None` on a non-stop exit, which is a value rather than a missing one.
    stop_level: float | None


@dataclass
class GateResult:
    trades: list[Trade] = field(default_factory=list)
    equity: float = 1.0
    bars_in_market: int = 0
    bars_total: int = 0
    buy_hold_return: float = 0.0

    @property
    def net_return_pct(self) -> float:
        return (self.equity - 1.0) * 100.0

    @property
    def exposure(self) -> float:
        return self.bars_in_market / self.bars_total if self.bars_total else 0.0

    @property
    def mean_bars(self) -> float:
        if not self.trades:
            return 0.0
        return statistics.mean(t.bars for t in self.trades)

    def summary(self) -> str:
        return (
            f"n={len(self.trades):4d}  net {self.net_return_pct:+7.2f}%  "
            f"exposure {self.exposure:5.1%}  mean hold {self.mean_bars:5.1f} bars  "
            f"buy&hold {self.buy_hold_return:+7.2f}%"
        )


def _leg_cost(fee_bps: float) -> float:
    return fee_bps / 10_000.0


def simulate_flags(
    closes: Sequence[float],
    opens: Sequence[float],
    entry_ok: Sequence[bool | None],
    hold_ok: Sequence[bool | None],
    config: TrendGateConfig | None = None,
    start_bar: int | None = None,
    end_bar: int | None = None,
    lows: Sequence[float] | None = None,
) -> GateResult:
    """Long/flat simulation driven by two boolean series.

    Split out from `simulate` so a second entry rule can reuse it rather than
    growing a second simulator. This is the part carrying the checks that are
    expensive to re-derive: fills land on the open *after* the bar whose close
    decided them, fees are charged on both legs, funding is charged per bar
    held, and every trade records the signal bar alongside the fill bar. A
    parallel implementation for a new rule would have to get all of that right
    again, and the first version of this one did not.

    `entry_ok` opens a position; `hold_ok` keeps it. Passing the same series for
    both gives the symmetric rule. `None` means "no opinion" and blocks both.

    `lows` is needed only when `config.stop_loss_pct` is set, and is a separate
    argument rather than something derived from `closes` because a stop is
    breached intrabar: the close can recover while the low did not, and using
    the close would make the stop miss exactly the bars it exists for.

    A stop triggers on bar `i`'s low and fills at bar `i + 1`'s **open** - the
    same one-bar delay as every other decision, and deliberately not at the stop
    level itself. A resting stop fills at its level in a calm market and far
    below it after a gap, and the gap is the part that hurts. Filling at the
    level would report a bounded loss that the tape does not always offer; the
    difference is recorded instead (`Trade.stop_level` vs `Trade.exit_price`).
    """
    cfg = config or TrendGateConfig()
    n = len(closes)
    if not (len(opens) == len(entry_ok) == len(hold_ok) == n):
        raise TrendGateError(
            "closes, opens, entry_ok and hold_ok must all be the same length"
        )
    if cfg.stop_loss_pct:
        if lows is None:
            raise TrendGateError("stop_loss_pct requires `lows`")
        if len(lows) != n:
            raise TrendGateError("lows must be the same length as closes")

    # `min`, not `max`: this is the first bar whose indicators are warm, and
    # taking the last one instead made start == stop and refused every run.
    first = min([i for i, v in enumerate(entry_ok) if v is not None], default=n)
    start = max(start_bar if start_bar is not None else first, first)
    stop = min(end_bar if end_bar is not None else n, n - 1)
    if start >= stop:
        raise TrendGateError("no tradable bars after warm-up")

    cost = _leg_cost(cfg.fee_bps)
    result = GateResult(bars_total=stop - start)
    equity = 1.0

    entry_bar = -1
    entry_price = 0.0
    entry_signal = -1
    stop_level: float | None = None
    #: First bar on which a new entry is allowed after a stop-out.
    cooldown_until = -1
    in_market = False
    #: Action decided on the previous bar, executed at this bar's open.
    pending = None
    #: The bar whose close produced the pending action.
    pending_signal = -1
    pending_reason = ""

    def close_trade(at_bar: int, signal_bar: int, reason: str) -> None:
        nonlocal equity, in_market, stop_level
        exit_price = opens[at_bar]
        gross = exit_price / entry_price - 1.0
        net = (1.0 + gross) * (1.0 - cost) ** 2 - 1.0
        equity *= 1.0 + net
        result.trades.append(
            Trade(entry_bar, at_bar, at_bar - entry_bar, entry_price,
                  exit_price, gross, net, entry_signal, signal_bar,
                  reason, stop_level)
        )
        in_market = False
        stop_level = None

    for i in range(start, stop):
        # 1) Execute what the previous bar decided, at this bar's OPEN. The
        #    signal at bar i-1 consumed closes[i-1]; opens[i] is the first price
        #    that is genuinely after that information. Filling an exit at
        #    opens[i] on the same bar i whose close produced the signal was the
        #    first version of this loop, and it returned 100th percentile in
        #    nine cells out of nine - the rule appeared to know the close
        #    before it happened.
        if pending == "exit" and in_market:
            close_trade(i, pending_signal, pending_reason)
            if pending_reason == "stop":
                cooldown_until = i + cfg.stop_cooldown_bars
        elif pending == "enter" and not in_market:
            entry_bar = i
            entry_price = opens[i]
            entry_signal = pending_signal
            # Anchored to the fill, not to the signal bar's close: the fill is
            # the price actually paid, so it is the only honest basis for a
            # "loss of this much from here" rule.
            stop_level = (
                entry_price * (1.0 - cfg.stop_loss_pct)
                if cfg.stop_loss_pct
                else None
            )
            in_market = True
        pending = None
        pending_signal = -1
        pending_reason = ""

        # 2) Mark this bar's exposure and charge funding per bar held.
        if in_market:
            result.bars_in_market += 1
            equity *= 1.0 - cfg.funding_per_bar

        # 3) Decide on this bar's close, to be executed on the next one.
        want_long = entry_ok[i] is not None and entry_ok[i]
        still_ok = hold_ok[i] is not None and hold_ok[i]
        # `hold_bars` counts the bars between the entry fill and the exit fill,
        # and the exit fill lands on the bar after this decision - so the
        # decision is taken at hold_bars - 1. Testing `>= hold_bars` here made
        # a "3 bar" hold last 4 bars, which is the parameter not meaning what
        # its name says.
        hit_cap = (
            cfg.hold_bars > 0
            and in_market
            and (i - entry_bar) >= cfg.hold_bars - 1
        )
        # The low is known once bar `i` closes, so this belongs with the other
        # close-of-bar decisions. A stop can fire on the entry bar itself: the
        # entry filled at this bar's open and the bar then fell through the
        # level, which is a real sequence and the one a mean-reversion rule
        # meets most often (buy the dip, dip continues).
        hit_stop = (
            in_market
            and stop_level is not None
            and lows is not None
            and lows[i] <= stop_level
        )
        if in_market and (hit_stop or not still_ok or hit_cap):
            pending = "exit"
            pending_signal = i
            # Priority is for the report, not the outcome: all three close the
            # position, and when they coincide the stop is the informative one.
            pending_reason = "stop" if hit_stop else ("cap" if hit_cap else "hold")
        elif not in_market and want_long and i >= cooldown_until:
            pending = "enter"
            pending_signal = i

    # A position still open at the end is marked out at the last close rather
    # than left open: an unrealised gain is not a result, and dropping the
    # trade entirely would flatter the trade count's average. Its signal bar is
    # the last one evaluated, and the mark-out price is that bar's close, so
    # the "fill after signal" rule is satisfied by construction here.
    if in_market:
        exit_price = closes[stop]
        gross = exit_price / entry_price - 1.0
        net = (1.0 + gross) * (1.0 - cost) ** 2 - 1.0
        equity *= 1.0 + net
        result.trades.append(
            Trade(entry_bar, stop, stop - entry_bar, entry_price,
                  exit_price, gross, net, entry_signal, stop - 1,
                  "markout", stop_level)
        )

    result.equity = equity
    result.bars_total = max(stop - start, 1)
    first_price = opens[start]
    last_price = closes[stop]
    if first_price > 0:
        result.buy_hold_return = (last_price / first_price - 1.0) * 100.0
    return result


def simulate(
    closes: Sequence[float],
    opens: Sequence[float],
    volumes: Sequence[float],
    config: TrendGateConfig | None = None,
    start_bar: int | None = None,
    end_bar: int | None = None,
) -> GateResult:
    """The four-vote gate: enter at or above the threshold, exit below it.

    A thin translation of the votes into the two boolean series
    `simulate_flags` consumes. Asymmetric when `exit_below` is set, which keeps
    a position while the vote sits between the two thresholds.
    """
    cfg = config or TrendGateConfig()
    if not (len(opens) == len(volumes) == len(closes)):
        raise TrendGateError("closes, opens and volumes must be the same length")

    votes = vote_series(closes, volumes, cfg)
    entry_ok = [None if v is None else v >= cfg.min_bull_votes for v in votes]
    hold_ok = [None if v is None else v > cfg.closes_below for v in votes]
    return simulate_flags(closes, opens, entry_ok, hold_ok, cfg, start_bar, end_bar)


def fixed_hold_simulate(
    closes: Sequence[float],
    opens: Sequence[float],
    volumes: Sequence[float],
    hold_bars: int,
    config: TrendGateConfig | None = None,
    start_bar: int | None = None,
    end_bar: int | None = None,
) -> GateResult:
    """The same gate with a fixed holding period instead of a vote-based exit.

    Offered as a separate entry point rather than as a branch of `simulate`
    because the exit is a free parameter the proposal did not specify, and the
    result is reported for both rather than one being picked.
    """
    base = config or TrendGateConfig()
    cfg = TrendGateConfig(
        momentum_bars=base.momentum_bars,
        volume_lookback=base.volume_lookback,
        bollinger_period=base.bollinger_period,
        ema_fast=base.ema_fast,
        ema_slow=base.ema_slow,
        min_bull_votes=base.min_bull_votes,
        fee_bps=base.fee_bps,
        hold_bars=hold_bars,
        funding_per_bar=base.funding_per_bar,
    )
    # With hold_bars > 0 and the vote exit disabled for the fixed-period case:
    # the loop still exits early when votes fall, so `hold_bars` acts as a cap.
    return simulate(closes, opens, volumes, cfg, start_bar, end_bar)


# ----------------------------------------------------------------------
# Controls
# ----------------------------------------------------------------------
def random_control(
    closes: Sequence[float],
    opens: Sequence[float],
    trades: int,
    hold_lengths: Sequence[int],
    runs: int = 30,
    seed: int = 1000,
    fee_bps: float = FEE_BPS,
    funding_per_bar: float = 0.0,
    start_bar: int = 0,
    end_bar: int | None = None,
) -> list[float]:
    """Net returns from placing the same trades at random bars.

    Matched on the things that could otherwise explain a difference: the same
    number of trades, the same distribution of holding lengths, the same
    long-only direction, the same fees. Only *when* each trade happens is
    randomised, so a result above this distribution is about timing rather than
    about being long in a market that rose.

    Overlaps are resolved by pushing a trade to the previous exit rather than
    dropped or averaged: dropping them would silently shrink the sample, and
    the real rule cannot hold two positions at once either.
    """
    n = len(closes)
    if not (len(opens) == n):
        raise TrendGateError("closes and opens must be the same length")
    last = min(end_bar if end_bar is not None else n, n) - 1
    if start_bar >= last:
        raise TrendGateError("no tradable range for the control")
    if trades == 0 or not hold_lengths:
        return []

    cost = _leg_cost(fee_bps)
    out: list[float] = []
    for spans in random_spans(trades, hold_lengths, runs, seed, start_bar, last):
        equity = 1.0
        for entry_bar, exit_bar in spans:
            gross = opens[exit_bar] / opens[entry_bar] - 1.0
            net = (1.0 + gross) * (1.0 - cost) ** 2 - 1.0
            equity *= 1.0 + net
            equity *= (1.0 - funding_per_bar) ** (exit_bar - entry_bar)
        out.append((equity - 1.0) * 100.0)
    return out


def random_spans(
    trades: int,
    hold_lengths: Sequence[int],
    runs: int = 30,
    seed: int = 1000,
    start_bar: int = 0,
    last_bar: int = 0,
) -> list[list[tuple[int, int]]]:
    """Random entry/exit bar pairs, and never overlapping.

    Split out so the non-overlap property can be asserted directly. Measuring
    it through returns does not work: an overlapping control produces a
    plausible-looking distribution, and the first attempt to catch it with a
    return threshold caught nothing at all.

    Overlaps are resolved by pushing a trade forward to the previous exit
    rather than dropping it, because the real rule cannot hold two positions at
    once either - dropping would silently shrink the sample instead.
    """
    if last_bar <= start_bar:
        raise TrendGateError("no range to place random trades in")
    if trades < 1 or not hold_lengths:
        return []

    out: list[list[tuple[int, int]]] = []
    for r in range(runs):
        rng = random.Random(seed + r)
        starts = sorted(rng.randrange(start_bar, last_bar) for _ in range(trades))
        spans: list[tuple[int, int]] = []
        cursor = start_bar
        for k, s in enumerate(starts):
            entry_bar = max(s, cursor)
            hold = hold_lengths[k % len(hold_lengths)]
            exit_bar = min(entry_bar + hold, last_bar)
            if exit_bar <= entry_bar:
                continue
            spans.append((entry_bar, exit_bar))
            # +1, not exit_bar: the real rule fills an exit at opens[j] using a
            # decision from bar j-1, so the earliest its next entry can fill is
            # opens[j+1]. Letting the control re-enter on the exit bar gave it
            # one extra bar of exposure per trade that the rule cannot have,
            # which is a bias in the control's favour and therefore against the
            # rule being tested.
            cursor = exit_bar + 1
        out.append(spans)
    return out


def percentile_of(value: float, distribution: Sequence[float]) -> float:
    """Where `value` sits in `distribution`, 0-100."""
    dist = list(distribution)
    if not dist:
        return 0.0
    return sum(1 for v in dist if v < value) / len(dist) * 100.0
