"""Mean reversion: RSI oversold and Keltner band stretch, long-only.

Why this direction and not another
----------------------------------
RESEARCH.md section 2.7 measured **one** effect that survived its controls:
cross-sectional reversal, IC -0.041 at 24h, t = -4.1 to -6.1. Everything
time-series came back null or unmeasurable (sections 1-6, 8). Mean reversion is
therefore not a fresh guess - it is the family the one measured effect belongs
to. But the two forms are not the same trade, and the distinction decides the
result:

* **Cross-sectional** - buy what performed *relatively* worst, short what
  performed relatively best. The common market factor cancels by construction,
  which is exactly why this is the one that measured.
* **Time series** - buy when *this* symbol is oversold. No hedge, so the
  common factor is the whole position. In a market that keeps falling, "oversold"
  stays oversold and the rule is long all the way down.

This module implements the time-series form, because that is what RSI and
Keltner bands are normally used for. The cross-sectional form is measured
separately so the two can be compared rather than conflated.

Two rules, two philosophies
---------------------------
**RSI(14)** - bounded oscillator. Entry under `oversold`, exit back above
`exit_level`. Bounded means it cannot go below 0, so a deeply falling market
pins it near 0 and the rule sits long. That is the known failure mode of RSI
mean reversion, not an implementation detail.

**Keltner (EMA20 / ATR10)** - volatility-scaled bands. Entry when the close is
`multiplier` ATRs below the EMA, exit at the EMA. Adaptive rather than fixed:
the band widens when volatility widens, so the same rule means the same thing
in a quiet and a violent market. This is the reason to prefer it to a fixed
percentage band in crypto, where volatility regimes shift by a factor of three.

Defaults are the conventional ones (RSI 30/50, Keltner 20/2.0/10) and are
documented rather than tuned. Choices that experience says matter - the regime
filter, the exit level, the band multiplier - are exposed and reported
separately instead of being picked silently.

What experience says will go wrong
----------------------------------
Mean reversion has a fat left tail: you buy weakness, and occasionally the
weakness continues. So the distribution of outcomes is many small wins and a
few large losses, which makes the **stop** and the **sizing** matter more than
the entry. A long-only version also carries the market, so buy & hold is the
reference rather than zero.

Costs are the other candidate explanation, and this project has already been
misled once by reading a fee bill as a signal (section 2.10). Gross and net are
therefore both reported, always.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agent.indicators import adx, atr, ema, rsi

from backtest.trend_gate import (
    ExecutionConfig,
    GateResult,
    TrendGateError,
    percentile_of,  # noqa: F401  (re-exported for callers)
    random_control,  # noqa: F401
    simulate_flags,
)

RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
RSI_EXIT = 50.0

KELTNER_PERIOD = 20
KELTNER_MULTIPLIER = 2.0
KELTNER_ATR_PERIOD = 10

ADX_PERIOD = 14
ADX_MAX = 25.0


class MeanReversionError(ValueError):
    """Raised when a rule's settings cannot describe a tradeable rule.

    Covers this rule's own parameters only. A bad `fee_bps`, `hold_bars` or
    `stop_loss_pct` raises `TrendGateError` instead, because those fields are
    defined by `ExecutionConfig` and validated there - one definition, one
    error. Wrapping it back into this type was considered and rejected: it
    would be a compatibility shim around a relation that is now uniform, and
    the exception's origin would no longer say where the check lives.
    """


@dataclass
class MeanReversionConfig(ExecutionConfig):
    """This rule's parameters, on top of everything executable.

    Inherits `fee_bps`, `hold_bars`, `funding_per_bar`, `stop_loss_pct` and
    `stop_cooldown_bars` from `ExecutionConfig` rather than restating them.
    Before the split they were declared here, in `TrendGateConfig` and in
    `PullbackConfig`, and each rule copied them into a `TrendGateConfig` to
    hand to `simulate_flags` - three declarations and two copies for one set of
    values.
    """
    # -- rule selection ----------------------------------------------------
    #: "rsi" or "keltner". Two named rules rather than one with a switch buried
    #: in it, because they fail differently (RSI pins at 0 in a crash, Keltner
    #: widens) and the results are reported per rule.
    rule: str = "rsi"

    # -- RSI ---------------------------------------------------------------
    rsi_period: int = RSI_PERIOD
    #: Enter at or below this. 30 is the conventional line.
    oversold: float = RSI_OVERSOLD
    #: Exit once RSI recovers to at least this. 50 is "back to neutral"; a
    #: higher value holds longer and is a different trade.
    exit_level: float = RSI_EXIT

    # -- Keltner -----------------------------------------------------------
    keltner_period: int = KELTNER_PERIOD
    keltner_multiplier: float = KELTNER_MULTIPLIER
    keltner_atr_period: int = KELTNER_ATR_PERIOD

    # -- regime filter -----------------------------------------------------
    #: Optional: only take entries while ADX is below this. Experience says
    #: mean reversion works in ranges and fails in trends, and ADX is the
    #: standard way to tell them apart. Disabled by default (0) so the plain
    #: rule is measured first and the filter is a separate, labelled step.
    adx_max: float = 0.0
    adx_period: int = ADX_PERIOD

    # Execution and stop fields are inherited; see `ExecutionConfig`.

    def __post_init__(self) -> None:
        # Not optional: skipping it silently drops every execution check.
        super().__post_init__()
        if self.rule not in ("rsi", "keltner"):
            raise MeanReversionError("rule must be 'rsi' or 'keltner'")
        if not 0.0 < self.oversold < self.exit_level <= 100.0:
            raise MeanReversionError(
                "need 0 < oversold < exit_level <= 100, or the exit can never "
                "be reached after an entry"
            )
        if self.rsi_period < 2:
            raise MeanReversionError("rsi_period must be >= 2")
        if self.keltner_period < 2 or self.keltner_atr_period < 1:
            raise MeanReversionError("Keltner periods must be >= 2 and >= 1")
        if self.keltner_multiplier <= 0:
            raise MeanReversionError("keltner_multiplier must be > 0")
        if self.adx_max and self.adx_max <= 0:
            raise MeanReversionError("adx_max must be > 0 when enabled")

    @property
    def stop_is_on(self) -> bool:
        return self.stop_loss_pct > 0.0


def keltner_lines(
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    config: MeanReversionConfig | None = None,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """(upper, mid, lower) Keltner channels.

    Mid is an EMA; the half-width is `multiplier` ATRs. Note the two warm-ups
    differ - `atr(period)` is first defined at index `period - 1` and
    `ema(period)` at `period - 1` as well, but ATR is built on true range which
    needs the previous close, so the honest first usable bar is the later of
    the two.
    """
    cfg = config or MeanReversionConfig()
    n = len(closes)
    if not (len(highs) == len(lows) == n):
        raise MeanReversionError("closes, highs and lows must be the same length")

    mid = ema(closes, cfg.keltner_period)
    width = atr(highs, lows, closes, cfg.keltner_atr_period)
    upper: list[float | None] = [None] * n
    lower: list[float | None] = [None] * n
    for i in range(n):
        if mid[i] is None or width[i] is None:
            continue
        band = cfg.keltner_multiplier * width[i]
        upper[i] = mid[i] + band
        lower[i] = mid[i] - band
    return upper, mid, lower


def entry_flags(
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    config: MeanReversionConfig | None = None,
) -> tuple[list[bool | None], list[bool | None]]:
    """The rule as two boolean series: (entry_ok, hold_ok).

    `None` marks bars where the indicators are not warm - distinct from False,
    which means the condition was evaluated and did not hold. Collapsing the
    two would make the warm-up read as "the market is not oversold" rather than
    as "the rule cannot speak yet".
    """
    cfg = config or MeanReversionConfig()
    n = len(closes)
    if not (len(highs) == len(lows) == n):
        raise MeanReversionError("closes, highs and lows must be the same length")
    if n == 0:
        raise MeanReversionError("no bars")

    if cfg.rule == "rsi":
        osc = rsi(closes, cfg.rsi_period)
        warm = cfg.rsi_period
        # Exit needs the oscillator, entry needs the oscillator, so there is no
        # second reference series and `hold_ok` is the same series compared to
        # a different level.
        entry_at = lambda i: (
            None if osc[i] is None else osc[i] <= cfg.oversold
        )
        hold_at = lambda i: (
            None if osc[i] is None else osc[i] < cfg.exit_level
        )
    else:
        upper, mid, lower = keltner_lines(closes, highs, lows, cfg)
        warm = max(cfg.keltner_period, cfg.keltner_atr_period)
        entry_at = lambda i: (
            None if lower[i] is None else closes[i] < lower[i]
        )
        hold_at = lambda i: (
            None if mid[i] is None else closes[i] < mid[i]
        )

    regime: list[bool | None] | None = None
    if cfg.adx_max:
        trend = adx(highs, lows, closes, cfg.adx_period)
        regime = [None if v is None else v < cfg.adx_max for v in trend]
        warm = max(warm, cfg.adx_period)

    entry_ok: list[bool | None] = []
    hold_ok: list[bool | None] = []
    for i in range(n):
        if i < warm:
            entry_ok.append(None)
            hold_ok.append(None)
            continue
        e = entry_at(i)
        h = hold_at(i)
        if e is not None and regime is not None:
            r = regime[i]
            # A `None` regime is a warm-up bar for ADX only; the rule's own
            # indicators are warm by now, so the honest answer is "no entry".
            e = False if r is None else (e and r)
        entry_ok.append(e)
        hold_ok.append(h)
    return entry_ok, hold_ok


def simulate_mean_reversion(
    closes: Sequence[float],
    opens: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    config: MeanReversionConfig | None = None,
    start_bar: int | None = None,
    end_bar: int | None = None,
) -> GateResult:
    """Run the rule through the shared simulator.

    Reusing `simulate_flags` rather than writing a second loop: that loop is
    where the fill-timing guarantee lives ("the fill is at the open after the
    bar whose close decided it"), and it was wrong once already in a way that
    turned a losing rule into a 100th-percentile one. A parallel implementation
    would have to earn that back.
    """
    cfg = config or MeanReversionConfig()
    entry, hold = entry_flags(closes, highs, lows, cfg)
    return simulate_flags(
        closes,
        opens,
        entry,
        hold,
        # `cfg` *is* an ExecutionConfig, so it is passed straight through
        # rather than copied field by field into a `TrendGateConfig`. The copy
        # was where a newly added execution field could be forgotten.
        cfg,
        start_bar,
        end_bar,
        # Passed unconditionally: `simulate_flags` rejects a stop without lows,
        # so the coupling is enforced in one place rather than at every call
        # site, and a caller that adds a stop later cannot forget this argument.
        lows,
    )


def describe(config: MeanReversionConfig | None = None) -> str:
    """One line naming the rule, so a results table says which one produced it."""
    cfg = config or MeanReversionConfig()
    if cfg.rule == "rsi":
        body = f"RSI{cfg.rsi_period} <= {cfg.oversold:g}, exit >= {cfg.exit_level:g}"
    else:
        body = (
            f"close < EMA{cfg.keltner_period} - "
            f"{cfg.keltner_multiplier:g}*ATR{cfg.keltner_atr_period}, "
            f"exit >= EMA{cfg.keltner_period}"
        )
    adx_part = f" | ADX{cfg.adx_period} < {cfg.adx_max:g}" if cfg.adx_max else ""
    stop_part = ""
    if cfg.stop_is_on:
        stop_part = f" | stop {cfg.stop_loss_pct:.1%}"
        if cfg.stop_cooldown_bars:
            stop_part += f" + {cfg.stop_cooldown_bars}bar cooldown"
    return f"{body}{adx_part}{stop_part} | fee {cfg.fee_bps}bp"
