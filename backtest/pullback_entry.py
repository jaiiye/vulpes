"""A pullback entry: small 24h move, EMA reclaiming the mid band, volume up.

The rule, long-only
-------------------
    momentum   0 <= close[i] / close[i-24] - 1 <= 3%      (the move is small)
    cross      EMA(21) crosses above SMA(20) on bar i     (reclaiming the mid)
    volume     volume[i] > mean(volume[i-480 .. i])       (participation)

all three on the same bar, which is the entry signal. The position is held
while price stays above the mid band.

Why the direction is different from `trend_gate`
------------------------------------------------
The four-vote gate in `trend_gate.py` was a *continuation* rule: it bought
strength. Measured on the archive, that is the wrong way round. The only effect
this project has been able to measure at a 24h horizon is **reversal** -
cross-sectional IC -0.041 at 24h, t = -4.1 (RESEARCH.md section 2). Capping the
24h move at 3% instead of requiring it to be positive is what moves this rule
onto the side of the effect that exists, and it is the reason the proposal is
worth testing rather than another variation on the same bet.

Three things the proposal left open, and what each choice costs
---------------------------------------------------------------
These are stated rather than resolved silently, because each one changes the
answer and picking the best of them after the fact is how a search becomes a
finding.

**Which EMA.** Not specified. EMA(21) is used for continuity with the previous
rule; `ema_period` is settable.

**Cross or level.** "Crosses from below through the mid band" is an *event*, so
literally it fires only on the transition bar. That is a much rarer signal than
"is above the mid band", and with the other two filters on the same bar it can
leave too few trades to say anything. Both are measured: `require_cross=True`
is the literal reading, `False` is the level reading.

**Does the 3% band have a floor.** "Within 3%" is a ceiling on the gain; it says
nothing about falls. `min_momentum=None` takes it literally (a 20% drop passes);
`min_momentum=0.0` reads it as "flat to slightly up", which excludes buying a
falling knife. Both are measured.

The exit is also unspecified. Holding while price stays above the mid band is
the natural companion and adds no parameter; a fixed cap is available because
section 2.10 measured turnover as the binding cost - 228 round trips cost about
16 percentage points against a gross edge of 9 - so the holding period is a
first-order choice, not a detail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agent.indicators import ema, sma

from backtest.trend_gate import (
    ExecutionConfig,
    GateResult,
    Trade,  # noqa: F401  (re-exported: callers get trades from here)
    TrendGateError,
    random_control,  # noqa: F401
    percentile_of,  # noqa: F401
    random_spans,  # noqa: F401
    simulate_flags,
)

MOMENTUM_BARS = 24
#: The fast line. 21 pairs with the 20-bar mid band, which is the reading the
#: rule was first measured with; the EMA-pair reading uses 9 against 26.
EMA_PERIOD = 21
EMA_SLOW_PERIOD = 26
BOLLINGER_PERIOD = 20
VOLUME_LOOKBACK = 480
MAX_MOMENTUM = 0.03


MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9


def macd_lines(
    closes: Sequence[float],
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> tuple[list[float | None], list[float | None]]:
    """MACD line and signal line, aligned to `closes` with None warm-up.

    Written out rather than composed from `indicators.ema` twice, because the
    signal line is an EMA *of the MACD line*, and the MACD line carries a None
    prefix. Feeding that to `ema` - which expects plain floats - would either
    raise or silently skip the warm-up. The defined region is extracted, the
    signal computed on it, and the result mapped back to the original indices.
    """
    if not (2 <= fast < slow) or signal < 1:
        raise PullbackConfigError("bad MACD periods")
    ef, es = ema(closes, fast), ema(closes, slow)
    line: list[float | None] = [None] * len(closes)
    for i in range(len(closes)):
        if ef[i] is not None and es[i] is not None:
            line[i] = ef[i] - es[i]

    start = next((i for i, v in enumerate(line) if v is not None), len(closes))
    tail = [line[i] for i in range(start, len(closes))]
    if not tail:
        return line, [None] * len(closes)
    sig_tail = ema(tail, signal)
    sig: list[float | None] = [None] * len(closes)
    for k, i in enumerate(range(start, len(closes))):
        sig[i] = sig_tail[k]
    return line, sig


def ema_pair(min_momentum: float | None = 0.0,
             max_momentum: float = MAX_MOMENTUM,
             require_volume: bool = True,
             **overrides) -> "PullbackConfig":
    """EMA(9) crossing EMA(26), with the 0..3% band.

    A named constructor rather than new defaults: every number already reported
    for this rule was produced with EMA(21) against the 20-bar mid band, and
    changing the defaults would make those unreproducible without saying so.
    """
    base = dict(
        ema_period=9,
        ema_slow_period=26,
        cross_reference="ema",
        min_momentum=min_momentum,
        max_momentum=max_momentum,
        require_volume=require_volume,
    )
    base.update(overrides)
    return PullbackConfig(**base)


class PullbackConfigError(ValueError):
    """Raised when the rule's settings cannot describe a tradeable rule.

    Covers this rule's own parameters only. A bad `hold_bars`, `fee_bps` or
    `funding_per_bar` raises `TrendGateError`, because those fields are defined
    by `ExecutionConfig` and validated there - one definition, one error.

    Defined above `PullbackConfig` rather than below it. It was below, which
    worked because the class body only calls it at instantiation, but it put an
    exception's definition after its only user.
    """


@dataclass
class PullbackConfig(ExecutionConfig):
    """This rule's parameters, on top of everything executable.

    Inherits the execution fields from `ExecutionConfig` instead of restating
    them. The rule previously restated three of them and then copied them into
    a `TrendGateConfig` to hand to `simulate_flags`; because only three were
    restated, the stop this repository implements and tests was unreachable
    from this rule.
    """
    momentum_bars: int = MOMENTUM_BARS
    #: Ceiling on the 24h change. The point of the rule: do not buy something
    #: that has already run.
    max_momentum: float = MAX_MOMENTUM
    #: Floor on the 24h change. `None` is the literal reading of "within 3%"
    #: and admits a falling market; 0.0 requires flat-to-up.
    min_momentum: float | None = None

    # -- context: was there an impulse, and have we pulled back from it? ------
    # The chart that motivated these shows a sharp rally, a sideways stretch,
    # then a return to the moving averages followed by a cross. A bare cross
    # fires anywhere; these two conditions say it has to fire *after a rally
    # that has come back*. Disabled by default (lookback 0) so every result
    # already reported is unchanged.
    impulse_lookback: int = 0
    #: The rally over that window must be at least this large, high versus low.
    min_impulse: float = 0.0
    #: And the current close must be at least this far below the window high,
    #: so the entry is a pullback rather than a breakout.
    min_pullback: float = 0.0
    #: Optional ceiling on the pullback: beyond this the structure is broken
    #: rather than resting. None means no ceiling, which is the looser reading.
    max_pullback: float | None = None

    #: The fast line. With `cross_reference="ema"` it is compared against
    #: `ema_slow_period`; with "sma", against `bollinger_period`.
    ema_period: int = EMA_PERIOD
    ema_slow_period: int = EMA_SLOW_PERIOD
    bollinger_period: int = BOLLINGER_PERIOD
    #: What the fast EMA has to cross. The two are NOT interchangeable and the
    #: difference is the reason this knob exists:
    #:
    #: * "sma" - EMA(21) against SMA(20), the mid band. Measured: EMA(21) sits
    #:   *below* SMA(20) on every bar of a linear ramp (0 of 700 bars), because
    #:   the exponential lag is 10 bars against the simple average's 9.5. So it
    #:   is not a trend filter at all - it needs the last ~20 bars to be strong
    #:   relative to what came before, i.e. an acceleration.
    #: * "ema" - EMA(9) against EMA(26). Both carry infinite memory, with lags
    #:   of 4 and 12.5 bars, so this IS satisfied throughout a steady uptrend.
    #:   It is a trend condition, and it therefore points the opposite way to
    #:   the only effect this project has been able to measure (24h reversal,
    #:   RESEARCH.md section 2.11) - which is why the two are reported apart.
    cross_reference: str = "sma"
    #: What does the crossing. "ema" is the moving average crossing the
    #: reference; "price" is the close crossing it.
    #:
    #: Not cosmetic. Both charts that motivated this rule show the *close*
    #: reclaiming the mid band while EMA(9) is still below EMA(26) - the moving
    #: averages had not crossed yet. So the entry being described is earlier
    #: than a golden cross, and "price" is the setting that expresses it.
    cross_subject: str = "ema"
    #: Require the fast EMA to be *below* the slow one at entry: the trade is
    #: taken before the cross, betting it follows. Measured separately because
    #: it removes any chance of the entry being the cross itself.
    require_lagging_cross: bool = False
    #: Require the MACD line to be above its signal line. The earlier of the
    #: two confirmations the charts show.
    require_macd: bool = False
    macd_fast: int = MACD_FAST
    macd_slow: int = MACD_SLOW
    macd_signal: int = MACD_SIGNAL
    volume_lookback: int = VOLUME_LOOKBACK
    #: Whether volume above its trailing mean is required. Kept switchable
    #: because a later proposal dropped the condition without saying so, and
    #: choosing for the caller would make the results unattributable.
    require_volume: bool = True

    #: True: the entry fires only on the bar the fast line crosses up through
    #: the reference. False: it fires on any bar the fast line is above it.
    require_cross: bool = True
    #: Exit when the fast line falls back below the reference.
    hold_above_mid: bool = True

    # Execution and stop fields are inherited; see `ExecutionConfig`.

    def __post_init__(self) -> None:
        # Not optional: skipping it silently drops every execution check.
        super().__post_init__()
        if self.min_momentum is not None and self.min_momentum > self.max_momentum:
            raise PullbackConfigError(
                "min_momentum must not exceed max_momentum, or no bar can pass"
            )
        if self.ema_period < 2 or self.bollinger_period < 2:
            raise PullbackConfigError("periods must be >= 2")
        if self.cross_reference not in ("sma", "ema"):
            raise PullbackConfigError("cross_reference must be 'sma' or 'ema'")
        if self.cross_subject not in ("ema", "price"):
            raise PullbackConfigError("cross_subject must be 'ema' or 'price'")
        if self.cross_reference == "ema":
            if self.ema_slow_period < 2:
                raise PullbackConfigError("ema_slow_period must be >= 2")
            if self.ema_period >= self.ema_slow_period:
                raise PullbackConfigError(
                    "ema_period must be shorter than ema_slow_period"
                )
        if self.volume_lookback < 1:
            raise PullbackConfigError("volume_lookback must be >= 1")
        if self.impulse_lookback < 0:
            raise PullbackConfigError("impulse_lookback must be >= 0")
        if self.min_impulse < 0 or self.min_pullback < 0:
            raise PullbackConfigError("impulse and pullback sizes must be >= 0")
        if (
            self.max_pullback is not None
            and self.max_pullback < self.min_pullback
        ):
            raise PullbackConfigError(
                "max_pullback must not be below min_pullback, or no bar can pass"
            )


def entry_flags(
    closes: Sequence[float],
    volumes: Sequence[float],
    config: PullbackConfig | None = None,
) -> tuple[list[bool | None], list[bool | None]]:
    """The rule as two boolean series: (entry_ok, hold_ok).

    `None` marks bars where the indicators are not warm. Returning None rather
    than False keeps the warm-up out of the "condition failed" bucket, which is
    the distinction that a bare False would erase.
    """
    cfg = config or PullbackConfig()
    n = len(closes)
    if len(volumes) != n:
        raise PullbackConfigError(
            f"closes has {n} bars but volumes has {len(volumes)}"
        )
    if n == 0:
        raise PullbackConfigError("no bars")

    fast = ema(closes, cfg.ema_period)
    slow = ema(closes, cfg.ema_slow_period)
    if cfg.cross_reference == "ema":
        reference = slow
        ref_warm = cfg.ema_slow_period
    else:
        reference = sma(closes, cfg.bollinger_period)
        ref_warm = cfg.bollinger_period
    # What has to cross the reference: the fast average, or the close itself.
    subject = closes if cfg.cross_subject == "price" else fast
    vol_mean = sma(volumes, cfg.volume_lookback)
    if cfg.require_macd:
        macd_line, macd_sig = macd_lines(
            closes, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
        )
        macd_warm = cfg.macd_slow + cfg.macd_signal
    else:
        macd_line = macd_sig = None
        macd_warm = 0

    entry_ok: list[bool | None] = []
    hold_ok: list[bool | None] = []
    for i in range(n):
        if (
            i - cfg.momentum_bars < 0
            or i - ref_warm + 1 < 0
            or i - cfg.ema_period + 1 < 0
            or i - cfg.volume_lookback + 1 < 0
            or (cfg.impulse_lookback and i - cfg.impulse_lookback + 1 < 0)
            or (macd_warm and i - macd_warm + 1 < 0)
        ):
            entry_ok.append(None)
            hold_ok.append(None)
            continue

        past = closes[i - cfg.momentum_bars]
        if past <= 0:
            entry_ok.append(None)
            hold_ok.append(None)
            continue

        change = closes[i] / past - 1.0
        small_move = change <= cfg.max_momentum and (
            cfg.min_momentum is None or change >= cfg.min_momentum
        )
        context = True
        if cfg.impulse_lookback:
            window = closes[i - cfg.impulse_lookback + 1: i + 1]
            high, low = max(window), min(window)
            prior_style = low > 0 and (high / low - 1.0) >= cfg.min_impulse
            if prior_style:
                # How far the close now sits under that high. The entry is a
                # pullback, so it must be below it and not so far below that
                # the move has failed.
                drop = high / closes[i] - 1.0 if closes[i] > 0 else -1.0
                prior_style = drop >= cfg.min_pullback and (
                    cfg.max_pullback is None or drop <= cfg.max_pullback
                )
            context = prior_style
        if cfg.require_volume:
            mean_volume = vol_mean[i]
            volume_up = (
                mean_volume is not None
                and mean_volume > 0
                and volumes[i] > mean_volume
            )
        else:
            volume_up = True
        above = (
            reference[i] is not None
            and subject[i] is not None
            and subject[i] > reference[i]
        )
        if cfg.require_cross:
            # The transition, not the state: the previous bar was at or below
            # the reference and this bar is above it.
            crossed = above and i > 0 and reference[i - 1] is not None and (
                subject[i - 1] is not None and subject[i - 1] <= reference[i - 1]
            )
        else:
            crossed = above

        # Early entry: the averages have not crossed yet, so the trade is taken
        # before the golden cross rather than on it.
        lagging = True
        if cfg.require_lagging_cross:
            lagging = (
                fast[i] is not None and slow[i] is not None and fast[i] < slow[i]
            )

        macd_ok = True
        if cfg.require_macd:
            macd_ok = (
                macd_line is not None
                and macd_sig is not None
                and macd_line[i] is not None
                and macd_sig[i] is not None
                and macd_line[i] > macd_sig[i]
            )

        entry_ok.append(
            bool(small_move and volume_up and crossed and context and lagging
                 and macd_ok)
        )
        hold_ok.append(
            bool(above) if cfg.hold_above_mid else True
        )
    return entry_ok, hold_ok


def simulate_pullback(
    closes: Sequence[float],
    opens: Sequence[float],
    volumes: Sequence[float],
    config: PullbackConfig | None = None,
    start_bar: int | None = None,
    end_bar: int | None = None,
    lows: Sequence[float] | None = None,
) -> GateResult:
    """Run the rule through the shared simulator.

    `lows` is needed only when `config.stop_loss_pct` is set, and it is a
    separate argument for the same reason it is on `simulate_flags`: a stop is
    breached intrabar, so it must read the low and not the close.

    It is also the second half of a gap this rule had. `PullbackConfig` used to
    restate three of the four execution fields, so `stop_loss_pct` could not be
    set - and this function had no `lows`, so even with the field it could not
    have fired. Inheriting `ExecutionConfig` fixes the first half; this
    parameter fixes the second, and the test asserts a stop actually fires
    rather than that either half exists.

    Reusing `simulate_flags` rather than writing a second loop is deliberate:
    that loop is where the fill-timing guarantee lives, and it was wrong once
    already in a way that turned a losing rule into a 100th-percentile one. A
    parallel implementation would have to earn that back.
    """
    cfg = config or PullbackConfig()
    entry, hold = entry_flags(closes, volumes, cfg)
    # `cfg` *is* an ExecutionConfig, so it goes straight to the loop rather than
    # through a translation helper. The helper existed to copy three fields into
    # a `TrendGateConfig`; a fourth execution field added later would have had
    # to remember to extend it, and the stop added in this repository's section
    # 2.16 is exactly the field that did not get extended.
    return simulate_flags(
        closes, opens, entry, hold, cfg, start_bar, end_bar, lows
    )


def describe(config: PullbackConfig | None = None) -> str:
    """One-line summary, so a table of results says which rule produced it."""
    cfg = config or PullbackConfig()
    band = (
        f"{cfg.min_momentum:.1%}..{cfg.max_momentum:.1%}"
        if cfg.min_momentum is not None
        else f"<= {cfg.max_momentum:.1%}"
    )
    cross = "cross" if cfg.require_cross else "level"
    if cfg.cross_reference == "ema":
        line = f"EMA{cfg.ema_period} vs EMA{cfg.ema_slow_period}"
    else:
        line = f"EMA{cfg.ema_period} vs SMA{cfg.bollinger_period}"
    vol = f"vol>{cfg.volume_lookback}bar mean" if cfg.require_volume else "no vol"
    ctx = ""
    if cfg.impulse_lookback:
        ceiling = "inf" if cfg.max_pullback is None else f"{cfg.max_pullback:.1%}"
        ctx = (
            f" | impulse>={cfg.min_impulse:.1%} over {cfg.impulse_lookback}bars"
            f", pullback {cfg.min_pullback:.1%}..{ceiling}"
        )
    return f"24h {band} | {line} ({cross}) | {vol}{ctx} | fee {cfg.fee_bps}bp"
