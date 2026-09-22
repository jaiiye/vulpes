"""Combining return streams, and the tests that stop it being self-deception.

Built to answer one question: two strategies have each tested positive in this
repo (`cross_section` and `mean_reversion`), so should they be run together?
The answer turned out to be "they are independent, but the improvement does not
reproduce" - which is a result this module exists to make checkable rather than
a thing to assert in prose.

What it does
------------
`sharpe`, `correlation`, `risk_parity_weights`, `blend`, `weight_sweep`,
`halves`. All on per-period return series, so callers do not need to agree on a
frequency - the numbers are per-period Sharpes, and rescaling to annual is the
caller's business.

Why risk parity and not equal capital
------------------------------------
The two strategies' volatilities differ by about 4x (a long-short basket
against a long-only rule at ~16% exposure). Equal capital is therefore not equal
risk: it puts ~80% of the combined variance on whichever book is noisier, and
if that one also has the lower Sharpe - as it does here - the blend scores
*worse* than the better component alone. That is not a failure of combining, it
is arithmetic, and it is the specific trap this module is written around.

The unpromising part
--------------------
A full-sample blend can look good while being the average of "nothing" and
"great". `halves` exists so that is a number rather than a suspicion, because
it has already happened twice in this repo - a stop-loss whose optimal level
appeared only in the pooled sample (§2.16), and this blend (§2.20).
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence


class CombineError(ValueError):
    """Raised when the inputs cannot describe a combination."""


def _check(*series: Sequence[float]) -> None:
    if not series:
        raise CombineError("no series supplied")
    n = len(series[0])
    if n < 2:
        raise CombineError("need at least 2 periods")
    for s in series:
        if len(s) != n:
            raise CombineError("all series must have the same length, and be aligned")


def correlation(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation. 0.0 when either series is flat.

    Flat rather than `None`: with no variance there is no relationship to
    measure, and 0.0 (independent) is the conservative assumption for blending -
    it claims diversification that the data does not contradict.

    Note what a correlation of ~0 does *not* establish. At n=202 the standard
    error is about 0.07, so anything up to roughly 0.14 is indistinguishable
    from zero. "Independent" here means "not distinguishable from independent",
    not "proven independent".
    """
    _check(a, b)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else 0.0


def sharpe(returns: Sequence[float]) -> float:
    """Per-period Sharpe: mean / population stdev, scaled by sqrt(periods).

    Population (not sample) stdev so a single series gives the same number
    however it is sliced - and 0.0 for a flat series rather than a division by
    zero, because "no risk and no return" is a legitimate input when one leg of
    a blend never trades.
    """
    _check(returns)
    sd = statistics.pstdev(returns)
    if sd == 0:
        return 0.0
    return statistics.fmean(returns) / sd * math.sqrt(len(returns))


def risk_parity_weights(series: Sequence[Sequence[float]]) -> list[float]:
    """Inverse-volatility weights, normalised to sum to 1.

    `series` is one return stream per component. A flat component gets an
    infinite inverse-volatility, so a zero-volatility leg takes the whole
    allocation - which is why the guard raises instead of inventing a cap.
    """
    if not series:
        raise CombineError("no series supplied")
    _check(*series)
    inv = []
    for s in series:
        sd = statistics.pstdev(s)
        if sd <= 0:
            raise CombineError(
                "a component has zero volatility; inverse-volatility weighting "
                "is undefined and silently capping it would pick a weight the "
                "data did not choose"
            )
        inv.append(1.0 / sd)
    total = sum(inv)
    return [x / total for x in inv]


def blend(series: Sequence[Sequence[float]], weights: Sequence[float]) -> list[float]:
    """Weighted sum of return streams, period by period.

    Weights must sum to 1 within a tolerance: a blend that is not fully
    invested is a different (and less comparable) object, and normalising
    silently would hide a caller's mistake about what they passed.
    """
    if not series:
        raise CombineError("no series supplied")
    _check(*series)
    if len(weights) != len(series):
        raise CombineError("need one weight per series")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise CombineError(f"weights must sum to 1, got {sum(weights)!r}")
    return [sum(w * s[i] for w, s in zip(weights, series))
            for i in range(len(series[0]))]


def weight_sweep(
    a: Sequence[float], b: Sequence[float], steps: int = 20
) -> tuple[float, float]:
    """(weight on `a`, Sharpe) at the best of a uniform weight grid.

    A grid rather than an analytic optimum because the analytic version assumes
    the sample means and covariances are known, which at ~200 periods they are
    not - and an optimum that is sharper than the estimate it came from invites
    exactly the overfitting the sweep is meant to expose.
    """
    if steps < 1:
        raise CombineError("steps must be >= 1")
    _check(a, b)
    best_w, best_s = 0.0, -math.inf
    for i in range(steps + 1):
        w = i / steps
        s = sharpe([w * x + (1 - w) * y for x, y in zip(a, b)])
        if s > best_s:
            best_w, best_s = w, s
    return best_w, best_s


def halves(returns: Sequence[float]) -> tuple[list[float], list[float]]:
    """Split into two contiguous halves.

    Contiguous rather than interleaved on purpose: the failure this is meant to
    catch is a result carried by one stretch of the sample, and interleaving
    would average that away instead of revealing it.
    """
    _check(returns)
    mid = len(returns) // 2
    return list(returns[:mid]), list(returns[mid:])


def sign_test(successes: int, trials: int) -> float:
    """Two-sided binomial tail probability under p=0.5.

    The unit of observation is a **window**, not a symbol or a bar, because the
    question it answers is "does this replicate" and only windows are
    independent of each other. Symbols inside a window share the market factor
    and bars inside a window are sequential, so counting them as trials would
    inflate the significance by a factor the correction in `universe` exists to
    remove.

    Deliberately crude - it ignores effect sizes entirely, which is what makes
    it the hardest test to pass and the first one to quote.
    """
    if trials < 1:
        raise CombineError("trials must be >= 1")
    if not 0 <= successes <= trials:
        raise CombineError("successes must be in 0..trials")
    # Two-sided means `2 * min(lower tail, upper tail)`, capped at 1. Doubling
    # only the *upper* tail is a common shortcut and it fails below the midpoint
    # - 1 success out of 6 came out as P = 1.0 instead of the correct 0.219,
    # breaking the symmetry with 5 of 6. A test on that symmetry caught it.
    upper = sum(math.comb(trials, i) for i in range(successes, trials + 1))
    lower = sum(math.comb(trials, i) for i in range(0, successes + 1))
    return min(1.0, 2.0 * min(upper, lower) / 2 ** trials)


def equal_windows(n_periods: int, count: int) -> list[tuple[int, int]]:
    """`count` contiguous, non-overlapping [start, end) splits of `n_periods`.

    A remainder is dropped rather than distributed, so every window is the same
    length. Unequal windows would make a per-window comparison partly a
    comparison of window lengths, and the discarded tail is at most
    `count - 1` periods.
    """
    if count < 1:
        raise CombineError("count must be >= 1")
    if n_periods < count:
        raise CombineError("not enough periods for that many windows")
    size = n_periods // count
    return [(i * size, (i + 1) * size) for i in range(count)]


def replicates(values: Sequence[float], threshold: float) -> dict[str, float]:
    """How many of a per-window measure clear `threshold`, and the sign test.

    Bundled because the count and its p-value are the same claim: "5 of 6" is
    not evidence on its own, and neither is a p-value without the count behind
    it.
    """
    _check(values)
    k = sum(1 for v in values if v > threshold)
    return {
        "count": float(k),
        "trials": float(len(values)),
        "fraction": k / len(values),
        "p_value": sign_test(k, len(values)),
        "median": statistics.median(values),
        "range": (min(values), max(values)),
    }


def stability(
    a: Sequence[float], b: Sequence[float]
) -> dict[str, float | tuple[float, ...]]:
    """Full-sample and split-half numbers for a two-way blend, side by side.

    Bundled so a caller cannot report the full-sample Sharpe without also
    having the halves in hand. That is the whole point: every headline in this
    repo that failed later failed at this step, and both times the full-sample
    number was the one quoted first.
    """
    _check(a, b)
    w = risk_parity_weights([a, b])
    combined = blend([a, b], w)
    ah, bh_ = halves(a), halves(b)
    return {
        "correlation": correlation(a, b),
        "sharpe_a": sharpe(a),
        "sharpe_b": sharpe(b),
        "sharpe_blend": sharpe(combined),
        "risk_parity_weight_a": w[0],
        "halves_sharpe_a": (sharpe(ah[0]), sharpe(ah[1])),
        "halves_sharpe_b": (sharpe(bh_[0]), sharpe(bh_[1])),
        "halves_sharpe_blend": tuple(
            sharpe(blend([ah[i], bh_[i]], w)) for i in (0, 1)
        ),
    }
