"""Is a measured edge real, or is it the bid-ask spread moving?

The question recurs every time a reversal-like rule tests positive on
lower-liquidity symbols: a stale quote bouncing between the bid and the ask
produces exactly the pattern a mean-reversion rule is looking for, and the
"profit" is then the trader's own spread rather than information. The estimates
here are the standard price-based diagnostics, and they are kept together
because they are only interpretable as a set - each one alone has a failure mode
the others cover.

Three tests, and what each one rules out
----------------------------------------
1. **Autocorrelation by lag.** The bounce is a lag-1 phenomenon: it makes
   `r_t` and `r_{t-1}` negatively related and leaves longer lags alone. A
   genuine reversal-following-reaction keeps going for several bars. So a
   negative autocorrelation that is *only* at lag 1 is the bounce; one that
   persists at lag 24 is not.

2. **Roll's spread.** Under the model "the only reason consecutive returns are
   negatively related is that one landed on the bid and the next on the ask",
   the effective spread is `2*sqrt(-Cov(r_t, r_{t-1}))`. It is an **upper
   bound**, not an estimate, and the distinction matters: measured on this
   archive, lag-24 autocorrelation is as negative as lag-1, so a large part of
   the covariance Roll attributes to the spread is genuine reversal. Reporting
   the number as "the spread" would overstate the cost by roughly an order of
   magnitude on names with millions of dollars of daily volume. It is useful as
   a *pessimistic* cost assumption, and that is how it is used.

3. **Variance ratio.** `Var(q-period return) / (q * Var(1-period return))`.
   Below 1 means returns partially undo themselves (reversal or bounce); above
   1 means they reinforce (trending). Unlike Roll it makes no assumption about
   *why*, so it survives the contamination above - it just cannot separate the
   two causes.

What none of them can do
------------------------
Tell you whether the edge is *tradeable*. An edge can be entirely real and
still smaller than the spread you must cross to collect it. That is a cost
question, and it is answered by re-running the result with the spread as the
fee rather than by any statistic in this module.
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence

MIN_SAMPLES = 100


class MicrostructureError(ValueError):
    """Raised when a series is too short or degenerate to estimate from."""


def _clean(returns: Sequence[float | None]) -> list[float]:
    """Drop missing values, keeping order. Gaps are not bridged."""
    return [float(r) for r in returns if r is not None]


def autocovariance(
    returns: Sequence[float | None], lag: int = 1
) -> float | None:
    """Covariance between returns and their `lag`-back value.

    `None` rather than 0.0 when there is not enough data: a zero covariance is
    a meaningful estimate ("no relationship") and returning it for "cannot say"
    would let a short series vote for the null hypothesis.
    """
    r = _clean(returns)
    if lag < 1:
        raise MicrostructureError("lag must be >= 1")
    n = len(r) - lag
    if n < MIN_SAMPLES:
        return None
    mean = statistics.fmean(r)
    return sum((r[i] - mean) * (r[i + lag] - mean) for i in range(n)) / n


def autocorrelation(
    returns: Sequence[float | None], lag: int = 1
) -> float | None:
    """`autocovariance` divided by the variance, so it is scale-free.

    The normalised form is the one to compare across symbols: raw
    autocovariance scales with the symbol's volatility, so quiet symbols would
    look clean whether or not they are.
    """
    r = _clean(returns)
    if lag < 1:
        raise MicrostructureError("lag must be >= 1")
    if len(r) < MIN_SAMPLES + lag:
        return None
    cov = autocovariance(r, lag)
    if cov is None:
        return None
    var = statistics.pvariance(r)
    if var <= 0:
        return None
    return cov / var


def roll_spread(returns: Sequence[float | None]) -> float | None:
    """Effective spread from the lag-1 autocovariance, as a **fraction** of price.

    `2 * sqrt(-Cov(r_t, r_{t-1}))` when that covariance is negative, else 0.0 -
    a non-negative covariance means the bounce model has nothing to say, and
    reporting a negative spread would be worse than reporting none.

    Read this as an upper bound. It is the spread you get if *every* negative
    lag-1 covariance came from trading at alternating sides of the book, and
    on this archive a large share of it is genuine multi-bar reversal instead
    (lag-24 is as negative as lag-1). Good as a pessimistic cost input, not as
    a measurement of the book.
    """
    r = _clean(returns)
    if len(r) < MIN_SAMPLES + 1:
        return None
    cov = autocovariance(r, 1)
    if cov is None or cov >= 0:
        return 0.0
    return 2.0 * math.sqrt(-cov)


def roll_spread_bps(returns: Sequence[float | None]) -> float | None:
    """`roll_spread` in basis points, one-way."""
    s = roll_spread(returns)
    return None if s is None else s * 10_000.0


def variance_ratio(returns: Sequence[float | None], q: int = 2) -> float | None:
    """`Var(q-period) / (q * Var(1-period))`.

    Below 1 = mean reverting, above 1 = trending, 1 = a random walk. Makes no
    assumption about the cause, so it is unaffected by the contamination that
    makes Roll an upper bound - but for the same reason it cannot say whether
    the reversal is information or microstructure.
    """
    r = _clean(returns)
    if q < 2:
        raise MicrostructureError("q must be >= 2")
    if len(r) < q * MIN_SAMPLES:
        return None
    one = statistics.pvariance(r)
    if one <= 0:
        return None
    agg = [sum(r[i:i + q]) for i in range(len(r) - q + 1)]
    return statistics.pvariance(agg) / (q * one)


Level = tuple[float, float]        # (price, size in coin units)


def _walk(levels: Sequence[Level], notional: float) -> float | None:
    """Average fill price for spending `notional` across `levels`.

    Returns None when the visible depth cannot fill it: pretending the order
    filled at the last visible price would quote a cost for size the book never
    offered, which is the optimistic direction and the one that matters.
    """
    remaining = notional
    spent = 0.0
    got = 0.0
    for price, size in levels:
        if price <= 0 or size <= 0:
            continue
        available = price * size
        take = min(available, remaining)
        got += take / price
        spent += take
        remaining -= take
        if remaining <= 1e-9:
            break
    if remaining > 1e-9 or got <= 0:
        return None
    return spent / got


def round_trip_cost_bps(
    bids: Sequence[Level], asks: Sequence[Level], notional: float
) -> float | None:
    """Cost of buying and then selling `notional`, in bps of the mid.

    This is the number that decides whether an edge is tradeable, and it is not
    the quoted spread: a market order walks the book, so the cost grows with
    size. Both legs are charged.

    `None` when either side cannot fill the size - an honest "the book does not
    support this order" rather than a number derived from depth that is not
    there.
    """
    if notional <= 0:
        raise MicrostructureError("notional must be > 0")
    if not bids or not asks:
        return None
    best_bid, best_ask = bids[0][0], asks[0][0]
    if best_bid <= 0 or best_ask <= 0 or best_ask < best_bid:
        return None
    mid = (best_bid + best_ask) / 2.0

    buy = _walk(asks, notional)
    sell = _walk(bids, notional)
    if buy is None or sell is None:
        return None
    return ((buy - mid) + (mid - sell)) / mid * 10_000.0


def top_of_book_spread_bps(
    bids: Sequence[Level], asks: Sequence[Level]
) -> float | None:
    """Quoted spread at the touch, in bps. The cheapest possible leg, and the
    one usually quoted as "the spread" - it is a lower bound on execution cost,
    not an estimate of it."""
    if not bids or not asks:
        return None
    bid, ask = bids[0][0], asks[0][0]
    if bid <= 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid * 10_000.0


def depth_within_bps(
    asks: Sequence[Level], reference: float, within_bps: float
) -> float:
    """Notional resting within `within_bps` above `reference`.

    The complement of a cost estimate: it answers "how much size is available
    at this price", which is what caps a strategy independently of its edge.
    """
    if reference <= 0:
        raise MicrostructureError("reference price must be > 0")
    cap = reference * (1.0 + within_bps / 10_000.0)
    return sum(p * s for p, s in asks if 0 < p <= cap)


def reversal_profile(
    returns: Sequence[float | None], lags: Sequence[int] = (1, 2, 3, 6, 12, 24, 48)
) -> dict[int, float | None]:
    """Autocorrelation at each lag, as a dict.

    The profile rather than a single number is what diagnoses the bounce: a
    `{1: -0.03, 2: 0.00, 3: 0.00}` shape is microstructure, while
    `{1: -0.03, 24: -0.03}` is not, and neither value alone says which.
    """
    return {lag: autocorrelation(returns, lag) for lag in lags}


def bounce_share(returns: Sequence[float | None]) -> float | None:
    """Fraction of the total negative autocorrelation that sits at lag 1.

    A summary of `reversal_profile` for when one number is needed: near 1.0 the
    negative relationship is entirely adjacent-bar (bounce-like), near 0 it is
    spread across several lags (reversal-like). `None` when there is no negative
    autocorrelation to attribute.
    """
    lags = (1, 2, 3, 6, 12, 24, 48)
    prof = reversal_profile(returns, lags)
    total = sum(-v for v in prof.values() if v is not None and v < 0)
    if total <= 0:
        return None
    first = prof.get(1)
    if first is None or first >= 0:
        return 0.0
    return -first / total
