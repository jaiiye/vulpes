"""Universe construction: many symbols that are not the same bet.

Why this exists
---------------
The directional backtest runs BTC, ETH, SOL. Measured over the 170-day window
those three have pairwise hourly correlations of +0.888 / +0.838 / +0.850, and
the 40 most liquid symbols average +0.517 against each other. Reporting the
pooled trade count as one sample therefore overstates the evidence, and the
standard correction for k correlated series is

    n_eff = n / (1 + (k - 1) * rho_bar)

which turns 130 pooled trades into roughly 48 independent ones.

Two different quantities are both called "effective sample size"
----------------------------------------------------------------
Conflating them is the error the research log records as its most repeated one
(six separate incidents), so this module keeps them apart by name:

* `effective_bets` - how many *uncorrelated series* a universe is worth, defined
  through the variance of the equal-weighted mean:

      N = k / (1 + (k - 1) * rho_bar)

  Use it for "is this really 40 bets or 2?".

* `sample_inflation` - how much a pooled *observation count* overstates itself:

      f = 1 + (k - 1) * rho_bar

  Use it for "130 trades is really how many?".

The two are reciprocals up to the factor k, so quoting the wrong one is an
error of `(k-1) * rho_bar` - a factor of 2.7 on three majors, 21 on forty
symbols. Both are returned side by side by `correlation_report` rather than one
being chosen silently.

There is also a third quantity in circulation, the participation ratio of the
correlation matrix's eigenvalues, which counts *factors* rather than series and
gives larger numbers for the same panel (3.2 against 1.9 on forty symbols). It
needs a linear algebra dependency this project does not carry, so it is not
implemented here - but it is why a bare "N_eff" with no stated definition is not
a usable claim.

What this does NOT do
---------------------
It does not make `engine.Backtester` hold a basket, and it does not fix the
statistics by filtering. Dropping correlated names changes *which* symbols are
traded, not how many independent observations exist. The cross-section module is
what raises the observation count; this module is what stops a universe from
being counted as more bets than it is.
"""

from __future__ import annotations

import math
import statistics
from typing import Iterable

#: Kept as a module-level constant rather than imported from cross_section so
#: this module has no dependency on the portfolio builder: it operates on a
#: `Panel` by duck typing (`times`, `symbols`, `price`), which keeps the tests
#: able to use small stand-in objects.
RETURN_LOOKBACK = 1


class UniverseError(ValueError):
    """Raised when a universe or its inputs are malformed."""


# ----------------------------------------------------------------------
# Returns and correlations
# ----------------------------------------------------------------------
def returns_from(panel, lookback: int = RETURN_LOOKBACK) -> dict[str, list[float | None]]:
    """Simple returns per symbol, aligned row-for-row with `panel.times`.

    `None` marks a return that cannot be computed - a missing bar at either end
    of the pair. Gaps are skipped rather than bridged: carrying the previous
    price across a halt would invent a zero return, which biases every
    correlation computed from it toward zero exactly where the data is worst.
    """
    if lookback < 1:
        raise UniverseError("lookback must be >= 1")
    out: dict[str, list[float | None]] = {}
    for s in panel.symbols:
        series: list[float | None] = []
        for i in range(len(panel.times)):
            now = panel.price(s, i)
            then = panel.price(s, i - lookback) if i - lookback >= 0 else None
            if now is None or then is None or then <= 0:
                series.append(None)
            else:
                series.append((now - then) / then)
        out[s] = series
    return out


def correlation_or_none(
    a: Iterable[float | None],
    b: Iterable[float | None],
    min_pairs: int = 30,
) -> float | None:
    """Pearson correlation over the rows where both series have a value.

    Returns None below `min_pairs` overlapping rows: a correlation from a
    handful of points is noise, and silently returning it invites it to be
    averaged into a universe-level number that looks well supported.
    A zero-variance series also returns None - the correlation is undefined,
    not zero.

    Named `_or_none` rather than `correlation` because `combine.correlation` is
    the same statistic with the opposite convention for that last case: it
    returns 0.0 for a flat series, because a blend has to divide by something
    and "independent" claims the least. Both behaviours are deliberate, the two
    were called the same thing, and an import picking the wrong one produced a
    different number with no error. The name now says which one this is.
    """
    xs: list[float] = []
    ys: list[float] = []
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        if math.isnan(x) or math.isnan(y):
            continue
        xs.append(x)
        ys.append(y)
    if len(xs) < min_pairs:
        return None

    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)


def pairwise_correlations(
    returns: dict[str, list[float | None]],
    symbols: Iterable[str] | None = None,
    min_pairs: int = 30,
) -> dict[tuple[str, str], float]:
    """Every unordered pair's correlation, skipping pairs that cannot be scored.

    Pairs are keyed `(a, b)` with `a < b` so a lookup never depends on
    argument order - the kind of asymmetry that silently returns a missing key
    and reads as "no correlation" rather than "wrong key".
    """
    names = sorted(symbols if symbols is not None else returns)
    out: dict[tuple[str, str], float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a not in returns or b not in returns:
                continue
            c = correlation_or_none(returns[a], returns[b], min_pairs=min_pairs)
            if c is not None:
                out[(a, b) if a < b else (b, a)] = c
    return out


def mean_pairwise_correlation(
    returns: dict[str, list[float | None]],
    symbols: Iterable[str] | None = None,
    min_pairs: int = 30,
) -> float | None:
    """Average of every scoreable pair. None when no pair can be scored."""
    pairs = pairwise_correlations(returns, symbols, min_pairs=min_pairs)
    if not pairs:
        return None
    return statistics.mean(pairs.values())


# ----------------------------------------------------------------------
# The two sample-size corrections, kept apart
# ----------------------------------------------------------------------
def sample_inflation(k: int, rho_bar: float) -> float:
    """Multiplier on a pooled observation count: `1 + (k-1) * rho_bar`.

    This is the one to use on "130 pooled trades", giving the divisor that
    turns it into independent observations.
    """
    if k < 1:
        raise UniverseError("k must be >= 1")
    if not -1.0 <= rho_bar <= 1.0:
        raise UniverseError("rho_bar must be a correlation")
    return 1.0 + (k - 1) * rho_bar


def effective_bets(k: int, rho_bar: float) -> float:
    """How many uncorrelated series a `k`-symbol universe is worth.

    Derived from the variance of the equal-weighted mean of `k` standardised
    series with average pairwise correlation `rho_bar`:

        Var(mean) = (1 + (k - 1) * rho_bar) / k

    so the equivalent count of independent series is `1 / Var(mean)`, which
    reduces to `k / (1 + (k - 1) * rho_bar)`.

    Note this is NOT the participation ratio of the correlation matrix's
    eigenvalues (that counts factors and returns a larger number for the same
    panel). The definition is stated here because an unlabelled "N_eff" has
    caused a wrong conclusion in this project's history already.
    """
    if k < 1:
        raise UniverseError("k must be >= 1")
    if not -1.0 <= rho_bar <= 1.0:
        raise UniverseError("rho_bar must be a correlation")
    denom = sample_inflation(k, rho_bar)
    if denom <= 0.0:
        # Only reachable with a strongly negative average correlation, where the
        # equal-weighted mean is less variable than independence. Treating that
        # as k independent bets is the conservative reading.
        return float(k)
    return k / denom


def effective_observations(n: int, k: int, rho_bar: float) -> float:
    """Independent observations behind `n` pooled trades across `k` symbols."""
    if n < 0:
        raise UniverseError("n must be >= 0")
    return n / sample_inflation(k, rho_bar)


# ----------------------------------------------------------------------
# Grouping
# ----------------------------------------------------------------------
def cluster_by_correlation(
    returns: dict[str, list[float | None]],
    symbols: Iterable[str] | None = None,
    threshold: float = 0.8,
    min_pairs: int = 30,
) -> list[list[str]]:
    """Groups of symbols whose correlation exceeds `threshold`, transitively.

    Connected components rather than a dendrogram: the output feeds a decision
    ("one name per group"), and a component is the smallest thing that is
    guaranteed to have no cross-group pair above the threshold. A cut through a
    dendrogram gives no such guarantee, and the difference is invisible in the
    result.

    Threshold rather than the more usual distance metric because it is the
    quantity a reviewer can check against the measured pair correlations.

    Symbols that score against nothing are singletons and are INCLUDED: they are
    by definition the most diversifying names available, so dropping them would
    discard exactly what the caller is looking for.
    """
    if not -1.0 <= threshold <= 1.0:
        raise UniverseError("threshold must be a correlation")
    names = sorted(symbols if symbols is not None else returns)
    parent = {s: s for s in names}

    def find(s: str) -> str:
        root = s
        while parent[root] != root:
            root = parent[root]
        # Path compression, so a long chain of merges does not turn the next
        # lookup into a walk.
        while parent[s] != root:
            parent[s], s = root, parent[s]
        return root

    for (a, b), c in pairwise_correlations(returns, names, min_pairs).items():
        if c >= threshold:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    groups: dict[str, list[str]] = {}
    for s in names:
        groups.setdefault(find(s), []).append(s)
    # Largest first, then alphabetical: deterministic output so a test or a run
    # does not depend on dict ordering.
    return sorted((sorted(v) for v in groups.values()), key=lambda g: (-len(g), g))


def select_diversified(
    returns: dict[str, list[float | None]],
    symbols: Iterable[str] | None = None,
    max_symbols: int | None = None,
    threshold: float = 0.8,
    liquidity: dict[str, float] | None = None,
    min_pairs: int = 30,
) -> list[str]:
    """At most one symbol per correlation group, most liquid first.

    The result is a universe whose members are not near-duplicates of each
    other by construction, which is the property the three-major backtest
    lacked. `max_symbols` truncates by the same liquidity ordering, so a bigger
    cap expands the universe instead of reshuffling it.

    Liquidity defaults to equal (alphabetical) instead of being inferred: the
    caller holds the volume data, and guessing here would silently pick a
    different universe whenever the guess disagreed.
    """
    if max_symbols is not None and max_symbols < 1:
        raise UniverseError("max_symbols must be >= 1 when given")
    names = sorted(symbols if symbols is not None else returns)
    groups = cluster_by_correlation(returns, names, threshold, min_pairs)

    def rank(s: str) -> tuple[float, str]:
        # Descending liquidity, then name: a total order, so ties cannot make
        # the output depend on iteration order.
        return (-(liquidity or {}).get(s, 0.0), s)

    picked = [sorted(g, key=rank)[0] for g in groups]
    if max_symbols is not None and len(picked) > max_symbols:
        picked = sorted(picked, key=rank)[:max_symbols]
    return sorted(picked)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def correlation_report(
    returns: dict[str, list[float | None]],
    symbols: Iterable[str] | None = None,
    threshold: float = 0.8,
    pooled_observations: int | None = None,
    min_pairs: int = 30,
) -> str:
    """The numbers above, printed with their definitions attached.

    Every figure carries its name because this project has already drawn a wrong
    conclusion from an unlabelled statistic once. `pooled_observations` is
    optional: it is the trade count whose independence is in question.
    """
    names = sorted(symbols if symbols is not None else returns)
    k = len(names)
    rho = mean_pairwise_correlation(returns, names, min_pairs)
    if rho is None:
        return f"{k} symbols: no pair could be scored"
    pairs = pairwise_correlations(returns, names, min_pairs)
    groups = cluster_by_correlation(returns, names, threshold, min_pairs)
    biggest = max(len(g) for g in groups) if groups else 0

    lines = [
        f"{k} symbols, {len(pairs)} scoreable pairs",
        f"  mean pairwise correlation  {rho:+.3f}"
        f"   (range {min(pairs.values()):+.3f} .. {max(pairs.values()):+.3f})",
        f"  effective_bets             {effective_bets(k, rho):.2f}"
        "   (uncorrelated series: k / (1 + (k-1)*rho))",
        f"  sample_inflation           {sample_inflation(k, rho):.2f}"
        "   (divisor on a pooled count: 1 + (k-1)*rho)",
        f"  groups at rho >= {threshold:.2f}      {len(groups)}"
        f"   (largest {biggest})",
    ]
    if pooled_observations is not None:
        eff = effective_observations(pooled_observations, k, rho)
        lines.append(
            f"  {pooled_observations} pooled observations -> {eff:.0f} independent"
        )
    lines.append(
        "  NOTE: these are the sample-mean definitions. The participation ratio "
        "of the eigenvalue spectrum (counts factors, not series) is larger for "
        "the same panel and is deliberately not used here."
    )
    return "\n".join(lines)


def liquidity_medians(panel) -> dict[str, float]:
    """Median dollar volume per symbol, for ranking a universe by tradability.

    Median rather than mean: one listing spike should not decide which of two
    symbols is the liquid one.

    Prices are read through `panel.price` rather than off `panel.close` so this
    stays on the same interface the rest of the module uses - reaching into the
    raw column would also pick up rows the `price` accessor deliberately
    filters (NaN, missing).
    """
    if not getattr(panel, "volume", None):
        raise UniverseError("panel has no volume, cannot rank liquidity")
    out: dict[str, float] = {}
    for s in panel.symbols:
        vols = panel.volume.get(s) or []
        vals = []
        for i in range(len(vols)):
            v = vols[i]
            if v is None:
                continue
            c = panel.price(s, i)
            if c is None:
                continue
            vals.append(c * v)
        if vals:
            out[s] = statistics.median(vals)
    return out


def coverage(panel) -> dict[str, float]:
    """Fraction of rows where each symbol has a usable price.

    Needed before pooling anything across symbols. A universe selected on
    liquidity alone will contain recent listings, and a symbol that only
    existed for the last 66 days of a 170-day window contributes trades from a
    *different period* - so the pooled statistics mix windows and the comparison
    is no longer like-for-like. Measured on one 35-symbol pool: 31 symbols at
    4896/4896 rows, and four between 8.9% and 74.5%.
    """
    total = len(panel.times)
    if total == 0:
        raise UniverseError("panel has no rows")
    out: dict[str, float] = {}
    for s in panel.symbols:
        have = sum(1 for i in range(total) if panel.price(s, i) is not None)
        out[s] = have / total
    return out


def filter_by_coverage(panel, min_fraction: float = 0.95) -> list[str]:
    """Symbols present for at least `min_fraction` of the panel's rows.

    Default 0.95 rather than 1.0: a handful of missing hours is a data gap, not
    a different window, and requiring perfection would drop symbols over a few
    seconds of downtime.
    """
    if not 0.0 < min_fraction <= 1.0:
        raise UniverseError("min_fraction must be in (0, 1]")
    cov = coverage(panel)
    return sorted(s for s, f in cov.items() if f >= min_fraction)


def liquidity_quintiles(panel, buckets: int = 5) -> list[list[str]]:
    """Symbols split into `buckets` liquidity tiers by median dollar volume.

    Returns least liquid first, so index 0 is the tier a capacity-constrained
    strategy is *forced* into rather than the one it would choose.

    Finer than `cross_section.liquid_split`'s halves, and that resolution is the
    point: on the researched universe the reversal effect is present in tiers
    1-4 and absent in tier 5, which a two-way split reports as "present in the
    illiquid half" and hides the fact that the boundary is at the top rather
    than the middle.

    Median rather than mean dollar volume per bar: one listing spike should not
    move a symbol a whole tier. Tiers are equal-sized (a remainder goes to the
    last, most liquid one) so that a per-tier comparison is not partly a
    comparison of tier sizes.
    """
    if buckets < 2:
        raise UniverseError("buckets must be >= 2")
    if not getattr(panel, "volume", None):
        raise UniverseError("panel has no volume, cannot rank liquidity")
    dv = liquidity_medians(panel)
    if len(dv) < buckets:
        raise UniverseError(f"need at least {buckets} symbols with volume")
    order = sorted(dv, key=lambda s: dv[s])
    size = len(order) // buckets
    out = []
    for i in range(buckets):
        lo = i * size
        hi = (i + 1) * size if i < buckets - 1 else len(order)
        out.append(order[lo:hi])
    return out


def is_perp_name(symbol: str) -> bool:
    """False for the spot pairs the archive interleaves with perps.

    Hyperliquid names spot pairs `@<index>`; the archive holds both, and on the
    2026-09-15 partition that is 117 of 882 rows. They are not tradable by this
    system at all - the bot reads its universe from the perp `meta` endpoint -
    so a universe selected purely on dollar volume can pick one up and then
    fail at execution.

    Measured before adding this: the `@` names were 1.8% of dollar volume but
    took 4 of the top 20 ranks, which is high enough to enter any top-N cut.
    Their prices also run to `2e-07`, so a "return" computed on one is a ratio
    of two rounding errors and its correlation with anything is noise.
    """
    return not symbol.startswith("@")


def tradeable_names(symbols: Iterable[str]) -> list[str]:
    """The subset a perp strategy could actually route, sorted."""
    return sorted(s for s in symbols if is_perp_name(s))
