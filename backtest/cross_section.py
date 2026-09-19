"""Cross-sectional portfolios: many symbols, one signal, long and short.

Why this exists
---------------
Every time-series test in this repo came back null or sign-unstable, and the
reason was structural rather than a bad signal: with three symbols the return
series is dominated by the common market factor, so a "signal IC" mostly measures
whether the window happened to trend. Two windows of the same length can then
disagree in sign while neither is wrong.

A cross-sectional test removes the common factor by construction. Each bar ranks
the universe against itself and measures whether the signal separated *relative*
winners from *relative* losers. That is one independent observation per bar
instead of one per window, which is where the statistical power comes from:
measured here, the effective sample goes from 15-24 (a time-series test on a
slow moving aggregate) to 250-500.

What it found
-------------
Short-horizon reversal, in the less liquid half of the universe. IC -0.041 at
24h and -0.046 on 72h volatility, t = -4.1 and -3.3, and the effect survives
costs, worst-case delisting and a random-portfolio control. In the liquid half
the same signal is t = -1.0. Details and caveats in RESEARCH.md.

What this module does NOT do
----------------------------
It does not plug into `engine.Backtester`. That engine holds one position in one
symbol (`max_open_positions: 1`); this holds a basket of both. Forcing them
together would mean rewriting the position model, and the honest sequencing is to
establish the effect standalone before touching the live path.
"""

from __future__ import annotations

import math
import random
import statistics
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Callable, Iterable

#: A signal maps (panel, bar index, symbol) to a value, or None when it cannot be
#: computed. Higher values are expected to have HIGHER forward returns; a reversal
#: signal therefore returns the negated past return rather than a special case
#: being threaded through the portfolio builder.
SignalFn = Callable[["Panel", int, str], "float | None"]


class CrossSectionError(ValueError):
    """Raised when a panel or its use is malformed."""


@dataclass
class Panel:
    """Aligned close/volume series for many symbols.

    Rows are aligned to `times`: `close[symbol][i]` belongs to `times[i]`. A
    missing bar is `None` rather than 0.0, because a delisted symbol and a symbol
    that traded at zero are different things and only one of them should be
    silently tradeable.
    """

    times: list[int]
    symbols: list[str]
    close: dict[str, list[float | None]]
    volume: dict[str, list[float | None]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.times:
            raise CrossSectionError("panel has no timestamps")
        if not self.symbols:
            raise CrossSectionError("panel has no symbols")
        n = len(self.times)
        for s in self.symbols:
            series = self.close.get(s)
            if series is None:
                raise CrossSectionError(f"no close series for {s}")
            if len(series) != n:
                raise CrossSectionError(
                    f"close series for {s} has {len(series)} rows, expected {n}"
                )

    def __len__(self) -> int:
        return len(self.times)

    def price(self, symbol: str, i: int) -> float | None:
        """Close at row `i`, or None if the symbol has no bar there."""
        series = self.close.get(symbol)
        if series is None or not (0 <= i < len(series)):
            return None
        value = series[i]
        # NaN is treated as missing too: it survives arithmetic and would
        # otherwise propagate into a weight as a silent zero.
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)

    def present(self, i: int) -> list[str]:
        """Symbols with a usable price at row `i`."""
        return [s for s in self.symbols if self.price(s, i) is not None]

    def last_row_of(self, symbol: str) -> int:
        """Last row index where the symbol has a price, or -1."""
        series = self.close.get(symbol) or []
        for i in range(len(series) - 1, -1, -1):
            v = series[i]
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                return i
        return -1


# ----------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------
def past_return(lookback: int) -> SignalFn:
    """Trailing return over `lookback` rows.

    Returned as-is, so a positive value means the symbol rose. Pair it with
    `negate` for a reversal signal, which is what the research found.
    """

    def signal(panel: Panel, i: int, symbol: str) -> float | None:
        if i - lookback < 0:
            return None
        now = panel.price(symbol, i)
        then = panel.price(symbol, i - lookback)
        if now is None or then is None or then <= 0:
            return None
        return (now - then) / then

    return signal


def realized_vol(lookback: int) -> SignalFn:
    """Standard deviation of row-to-row returns over `lookback` rows."""

    def signal(panel: Panel, i: int, symbol: str) -> float | None:
        if i - lookback < 0:
            return None
        rets: list[float] = []
        for k in range(i - lookback + 1, i + 1):
            a = panel.price(symbol, k - 1)
            b = panel.price(symbol, k)
            if a is None or b is None or a <= 0:
                continue
            rets.append((b - a) / a)
        if len(rets) < lookback:
            return None
        return statistics.pstdev(rets)

    return signal


def turnover_signal(lookback: int) -> SignalFn:
    """Average dollar volume over `lookback` rows."""

    def signal(panel: Panel, i: int, symbol: str) -> float | None:
        if i - lookback < 0 or not panel.volume:
            return None
        total = 0.0
        count = 0
        for k in range(i - lookback + 1, i + 1):
            c = panel.price(symbol, k)
            v = panel.volume.get(symbol, [])[k] if panel.volume else None
            if c is None or v is None:
                continue
            total += c * v
            count += 1
        return total / count if count else None

    return signal


def negate(signal: SignalFn) -> SignalFn:
    """Flip a signal's sign, so its high end becomes the short leg."""

    def flipped(panel: Panel, i: int, symbol: str) -> float | None:
        v = signal(panel, i, symbol)
        return None if v is None else -v

    return flipped


def random_signal(seed: int) -> SignalFn:
    """Noise with the same shape as a real signal.

    This is the control that mattered most in the research: a long-short basket
    in a falling market earns money from the short leg whether or not the signal
    ranks anything, so a result is only meaningful next to the distribution
    produced by ranking at random.
    """
    rng = random.Random(seed)
    cache: dict[tuple[int, str], float] = {}

    def signal(panel: Panel, i: int, symbol: str) -> float | None:
        key = (i, symbol)
        if key not in cache:
            cache[key] = rng.random()
        return cache[key]

    return signal


# ----------------------------------------------------------------------
# Portfolio
# ----------------------------------------------------------------------
@dataclass
class CrossSectionConfig:
    """Knobs for the basket. Defaults are the ones the research used."""

    #: Fraction of the ranked universe in each leg. 0.2 = quintiles.
    quantile: float = 0.2
    #: Minimum ranked symbols for a period to trade at all. Below this the
    #: quantiles are a handful of names and the result is one symbol's move.
    min_symbols: int = 20
    #: Fee per leg in basis points, charged on the notional traded. Entry and
    #: exit are separate legs.
    fee_bps: float = 3.5
    #: What a symbol's exit price is worth when it has no bar at the exit row.
    #: 0.0 assumes total loss, which is the honest reading for a pair that stops
    #: trading; 1.0 assumes the position is closed at the last price seen, which
    #: understates the loss. The two bracket the answer.
    late_exit_factor: float = 0.0
    long_only: bool = False


@dataclass
class PeriodResult:
    entry_row: int
    exit_row: int
    entry_time: int
    exit_time: int
    n_long: int
    n_short: int
    long_return: float
    short_return: float
    turnover: float
    cost: float
    net_return: float


@dataclass
class CrossSectionResult:
    periods: list[PeriodResult] = field(default_factory=list)
    symbol: str = ""
    config: CrossSectionConfig | None = None

    def __len__(self) -> int:
        return len(self.periods)

    @property
    def gross_return_pct(self) -> float:
        """Compounded return before costs."""
        eq = 1.0
        for p in self.periods:
            eq *= 1.0 + (p.long_return - p.short_return)
        return (eq - 1.0) * 100.0

    @property
    def net_return_pct(self) -> float:
        eq = 1.0
        for p in self.periods:
            eq *= 1.0 + p.net_return
        return (eq - 1.0) * 100.0

    @property
    def cost_pct(self) -> float:
        return sum(p.cost for p in self.periods) * 100.0

    @property
    def mean_turnover_pct(self) -> float:
        if not self.periods:
            return 0.0
        return statistics.mean(p.turnover for p in self.periods) * 100.0

    @property
    def mean_net_pct(self) -> float:
        if not self.periods:
            return 0.0
        return statistics.mean(p.net_return for p in self.periods) * 100.0

    @property
    def median_net_pct(self) -> float:
        """Reported next to the mean because a mean carried by three periods is
        not the same claim as one that holds period to period."""
        if not self.periods:
            return 0.0
        return statistics.median(p.net_return for p in self.periods) * 100.0

    def summary(self) -> str:
        if not self.periods:
            return "no periods"
        return (
            f"n={len(self.periods)}  gross {self.gross_return_pct:+.2f}%  "
            f"cost {self.cost_pct:.2f}%  net {self.net_return_pct:+.2f}%  "
            f"mean/period {self.mean_net_pct:+.3f}%  "
            f"median/period {self.median_net_pct:+.3f}%  "
            f"turnover {self.mean_turnover_pct:.1f}%"
        )


class CrossSectionBacktester:
    """Ranks the universe each period and holds the two tails.

    Deliberately independent of `Backtester`: no discipline gates, no single
    position, no stops. Those belong to the directional engine and reusing them
    here would smuggle in assumptions (one position at a time, per-symbol
    cooldowns) that do not apply to a basket.
    """

    def __init__(
        self,
        panel: Panel,
        signal: SignalFn,
        hold_rows: int,
        config: CrossSectionConfig | None = None,
        start_row: int = 0,
    ) -> None:
        if hold_rows < 1:
            raise CrossSectionError("hold_rows must be >= 1")
        self.panel = panel
        self.signal = signal
        self.hold_rows = int(hold_rows)
        self.cfg = config or CrossSectionConfig()
        self.start_row = int(start_row)

    def run(self) -> CrossSectionResult:
        panel = self.panel
        cfg = self.cfg
        result = CrossSectionResult(symbol=",".join(panel.symbols[:3]), config=cfg)

        # Tracked as actual name sets rather than an assumed turnover, so the
        # cost reflects the overlap between consecutive baskets.
        prev_long: set[str] = set()
        prev_short: set[str] = set()

        row = self.start_row
        while row + self.hold_rows < len(panel):
            exit_row = row + self.hold_rows
            ranked = self._rank(row)
            if ranked is None:
                row += self.hold_rows
                continue
            long_leg, short_leg = ranked

            long_ret = self._leg_return(long_leg, row, exit_row)
            short_ret = (
                0.0 if cfg.long_only else self._leg_return(short_leg, row, exit_row)
            )
            if long_ret is None:
                row += self.hold_rows
                continue
            if short_ret is None:
                short_ret = 0.0

            long_turn = _turnover(long_leg, prev_long)
            short_turn = 0.0 if cfg.long_only else _turnover(short_leg, prev_short)
            turnover = (long_turn + short_turn) / (1.0 if cfg.long_only else 2.0)
            # Two legs, each turning over `turnover` of its notional.
            legs = 1.0 if cfg.long_only else 2.0
            cost = turnover * legs * cfg.fee_bps / 10_000.0

            result.periods.append(
                PeriodResult(
                    entry_row=row,
                    exit_row=exit_row,
                    entry_time=panel.times[row],
                    exit_time=panel.times[exit_row],
                    n_long=len(long_leg),
                    n_short=len(short_leg),
                    long_return=long_ret,
                    short_return=short_ret,
                    turnover=turnover,
                    cost=cost,
                    net_return=long_ret - short_ret - cost,
                )
            )
            prev_long, prev_short = long_leg, short_leg
            row += self.hold_rows

        return result

    # ------------------------------------------------------------------
    def _rank(self, row: int) -> tuple[set[str], set[str]] | None:
        """Split the ranked universe into the two tails, or None to skip."""
        cfg = self.cfg
        scored: list[tuple[float, str]] = []
        for s in self.panel.symbols:
            if self.panel.price(s, row) is None:
                continue
            v = self.signal(self.panel, row, s)
            if v is None or math.isnan(v):
                continue
            scored.append((v, s))
        if len(scored) < cfg.min_symbols:
            return None

        scored.sort(key=lambda t: t[0])
        k = max(1, int(len(scored) * cfg.quantile))
        low = {s for _, s in scored[:k]}          # signal says "will fall"
        high = {s for _, s in scored[-k:]}        # signal says "will rise"
        if cfg.long_only:
            # Long the high end, hold cash otherwise. No short leg at all.
            return high, set()
        # Short the low end, long the high end. The signal is direction-neutral:
        # callers that mean "reversal" pass a negated momentum, they do not get
        # a different portfolio builder.
        return high, low

    def _leg_return(
        self, names: set[str], entry_row: int, exit_row: int
    ) -> float | None:
        """Equal-weighted return of a leg, honouring deaths inside the period."""
        rets: list[float] = []
        for s in names:
            entry = self.panel.price(s, entry_row)
            if entry is None or entry <= 0:
                continue
            exit_px = self.panel.price(s, exit_row)
            if exit_px is not None and exit_px > 0:
                rets.append((exit_px - entry) / entry)
                continue
            # No bar at the exit. The symbol stopped trading inside the period,
            # which is the case a long-the-losers basket is most exposed to.
            # Bracket it rather than dropping the name: dropping it is exactly
            # the survivorship bias that flatters such a strategy.
            rets.append(self.cfg.late_exit_factor - 1.0)
        if not rets:
            return None
        return sum(rets) / len(rets)


def _turnover(new: set[str], old: set[str]) -> float:
    """Fraction of a leg's names replaced, as a share of the new leg's size."""
    if not new:
        return 0.0
    return len(new - old) / len(new)


# ----------------------------------------------------------------------
# Comparison helper
# ----------------------------------------------------------------------
def random_benchmark(
    panel: Panel,
    hold_rows: int,
    config: CrossSectionConfig | None = None,
    runs: int = 30,
    seed: int = 1000,
    start_row: int = 0,
) -> list[float]:
    """Net returns from ranking at random, for the same panel and settings.

    The point of the control: a long-short basket in a falling market earns from
    the short leg regardless of the signal, so a rule's return means nothing
    until it is placed against this distribution.
    """
    out: list[float] = []
    for i in range(runs):
        bt = CrossSectionBacktester(
            panel, random_signal(seed + i), hold_rows, config, start_row
        )
        out.append(bt.run().net_return_pct)
    return out


def percentile_of(value: float, distribution: Iterable[float]) -> float:
    """Where `value` sits in `distribution`, 0-100."""
    dist = list(distribution)
    if not dist:
        return 0.0
    below = sum(1 for v in dist if v < value)
    return below / len(dist) * 100.0


# ----------------------------------------------------------------------
# Archive loader (kept apart so the core above stays testable offline)
# ----------------------------------------------------------------------
def load_archive_panel(
    coins: Iterable[str],
    interval_minutes: int = 240,
    start_ms: int | None = None,
    end_ms: int | None = None,
    archive: str = "data/canonical/candles",
    min_coverage: int = 0,
) -> Panel:
    """Build a `Panel` from the one-second candle archive.

    `min_coverage` drops symbols with fewer than that many bars in range. Use it
    with care: requiring long coverage removes symbols whose data stops early,
    and those are the delistings a long-the-losers basket needs to see. The
    research deliberately set it to 0 and let `late_exit_factor` handle deaths.
    """
    import json
    import subprocess
    from pathlib import Path

    root = Path(archive)
    if not root.is_dir():
        raise CrossSectionError(
            f"no candle archive at {root}. Run: "
            "python sync_reservoir.py --datasets candles"
        )
    names = [c for c in coins if c]
    if not names:
        raise CrossSectionError("no coins requested")

    where = ["coin IN (" + ", ".join(f"'{c}'" for c in names) + ")"]
    if start_ms is not None:
        where.append(f"timestamp >= to_timestamp({int(start_ms) // 1000})")
    if end_ms is not None:
        where.append(f"timestamp <  to_timestamp({int(end_ms) // 1000})")

    sql = f"""
SELECT coin,
       epoch_ms(time_bucket(INTERVAL '{int(interval_minutes)} minutes', timestamp)) AS t,
       arg_max(close, timestamp)::DOUBLE AS c,
       SUM(volume)::DOUBLE AS v
FROM read_parquet('{root}/*.parquet') AS x(
    coin, timestamp, open, high, low, close, volume, filename)
WHERE {' AND '.join(where)}
GROUP BY 1, 2
ORDER BY 2, 1;
"""
    try:
        proc = subprocess.run(
            ["duckdb", "-json"], input=sql, capture_output=True, text=True,
            timeout=1800,
        )
    except FileNotFoundError as exc:
        raise CrossSectionError("duckdb was not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise CrossSectionError("reading the candle archive timed out") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise CrossSectionError(
            f"reading the candle archive failed: "
            f"{detail[-1] if detail else 'no output'}"
        )
    rows = json.loads(proc.stdout or "[]")
    return build_panel(rows, min_coverage=min_coverage)


def build_panel(rows: list[dict], min_coverage: int = 0) -> Panel:
    """Assemble rows of (coin, t, c, v) into an aligned panel.

    Split out from the loader so tests can build panels without duckdb.
    """
    by_time: dict[int, dict[str, tuple[float, float]]] = {}
    for r in rows:
        t = int(r["t"])
        by_time.setdefault(t, {})[str(r["coin"])] = (
            float(r["c"]), float(r.get("v") or 0.0)
        )
    if not by_time:
        raise CrossSectionError("no rows to build a panel from")

    times = sorted(by_time)
    seen: dict[str, int] = {}
    for t in times:
        for s in by_time[t]:
            seen[s] = seen.get(s, 0) + 1
    symbols = sorted(s for s, n in seen.items() if n >= min_coverage)
    if not symbols:
        raise CrossSectionError(
            f"no symbol reaches min_coverage={min_coverage} bars"
        )

    close: dict[str, list[float | None]] = {s: [] for s in symbols}
    volume: dict[str, list[float | None]] = {s: [] for s in symbols}
    for t in times:
        row = by_time[t]
        for s in symbols:
            hit = row.get(s)
            if hit is None:
                close[s].append(None)
                volume[s].append(None)
            else:
                close[s].append(hit[0])
                volume[s].append(hit[1])
    return Panel(times=times, symbols=symbols, close=close, volume=volume)


def liquid_split(
    panel: Panel, fraction: float = 0.5
) -> tuple[set[str], set[str]]:
    """Split symbols into (more liquid, less liquid) by median dollar volume.

    The research found the reversal effect only in the less liquid half, so any
    claim about a basket has to say which half it holds. Returning the two sets
    makes that a choice the caller states rather than a default they inherit.
    """
    if not panel.volume:
        raise CrossSectionError("panel has no volume, cannot rank liquidity")
    medians: dict[str, float] = {}
    for s in panel.symbols:
        vals = [
            c * v
            for c, v in zip(panel.close[s], panel.volume[s])
            if c is not None and v is not None
        ]
        if vals:
            medians[s] = statistics.median(vals)
    if not medians:
        raise CrossSectionError("no symbol has usable volume")
    ordered = sorted(medians.values())
    cut = ordered[int(len(ordered) * (1.0 - fraction))]
    liquid = {s for s, v in medians.items() if v >= cut}
    illiquid = set(medians) - liquid
    return liquid, illiquid


def restrict(panel: Panel, symbols: set[str]) -> Panel:
    """A panel holding only `symbols`, keeping the time axis unchanged."""
    keep = [s for s in panel.symbols if s in symbols]
    if not keep:
        raise CrossSectionError("no symbol survives the restriction")
    return Panel(
        times=list(panel.times),
        symbols=keep,
        close={s: list(panel.close[s]) for s in keep},
        volume={s: list(panel.volume[s]) for s in keep} if panel.volume else {},
    )
