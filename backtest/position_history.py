"""Whale position history from the Reservoir archive.

The smart money factor reads per-wallet positions. Historically the only way to
get those was `userFunding`, which yields a position size per wallet per hour
but no entry price - see `whale_history.py`. The archive's daily snapshots
carry more: size, signed notional and entry price, for every open position on
the venue.

That extra column matters more than it looks. Without an entry price the
unrealised PnL of each wallet is unknown, so the scorer has to drop the wallet
quality term and substitute the midpoint of its range. With it, the real
`winner_ratio` can be computed and confidence is the number production would
have produced. See `SmartMoneySnapshot.pnl_available`.

Three honest limits, all of which are reported with every run:

  * **The wallet set is not the leaderboard.** Production picks wallets by
    profitability over day/week/month windows. Rebuilding that needs the fills
    archive; this module approximates it with size and persistence, which are
    different things. A wallet that holds a large position is not thereby a
    wallet that is right.
  * **Selection uses only data before the backtest window.** Choosing wallets
    from the window being measured would leak the answer - the survivors are
    exactly the ones that did well.
  * **Snapshots are daily.** Production refreshes every 15 minutes. The
    backtest's view of whale positioning therefore lags by up to a day.

The snapshot files are partitioned `date=YYYY-MM-DD` with no timestamp column,
so the instant is taken as the end of that UTC day. The archive's own filenames
carry a block timestamp, but the sync flattens the filename and drops it; the
resulting error is under an hour against a 26-hour staleness window.
"""

from __future__ import annotations

import calendar
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .whale_history import WalletPositionSeries

#: Hyperliquid addresses are 20-byte hex. Validated before use because the
#: leaderboard loader builds an SQL `IN` list out of them by string
#: interpolation, and a value containing a quote would change the statement.
_WALLET_RE = re.compile(r"^0x[0-9a-f]{40}$")

DEFAULT_STORE = "data/canonical/positions"

#: Production's `max_wallets`. Kept equal so the factor runs on a comparably
#: sized sample rather than a much larger or smaller one.
DEFAULT_LIMIT = 150
#: Days a wallet must appear on to be eligible. A quarter of the live
#: leaderboard's longest window: long enough to exclude one-off whales, short
#: enough that the pool does not collapse.
DEFAULT_MIN_DAYS = 60
#: Stands in for the config's `min_account_value`. The synced column set does
#: not include account value - it was dropped in favour of `entry_price`, which
#: is needed for the quality term - so position size is the proxy.
DEFAULT_MIN_NOTIONAL = 10_000.0

#: Snapshots are ~24h apart, so a position stays current for a little over a
#: day. Below this the factor would see an empty book for most bars; far above
#: it would carry positions through the archive's known 51-day gap.
DEFAULT_MAX_AGE_MS = 26 * 3_600_000

#: Days between the last usable snapshot and the next one. The archive has no
#: ABCI state for 2025-10-25 through 2025-12-14; a window spanning it will see
#: the book go empty, which is a property of the source, not of this code.
KNOWN_GAP = ("2025-10-25", "2025-12-14")


class PositionHistoryError(RuntimeError):
    """The archive could not be read or held no usable history."""


@dataclass
class PositionHistory:
    """Selected wallets and their reconstructed series."""

    series: dict[str, WalletPositionSeries] = field(default_factory=dict)
    market: str = ""
    cutoff_pool: int = 0
    #: How the wallet set was chosen, in the words of whichever loader built
    #: it. The two loaders select on entirely different grounds - one on
    #: position size and persistence, the other on a reconstructed PnL ranking -
    #: so a shared description would be wrong for one of them.
    selection: str = ""
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.series)

    def summary(self) -> str:
        if not self.series:
            return "  no whale snapshot history selected"
        days = {len(s) for s in self.series.values()}
        points = sum(len(s) for s in self.series.values())
        lines = [
            f"  wallets selected: {len(self.series)}",
            f"  position points: {points:,} "
            f"({min(days)}-{max(days)} per wallet)",
        ]
        if self.selection:
            lines.append(f"  basis: {self.selection}")
        if self.cutoff_pool:
            pct = len(self.series) / self.cutoff_pool * 100
            lines.append(
                f"  {len(self.series):,} of {self.cutoff_pool:,} eligible wallets "
                f"held a position on a selected day ({pct:.1f}%)"
            )
        return "\n".join(lines)


def end_of_day_ms(day: str) -> int:
    """UTC milliseconds at the end of a `YYYY-MM-DD` day."""
    year, month, date = (int(part) for part in day.split("-"))
    return (calendar.timegm((year, month, date, 23, 59, 59)) ) * 1000


def _run_sql(sql: str, timeout: int = 900) -> list[dict]:
    """Run a query and parse its JSON output.

    The statement goes in on **stdin**, not as a `-c` argument. A single
    argument is capped at 128 KB by the kernel (`MAX_ARG_STRLEN`), and the
    leaderboard loader builds an `IN` list of every selected wallet - 3,214 of
    them is already 142 KB, so `-c` fails with "Argument list too long" on a
    query that is otherwise valid. Stdin has no such limit.
    """
    try:
        proc = subprocess.run(
            ["duckdb", "-json"],
            input=sql,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise PositionHistoryError("duckdb was not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise PositionHistoryError(f"duckdb timed out after {timeout}s") from exc

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise PositionHistoryError(
            f"duckdb failed: {detail[-1] if detail else 'no output'}"
        )
    text = proc.stdout.strip()
    if not text:
        return []
    try:
        rows = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PositionHistoryError(
            f"unparseable duckdb output: {text[:200]}"
        ) from exc
    if not isinstance(rows, list):
        raise PositionHistoryError("unexpected duckdb output shape")
    return rows


def load_position_history(
    store: str | Path = DEFAULT_STORE,
    market: str = "BTC",
    *,
    select_before_ms: int,
    limit: int = DEFAULT_LIMIT,
    min_days: int = DEFAULT_MIN_DAYS,
    min_notional: float = DEFAULT_MIN_NOTIONAL,
) -> PositionHistory:
    """Select wallets from pre-window data, then load their full series.

    Selection and loading are one query so the cutoff is applied to a single
    consistent snapshot of the archive. Splitting them would let a file land
    between the two and make the selection unreproducible.
    """
    root = Path(store)
    if not root.exists():
        raise PositionHistoryError(
            f"no snapshot store at {root}. Run: "
            "python sync_reservoir.py --datasets positions"
        )
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise PositionHistoryError(f"{root} holds no parquet files")

    # The cutoff is compared as a string: the partition names sort
    # lexicographically in date order, so no cast is needed.
    cutoff_day = _day_before(select_before_ms)
    glob = str(root / "*.parquet")
    symbol = market.upper()

    sql = f"""
WITH p AS (
    SELECT user,
           regexp_extract(filename, 'date=(\\d{{4}}-\\d{{2}}-\\d{{2}})', 1) AS day,
           size, notional, entry_price
    FROM read_parquet('{glob}') AS t(user, market, size, notional, entry_price, filename)
    WHERE market = '{symbol}' AND size <> 0
),
eligible AS (
    SELECT user, MEDIAN(ABS(notional)) AS size_score
    FROM p
    WHERE day < '{cutoff_day}'
    GROUP BY user
    HAVING COUNT(DISTINCT day) >= {int(min_days)}
       AND MEDIAN(ABS(notional)) >= {float(min_notional)}
),
sel AS (
    SELECT user FROM eligible ORDER BY size_score DESC LIMIT {int(limit)}
),
pool AS (SELECT COUNT(*) AS n FROM eligible)
SELECT p.user, p.day, p.size, p.entry_price, pool.n AS pool
FROM p JOIN sel USING (user) CROSS JOIN pool
ORDER BY p.user, p.day;
"""

    rows = _run_sql(sql)
    if not rows:
        raise PositionHistoryError(
            f"no {symbol} wallets met the selection rules "
            f"(min_days={min_days}, min_notional={min_notional:,.0f}) "
            f"before {cutoff_day}"
        )

    # Group the flat result into per-wallet series. A list of points is the
    # shape `HistoricalSmartMoney` already consumes, so the factor's scoring
    # path stays untouched.
    per_wallet: dict[str, dict[str, list]] = {}
    for row in rows:
        wallet = str(row.get("user", "")).lower()
        if not wallet:
            continue
        # Coerce every value before appending any of them. Appending inside the
        # `try` lets the second conversion raise after the first has already
        # been written, which leaves `points` and `entries` different lengths -
        # and the lookup helpers index each series independently, so the
        # mismatch would surface as a missing entry price for one day rather
        # than as an error.
        try:
            ts = end_of_day_ms(str(row["day"]))
            size = float(row["size"])
            entry_price = float(row["entry_price"])
        except (KeyError, TypeError, ValueError):
            continue
        parts = per_wallet.setdefault(wallet, {"size": [], "entry": []})
        parts["size"].append((ts, size))
        parts["entry"].append((ts, entry_price))

    series: dict[str, WalletPositionSeries] = {}
    for wallet, parts in per_wallet.items():
        if not parts["size"]:
            continue
        series[wallet] = WalletPositionSeries(
            wallet=wallet,
            coin=symbol,
            points=parts["size"],
            entries=parts["entry"],
        )

    if not series:
        raise PositionHistoryError("selected wallets produced no position points")

    try:
        pool_size = int(rows[0].get("pool") or 0)
    except (TypeError, ValueError):
        pool_size = 0

    history = PositionHistory(
        series=series,
        market=symbol,
        cutoff_pool=pool_size,
        selection=(
            f"top {limit} by median notional among wallets seen on "
            f"{min_days}+ days, using data before {cutoff_day} so the choice "
            "cannot see the window it is measured on"
        ),
    )
    history.notes.append(
        "wallet set approximated by position size and persistence, not by "
        "leaderboard profitability: rebuilding that needs the fills archive"
    )
    history.notes.append(
        "snapshots are daily, so the whale view updates once a day where the "
        "live agent refreshes every 15 minutes"
    )
    if _spans_known_gap(series):
        history.notes.append(
            f"window includes the archive's known gap {KNOWN_GAP[0]} to "
            f"{KNOWN_GAP[1]}, where no snapshot exists and the book reads empty"
        )
    return history


def load_leaderboard_history(
    leaderboard,
    market: str = "BTC",
    store: str | Path = DEFAULT_STORE,
) -> PositionHistory:
    """Positions for the wallets the reconstructed leaderboard selected.

    This is the honest version of the wallet set. `load_position_history`
    approximates the leaderboard with size and persistence, because it was
    written before the fills archive was available; this reads the ranking
    itself, built from realised PnL over the same day/week/month windows the
    live agent uses.

    The set varies over time, and that is the point. A wallet contributes a
    point on a day only if it was selected for that day, so a wallet that fell
    out of the ranking stops being consulted immediately rather than being
    carried through the rest of the run on the strength of having once been
    good. A run-wide set would be a set that never existed.
    """
    root = Path(store)
    if not root.exists():
        raise PositionHistoryError(
            f"no snapshot store at {root}. Run: "
            "python sync_reservoir.py --datasets positions"
        )
    # The wallet list is interpolated into SQL, so it is validated first. Every
    # address in the archive is lowercase hex, but "the archive is trustworthy"
    # is an assumption about data, and this is the one place where a stray
    # quote would turn a read into a different statement.
    candidates = sorted(leaderboard.wallets)
    wallets = [w for w in candidates if _WALLET_RE.match(w)]
    rejected = len(candidates) - len(wallets)
    if not wallets:
        raise PositionHistoryError("the leaderboard selected no usable wallets")

    symbol = market.upper()
    glob = str(root / "*.parquet")
    in_list = ",".join(f"'{w}'" for w in wallets)

    sql = f"""
SELECT lower(user) AS wallet,
       regexp_extract(filename, 'date=(\\d{{4}}-\\d{{2}}-\\d{{2}})', 1) AS day,
       size, entry_price
FROM read_parquet('{glob}') AS t(user, market, size, notional, entry_price, filename)
WHERE market = '{symbol}' AND size <> 0 AND lower(user) IN ({in_list})
ORDER BY wallet, day;
"""
    rows = _run_sql(sql)
    if not rows:
        raise PositionHistoryError(
            f"no positions for the {len(wallets)} selected wallets in {symbol}"
        )

    per_wallet: dict[str, dict[str, list]] = {}
    for row in rows:
        wallet = str(row.get("wallet", "")).lower()
        day = str(row.get("day", ""))[:10]
        if not wallet or not day:
            continue
        # Only days the wallet was actually on the board. This single check is
        # what makes the set time-varying.
        persistence = leaderboard.by_day.get(day, {}).get(wallet)
        if persistence is None:
            continue
        # Coerced before any append, so the three parallel series stay the same
        # length even when one row is unparseable. See the same note in
        # `load_position_history`.
        try:
            ts = end_of_day_ms(day)
            size = float(row["size"])
            entry_price = float(row["entry_price"])
            hits = int(persistence)
        except (KeyError, TypeError, ValueError):
            continue
        parts = per_wallet.setdefault(
            wallet, {"size": [], "entry": [], "persist": []}
        )
        parts["size"].append((ts, size))
        parts["entry"].append((ts, entry_price))
        parts["persist"].append((ts, hits))

    series: dict[str, WalletPositionSeries] = {}
    for wallet, parts in per_wallet.items():
        if not parts["size"]:
            continue
        series[wallet] = WalletPositionSeries(
            wallet=wallet,
            coin=symbol,
            points=parts["size"],
            entries=parts["entry"],
            persistences=parts["persist"],
        )
    if not series:
        raise PositionHistoryError(
            "no selected wallet held a position on any day it was selected"
        )

    history = PositionHistory(
        series=series,
        market=symbol,
        cutoff_pool=len(wallets),
        selection=(
            f"reconstructed {len(leaderboard.windows)}-window PnL ranking over "
            f"{len(leaderboard.by_day)} days "
            f"({min(leaderboard.by_day)} -> {max(leaderboard.by_day)}), "
            f"top {leaderboard.top_per_window} per window unioned and capped at "
            f"{leaderboard.max_wallets}; a wallet is counted only on days it "
            "was selected, so the set moves with the ranking"
        ),
    )
    if rejected:
        history.notes.append(
            f"{rejected} wallet address(es) were not valid hex and were "
            "dropped before the query"
        )
    history.notes.append(
        f"wallet set rebuilt from {len(leaderboard.by_day)} days of realised-PnL "
        f"rankings: {len(wallets):,} wallets were selected on at least one day, "
        f"of which {len(series):,} held a {symbol} position on a day they were "
        "selected"
    )
    history.notes.append(
        "a wallet contributes only on days it was selected, so the set changes "
        "with the ranking instead of being fixed for the whole run"
    )
    history.notes.extend(leaderboard.notes)
    return history


def _day_before(ms: int) -> str:
    """The UTC date one day before `ms`, as a `YYYY-MM-DD` string."""
    import time as _time

    return _time.strftime("%Y-%m-%d", _time.gmtime((ms - 86_400_000) / 1000))


def _spans_known_gap(series: dict[str, WalletPositionSeries]) -> bool:
    start = end_of_day_ms(KNOWN_GAP[0])
    end = end_of_day_ms(KNOWN_GAP[1])
    for entry in series.values():
        if entry.points and entry.points[0][0] < end and entry.points[-1][0] > start:
            return True
    return False
