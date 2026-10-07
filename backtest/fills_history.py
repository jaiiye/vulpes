"""Whale positions and a per-coin wallet ranking, both from the wide fills.

Why this is not `leaderboard.py`
--------------------------------
`leaderboard.py` already reconstructs the venue's ranking from fills, but from
the **three-column projection** (`address`, `timestamp`, `realized_pnl`). That
is enough to rank wallets *globally* and nothing more, and two consequences
follow directly from the columns that are missing:

  * **No `coin` column, so the ranking cannot be per-coin.** The factor asks
    "what are the good wallets doing in *this* market", and a wallet that made
    its money in a token that is not the one being traded is not evidence about
    this one.
  * **No `crossed` column, so market makers cannot be recognised.** They are
    the majority of realised PnL on the venue by construction - they quote both
    sides all day and collect the spread - so a pure PnL ranking puts them at
    the top. Measured on 2026-03-10, the 57 wallets above 10k fills traded a
    **median of 12 coins** at a **36% taker ratio**, against 74% for the
    1-3 coin wallets. They are not making a directional call, and reading their
    inventory as one is a category error.

The wide projection carries `coin`, `crossed`, `size`, `side`, `start_position`
and `price` across 435 days, which fixes both, and changes a third thing:

  * **Positions are rebuilt per fill, not per daily snapshot.** The snapshot
    path (`position_history.py`) lags by up to a day and carries a known 51-day
    hole; fills are as fine as the archive is.

How the position series is built
--------------------------------
Every fill reports `start_position`, the position *before* it. Rather than
trusting that per row, the series is rebuilt by accumulation:

    position_t = first_start_position + cumsum(signed size up to t)

which uses `start_position` only for its one reliable contribution - where the
wallet entered the archive - and is then self-consistent by construction. On
two high-frequency wallets (56,985 fills) the two agree on **99.87-100%** of
rows, so the accumulation reproduces the archive's own bookkeeping.

What it does not provide, and every run reports
-----------------------------------------------
  * **No entry price, so no unrealised PnL.** Cost basis would have to be
    rebuilt per wallet by a running weighted average, and nothing here needs
    it yet: the wallet set is chosen by realised PnL, which the archive carries
    directly. `PositionHistory` therefore reports the same
    `pnl_available=False` the funding-derived path does, and the scorer drops
    the quality term rather than guessing it.
  * **Realised PnL only**, for the same reason as `leaderboard.py`: a wallet
    sitting on a large open winner is understated here.
  * **A market-maker filter that is behavioural, not definitive.** The
    thresholds below are set from the measured distribution, not from a label
    the venue publishes, so they exclude the *shape* of market making (many
    fills, low taker share, many coins) rather than the activity itself.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .position_history import PositionHistory, PositionHistoryError
from .whale_history import WalletPositionSeries

#: The wide projection, not `data/canonical/fills`. The three-column store has
#: no `coin` and no `crossed`, which are the two columns this module exists for.
DEFAULT_FILLS_STORE = "data/fills_wide/fills"

#: Same three horizons as the live config and as `leaderboard.py`, so a
#: discrepancy against either is about the data rather than the rule.
DEFAULT_WINDOWS: tuple[tuple[str, int], ...] = (
    ("day", 1),
    ("week", 7),
    ("month", 30),
)
#: Production's `top_per_window` and `max_wallets`, kept equal so the set is
#: comparably sized to the one the live agent consults.
DEFAULT_TOP_PER_WINDOW = 60
DEFAULT_MAX_WALLETS = 150

#: A wallet averaging more fills per day than this is quoting, not taking a
#: view. The 10k+ bucket on the measured day is ~12 coins wide; this bound sits
#: well above every directional wallet observed and well below the quoters.
DEFAULT_MAX_FILLS_PER_DAY = 2_000.0
#: Share of fills that cross the spread. Measured medians: 74% for 1-coin
#: wallets against 36% for the 10k+ quoters.
DEFAULT_MIN_TAKER_RATIO = 0.5

#: Where a completed rebuild is cached. The windowed scan costs minutes over a
#: billion fills while the backtest it feeds runs in seconds, so recomputing it
#: per invocation is what makes the comparisons this feeds impractical rather
#: than slow.
DEFAULT_CACHE = "data/canonical/fills_history"

#: Hyperliquid addresses are 20-byte hex. Validated before interpolation
#: because the wallet list is built into an SQL `IN` list by string
#: concatenation, and one stray quote would change the statement.
_WALLET_RE = re.compile(r"^0x[0-9a-f]{40}$")

#: Seconds per day, for turning a trailing window into a lower bound.
_DAY_MS = 86_400_000


class FillsHistoryError(RuntimeError):
    """The fills archive could not be read or held nothing usable."""


def _run_sql(sql: str, timeout: int = 1800) -> list[dict]:
    """Run a query and parse its JSON output.

    The statement arrives on **stdin**, not as a `-c` argument: a single
    argument is capped at 128 KB by the kernel (`MAX_ARG_STRLEN`) and the
    wallet list alone can exceed that. Same constraint, same fix as
    `leaderboard._run_sql` and `position_history._run_sql`.
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
        raise FillsHistoryError("duckdb was not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise FillsHistoryError(f"duckdb timed out after {timeout}s") from exc

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise FillsHistoryError(
            f"duckdb failed: {detail[-1] if detail else 'no output'}"
        )
    text = proc.stdout.strip()
    if not text:
        return []
    try:
        rows = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FillsHistoryError(f"unparseable duckdb output: {text[:200]}") from exc
    if not isinstance(rows, list):
        raise FillsHistoryError("unexpected duckdb output shape")
    return rows


def _files_for_window(root: Path, end_ms: int, days: int) -> list[str]:
    """Partition files covering `days` before `end_ms`, as a SQL list literal.

    Passing the file list rather than a glob over the whole store is the
    difference between reading 30 footers and reading 435. The archive is
    partitioned by day, so the trailing window maps straight onto filenames.
    """
    first = (datetime.fromtimestamp(end_ms / 1000, timezone.utc)
             - timedelta(days=days)).date()
    last = datetime.fromtimestamp(end_ms / 1000, timezone.utc).date()
    picked = []
    day = first
    while day <= last:
        path = root / f"date={day.isoformat()}.parquet"
        if path.exists():
            picked.append(f"'{path}'")
        day += timedelta(days=1)
    if not picked:
        raise FillsHistoryError(
            f"no daily files under {root} for the {days} days before "
            f"{last.isoformat()}"
        )
    return picked


def select_wallets(
    store: str | Path = DEFAULT_FILLS_STORE,
    market: str = "BTC",
    *,
    as_of_ms: int,
    windows: tuple[tuple[str, int], ...] = DEFAULT_WINDOWS,
    top_per_window: int = DEFAULT_TOP_PER_WINDOW,
    max_wallets: int = DEFAULT_MAX_WALLETS,
    max_fills_per_day: float | None = DEFAULT_MAX_FILLS_PER_DAY,
    min_taker_ratio: float | None = DEFAULT_MIN_TAKER_RATIO,
) -> tuple[dict[str, int], list[str]]:
    """Per-coin realised-PnL ranking, as `{wallet: persistence}` plus notes.

    The rule mirrors `leaderboard.py` - rank each window by realised PnL, take
    the top `top_per_window`, union, weight by how many windows a wallet
    appeared in - with two additions the wide columns allow: the ranking is
    over one coin, and market makers are dropped before the top is taken.

    Filtering **before** the cut matters. Dropping them afterwards would leave
    a shortlist whose length no longer means what it says, and the failure
    would be silent: the set would look full.
    """
    root = Path(store)
    if not root.exists():
        raise FillsHistoryError(
            f"no fills store at {root}. Run: python sync_reservoir.py "
            "--datasets fills --columns wide"
        )
    symbol = market.upper()

    # No wallet can clear the fill-count cut unless it has been seen inside at
    # least the longest window, so one scan of that window answers both the
    # ranking and the filter.
    widest = max(days for _, days in windows)
    files = _files_for_window(root, as_of_ms, widest)
    glob = ", ".join(files)

    having = []
    if max_fills_per_day is not None:
        per_day = float(max_fills_per_day)
        having.append(f"COUNT(*) <= {per_day} * {float(widest)}")
    if min_taker_ratio is not None:
        having.append(
            f"AVG(CASE WHEN crossed THEN 1.0 ELSE 0.0 END) >= "
            f"{float(min_taker_ratio)}"
        )
    having_sql = (" HAVING " + " AND ".join(having)) if having else ""

    # `epoch_ms` on the column rather than a timestamp literal: the archive
    # stores timestamptz, and comparing an integer bound against it would rely
    # on an implicit cast whose timezone is the process default.
    def rank_sql(lo_ms: int, hi_ms: int) -> str:
        return f"""
SELECT lower(address) AS wallet
FROM read_parquet([{glob}])
WHERE coin = '{symbol}'
  AND epoch_ms(timestamp) > {int(lo_ms)} AND epoch_ms(timestamp) <= {int(hi_ms)}
GROUP BY wallet
{having_sql}
ORDER BY SUM(CAST(realized_pnl AS DOUBLE)) DESC
LIMIT {int(top_per_window)};
"""

    selected: dict[str, int] = {}
    for _, days in sorted(windows, key=lambda w: w[1]):
        lo_ms = _window_lower_ms(as_of_ms, days)
        for row in _run_sql(rank_sql(lo_ms, as_of_ms)):
            wallet = str(row.get("wallet") or "").lower()
            if wallet:
                selected[wallet] = selected.get(wallet, 0) + 1

    notes: list[str] = []
    if len(selected) > max_wallets:
        # Cut by persistence first, then alphabetically: the persistence order
        # is the factor's own weighting, and ties have to break deterministically
        # or the same archive yields two different wallet sets.
        ordered = sorted(selected.items(), key=lambda kv: (-kv[1], kv[0]))
        selected = dict(ordered[:max_wallets])
        notes.append(
            f"capped to {max_wallets} wallets by persistence "
            f"(from {len(ordered)})"
        )
    return selected, notes


def _window_lower_ms(as_of_ms: int, days: int) -> int:
    """Exclusive lower bound of a trailing window, in epoch milliseconds."""
    return int(as_of_ms) - days * _DAY_MS


def load_fills_history(
    market: str = "BTC",
    store: str | Path = DEFAULT_FILLS_STORE,
    *,
    select_before_ms: int,
    max_wallets: int = DEFAULT_MAX_WALLETS,
    top_per_window: int = DEFAULT_TOP_PER_WINDOW,
    max_fills_per_day: float | None = DEFAULT_MAX_FILLS_PER_DAY,
    min_taker_ratio: float | None = DEFAULT_MIN_TAKER_RATIO,
    cache: str | Path | None = DEFAULT_CACHE,
) -> PositionHistory:
    """Selected wallets and their per-fill position series in one coin.

    Selection uses only fills before `select_before_ms`, so a backtest cannot
    be told which wallets turned out to be right. Loading then spans the whole
    archive for the selected wallets: the point of the series is to be read
    *forward* from the cutoff, and a wallet chosen on pre-window data is still
    the wallet it was on the day being replayed.
    """
    root = Path(store)
    symbol = market.upper()
    wallets, notes = select_wallets(
        store=store, market=symbol, as_of_ms=select_before_ms,
        top_per_window=top_per_window, max_wallets=max_wallets,
        max_fills_per_day=max_fills_per_day, min_taker_ratio=min_taker_ratio,
    )
    if not wallets:
        raise FillsHistoryError(
            f"no wallet in {symbol} cleared the filters before the cutoff"
        )

    series = _rebuild_series(root, symbol, sorted(wallets), cache)
    if not series:
        raise FillsHistoryError(
            f"the {len(wallets)} selected wallets hold no {symbol} positions"
        )

    notes.append(
        f"ranking is per-coin realised PnL; "
        f"{len(wallets)} wallets selected, {len(series)} carry positions"
    )
    return PositionHistory(
        series=series,
        market=symbol,
        cutoff_pool=0,
        selection=(
            f"fills: {symbol} realised PnL over "
            f"{'/'.join(name for name, _ in DEFAULT_WINDOWS)}, "
            f"market makers excluded (fills/day <= {max_fills_per_day}, "
            f"taker share >= {min_taker_ratio})"
        ),
        notes=notes,
    )


def _series_sql(root: Path, symbol: str, wallets: list[str]) -> str:
    """The statement that reconstructs every selected wallet's series.

    Split out from execution so one statement can either be read back as rows
    or streamed straight into the cache file. A cache written from a *second*
    copy of this query would be a cache of something else.

    The wallet list is validated rather than sanitised: it is interpolated into
    the statement, and every address in the archive is lowercase hex, so
    anything that fails the pattern is a bug upstream and is dropped loudly
    instead of silently changing what the query selects.
    """
    usable = [w for w in wallets if _WALLET_RE.match(w)]
    if not usable:
        raise FillsHistoryError("no syntactically valid wallet address to query")

    files = sorted(root.glob("*.parquet"))
    if not files:
        raise FillsHistoryError(f"{root} holds no parquet files")
    in_list = ",".join(f"'{w}'" for w in usable)
    glob = str(root / "*.parquet")

    # Grouping by (wallet, timestamp) first collapses same-instant fills, whose
    # order the archive does not define. `MIN(start_position)` stands in for
    # "the position entering that instant": exact unless several fills share
    # the very first timestamp and they are reductions, which the 99.87-100%
    # agreement between accumulation and `start_position` says is rare. The
    # accumulation then never reads `start_position` again, so a wrong base
    # offsets one wallet's whole series rather than drifting.
    #
    # No ORDER BY: the reading side sorts per wallet, which it has to do anyway
    # to survive an unordered cache read, and sorting here would make the cache
    # write pay for the same ordering a second time.
    return f"""
WITH step AS (
    SELECT lower(address) AS wallet,
           epoch_ms(timestamp) AS ts,
           MIN(CAST(start_position AS DOUBLE)) AS pos_before,
           SUM(CASE WHEN side = 'buy' THEN CAST(size AS DOUBLE)
                    ELSE -CAST(size AS DOUBLE) END) AS delta
    FROM read_parquet('{glob}')
    WHERE coin = '{symbol}' AND lower(address) IN ({in_list})
    GROUP BY 1, 2
),
seq AS (
    SELECT wallet, ts,
           FIRST_VALUE(pos_before) OVER w
             + SUM(delta) OVER w AS position
    FROM step
    WINDOW w AS (PARTITION BY wallet ORDER BY ts ROWS UNBOUNDED PRECEDING)
)
SELECT wallet, ts, position FROM seq
"""


def _series_from_rows(
    symbol: str, rows: list[dict]
) -> dict[str, WalletPositionSeries]:
    """Group the statement's rows into one sorted series per wallet.

    Sorting here rather than in SQL is what lets the cache be an unordered
    parquet: `position_at` bisects the point list, so an unsorted one would not
    fail loudly, it would answer the wrong question.
    """
    per_wallet: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        try:
            wallet = str(row["wallet"]).lower()
            ts = int(row["ts"])
            position = float(row["position"])
        except (KeyError, TypeError, ValueError):
            continue
        per_wallet.setdefault(wallet, []).append((ts, position))

    return {
        wallet: WalletPositionSeries(
            wallet=wallet, coin=symbol, points=sorted(points)
        )
        for wallet, points in per_wallet.items()
        if points
    }


def _cache_paths(cache_dir: Path, symbol: str) -> tuple[Path, Path]:
    return cache_dir / f"{symbol}.parquet", cache_dir / f"{symbol}.json"


def _signature(root: Path, symbol: str, wallets: list[str]) -> dict:
    """What a cached series file must match to still describe this rebuild.

    Names, sizes and mtimes of every fills file rather than a count: the sync
    keeps adding days, and the archive is also **rewritten in place** - two
    days changed under us on 2026-10-02 - so anything coarser would serve a
    series built from a different archive than the one on disk.

    The wallet list is here because the rebuild only covers the wallets it is
    handed: the same archive yields different series for different sets, and a
    cache keyed on the archive alone would return the wrong one.
    """
    files = sorted(root.glob("*.parquet"))
    return {
        "symbol": symbol,
        "wallets": list(wallets),
        "files": [
            [f.name, f.stat().st_size, int(f.stat().st_mtime)] for f in files
        ],
    }


def _read_cache(
    cache_dir: Path, symbol: str, signature: dict
) -> list[dict] | None:
    """Rows from the cache, or None when it does not describe this rebuild."""
    parquet, meta = _cache_paths(cache_dir, symbol)
    if not parquet.exists() or not meta.exists():
        return None
    try:
        stored = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if stored != signature:
        return None
    try:
        return _run_sql(
            f"SELECT wallet, ts, position FROM read_parquet('{parquet}');"
        )
    except FillsHistoryError:
        # A cache that cannot be read is a cache miss, not a run failure: the
        # rebuild it was meant to skip is still available.
        return None


def _write_cache(
    cache_dir: Path, symbol: str, signature: dict, sql: str
) -> None:
    """Write the rebuild to parquet, then the signature describing it.

    The signature lands *last* and the parquet goes through a rename, so a run
    interrupted mid-write leaves a cache that fails the signature check rather
    than one that passes it while holding half a file.
    """
    parquet, meta = _cache_paths(cache_dir, symbol)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = parquet.with_suffix(".parquet.tmp")
    _run_sql(f"COPY ({sql}) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD);")
    tmp.replace(parquet)
    meta.write_text(json.dumps(signature), encoding="utf-8")


def _rebuild_series(
    root: Path,
    symbol: str,
    wallets: list[str],
    cache: str | Path | None = DEFAULT_CACHE,
) -> dict[str, WalletPositionSeries]:
    """Position series per wallet, from the cache when it is still valid.

    The rebuild is a windowed scan of every fills file - minutes, against a
    backtest that runs in seconds - so a run that changes only the strategy
    should not pay for it twice.
    """
    if cache is None:
        return _series_from_rows(symbol, _run_sql(_series_sql(root, symbol, wallets)))

    cache_dir = Path(cache)
    signature = _signature(root, symbol, wallets)
    rows = _read_cache(cache_dir, symbol, signature)
    if rows is None:
        _write_cache(cache_dir, symbol, signature,
                     _series_sql(root, symbol, wallets))
        rows = _read_cache(cache_dir, symbol, signature)
    return _series_from_rows(symbol, rows or [])
