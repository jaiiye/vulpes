"""Rebuilds the wallet leaderboard from the fills archive.

Production selects the smart money set from the venue's published leaderboard,
ranked by realised PnL over day/week/month windows. No archive of that
leaderboard exists, so it has to be reconstructed from the one thing that is
archived: every fill, with the PnL it realised.

The ranking criterion is matched deliberately rather than invented. The live
`LeaderboardWalletSource` sorts each window by its `pnl` figure - not by ROI,
not by volume, not by position size - takes the top `top_per_window` from each,
unions them, and keeps the wallets that persist across windows. This module
does the same, in the same order, so a discrepancy between the two is a
property of the data rather than of two different ranking rules.

What the reconstruction is not, and every run reports:

  * **Realised PnL only.** The venue's figure can include unrealised movement;
    summing fills cannot. A wallet sitting on a large open winner looks worse
    here than it does in production.
  * **No account-value floor.** Production drops wallets below
    `min_account_value`. That column belongs to the account-values dataset,
    which is not part of the research projection, so the floor is omitted
    rather than approximated by something else.
  * **Trailing windows, not venue windows.** 1/7/30 days back from the clock,
    which is what day/week/month mean operationally but need not be the exact
    interval the venue uses.

The output drives a **time-varying** wallet set. A wallet contributes position
points only on the days it was actually selected, so `position_at` returns None
on every other day and the factor reads the set that existed at that moment
rather than one chosen with hindsight.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_FILLS_STORE = "data/canonical/fills"
#: Where a completed reconstruction is cached. The ranking costs minutes of
#: windowed aggregation over a billion fills, while the backtest it feeds runs
#: in seconds, so recomputing on every invocation makes iteration impractical.
DEFAULT_CACHE = "data/canonical/leaderboard/board.json"

#: (window name, trailing days). Matches the live config's `windows: [day,
#: week, month]` so the union covers the same three horizons.
DEFAULT_WINDOWS: tuple[tuple[str, int], ...] = (
    ("day", 1),
    ("week", 7),
    ("month", 30),
)
#: Production's `top_per_window`.
DEFAULT_TOP_PER_WINDOW = 60
#: Production's `max_wallets`.
DEFAULT_MAX_WALLETS = 150


class LeaderboardError(RuntimeError):
    """The fills archive could not be read or held nothing usable."""


@dataclass
class Leaderboard:
    """Selected wallets per UTC day, and how each was selected."""

    #: day string (YYYY-MM-DD) -> {wallet: persistence}
    by_day: dict[str, dict[str, int]] = field(default_factory=dict)
    windows: tuple[tuple[str, int], ...] = DEFAULT_WINDOWS
    top_per_window: int = DEFAULT_TOP_PER_WINDOW
    max_wallets: int = DEFAULT_MAX_WALLETS
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.by_day)

    @property
    def wallets(self) -> set[str]:
        """Every wallet selected on at least one day."""
        out: set[str] = set()
        for selected in self.by_day.values():
            out.update(selected)
        return out

    def summary(self) -> str:
        if not self.by_day:
            return "  no leaderboard reconstructed"
        sizes = [len(v) for v in self.by_day.values()]
        counts: dict[int, int] = {}
        for selected in self.by_day.values():
            for wallet in selected:
                counts[wallet] = counts.get(wallet, 0) + 1
        repeat = sum(1 for n in counts.values() if n >= 2)
        days = sorted(self.by_day)
        lines = [
            f"  days: {len(self.by_day)} ({days[0]} -> {days[-1]})",
            f"  wallets per day: {min(sizes)}-{max(sizes)} (cap {self.max_wallets})",
            f"  distinct wallets ever selected: {len(counts):,}",
            f"  selected on 2+ days: {repeat:,} "
            f"({repeat / max(len(counts), 1) * 100:.0f}%)",
        ]
        return "\n".join(lines)


def _run_sql(sql: str, timeout: int = 1800) -> list[dict]:
    """Run a query and parse its JSON output.

    The statement arrives on **stdin** rather than as a `-c` argument. A single
    argument is capped at 128 KB by the kernel (`MAX_ARG_STRLEN`), which this
    query can approach once the window frames are expanded, and stdin has no
    such limit. See the same note in `position_history._run_sql`.
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
        raise LeaderboardError("duckdb was not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise LeaderboardError(f"duckdb timed out after {timeout}s") from exc

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise LeaderboardError(
            f"duckdb failed: {detail[-1] if detail else 'no output'}"
        )
    text = proc.stdout.strip()
    if not text:
        return []
    try:
        rows = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LeaderboardError(f"unparseable duckdb output: {text[:200]}") from exc
    if not isinstance(rows, list):
        raise LeaderboardError("unexpected duckdb output shape")
    return rows


def _signature(
    root: Path,
    windows: tuple[tuple[str, int], ...],
    top_per_window: int,
    max_wallets: int,
    first_day: str | None,
    last_day: str | None,
    max_fills_per_day: float | None = None,
) -> dict:
    """What a cached board must match to still describe the archive.

    Names, sizes and mtimes of every fills file, not just a count: the sync
    keeps adding days, and a cache keyed on anything coarser would serve a
    ranking built from a different archive than the one on disk.

    Every selection parameter is in here for the same reason. A cache that
    tracks the data but not the rule would answer a different question with a
    confident-looking board, which is worse than no cache at all.
    """
    files = sorted(root.glob("*.parquet"))
    return {
        "files": [
            [f.name, f.stat().st_size, int(f.stat().st_mtime)] for f in files
        ],
        "windows": [list(w) for w in windows],
        "top_per_window": int(top_per_window),
        "max_wallets": int(max_wallets),
        "first_day": first_day,
        "last_day": last_day,
        "max_fills_per_day": (
            None if max_fills_per_day is None else float(max_fills_per_day)
        ),
    }


def _load_cache(path: Path | None, signature: dict) -> Leaderboard | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("signature") != signature:
        return None
    raw = payload.get("by_day")
    if not isinstance(raw, dict) or not raw:
        return None
    days: dict[str, dict[str, int]] = {}
    for day, wallets in raw.items():
        if not isinstance(wallets, dict):
            continue
        entry: dict[str, int] = {}
        for wallet, hits in wallets.items():
            try:
                entry[str(wallet)] = int(hits)
            except (TypeError, ValueError):
                continue
        if entry:
            days[str(day)] = entry
    if not days:
        return None
    windows = tuple(
        (str(name), int(days_))
        for name, days_ in payload.get("windows") or signature["windows"]
    )
    return Leaderboard(
        by_day=days,
        windows=windows,
        top_per_window=int(signature["top_per_window"]),
        max_wallets=int(signature["max_wallets"]),
        notes=list(payload.get("notes") or []),
    )


def _save_cache(path: Path | None, board: Leaderboard, signature: dict) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "signature": signature,
                    "windows": [list(w) for w in board.windows],
                    "notes": board.notes,
                    "by_day": board.by_day,
                }
            ),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError:
        # A cache miss costs minutes, not correctness, so a write failure is
        # not worth failing the run over.
        pass


def reconstruct(
    store: str | Path = DEFAULT_FILLS_STORE,
    *,
    windows: tuple[tuple[str, int], ...] = DEFAULT_WINDOWS,
    top_per_window: int = DEFAULT_TOP_PER_WINDOW,
    max_wallets: int = DEFAULT_MAX_WALLETS,
    first_day: str | None = None,
    last_day: str | None = None,
    max_fills_per_day: float | None = None,
    cache_path: str | Path | None = DEFAULT_CACHE,
) -> Leaderboard:
    """Rank wallets by trailing realised PnL, one ranking per UTC day.

    The whole computation is one query so the ranking is taken against a single
    consistent view of the archive. Splitting the trailing sums from the
    per-day ranking would allow a file to land between the two and produce a
    ranking that never existed.

    A completed board is cached and reused while the archive it was built from
    is unchanged. Pass `cache_path=None` to force a rebuild.
    """
    root = Path(store)
    if not root.exists():
        raise LeaderboardError(
            f"no fills store at {root}. Run: "
            "python sync_reservoir.py --datasets fills --profile <name>"
        )
    files = sorted(root.glob("*.parquet"))
    if not files:
        raise LeaderboardError(f"{root} holds no parquet files")

    cache = Path(cache_path) if cache_path else None
    signature = _signature(
        root, windows, top_per_window, max_wallets, first_day, last_day,
        max_fills_per_day,
    )
    cached = _load_cache(cache, signature)
    if cached is not None:
        cached.notes.append(
            "served from cache; the fills archive is unchanged since it was "
            "built, so this ranking describes the same data"
        )
        return cached

    glob = str(root / "*.parquet")
    top = max(5, min(200, int(top_per_window)))
    cap = max(1, int(max_wallets))

    # Trailing sums are expressed as RANGE frames over an integer day number.
    # A date offset would read better, but a numeric frame is unambiguous and
    # does not depend on the calendar arithmetic DuckDB chooses internally.
    #
    # `RANGE` rather than `ROWS`: a wallet with no fill on a given day has no
    # row at all, and a ROWS frame would silently shorten the window by one day
    # for every gap - measuring a different period for an active trader than
    # for an intermittent one.
    frame = ",\n           ".join(
        f"SUM(pnl) OVER (PARTITION BY address ORDER BY day_num "
        f"RANGE BETWEEN {days - 1} PRECEDING AND CURRENT ROW) AS pnl_{name}"
        for name, days in windows
    )
    rank = ",\n           ".join(
        f"ROW_NUMBER() OVER (PARTITION BY d ORDER BY pnl_{name} DESC) AS r_{name}"
        for name, _ in windows
    )
    rank_cols = ", ".join(f"r_{name}" for name, _ in windows)
    persist = " + ".join(
        f"CASE WHEN r_{name} <= {top} THEN 1 ELSE 0 END" for name, _ in windows
    )
    predicate = " OR ".join(f"r_{name} <= {top}" for name, _ in windows)
    # Ties broken by the SUM of ranks, and ranked ascending because a rank of 1
    # is the best: the smaller the sum, the better the wallet placed. Sorting
    # this descending kept the worst of the tied wallets, which is the opposite
    # of what the rule is for.
    #
    # Production breaks the same tie on account value; that column is absent
    # here, and rank position is the closest available proxy for conviction.
    tiebreak = "+".join(f"c.r_{name}" for name, _ in windows)

    date_filter = ""
    if first_day:
        date_filter += f" AND d >= DATE '{first_day}'"
    if last_day:
        date_filter += f" AND d <= DATE '{last_day}'"

    # Optional activity ceiling. Ranking by absolute PnL favours whoever books
    # the largest number, and measured against this archive that is
    # overwhelmingly market makers: of the wallets entering the top 60 on
    # 30-day PnL, 53% trade more than 1,000 times a day and 86% more than 200,
    # while under 0.4% trade fewer than ten. Median monthly PnL rises
    # monotonically with fill count, which is the signature of a ranking that
    # is really selecting for activity.
    #
    # The ceiling is what tests whether that matters. A position held by a
    # liquidity provider is inventory, not a directional view, so if the factor
    # only earns its keep on the high-frequency accounts the signal is being
    # read off the wrong thing.
    activity_frame = "0.0 AS fills_per_day"
    activity_where = ""
    if max_fills_per_day is not None:
        window_days = max(days for _, days in windows)
        activity_frame = (
            f"SUM(fills) OVER (PARTITION BY address ORDER BY day_num "
            f"RANGE BETWEEN {window_days - 1} PRECEDING AND CURRENT ROW) "
            f"/ {window_days}.0 AS fills_per_day"
        )
        activity_where = f"WHERE fills_per_day <= {float(max_fills_per_day)}"

    sql = f"""
WITH daily AS (
    SELECT address,
           CAST(timestamp AS DATE) AS d,
           CAST(floor(epoch(CAST(timestamp AS DATE)) / 86400.0) AS BIGINT) AS day_num,
           SUM(realized_pnl)::DOUBLE AS pnl,
           COUNT(*) AS fills
    FROM read_parquet('{glob}')
    WHERE address IS NOT NULL AND address <> ''{date_filter}
    GROUP BY ALL
),
rolling AS (
    SELECT address, d, day_num,
           {frame},
           {activity_frame}
    FROM daily
),
ranked AS (
    SELECT address, d, day_num, fills_per_day,
           {rank}
    FROM rolling
    {activity_where}
),
chosen AS (
    SELECT address, d, day_num, {rank_cols},
           ({persist}) AS persistence
    FROM ranked
    WHERE {predicate}
)
SELECT address, CAST(d AS VARCHAR) AS day, persistence,
       ({tiebreak}) AS rank_sum
FROM chosen c
QUALIFY ROW_NUMBER() OVER (
    PARTITION BY c.d
    ORDER BY c.persistence DESC, {tiebreak} ASC, c.address
) <= {cap}
ORDER BY c.d, c.persistence DESC, rank_sum ASC, c.address;
"""

    rows = _run_sql(sql)
    if not rows:
        raise LeaderboardError("no wallet ranked on any day")

    by_day: dict[str, dict[str, int]] = {}
    for row in rows:
        wallet = str(row.get("address", "")).lower()
        day = str(row.get("day", ""))[:10]
        if not wallet or not day:
            continue
        try:
            persistence = int(row.get("persistence") or 1)
        except (TypeError, ValueError):
            persistence = 1
        by_day.setdefault(day, {})[wallet] = persistence

    if not by_day:
        raise LeaderboardError("reconstruction produced no usable days")

    board = Leaderboard(
        by_day=by_day,
        windows=windows,
        top_per_window=top,
        max_wallets=cap,
    )
    board.notes.append(
        "ranking uses realised PnL summed from fills; the venue's figure can "
        "include unrealised movement, so an open winner ranks lower here"
    )
    board.notes.append(
        "no account-value floor: that column is not in the research projection, "
        "so production's min_account_value filter is omitted rather than faked"
    )
    if max_fills_per_day is not None:
        board.notes.append(
            f"activity ceiling: wallets averaging more than "
            f"{max_fills_per_day:g} fills/day over the longest ranking window "
            "were excluded. Production applies no such ceiling - this is a "
            "diagnostic, not a reproduction"
        )
    _save_cache(cache, board, signature)
    return board
