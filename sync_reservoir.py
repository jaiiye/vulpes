#!/usr/bin/env python3
"""Sync the Hydromancer Reservoir archive into the local data store.

The archive is requester-pays: every byte transferred is billed to this
account. Two facts shape this script.

First, **column projection is the only lever that reduces transfer**. Measured
against the real files, `coin` in the candle file and `address` in the fills
file carry no min/max statistics, so `WHERE coin IN (...)` cannot prune a
single row group - the filter runs after the bytes have arrived. Projecting
columns, by contrast, is worth 70% of the whole archive, almost entirely
because `fills` is dominated by identifier columns nobody needs.

Second, the first 100 GB of egress each calendar month is free and does not
carry over. So the bill depends on how the download is spread, which is why
this script reports cost before it moves anything and refuses to exceed a cap.

    python sync_reservoir.py --dry-run                  # plan and cost, no data
    python sync_reservoir.py --datasets positions       # one dataset
    python sync_reservoir.py --limit 30 --max-gb 40     # bounded first pass

Re-running is safe and cheap: dates already present locally are skipped, so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

BUCKET = "hydromancer-reservoir"
REGION = "ap-northeast-1"
DEFAULT_STORE = "data/canonical"

FREE_EGRESS_GB_PER_MONTH = 100.0
EGRESS_USD_PER_GB = 0.114

DEFAULT_SYMBOLS = ("BTC", "ETH", "SOL")

#: Parallel range requests per COPY. Measured on this link: raising it from the
#: default cut one object's read from 85s to 25s.
DUCKDB_THREADS = 8


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncTarget:
    """One archive dataset, and the columns worth paying to transfer."""

    name: str
    prefix: str
    #: Columns a backtest needs today. Cheapest, and the recommended default.
    research: tuple[str, ...]
    #: Everything except pure identifiers - hashes, order ids, addresses of
    #: counterparties. These carry no analytical value and, in `fills`, 57% of
    #: the bytes.
    wide: tuple[str, ...]
    #: Enough to reconstruct every position change from the fills themselves,
    #: plus what each trade cost and whether it was a maker or a taker. `wide`
    #: costs about a third more and is only needed for what `flow` omits.
    #:
    #: `start_position` is deliberately left out although it is the single most
    #: expensive column in the dataset (11.4% of the whole archive, 14.5 GB).
    #: It is a convenience, not a requirement: `size` plus `direction` accumulate
    #: into the same series, and the `positions` snapshots already in the store
    #: supply the daily anchor that makes the accumulation start from a known
    #: value. Leave the column and every sync of this dataset pays for it.
    flow: tuple[str, ...] = ()
    #: Set when one day is many files, one per market, selected by path.
    per_symbol_files: bool = False
    #: Optional SQL predicate. Present for completeness; it saves no transfer.
    filter_sql: str = ""
    note: str = ""

    def columns_for(self, mode: str) -> tuple[str, ...]:
        if mode == "research":
            return self.research
        if mode == "wide":
            return self.wide
        if mode == "flow":
            if not self.flow:
                raise ValueError(f"{self.name} has no flow column set")
            return self.flow
        raise ValueError(f"unknown column mode {mode!r}")


#: Columns that exist only in the later part of the archive, and so cannot be
#: projected. Measured by reading each day's own footer on 2026-09-25:
#:
#:     2025-08-01 .. 2026-03-19   27 columns
#:     2026-06-01 .. 2026-09-20   29 columns   <- adds these two
#:
#: A COPY that names a column the day does not have fails on that day, and
#: `wide` named both - so 235 of 423 days failed with a binder error while every
#: column-set test passed, because `REAL_SCHEMAS` records **one** day's schema
#: and the archive's schema is not constant. Neither column is worth that: both
#: are breakdowns of the fee, and `fee` is already the total.
COLUMNS_ADDED_MID_ARCHIVE = ("deployer_fee", "priority_gas")


SYNC_TARGETS: tuple[SyncTarget, ...] = (
    SyncTarget(
        name="positions",
        prefix="by_dex/hyperliquid/snapshots/perp",
        research=("user", "market", "size", "notional", "entry_price"),
        wide=(
            "user", "market", "size", "notional", "entry_price",
            "liquidation_price", "leverage", "leverage_type",
            "funding_pnl", "account_value", "account_mode",
        ),
        note="unlocks the 40%-weight smart money factor; cheapest high-value set",
    ),
    SyncTarget(
        name="candles",
        prefix="by_dex/hyperliquid/candles/1s",
        research=("coin", "timestamp", "open", "high", "low", "close", "volume"),
        wide=(
            "coin", "dex", "asset_class", "base_symbol", "quote_symbol",
            "timestamp", "open", "high", "low", "close", "volume",
            "volume_quote", "trade_count",
        ),
        note="1s only; coarser bars are aggregated locally",
    ),
    SyncTarget(
        name="orderbook",
        prefix="by_dex/hyperliquid/orderbook/1m/perps",
        research=("block_time_ms", "block_number", "bids", "asks"),
        wide=("block_time_ms", "block_number", "bids", "asks"),
        per_symbol_files=True,
        note="one file per coin per day, so symbol selection is by path",
    ),
    SyncTarget(
        name="account_values",
        prefix="global/snapshots/account_values",
        research=("user", "dex", "account_value"),
        wide=(
            "user", "dex", "collateral_token", "account_value",
            "total_long_notional", "total_short_notional", "account_mode",
        ),
        note="barely compressible: `user` alone is 91% of the file",
    ),
    SyncTarget(
        name="fills",
        prefix="by_dex/hyperliquid/fills/perp/all",
        research=("address", "timestamp", "realized_pnl"),
        wide=(
            "address", "timestamp", "realized_pnl", "coin", "base_symbol",
            "quote_symbol", "asset_class", "dex", "price", "size", "side",
            "direction", "fee", "fee_token", "start_position", "crossed",
            "is_liquidation", "liquidation_mark_px", "liquidation_method",
            "builder_fee",
        ),
        flow=(
            "address", "timestamp", "realized_pnl",
            "coin", "size", "side", "direction", "price",
            "crossed", "is_liquidation", "fee",
        ),
        note="76% of the archive; identifiers are 57% of it",
    ),
)

# Per-column compressed sizes, read from each file's own footer with
# `parquet_metadata` on a 2026-09-16 object. These are the basis for every
# cost figure the script prints, and they are measurements rather than
# estimates.
#
# The ratio is what matters, not the absolute values: it is a property of the
# schema and the compression, so it holds across files even as one day's volume
# differs from another's.
COLUMN_BYTES: dict[str, dict[str, int]] = {
    "candles": {
        "volume_quote": 8_731_899, "volume": 5_515_570, "close": 3_704_413,
        "low": 3_694_532, "high": 3_682_079, "open": 3_680_466,
        "timestamp": 3_434_808, "trade_count": 741_571,
        "base_symbol": 10_513, "coin": 9_963, "quote_symbol": 2_139,
        "asset_class": 1_656, "dex": 1_585,
    },
    "positions": {
        "user": 10_177_569, "funding_pnl": 2_336_138,
        "liquidation_price": 1_840_514, "notional": 1_700_751,
        "entry_price": 1_583_455, "account_value": 1_286_787,
        "size": 1_056_019, "leverage": 155_144, "account_mode": 51_991,
        "leverage_type": 38_174, "market": 2_024,
    },
    "account_values": {
        "user": 85_161_328, "account_value": 6_299_818,
        "total_long_notional": 1_279_176, "total_short_notional": 604_949,
        "account_mode": 317_259, "dex": 2_839, "collateral_token": 2_736,
    },
    "fills": {
        "tx_hash": 109_397_747, "client_order_id": 70_514_540,
        "start_position": 47_830_412, "trade_id": 33_404_661,
        "fee": 33_256_328, "order_id": 24_887_420, "realized_pnl": 24_725_516,
        "size": 21_859_831, "price": 15_536_364, "address": 15_121_862,
        "timestamp": 7_233_159, "base_symbol": 3_346_514, "coin": 3_343_591,
        "builder_fee": 2_675_109, "priority_gas": 2_292_516,
        "direction": 1_943_791, "builder": 848_227, "twap_id": 669_567,
        "crossed": 581_655, "is_liquidation": 51_401, "side": 47_468,
        "liquidation_mark_px": 31_380, "liquidation_method": 13_978,
        "quote_symbol": 10_545, "deployer_fee": 9_123, "dex": 8_080,
        "fee_token": 8_031, "asset_class": 8_031,
    },
    # The order book keeps all four of its columns, so no projection applies.
    # Its saving comes from the path: one file per market, so three symbols are
    # taken out of ~178 rather than every column of every file.
    "orderbook": {
        "block_time_ms": 1, "block_number": 1, "bids": 1, "asks": 1,
    },
}

# Actual source sizes, as reported by a real `--dry-run` on 2026-09-18. These
# are totals, not the per-day sample multiplied out, and they came in lower than
# that extrapolation did - so the earlier figures were pessimistic. Kept here so
# the projection's effect on the archive can be asserted without touching the
# network.
#
# `orderbook` is the one exception to "every object the archive holds": it is
# the size of the **default three symbols**, not of the dataset. `plan_dataset`
# drops every object whose file name is not in `--symbols` before summing, so
# this figure is BTC+ETH+SOL, and the whole dataset is 178 markets a day -
# measured by listing the prefix on 2026-09-24: 51,841 objects over 282 days,
# 13.7 GB. The distinction matters because 0.31 GB reads as "this dataset is
# nearly free", which is true only for three markets; taking all of them is
# 44 times that and would not fit the egress allowance alongside a wide fills
# sync. Every other figure here is the whole dataset.
MEASURED_SOURCE_GB: dict[str, float] = {
    "positions": 5.09,
    "candles": 10.68,
    "orderbook": 0.31,
    "account_values": 15.73,
    "fills": 127.58,
}

TARGETS_BY_NAME = {t.name: t for t in SYNC_TARGETS}


class SyncError(RuntimeError):
    """The sync cannot proceed."""


# ---------------------------------------------------------------------------
# Cost arithmetic (pure; tested without touching AWS)
# ---------------------------------------------------------------------------


def to_gb(num_bytes: float) -> float:
    return num_bytes / 1024**3


def egress_cost_usd(total_gb: float, months: int = 1) -> float:
    if total_gb <= 0 or months < 1:
        return 0.0
    billable = max(0.0, total_gb - FREE_EGRESS_GB_PER_MONTH * months)
    return billable * EGRESS_USD_PER_GB


def months_to_stay_free(total_gb: float) -> int:
    """Ceiling division: exactly 100 GB still fits in one month."""
    if total_gb <= 0:
        return 1
    return max(1, math.ceil(total_gb / FREE_EGRESS_GB_PER_MONTH))


def human_gb(num_bytes: float) -> str:
    return f"{to_gb(num_bytes):,.2f} GB"


def projected_fraction(dataset: str, columns: tuple[str, ...]) -> float:
    """Share of a source file a column-projected read still fetches.

    DuckDB reads only the column chunks a query references, so the floor is the
    needed columns over all of them. This is the leverage that makes the
    archive affordable: `fills` is 76% of the bytes and drops to about an
    eighth of itself because its identifiers are most of the file.

    Row-group pruning would be the other lever and is not available: neither
    `coin` in the candle file nor `address` in the fills file carries min/max
    statistics, so a `WHERE` on either saves nothing on transfer.
    """
    sizes = COLUMN_BYTES.get(dataset)
    if not sizes:
        return 1.0
    total = sum(sizes.values())
    if total <= 0:
        return 1.0
    wanted = sum(size for name, size in sizes.items() if name in columns)
    return wanted / total


def date_from_key(key: str) -> str:
    """The `date=YYYY-MM-DD` partition a key lives under, or ''."""
    for part in key.split("/"):
        if part.startswith("date="):
            return part[len("date="):]
    return ""


def completed_file(path: Path) -> bool:
    """True only when a day-file is present AND has content.

    A COPY that stalls partway leaves a zero-length file behind. Treating mere
    existence as "done" makes that day permanently skipped, so an interrupted
    run would quietly produce a store with holes in it - and the holes would be
    invisible, because the plan would keep reporting those days as fetched.

    That is not hypothetical: a stalled `duckdb` created a 0-byte
    `date=2025-07-30.parquet` that the next run would have walked straight past.
    """
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


# ---------------------------------------------------------------------------
# AWS access
# ---------------------------------------------------------------------------


def aws_json(
    args: list[str],
    timeout: int = 300,
    attempts: int = 4,
    env: dict[str, str] | None = None,
) -> dict:
    """Run an `aws s3api` call, retrying transient transport failures.

    Listing tens of thousands of keys takes dozens of sequential requests, and
    on this network an `SSL: UNEXPECTED_EOF_WHILE_READING` mid-pagination is
    routine. Without a retry a single blip aborts a plan that has already spent
    a minute enumerating. Only transport errors are retried: a credentials or
    permissions failure is deterministic and must surface immediately.

    `env` must be the environment from `duckdb_env`. Passing it is the whole
    reason a named profile works: the CLI resolves the profile itself, so an
    inherited environment with an empty default profile fails here even when
    the credentials exist under a name.
    """
    cmd = ["aws", *args, "--region", REGION, "--output", "json"]
    transient = (
        "UNEXPECTED_EOF_WHILE_READING", "Connection reset", "timed out",
        "TLS", "SSL", "BrokenPipe", "throttl",
    )
    last = ""
    for attempt in range(1, attempts + 1):
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, env=env
            )
        except FileNotFoundError as exc:
            raise SyncError(
                "the aws CLI was not found on PATH; install it and configure "
                "credentials first"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            last = f"aws call timed out after {timeout}s"
            if attempt == attempts:
                raise SyncError(last) from exc
            time.sleep(2 * attempt)
            continue

        if proc.returncode == 0:
            try:
                return json.loads(proc.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise SyncError(
                    f"unparseable aws output: {proc.stdout[:200]}"
                ) from exc

        detail = (proc.stderr or proc.stdout).strip().splitlines()
        last = detail[-1] if detail else "no output"
        if not any(marker.lower() in last.lower() for marker in transient):
            raise SyncError(f"aws call failed ({proc.returncode}): {last}")
        if attempt < attempts:
            time.sleep(2 * attempt)

    raise SyncError(f"aws call failed after {attempts} attempts: {last}")


def list_objects(
    prefix: str, env: dict[str, str] | None = None
) -> list[tuple[str, int]]:
    """Every object under a prefix, as (key, size). Paginates."""
    out: list[tuple[str, int]] = []
    token: str | None = None
    while True:
        args = [
            "s3api", "list-objects-v2",
            "--bucket", BUCKET,
            "--prefix", prefix.rstrip("/") + "/",
            "--request-payer", "requester",
        ]
        if token:
            args += ["--starting-token", token]
        payload = aws_json(args, env=env)
        for obj in payload.get("Contents") or []:
            out.append((obj.get("Key", ""), int(obj.get("Size", 0))))
        if not payload.get("IsTruncated"):
            return out
        token = payload.get("NextContinuationToken")
        if not token:
            return out


def available_profiles() -> list[str]:
    """Profile names present in the AWS CLI config files.

    Names only, never values. This exists to make a credential failure
    actionable: the usual arrangement on a workstation is that the keys live in
    a named profile while the default profile is empty, and the CLI reports that
    as a bare "no credentials found" - which reads as "no credentials exist"
    rather than "the ones you have are under a different name".
    """
    names: list[str] = []
    for path in (
        Path.home() / ".aws" / "credentials",
        Path.home() / ".aws" / "config",
    ):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not (line.startswith("[") and line.endswith("]")):
                continue
            name = line[1:-1].strip()
            if name.startswith("profile "):
                name = name[len("profile "):].strip()
            if name and name not in names:
                names.append(name)
    return names


def duckdb_env(profile: str | None = None) -> dict[str, str]:
    """Environment for DuckDB, with credentials resolved.

    DuckDB does not read `~/.aws/credentials` on its own, and it certainly does
    not understand the `aws login` store. Rather than fail with an opaque
    "Anonymous users cannot invoke requests against Requester Pays buckets",
    resolve the credentials through the CLI - which does understand both - and
    hand them over in the environment rather than on a command line.
    """
    env = dict(os.environ)
    if profile:
        env["AWS_PROFILE"] = profile
    if env.get("AWS_ACCESS_KEY_ID"):
        return env
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
        env.pop(key, None)
    try:
        proc = subprocess.run(
            ["aws", "configure", "export-credentials", "--format", "env"],
            capture_output=True, text=True, timeout=60, env=env,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise SyncError(f"could not resolve AWS credentials: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()
        used = env.get("AWS_PROFILE") or "default"
        found = available_profiles()
        hint = (
            f" Profiles available: {', '.join(found)}. Pass --profile to use one."
            if found
            else " No profiles were found in ~/.aws."
        )
        raise SyncError(
            f"no usable AWS credentials for the '{used}' profile. "
            f"`aws configure export-credentials` failed: "
            f"{detail[-1] if detail else 'no output'}.{hint}"
        )
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("export ") and "=" in line:
            name, _, value = line[len("export "):].partition("=")
            env[name.strip()] = value.strip().strip("'\"")
    if not env.get("AWS_ACCESS_KEY_ID"):
        raise SyncError("credential export produced no access key")
    return env


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


@dataclass
class DayPlan:
    date: str
    source_bytes: int
    keys: list[str] = field(default_factory=list)


@dataclass
class DatasetPlan:
    target: SyncTarget
    #: Days already in the store, so a re-run resumes instead of refetching.
    present: int = 0
    days: list[DayPlan] = field(default_factory=list)
    #: Share of each source file the projection must still fetch.
    fraction: float = 1.0

    @property
    def source_bytes(self) -> int:
        return sum(d.source_bytes for d in self.days)

    @property
    def transfer_bytes(self) -> int:
        return int(self.source_bytes * self.fraction)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def build_copy_sql(
    target: SyncTarget,
    keys: list[str],
    columns: tuple[str, ...],
    destination: Path,
    glob: bool,
) -> str:
    """A projection-only COPY. Aggregation belongs to a later stage."""
    listed = ", ".join(f"'{k}'" for k in keys)
    if glob:
        # A per-market dataset such as the order book. Two things about this
        # are not obvious, and the first version got both wrong.
        #
        # **Every selected key has to be read.** It used to read `keys[0]`
        # alone, on the stated reasoning that "one glob replaces the explicit
        # list" - but the keys come from a listing and are complete object
        # paths, so that is not a glob: it reads exactly one market. Measured
        # over 282 days, `--symbols BTC,ETH,SOL` wrote BTC alone at 81 MB where
        # all three markets are 240 MB, with no error and no warning. The
        # single-symbol case looks correct, which is why a probe with one
        # symbol cannot find this.
        #
        # **The market has to be materialised as a column.** The source
        # carries only `block_time_ms/block_number/bids/asks`; the market is
        # in the file name. Without it a day-file holding three markets is
        # three unidentified books, and `datalake.orderbook_from_hydromancer`
        # reads `row["symbol"]` - so every book would normalise to an empty
        # symbol and report nothing about it.
        source = f"read_parquet([{listed}], filename=true)"
        columns = (
            "regexp_extract(filename, '/([^/]+)\\.parquet$', 1) AS symbol",
            *columns,
        )
    else:
        source = f"read_parquet([{listed}])"

    cols = ", ".join(columns)
    predicate = f"\n  WHERE {target.filter_sql}" if target.filter_sql else ""
    return (
        f"COPY (\n  SELECT {cols}\n  FROM {source}{predicate}\n) "
        f"TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD);"
    )


#: Seconds allowed for one day's COPY. Dataset-dependent, and the default is
#: wrong for the largest one.
#:
#: The timeout does two jobs and they pull in opposite directions. It is what
#: keeps a long run moving past a stalled read - one such stall held a COPY open
#: for minutes writing nothing. But multiplied by this link's per-stream rate it
#: is also a **ceiling on how large a day can ever succeed**:
#:
#:     bandwidth per day-stream   ~0.06 MB/s  (measured, 4 workers)
#:     600 s x 0.06 MB/s          ~36 MB      <- the largest day that can finish
#:     fills flow projection      ~93 MB/day  (average)
#:
#: Every fills day is larger than that, so all of them timed out - 228 days,
#: each retried three times, over a day of transfer that wrote nothing and
#: pulled the same bytes three times over. Raise this for `fills`; leave it for
#: the small datasets, where a hang is the likelier failure than a slow success.
DEFAULT_COPY_TIMEOUT = 600

#: Retries per day. What a retry costs is the bytes already moved when it
#: failed, and that depends entirely on the failure mode rather than on the
#: timeout: an SSL connect error fails immediately and re-pays nothing, while a
#: timeout re-pays the whole window.
#:
#: So lowering this to save transfer is the wrong instinct once the timeout is
#: large. The timeout was raised to 3600s precisely so that a timeout stops
#: meaning "this day is too big", which leaves the immediate failures - and this
#: link produces them routinely - for the retries to absorb. Measured the hard
#: way: at `--copy-attempts 1`, three days were lost to SSL errors in a single
#: batch that one more attempt would have covered.
DEFAULT_COPY_ATTEMPTS = 3


def run_copy(sql: str, env: dict[str, str], timeout: int = DEFAULT_COPY_TIMEOUT,
             attempts: int = DEFAULT_COPY_ATTEMPTS,
             threads: int = DUCKDB_THREADS) -> None:
    """Run one projection COPY, with a timeout and a retry.

    A stalled read is routine on this link and DuckDB will sit on it
    indefinitely - one such stall held a COPY open for minutes while writing
    nothing. Without a timeout the whole backfill stops behind a single hung
    subprocess, so the timeout is the thing that keeps a 364-day run moving.

    It is also a throughput ceiling; see `DEFAULT_COPY_TIMEOUT` for the
    arithmetic and for why `fills` needs a larger one.

    `threads` is per DuckDB process, and the caller scales it down as the
    number of concurrent processes goes up. Left fixed it oversubscribes: eight
    processes at eight threads each is sixty-four threads on eight cores, which
    measured a load average of 8.1 and gained only 2.2x over serial.
    """
    # INSTALL is idempotent and the extension is cached after the first run.
    #
    # The two tuning settings are not decoration. Reading one 13 MB object took
    # 85s without them and 25s with - a 3.4x difference measured on this link.
    # `enable_http_metadata_cache` stops DuckDB refetching the footer for every
    # reference, and `threads` lets the range requests for separate column
    # chunks overlap instead of queueing one behind another.
    preamble = (
        "INSTALL httpfs; LOAD httpfs;\n"
        f"SET s3_region = '{REGION}';\n"
        "SET s3_requester_pays = true;\n"
        "SET enable_http_metadata_cache = true;\n"
        f"SET threads = {max(1, int(threads))};\n"
    )
    last = ""
    for attempt in range(1, attempts + 1):
        try:
            proc = subprocess.run(
                ["duckdb"],
                input=preamble + sql,
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            last = f"timed out after {timeout}s"
            if attempt < attempts:
                time.sleep(5 * attempt)
                continue
            raise SyncError(f"duckdb {last}")
        except OSError as exc:
            raise SyncError(f"could not run duckdb: {exc}") from exc

        if proc.returncode == 0:
            return
        # The **first** non-empty line, not the last. DuckDB prints the message
        # first and then echoes the statement with a caret under the offending
        # token, so the last line is usually the caret - which is how a binder
        # error over 235 days was reported as
        # `duckdb failed after 1 attempts:                     ^` and cost a
        # detour through the remote schema to identify.
        detail = [
            line.strip()
            for line in (proc.stderr or proc.stdout).strip().splitlines()
            if line.strip()
        ]
        last = detail[0] if detail else "no output"
        if attempt < attempts:
            time.sleep(5 * attempt)
    raise SyncError(f"duckdb failed after {attempts} attempts: {last}")


def s3_path(key: str) -> str:
    return f"s3://{BUCKET}/{key}"


def plan_dataset(
    target: SyncTarget,
    store: Path,
    symbols: tuple[str, ...],
    limit: int | None,
    column_mode: str = "research",
    env: dict[str, str] | None = None,
) -> DatasetPlan:
    objects = list_objects(target.prefix, env=env)
    if not objects:
        raise SyncError(f"no objects under {target.prefix}")

    # Group by day. The order book stores one file per market, so its symbol
    # selection happens here rather than in SQL.
    by_date: dict[str, list[tuple[str, int]]] = {}
    for key, size in objects:
        if target.per_symbol_files:
            stem = key.rsplit("/", 1)[-1].removesuffix(".parquet")
            if stem.upper() not in symbols:
                continue
        date = date_from_key(key)
        if date:
            by_date.setdefault(date, []).append((key, size))

    out_dir = store / target.name
    plan = DatasetPlan(
        target=target,
        # Not cosmetic: the cost this script reports, and the cap it enforces,
        # are both derived from it. Leaving it at 1.0 overstated the transfer
        # by about 2.5x.
        fraction=projected_fraction(target.name, target.columns_for(column_mode)),
    )
    dates = sorted(by_date)
    for date in dates:
        entries = by_date[date]
        if completed_file(out_dir / f"date={date}.parquet"):
            plan.present += 1
            continue
        plan.days.append(
            DayPlan(
                date=date,
                source_bytes=sum(size for _, size in entries),
                keys=[key for key, _ in entries],
            )
        )

    if limit is not None:
        plan.days = plan.days[:limit]
    return plan


def report(plans: list[DatasetPlan], mode: str, dry_run: bool) -> float:
    total_source = sum(p.source_bytes for p in plans)
    total_transfer = sum(p.transfer_bytes for p in plans)
    total_days = sum(len(p.days) for p in plans)

    print("=" * 78)
    print(" RESERVOIR SYNC PLAN" + ("  (dry run)" if dry_run else ""))
    print("=" * 78)
    print(f"  columns: {mode}")
    print()
    print(f"  {'dataset':16s}{'to fetch':>10s}{'have':>7s}"
          f"{'source':>14s}{'transfer':>14s}")
    for plan in plans:
        print(
            f"  {plan.target.name:16s}{len(plan.days):>10d}{plan.present:>7d}"
            f"{human_gb(plan.source_bytes):>14s}"
            f"{human_gb(plan.transfer_bytes):>14s}"
        )
    print(
        f"  {'TOTAL':16s}{total_days:>10d}{'':7s}"
        f"{human_gb(total_source):>14s}{human_gb(total_transfer):>14s}"
    )
    print()

    if total_days == 0:
        print("  Nothing to fetch; the store is up to date.")
        return 0.0

    transfer_gb = to_gb(total_transfer)
    one_month = egress_cost_usd(transfer_gb, months=1)
    free_months = months_to_stay_free(transfer_gb)
    print("  COST OF THIS RUN")
    print(f"    transfer        {transfer_gb:,.2f} GB")
    print(f"    in one month    ${one_month:,.2f}")
    if free_months > 1:
        unit = "month" if free_months == 1 else "months"
        print(f"    spread over {free_months} {unit}   $0.00")
    else:
        print("    inside the 100 GB monthly allowance: $0.00")
    print(f"    the source files total {human_gb(total_source)}, so projection "
          f"avoids {human_gb(total_source - total_transfer)} "
          f"({100 * (1 - total_transfer / max(total_source, 1)):.0f}%)")
    print()
    for plan in plans:
        if plan.target.note:
            print(f"    {plan.target.name}: {plan.target.note}")
    print()
    return transfer_gb


def sync_dataset(
    plan: DatasetPlan,
    store: Path,
    columns: tuple[str, ...],
    env: dict[str, str],
    dry_run: bool,
    workers: int = 1,
    copy_timeout: int = DEFAULT_COPY_TIMEOUT,
    copy_attempts: int = DEFAULT_COPY_ATTEMPTS,
) -> int:
    """Fetch every pending day, optionally several at a time.

    Days are independent - each stages its own file and renames it into place -
    so overlapping them costs nothing in correctness. What it buys is speed, and
    the reason is worth recording: the transfer is **latency**-bound, not
    bandwidth-bound. One day moves about 34 MB and takes about 45 seconds, which
    is 0.76 MB/s. Fetching a parquet object over HTTP issues many small range
    requests and the reader waits on each round trip, so most of those 45
    seconds the link is idle. Overlapping days occupies it.

    Concurrency does not change the byte count, so it does not change the bill.

    It can, however, make it *slower*, which the help text for `--workers` did
    not say until it was measured: on the fills dataset twelve concurrent days
    moved 0.30 GB/h where four moved 0.85 GB/h. Beyond some point the streams
    congest each other, and every one of them is then inside the per-day timeout
    at once, so they all fail together.
    """
    out_dir = store / plan.target.name
    total = len(plan.days)
    if dry_run or not total:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep the *total* thread count near the core count rather than giving
    # every process the full budget. The read is partly CPU - decompression and
    # zstd re-compression - so processes competing for the same cores gain
    # little, as measured: 8x8 threads on 8 cores reached a load average of 8.1
    # for a 2.2x speedup over serial.
    cores = os.cpu_count() or 4
    per_process = max(1, cores // max(1, workers))

    def fetch(day) -> bool:
        destination = out_dir / f"date={day.date}.parquet"
        # Stage under a temporary name and rename on success. DuckDB creates
        # the output file before it has read a byte, so writing straight to the
        # final path leaves a zero-length file there whenever a read stalls -
        # and the next run would read that as a finished day.
        staging = destination.with_name(destination.name + ".tmp")
        staging.unlink(missing_ok=True)

        glob = plan.target.per_symbol_files and len(day.keys) > 1
        sql = build_copy_sql(
            plan.target,
            [s3_path(k) for k in day.keys],
            columns,
            staging,
            glob=glob,
        )
        try:
            run_copy(
                sql,
                env,
                timeout=copy_timeout,
                attempts=copy_attempts,
                threads=per_process,
            )
        except SyncError as exc:
            print(f"    {plan.target.name} {day.date}: FAILED: {exc}",
                  file=sys.stderr)
            staging.unlink(missing_ok=True)
            return False

        if not completed_file(staging):
            print(f"    {plan.target.name} {day.date}: wrote no data",
                  file=sys.stderr)
            staging.unlink(missing_ok=True)
            return False

        staging.replace(destination)
        return True

    written = 0
    finished = 0
    # `as_completed` is drained by this one thread, so the counters need no lock.
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(fetch, day) for day in plan.days]
        for future in as_completed(futures):
            if future.result():
                written += 1
            finished += 1
            if finished % 25 == 0 or finished == total:
                print(
                    f"    {plan.target.name}: {finished}/{total} days, "
                    f"{written} written",
                    flush=True,
                )
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sync the Hydromancer Reservoir archive locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--store", default=DEFAULT_STORE, help="local store root")
    p.add_argument(
        "--datasets",
        default=",".join(t.name for t in SYNC_TARGETS),
        help=f"comma-separated subset of: {', '.join(t.name for t in SYNC_TARGETS)}",
    )
    p.add_argument(
        "--columns",
        choices=("research", "wide", "flow"),
        default="research",
        help="research: only what a backtest needs today (default, cheapest). "
             "flow: enough to reconstruct position changes from the fills, plus "
             "what each trade cost and maker/taker. "
             "wide: also keep everything except identifier columns",
    )
    p.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="symbols to keep where the archive stores one file per market",
    )
    p.add_argument("--limit", type=int, default=None,
                   help="fetch at most this many days per dataset")
    p.add_argument(
        "--max-gb",
        type=float,
        default=100.0,
        help="refuse to start if the estimated transfer exceeds this",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and cost, move nothing")
    p.add_argument(
        "--profile",
        default=None,
        help="AWS CLI profile holding the archive credentials. Without it the "
             "default profile (or AWS_PROFILE) is used, and an empty default "
             "profile fails with a message that names the ones that do exist.",
    )
    p.add_argument(
        "--copy-timeout",
        type=int,
        default=DEFAULT_COPY_TIMEOUT,
        metavar="SECONDS",
        help=f"seconds allowed for one day's COPY (default "
             f"{DEFAULT_COPY_TIMEOUT}). Raise it for `fills`: multiplied by the "
             f"per-stream rate this is a ceiling on how large a day can ever "
             f"succeed, and the fills projection averages ~93 MB/day against a "
             f"~36 MB ceiling at the default.",
    )
    p.add_argument(
        "--copy-attempts",
        type=int,
        default=DEFAULT_COPY_ATTEMPTS,
        metavar="N",
        help=f"retries per day (default {DEFAULT_COPY_ATTEMPTS}). A retry "
             f"re-pays only the bytes moved before it failed, so it is nearly "
             f"free for an immediate failure (this link raises SSL connect "
             f"errors routinely) and expensive only for a timeout. Lower it "
             f"with that difference in mind, not just because the timeout is "
             f"large.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="days to fetch concurrently (default 4). The transfer is "
             "latency-bound - one day moves ~34 MB in ~45s, most of it spent "
             "waiting on range requests - so overlapping days helps. It does "
             "NOT scale with the count: measured on the fills dataset, 12 days "
             "concurrently moved 0.30 GB/h where 4 moved 0.85 GB/h, and every "
             "stream was then inside the same per-day timeout, so they timed "
             "out together. Bytes billed are unchanged either way.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    wanted = [s.strip() for s in args.datasets.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in TARGETS_BY_NAME]
    if unknown:
        print(f"unknown dataset(s): {', '.join(unknown)}", file=sys.stderr)
        return 2
    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())

    store = Path(args.store)

    # Credentials are resolved before planning, not after. Listing the archive
    # is itself an authenticated call, so waiting until the transfer starts
    # would report "planning failed" for what is really a credentials problem.
    try:
        env = duckdb_env(args.profile)
    except SyncError as exc:
        print(f"cannot start: {exc}", file=sys.stderr)
        return 1

    try:
        plans = [
            plan_dataset(
                TARGETS_BY_NAME[name], store, symbols, args.limit,
                args.columns, env,
            )
            for name in wanted
        ]
    except SyncError as exc:
        print(f"planning failed: {exc}", file=sys.stderr)
        return 1

    transfer_gb = report(plans, args.columns, args.dry_run)

    if args.dry_run or transfer_gb == 0:
        return 0

    if transfer_gb > args.max_gb:
        print(
            f"refusing to proceed: {transfer_gb:,.2f} GB exceeds --max-gb "
            f"{args.max_gb:g}. Narrow with --datasets or --limit, or raise "
            "the cap deliberately.",
            file=sys.stderr,
        )
        return 3

    print(f"  syncing ({max(1, args.workers)} day(s) at a time)...")
    total = 0
    for plan in plans:
        columns = plan.target.columns_for(args.columns)
        total += sync_dataset(
            plan,
            store,
            columns,
            env,
            dry_run=False,
            workers=args.workers,
            copy_timeout=args.copy_timeout,
            copy_attempts=args.copy_attempts,
        )
    print()
    print(f"  wrote {total} day-file(s) under {store}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted; re-running will resume")
        raise SystemExit(130)
