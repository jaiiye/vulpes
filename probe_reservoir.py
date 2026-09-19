#!/usr/bin/env python3
"""Measure the Hydromancer Reservoir archive before downloading anything.

The archive is requester-pays: every byte transferred is billed to this
account. Nobody publishes the volume, so the only honest way to plan a
download is to measure it first.

This script lists and summarises. It does **not** download data unless you ask
it to with `--sample`, which fetches a single object per dataset to read its
parquet schema. Listing costs about five millionths of a dollar per request;
the sample is a few megabytes. Nothing here is expensive, and nothing here is
irreversible.

    python probe_reservoir.py                  # coverage + size estimate
    python probe_reservoir.py --sample         # also verify the real schema
    python probe_reservoir.py --days 1         # budget against 1 month

Requires the AWS CLI, configured credentials, and network access. DuckDB is
only needed for `--sample`.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

BUCKET = "hydromancer-reservoir"
REGION = "ap-northeast-1"

# AWS bills data transfer out to the internet, first 100 GB per month free,
# aggregated across services. Beyond that, Tokyo is among the pricier regions.
FREE_EGRESS_GB_PER_MONTH = 100.0
EGRESS_USD_PER_GB = 0.114
GET_USD_PER_1000 = 0.0004


@dataclass
class Dataset:
    """One archive dataset, and what it would cost us to take."""

    name: str
    prefix: str
    note: str
    # True when files are split per coin, which means we can take only the
    # symbols we trade instead of every market on the venue.
    per_coin: bool = False
    # Filled in by the probe.
    days: int = 0
    first_date: str = ""
    last_date: str = ""
    sample_date: str = ""
    sample_bytes: int = 0
    sample_objects: int = 0
    error: str = ""

    @property
    def per_day_bytes(self) -> float:
        """Bytes for one full day of this dataset.

        This is the sampled day's total, deliberately NOT divided by the object
        count. A per-coin dataset holds hundreds of small files in a single
        day, and dividing here understates the whole dataset by that same
        factor - which is exactly how the order book first came out at 0.07 GB
        against a measured 44 MB per day.
        """
        return float(self.sample_bytes)

    def total_bytes(self) -> float:
        return self.per_day_bytes * self.days

    def total_objects(self) -> int:
        return self.sample_objects * self.days


DATASETS: list[Dataset] = [
    Dataset(
        "candles",
        "by_dex/hyperliquid/candles/1s",
        "1s OHLCV for every market; the only censored interval offered",
    ),
    Dataset(
        "orderbook",
        "by_dex/hyperliquid/orderbook/1m/perps",
        "L2 20 levels, one file per coin per day",
        per_coin=True,
    ),
    Dataset(
        "positions",
        "by_dex/hyperliquid/snapshots/perp",
        "per-user per-market open positions, end of day",
    ),
    Dataset(
        "account_values",
        "global/snapshots/account_values",
        "per-user per-dex aggregates; the min_account_value filter reads this",
    ),
    Dataset(
        "fills",
        "by_dex/hyperliquid/fills/perp/all",
        "every perp fill; the only way to rebuild a historical leaderboard",
    ),
]


class ProbeError(RuntimeError):
    """The archive could not be reached or understood."""


# ---------------------------------------------------------------------------
# Cost arithmetic (pure, so it can be tested without touching AWS)
# ---------------------------------------------------------------------------


def to_gb(num_bytes: float) -> float:
    """Bytes to binary gigabytes, matching how AWS meters S3."""
    return num_bytes / 1024**3


def egress_cost_usd(total_gb: float, months: int = 1) -> float:
    """Transfer cost, with the monthly free allowance applied per month.

    The allowance is per calendar month and does not roll over, so the same
    download costs nothing when spread over enough months and real money when
    it is not. That is the single biggest lever on the bill, which is why the
    number is reported as a range rather than as a point.
    """
    if total_gb <= 0 or months < 1:
        return 0.0
    billable = max(0.0, total_gb - FREE_EGRESS_GB_PER_MONTH * months)
    return billable * EGRESS_USD_PER_GB


def months_needed_to_stay_free(total_gb: float) -> int:
    """How many months to spread a download over to pay nothing for transfer.

    Ceiling division, not floor-plus-one. `int(x) + 1` always spends an extra
    month even when the volume divides evenly, so an exactly-100 GB download
    would be reported as needing two months when one is enough.
    """
    if total_gb <= 0:
        return 1
    return max(1, math.ceil(total_gb / FREE_EGRESS_GB_PER_MONTH))


def request_cost_usd(requests: int) -> float:
    return requests / 1000.0 * GET_USD_PER_1000


def human_gb(num_bytes: float) -> str:
    return f"{to_gb(num_bytes):,.2f} GB"


def projected_fraction(
    column_bytes: dict[str, int], needed: set[str]
) -> float:
    """Share of a parquet file a column-projected read still has to fetch.

    DuckDB reads only the column chunks a query references, so the floor is the
    sum of the needed columns over the sum of all of them. This is an upper
    bound in the useful direction: it is what a query must read, and no cheaper
    implementation can do better without also skipping pages inside a column.

    Row-group pruning is the other lever, and it applies only where the file
    carries min/max statistics for the filtered column. Neither `coin` in the
    candle file nor `address` in the fills file does, which is why filtering by
    symbol saves nothing on transfer - the filter runs after the bytes arrive.
    `timestamp` does carry statistics, so a narrowed time range can prune.
    """
    total = sum(column_bytes.values())
    if total <= 0:
        return 0.0
    wanted = sum(size for name, size in column_bytes.items() if name in needed)
    return wanted / total


def projected_bytes(column_bytes: dict[str, int], needed: set[str]) -> int:
    return sum(size for name, size in column_bytes.items() if name in needed)


# ---------------------------------------------------------------------------
# AWS access
# ---------------------------------------------------------------------------


def aws_json(args: list[str]) -> dict:
    """Run an `aws s3api` call and parse its JSON output."""
    cmd = ["aws", *args, "--region", REGION, "--output", "json"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except FileNotFoundError as exc:  # noqa: PERF203 - startup check
        raise ProbeError(
            "the aws CLI was not found on PATH; install it and configure "
            "credentials before probing"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ProbeError(f"aws call timed out: {' '.join(cmd)}") from exc

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise ProbeError(
            f"aws call failed ({proc.returncode}): "
            f"{detail[-1] if detail else 'no output'}"
        )
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise ProbeError(f"could not parse aws output: {proc.stdout[:200]}") from exc


def list_partitions(prefix: str) -> list[str]:
    """The `date=...` partitions under a dataset, in order."""
    partitions: list[str] = []
    token: str | None = None
    while True:
        args = [
            "s3api", "list-objects-v2",
            "--bucket", BUCKET,
            "--prefix", prefix.rstrip("/") + "/",
            "--delimiter", "/",
            "--request-payer", "requester",
        ]
        if token:
            args += ["--starting-token", token]
        payload = aws_json(args)
        for entry in payload.get("CommonPrefixes") or []:
            name = entry.get("Prefix", "").rstrip("/").split("/")[-1]
            if name.startswith("date="):
                partitions.append(name[len("date="):])
        if not payload.get("IsTruncated"):
            return sorted(partitions)
        token = payload.get("NextContinuationToken")
        if not token:
            return sorted(partitions)


def measure_day(prefix: str, date: str) -> tuple[int, int]:
    """(total_bytes, object_count) for one date partition."""
    total = 0
    count = 0
    token: str | None = None
    while True:
        args = [
            "s3api", "list-objects-v2",
            "--bucket", BUCKET,
            "--prefix", f"{prefix.rstrip('/')}/date={date}/",
            "--request-payer", "requester",
        ]
        if token:
            args += ["--starting-token", token]
        payload = aws_json(args)
        for obj in payload.get("Contents") or []:
            total += int(obj.get("Size", 0))
            count += 1
        if not payload.get("IsTruncated"):
            return total, count
        token = payload.get("NextContinuationToken")
        if not token:
            return total, count


def probe_coverage(dataset: Dataset) -> None:
    partitions = list_partitions(dataset.prefix)
    if not partitions:
        dataset.error = "no date partitions found"
        return
    dataset.days = len(partitions)
    dataset.first_date = partitions[0]
    dataset.last_date = partitions[-1]
    # Measure the most recent complete day rather than a fixed calendar date,
    # which may predate the dataset or fall in a known gap.
    dataset.sample_date = partitions[-2] if len(partitions) > 1 else partitions[-1]


# ---------------------------------------------------------------------------
# Schema verification
# ---------------------------------------------------------------------------


def sample_schema(prefix: str, date: str) -> tuple[str, list[tuple[str, str]]]:
    """Download one object and read its parquet schema.

    This is the only step that fetches bytes. It is worth doing once, because
    the published schema and the stored schema have disagreed elsewhere in
    this archive: the docs list columns that appear only after a certain date.
    """
    args = [
        "s3api", "list-objects-v2",
        "--bucket", BUCKET,
        "--prefix", f"{prefix.rstrip('/')}/date={date}/",
        "--max-items", "1",
        "--request-payer", "requester",
    ]
    payload = aws_json(args)
    contents = payload.get("Contents") or []
    if not contents:
        raise ProbeError(f"no objects under {prefix}/date={date}/")
    key = contents[0]["Key"]

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "sample.parquet"
        subprocess.run(
            ["aws", "s3api", "get-object",
             "--bucket", BUCKET, "--key", key,
             "--request-payer", "requester",
             "--region", REGION, str(target)],
            capture_output=True, text=True, timeout=300, check=True,
        )
        proc = subprocess.run(
            ["duckdb", "-json", "-c",
             f"DESCRIBE SELECT * FROM read_parquet('{target}')"],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            raise ProbeError(f"duckdb could not read the sample: {proc.stderr.strip()}")
        rows = json.loads(proc.stdout or "[]")
    return key, [(r.get("column_name", "?"), r.get("column_type", "?")) for r in rows]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(datasets: list[Dataset], months: int, show_sample: bool) -> int:
    print("=" * 78)
    print(" HYDROMANCER RESERVOIR - scope before download")
    print(f" bucket s3://{BUCKET}   region {REGION}   requester-pays: you pay transfer")
    print("=" * 78)
    print()

    measured: list[Dataset] = []
    for ds in datasets:
        print(f"--- {ds.name}  ({ds.prefix})")
        if ds.error:
            print(f"    ERROR: {ds.error}")
            print()
            continue

        print(f"    coverage   {ds.days} days   {ds.first_date} -> {ds.last_date}")
        print(
            f"    sampled    {ds.sample_date}: {ds.sample_objects} object(s), "
            f"{human_gb(ds.sample_bytes)}"
        )
        if ds.per_coin:
            print(
                "    NOTE       one file per coin per day, so a 3-symbol "
                "download is roughly 3/objects of the full size"
            )
        print(f"    {ds.note}")

        if show_sample:
            try:
                key, schema = sample_schema(ds.prefix, ds.sample_date)
                print(f"    schema of {key.rsplit('/', 1)[-1]}:")
                for column, kind in schema:
                    print(f"      {column:24s} {kind}")
            except (ProbeError, subprocess.SubprocessError) as exc:
                print(f"    schema probe failed: {exc}")
        print()
        measured.append(ds)

    if not measured:
        print("nothing measurable; check credentials and the prefix list")
        return 1

    total_bytes = sum(ds.total_bytes() for ds in measured)
    total_objects = sum(ds.total_objects() for ds in measured)

    print("=" * 78)
    print(" ESTIMATED FULL DOWNLOAD")
    print("=" * 78)
    print(f"  {'dataset':18s}{'days':>6s}{'per day':>12s}{'total':>14s}")
    for ds in measured:
        print(
            f"  {ds.name:18s}{ds.days:6d}"
            f"{human_gb(ds.per_day_bytes):>12s}{human_gb(ds.total_bytes()):>14s}"
        )
    print(f"  {'TOTAL':18s}{'':6s}{'':12s}{human_gb(total_bytes):>14s}")
    print()

    total_gb = to_gb(total_bytes)
    free_months = months_needed_to_stay_free(total_gb)
    transfer_now = egress_cost_usd(total_gb, months=1)
    transfer_spread = egress_cost_usd(total_gb, months=max(months, free_months))

    print("COST (transfer is what you pay; storage is the bucket owner's)")
    print(f"  volume                {total_gb:,.2f} GB")
    print(f"  objects               ~{int(total_objects):,}")
    print(f"  free allowance        {FREE_EGRESS_GB_PER_MONTH:g} GB per calendar "
          "month, not carried over")
    print(f"  if pulled in 1 month  ${transfer_now:,.2f}")
    unit = "month" if free_months == 1 else "months"
    print(f"  if spread over {free_months} {unit}  ${transfer_spread:,.2f}")
    print(f"  GET requests          ${request_cost_usd(total_objects):,.4f}")
    print()
    print(
        f"  Realistic floor is $0: spreading {total_gb:,.1f} GB over "
        f"{free_months} {unit} stays inside the free allowance."
    )
    print("  Verify the Tokyo per-GB rate in the AWS Pricing Calculator before")
    print("  committing; the rate above is a public figure, not a quote.")
    print()
    print("  This estimate is worst case: it assumes every byte is transferred.")
    print("  DuckDB reading parquet from S3 pulls only the column chunks a query")
    print("  needs, and a filter on a sorted column can skip row groups entirely.")
    print("  Measuring one real query is the next step, not this one.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Probe the Hydromancer Reservoir archive before downloading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--datasets",
        default="",
        help="comma-separated subset to probe, e.g. candles,positions",
    )
    p.add_argument(
        "--sample",
        action="store_true",
        help="download one object per dataset to verify the real parquet schema",
    )
    p.add_argument(
        "--days",
        type=int,
        default=1,
        help="months the download is spread over, for the cost model",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    selected = DATASETS
    if args.datasets:
        wanted = {s.strip() for s in args.datasets.split(",") if s.strip()}
        selected = [d for d in DATASETS if d.name in wanted]
        unknown = wanted - {d.name for d in DATASETS}
        if unknown:
            print(f"unknown dataset(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            print(f"known: {', '.join(d.name for d in DATASETS)}", file=sys.stderr)
            return 2

    try:
        for dataset in selected:
            try:
                probe_coverage(dataset)
                if not dataset.error:
                    dataset.sample_bytes, dataset.sample_objects = measure_day(
                        dataset.prefix, dataset.sample_date
                    )
            except ProbeError as exc:
                dataset.error = str(exc)
        return report(selected, months=max(1, args.days), show_sample=args.sample)
    except ProbeError as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130)
