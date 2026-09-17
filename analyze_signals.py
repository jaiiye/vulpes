#!/usr/bin/env python3
"""Forward signal-quality analysis over the dry-run decision journal.

Why this exists
---------------
The backtest can only replay the ~60% of the model that does not need whale
data, and it just showed no edge on that 60%. This tool closes the gap: it
measures the LIVE signal - the full model, whale factor included - using the
signals the dry-run has already been recording every 15 minutes.

It does not need months of history to say something useful. At ~96 signals a
day, two weeks gives roughly 1,300 observations, which is enough for a first
read on whether the direction call carries information.

Two distinct questions, deliberately kept separate:

  1. Does the signal predict? Measured on ALL signals, including the ones the
     discipline layer rejected. This is the fundamental question and needs the
     fewest samples.
  2. Do the gates help? Blocked signals are a free control group: if the ones
     the gates refused would have won, the gates are too strict.

The price series is read from the journal itself - every signal records the
mark price at decision time - so this needs no network access and cannot
disagree with what the agent actually saw.

Usage:
    python analyze_signals.py
    python analyze_signals.py --journal logs/journal.jsonl --bucket-hours 4
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The gate-reason vocabulary belongs to the module that emits the messages, so
# that a new gate is declared in exactly one place.
from agent.discipline import bucket_block_reason  # noqa: E402

DEFAULT_JOURNAL = "logs/journal.jsonl"

# Horizons to score, in hours. The agent holds for days, so the short end is
# about signal timing and the long end about whether the call was right at all.
DEFAULT_HORIZONS = (1.0, 4.0, 24.0)

# The horizon used for the bucketed breakdowns (by score, by confidence, by
# gate). Fixed rather than configurable so the tables stay comparable between
# runs; the main table already reports every horizon.
GROUP_HORIZON_HOURS = 4.0

# A forward price is only used if a journal entry exists within this many
# seconds of the target instant. Collection runs every 15 minutes, so 30
# minutes tolerates one missed pass; anything larger means the series has a gap
# and the measurement would be against the wrong moment.
MAX_GAP_SECONDS = 30 * 60

# Two-sided 95% confidence: |z| above this is worth mentioning.
Z_SIGNIFICANT = 1.96

LONG = "long"
SHORT = "short"
NEUTRAL = "neutral"


@dataclass
class SignalObservation:
    """One recorded signal decision."""

    ts: float
    symbol: str
    action: str
    score: float
    confidence: float
    price: float
    blocked_by: str | None = None

    @property
    def is_directional(self) -> bool:
        return self.action in (LONG, SHORT)


@dataclass
class HorizonResult:
    """Hit statistics for one horizon."""

    hours: float
    n: int = 0
    hits: int = 0
    skipped: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.n if self.n else 0.0

    @property
    def z(self) -> float:
        """z-score against the 50% null. 0 when the sample is too small."""
        if self.n < 2:
            return 0.0
        se = math.sqrt(0.25 / self.n)
        return (self.hit_rate - 0.5) / se if se else 0.0

    @property
    def is_significant(self) -> bool:
        return abs(self.z) >= Z_SIGNIFICANT

    def verdict(self) -> str:
        if self.n < 30:
            return f"样本不足 (n={self.n})"
        if not self.is_significant:
            return "无显著优势"
        return "正向优势" if self.z > 0 else "反向（信号反着用更好）"


@dataclass
class AnalysisReport:
    """Everything the analysis produces."""

    total_signals: int = 0
    observations: list[SignalObservation] = field(default_factory=list)
    price_points: int = 0
    span_days: float = 0.0
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_signals(path: str | Path) -> tuple[list[SignalObservation], list[str]]:
    """Read `signal` events from a journal, skipping malformed lines."""
    path = Path(path)
    if not path.exists():
        return [], [f"journal not found: {path}"]

    out: list[SignalObservation] = []
    bad = 0

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if rec.get("kind") != "signal":
            continue

        try:
            price = float(rec["price"])
        except (KeyError, TypeError, ValueError):
            # A signal without a usable price cannot be scored forward.
            continue
        if price <= 0:
            continue

        out.append(
            SignalObservation(
                ts=float(rec.get("ts") or 0.0),
                symbol=str(rec.get("symbol") or "").upper(),
                action=str(rec.get("action") or NEUTRAL).lower(),
                score=float(rec.get("score") or 0.0),
                confidence=float(rec.get("confidence") or 0.0),
                price=price,
                blocked_by=rec.get("blocked_by") or None,
            )
        )

    warnings = [f"{bad} malformed journal line(s) skipped"] if bad else []
    return out, warnings


def build_price_series(
    observations: list[SignalObservation],
) -> list[tuple[float, float]]:
    """The journal's own price series: (timestamp, price), deduplicated."""
    by_ts: dict[float, float] = {}
    for obs in observations:
        if obs.ts and obs.price > 0:
            # Later record for the same instant wins; they are identical anyway.
            by_ts[obs.ts] = obs.price
    return sorted(by_ts.items())


def price_at(
    series: list[tuple[float, float]], target_ts: float, max_gap: float
) -> float | None:
    """The price closest to `target_ts`, or None if the series has a gap there.

    Scans outward from the nearest timestamp rather than assuming the series is
    evenly spaced: collection pauses and missed runs are expected, and a stale
    price silently used as a forward return would fabricate edge.
    """
    if not series:
        return None

    lo, hi = 0, len(series)
    while lo < hi:
        mid = (lo + hi) // 2
        if series[mid][0] < target_ts:
            lo = mid + 1
        else:
            hi = mid

    best: tuple[float, float] | None = None
    for idx in (lo - 1, lo):
        if 0 <= idx < len(series):
            candidate = series[idx]
            if best is None or abs(candidate[0] - target_ts) < abs(best[0] - target_ts):
                best = candidate

    if best is None or abs(best[0] - target_ts) > max_gap:
        return None
    return best[1]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_horizon(
    observations: list[SignalObservation],
    series: list[tuple[float, float]],
    hours: float,
    max_gap: float = MAX_GAP_SECONDS,
    subset=None,
) -> HorizonResult:
    """Direction hit rate at one forward horizon.

    A long is a hit if price rose, a short if it fell. Neutral signals carry no
    direction and are excluded rather than counted as misses.
    """
    result = HorizonResult(hours=hours)
    offset = hours * 3600.0
    now = time.time()

    for obs in observations:
        if subset is not None and not subset(obs):
            continue
        if not obs.is_directional:
            continue
        # Not enough forward data yet; scoring it would bias toward whatever
        # the last few bars happened to do.
        if obs.ts + offset > now:
            result.skipped += 1
            continue

        forward = price_at(series, obs.ts + offset, max_gap)
        if forward is None:
            result.skipped += 1
            continue

        moved_up = forward > obs.price
        if obs.action == LONG:
            hit = moved_up
        else:
            hit = not moved_up

        result.n += 1
        if hit:
            result.hits += 1

    return result


def bucket_score(score: float) -> str:
    if score >= 70:
        return ">=70"
    if score >= 60:
        return "60-70"
    if score > 40:
        return "40-60"
    if score > 30:
        return "30-40"
    return "<30"


def bucket_confidence(conf: float) -> str:
    if conf >= 0.8:
        return ">=80%"
    if conf >= 0.6:
        return "60-80%"
    if conf >= 0.4:
        return "40-60%"
    return "<40%"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_horizon_table(results: list[HorizonResult], indent: str = "  ") -> str:
    lines = [
        f"{indent}{'horizon':>8s} {'n':>6s} {'跳过':>5s} {'命中率':>8s} {'z':>7s}  判定",
    ]
    for r in results:
        # `跳过` counts signals that could not be scored, either because the
        # forward price has not happened yet or because the journal has a gap
        # there. A large number means the sample is being eroded, which is worth
        # seeing next to the hit rate rather than trusting it blind.
        lines.append(
            f"{indent}{'+' + format_hours(r.hours):>8s} {r.n:>6d} {r.skipped:>5d} "
            f"{r.hit_rate * 100:>7.1f}% {r.z:>+7.2f}  {r.verdict()}"
        )
    return "\n".join(lines)


def format_hours(hours: float) -> str:
    return f"{hours:g}h"


def render_bucket_rows(
    observations: list[SignalObservation],
    series: list[tuple[float, float]],
    max_gap: float,
    names: tuple[str, ...],
    key_fn,
    width: int = 8,
) -> list[str]:
    """One table row per bucket that has at least one scoreable signal.

    `names` drives the display order; buckets absent from it are not shown.
    Buckets with nothing to score are dropped rather than printed as zeros: a
    row of `n=0` carries no information and buries the rows that do.
    """
    grouped: dict[str, list[SignalObservation]] = collections.defaultdict(list)
    for obs in observations:
        grouped[key_fn(obs)].append(obs)

    rows: list[str] = []
    for name in names:
        group = grouped.get(name)
        if not group:
            continue
        result = score_horizon(group, series, GROUP_HORIZON_HOURS, max_gap)
        if result.n == 0:
            continue
        rows.append(
            f"  {name:>{width}s} {result.n:>6d} "
            f"{result.hit_rate * 100:>7.1f}% {result.z:>+7.2f}"
        )
    return rows


def build_report(
    journal: str | Path = DEFAULT_JOURNAL,
    horizons: tuple[float, ...] = DEFAULT_HORIZONS,
    max_gap: float = MAX_GAP_SECONDS,
) -> tuple[AnalysisReport, str]:
    """Analyse a journal and return (report, rendered text)."""
    observations, warnings = load_signals(journal)
    report = AnalysisReport(
        total_signals=len(observations), observations=observations, warnings=warnings
    )

    if not observations:
        return report, (
            f"journal: {journal}\n"
            "没有可分析的信号记录。采集脚本需要先运行一段时间。\n"
            f"（提示：{'; '.join(warnings) if warnings else '文件为空或不存在'}）"
        )

    series = build_price_series(observations)
    report.price_points = len(series)
    stamps = [o.ts for o in observations if o.ts]
    if stamps:
        report.span_days = (max(stamps) - min(stamps)) / 86400.0

    out: list[str] = []
    out.append(f"journal: {journal}")
    out.append(
        f"信号总数: {report.total_signals}   价格点: {report.price_points}   "
        f"跨度: {report.span_days:.2f} 天"
    )

    directional = [o for o in observations if o.is_directional]
    out.append(
        f"可评分的方向性信号: {len(directional)}"
        f"（其中中性 {report.total_signals - len(directional)} 条，"
        "中性没有方向，不计入命中率）"
    )

    if report.span_days < 7:
        out.append(
            "  !! 跨度不足 7 天，下面的数字只能当作管道连通性的验证，"
            "还不能当作结论"
        )
    for warn in warnings:
        out.append(f"  !! {warn}")
    out.append("")

    if not directional:
        # The usual state early on: the score never crosses a threshold, so
        # there is nothing to score. A wall of empty tables would bury the one
        # genuinely useful number, which is the gate distribution below.
        out.append("=== 方向命中率 ===")
        out.append("  暂无方向性信号，无法评估。下方只报告门禁分布。")
        out.append("")
    else:
        # --- 1. Direction hit rate over all signals ----------------------
        out.append("=== 方向命中率（全部信号，含被门禁拒绝的）===")
        out.append(
            format_horizon_table(
                [score_horizon(observations, series, h, max_gap) for h in horizons]
            )
        )
        out.append("")
        out.append(
            "  说明：中性信号不计入（它们本就没有方向）；"
            "|z| >= 1.96 才算 95% 置信下有优势。"
        )
        out.append("")

        # --- 2. By score bucket -----------------------------------------
        out.append(f"=== 按得分分档（{GROUP_HORIZON_HOURS:g}h 前瞻）===")
        out.append(f"  {'score':>8s} {'n':>6s} {'命中率':>8s} {'z':>7s}")
        out.extend(
            render_bucket_rows(
                observations, series, max_gap,
                (">=70", "60-70", "40-60", "30-40", "<30"),
                key_fn=lambda o: bucket_score(o.score),
            )
        )
        out.append("")

        # --- 3. By confidence bucket ------------------------------------
        out.append(f"=== 按置信度分档（{GROUP_HORIZON_HOURS:g}h 前瞻）===")
        out.append(f"  {'conf':>8s} {'n':>6s} {'命中率':>8s} {'z':>7s}")
        out.extend(
            render_bucket_rows(
                observations, series, max_gap,
                (">=80%", "60-80%", "40-60%", "<40%"),
                key_fn=lambda o: bucket_confidence(o.confidence),
            )
        )
        out.append("")

    # --- 4. Blocked signals: the free control group -----------------------
    blocked = [o for o in observations if o.blocked_by]
    passed = [o for o in observations if not o.blocked_by]
    out.append("=== 被门禁拒绝的信号（对照组）===")
    out.append(
        f"  通过 {len(passed)} 条 / 拒绝 {len(blocked)} 条"
        f"（拒绝率 {len(blocked) / len(observations) * 100:.0f}%）"
    )

    r_passed = score_horizon(passed, series, GROUP_HORIZON_HOURS, max_gap)
    r_blocked = score_horizon(blocked, series, GROUP_HORIZON_HOURS, max_gap)
    out.append(
        f"  通过的信号    n={r_passed.n:>5d}  命中率 {r_passed.hit_rate * 100:>5.1f}%  z={r_passed.z:>+.2f}"
    )
    out.append(
        f"  被拒的信号    n={r_blocked.n:>5d}  命中率 {r_blocked.hit_rate * 100:>5.1f}%  z={r_blocked.z:>+.2f}"
    )
    if r_blocked.n >= 30 and r_blocked.is_significant and r_blocked.z > 0:
        out.append(
            "  !! 被拒信号显著命中：说明门禁过于严格，正在挡掉有效的信号"
        )
    out.append("")

    # Normalise each reason ONCE and group in the same pass, rather than
    # re-deriving the bucket for every observation on every gate's turn.
    by_gate: dict[str, list[SignalObservation]] = collections.defaultdict(list)
    for obs in blocked:
        by_gate[bucket_block_reason(obs.blocked_by)].append(obs)

    if by_gate:
        out.append("  拒绝原因分布（已归一化；原始文本带变量，直接计数会几乎每行一个值）:")
        rate_header = f"{GROUP_HORIZON_HOURS:g}h命中率"
        out.append(
            f"    {'gate':<16s} {'n':>5s} {'被拒占比':>9s} {rate_header:>9s} {'z':>7s}"
        )
        for gate, group in sorted(by_gate.items(), key=lambda kv: -len(kv[1])):
            result = score_horizon(group, series, GROUP_HORIZON_HOURS, max_gap)
            share = len(group) / len(blocked) * 100
            out.append(
                f"    {gate:<16s} {len(group):>5d} {share:>8.0f}% "
                f"{result.hit_rate * 100:>8.1f}% {result.z:>7.2f}"
            )
    out.append("")

    return report, "\n".join(out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Forward signal-quality analysis over the decision journal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--journal", default=DEFAULT_JOURNAL)
    p.add_argument(
        "--horizons",
        default="1,4,24",
        help="comma-separated forward horizons in hours",
    )
    p.add_argument(
        "--max-gap-minutes",
        type=float,
        default=MAX_GAP_SECONDS / 60,
        help="reject a forward price further than this from the target instant",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        horizons = tuple(float(h) for h in args.horizons.split(",") if h.strip())
    except ValueError:
        print("--horizons must be comma-separated numbers", file=sys.stderr)
        return 2
    if not horizons:
        print("--horizons must not be empty", file=sys.stderr)
        return 2

    report, text = build_report(
        journal=args.journal,
        horizons=horizons,
        max_gap=args.max_gap_minutes * 60,
    )
    print(text)
    return 0 if report.observations else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        raise SystemExit(130)
