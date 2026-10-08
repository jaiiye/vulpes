#!/usr/bin/env python3
"""Step 0 排雷：篮子实盘化之前必须清掉的三件事。

重写估算（RESEARCH.md §二.30）说约 1300 行、4-5 天，但那里列了三个不改行数、
只改难度的风险，其中两条成立就会改变设计。这个脚本在写任何篮子代码之前把
它们量出来：

  A. 分档口径对齐
     回测的非流动半用「每 4h 美元成交量中位数」分档，实盘只能拿到「日成交额」
     (dayNtlVlm)。比例因子不影响分档（只取相对中位数、切两半），但排名必须
     一致，否则测的和跑的是两个池。这里把两种效应分开量：
       - 纯口径差异：同一窗口内 4h 中位 vs 日中位（无时间漂移）
       - 时间漂移：归档末的日中位 vs 当下的 day_volume
     报告 Jaccard 重叠率与 Spearman 排名相关。

  B. market 单批量可行（dry-run）
     执行层默认 maker（limit + 5bp + 挂单等待 + 取消剩余），而成本结论是 taker
     口径。这里在 dry-run 下按 market 单连开 N 笔，看有没有被拒、被拒在哪。

  C. 串行耗时
     全仓库无 async / 批量下单，N 笔串行。这里量真实网络往返，外推到一轮篮子
     （池的 K 线 + 两条腿的下单）。只读接口的往返是下单耗时的下界。

用法：
    python3 probe_basket_step0.py              # 全部
    python3 probe_basket_step0.py --only a     # 只跑分档对齐
    python3 probe_basket_step0.py --legs 34 --per-coin 2000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent.config import BotConfig
from agent.execution import Broker
from agent.market_data import HyperliquidMarket
from agent.synthesizer import Journal
from backtest.cross_section import liquid_split, load_archive_panel
from backtest.universe import liquidity_medians

ARCHIVE = "data/canonical/candles"
WINDOW_DAYS = 30


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _spearman(x: list[float], y: list[float]) -> float:
    """Rank correlation. Hand-rolled to avoid a scipy dependency."""

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    return num / den if den else float("nan")


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _archive_bounds() -> tuple[int, int]:
    """(latest_ms, earliest_ms) of the candle archive."""
    sql = (
        "SELECT max(epoch_ms(timestamp)) hi, min(epoch_ms(timestamp)) lo "
        f"FROM read_parquet('{ARCHIVE}/*.parquet');"
    )
    proc = subprocess.run(
        ["duckdb", "-json"], input=sql, capture_output=True, text=True, timeout=900
    )
    if proc.returncode != 0:
        raise SystemExit(f"duckdb failed: {(proc.stderr or '').strip()[-200:]}")
    row = json.loads(proc.stdout or "[]")[0]
    return int(row["hi"]), int(row["lo"])


# ---------------------------------------------------------------------------
# A. 分档口径对齐
# ---------------------------------------------------------------------------
def part_a(window_days: int) -> dict:
    print("\n=== A. 分档口径对齐 ===")

    live = HyperliquidMarket(testnet=False)
    ctx = live.asset_contexts()
    day_volume = {c.name: c.day_volume for c in ctx if c.day_volume > 0}
    print(f"实盘: {len(day_volume)} 币有当日成交额 (asset_contexts)")

    hi, _lo = _archive_bounds()
    end = hi + 1
    start = end - int(timedelta(days=window_days).total_seconds() * 1000)
    fmt = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    print(f"归档窗: {fmt(start)} -> {fmt(end)}  ({window_days}d)")

    coins = sorted(day_volume)
    print(f"加载归档面板 ({len(coins)} 币)...")

    panel_4h = load_archive_panel(coins, interval_minutes=240, start_ms=start, end_ms=end)
    med_4h = liquidity_medians(panel_4h)
    print(f"  4h 面板: {len(panel_4h.symbols)} 币 x {len(panel_4h.times)} bar")

    panel_1d = load_archive_panel(coins, interval_minutes=1440, start_ms=start, end_ms=end)
    med_1d = liquidity_medians(panel_1d)
    print(f"  日 面板: {len(panel_1d.symbols)} 币 x {len(panel_1d.times)} bar")

    def lower_half(med: dict[str, float]) -> set[str]:
        ordered = sorted(med.values())
        cut = ordered[int(len(ordered) * 0.5)]
        return {s for s, v in med.items() if v < cut}

    illiq_4h = lower_half(med_4h)
    illiq_1d = lower_half(med_1d)

    # 实盘分档只在同时有两个口径的币上做，否则重叠率会被缺失币污染。
    common = sorted(set(med_4h) & set(med_1d) & set(day_volume))
    live_illiq = lower_half({s: day_volume[s] for s in common})

    # 纯口径差异：同一窗口，4h 中位 vs 日中位。
    both = sorted(set(med_4h) & set(med_1d))
    rho_unit = _spearman(
        [med_4h[s] for s in both], [med_1d[s] for s in both]
    )
    jac_unit = _jaccard(illiq_4h & set(both), illiq_1d & set(both))

    # 时间漂移：归档末的日中位 vs 当下 day_volume。
    rho_drift = _spearman(
        [med_1d[s] for s in common], [day_volume[s] for s in common]
    )
    jac_drift = _jaccard(illiq_1d & set(common), live_illiq)

    print(f"\n  纯口径差异 (4h中位 vs 日中位, 同窗口):")
    print(f"    Spearman {rho_unit:+.4f} | 非流动半重叠 {jac_unit:.1%}")
    print(f"  时间漂移   (归档末 vs 当下, 同口径):")
    print(f"    Spearman {rho_drift:+.4f} | 非流动半重叠 {jac_drift:.1%}")
    print(f"\n  共同币 {len(common)} | 归档非流动半 {len(illiq_1d & set(common))} "
          f"| 实盘非流动半 {len(live_illiq)}")

    verdict = (
        "对齐" if jac_unit >= 0.8 and jac_drift >= 0.7
        else "漂移过大：实盘必须按归档口径重建，或改回测口径"
    )
    print(f"  结论: {verdict}")

    return {
        "n_live": len(day_volume),
        "n_common": len(common),
        "rho_unit": rho_unit,
        "jaccard_unit": jac_unit,
        "rho_drift": rho_drift,
        "jaccard_drift": jac_drift,
        "verdict": verdict,
        "live_illiquid": sorted(live_illiq),
        "archive_illiquid": sorted(illiq_1d & set(common)),
    }


# ---------------------------------------------------------------------------
# B. market 单批量可行 (dry-run)
# ---------------------------------------------------------------------------
def part_b(legs: int, per_coin: float, universe: list[str]) -> dict:
    print("\n=== B. market 单批量可行 (dry-run) ===")

    cfg = BotConfig()
    cfg.execution.dry_run = True
    cfg.execution.order_type = "market"   # 成本结论是 taker 口径
    cfg.risk.leverage = 3

    market = HyperliquidMarket(testnet=True)
    journal = Journal("/tmp/step0_journal.jsonl")
    broker = Broker(cfg, market, journal)

    mids = market.all_mids()
    print(f"中间价: {len(mids)} 币")

    picked = [s for s in universe if mids.get(s, 0) > 0][:legs]
    print(f"对 {len(picked)} 个币各开 ${per_coin:,.0f} (market, dry-run)")

    ok, failed = [], []
    t0 = time.time()
    for i, sym in enumerate(picked):
        price = mids[sym]
        size = per_coin / price
        side = "long" if i % 2 == 0 else "short"
        try:
            pos = broker.open_position(
                symbol=sym,
                side=side,
                size=size,
                price=price,
                leverage=cfg.risk.leverage,
            )
            ok.append((sym, side, pos.size, pos.notional))
        except Exception as exc:
            failed.append((sym, type(exc).__name__, str(exc)[:120]))
    elapsed = time.time() - t0

    print(f"\n  成功 {len(ok)} / 失败 {len(failed)}  耗时 {elapsed:.2f}s "
          f"({elapsed / max(len(picked), 1) * 1000:.0f}ms/笔, 无网络)")
    for sym, err, msg in failed[:10]:
        print(f"    FAIL {sym}: {err}: {msg}")

    notional = sum(n for _, _, _, n in ok)
    print(f"  名义敞口 ${notional:,.0f} (毛)")

    zero_sized = [s for s, _, sz, _ in ok if sz <= 0]
    if zero_sized:
        print(f"  警告: {len(zero_sized)} 笔 size 取整后为 0: {zero_sized[:5]}")

    return {
        "attempted": len(picked),
        "ok": len(ok),
        "failed": len(failed),
        "failures": failed[:20],
        "elapsed_s": elapsed,
        "gross_notional": notional,
    }


# ---------------------------------------------------------------------------
# C. 串行耗时
# ---------------------------------------------------------------------------
def part_c(pool: int, legs: int, samples: int = 12) -> dict:
    print("\n=== C. 串行耗时 (真实网络往返) ===")

    market = HyperliquidMarket(testnet=False)

    def timeit(fn, n: int) -> tuple[float, float]:
        ts = []
        for _ in range(n):
            t0 = time.time()
            try:
                fn()
            except Exception:
                pass
            ts.append(time.time() - t0)
        return statistics.fmean(ts), max(ts)

    t_mids, t_mids_max = timeit(market.all_mids, samples)
    print(f"  all_mids()       平均 {t_mids * 1000:6.0f}ms  最慢 {t_mids_max * 1000:6.0f}ms")

    t_ctx, _ = timeit(market.asset_contexts, samples)
    print(f"  asset_contexts() 平均 {t_ctx * 1000:6.0f}ms  (全池 {pool} 币, 1 次请求)")

    syms = [c.name for c in market.asset_contexts()][:samples]
    it = iter(syms * 3)

    def one_candle() -> None:
        try:
            market.candles(next(it), interval="1h", lookback=48)
        except StopIteration:
            pass

    t_candle, t_candle_max = timeit(one_candle, samples)
    print(f"  candles(1 币)    平均 {t_candle * 1000:6.0f}ms  最慢 {t_candle_max * 1000:6.0f}ms")

    kline = pool * t_candle
    orders = legs * t_candle  # 下单没有就地下单样本，用只读往返作下界
    print(f"\n  外推一轮篮子:")
    print(f"    {pool} 币 K 线串行  ~{kline:.0f}s")
    print(f"    {legs} 笔下单串行   ~{orders:.0f}s  (下界：只读往返)")
    print(f"    合计           ~{kline + orders:.0f}s")

    has_key = bool(os.environ.get("HYPERLIQUID_PRIVATE_KEY"))
    print(f"\n  真实下单往返: {'可测 (HYPERLIQUID_PRIVATE_KEY 已设置)' if has_key else '未测 (无私钥；以上为只读下界)'}")

    return {
        "all_mids_ms": t_mids * 1000,
        "asset_contexts_ms": t_ctx * 1000,
        "candles_ms": t_candle * 1000,
        "candles_max_ms": t_candle_max * 1000,
        "kline_total_s": kline,
        "orders_total_s": orders,
        "has_private_key": has_key,
    }


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=["a", "b", "c"], help="只跑其中一项")
    p.add_argument("--legs", type=int, default=34, help="两腿总笔数 (默认 34)")
    p.add_argument("--per-coin", type=float, default=2000.0, help="每币名义 USD")
    p.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    p.add_argument("--pool", type=int, default=117, help="池的币数 (默认非流动半)")
    p.add_argument("--json", help="把结果写成 JSON")
    args = p.parse_args()

    out: dict = {}
    universe: list[str] = []

    if args.only in (None, "a"):
        out["a"] = part_a(args.window_days)
        universe = out["a"].get("live_illiquid", [])

    if args.only in (None, "b"):
        if not universe:
            ctx = HyperliquidMarket(testnet=False).asset_contexts()
            vols = sorted((c.day_volume, c.name) for c in ctx if c.day_volume > 0)
            universe = [n for _, n in vols[: max(1, len(vols) // 2)]]
        out["b"] = part_b(args.legs, args.per_coin, universe)

    if args.only in (None, "c"):
        out["c"] = part_c(args.pool, args.legs)

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\n已写入 {args.json}")


if __name__ == "__main__":
    main()
