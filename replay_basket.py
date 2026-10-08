#!/usr/bin/env python3
"""用实盘代码在归档上逐期重放，验收「实盘就是回测里那个策略」。

为什么不能直接拿回测的数字当验收：回测走 `CrossSectionBacktester`，实盘走
`agent.basket` + `agent.rebalance`。两者共用选币代码（`rank_legs`），但成本、换手、
腿平衡的记账是各自实现的。要证明它们是同一个策略，必须让**实盘代码**在历史数据
上跑一遍，而不是假设等价。这是 RESEARCH.md §二.30 里 Step 4 的验收。

口径对齐（能对比的前提，每一条都对应一处可能的分歧）：

  * 池与分档：直接用 `cross_section_cost.window_panels()`，与回测拿**同一个**
    panel，池的差异被完全排除，剩下的差别只可能来自执行路径。
  * 每期收益：该期腿内所有币从 row 到 exit_row 的等权价格变化 —— 与回测
    `_leg_return` 同定义。**不跟踪 entry_price**：换手保留下来的币也按当期价格
    算，这正是回测的算法。
  * 成本：来自 dry-run broker 的真实记账（换手产生的开平双边费），按**单腿规模**
    折算，与回测 `cost = turnover * 2 * fee_bps` 同口径。未换手的币不产生成本，
    这一点两边一致，所以换手率才是成本的真正驱动。

用法：
    python3 replay_basket.py                 # 默认 3.2 bp/leg、6 个窗口
    python3 replay_basket.py --fee 0         # 只看毛收益，隔离成本模型
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent.basket import BasketConfig, build_basket
from agent.basket_risk import check_plan
from agent.config import BotConfig
from agent.execution import Broker
from agent.rebalance import execute_rebalance, plan_rebalance
from agent.synthesizer import Journal, LONG
from backtest.cross_section import (
    CrossSectionBacktester,
    CrossSectionConfig,
    negate,
    past_return,
    slice_rows,
)
from cross_section_cost import (
    DEFAULT_INTERVAL,
    DEFAULT_LOOKBACK,
    DEFAULT_HOLD,
    DEFAULT_WINDOWS,
    archive_coins,
    load_archive_panel,
    window_panels,
)

ARCHIVE = "data/canonical/candles"


class FeeRecordingBroker:
    """包住 dry-run broker，只为把真实记账的费用捞出来。

    只记 `exit_fee_usd`：开仓那一边的费用已经在 `Position.entry_fee_usd` 上记过，
    `CloseResult.cost_usd` 是双边合计，直接加会重复。
    """

    def __init__(self, inner: Broker) -> None:
        self.inner = inner
        self.fees = 0.0

    def reset(self) -> None:
        self.fees = 0.0

    def open_position(self, **kw):
        pos = self.inner.open_position(**kw)
        self.fees += pos.entry_fee_usd or 0.0
        return pos

    def close_position(self, position, price, reason):
        result = self.inner.close_position(position, price, reason)
        if result is not None:
            self.fees += result.exit_fee_usd or 0.0
        return result

    def __getattr__(self, name):
        return getattr(self.inner, name)


def dollar_volume(panel, symbol: str, row: int, lookback: int) -> float:
    """截止 row 的一段成交额（美元）。归档的 volume 是币量，要乘价格。"""
    series = panel.volume.get(symbol) or []
    total = 0.0
    for i in range(max(0, row - lookback + 1), row + 1):
        v = series[i] if i < len(series) else None
        if not v:
            continue
        p = panel.price(symbol, i)
        if p:
            total += p * v
    return total


@dataclass
class ReplayResult:
    periods: list[dict] = field(default_factory=list)
    fees_total: float = 0.0

    @property
    def n(self) -> int:
        return len(self.periods)

    @property
    def compounded_pct(self) -> float:
        """复利总净收益，与回测 `net_return_pct` 同口径（相对单腿规模）。"""
        eq = 1.0
        for p in self.periods:
            eq *= 1.0 + p["net"]
        return (eq - 1.0) * 100.0

    @property
    def mean_net_pct(self) -> float:
        return statistics.mean(p["net"] for p in self.periods) * 100.0 if self.periods else 0.0

    @property
    def median_net_pct(self) -> float:
        return statistics.median(p["net"] for p in self.periods) * 100.0 if self.periods else 0.0

    @property
    def cost_pct(self) -> float:
        return sum(p["cost"] for p in self.periods) * 100.0

    @property
    def mean_turnover_pct(self) -> float:
        return statistics.mean(p["turnover"] for p in self.periods) * 100.0 if self.periods else 0.0

    @property
    def mean_legs(self) -> float:
        return statistics.mean(p["n_legs"] for p in self.periods) if self.periods else 0.0

    def summary(self) -> str:
        return (
            f"n={self.n}  net {self.compounded_pct:+.2f}%  cost {self.cost_pct:.2f}%  "
            f"mean/period {self.mean_net_pct:+.3f}%  median/period {self.median_net_pct:+.3f}%  "
            f"turnover {self.mean_turnover_pct:.1f}%  legs {self.mean_legs:.1f}"
        )


def replay_panel(
    panel,
    broker: FeeRecordingBroker,
    cfg: BasketConfig,
    lookback: int = DEFAULT_LOOKBACK,
    hold: int = DEFAULT_HOLD,
    equity: float = 1e9,
    late_exit_factor: float = 0.0,
    verbose: bool = False,
) -> ReplayResult:
    """在一个 panel 上逐期走实盘代码。

    `late_exit_factor` 与 `CrossSectionConfig` 同名参数同义（默认 0 = 全损）。
    """
    result = ReplayResult()
    held: dict = {}

    # 从 0 起步，与回测 `start_row=0` 一致。前面几期 `past_return` 算不出来
    # （i - lookback < 0），`build_basket` 自然返回 None 被跳过 —— 让跳过发生在
    # 同一个地方，而不是靠两边各自选一个起点，那样期数和覆盖区间都会错开。
    row = 0
    while row + hold < len(panel):
        exit_row = row + hold

        # 只用截止 row 的数据排名 —— 这一行是防前视的全部内容。
        sub = slice_rows(panel, 0, row + 1)
        basket = build_basket(sub, cfg)
        if basket is None:
            row += hold
            continue

        prices = {
            s: p
            for s in basket.symbols
            if (p := panel.price(s, row)) is not None and p > 0
        }
        day_vols = {
            s: dollar_volume(panel, s, row, hold) for s in basket.symbols
        }

        plan = plan_rebalance(held, basket, prices, day_vols, cfg)
        reason = check_plan(plan, equity, cfg)
        if reason:
            if verbose:
                print(f"  row {row}: REFUSED {reason}")
            row += hold
            continue

        broker.reset()
        outcome = execute_rebalance(broker, plan, prices, leverage=1)
        held = outcome.positions

        # --- 收益：该期腿内所有币从 row 到 exit_row 的等权变化（= 回测口径）
        longs = [s for s, p in held.items() if p.side == LONG]
        shorts = [s for s, p in held.items() if p.side != LONG]

        def leg_return(names: list[str]) -> float | None:
            rets = []
            for s in names:
                a = panel.price(s, row)
                if not a or a <= 0:
                    continue
                b = panel.price(s, exit_row)
                if b and b > 0:
                    rets.append((b - a) / a)
                    continue
                # 到期没有 bar：这个币在持有期内停止交易了。回测按
                # `late_exit_factor` 计（默认 0 = 全损）而不是丢掉它，因为丢掉
                # 它正是长-输家篮子最容易沾上的幸存者偏差。这里必须同口径，
                # 否则两边的毛收益都对不上。
                rets.append(late_exit_factor - 1.0)
            return statistics.fmean(rets) if rets else None

        long_ret = leg_return(longs)
        short_ret = leg_return(shorts)
        if long_ret is None:
            row += hold
            continue
        short_ret = short_ret or 0.0

        # --- 成本：真实记账，按单腿规模折算（= 回测口径）
        long_notional = sum(held[s].notional for s in longs)
        short_notional = sum(held[s].notional for s in shorts)
        leg = (long_notional + short_notional) / 2.0
        cost_rate = (broker.fees / leg) if leg > 0 else 0.0

        result.periods.append(
            {
                "row": row,
                "n_legs": (len(longs) + len(shorts)) / 2.0,
                "long_return": long_ret,
                "short_return": short_ret,
                "turnover": plan.turnover,
                "cost": cost_rate,
                "net": long_ret - short_ret - cost_rate,
                "imbalance": outcome.imbalance,
            }
        )
        result.fees_total += broker.fees
        row += hold

    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fee", type=float, default=3.2, help="bp per leg (taker)")
    ap.add_argument("--windows", type=int, default=DEFAULT_WINDOWS)
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    ap.add_argument("--hold", type=int, default=DEFAULT_HOLD)
    ap.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    ap.add_argument("--per-coin", type=float, default=2000.0)
    ap.add_argument("--quantile", type=float, default=0.2)
    ap.add_argument("--equity", type=float, default=1e9)
    ap.add_argument(
        "--late-exit-factor",
        type=float,
        default=0.0,
        help="exit price assumed for a symbol with no bar at exit (0 = total loss)",
    )
    ap.add_argument("--archive", default=ARCHIVE)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", help="write per-period detail")
    args = ap.parse_args()

    cfg = BasketConfig(
        lookback=args.lookback,
        hold=args.hold,
        quantile=args.quantile,
        per_coin_usd=args.per_coin,
    )

    print(f"加载归档 ({args.interval}m bars)...")
    t0 = time.time()
    coins = archive_coins(args.archive)
    panel = load_archive_panel(coins, interval_minutes=args.interval)
    print(f"  {len(panel.symbols)} 币 x {len(panel)} bar  ({time.time() - t0:.0f}s)")

    panels = window_panels(panel, args.windows)
    print(f"  {len(panels)} 个窗口，每个已 restrict 到各自的非流动半")

    # dry-run broker：订单不出网，费用按 fee_bps 模拟。
    bot_cfg = BotConfig()
    bot_cfg.execution.dry_run = True
    bot_cfg.execution.order_type = "market"
    bot_cfg.execution.fee_bps = args.fee
    tmp = tempfile.mkdtemp(prefix="replay-basket-")
    broker = FeeRecordingBroker(Broker(bot_cfg, None, Journal(f"{tmp}/journal.jsonl")))

    signal = negate(past_return(args.lookback))
    bt_cfg = CrossSectionConfig(
        quantile=args.quantile,
        min_symbols=cfg.min_symbols,
        fee_bps=args.fee,
        late_exit_factor=args.late_exit_factor,
    )

    print(
        f"\n{'窗口':>4} {'币':>4} {'期':>4} | "
        f"{'回测net':>9} {'重放net':>9} {'差':>8} | "
        f"{'回测成本':>8} {'重放成本':>8} | {'回测换手':>8} {'重放换手':>8}"
    )
    print("-" * 86)
    replay_all = ReplayResult()
    bt_all: list[float] = []

    for i, p in enumerate(panels):
        res = replay_panel(
            p,
            broker,
            cfg,
            args.lookback,
            args.hold,
            args.equity,
            args.late_exit_factor,
            args.verbose,
        )
        bt = CrossSectionBacktester(p, signal, hold_rows=args.hold, config=bt_cfg).run()
        bt_net = bt.net_return_pct
        replay_net = res.compounded_pct
        replay_all.periods.extend(res.periods)
        bt_all.append(bt_net)
        print(
            f"{i:>4} {len(p.symbols):>4} {res.n:>4} | "
            f"{bt_net:>+8.2f}% {replay_net:>+8.2f}% {replay_net - bt_net:>+7.2f} | "
            f"{bt.cost_pct:>7.2f}% {res.cost_pct:>7.2f}% | "
            f"{bt.mean_turnover_pct:>7.1f}% {res.mean_turnover_pct:>7.1f}%"
        )

    print("-" * 50)
    print(f"回测合计     : {statistics.fmean(bt_all):+.2f}% (窗口均值)")
    print(f"实盘重放合计 : {replay_all.compounded_pct:+.2f}% (复利)")
    print()
    print(f"实盘重放 {replay_all.summary()}")
    gap = replay_all.compounded_pct - statistics.fmean(bt_all)
    print(f"\n差异 {gap:+.2f} 个百分点")

    imb = [p["imbalance"] for p in replay_all.periods]
    if any(imb):
        print(f"腿不平衡的期数: {sum(1 for x in imb if x)} / {len(imb)}")

    if args.json:
        Path(args.json).write_text(
            json.dumps(replay_all.periods, indent=2, default=str)
        )
        print(f"\n明细写入 {args.json}")


if __name__ == "__main__":
    main()
