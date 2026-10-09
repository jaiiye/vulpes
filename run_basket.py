#!/usr/bin/env python3
"""篮子主循环：构建 → 风控 → 再平衡 → 持久化。

一次运行 = 一期，由 cron 每 `hold` 个 bar 调一次（hold=18、4h bar 即每 3 天）。

为什么「一次运行 = 一期」而不是常驻进程：篮子**没有需要盯盘的东西** —— 每条腿
的盈亏靠另一条腿对冲掉，期间没有任何需要响应的事件（这也是篮子不给单个币挂
止损的原因：挂了就是把对冲结构拆掉）。常驻除了多出一个「重启后失忆」的风险面
之外没有收益，而失忆的后果是下一期把已有持仓当成新仓再开一遍、敞口翻倍。

两个安全设计：

  * **不加 `--live` 就是 dry-run。** 真下单必须显式打开，因为这一跑是 34 笔。
  * **同一期不会重复建仓。** 持久化里记了 `basket_ts_ms`，与本次信号的那一刻
    相同就跳过。重复运行的来源很多（cron 重跑、上次没跑完又来一次、手工补跑），
    而重复建仓的后果是敞口翻倍 —— 且日志上看只是「正常开仓」。

用法：
    DRY_RUN_EQUITY_USD=200000 python3 run_basket.py           # dry-run
    python3 run_basket.py --live                               # 真下单
"""

from __future__ import annotations

import argparse
import sys
import time

from agent.basket import (
    DEFAULT_HOLD_OPTIMIZED,
    DEFAULT_LOOKBACK,
    BasketConfig,
    build_basket_live,
)
from agent.basket_risk import check_plan, plan_gross_notional
from agent.basket_state import (
    DEFAULT_BASKET_STATE_PATH,
    BasketStore,
    state_from_positions,
)
from agent.config import BotConfig
from agent.execution import Broker
from agent.market_data import HyperliquidMarket
from agent.rebalance import execute_rebalance, plan_rebalance
from agent.synthesizer import Journal


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--live", action="store_true", help="真下单（默认 dry-run）")
    ap.add_argument("--testnet", action="store_true")
    ap.add_argument("--state", default=str(DEFAULT_BASKET_STATE_PATH))
    ap.add_argument("--journal", default="logs/journal.jsonl")
    ap.add_argument(
        "--equity",
        type=float,
        default=0.0,
        help="0 = 读账户（dry-run 读 DRY_RUN_EQUITY_USD）",
    )
    ap.add_argument("--hold", type=int, default=DEFAULT_HOLD_OPTIMIZED)
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    ap.add_argument("--quantile", type=float, default=0.2)
    ap.add_argument("--per-coin", type=float, default=2000.0)
    ap.add_argument("--max-participation", type=float, default=0.05)
    ap.add_argument(
        "--max-gross-pct",
        type=float,
        default=0.0,
        help="毛敞口占权益的百分比上限，0 = 不限",
    )
    ap.add_argument("--leverage", type=int, default=1)
    ap.add_argument("--pause-sec", type=float, default=0.0, help="每笔下单之间的间隔")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = BasketConfig(
        lookback=args.lookback,
        hold=args.hold,
        quantile=args.quantile,
        per_coin_usd=args.per_coin,
        max_participation=args.max_participation,
        max_gross_pct_of_equity=args.max_gross_pct,
    )

    mode = "LIVE" if args.live else "DRY-RUN"
    print(f"=== 篮子 {mode} ===  {cfg.describe()}")

    market = HyperliquidMarket(testnet=args.testnet)
    bot_cfg = BotConfig()
    bot_cfg.execution.dry_run = not args.live
    # taker：成本结论（§二.28/§二.29）是走盘 + taker 费率口径，maker 挂单那套
    # 「等待成交、到期取消剩余」会让腿间不平衡。
    bot_cfg.execution.order_type = "market"
    broker = Broker(bot_cfg, market, Journal(args.journal))
    store = BasketStore(args.state)

    # --- 1. 恢复上一期。忘了会把已有持仓当成新仓再开一遍。
    prev = store.load()
    held = prev.positions_as_objects() if prev else {}
    if prev:
        dropped = prev.n_positions - len(held)
        print(f"恢复持仓: {prev.describe()}" + (f" ({dropped} 笔记录损坏)" if dropped else ""))

    # --- 2. 权益。读不到就不要建仓 —— 按猜测的权益算出来的规模不是规模。
    equity = args.equity if args.equity > 0 else broker.account_equity()
    print(f"权益 ${equity:,.2f}")

    # --- 3. 构建篮子
    t0 = time.time()
    skipped: list[tuple[str, str]] = []
    basket, pool = build_basket_live(
        market, cfg, on_skip=lambda s, w: skipped.append((s, w))
    )
    print(f"池 {pool}  构建 {time.time() - t0:.0f}s  跳过 {len(skipped)}")
    if args.verbose:
        for s, w in skipped[:5]:
            print(f"  跳过 {s}: {w[:80]}")
    if basket is None:
        print("无篮子（池不足或信号算不出来）—— 这一期不动，已有持仓继续持有")
        return 1

    print(basket.describe())

    # --- 4. 幂等：同一期不重复建仓
    if prev and prev.basket_ts_ms == basket.ts_ms:
        print(
            f"这一期（ts={basket.ts_ms}）已经建过了，跳过 —— 重复建仓会让敞口翻倍"
        )
        return 0

    # --- 5. 计划与风控
    prices = basket.prices
    day_vols = {c.name: c.day_volume for c in market.asset_contexts()}
    plan = plan_rebalance(held, basket, prices, day_vols, cfg)
    print(f"计划: {plan.describe()}")

    gross = plan_gross_notional(plan)
    if equity > 0:
        print(f"毛敞口 ${gross:,.0f} = 权益的 {gross / equity * 100:.1f}%")
    capped = sum(1 for o in plan.opens if o.capped)
    if capped:
        print(f"  {capped} 笔被参与率上限削过（实际没跑到目标规模）")

    reason = check_plan(plan, equity, cfg)
    if reason:
        print(f"**风控拒绝**: {reason}")
        print("这一期不下单，已有持仓保持不变")
        return 1

    # --- 6. 执行
    def on_event(kind: str, msg: str) -> None:
        if args.verbose or kind.endswith("_failed"):
            print(f"  [{kind}] {msg}")

    t1 = time.time()
    outcome = execute_rebalance(
        broker,
        plan,
        prices,
        leverage=args.leverage,
        on_event=on_event,
        pause_sec=args.pause_sec,
    )
    print(f"执行 {time.time() - t1:.0f}s: {outcome.describe()}")

    # --- 7. 持久化。写失败必须说出来：那一瞬间重启保护是没有的。
    if store.save(state_from_positions(outcome.positions, basket.ts_ms)):
        print(f"已保存 {len(outcome.positions)} 笔到 {args.state}")
    else:
        print(f"**持久化失败** —— 下一期将无法恢复这 {len(outcome.positions)} 笔持仓")

    if outcome.imbalance != 0:
        print(f"**警告**: 腿不平衡 {outcome.imbalance:+d}，这一期带有净方向敞口")
    return 0


if __name__ == "__main__":
    sys.exit(main())
