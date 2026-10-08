"""再平衡：把当前持仓变成目标篮子。

三个决定都写在这里，因为它们不是实现细节而是风险选择：

1. **先平后开。** 方向反了的币不能同时持有多空两边 —— 白占两份保证金，还多
   付一遍费。所以整轮的平仓全部先做完，再统一开仓。代价是中间有几十秒不是
   市场中性（旧的平了、新的没开全），但那比同时双向持仓便宜。

2. **单笔失败不终止整轮。** 34 笔里挂几笔不该让这一期完全没有篮子 —— 缺两个
   币只是信号略弱，而放弃整期等于这一期完全不参与。

3. **但最后要把两腿削平（`enforce_balance`）。** 这是 2 的另一半，也是最容易
   被漏掉的一半：策略的 edge 来自横截面反转，多空是对冲掉的。腿一旦不平衡，
   剩下的就是净方向敞口，而市场方向是这个策略**没有 edge 的维度**。宁可这一
   期小一点，也不带着一个没有 edge 的敞口过夜。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from agent.basket import Basket, BasketConfig
from agent.execution import Broker, Position
from agent.synthesizer import LONG, SHORT


@dataclass(frozen=True)
class LegOrder:
    """一条腿的一笔开仓。"""

    symbol: str
    side: str
    size: float
    price: float
    notional: float
    #: 名义被参与率上限削过。True 意味着这笔没有达到 `per_coin_usd`。
    capped: bool = False


@dataclass
class RebalancePlan:
    opens: list[LegOrder] = field(default_factory=list)
    closes: list[Position] = field(default_factory=list)
    #: 方向没变、继续持有的。它们不产生任何成本，这正是换手率要算的。
    holds: list[Position] = field(default_factory=list)

    @property
    def n_trades(self) -> int:
        return len(self.opens) + len(self.closes)

    @property
    def turnover(self) -> float:
        """两腿合计的换手率，与回测 `_turnover` 同口径（对新腿规模取比）。"""
        total = len(self.opens) + len(self.holds)
        if not total:
            return 0.0
        return len(self.opens) / total

    def describe(self) -> str:
        capped = sum(1 for o in self.opens if o.capped)
        return (
            f"{len(self.opens)} open / {len(self.closes)} close / "
            f"{len(self.holds)} hold  turnover {self.turnover:.0%}"
            + (f"  ({capped} capped by participation)" if capped else "")
        )


@dataclass
class RebalanceOutcome:
    #: 最终持仓，含持有不动的。键是 symbol。
    positions: dict[str, Position] = field(default_factory=dict)
    closed: list[str] = field(default_factory=list)
    failed_opens: list[tuple[str, str]] = field(default_factory=list)
    failed_closes: list[tuple[str, str]] = field(default_factory=list)
    #: 因削平两腿而被平掉的。
    truncated: list[str] = field(default_factory=list)
    #: 开仓成功的顺序，削平时从后往前平（最后开的最先走）。
    open_order: list[str] = field(default_factory=list)

    @property
    def n_long(self) -> int:
        return sum(1 for p in self.positions.values() if p.side == LONG)

    @property
    def n_short(self) -> int:
        return sum(1 for p in self.positions.values() if p.side == SHORT)

    @property
    def imbalance(self) -> int:
        """正=多头腿多，负=空头腿多。0 才是市场中性。"""
        return self.n_long - self.n_short

    @property
    def gross_notional(self) -> float:
        return sum(p.notional for p in self.positions.values())

    def describe(self) -> str:
        return (
            f"{self.n_long} long / {self.n_short} short "
            f"(imbalance {self.imbalance:+d})  gross ${self.gross_notional:,.0f}  "
            f"failed {len(self.failed_opens)} open / {len(self.failed_closes)} close"
            + (f"  truncated {len(self.truncated)}" if self.truncated else "")
        )


# ---------------------------------------------------------------------------
# 计划
# ---------------------------------------------------------------------------
def leg_notional(
    price: float, day_volume: float, cfg: BasketConfig
) -> tuple[float, bool]:
    """这一笔该下多少名义，以及有没有被参与率上限削过。

    容量那道坎的护栏：单笔不超过该币当日成交额的 `max_participation`。被削的
    笔会记在 `LegOrder.capped` 上，因为「这一期实际上没跑到目标规模」是必须能
    看见的事 —— 否则实测与回测的差距会被误读成策略失效。
    """
    want = cfg.per_coin_usd
    if day_volume and day_volume > 0:
        cap = cfg.max_participation * day_volume
        if want > cap:
            return cap, True
    return want, False


def plan_rebalance(
    current: dict[str, Position],
    target: Basket,
    prices: dict[str, float] | None = None,
    day_volumes: dict[str, float] | None = None,
    cfg: BasketConfig | None = None,
) -> RebalancePlan:
    """算出「要开哪些、要平哪些、哪些不动」。

    `prices` 缺省用篮子构建时的价格。拿不到价格的币直接跳过：宁可这一期少一个
    币，也不能按猜测的价格下一个错的单。
    """
    cfg = cfg or BasketConfig()
    prices = prices or {}
    day_volumes = day_volumes or {}
    plan = RebalancePlan()

    for s in sorted(target.symbols):
        want = target.side_of(s)
        if want is None:
            continue
        price = prices.get(s) or target.prices.get(s)
        if price is None or price <= 0:
            continue

        have = current.get(s)
        if have is not None and have.side != want:
            # 方向反了：先平掉，再按新方向开。不能同时持两边。
            plan.closes.append(have)
            have = None

        if have is None:
            notional, capped = leg_notional(price, day_volumes.get(s, 0.0), cfg)
            size = notional / price
            if size <= 0:
                continue
            plan.opens.append(
                LegOrder(s, want, size, price, notional, capped)
            )
        else:
            plan.holds.append(have)

    # 掉出新篮子的旧持仓，全部平掉。
    for s, pos in current.items():
        if s not in target.symbols:
            plan.closes.append(pos)

    return plan


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------
def execute_rebalance(
    broker: Broker,
    plan: RebalancePlan,
    prices: dict[str, float] | None = None,
    leverage: int = 1,
    on_event: Callable[[str, str], None] | None = None,
    pause_sec: float = 0.0,
) -> RebalanceOutcome:
    """按计划下单，然后削平两腿。

    失败是逐笔收集的，不是抛出去的：调用方需要知道「这一期建成了什么」，
    而不是在第一笔失败时就失去全部上下文。
    """
    prices = prices or {}
    outcome = RebalanceOutcome()
    emit = on_event or (lambda kind, msg: None)

    # --- 1. 先平后开（见模块 docstring）。
    for pos in plan.closes:
        price = prices.get(pos.symbol) or pos.entry_price
        try:
            broker.close_position(pos, price, "rebalance")
            outcome.closed.append(pos.symbol)
            emit("closed", f"{pos.symbol} {pos.side}")
        except Exception as exc:  # noqa: BLE001 - 见 docstring 第 2 点
            outcome.failed_closes.append((pos.symbol, f"{type(exc).__name__}: {exc}"))
            # 没平掉就还在手里，必须留在 positions 里，否则会漏掉这笔敞口。
            outcome.positions[pos.symbol] = pos
            emit("close_failed", f"{pos.symbol}: {exc}")

    # --- 2. 开新腿。
    for i, order in enumerate(plan.opens):
        if i and pause_sec > 0:
            time.sleep(pause_sec)
        try:
            pos = broker.open_position(
                symbol=order.symbol,
                side=order.side,
                size=order.size,
                price=order.price,
                leverage=leverage,
                entry_reasons=["cross-section reversal"],
            )
        except Exception as exc:  # noqa: BLE001 - 见 docstring 第 2 点
            outcome.failed_opens.append(
                (order.symbol, f"{type(exc).__name__}: {exc}")
            )
            emit("open_failed", f"{order.symbol}: {exc}")
            continue
        outcome.positions[pos.symbol] = pos
        outcome.open_order.append(pos.symbol)
        emit("opened", f"{pos.symbol} {order.side} ${order.notional:,.0f}")

    # --- 3. 持有不动的。
    for pos in plan.holds:
        outcome.positions[pos.symbol] = pos

    # --- 4. 削平两腿。
    enforce_balance(broker, outcome, prices, on_event=emit)
    return outcome


def enforce_balance(
    broker: Broker,
    outcome: RebalanceOutcome,
    prices: dict[str, float] | None = None,
    on_event: Callable[[str, str], None] | None = None,
) -> None:
    """把多的那一腿削到和少的一样长。

    为什么削而不是带着不平衡跑：见模块 docstring 第 3 点。这里从**最后开成的**
    开始平，因为先开成的那批是这一期信号最强的（虽然没有刻意排序），而且先平的
    话要连着多付一遍往返成本。
    """
    prices = prices or {}
    emit = on_event or (lambda kind, msg: None)
    imbalance = outcome.imbalance
    if imbalance == 0:
        return

    surplus_side = LONG if imbalance > 0 else SHORT
    # 只考虑这一期新开的：持有不动的是上一期就建好的，动它们的成本更高。
    candidates = [
        s for s in reversed(outcome.open_order)
        if s in outcome.positions and outcome.positions[s].side == surplus_side
    ]
    for s in candidates:
        if outcome.imbalance == 0:
            break
        pos = outcome.positions[s]
        price = prices.get(s) or pos.entry_price
        try:
            broker.close_position(pos, price, "leg-balance")
        except Exception as exc:  # noqa: BLE001
            emit("truncate_failed", f"{s}: {exc}")
            continue
        del outcome.positions[s]
        outcome.truncated.append(s)
        emit("truncated", f"{s} {surplus_side}")
