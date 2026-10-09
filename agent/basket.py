"""篮子构建：选池、横截面排序、两条腿。

这里刻意不实现选币逻辑。选币在 `backtest.cross_section.rank_legs`，回测和实盘
共用同一段代码；信号是 `negate(past_return(lookback))`，也就是
`cross_section_cost.py` 跑出净收益的那一个。实盘只是「当下这一行」，其余全部复用。

理由记录在 RESEARCH.md §二.30/§二.31：如果实盘自己实现一遍排序，那么当实测与
回测不一致时，无法归因于是策略失效还是两份实现漂了。

参数与回测对齐（cross_section_cost.py 的默认值），改任何一个都意味着跑的不是
回测里那个策略：

    4h bar | lookback 6 (=24h) | hold 6 (=24h) | quantile 0.2 | 反转
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from agent.market_data import HyperliquidMarket
from backtest.cross_section import (
    Panel,
    build_panel,
    negate,
    past_return,
    rank_legs,
)

#: 与 `cross_section_cost.DEFAULT_INTERVAL` 一致。
INTERVAL_MIN = 240
INTERVAL_STR = "4h"
#: 与 `cross_section_cost.DEFAULT_LOOKBACK` / `DEFAULT_HOLD` 一致，单位是 bar。
DEFAULT_LOOKBACK = 6
DEFAULT_HOLD = 6

#: 实盘真正用的持有期（§二.33 实测优化的结论）。**刻意不等于** DEFAULT_HOLD：
#: 6 是「对齐 IC 尺度」选的（IC 在 24h 上测），而最优持有是 72h —— 成本按交易
#: 次数收、毛收益按持有时间长，所以成本一进来，对齐 IC 的尺度就不再是最优尺度。
#: 实测：6 -> 年化 -5.3%，18 -> +55.7%，机制是每年交易次数从 365 降到 122。
DEFAULT_HOLD_OPTIMIZED = 18

#: 拉 K 线时每个币之间的间隔，理由见 `live_panel`。
DEFAULT_PAUSE_SEC = 0.1


@dataclass
class BasketConfig:
    """篮子参数。默认值即回测跑过的那一组。"""

    lookback: int = DEFAULT_LOOKBACK
    #: 调仓间隔。默认取优化后的 72h，不是回测对齐 IC 的 24h，理由见
    #: `DEFAULT_HOLD_OPTIMIZED`。
    hold: int = DEFAULT_HOLD_OPTIMIZED
    quantile: float = 0.2
    min_symbols: int = 20
    #: 取流动性最差的这一半作为池。效应只在这一半里被测到。
    illiquid_fraction: float = 0.5
    #: 每个币的名义敞口。
    per_coin_usd: float = 2000.0
    #: 单笔名义不超过该币当日成交额的这个比例。容量那道坎的护栏。
    max_participation: float = 0.05
    #: 拉 K 线时留几根余量：信号要 lookback 根，再加当期一根。
    candle_margin: int = 2

    # --- 篮子级风控。0 = 不限。见 `basket_risk.check_plan`。 ---
    #: 建仓后毛名义的绝对上限。
    max_gross_notional: float = 0.0
    #: 毛名义相对账户权益的百分比上限。34 笔一起建，一次建错不是一笔止损的事。
    max_gross_pct_of_equity: float = 0.0
    #: 每条腿最多几个币。
    max_legs_per_side: int = 0
    #: 单笔名义低于这个就不下 —— venue 有最小 size，太小的单还占一个腿的名额。
    min_leg_notional: float = 10.0

    @property
    def rows_needed(self) -> int:
        return self.lookback + self.candle_margin

    @property
    def interval_hours(self) -> float:
        return INTERVAL_MIN / 60.0

    def describe(self) -> str:
        return (
            f"lookback {self.lookback * self.interval_hours:.0f}h "
            f"hold {self.hold * self.interval_hours:.0f}h "
            f"q{self.quantile} illiquid {self.illiquid_fraction:.0%} "
            f"${self.per_coin_usd:,.0f}/coin"
        )


@dataclass
class Basket:
    """一期篮子的两条腿。

    `longs` 是信号值最高的一尾。因为信号是 negated momentum，所以它其实是
    「过去跌得最多」的那一批 —— 即反转做多的一端，不是追涨。
    """

    longs: set[str]
    shorts: set[str]
    ts_ms: int = 0
    #: 构建时的实际价格，用于下单与后续盯市。缺失的币不在篮子里。
    prices: dict[str, float] = field(default_factory=dict)

    @property
    def symbols(self) -> set[str]:
        return self.longs | self.shorts

    @property
    def n_legs(self) -> int:
        return len(self.longs) + len(self.shorts)

    def side_of(self, symbol: str) -> str | None:
        if symbol in self.longs:
            return "long"
        if symbol in self.shorts:
            return "short"
        return None

    def describe(self) -> str:
        return (
            f"{len(self.longs)} long / {len(self.shorts)} short "
            f"@ {time.strftime('%Y-%m-%d %H:%M', time.gmtime(self.ts_ms / 1000))} UTC"
        )


# ---------------------------------------------------------------------------
# 选池
# ---------------------------------------------------------------------------
def illiquid_half(
    market: HyperliquidMarket, fraction: float = 0.5
) -> set[str]:
    """当日的非流动半（按 `dayNtlVlm` 排）。

    Step 0 量过（RESEARCH.md §二.31）：`day_volume` 的排名与回测用的「4h 美元
    成交量中位数」Spearman +0.997，分出来的非流动半重叠 100%。所以直接用
    `dayNtlVlm` 就是回测口径，不需要任何换算。

    分档边界与 `liquid_split` 逐字一致（`v < cut`，cut 取 `1-fraction` 分位），
    这样两半的定义只有一处。
    """
    vols: dict[str, float] = {}
    for c in market.asset_contexts():
        if c.day_volume > 0:
            vols[c.name] = c.day_volume
    if len(vols) < 2:
        return set()
    ordered = sorted(vols.values())
    cut = ordered[int(len(ordered) * (1.0 - fraction))]
    return {s for s, v in vols.items() if v < cut}


# ---------------------------------------------------------------------------
# 实盘面板
# ---------------------------------------------------------------------------
def live_panel(
    market: HyperliquidMarket,
    symbols: Iterable[str],
    interval: str = INTERVAL_STR,
    rows: int = 8,
    pause_sec: float = DEFAULT_PAUSE_SEC,
    on_skip=None,
) -> Panel:
    """把实盘 K 线组装成回测那个 `Panel`。

    单个币拉不到就跳过，不让一个币的失败带走整个池 —— 池少一两个币只是信号
    略弱，而整体失败会让这一期完全没有篮子。

    `pause_sec` 是每个币之间的间隔。它必须存在：一次建仓要拉近 90 个币的 K 线，
    串着发会撞上 429，而丢掉的币不是随机丢的（谁先撞上限速谁先没），池会因此
    偏斜。实测（未节流）89 个币里丢了 10 个。约 +9s 的代价换一个不偏的池。

    `rows` 至少要有 `lookback + 1` 根，否则信号在最后一行算不出来（返回 None），
    篮子会静默地变小。
    """
    out: list[dict] = []
    for i, s in enumerate(sorted(symbols)):
        if i and pause_sec > 0:
            time.sleep(pause_sec)
        try:
            cs = market.candles(s, interval=interval, lookback=rows)
        except Exception as exc:  # noqa: BLE001 - 单币失败不该带崩全池
            if on_skip is not None:
                on_skip(s, f"{type(exc).__name__}: {exc}")
            continue
        for t, c, v in zip(cs.times, cs.closes, cs.volumes):
            out.append({"coin": s, "t": int(t), "c": float(c), "v": float(v)})
    if not out:
        raise ValueError("no candles returned for any symbol in the pool")
    return build_panel(out)


# ---------------------------------------------------------------------------
# 构建
# ---------------------------------------------------------------------------
def build_basket(
    panel: Panel, cfg: BasketConfig | None = None
) -> Basket | None:
    """在面板的最后一行排名取两尾。

    返回 None 表示这一期不该交易（可排名的币少于 `min_symbols`）—— 调用方必须
    处理 None，而不是拿一个空篮子去下单。
    """
    cfg = cfg or BasketConfig()
    row = len(panel) - 1
    signal = negate(past_return(cfg.lookback))
    legs = rank_legs(
        panel, row, signal, quantile=cfg.quantile, min_symbols=cfg.min_symbols
    )
    if legs is None:
        return None
    high, low = legs
    prices = {s: p for s in high | low if (p := panel.price(s, row)) is not None}
    return Basket(
        longs=high, shorts=low, ts_ms=panel.times[row], prices=prices
    )


def build_basket_live(
    market: HyperliquidMarket,
    cfg: BasketConfig | None = None,
    pool: set[str] | None = None,
    on_skip=None,
    pause_sec: float = DEFAULT_PAUSE_SEC,
) -> tuple[Basket | None, int]:
    """一步到位：选池 → 拉 K 线 → 排名取两尾。

    返回 (篮子, 池的大小)。篮子为 None 时不要下单。
    """
    cfg = cfg or BasketConfig()
    symbols = pool if pool is not None else illiquid_half(market, cfg.illiquid_fraction)
    if not symbols:
        return None, 0
    panel = live_panel(
        market,
        symbols,
        rows=cfg.rows_needed,
        pause_sec=pause_sec,
        on_skip=on_skip,
    )
    return build_basket(panel, cfg), len(symbols)
