"""篮子持仓的持久化。

为什么必须有：`Broker.live_positions()` 在 dry-run 下返回 []（没有交易所真相），
而 live 下重启之后同样需要知道手上有什么。忘了会怎样 —— 下一期会把 34 笔已开的
仓当成新仓再开一遍，敞口翻倍，而在日志上看只是「正常开仓」。这不是理论风险，
它是「重启即失忆」这类缺陷的标准形态。

写路径复用 `state.atomic_write_json`，不另写一份：那里的每一个属性（per-writer
temp、fsync 早于 rename、拒绝非有限值）都是踩出来的，复制一份就会在某一次修改
中丢掉其中一条。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.execution import Position
from agent.state import atomic_write_json

BASKET_STATE_VERSION = 1
DEFAULT_BASKET_STATE_PATH = "logs/basket_state.json"


@dataclass
class BasketState:
    """一个篮子期的持仓，需要活过重启的东西。"""

    version: int = BASKET_STATE_VERSION
    saved_at: float = 0.0
    #: 篮子对应的信号时间戳。用来判断「这一期是否已经建过了」，这样重启后不会
    #: 在同一个 4h 窗口里再建一次。
    basket_ts_ms: int = 0
    #: symbol -> `Position.to_dict()`
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    longs: list[str] = field(default_factory=list)
    shorts: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    def positions_as_objects(self) -> dict[str, Position]:
        """还原成 `Position`，跳过坏掉的单条记录。

        一笔坏数据不该让整个篮子起不来 —— 但也不能静默丢掉，所以调用方要拿
        返回值和 `positions` 的长度对一下。
        """
        out: dict[str, Position] = {}
        for s, d in (self.positions or {}).items():
            if not isinstance(d, dict):
                continue
            try:
                out[str(s)] = Position.from_dict(d)
            except Exception:  # noqa: BLE001 - 单条损坏不该带崩启动
                continue
        return out

    @property
    def n_positions(self) -> int:
        return len(self.positions or {})

    def is_empty(self) -> bool:
        return not self.positions

    def describe(self) -> str:
        if not self.saved_at:
            return "no basket state"
        age_h = (time.time() - self.saved_at) / 3600.0
        return (
            f"{self.n_positions} positions "
            f"({len(self.longs)} long / {len(self.shorts)} short) "
            f"saved {age_h:.1f}h ago"
        )


class BasketStore:
    """`BasketState` 的原子 JSON 持久化。"""

    def __init__(self, path: str | Path | None = DEFAULT_BASKET_STATE_PATH) -> None:
        self.path = Path(path) if path else None

    # ------------------------------------------------------------------
    def load(self) -> BasketState | None:
        """读状态。缺失、不可读、版本不符都返回 None（当作没有）。"""
        if self.path is None or not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        try:
            if int(payload.get("version", 0)) != BASKET_STATE_VERSION:
                return None
        except (TypeError, ValueError):
            return None

        state = BasketState()
        for name in BasketState.__dataclass_fields__:
            if name in payload:
                setattr(state, name, payload[name])

        # 防御性强制：手改过或写坏的文件不能让交易循环在深处抛类型错误。
        if not isinstance(state.positions, dict):
            state.positions = {}
        if not isinstance(state.longs, list):
            state.longs = []
        if not isinstance(state.shorts, list):
            state.shorts = []
        try:
            state.basket_ts_ms = int(state.basket_ts_ms)
        except (TypeError, ValueError):
            state.basket_ts_ms = 0
        return state

    def save(self, state: BasketState) -> bool:
        if self.path is None:
            return False
        state.version = BASKET_STATE_VERSION
        state.saved_at = time.time()
        return atomic_write_json(self.path, {
            "version": state.version,
            "saved_at": state.saved_at,
            "basket_ts_ms": state.basket_ts_ms,
            "positions": state.positions,
            "longs": list(state.longs),
            "shorts": list(state.shorts),
        })

    def clear(self) -> None:
        if self.path is None:
            return
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def state_from_positions(
    positions: dict[str, Position], basket_ts_ms: int = 0
) -> BasketState:
    """从当前持仓构造一个可持久化的状态。"""
    from agent.synthesizer import LONG

    return BasketState(
        basket_ts_ms=basket_ts_ms,
        positions={s: p.to_dict() for s, p in positions.items()},
        longs=sorted(s for s, p in positions.items() if p.side == LONG),
        shorts=sorted(s for s, p in positions.items() if p.side != LONG),
    )
