"""Crash-safe persistence of agent state.

Why this exists: the frequency guardrails (daily signal cap, cooldown, loss
streak halt) are the only part of this agent with evidence behind them, and
they lived purely in memory. A restart silently reset all of them, and also
lost track of the open position, which defeated the max_open_positions check
and left a position with no exit management.

Writes go through a per-writer temp file, are fsynced, and are then renamed
into place, so a crash mid-write cannot leave a corrupt file that would abort
the next startup. Reads are defensive: anything unexpected yields None and the
caller starts from a clean state rather than refusing to boot.

The two failure modes worth naming, because both end with a guardrail that
looks present but never acts:

  * Non-finite floats. `Infinity`/`NaN` compare false against everything, so a
    single one silently disables the guardrail that reads it. They are rejected
    on the way in and refused on the way out.
  * Concurrent writers. A fixed temp filename would let two processes interleave
    into the same file and publish the mixture, which loads as "corrupt" and
    resets every guardrail. Each writer gets its own temp file instead.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STATE_VERSION = 1
DEFAULT_STATE_PATH = "logs/agent_state.json"


@dataclass
class AgentState:
    """Everything that must survive a restart."""

    version: int = STATE_VERSION
    saved_at: float = field(default_factory=time.time)

    # --- Discipline: the validated edge ---------------------------------
    # Resetting these on restart would let the agent trade far more often than
    # the configuration permits, which is precisely the failure mode the
    # 22-agent experiment identified as fatal.
    last_trade_ts: dict[str, float] = field(default_factory=dict)
    trades_today: dict[str, int] = field(default_factory=dict)
    day_anchor: float = 0.0
    consecutive_losses: int = 0
    halted_until: float = 0.0
    halt_reason: str = ""

    # --- Open position ---------------------------------------------------
    # Plain dict so this module has no dependency on the execution layer.
    position: dict[str, Any] | None = None

    # --- Risk accounting -------------------------------------------------
    peak_equity: float = 0.0
    realised_pnl: float = 0.0

    # --- Agent-level halt -------------------------------------------------
    # Persisted on purpose: a drawdown or critical halt must survive a restart,
    # otherwise bouncing the process is an easy way to bypass the limit.
    agent_halted: bool = False
    agent_halt_reason: str = ""

    def describe(self) -> str:
        age_min = (time.time() - self.saved_at) / 60.0
        if self.position:
            pos = (
                f"{self.position.get('side')} {self.position.get('size')} "
                f"{self.position.get('symbol')} @ {self.position.get('entry_price')}"
            )
        else:
            pos = "flat"
        return (
            f"saved {age_min:.1f} min ago | position {pos} | "
            f"trades_today {self.trades_today or '{}'} | "
            f"loss_streak {self.consecutive_losses} | "
            f"peak_equity {self.peak_equity:.2f}"
        )


class StateStore:
    """Atomic JSON persistence for :class:`AgentState`."""

    def __init__(self, path: str | Path | None = DEFAULT_STATE_PATH) -> None:
        self.path = Path(path) if path else None

    # ------------------------------------------------------------------
    def load(self) -> AgentState | None:
        """Read state. Returns None if absent, unreadable or incompatible."""
        if self.path is None or not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        try:
            if int(payload.get("version", 0)) != STATE_VERSION:
                return None
        except (TypeError, ValueError):
            return None

        state = AgentState()
        for name in AgentState.__dataclass_fields__:
            if name in payload:
                setattr(state, name, payload[name])

        # Defensive coercion: a hand-edited or partially written file must not
        # crash the agent with a type error deep in the trading loop.
        if not isinstance(state.last_trade_ts, dict):
            state.last_trade_ts = {}
        if not isinstance(state.trades_today, dict):
            state.trades_today = {}
        if state.position is not None and not isinstance(state.position, dict):
            state.position = None

        state.last_trade_ts = _float_map(state.last_trade_ts)
        state.trades_today = _int_map(state.trades_today)
        state.day_anchor = _as_float(state.day_anchor)
        state.halted_until = _as_float(state.halted_until)
        state.peak_equity = _as_float(state.peak_equity)
        state.realised_pnl = _as_float(state.realised_pnl)
        state.consecutive_losses = int(_as_float(state.consecutive_losses))
        state.halt_reason = str(state.halt_reason or "")
        state.agent_halted = bool(state.agent_halted)
        state.agent_halt_reason = str(state.agent_halt_reason or "")
        return state

    def save(self, state: AgentState) -> bool:
        """Write state atomically. Returns False on failure; never raises."""
        if self.path is None:
            return False
        state.version = STATE_VERSION
        state.saved_at = time.time()

        tmp_path: str | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)

            # A per-writer temp file rather than a fixed `<name>.tmp`. Two
            # processes - a cron run and a manual run, which this project
            # explicitly supports - would otherwise interleave into the same
            # temp file and publish the mixture via `os.replace`.
            fd, tmp_path = tempfile.mkstemp(
                dir=str(self.path.parent),
                prefix=f"{self.path.name}.",
                suffix=".tmp",
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                # `allow_nan=False` refuses to emit `Infinity`/`NaN`. Those are
                # not valid JSON, and a non-finite value read back would make
                # every guardrail comparison silently false. Failing the save
                # keeps the previous (valid) file instead of poisoning the next
                # startup.
                handle.write(
                    json.dumps(asdict(state), default=str, allow_nan=False)
                )
                # Flush to the device before renaming. Without this the rename
                # can be durable while the contents are not, so an unclean
                # shutdown leaves a zero-length or truncated file behind.
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(tmp_path, self.path)
            tmp_path = None
            return True
        except (OSError, ValueError, TypeError):
            # A persistence failure must not take down the trading loop, but it
            # does remove the restart protection, so the caller logs it.
            return False
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def clear(self) -> None:
        if self.path is None:
            return
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------
# Every value read from disk passes through these. They are deliberately total:
# anything unusable falls back rather than raising. They must ALSO reject
# infinities and NaN, because those compare false against everything - a single
# one loaded into `peak_equity`, `halted_until` or `last_trade_ts` would
# silently disable a guardrail for good. `nan >= limit` is False, so the
# drawdown limit never fires; `now < nan` is False, so the loss-streak halt
# never applies; `(now - nan) < cooldown` is False, so the cooldown never
# engages. The guardrail would look present and simply never act.


def _finite(value: Any) -> float | None:
    """Parse `value` as a finite float, or None if it is not one."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_float(value: Any, default: float = 0.0) -> float:
    number = _finite(value)
    return default if number is None else number


def _float_map(data: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in data.items():
        number = _finite(value)
        if number is not None:
            out[str(key)] = number
    return out


def _int_map(data: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in data.items():
        number = _finite(value)
        if number is not None:
            out[str(key)] = int(number)
    return out
