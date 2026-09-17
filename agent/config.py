"""Configuration loading and validation.

Mirrors the Chainstack-style YAML-driven bot config, but the strategy kernel is
a multi-factor signal synthesizer instead of a grid.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the bot configuration is invalid."""


# ---------------------------------------------------------------------------
# Weight / threshold presets
# ---------------------------------------------------------------------------

# Ranking windows offered by the Hyperliquid leaderboard.
VALID_LEADERBOARD_WINDOWS = frozenset({"day", "week", "month", "allTime"})

RISK_PRESETS: dict[str, dict[str, Any]] = {
    "conservative": {
        "long_threshold": 60.0,
        "short_threshold": 40.0,
        "max_signals_per_day": 3,
        "cooldown_minutes": 240,
        "leverage": 2,
        "risk_per_trade_pct": 1.0,
    },
    "balanced": {
        "long_threshold": 55.0,
        "short_threshold": 45.0,
        "max_signals_per_day": 5,
        "cooldown_minutes": 120,
        "leverage": 3,
        "risk_per_trade_pct": 2.0,
    },
    "aggressive": {
        "long_threshold": 52.0,
        "short_threshold": 48.0,
        "max_signals_per_day": 8,
        "cooldown_minutes": 60,
        "leverage": 5,
        "risk_per_trade_pct": 3.0,
    },
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Weights:
    """Factor weights. Must be positive; normalised at load time."""

    smart_money: float = 0.40
    technical: float = 0.35
    market: float = 0.25

    def normalized(self) -> "Weights":
        total = self.smart_money + self.technical + self.market
        if total <= 0:
            raise ConfigError("factor weights must sum to a positive number")
        return Weights(
            smart_money=self.smart_money / total,
            technical=self.technical / total,
            market=self.market / total,
        )


@dataclass
class Indicators:
    """Technical indicator parameters."""

    entry_timeframe: str = "1h"
    trend_timeframe: str = "4h"
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    rsi_period: int = 14
    adx_period: int = 14
    ema_fast: int = 21
    ema_slow: int = 55
    atr_period: int = 14
    lookback_candles: int = 300


@dataclass
class Discipline:
    """Fox-style selectivity guardrails.

    The 22-agent experiment showed that trade *count* dominates outcomes:
    agents with >400 trades all lost, agents with <120 trades all won.
    These fields exist to hard-cap activity, not to be tuned upward.
    """

    max_signals_per_day: int = 5
    cooldown_minutes: int = 120
    long_threshold: float = 55.0
    short_threshold: float = 45.0
    require_smart_money_alignment: bool = True
    btc_trend_filter: bool = True
    btc_trend_timeframe: str = "4h"
    min_score_gap: float = 0.0


@dataclass
class Risk:
    """Position and exit management. Exits default to OFF, as in Chainstack."""

    account_allocation_pct: float = 10.0
    leverage: int = 3
    risk_per_trade_pct: float = 2.0
    max_position_usd: float = 0.0  # 0 = derive from account equity
    stop_loss_enabled: bool = False
    stop_loss_pct: float = 3.0
    take_profit_enabled: bool = False
    take_profit_pct: float = 6.0
    # Peak-to-trough equity drawdown that halts new entries. Enforced by the
    # bot against persisted peak equity.
    max_drawdown_pct: float = 15.0
    max_open_positions: int = 1
    # NOTE: there is deliberately no `reduce_only_on_exit` switch. Exits go
    # through the SDK's market_close, which is inherently reduce-only, so such
    # a flag could only ever be decoration.


@dataclass
class Execution:
    """Order execution parameters."""

    order_type: str = "limit"  # "limit" | "market"
    limit_offset_bps: float = 5.0
    slippage_bps: float = 20.0
    max_spread_bps: float = 25.0
    poll_interval_seconds: int = 60
    dry_run: bool = True
    testnet: bool = True
    base_url: str | None = None


@dataclass
class SmartMoneySettings:
    """Smart money source settings.

    Wallet selection samples several leaderboard windows and keeps the wallets
    that persist across them. Ranking by a single window produced an unstable,
    tiny and highly concentrated sample: measured overlap between the week and
    month top-25 sets was only 0.11.
    """

    # Windows to rank by. Valid: day, week, month, allTime.
    windows: list[str] = field(default_factory=lambda: ["day", "week", "month"])
    # How many accounts to take from each window before unioning.
    top_per_window: int = 60
    # Minimum windows a wallet must rank in to be included.
    # 1 = use the whole union and let persistence act purely as a *weight*.
    # Measured on BTC, hard-filtering at 2 cut holders from 18 to 3, because
    # the wallets that persist across windows are large diversified accounts
    # that often hold nothing in a given symbol. Raise this only if you want a
    # smaller, stricter candidate set and accept the coverage loss.
    min_persistence: int = 1
    # Hard cap on the candidate set, which bounds the number of HTTP reads.
    max_wallets: int = 150
    # Accounts below this value are ignored; their positions are noise.
    min_account_value: float = 10_000.0
    # Manual include / exclude lists. Whitelisted wallets always survive the
    # persistence filter, so hand-picked track records are not screened out.
    whitelist: list[str] = field(default_factory=list)
    blacklist: list[str] = field(default_factory=list)


@dataclass
class Safety:
    """Backstops that apply regardless of the risk block's exit switches."""

    hard_stop_pct: float = 5.0


@dataclass
class BotConfig:
    """Top level configuration."""

    name: str = "fox_lite"
    symbol: str = "BTC"
    weights: Weights = field(default_factory=Weights)
    indicators: Indicators = field(default_factory=Indicators)
    discipline: Discipline = field(default_factory=Discipline)
    risk: Risk = field(default_factory=Risk)
    execution: Execution = field(default_factory=Execution)
    safety: Safety = field(default_factory=Safety)
    smart_money: SmartMoneySettings = field(default_factory=SmartMoneySettings)
    min_confidence: float = 0.35
    risk_preset: str | None = None
    use_hyperfeed: bool = False  # set true only if SENPI_API_KEY is present
    raw: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        w = self.weights
        return (
            f"[{self.name}] {self.symbol} | "
            f"weights sm={w.smart_money:.0%} ta={w.technical:.0%} mkt={w.market:.0%} | "
            f"thresholds long>{self.discipline.long_threshold:g} "
            f"short<{self.discipline.short_threshold:g} | "
            f"cap={self.discipline.max_signals_per_day}/day "
            f"cooldown={self.discipline.cooldown_minutes}m | "
            f"lev={self.risk.leverage}x alloc={self.risk.account_allocation_pct:g}% | "
            f"dry_run={self.execution.dry_run} testnet={self.execution.testnet}"
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _build(dataclass_type, data: dict[str, Any] | None, path: str):
    """Instantiate a dataclass from a dict, rejecting unknown keys."""
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError(f"`{path}` must be a mapping, got {type(data).__name__}")
    allowed = set(dataclass_type.__dataclass_fields__)
    # `raw` is internal bookkeeping, not user-facing
    allowed.discard("raw")
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(
            f"unknown key(s) in `{path}`: {', '.join(sorted(unknown))}. "
            f"valid: {', '.join(sorted(allowed))}"
        )
    return dataclass_type(**data)


def load_config(path: str | Path) -> BotConfig:
    """Load, validate and normalise a bot config from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        try:
            data = yaml.safe_load(fh) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"config root must be a mapping in {path}")

    allowed_top = set(BotConfig.__dataclass_fields__)
    allowed_top.discard("raw")
    unknown = set(data) - allowed_top
    if unknown:
        raise ConfigError(
            f"unknown top-level key(s): {', '.join(sorted(unknown))}. "
            f"valid: {', '.join(sorted(allowed_top))}"
        )

    risk_preset = data.pop("risk_preset", None)
    preset: dict[str, Any] = {}
    if risk_preset is not None:
        if risk_preset not in RISK_PRESETS:
            raise ConfigError(
                f"unknown risk_preset `{risk_preset}`; "
                f"choose from {', '.join(RISK_PRESETS)}"
            )
        preset = dict(RISK_PRESETS[risk_preset])

    # Preset supplies defaults; explicit YAML values win.
    discipline_data = dict(data.get("discipline") or {})
    risk_data = dict(data.get("risk") or {})
    for key, value in preset.items():
        if key in Discipline.__dataclass_fields__:
            discipline_data.setdefault(key, value)
        elif key in Risk.__dataclass_fields__:
            risk_data.setdefault(key, value)

    cfg = BotConfig(
        name=data.get("name", "fox_lite"),
        symbol=str(data.get("symbol", "BTC")).upper(),
        weights=_build(Weights, data.get("weights"), "weights"),
        indicators=_build(Indicators, data.get("indicators"), "indicators"),
        discipline=_build(Discipline, discipline_data, "discipline"),
        risk=_build(Risk, risk_data, "risk"),
        execution=_build(Execution, data.get("execution"), "execution"),
        safety=_build(Safety, data.get("safety"), "safety"),
        smart_money=_build(
            SmartMoneySettings, data.get("smart_money"), "smart_money"
        ),
        min_confidence=float(data.get("min_confidence", 0.35)),
        risk_preset=risk_preset,
        use_hyperfeed=bool(data.get("use_hyperfeed", False)),
        raw={**data, "risk_preset": risk_preset},
    )

    _validate(cfg, risk_preset)
    return cfg


def _validate(cfg: BotConfig, risk_preset: str | None) -> None:
    """Reject configurations that are economically or logically unsafe."""

    cfg.weights = cfg.weights.normalized()

    d, r, e = cfg.discipline, cfg.risk, cfg.execution

    if not (0 <= cfg.min_confidence <= 1):
        raise ConfigError("min_confidence must be within [0, 1]")

    sm = cfg.smart_money

    if not sm.windows:
        raise ConfigError("smart_money.windows must not be empty")
    invalid_windows = [w for w in sm.windows if w not in VALID_LEADERBOARD_WINDOWS]
    if invalid_windows:
        raise ConfigError(
            f"unknown smart_money window(s): {', '.join(invalid_windows)}. "
            f"valid: {', '.join(sorted(VALID_LEADERBOARD_WINDOWS))}"
        )
    if len(set(sm.windows)) != len(sm.windows):
        raise ConfigError("smart_money.windows contains duplicates")
    if sm.top_per_window < 5:
        raise ConfigError(
            "smart_money.top_per_window must be >= 5; a smaller sample is noise"
        )
    if sm.min_persistence < 1:
        raise ConfigError("smart_money.min_persistence must be >= 1")
    if sm.min_persistence > len(sm.windows):
        raise ConfigError(
            f"smart_money.min_persistence ({sm.min_persistence}) cannot exceed the "
            f"number of windows ({len(sm.windows)}); no wallet could ever satisfy it"
        )
    if sm.max_wallets < 0:
        raise ConfigError("smart_money.max_wallets must be >= 0 (0 = no cap)")
    if sm.max_wallets and sm.max_wallets < sm.top_per_window:
        raise ConfigError(
            f"smart_money.max_wallets ({sm.max_wallets}) is below top_per_window "
            f"({sm.top_per_window}); the union would be cut below a single "
            "window's contribution"
        )
    if sm.min_account_value < 0:
        raise ConfigError("smart_money.min_account_value must be >= 0")

    for label, addresses in (
        ("whitelist", sm.whitelist),
        ("blacklist", sm.blacklist),
    ):
        for addr in addresses:
            if not (isinstance(addr, str) and addr.startswith("0x") and len(addr) == 42):
                raise ConfigError(
                    f"smart_money.{label} entry `{addr}` is not a 0x-prefixed "
                    "42-character address"
                )

    if set(sm.whitelist) & set(sm.blacklist):
        overlap = sorted(set(sm.whitelist) & set(sm.blacklist))
        raise ConfigError(
            f"address(es) in both smart_money.whitelist and blacklist: {', '.join(overlap)}"
        )
    if not (0 < cfg.safety.hard_stop_pct < 100):
        raise ConfigError("safety.hard_stop_pct must be within (0, 100)")
    if r.stop_loss_enabled and cfg.safety.hard_stop_pct < r.stop_loss_pct:
        raise ConfigError(
            "safety.hard_stop_pct must be >= risk.stop_loss_pct, otherwise the "
            "safety backstop would fire before the configured stop"
        )

    if not (0 < d.long_threshold < 100):
        raise ConfigError("discipline.long_threshold must be within (0, 100)")
    if not (0 < d.short_threshold < 100):
        raise ConfigError("discipline.short_threshold must be within (0, 100)")
    if d.short_threshold >= d.long_threshold:
        raise ConfigError(
            "discipline.short_threshold must be strictly below long_threshold "
            f"(got short={d.short_threshold:g}, long={d.long_threshold:g})"
        )
    if d.max_signals_per_day < 1:
        raise ConfigError("discipline.max_signals_per_day must be >= 1")
    if d.cooldown_minutes < 0:
        raise ConfigError("discipline.cooldown_minutes must be >= 0")

    if not (0 < r.account_allocation_pct <= 100):
        raise ConfigError("risk.account_allocation_pct must be within (0, 100]")
    if r.leverage < 1:
        raise ConfigError("risk.leverage must be >= 1")
    if r.leverage > 10:
        raise ConfigError(
            f"risk.leverage {r.leverage}x refused. The 22-agent experiment showed "
            "self-escalated leverage (up to 25x) accelerated losses. Cap is 10x."
        )
    if not (0 < r.risk_per_trade_pct <= 100):
        raise ConfigError("risk.risk_per_trade_pct must be within (0, 100]")
    if r.max_open_positions < 1:
        raise ConfigError("risk.max_open_positions must be >= 1")
    if r.stop_loss_enabled and not (0 < r.stop_loss_pct < 100):
        raise ConfigError("risk.stop_loss_pct must be within (0, 100)")
    if r.take_profit_enabled and r.take_profit_pct <= 0:
        raise ConfigError("risk.take_profit_pct must be > 0")
    if not (0 < r.max_drawdown_pct <= 100):
        raise ConfigError("risk.max_drawdown_pct must be within (0, 100]")
    if r.take_profit_enabled and r.stop_loss_enabled and r.take_profit_pct < r.stop_loss_pct:
        raise ConfigError(
            f"risk.take_profit_pct ({r.take_profit_pct:g}) is below stop_loss_pct "
            f"({r.stop_loss_pct:g}); the trade would have negative expectancy "
            "before fees"
        )

    if e.order_type not in ("limit", "market"):
        raise ConfigError("execution.order_type must be 'limit' or 'market'")
    if e.poll_interval_seconds < 5:
        raise ConfigError("execution.poll_interval_seconds must be >= 5")

    # Live trading requires an explicit key and is mainnet-gated below.
    if not e.dry_run and not os.getenv("HYPERLIQUID_PRIVATE_KEY"):
        raise ConfigError(
            "execution.dry_run is false but HYPERLIQUID_PRIVATE_KEY is not set. "
            "Export the API wallet key (never the main wallet key) before going live."
        )

    if not e.dry_run and not e.testnet:
        if risk_preset is None:
            raise ConfigError(
                "refusing to trade mainnet without an explicit risk_preset. "
                "Set risk_preset to conservative / balanced / aggressive."
            )
        if r.leverage > 3:
            raise ConfigError(
                f"refusing mainnet leverage of {r.leverage}x. Max 3x for live mainnet."
            )
