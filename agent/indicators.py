"""Pure-python technical indicators.

No numpy/pandas dependency: candle counts here are in the hundreds, so plain
lists are fast enough and keep the install footprint at zero.
"""

from __future__ import annotations

from typing import Sequence


class IndicatorError(ValueError):
    """Raised when there is not enough data to compute an indicator."""


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


def sma(values: Sequence[float], period: int) -> list[float | None]:
    if period < 1:
        raise IndicatorError("period must be >= 1")
    out: list[float | None] = [None] * len(values)
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def ema(values: Sequence[float], period: int) -> list[float | None]:
    """Exponential moving average, seeded with an SMA of the first `period`."""
    if period < 1:
        raise IndicatorError("period must be >= 1")
    n = len(values)
    out: list[float | None] = [None] * n
    if n < period:
        return out

    seed = sum(values[:period]) / period
    out[period - 1] = seed
    k = 2.0 / (period + 1)
    prev = seed
    for i in range(period, n):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def rsi(values: Sequence[float], period: int = 14) -> list[float | None]:
    """Wilder's RSI."""
    n = len(values)
    out: list[float | None] = [None] * n
    if n <= period:
        return out

    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        delta = values[i] - values[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta

    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_value(avg_gain, avg_loss)

    for i in range(period + 1, n):
        delta = values[i] - values[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# ---------------------------------------------------------------------------
# True range family
# ---------------------------------------------------------------------------


def true_range(
    highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]
) -> list[float]:
    n = min(len(highs), len(lows), len(closes))
    out = [0.0] * n
    if n == 0:
        return out
    out[0] = highs[0] - lows[0]
    for i in range(1, n):
        out[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    return out


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> list[float | None]:
    """Wilder's Average True Range."""
    tr = true_range(highs, lows, closes)
    n = len(tr)
    out: list[float | None] = [None] * n
    if n < period:
        return out

    prev = sum(tr[:period]) / period
    out[period - 1] = prev
    for i in range(period, n):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out


# ---------------------------------------------------------------------------
# ADX / directional movement
# ---------------------------------------------------------------------------


def adx(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> list[float | None]:
    """Wilder's ADX."""
    n = min(len(highs), len(lows), len(closes))
    out: list[float | None] = [None] * n
    if n < 2 * period:
        return out

    plus_dm = [0.0] * n
    minus_dm = [0.0] * n
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    tr = true_range(highs, lows, closes)

    # Wilder smoothing over the first window
    tr_s = sum(tr[1 : period + 1])
    plus_s = sum(plus_dm[1 : period + 1])
    minus_s = sum(minus_dm[1 : period + 1])

    dx_values: list[float] = []
    for i in range(period + 1, n):
        tr_s = tr_s - (tr_s / period) + tr[i]
        plus_s = plus_s - (plus_s / period) + plus_dm[i]
        minus_s = minus_s - (minus_s / period) + minus_dm[i]

        if tr_s == 0:
            continue
        plus_di = 100.0 * plus_s / tr_s
        minus_di = 100.0 * minus_s / tr_s
        denom = plus_di + minus_di
        dx_values.append(100.0 * abs(plus_di - minus_di) / denom if denom else 0.0)

    if len(dx_values) < period:
        return out

    prev = sum(dx_values[:period]) / period
    start = period + 1 + (period - 1)
    if start < n:
        out[start] = prev
    for idx in range(start + 1, n):
        dx_idx = idx - (period + 1)
        if dx_idx >= len(dx_values):
            break
        prev = (prev * (period - 1) + dx_values[dx_idx]) / period
        out[idx] = prev
    return out


# ---------------------------------------------------------------------------
# Supertrend
# ---------------------------------------------------------------------------


def supertrend(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 10,
    multiplier: float = 3.0,
) -> tuple[list[float | None], list[str | None]]:
    """Return (supertrend line, direction) where direction is 'buy' or 'sell'."""
    n = min(len(highs), len(lows), len(closes))
    atr_vals = atr(highs, lows, closes, period)

    line: list[float | None] = [None] * n
    direction: list[str | None] = [None] * n

    final_upper = 0.0
    final_lower = 0.0
    prev_dir: str | None = None
    started = False

    for i in range(n):
        a = atr_vals[i]
        if a is None or a == 0:
            continue

        mid = (highs[i] + lows[i]) / 2.0
        upper = mid + multiplier * a
        lower = mid - multiplier * a

        if not started:
            final_upper, final_lower = upper, lower
            prev_dir = "buy" if closes[i] >= mid else "sell"
            started = True
        else:
            final_upper = (
                upper if (upper < final_upper or closes[i - 1] > final_upper) else final_upper
            )
            final_lower = (
                lower if (lower > final_lower or closes[i - 1] < final_lower) else final_lower
            )

            # A close through the band flips the trend. The bands are already
            # ratcheted above; the previous close guards in the band update
            # prevent the line from being pulled back once a trend is running.
            if prev_dir == "buy" and closes[i] < final_lower:
                prev_dir = "sell"
            elif prev_dir == "sell" and closes[i] > final_upper:
                prev_dir = "buy"

        line[i] = final_lower if prev_dir == "buy" else final_upper
        direction[i] = prev_dir

    return line, direction


# ---------------------------------------------------------------------------
# Candle container
# ---------------------------------------------------------------------------


class CandleSeries:
    """OHLCV series with lazily computed indicators."""

    __slots__ = (
        "_opens",
        "_highs",
        "_lows",
        "_closes",
        "_volumes",
        "_times",
        "_atr_period",
        "_cache",
    )

    def __init__(
        self,
        opens: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        volumes: Sequence[float],
        times: Sequence[int],
        atr_period: int = 14,
    ) -> None:
        self._opens = list(opens)
        self._highs = list(highs)
        self._lows = list(lows)
        self._closes = list(closes)
        self._volumes = list(volumes)
        self._times = list(times)
        self._atr_period = atr_period
        self._cache: dict[str, object] = {}

    def __len__(self) -> int:
        return len(self._closes)

    # All six arrays are exposed. Only closes/highs/lows used to be, so the
    # backtest reached into `_opens`, `_times` and `_volumes` directly; a
    # half-public container invites exactly that kind of private access.
    @property
    def opens(self) -> list[float]:
        return self._opens

    @property
    def closes(self) -> list[float]:
        return self._closes

    @property
    def highs(self) -> list[float]:
        return self._highs

    @property
    def lows(self) -> list[float]:
        return self._lows

    @property
    def volumes(self) -> list[float]:
        return self._volumes

    @property
    def times(self) -> list[int]:
        return self._times

    @property
    def last_price(self) -> float:
        if not self._closes:
            raise IndicatorError("empty candle series")
        return self._closes[-1]

    # ------------------------------------------------------------------
    # Serialisation, used by the backtest data cache
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, list]:
        return {
            "o": self._opens,
            "h": self._highs,
            "l": self._lows,
            "c": self._closes,
            "v": self._volumes,
            "t": self._times,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CandleSeries":
        try:
            return cls(
                data["o"], data["h"], data["l"], data["c"], data["v"], data["t"]
            )
        except (KeyError, TypeError) as exc:
            raise IndicatorError(f"malformed candle payload: {exc}") from exc

    def slice(self, start: int, end: int) -> "CandleSeries":
        """A sub-series by index, used to expose only past bars."""
        return CandleSeries(
            self._opens[start:end],
            self._highs[start:end],
            self._lows[start:end],
            self._closes[start:end],
            self._volumes[start:end],
            self._times[start:end],
        )

    @property
    def last_time(self) -> int:
        return self._times[-1] if self._times else 0

    def _cached(self, key: str, factory):
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    def ema(self, period: int) -> list[float | None]:
        return self._cached(f"ema{period}", lambda: ema(self._closes, period))

    def rsi(self, period: int = 14) -> list[float | None]:
        return self._cached(f"rsi{period}", lambda: rsi(self._closes, period))

    def atr(self, period: int | None = None) -> list[float | None]:
        p = period or self._atr_period
        return self._cached(
            f"atr{p}", lambda: atr(self._highs, self._lows, self._closes, p)
        )

    def adx(self, period: int = 14) -> list[float | None]:
        return self._cached(
            f"adx{period}", lambda: adx(self._highs, self._lows, self._closes, period)
        )

    def supertrend(self, period: int = 10, multiplier: float = 3.0):
        return self._cached(
            f"st{period}_{multiplier}",
            lambda: supertrend(
                self._highs, self._lows, self._closes, period, multiplier
            ),
        )
