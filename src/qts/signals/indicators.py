"""Technical indicators as pure stateless functions on numpy arrays.

All functions return NaN at positions where insufficient data exists
(warm-up period). Functions are stateless and have no side effects.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _ema(data: NDArray[np.float64], period: int) -> NDArray[np.float64]:
    """Compute Exponential Moving Average (EMA) for a 1-D price array.

    Formula:
        alpha = 2 / (period + 1)
        EMA_t = alpha * price_t + (1 - alpha) * EMA_{t-1}
        Seeded with SMA of first `period` values.

    Output range: Same units as input.
    Staleness: First (period - 1) values are NaN (insufficient data).

    Args:
        data: 1-D array of float64 values.
        period: Look-back window for EMA calculation.

    Returns:
        1-D array of EMA values, same length as input.
    """
    n = len(data)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out

    alpha = 2.0 / (period + 1)
    # Seed with SMA of first `period` values
    seed = float(np.mean(data[:period]))
    out[period - 1] = seed
    for i in range(period, n):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


def compute_rsi(prices: NDArray[np.float64], period: int = 14) -> NDArray[np.float64]:
    """Compute the Relative Strength Index (RSI).

    Formula:
        delta   = price[t] - price[t-1]
        gain    = max(delta, 0)
        loss    = max(-delta, 0)
        avg_gain = EMA(gain, period)   (Wilder's smoothing: alpha = 1/period)
        avg_loss = EMA(loss, period)
        RS      = avg_gain / avg_loss
        RSI     = 100 - 100 / (1 + RS)

    Output range: [0, 100] where 0 = extreme oversold, 100 = extreme overbought.
    Staleness: First `period` values are NaN.

    Args:
        prices: 1-D array of close prices.
        period: Look-back period (default 14).

    Returns:
        1-D array of RSI values in [0, 100], NaN for warm-up bars.
    """
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return out

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Wilder smoothing: alpha = 1/period
    alpha = 1.0 / period

    # Seed average gain/loss using SMA of first `period` changes
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    # Compute RSI starting at index `period` (prices[period])
    for i in range(period, len(deltas)):
        avg_gain = alpha * gains[i] + (1.0 - alpha) * avg_gain
        avg_loss = alpha * losses[i] + (1.0 - alpha) * avg_loss
        if avg_loss == 0.0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    # Fill the initial RSI at index period using seed averages
    if avg_loss == 0.0:
        out[period] = 100.0
    else:
        # Recalculate seed RSI from initial averages
        seed_gain = float(np.mean(gains[:period]))
        seed_loss = float(np.mean(losses[:period]))
        if seed_loss == 0.0:
            out[period] = 100.0
        else:
            rs_seed = seed_gain / seed_loss
            out[period] = 100.0 - 100.0 / (1.0 + rs_seed)

    return out


def compute_macd(
    prices: NDArray[np.float64],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute Moving Average Convergence/Divergence (MACD).

    Formula:
        MACD      = EMA(prices, fast) - EMA(prices, slow)
        Signal    = EMA(MACD, signal)
        Histogram = MACD - Signal

    Output range: Unbounded; same units as price differences.
    Staleness: First (slow - 1) values of MACD are NaN; first
               (slow + signal - 2) values of signal/histogram are NaN.

    Args:
        prices: 1-D array of close prices.
        fast: Fast EMA period (default 12).
        slow: Slow EMA period (default 26).
        signal: Signal EMA period (default 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram), each a 1-D float64 array.
    """
    ema_fast = _ema(prices, fast)
    ema_slow = _ema(prices, slow)

    macd_line = ema_fast - ema_slow
    # Mask where either EMA is NaN
    macd_line = np.where(np.isnan(ema_fast) | np.isnan(ema_slow), np.nan, macd_line)

    signal_line = _ema(macd_line[~np.isnan(macd_line)], signal)
    # Re-expand signal line to full length
    signal_full = np.full(len(prices), np.nan, dtype=np.float64)
    valid_mask = ~np.isnan(macd_line)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) >= signal:
        signal_full[valid_indices] = signal_line

    histogram = np.where(
        np.isnan(macd_line) | np.isnan(signal_full),
        np.nan,
        macd_line - signal_full,
    )

    return macd_line, signal_full, histogram


def compute_bollinger_bands(
    prices: NDArray[np.float64],
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute Bollinger Bands.

    Formula:
        Middle = SMA(prices, period)
        StdDev = rolling standard deviation over `period`
        Upper  = Middle + num_std * StdDev
        Lower  = Middle - num_std * StdDev

    Output range: Same units as prices; upper >= middle >= lower.
    Staleness: First (period - 1) values are NaN.

    Args:
        prices: 1-D array of close prices.
        period: Look-back window for SMA and StdDev (default 20).
        num_std: Number of standard deviations for bands (default 2.0).

    Returns:
        Tuple of (upper, middle, lower), each a 1-D float64 array.
    """
    n = len(prices)
    upper = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)

    if n < period:
        return upper, middle, lower

    for i in range(period - 1, n):
        window = prices[i - period + 1 : i + 1]
        sma = float(np.mean(window))
        std = float(np.std(window, ddof=0))
        middle[i] = sma
        upper[i] = sma + num_std * std
        lower[i] = sma - num_std * std

    return upper, middle, lower


def compute_atr(
    high: NDArray[np.float64],
    low: NDArray[np.float64],
    close: NDArray[np.float64],
    period: int = 14,
) -> NDArray[np.float64]:
    """Compute Average True Range (ATR).

    Formula:
        TrueRange  = max(H - L, |H - prevClose|, |L - prevClose|)
        ATR        = SMA(TrueRange, period)

    Output range: [0, +inf]; same units as prices.
    Staleness: First `period` values are NaN.

    Args:
        high: 1-D array of bar high prices.
        low: 1-D array of bar low prices.
        close: 1-D array of bar close prices.
        period: SMA look-back period for ATR smoothing (default 14).

    Returns:
        1-D float64 array of ATR values, NaN for warm-up bars.
    """
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return out

    tr = np.full(n, np.nan, dtype=np.float64)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # SMA of TR over period (starting at index `period`)
    for i in range(period, n):
        window = tr[i - period + 1 : i + 1]
        if not np.any(np.isnan(window)):
            out[i] = float(np.mean(window))

    return out


def compute_momentum(prices: NDArray[np.float64], period: int = 5) -> NDArray[np.float64]:
    """Compute price momentum as rate of change.

    Formula:
        momentum[t] = (price[t] - price[t - period]) / price[t - period]

    Output range: (-1, +inf) in practice; 0.0 means no change.
    Staleness: First `period` values are NaN.

    Args:
        prices: 1-D array of close prices.
        period: Look-back period for momentum (default 5).

    Returns:
        1-D float64 array of momentum values, NaN for warm-up bars.
    """
    n = len(prices)
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return out

    for i in range(period, n):
        prev = prices[i - period]
        if prev != 0.0:
            out[i] = (prices[i] - prev) / prev
        else:
            out[i] = np.nan

    return out


def compute_bb_position(price: float, upper: float, lower: float) -> float:
    """Compute the price position within Bollinger Bands.

    Formula:
        bb_position = (price - lower) / (upper - lower)

    Clamped to [0, 1]: 0.0 = at or below lower band, 1.0 = at or above upper.
    Output range: [0.0, 1.0].
    Staleness: Returns 0.5 (neutral) if bands are equal (zero width).

    Args:
        price: Current close price.
        upper: Upper Bollinger Band value.
        lower: Lower Bollinger Band value.

    Returns:
        Float in [0.0, 1.0] representing position within the bands.
    """
    band_width = upper - lower
    if band_width == 0.0:
        return 0.5
    raw = (price - lower) / band_width
    return float(np.clip(raw, 0.0, 1.0))


def normalise_rsi(rsi: float) -> float:
    """Map RSI from [0, 100] to [-1, 1].

    Formula:
        normalised = (rsi - 50) / 50

    Output range: [-1.0, 1.0].
    Staleness: NaN input returns NaN output.

    Args:
        rsi: RSI value in [0, 100].

    Returns:
        Normalised RSI in [-1, 1]; negative = oversold, positive = overbought.
    """
    if np.isnan(rsi):
        return float("nan")
    return (rsi - 50.0) / 50.0
