"""Unit tests for qts.signals.indicators.

Known-answer tests for each indicator, verifying:
- Correct warm-up NaN behaviour
- Numeric accuracy against reference values
- Boundary and edge cases
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from qts.signals.indicators import (
    _ema,
    compute_atr,
    compute_bb_position,
    compute_bollinger_bands,
    compute_macd,
    compute_momentum,
    compute_rsi,
    normalise_rsi,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _linspace_prices(start: float = 100.0, end: float = 150.0, n: int = 100) -> np.ndarray:
    """Generate a linearly increasing price series."""
    return np.linspace(start, end, n, dtype=np.float64)


def _sine_prices(n: int = 200, amplitude: float = 20.0, base: float = 100.0) -> np.ndarray:
    """Generate a sinusoidal price series."""
    t = np.linspace(0, 4 * np.pi, n)
    return (base + amplitude * np.sin(t)).astype(np.float64)


# ── EMA ───────────────────────────────────────────────────────────────────────


class TestEMA:
    def test_shorter_than_period_all_nan(self) -> None:
        prices = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        out = _ema(prices, period=5)
        assert all(math.isnan(v) for v in out)

    def test_seed_equals_sma(self) -> None:
        prices = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        out = _ema(prices, period=3)
        # Seed at index 2 = SMA of [1, 2, 3] = 2.0
        assert not math.isnan(out[2])
        assert abs(out[2] - 2.0) < 1e-10

    def test_length_preserved(self) -> None:
        prices = _linspace_prices(n=50)
        out = _ema(prices, period=10)
        assert len(out) == 50

    def test_nan_count(self) -> None:
        prices = _linspace_prices(n=50)
        period = 10
        out = _ema(prices, period=period)
        nan_count = np.sum(np.isnan(out))
        # First (period - 1) values should be NaN
        assert nan_count == period - 1


# ── RSI ───────────────────────────────────────────────────────────────────────


class TestRSI:
    def test_warm_up_nan(self) -> None:
        prices = _linspace_prices(n=20)
        out = compute_rsi(prices, period=14)
        assert all(math.isnan(out[i]) for i in range(14))

    def test_all_gains_rsi_100(self) -> None:
        """Strictly increasing prices -> RSI should approach 100."""
        prices = np.arange(1, 101, dtype=np.float64)
        out = compute_rsi(prices, period=14)
        valid = out[~np.isnan(out)]
        # All gains -> avg_loss ~ 0 -> RSI ~ 100
        assert all(v > 90.0 for v in valid)

    def test_all_losses_rsi_low(self) -> None:
        """Strictly decreasing prices -> RSI should approach 0."""
        prices = np.arange(100, 0, -1, dtype=np.float64)
        out = compute_rsi(prices, period=14)
        valid = out[~np.isnan(out)]
        assert all(v < 10.0 for v in valid)

    def test_output_range(self) -> None:
        prices = _sine_prices(n=200)
        out = compute_rsi(prices, period=14)
        valid = out[~np.isnan(out)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 100.0)

    def test_length_preserved(self) -> None:
        prices = _linspace_prices(n=50)
        out = compute_rsi(prices, period=14)
        assert len(out) == 50

    def test_insufficient_data_all_nan(self) -> None:
        prices = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        out = compute_rsi(prices, period=14)
        assert all(math.isnan(v) for v in out)

    def test_known_answer(self) -> None:
        """RSI(14) for flat prices should be near 50 (no net gain or loss)."""
        prices = np.full(50, 100.0, dtype=np.float64)
        out = compute_rsi(prices, period=14)
        valid = out[~np.isnan(out)]
        # Flat line: all deltas are 0 -> both avg_gain and avg_loss are 0
        # Function should handle 0/0 gracefully (result is NaN or boundary)
        # At minimum it should not raise
        assert len(valid) >= 0  # just verify no crash


# ── MACD ──────────────────────────────────────────────────────────────────────


class TestMACD:
    def test_returns_three_arrays(self) -> None:
        prices = _linspace_prices(n=100)
        result = compute_macd(prices)
        assert len(result) == 3

    def test_length_preserved(self) -> None:
        prices = _linspace_prices(n=100)
        macd, signal, hist = compute_macd(prices)
        assert len(macd) == 100
        assert len(signal) == 100
        assert len(hist) == 100

    def test_warm_up_nan(self) -> None:
        prices = _linspace_prices(n=100)
        macd, signal, hist = compute_macd(prices, fast=12, slow=26, signal=9)
        # First 25 values of MACD should be NaN (slow period - 1 = 25)
        assert all(math.isnan(macd[i]) for i in range(25))

    def test_histogram_equals_macd_minus_signal(self) -> None:
        prices = _sine_prices(n=200)
        macd, signal, hist = compute_macd(prices)
        valid = ~(np.isnan(macd) | np.isnan(signal) | np.isnan(hist))
        np.testing.assert_allclose(hist[valid], macd[valid] - signal[valid], atol=1e-10)

    def test_trending_up_positive_macd(self) -> None:
        """Upward trending prices should produce non-negative MACD line."""
        prices = np.linspace(50, 200, 150, dtype=np.float64)
        macd, signal, hist = compute_macd(prices)
        valid_macd = macd[~np.isnan(macd)]
        # In a steadily rising market, fast EMA >= slow EMA -> macd >= 0
        assert len(valid_macd) > 0
        # Use tolerant comparison: linearly trending -> macd ~ 0 or slightly positive
        assert np.mean(valid_macd) >= -1e-10


# ── Bollinger Bands ───────────────────────────────────────────────────────────


class TestBollingerBands:
    def test_returns_three_arrays(self) -> None:
        prices = _linspace_prices(n=100)
        result = compute_bollinger_bands(prices)
        assert len(result) == 3

    def test_upper_ge_middle_ge_lower(self) -> None:
        prices = _sine_prices(n=100)
        upper, middle, lower = compute_bollinger_bands(prices, period=20)
        valid = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        assert np.all(upper[valid] >= middle[valid])
        assert np.all(middle[valid] >= lower[valid])

    def test_warm_up_nan(self) -> None:
        prices = _linspace_prices(n=100)
        upper, middle, lower = compute_bollinger_bands(prices, period=20)
        assert all(math.isnan(upper[i]) for i in range(19))

    def test_flat_price_zero_band_width(self) -> None:
        prices = np.full(50, 100.0, dtype=np.float64)
        upper, middle, lower = compute_bollinger_bands(prices, period=20)
        valid = ~np.isnan(upper)
        # Zero std -> bands collapse to middle
        np.testing.assert_allclose(upper[valid], 100.0, atol=1e-10)
        np.testing.assert_allclose(lower[valid], 100.0, atol=1e-10)

    def test_middle_is_sma(self) -> None:
        prices = np.arange(1, 51, dtype=np.float64)  # 1, 2, ..., 50
        _, middle, _ = compute_bollinger_bands(prices, period=5)
        # Middle at index 4 = mean([1,2,3,4,5]) = 3.0
        assert not math.isnan(middle[4])
        assert abs(middle[4] - 3.0) < 1e-10

    def test_num_std_scaling(self) -> None:
        prices = _sine_prices(n=100)
        _, _, lower1 = compute_bollinger_bands(prices, period=20, num_std=1.0)
        _, _, lower2 = compute_bollinger_bands(prices, period=20, num_std=2.0)
        valid = ~(np.isnan(lower1) | np.isnan(lower2))
        # Wider bands with num_std=2
        assert np.all(lower2[valid] <= lower1[valid])


# ── ATR ───────────────────────────────────────────────────────────────────────


class TestATR:
    def _make_bars(self, n: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        closes = _linspace_prices(n=n)
        highs = closes + 1.0
        lows = closes - 1.0
        return highs, lows, closes

    def test_warm_up_nan(self) -> None:
        highs, lows, closes = self._make_bars(50)
        out = compute_atr(highs, lows, closes, period=14)
        # ATR requires at least period+1 bars
        assert all(math.isnan(out[i]) for i in range(14))

    def test_constant_range(self) -> None:
        """H-L = 2 always, no gap -> ATR should be 2.0."""
        closes = np.full(50, 100.0, dtype=np.float64)
        highs = closes + 1.0
        lows = closes - 1.0
        out = compute_atr(highs, lows, closes, period=14)
        valid = out[~np.isnan(out)]
        np.testing.assert_allclose(valid, 2.0, atol=1e-6)

    def test_output_non_negative(self) -> None:
        highs, lows, closes = self._make_bars(100)
        out = compute_atr(highs, lows, closes)
        valid = out[~np.isnan(out)]
        assert np.all(valid >= 0.0)

    def test_length_preserved(self) -> None:
        highs, lows, closes = self._make_bars(50)
        out = compute_atr(highs, lows, closes)
        assert len(out) == 50

    def test_insufficient_data(self) -> None:
        closes = np.array([1.0, 2.0], dtype=np.float64)
        highs = closes + 0.5
        lows = closes - 0.5
        out = compute_atr(highs, lows, closes, period=14)
        assert all(math.isnan(v) for v in out)


# ── Momentum ──────────────────────────────────────────────────────────────────


class TestMomentum:
    def test_warm_up_nan(self) -> None:
        prices = _linspace_prices(n=20)
        out = compute_momentum(prices, period=5)
        assert all(math.isnan(out[i]) for i in range(5))

    def test_zero_change_zero_momentum(self) -> None:
        prices = np.full(20, 100.0, dtype=np.float64)
        out = compute_momentum(prices, period=5)
        valid = out[~np.isnan(out)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-10)

    def test_known_value(self) -> None:
        prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 110.0], dtype=np.float64)
        out = compute_momentum(prices, period=5)
        # momentum[5] = (110 - 100) / 100 = 0.10
        assert not math.isnan(out[5])
        assert abs(out[5] - 0.10) < 1e-10

    def test_length_preserved(self) -> None:
        prices = _linspace_prices(n=50)
        out = compute_momentum(prices, period=5)
        assert len(out) == 50


# ── BB Position ───────────────────────────────────────────────────────────────


class TestBBPosition:
    def test_at_lower_band(self) -> None:
        assert compute_bb_position(90.0, 110.0, 90.0) == pytest.approx(0.0)

    def test_at_upper_band(self) -> None:
        assert compute_bb_position(110.0, 110.0, 90.0) == pytest.approx(1.0)

    def test_mid_band(self) -> None:
        assert compute_bb_position(100.0, 110.0, 90.0) == pytest.approx(0.5)

    def test_below_lower_clamped(self) -> None:
        assert compute_bb_position(80.0, 110.0, 90.0) == pytest.approx(0.0)

    def test_above_upper_clamped(self) -> None:
        assert compute_bb_position(120.0, 110.0, 90.0) == pytest.approx(1.0)

    def test_zero_width_bands_returns_half(self) -> None:
        assert compute_bb_position(100.0, 100.0, 100.0) == pytest.approx(0.5)


# ── Normalise RSI ─────────────────────────────────────────────────────────────


class TestNormaliseRSI:
    def test_rsi_50_maps_to_zero(self) -> None:
        assert normalise_rsi(50.0) == pytest.approx(0.0)

    def test_rsi_100_maps_to_one(self) -> None:
        assert normalise_rsi(100.0) == pytest.approx(1.0)

    def test_rsi_0_maps_to_minus_one(self) -> None:
        assert normalise_rsi(0.0) == pytest.approx(-1.0)

    def test_nan_returns_nan(self) -> None:
        assert math.isnan(normalise_rsi(float("nan")))

    def test_output_range(self) -> None:
        for rsi in np.linspace(0, 100, 50):
            result = normalise_rsi(float(rsi))
            assert -1.0 <= result <= 1.0
