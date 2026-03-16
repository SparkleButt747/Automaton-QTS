"""Property-based tests for technical indicators using Hypothesis."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from qts.signals.indicators import (
    compute_atr,
    compute_bb_position,
    compute_bollinger_bands,
    compute_macd,
    compute_momentum,
    compute_rsi,
    normalise_rsi,
)

# ── Strategies ────────────────────────────────────────────────────────────────

_PRICE_STRATEGY = st.floats(
    min_value=1.0, max_value=100_000.0, allow_nan=False, allow_infinity=False
)
_PERIOD_STRATEGY = st.integers(min_value=2, max_value=50)


# A valid positive price series of length n
def _price_array_strategy(min_size: int = 2, max_size: int = 300) -> st.SearchStrategy:  # type: ignore[type-arg]
    return arrays(
        dtype=np.float64,
        shape=st.integers(min_value=min_size, max_value=max_size),
        elements=st.floats(
            min_value=1.0, max_value=100_000.0, allow_nan=False, allow_infinity=False
        ),
    )


# ── RSI ───────────────────────────────────────────────────────────────────────


class TestRSIProperty:
    @given(
        prices=_price_array_strategy(min_size=20, max_size=300),
        period=st.integers(min_value=2, max_value=14),
    )
    @settings(max_examples=200)
    def test_rsi_always_in_0_100(self, prices: np.ndarray, period: int) -> None:
        """RSI output must always be in [0, 100] for any positive price series."""
        # Need at least period+1 prices for non-NaN values
        assume(len(prices) > period)
        out = compute_rsi(prices, period=period)
        valid = out[~np.isnan(out)]
        if len(valid) > 0:
            assert float(np.min(valid)) >= 0.0, f"RSI below 0: {float(np.min(valid))}"
            assert float(np.max(valid)) <= 100.0, f"RSI above 100: {float(np.max(valid))}"

    @given(
        prices=_price_array_strategy(min_size=30, max_size=200),
        period=st.integers(min_value=2, max_value=14),
    )
    @settings(max_examples=100)
    def test_rsi_length_preserved(self, prices: np.ndarray, period: int) -> None:
        """RSI output must have the same length as input."""
        out = compute_rsi(prices, period=period)
        assert len(out) == len(prices)

    @given(
        prices=_price_array_strategy(min_size=30, max_size=200),
        period=st.integers(min_value=2, max_value=14),
    )
    @settings(max_examples=100)
    def test_rsi_warmup_nan(self, prices: np.ndarray, period: int) -> None:
        """RSI first `period` values must be NaN."""
        assume(len(prices) >= period + 1)
        out = compute_rsi(prices, period=period)
        for i in range(period):
            assert math.isnan(out[i]), f"Expected NaN at index {i}, got {out[i]}"


# ── MACD ──────────────────────────────────────────────────────────────────────


class TestMACDProperty:
    @given(
        prices=_price_array_strategy(min_size=50, max_size=300),
        fast=st.integers(min_value=2, max_value=12),
        slow=st.integers(min_value=13, max_value=30),
        signal=st.integers(min_value=2, max_value=9),
    )
    @settings(max_examples=200)
    def test_histogram_equals_macd_minus_signal(
        self,
        prices: np.ndarray,
        fast: int,
        slow: int,
        signal: int,
    ) -> None:
        """MACD histogram must equal macd_line - signal_line within float tolerance."""
        assume(fast < slow)
        macd_line, signal_line, histogram = compute_macd(
            prices, fast=fast, slow=slow, signal=signal
        )
        valid = ~(np.isnan(macd_line) | np.isnan(signal_line) | np.isnan(histogram))
        if valid.any():
            expected = macd_line[valid] - signal_line[valid]
            np.testing.assert_allclose(
                histogram[valid],
                expected,
                atol=1e-10,
                err_msg="histogram != macd_line - signal_line",
            )

    @given(prices=_price_array_strategy(min_size=50, max_size=300))
    @settings(max_examples=100)
    def test_macd_length_preserved(self, prices: np.ndarray) -> None:
        """All MACD arrays must have the same length as input."""
        macd_line, signal_line, histogram = compute_macd(prices)
        assert len(macd_line) == len(prices)
        assert len(signal_line) == len(prices)
        assert len(histogram) == len(prices)


# ── Bollinger Bands ───────────────────────────────────────────────────────────


class TestBollingerBandsProperty:
    @given(
        prices=_price_array_strategy(min_size=25, max_size=300),
        period=st.integers(min_value=2, max_value=20),
        num_std=st.floats(min_value=0.5, max_value=3.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_upper_ge_middle_ge_lower(
        self, prices: np.ndarray, period: int, num_std: float
    ) -> None:
        """Bollinger upper >= middle >= lower must always hold."""
        upper, middle, lower = compute_bollinger_bands(prices, period=period, num_std=num_std)
        valid = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        if valid.any():
            assert np.all(upper[valid] >= middle[valid]), "upper < middle found"
            assert np.all(middle[valid] >= lower[valid]), "middle < lower found"

    @given(
        prices=_price_array_strategy(min_size=25, max_size=300),
        period=st.integers(min_value=2, max_value=20),
    )
    @settings(max_examples=100)
    def test_bb_length_preserved(self, prices: np.ndarray, period: int) -> None:
        """All Bollinger Band arrays must have the same length as input."""
        upper, middle, lower = compute_bollinger_bands(prices, period=period)
        assert len(upper) == len(prices)
        assert len(middle) == len(prices)
        assert len(lower) == len(prices)

    @given(
        price=_PRICE_STRATEGY,
        upper=_PRICE_STRATEGY,
        lower=_PRICE_STRATEGY,
    )
    @settings(max_examples=300, suppress_health_check=[HealthCheck.filter_too_much])
    def test_bb_position_in_0_1_when_within_bands(
        self, price: float, upper: float, lower: float
    ) -> None:
        """BB position must be in [0, 1] when price is between lower and upper bands."""
        assume(upper > lower)
        assume(lower <= price <= upper)
        result = compute_bb_position(price, upper, lower)
        assert 0.0 <= result <= 1.0, f"BB position out of [0,1]: {result}"

    @given(
        price=_PRICE_STRATEGY,
        upper=_PRICE_STRATEGY,
        lower=_PRICE_STRATEGY,
    )
    @settings(max_examples=300)
    def test_bb_position_always_in_0_1(self, price: float, upper: float, lower: float) -> None:
        """BB position must always be in [0, 1] regardless of price position."""
        assume(upper >= lower)
        result = compute_bb_position(price, upper, lower)
        assert 0.0 <= result <= 1.0, f"BB position out of [0,1]: {result}"


# ── ATR ───────────────────────────────────────────────────────────────────────


class TestATRProperty:
    @given(
        n=st.integers(min_value=20, max_value=200),
        period=st.integers(min_value=2, max_value=14),
        base=st.floats(min_value=10.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
        range_size=st.floats(
            min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False
        ),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=200)
    def test_atr_always_non_negative(
        self, n: int, period: int, base: float, range_size: float, seed: int
    ) -> None:
        """ATR must always be non-negative for valid OHLC data (high >= low, close > 0)."""
        assume(n > period)
        rng = np.random.default_rng(seed)
        # Build valid OHLC: high >= close >= low > 0
        closes = np.full(n, base, dtype=np.float64)
        closes += rng.uniform(-range_size * 0.1, range_size * 0.1, n)
        closes = np.clip(closes, 1.0, None)
        highs = closes + abs(rng.uniform(0.0, range_size, n))
        lows = closes - abs(rng.uniform(0.0, range_size * 0.5, n))
        lows = np.clip(lows, 0.01, None)

        out = compute_atr(highs, lows, closes, period=period)
        valid = out[~np.isnan(out)]
        if len(valid) > 0:
            assert float(np.min(valid)) >= 0.0, f"ATR negative: {float(np.min(valid))}"

    @given(
        n=st.integers(min_value=20, max_value=200),
        period=st.integers(min_value=2, max_value=14),
        base=st.floats(min_value=10.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_atr_length_preserved(self, n: int, period: int, base: float) -> None:
        """ATR output length must equal input length."""
        closes = np.full(n, base, dtype=np.float64)
        highs = closes + 1.0
        lows = closes - 1.0
        out = compute_atr(highs, lows, closes, period=period)
        assert len(out) == n


# ── Momentum ──────────────────────────────────────────────────────────────────


class TestMomentumProperty:
    @given(
        base=_PRICE_STRATEGY,
        n=st.integers(min_value=10, max_value=200),
        period=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=200)
    def test_momentum_zero_when_all_equal(self, base: float, n: int, period: int) -> None:
        """Momentum must be 0 when all prices are equal (no price change)."""
        assume(n > period)
        prices = np.full(n, base, dtype=np.float64)
        out = compute_momentum(prices, period=period)
        valid = out[~np.isnan(out)]
        if len(valid) > 0:
            np.testing.assert_allclose(
                valid,
                0.0,
                atol=1e-10,
                err_msg="Momentum != 0 for flat price series",
            )

    @given(
        prices=_price_array_strategy(min_size=10, max_size=200),
        period=st.integers(min_value=2, max_value=10),
    )
    @settings(max_examples=100)
    def test_momentum_length_preserved(self, prices: np.ndarray, period: int) -> None:
        """Momentum output length must equal input length."""
        out = compute_momentum(prices, period=period)
        assert len(out) == len(prices)


# ── normalise_rsi ─────────────────────────────────────────────────────────────


class TestNormaliseRSIProperty:
    @given(rsi=st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
    @settings(max_examples=300)
    def test_normalise_rsi_maps_to_minus1_to_1(self, rsi: float) -> None:
        """normalise_rsi must map [0, 100] to [-1, 1]."""
        result = normalise_rsi(rsi)
        assert -1.0 <= result <= 1.0, f"normalise_rsi({rsi}) = {result} is out of [-1, 1]"

    @given(rsi=st.floats(min_value=0.0, max_value=100.0, allow_nan=False))
    @settings(max_examples=300)
    def test_normalise_rsi_monotone(self, rsi: float) -> None:
        """normalise_rsi must be monotonically increasing (higher RSI -> higher output)."""
        delta = 0.01
        if rsi + delta <= 100.0:
            result_lo = normalise_rsi(rsi)
            result_hi = normalise_rsi(rsi + delta)
            assert (
                result_hi >= result_lo - 1e-12
            ), f"normalise_rsi not monotone: f({rsi})={result_lo} > f({rsi+delta})={result_hi}"

    def test_normalise_rsi_nan_returns_nan(self) -> None:
        """normalise_rsi(NaN) must return NaN."""
        assert math.isnan(normalise_rsi(float("nan")))
