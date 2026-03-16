"""Integration tests: Walk-forward validation with SMACrossoverStrategy on synthetic data.

Verifies:
- WalkForwardEngine runs end-to-end without errors
- Produces WalkForwardResult with 5 windows
- Each window has non-None train_result
- Results are deterministic (same inputs -> identical results)
- WalkForwardResult structure is valid
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from qts.models.base import Bar
from qts.simulation.backtest import BacktestSettings
from qts.simulation.walk_forward import (
    WalkForwardConfig,
    WalkForwardEngine,
    WalkForwardResult,
)
from qts.strategies.sma_crossover import SMACrossoverStrategy

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_bars(
    n: int = 600,
    symbol: str = "SYNTH",
    start_price: float = 100.0,
    trend: float = 0.001,
    noise: float = 0.005,
    seed: int = 42,
) -> list[Bar]:
    """Generate synthetic OHLCV bars with a mild upward trend."""
    rng = np.random.default_rng(seed)
    bars = []
    price = start_price
    base_time = datetime(2024, 1, 1)

    for i in range(n):
        daily_return = trend + rng.normal(0, noise)
        price = price * (1.0 + daily_return)
        high = price * (1 + abs(rng.normal(0, 0.002)))
        low = price * (1 - abs(rng.normal(0, 0.002)))
        bars.append(
            Bar(
                timestamp=base_time + timedelta(days=i),
                symbol=symbol,
                open=price * (1 + rng.normal(0, 0.001)),
                high=high,
                low=low,
                close=price,
                volume=float(rng.integers(1000, 10000)),
            )
        )
    return bars


def _sma_factory() -> SMACrossoverStrategy:
    """Create a fresh SMACrossoverStrategy for walk-forward testing."""
    return SMACrossoverStrategy(fast_period=10, slow_period=30, quantity=1.0)


# ── Integration Tests ─────────────────────────────────────────────────────────


class TestWalkForwardIntegration:
    def test_produces_walk_forward_result(self) -> None:
        """Full pipeline returns a WalkForwardResult instance."""
        bars = _make_bars(n=600, seed=42)
        settings = BacktestSettings(initial_capital=100_000.0)
        config = WalkForwardConfig(n_windows=5, min_bars_per_window=100)
        engine = WalkForwardEngine(_sma_factory, settings, config)

        result = engine.run(bars)
        assert isinstance(result, WalkForwardResult)

    def test_result_has_five_windows(self) -> None:
        """WalkForwardResult contains exactly 5 windows."""
        bars = _make_bars(n=600, seed=42)
        settings = BacktestSettings(initial_capital=100_000.0)
        config = WalkForwardConfig(n_windows=5, min_bars_per_window=100)
        engine = WalkForwardEngine(_sma_factory, settings, config)

        result = engine.run(bars)
        assert len(result.windows) == 5

    def test_each_window_has_non_none_train_result(self) -> None:
        """Each walk-forward window has a non-None train_result."""
        bars = _make_bars(n=600, seed=42)
        settings = BacktestSettings(initial_capital=100_000.0)
        config = WalkForwardConfig(n_windows=5, min_bars_per_window=100)
        engine = WalkForwardEngine(_sma_factory, settings, config)

        result = engine.run(bars)
        for i, window in enumerate(result.windows):
            assert window.train_result is not None, (
                f"Window {i} has None train_result"
            )

    def test_each_window_has_non_none_validate_result(self) -> None:
        """Each walk-forward window has a non-None validate_result."""
        bars = _make_bars(n=600, seed=42)
        settings = BacktestSettings(initial_capital=100_000.0)
        config = WalkForwardConfig(n_windows=5, min_bars_per_window=100)
        engine = WalkForwardEngine(_sma_factory, settings, config)

        result = engine.run(bars)
        for i, window in enumerate(result.windows):
            assert window.validate_result is not None, (
                f"Window {i} has None validate_result"
            )

    def test_each_window_has_non_none_test_result(self) -> None:
        """Each walk-forward window has a non-None test_result."""
        bars = _make_bars(n=600, seed=42)
        settings = BacktestSettings(initial_capital=100_000.0)
        config = WalkForwardConfig(n_windows=5, min_bars_per_window=100)
        engine = WalkForwardEngine(_sma_factory, settings, config)

        result = engine.run(bars)
        for i, window in enumerate(result.windows):
            assert window.test_result is not None, (
                f"Window {i} has None test_result"
            )

    def test_deterministic_same_inputs_same_results(self) -> None:
        """Running twice with identical inputs produces identical results."""
        bars = _make_bars(n=600, seed=7)
        settings = BacktestSettings(initial_capital=100_000.0)
        config = WalkForwardConfig(n_windows=5, min_bars_per_window=100)

        engine1 = WalkForwardEngine(_sma_factory, settings, config)
        result1 = engine1.run(bars)

        engine2 = WalkForwardEngine(_sma_factory, settings, config)
        result2 = engine2.run(bars)

        assert result1.aggregate_test_sharpe == result2.aggregate_test_sharpe
        assert result1.aggregate_test_return == result2.aggregate_test_return
        assert result1.aggregate_test_max_drawdown == result2.aggregate_test_max_drawdown
        assert result1.per_window_test_sharpes == result2.per_window_test_sharpes

    def test_aggregate_sharpe_is_finite(self) -> None:
        """Aggregate test Sharpe ratio should be a finite float."""
        bars = _make_bars(n=600, seed=42)
        settings = BacktestSettings(initial_capital=100_000.0)
        config = WalkForwardConfig(n_windows=5, min_bars_per_window=100)
        engine = WalkForwardEngine(_sma_factory, settings, config)

        result = engine.run(bars)
        import math
        assert math.isfinite(result.aggregate_test_sharpe)

    def test_aggregate_max_drawdown_non_negative(self) -> None:
        """Aggregate max drawdown should be >= 0."""
        bars = _make_bars(n=600, seed=42)
        settings = BacktestSettings(initial_capital=100_000.0)
        config = WalkForwardConfig(n_windows=5, min_bars_per_window=100)
        engine = WalkForwardEngine(_sma_factory, settings, config)

        result = engine.run(bars)
        assert result.aggregate_test_max_drawdown >= 0.0

    def test_per_window_sharpes_length_matches_windows(self) -> None:
        """per_window_test_sharpes should have length equal to n_windows."""
        bars = _make_bars(n=600, seed=42)
        settings = BacktestSettings(initial_capital=100_000.0)
        config = WalkForwardConfig(n_windows=5, min_bars_per_window=100)
        engine = WalkForwardEngine(_sma_factory, settings, config)

        result = engine.run(bars)
        assert len(result.per_window_test_sharpes) == 5

    def test_insufficient_data_raises_value_error(self) -> None:
        """Should raise ValueError when not enough bars for configured windows."""
        bars = _make_bars(n=50, seed=42)  # 50 / 5 = 10 bars per window < 100 min
        settings = BacktestSettings(initial_capital=100_000.0)
        config = WalkForwardConfig(n_windows=5, min_bars_per_window=100)
        engine = WalkForwardEngine(_sma_factory, settings, config)

        with pytest.raises(ValueError, match="Insufficient data"):
            engine.run(bars)

    def test_is_significant_field_is_bool(self) -> None:
        """is_significant field should be a boolean."""
        bars = _make_bars(n=600, seed=42)
        settings = BacktestSettings(initial_capital=100_000.0)
        config = WalkForwardConfig(n_windows=5, min_bars_per_window=100)
        engine = WalkForwardEngine(_sma_factory, settings, config)

        result = engine.run(bars)
        assert isinstance(result.is_significant, bool)

    def test_train_result_equity_curve_non_empty(self) -> None:
        """Train result equity curves should be non-empty."""
        bars = _make_bars(n=600, seed=42)
        settings = BacktestSettings(initial_capital=100_000.0)
        config = WalkForwardConfig(n_windows=5, min_bars_per_window=100)
        engine = WalkForwardEngine(_sma_factory, settings, config)

        result = engine.run(bars)
        for window in result.windows:
            assert len(window.train_result.equity_curve) > 0  # type: ignore[union-attr]

    def test_with_500_plus_bars(self) -> None:
        """Explicitly test with 500+ bars as required by spec."""
        bars = _make_bars(n=500, seed=42)
        settings = BacktestSettings(initial_capital=100_000.0)
        config = WalkForwardConfig(n_windows=5, min_bars_per_window=100)
        engine = WalkForwardEngine(_sma_factory, settings, config)

        result = engine.run(bars)
        assert isinstance(result, WalkForwardResult)
        assert len(result.windows) == 5

    def test_window_train_bars_counts_are_positive(self) -> None:
        """Each window should have a positive number of train bars."""
        bars = _make_bars(n=600, seed=42)
        settings = BacktestSettings(initial_capital=100_000.0)
        config = WalkForwardConfig(n_windows=5, min_bars_per_window=100)
        engine = WalkForwardEngine(_sma_factory, settings, config)

        result = engine.run(bars)
        for window in result.windows:
            assert len(window.train_bars) > 0
            assert len(window.validate_bars) > 0
            assert len(window.test_bars) > 0
