"""Walk-forward validation engine for out-of-sample strategy testing."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from qts.models.base import Bar
from qts.simulation.backtest import BacktestEngine, BacktestResult, BacktestSettings
from qts.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Configuration for walk-forward validation.

    Attributes:
        train_ratio: Fraction of each window used for training.
        validate_ratio: Fraction used for validation.
        test_ratio: Fraction used for out-of-sample testing.
        n_windows: Number of rolling windows.
        min_bars_per_window: Minimum bars required per window.
    """

    train_ratio: float = 0.70
    validate_ratio: float = 0.15
    test_ratio: float = 0.15
    n_windows: int = 5
    min_bars_per_window: int = 100


@dataclass(slots=True)
class WalkForwardWindow:
    """Single walk-forward validation window with results."""

    window_index: int
    train_bars: list[Bar]
    validate_bars: list[Bar]
    test_bars: list[Bar]
    train_result: BacktestResult | None = None
    validate_result: BacktestResult | None = None
    test_result: BacktestResult | None = None


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Aggregate results from walk-forward validation."""

    windows: tuple[WalkForwardWindow, ...]
    aggregate_test_sharpe: float
    aggregate_test_return: float
    aggregate_test_max_drawdown: float
    per_window_test_sharpes: tuple[float, ...]
    is_significant: bool  # True if majority of test windows have positive Sharpe
    monte_carlo_p_value: float | None = None


class WalkForwardEngine:
    """Walk-forward validation engine.

    Splits data into rolling train/validate/test windows, runs backtests
    on each, and aggregates out-of-sample performance metrics.

    Args:
        strategy_factory: Callable that returns a fresh Strategy instance.
        settings: Backtest settings (capital, commission, etc.).
        config: Walk-forward configuration.
    """

    def __init__(
        self,
        strategy_factory: Callable[[], Strategy],
        settings: BacktestSettings,
        config: WalkForwardConfig | None = None,
    ) -> None:
        self._strategy_factory = strategy_factory
        self._settings = settings
        self._config = config or WalkForwardConfig()

    def run(self, bars: list[Bar]) -> WalkForwardResult:
        """Run walk-forward validation on the given bar data.

        Args:
            bars: Full historical bar data to split into windows.

        Returns:
            Aggregate walk-forward results.

        Raises:
            ValueError: If insufficient data for the configured windows.
        """
        windows = self._create_windows(bars)

        for window in windows:
            window.train_result = self._run_backtest(window.train_bars)
            window.validate_result = self._run_backtest(window.validate_bars)
            window.test_result = self._run_backtest(window.test_bars)

        return self._aggregate(windows)

    def _create_windows(self, bars: list[Bar]) -> list[WalkForwardWindow]:
        """Split bars into rolling walk-forward windows."""
        n = len(bars)
        n_windows = self._config.n_windows

        # Calculate window size with overlap
        # Each window starts later into the data
        window_size = n // n_windows
        if window_size < self._config.min_bars_per_window:
            raise ValueError(
                f"Insufficient data: {n} bars for {n_windows} windows "
                f"(need {self._config.min_bars_per_window} per window, got {window_size})"
            )

        windows: list[WalkForwardWindow] = []
        step = (n - window_size) // max(1, n_windows - 1) if n_windows > 1 else 0

        for i in range(n_windows):
            start = i * step
            end = min(start + window_size, n)
            window_bars = bars[start:end]

            train_end = int(len(window_bars) * self._config.train_ratio)
            validate_end = train_end + int(len(window_bars) * self._config.validate_ratio)

            windows.append(
                WalkForwardWindow(
                    window_index=i,
                    train_bars=window_bars[:train_end],
                    validate_bars=window_bars[train_end:validate_end],
                    test_bars=window_bars[validate_end:],
                )
            )

        return windows

    def _run_backtest(self, bars: list[Bar]) -> BacktestResult | None:
        """Run backtest on a subset of bars."""
        if len(bars) < 10:
            return None
        strategy = self._strategy_factory()
        engine = BacktestEngine(
            strategy=strategy,
            bars=bars,
            settings=self._settings,
        )
        try:
            return engine.run()
        except Exception as exc:
            logger.warning("Backtest failed on window: %s", exc)
            return None

    def _aggregate(self, windows: list[WalkForwardWindow]) -> WalkForwardResult:
        """Aggregate results across all windows."""
        test_sharpes: list[float] = []
        test_returns: list[float] = []
        test_drawdowns: list[float] = []

        for w in windows:
            if w.test_result is not None:
                test_sharpes.append(w.test_result.sharpe_ratio)
                test_returns.append(w.test_result.total_return)
                test_drawdowns.append(w.test_result.max_drawdown)

        avg_sharpe = float(np.mean(test_sharpes)) if test_sharpes else 0.0
        avg_return = float(np.mean(test_returns)) if test_returns else 0.0
        max_dd = float(np.max(test_drawdowns)) if test_drawdowns else 0.0

        positive_count = sum(1 for s in test_sharpes if s > 0)
        is_significant = positive_count > len(test_sharpes) / 2 if test_sharpes else False

        return WalkForwardResult(
            windows=tuple(windows),
            aggregate_test_sharpe=avg_sharpe,
            aggregate_test_return=avg_return,
            aggregate_test_max_drawdown=max_dd,
            per_window_test_sharpes=tuple(test_sharpes),
            is_significant=is_significant,
        )


def monte_carlo_permutation_test(
    returns: list[float],
    n_permutations: int = 1000,
    seed: int = 42,
) -> float:
    """Test statistical significance of returns via permutation.

    Shuffles returns n_permutations times, computes Sharpe each time,
    and returns the fraction of shuffled Sharpes >= observed Sharpe.

    Args:
        returns: List of trade or period returns.
        n_permutations: Number of random shuffles.
        seed: Random seed for reproducibility.

    Returns:
        p-value in [0, 1]. Lower = more significant.
    """
    if not returns or all(r == 0 for r in returns):
        return 1.0

    arr = np.array(returns, dtype=np.float64)
    observed_sharpe = _compute_sharpe(arr)

    rng = np.random.default_rng(seed)
    count_ge = 0

    for _ in range(n_permutations):
        shuffled = rng.permutation(arr)
        if _compute_sharpe(shuffled) >= observed_sharpe:
            count_ge += 1

    return count_ge / n_permutations


def _compute_sharpe(returns: np.ndarray) -> float:
    """Compute Sharpe ratio from array of returns."""
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std < 1e-12:
        return 0.0
    return float(np.mean(returns)) / std
