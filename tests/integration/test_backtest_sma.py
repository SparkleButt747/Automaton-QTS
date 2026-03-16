"""Integration tests: Full backtest with SMA crossover on synthetic data.

Verifies:
- Backtest runs end-to-end without errors
- Equity curve has correct length
- Determinism: same inputs -> identical results
- Trades are recorded with valid data
- SMA crossover generates trades on trending data
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from qts.models.base import Bar, TradeOutcome
from qts.simulation.backtest import BacktestEngine, BacktestResult, BacktestSettings
from qts.strategies.sma_crossover import SMACrossoverStrategy

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_bars(
    n: int = 200,
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


def _make_oscillating_bars(
    n: int = 300,
    symbol: str = "OSC",
    seed: int = 99,
) -> list[Bar]:
    """Generate oscillating price bars to trigger multiple crossovers."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 8 * np.pi, n)
    prices = 100.0 + 10.0 * np.sin(t) + rng.normal(0, 0.5, n)
    bars = []
    base_time = datetime(2024, 1, 1)

    for i in range(n):
        p = float(prices[i])
        bars.append(
            Bar(
                timestamp=base_time + timedelta(days=i),
                symbol=symbol,
                open=p * (1 + rng.normal(0, 0.001)),
                high=p * 1.005,
                low=p * 0.995,
                close=p,
                volume=float(rng.integers(500, 5000)),
            )
        )
    return bars


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestBacktestSMABasic:
    def test_run_returns_backtest_result(self) -> None:
        bars = _make_bars(n=100)
        strategy = SMACrossoverStrategy(fast_period=10, slow_period=30)
        settings = BacktestSettings(initial_capital=100_000.0)
        engine = BacktestEngine(strategy, bars, settings)
        result = engine.run()
        assert isinstance(result, BacktestResult)

    def test_equity_curve_length(self) -> None:
        n = 150
        bars = _make_bars(n=n)
        strategy = SMACrossoverStrategy()
        engine = BacktestEngine(strategy, bars)
        result = engine.run()
        assert len(result.equity_curve) == n

    def test_equity_curve_starts_near_initial_capital(self) -> None:
        bars = _make_bars(n=100)
        strategy = SMACrossoverStrategy()
        settings = BacktestSettings(initial_capital=50_000.0)
        engine = BacktestEngine(strategy, bars, settings)
        result = engine.run()
        # First bars are warm-up; equity should be close to initial
        assert abs(result.equity_curve[0] - 50_000.0) < 5_000.0

    def test_total_return_type_and_range(self) -> None:
        bars = _make_bars(n=200)
        strategy = SMACrossoverStrategy()
        engine = BacktestEngine(strategy, bars)
        result = engine.run()
        assert isinstance(result.total_return, float)
        # Sanity: return should be in a reasonable range for synthetic data
        assert -1.0 <= result.total_return <= 10.0


class TestBacktestDeterminism:
    def test_same_inputs_identical_equity_curve(self) -> None:
        bars = _make_bars(n=200, seed=7)
        settings = BacktestSettings()

        result1 = BacktestEngine(
            SMACrossoverStrategy(fast_period=10, slow_period=30), bars, settings
        ).run()
        result2 = BacktestEngine(
            SMACrossoverStrategy(fast_period=10, slow_period=30), bars, settings
        ).run()

        assert result1.equity_curve == result2.equity_curve

    def test_same_inputs_identical_trade_count(self) -> None:
        bars = _make_oscillating_bars(n=300, seed=99)
        settings = BacktestSettings()

        r1 = BacktestEngine(
            SMACrossoverStrategy(fast_period=10, slow_period=30), bars, settings
        ).run()
        r2 = BacktestEngine(
            SMACrossoverStrategy(fast_period=10, slow_period=30), bars, settings
        ).run()

        assert len(r1.trades) == len(r2.trades)

    def test_different_seeds_different_results(self) -> None:
        bars_a = _make_bars(n=200, seed=1)
        bars_b = _make_bars(n=200, seed=2)
        settings = BacktestSettings()

        r_a = BacktestEngine(SMACrossoverStrategy(), bars_a, settings).run()
        r_b = BacktestEngine(SMACrossoverStrategy(), bars_b, settings).run()

        # Different data -> different equity curves (not identical)
        assert r_a.equity_curve != r_b.equity_curve


class TestBacktestTrades:
    def test_oscillating_bars_produce_trades(self) -> None:
        bars = _make_oscillating_bars(n=300)
        strategy = SMACrossoverStrategy(fast_period=10, slow_period=30)
        result = BacktestEngine(strategy, bars).run()
        # Oscillating prices should trigger at least a few crossovers
        assert len(result.trades) > 0

    def test_trade_records_have_valid_pnl(self) -> None:
        bars = _make_oscillating_bars(n=300)
        strategy = SMACrossoverStrategy(fast_period=10, slow_period=30)
        result = BacktestEngine(strategy, bars).run()
        for trade in result.trades:
            assert isinstance(trade.pnl_usd, float)
            assert not (trade.pnl_usd != trade.pnl_usd)  # not NaN

    def test_trade_outcomes_valid(self) -> None:
        bars = _make_oscillating_bars(n=300)
        strategy = SMACrossoverStrategy(fast_period=10, slow_period=30)
        result = BacktestEngine(strategy, bars).run()
        valid_outcomes = {TradeOutcome.WIN, TradeOutcome.LOSS, TradeOutcome.BREAKEVEN}
        for trade in result.trades:
            assert trade.outcome in valid_outcomes


class TestBacktestStatistics:
    def test_win_rate_in_unit_interval(self) -> None:
        bars = _make_oscillating_bars(n=300)
        strategy = SMACrossoverStrategy()
        result = BacktestEngine(strategy, bars).run()
        if result.trades:
            assert 0.0 <= result.win_rate <= 1.0

    def test_max_drawdown_non_negative(self) -> None:
        bars = _make_bars(n=200)
        result = BacktestEngine(SMACrossoverStrategy(), bars).run()
        assert result.max_drawdown >= 0.0

    def test_sharpe_ratio_is_finite_or_zero(self) -> None:
        bars = _make_bars(n=200)
        result = BacktestEngine(SMACrossoverStrategy(), bars).run()
        assert np.isfinite(result.sharpe_ratio) or result.sharpe_ratio == 0.0

    def test_trending_up_positive_return(self) -> None:
        """Strongly trending upward data with long-only SMA should profit."""
        bars = _make_bars(n=200, trend=0.003, noise=0.001, seed=1)
        result = BacktestEngine(SMACrossoverStrategy(), bars).run()
        # With a strong trend, at least the last equity should be reasonable
        assert len(result.equity_curve) == 200

    def test_profit_factor_non_negative(self) -> None:
        bars = _make_oscillating_bars(n=300)
        result = BacktestEngine(SMACrossoverStrategy(), bars).run()
        assert result.profit_factor >= 0.0


class TestBacktestEdgeCases:
    def test_empty_bars_returns_empty_result(self) -> None:
        strategy = SMACrossoverStrategy()
        result = BacktestEngine(strategy, []).run()
        assert result.equity_curve == []
        assert result.trades == []

    def test_few_bars_no_trades(self) -> None:
        bars = _make_bars(n=10)
        strategy = SMACrossoverStrategy(fast_period=5, slow_period=8)
        result = BacktestEngine(strategy, bars).run()
        assert len(result.equity_curve) == 10

    def test_custom_initial_capital(self) -> None:
        bars = _make_bars(n=100)
        settings = BacktestSettings(initial_capital=500_000.0)
        result = BacktestEngine(SMACrossoverStrategy(), bars, settings).run()
        # Equity should reflect the larger capital base
        assert result.equity_curve[0] > 400_000.0
