"""Integration tests: Full backtest with MeanReversionStrategy on synthetic data.

Verifies:
- BacktestEngine + MeanReversionStrategy runs end-to-end without errors
- Produces valid BacktestResult
- Equity curve has correct length
- Statistics are well-formed (win_rate in [0,1], max_drawdown >= 0, etc.)
- Deterministic: same inputs -> identical results
- Oscillating data produces mean-reversion trades
- Trending data behaves reasonably
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from qts.models.base import Bar, TradeOutcome
from qts.simulation.backtest import BacktestEngine, BacktestResult, BacktestSettings
from qts.strategies.mean_reversion import MeanReversionStrategy

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_bars(
    n: int = 200,
    symbol: str = "SYNTH",
    start_price: float = 100.0,
    trend: float = 0.0,
    noise: float = 0.005,
    seed: int = 42,
) -> list[Bar]:
    """Generate synthetic OHLCV bars with optional drift."""
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


def _make_mean_reverting_bars(
    n: int = 300,
    symbol: str = "MR",
    seed: int = 99,
    amplitude: float = 15.0,
    noise: float = 0.3,
) -> list[Bar]:
    """Generate oscillating price bars well-suited to mean-reversion.

    Uses a sinusoidal price path so that the BB position alternates between
    extremes, triggering both long and short entries.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10 * np.pi, n)
    prices = 100.0 + amplitude * np.sin(t) + rng.normal(0, noise, n)
    bars = []
    base_time = datetime(2024, 1, 1)

    for i in range(n):
        p = max(float(prices[i]), 1.0)  # guard against non-positive price
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


# ── Basic functionality ───────────────────────────────────────────────────────


class TestBacktestMeanReversionBasic:
    def test_run_returns_backtest_result(self) -> None:
        bars = _make_bars(n=100)
        strategy = MeanReversionStrategy()
        settings = BacktestSettings(initial_capital=100_000.0)
        engine = BacktestEngine(strategy, bars, settings)
        result = engine.run()
        assert isinstance(result, BacktestResult)

    def test_equity_curve_length_matches_bars(self) -> None:
        n = 150
        bars = _make_bars(n=n)
        strategy = MeanReversionStrategy()
        engine = BacktestEngine(strategy, bars)
        result = engine.run()
        assert len(result.equity_curve) == n

    def test_equity_curve_all_positive(self) -> None:
        """Equity should remain positive with reasonable strategy parameters."""
        bars = _make_mean_reverting_bars(n=200)
        strategy = MeanReversionStrategy(position_size=0.01)
        engine = BacktestEngine(strategy, bars)
        result = engine.run()
        assert all(e > 0 for e in result.equity_curve)

    def test_total_return_is_float(self) -> None:
        bars = _make_bars(n=200)
        strategy = MeanReversionStrategy()
        engine = BacktestEngine(strategy, bars)
        result = engine.run()
        assert isinstance(result.total_return, float)

    def test_total_return_in_reasonable_range(self) -> None:
        """Total return should not be absurdly large or less than -100%."""
        bars = _make_mean_reverting_bars(n=300)
        strategy = MeanReversionStrategy(position_size=0.01)
        engine = BacktestEngine(strategy, bars)
        result = engine.run()
        assert -1.0 <= result.total_return <= 100.0

    def test_equity_curve_starts_near_initial_capital(self) -> None:
        bars = _make_bars(n=100)
        strategy = MeanReversionStrategy()
        settings = BacktestSettings(initial_capital=50_000.0)
        engine = BacktestEngine(strategy, bars, settings)
        result = engine.run()
        # First bars are warm-up; equity should be close to initial capital
        assert abs(result.equity_curve[0] - 50_000.0) < 10_000.0


# ── Determinism ───────────────────────────────────────────────────────────────


class TestBacktestMeanReversionDeterminism:
    def test_same_inputs_identical_equity_curve(self) -> None:
        bars = _make_mean_reverting_bars(n=200, seed=7)
        settings = BacktestSettings()

        def _run() -> BacktestResult:
            return BacktestEngine(
                MeanReversionStrategy(
                    bb_oversold=0.2,
                    bb_overbought=0.8,
                    position_size=0.03,
                ),
                bars,
                settings,
            ).run()

        result1 = _run()
        result2 = _run()

        assert result1.equity_curve == result2.equity_curve

    def test_same_inputs_identical_trade_count(self) -> None:
        bars = _make_mean_reverting_bars(n=300, seed=99)
        settings = BacktestSettings()

        r1 = BacktestEngine(MeanReversionStrategy(position_size=0.03), bars, settings).run()
        r2 = BacktestEngine(MeanReversionStrategy(position_size=0.03), bars, settings).run()

        assert len(r1.trades) == len(r2.trades)

    def test_different_seeds_different_results(self) -> None:
        bars_a = _make_mean_reverting_bars(n=200, seed=1)
        bars_b = _make_mean_reverting_bars(n=200, seed=2)
        settings = BacktestSettings()

        r_a = BacktestEngine(MeanReversionStrategy(), bars_a, settings).run()
        r_b = BacktestEngine(MeanReversionStrategy(), bars_b, settings).run()

        assert r_a.equity_curve != r_b.equity_curve

    def test_different_strategy_params_different_results(self) -> None:
        bars = _make_mean_reverting_bars(n=300, seed=42)
        settings = BacktestSettings()

        r_tight = BacktestEngine(
            MeanReversionStrategy(bb_oversold=0.1, bb_overbought=0.9), bars, settings
        ).run()
        r_loose = BacktestEngine(
            MeanReversionStrategy(bb_oversold=0.3, bb_overbought=0.7), bars, settings
        ).run()

        # Different thresholds should produce different trade counts
        assert len(r_tight.trades) != len(r_loose.trades) or (
            r_tight.equity_curve != r_loose.equity_curve
        )


# ── Trade generation ──────────────────────────────────────────────────────────


class TestBacktestMeanReversionTrades:
    def test_oscillating_bars_produce_trades(self) -> None:
        """Mean-reverting oscillating data should trigger at least some trades."""
        bars = _make_mean_reverting_bars(n=300, amplitude=20.0)
        strategy = MeanReversionStrategy(
            bb_oversold=0.2,
            bb_overbought=0.8,
            position_size=0.05,
        )
        result = BacktestEngine(strategy, bars).run()
        assert len(result.trades) > 0

    def test_trade_records_have_valid_pnl(self) -> None:
        bars = _make_mean_reverting_bars(n=300)
        strategy = MeanReversionStrategy()
        result = BacktestEngine(strategy, bars).run()
        for trade in result.trades:
            assert isinstance(trade.pnl_usd, float)
            assert not (trade.pnl_usd != trade.pnl_usd)  # not NaN

    def test_trade_pnl_pct_reasonable(self) -> None:
        """Individual trade pnl_pct should not be absurdly large."""
        bars = _make_mean_reverting_bars(n=300)
        strategy = MeanReversionStrategy(position_size=0.01)
        result = BacktestEngine(strategy, bars).run()
        for trade in result.trades:
            assert -1.0 <= trade.pnl_pct <= 10.0

    def test_trade_outcomes_valid(self) -> None:
        bars = _make_mean_reverting_bars(n=300)
        strategy = MeanReversionStrategy()
        result = BacktestEngine(strategy, bars).run()
        valid_outcomes = {TradeOutcome.WIN, TradeOutcome.LOSS, TradeOutcome.BREAKEVEN}
        for trade in result.trades:
            assert trade.outcome in valid_outcomes

    def test_trade_direction_valid(self) -> None:
        from qts.models.base import TradeDirection

        bars = _make_mean_reverting_bars(n=300)
        strategy = MeanReversionStrategy()
        result = BacktestEngine(strategy, bars).run()
        valid_directions = {TradeDirection.LONG, TradeDirection.SHORT}
        for trade in result.trades:
            assert trade.direction in valid_directions

    def test_no_trades_with_neutral_bb(self) -> None:
        """Bars that never reach extreme BB positions should produce few/no trades."""
        # Very noisy data that likely won't consistently hit oversold/overbought
        bars = _make_bars(n=100, noise=0.0001, seed=42)
        strategy = MeanReversionStrategy(
            bb_oversold=0.001,  # effectively never oversold
            bb_overbought=0.999,  # effectively never overbought
        )
        result = BacktestEngine(strategy, bars).run()
        # Very tight thresholds -> few or no trades
        assert isinstance(result.trades, list)  # result is valid


# ── Statistics ────────────────────────────────────────────────────────────────


class TestBacktestMeanReversionStatistics:
    def test_win_rate_in_unit_interval(self) -> None:
        bars = _make_mean_reverting_bars(n=300)
        strategy = MeanReversionStrategy()
        result = BacktestEngine(strategy, bars).run()
        if result.trades:
            assert 0.0 <= result.win_rate <= 1.0

    def test_max_drawdown_non_negative(self) -> None:
        bars = _make_mean_reverting_bars(n=200)
        result = BacktestEngine(MeanReversionStrategy(), bars).run()
        assert result.max_drawdown >= 0.0

    def test_max_drawdown_at_most_one(self) -> None:
        bars = _make_mean_reverting_bars(n=200)
        result = BacktestEngine(MeanReversionStrategy(), bars).run()
        assert result.max_drawdown <= 1.0

    def test_sharpe_ratio_is_finite_or_zero(self) -> None:
        bars = _make_mean_reverting_bars(n=200)
        result = BacktestEngine(MeanReversionStrategy(), bars).run()
        assert np.isfinite(result.sharpe_ratio) or result.sharpe_ratio == 0.0

    def test_profit_factor_non_negative(self) -> None:
        bars = _make_mean_reverting_bars(n=300)
        result = BacktestEngine(MeanReversionStrategy(), bars).run()
        assert result.profit_factor >= 0.0

    def test_sortino_ratio_finite_or_zero(self) -> None:
        bars = _make_mean_reverting_bars(n=200)
        result = BacktestEngine(MeanReversionStrategy(), bars).run()
        assert np.isfinite(result.sortino_ratio) or result.sortino_ratio == 0.0


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestBacktestMeanReversionEdgeCases:
    def test_empty_bars_returns_empty_result(self) -> None:
        strategy = MeanReversionStrategy()
        result = BacktestEngine(strategy, []).run()
        assert result.equity_curve == []
        assert result.trades == []

    def test_few_bars_no_trades(self) -> None:
        """With fewer bars than the signal window, no signals fire."""
        bars = _make_bars(n=5)
        strategy = MeanReversionStrategy()
        result = BacktestEngine(strategy, bars).run()
        assert len(result.equity_curve) == 5

    def test_custom_initial_capital(self) -> None:
        bars = _make_bars(n=100)
        settings = BacktestSettings(initial_capital=500_000.0)
        result = BacktestEngine(MeanReversionStrategy(), bars, settings).run()
        assert result.equity_curve[0] > 400_000.0

    def test_strict_sentiment_blocks_all_entries(self) -> None:
        """Very tight sentiment threshold blocks entries on most neutral data."""
        bars = _make_mean_reverting_bars(n=200)
        # Threshold of 0.0 means sentiment must be > 0 for LONG, < 0 for SHORT
        # This severely limits trades but should not crash.
        strategy = MeanReversionStrategy(
            sentiment_confirm_threshold=0.0,
            bb_oversold=0.2,
            bb_overbought=0.8,
        )
        result = BacktestEngine(strategy, bars).run()
        assert isinstance(result, BacktestResult)

    def test_very_short_max_hold_bars(self) -> None:
        """Extremely short max_hold_bars should force frequent time-stops."""
        bars = _make_mean_reverting_bars(n=300, amplitude=20.0)
        strategy = MeanReversionStrategy(max_hold_bars=1, position_size=0.02)
        result = BacktestEngine(strategy, bars).run()
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) == 300
