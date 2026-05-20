"""Unit tests for qts.optimisation.tuner.

Covers:
- T-OTUN-1: run_strategy_study raises ValueError when train_terrains is empty
- T-OTUN-2: run_strategy_study with a no-op strategy completes 2 trials and returns a study
- T-OTUN-3: TunerResult collects best_params / best_score / n_completed
- T-OTUN-4: _split_trials distributes evenly with remainder
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import optuna
import pytest

from qts.config import RiskLimits
from qts.models.base import Bar, Catalyst, LiquidityLevel, SentimentLevel, Trend, VolLevel
from qts.models.terrain import MacroRegime
from qts.optimisation.search_space import sample_momentum_params
from qts.optimisation.tuner import TunerConfig, TunerResult, _split_trials, run_strategy_study
from qts.terrain.builder import TerrainBuilder


def _make_risk_limits() -> RiskLimits:
    return RiskLimits(
        max_daily_drawdown_pct=0.02,
        max_position_size_pct=0.05,
        max_open_positions=5,
        circuit_breaker_cooldown_seconds=3600,
        sentiment_signal_max_scalar=2.0,
    )


def _make_tiny_terrain() -> object:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    bars = [
        Bar(
            symbol="BTCUSDT",
            timestamp=start + timedelta(hours=i),
            open=30000.0 + i,
            high=30001.0 + i,
            low=29999.0 + i,
            close=30000.5 + i,
            volume=1000.0,
        )
        for i in range(60)
    ]
    regime = MacroRegime(
        trend=Trend.BULL,
        volatility=VolLevel.LOW,
        liquidity=LiquidityLevel.ABUNDANT,
        sentiment=SentimentLevel.NEUTRAL,
        catalyst=Catalyst.NONE,
        expected_drift=0.0001,
        expected_vol=0.005,
        correlation_regime=0.4,
        scenario_description="tiny tuner terrain",
    )
    return (
        TerrainBuilder().set_identity("tiny", "BTCUSDT").set_regime(regime).set_bars(bars).build()
    )


def _noop_strategy_factory(params: object) -> object:
    """Strategy that never trades — returns no orders, no fills. Sharpe is 0.0."""

    class _Noop:
        def __init__(self) -> None:
            self.params = params

        def on_bar(self, *_a: object, **_k: object) -> list[object]:
            return []

        def on_fill(self, *_a: object, **_k: object) -> None:
            pass

    return _Noop()


class TestRunStrategyStudy:
    def test_empty_terrains_raises(self) -> None:  # T-OTUN-1
        with pytest.raises(ValueError, match="At least one training terrain"):
            run_strategy_study(train_terrains=[], risk_limits=_make_risk_limits())

    def test_completes_small_run(self) -> None:  # T-OTUN-2, T-OTUN-3
        """Drive a 2-trial study with a no-op strategy. Completes cleanly."""
        cfg = TunerConfig(
            n_trials=2,
            n_workers=1,
            sampler_seed=42,
            pruner_n_startup_trials=10,
            study_name="test_tuner_run",
        )
        result = run_strategy_study(
            train_terrains=[_make_tiny_terrain()],
            risk_limits=_make_risk_limits(),
            config=cfg,
            strategy_factory=_noop_strategy_factory,
            param_sampler=sample_momentum_params,
        )

        assert isinstance(result, TunerResult)
        assert isinstance(result.study, optuna.Study)
        assert result.n_completed + result.n_pruned == 2
        # With a noop strategy, all trials should complete (no errors, no prune).
        assert result.n_completed == 2
        # best_params should be a non-empty dict (sampler proposed something).
        assert len(result.best_params) > 0


class TestSplitTrials:
    def test_even_split(self) -> None:  # T-OTUN-4a
        assert _split_trials(10, 2) == [5, 5]

    def test_split_with_remainder(self) -> None:  # T-OTUN-4b
        assert _split_trials(7, 3) == [3, 2, 2]

    def test_single_worker(self) -> None:  # T-OTUN-4c
        assert _split_trials(5, 1) == [5]
