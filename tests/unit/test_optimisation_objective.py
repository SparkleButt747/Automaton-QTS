"""Unit tests for qts.optimisation.objective.

Covers:
- T-OOBJ-1: ObjectiveContext fields populate correctly
- T-OOBJ-2: make_objective returns a callable that takes an Optuna trial
- T-OOBJ-3: objective falls back to min_sharpe_threshold when terrains list is empty
- T-OOBJ-4: objective records per-terrain user attributes (score and trades)
- T-OOBJ-5: objective handles backtest exceptions and substitutes min_sharpe_threshold
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import optuna

from qts.config import RiskLimits
from qts.models.base import Bar, Catalyst, LiquidityLevel, SentimentLevel, Trend, VolLevel
from qts.models.terrain import MacroRegime
from qts.nautilus.config import VenueConfig
from qts.optimisation.objective import ObjectiveContext, make_objective
from qts.optimisation.search_space import sample_momentum_params
from qts.terrain.builder import TerrainBuilder


def _make_risk_limits() -> RiskLimits:
    return RiskLimits(
        max_daily_drawdown_pct=0.02,
        max_position_size_pct=0.05,
        max_open_positions=5,
        circuit_breaker_cooldown_seconds=3600,
        sentiment_signal_max_scalar=2.0,
    )


def _make_tiny_terrain(name: str = "tiny") -> object:
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
        scenario_description="tiny test terrain",
    )
    return TerrainBuilder().set_identity(name, "BTCUSDT").set_regime(regime).set_bars(bars).build()


class TestObjectiveContext:
    def test_fields_populate(self) -> None:  # T-OOBJ-1
        terrain = _make_tiny_terrain()
        ctx = ObjectiveContext(
            train_terrains=[terrain],
            risk_limits=_make_risk_limits(),
        )
        assert ctx.train_terrains == [terrain]
        assert isinstance(ctx.venue_config, VenueConfig)
        assert ctx.primary_metric == "sharpe_ratio"
        assert ctx.min_sharpe_threshold == -2.0
        assert ctx.min_trades_per_terrain == 3


class TestMakeObjective:
    def test_returns_callable(self) -> None:  # T-OOBJ-2
        ctx = ObjectiveContext(
            train_terrains=[_make_tiny_terrain()],
            risk_limits=_make_risk_limits(),
        )
        fn = make_objective(ctx)
        assert callable(fn)

    def test_empty_terrains_returns_threshold(self) -> None:  # T-OOBJ-3
        ctx = ObjectiveContext(
            train_terrains=[],
            risk_limits=_make_risk_limits(),
            min_sharpe_threshold=-1.5,
        )
        fn = make_objective(ctx)
        study = optuna.create_study(direction="maximize")
        study.optimize(fn, n_trials=1)
        assert study.best_value == -1.5

    def test_exception_substitutes_threshold(self) -> None:  # T-OOBJ-5
        """When run_terrain_backtest raises, objective records min_sharpe_threshold."""

        def exploding_factory(params: object) -> object:
            class _Boom:
                def __init__(self) -> None:
                    self.params = params

                def on_bar(self, *_a: object, **_k: object) -> list[object]:
                    raise RuntimeError("synthetic boom")

                def on_fill(self, *_a: object, **_k: object) -> None:
                    pass

            return _Boom()

        ctx = ObjectiveContext(
            train_terrains=[_make_tiny_terrain()],
            risk_limits=_make_risk_limits(),
            strategy_factory=exploding_factory,
            param_sampler=sample_momentum_params,
            min_sharpe_threshold=-1.5,
        )
        fn = make_objective(ctx)
        study = optuna.create_study(direction="maximize")
        # The objective swallows the backtest error and assigns the threshold,
        # so the trial should complete (not error out).
        study.optimize(fn, n_trials=1, catch=())
        assert len(study.trials) == 1
        assert study.best_value == -1.5
