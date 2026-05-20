"""Multi-terrain objective function for Optuna optimisation.

Mirrors the racing lab's optimize/objective.py:
- ObjectiveContext holds the evaluation configuration
- make_objective() returns a closure that Optuna calls per trial
- Each trial runs the strategy across ALL training terrains
- The objective is the mean Sharpe ratio (must work in ALL regimes)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import optuna

from qts.config import RiskLimits, StrategyParams
from qts.models.terrain import MarketTerrain
from qts.nautilus.config import BacktestResult, VenueConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ObjectiveContext:
    """Context for the multi-terrain objective function.

    Analogous to racing lab's ObjectiveContext — holds everything needed
    to evaluate a strategy parameter set across diverse market conditions.
    """

    train_terrains: list[MarketTerrain]
    risk_limits: RiskLimits
    venue_config: VenueConfig = field(default_factory=VenueConfig)
    strategy_factory: Callable[..., Any] | None = None
    param_sampler: Callable[[optuna.Trial], StrategyParams] | None = None

    # Scoring configuration
    primary_metric: str = "sharpe_ratio"
    min_sharpe_threshold: float = -2.0  # prune trials below this
    min_trades_per_terrain: int = 3


def make_objective(context: ObjectiveContext) -> Callable[[optuna.Trial], float]:
    """Create an Optuna objective function for multi-terrain strategy evaluation.

    Returns a closure that:
    1. Samples strategy parameters from the search space
    2. For each training terrain: runs a Nautilus backtest, collects Sharpe
    3. Reports intermediate values for pruning
    4. Returns mean Sharpe across all terrains

    This is the direct equivalent of racing lab's make_objective() — training
    across multiple terrains prevents overfitting to one market condition.
    """

    def _objective(trial: optuna.Trial) -> float:
        from qts.nautilus.runner import run_terrain_backtest  # noqa: PLC0415
        from qts.optimisation.search_space import (  # noqa: PLC0415
            sample_momentum_params,
        )
        from qts.strategies.momentum import MomentumStrategy  # noqa: PLC0415

        # 1. Sample parameters
        if context.param_sampler is not None:
            params = context.param_sampler(trial)
        else:
            params = sample_momentum_params(trial)

        # 2. Create strategy
        if context.strategy_factory is not None:
            strategy = context.strategy_factory(params)
        else:
            strategy = MomentumStrategy(
                params=params,
                risk_limits=context.risk_limits,
            )

        # 3. Evaluate across all training terrains
        scores: list[float] = []

        for idx, terrain in enumerate(context.train_terrains):
            try:
                result = run_terrain_backtest(
                    terrain=terrain,
                    strategy=strategy,
                    venue_config=context.venue_config,
                    log_level="ERROR",
                )

                score = _extract_score(result, context.primary_metric)
                scores.append(score)

                # Report intermediate value for pruning
                running_mean = float(np.mean(scores))
                trial.report(running_mean, step=idx)

                if trial.should_prune():
                    raise optuna.TrialPruned()

                # Store per-terrain scores as user attributes
                trial.set_user_attr(f"score_{terrain.name}", score)
                trial.set_user_attr(f"trades_{terrain.name}", result.total_trades)

            except optuna.TrialPruned:
                raise
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Backtest failed on terrain '%s' for trial %d",
                    terrain.name,
                    trial.number,
                )
                scores.append(context.min_sharpe_threshold)

        if not scores:
            return context.min_sharpe_threshold

        # 4. Aggregate: mean score across all terrains
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores)) if len(scores) > 1 else 0.0

        trial.set_user_attr("score_mean", mean_score)
        trial.set_user_attr("score_std", std_score)
        trial.set_user_attr("score_min", float(np.min(scores)))
        trial.set_user_attr("score_max", float(np.max(scores)))
        trial.set_user_attr("n_terrains_evaluated", len(scores))

        logger.info(
            "Trial %d: mean_sharpe=%.4f (std=%.4f, min=%.4f, max=%.4f)",
            trial.number,
            mean_score,
            std_score,
            float(np.min(scores)),
            float(np.max(scores)),
        )

        return mean_score

    return _objective


def _extract_score(result: BacktestResult, metric: str) -> float:
    """Extract the scoring metric from a backtest result."""
    return float(getattr(result, metric, 0.0))
