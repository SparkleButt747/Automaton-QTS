"""Optuna-based strategy parameter tuner — the optimisation loop.

Mirrors racing lab's optimize/runner.py:
- TPE sampler for Bayesian parameter search
- MedianPruner for early stopping of bad trials
- Multi-worker support via multiprocessing
- Train across diverse terrains to prevent overfitting
"""

from __future__ import annotations

import logging
import multiprocessing
from dataclasses import dataclass, field
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

import optuna

from qts.config import RiskLimits
from qts.models.terrain import MarketTerrain
from qts.nautilus.config import VenueConfig
from qts.optimisation.objective import ObjectiveContext, make_objective

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TunerConfig:
    """Configuration for the strategy parameter tuner.

    Analogous to racing lab's OptimCfg.
    """

    n_trials: int = 100
    n_workers: int = 1
    sampler_seed: int = 42
    pruner_n_startup_trials: int = 10
    pruner_n_warmup_steps: int = 3
    study_name: str = "qts_strategy_tuning"
    storage_path: Path | None = None  # SQLite or journal path
    primary_metric: str = "sharpe_ratio"
    direction: str = "maximize"


@dataclass
class TunerResult:
    """Results from a tuning run."""

    study: optuna.Study
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    n_completed: int = 0
    n_pruned: int = 0


def run_strategy_study(
    train_terrains: list[MarketTerrain],
    risk_limits: RiskLimits,
    config: TunerConfig | None = None,
    venue_config: VenueConfig | None = None,
    strategy_factory: Any | None = None,
    param_sampler: Any | None = None,
) -> TunerResult:
    """Run Optuna strategy parameter tuning across diverse terrains.

    This is the main entry point for the optimisation loop. Mirrors
    racing lab's run_controller_study().

    Args:
        train_terrains: Terrains to train on (diverse regimes).
        risk_limits: Risk limits applied during evaluation.
        config: Tuner configuration.
        venue_config: Nautilus venue configuration.
        strategy_factory: Optional callable(params) -> Strategy.
        param_sampler: Optional callable(trial) -> StrategyParams.

    Returns:
        TunerResult with the completed study and best parameters.
    """
    cfg = config or TunerConfig()
    vc = venue_config or VenueConfig()

    if not train_terrains:
        raise ValueError("At least one training terrain is required")

    # Build objective context
    context = ObjectiveContext(
        train_terrains=train_terrains,
        risk_limits=risk_limits,
        venue_config=vc,
        strategy_factory=strategy_factory,
        param_sampler=param_sampler,
        primary_metric=cfg.primary_metric,
    )

    # Create sampler and pruner (mirrors racing lab's storage.py)
    sampler = optuna.samplers.TPESampler(seed=cfg.sampler_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=cfg.pruner_n_startup_trials,
        n_warmup_steps=cfg.pruner_n_warmup_steps,
    )

    # Create or load study
    storage = None
    if cfg.storage_path is not None:
        cfg.storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{cfg.storage_path}"

    study = optuna.create_study(
        study_name=cfg.study_name,
        direction=cfg.direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    # Build the objective
    objective = make_objective(context)

    # Run optimisation
    if cfg.n_workers <= 1:
        # Single-worker mode
        study.optimize(
            objective,
            n_trials=cfg.n_trials,
            catch=(Exception,),
            gc_after_trial=True,
        )
    else:
        # Multi-worker mode via multiprocessing
        _run_parallel(study, objective, cfg)

    # Collect results
    result = TunerResult(study=study)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    result.n_completed = len(completed)
    result.n_pruned = len(pruned)

    if completed:
        result.best_params = study.best_params
        result.best_score = study.best_value
        logger.info(
            "Tuning complete: %d completed, %d pruned, best_score=%.4f",
            result.n_completed,
            result.n_pruned,
            result.best_score,
        )
        logger.info("Best params: %s", result.best_params)
    else:
        logger.warning("No trials completed successfully")

    return result


def _run_parallel(
    study: optuna.Study,
    objective: Any,
    config: TunerConfig,
) -> None:
    """Run optimisation with multiple workers."""
    trials_per_worker = _split_trials(config.n_trials, config.n_workers)

    def _worker(n_trials: int) -> None:
        study.optimize(
            objective,
            n_trials=n_trials,
            catch=(Exception,),
            gc_after_trial=True,
        )

    ctx = multiprocessing.get_context("spawn")
    processes: list[BaseProcess] = []

    for worker_trials in trials_per_worker:
        p = ctx.Process(target=_worker, args=(worker_trials,))
        processes.append(p)
        p.start()

    for proc in processes:
        proc.join()


def _split_trials(total: int, n_workers: int) -> list[int]:
    """Split trials evenly across workers."""
    base = total // n_workers
    remainder = total % n_workers
    return [base + (1 if i < remainder else 0) for i in range(n_workers)]
