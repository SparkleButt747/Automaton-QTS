"""Orchestrate the news-reactive Optuna sweep: bank -> warm cache -> study ->
best params -> held-out real-day validity check.

The real run uses the live LlamaCpp classifier (cache warmed once). Tests inject a
stub classifier so CI needs no LLM.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna

from qts.nautilus.config import VenueConfig
from qts.optimisation.episode_bank import CouplingRanges, generate_episode_bank
from qts.optimisation.news_objective import NewsObjectiveContext, make_news_objective
from qts.optimisation.tuner import TunerConfig, build_study

if TYPE_CHECKING:
    from qts.config import RiskLimits
    from qts.data.real_episode import RealEpisode
    from qts.world.scenario import ScenarioConfig

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path("data/news_cache")


@dataclass
class RealDayVerdict:
    trades: int
    strat_return: float
    hold_return: float
    excess: float
    passed: bool


@dataclass
class NewsSweepResult:
    study: optuna.Study
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    n_completed: int = 0
    n_pruned: int = 0
    real_day: RealDayVerdict | None = None


def run_news_sweep(
    base_scenario: ScenarioConfig,
    risk_limits: RiskLimits,
    n_episodes: int = 50,
    n_trials: int = 150,
    seed: int = 42,
    classifier: Any | None = None,
    real_episode: RealEpisode | None = None,
    venue_config: VenueConfig | None = None,
    coupling_ranges: CouplingRanges | None = None,
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> NewsSweepResult:
    """Run the sweep. If `classifier` is None, build the live LlamaCpp classifier and
    warm its cache for every episode's text events before tuning."""
    vc = venue_config or VenueConfig()
    ranges = coupling_ranges or CouplingRanges()
    bank = generate_episode_bank(n_episodes, seed, base_scenario, ranges)

    if classifier is None:
        from qts.macro.news_classifier import NewsClassifier  # noqa: PLC0415
        from qts.oversight.llm_client import create_llm_client  # noqa: PLC0415

        llm = create_llm_client(backend="llamacpp")
        classifier = NewsClassifier(llm_client=llm, cache_dir=cache_dir)
        all_text = [e for ep in bank for e in ep.text_events]
        asyncio.run(classifier.warm_cache_for(all_text))

    ctx = NewsObjectiveContext(
        episodes=bank, classifier=classifier, risk_limits=risk_limits, venue_config=vc
    )
    cfg = TunerConfig(n_trials=n_trials, study_name="qts_news_sweep", direction="maximize")
    study = build_study(cfg)
    study.optimize(
        make_news_objective(ctx), n_trials=n_trials, catch=(Exception,), gc_after_trial=True
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    result = NewsSweepResult(study=study, n_completed=len(completed), n_pruned=len(pruned))
    if completed:
        result.best_params = study.best_params
        result.best_score = study.best_value

    if real_episode is not None and result.best_params:
        result.real_day = _evaluate_real_day(
            result.best_params, real_episode, classifier, risk_limits, vc
        )
        logger.info("Real-day verdict: %s", result.real_day)

    return result


def _evaluate_real_day(
    best_params: dict[str, Any],
    episode: RealEpisode,
    classifier: Any,
    risk_limits: RiskLimits,
    venue_config: VenueConfig,
) -> RealDayVerdict:
    """Apply best params to the held-out real day. Pass = trades fire AND beat hold
    AND positive return (on a dovish/up day, beating hold while profitable => long)."""
    from qts.nautilus.real_runner import run_real_backtest  # noqa: PLC0415
    from qts.optimisation.news_objective import _build_inner_params, _hold_return  # noqa: PLC0415
    from qts.strategies.momentum import MomentumStrategy  # noqa: PLC0415
    from qts.strategies.news_reactive import NewsReactiveMomentum  # noqa: PLC0415

    strategy = NewsReactiveMomentum(
        inner=MomentumStrategy(
            params=_build_inner_params(float(best_params["entry_threshold"])),
            risk_limits=risk_limits,
        ),
        classifier=classifier,
        belief_half_life=timedelta(minutes=float(best_params["belief_half_life_minutes"])),
        news_signal_weight=float(best_params["news_signal_weight"]),
    )
    result = run_real_backtest(episode, strategy, venue_config=venue_config, log_level="ERROR")
    hold = _hold_return(episode.terrain)
    excess = result.total_return - hold
    passed = result.total_trades > 0 and excess > 0 and result.total_return > 0
    return RealDayVerdict(
        trades=result.total_trades,
        strat_return=result.total_return,
        hold_return=hold,
        excess=excess,
        passed=passed,
    )
