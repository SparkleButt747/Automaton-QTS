"""CVaR-of-excess-vs-hold objective for the news-reactive Optuna sweep.

Each trial samples (news_signal_weight, belief_half_life, entry_threshold), builds
a NewsReactiveMomentum (frozen momentum base), backtests it over every episode in
a frozen bank via the v2.1 NewsDataPoint custom-data path, and scores the
25th-percentile of per-episode (strategy_return - buy_and_hold_return). The lower
quantile rewards robustness and self-guards against a do-nothing strategy (it loses
in rally episodes).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna

from qts.config import SentimentFusionWeights, SignalWeights, StrategyParams
from qts.nautilus.config import VenueConfig

if TYPE_CHECKING:
    from qts.config import RiskLimits
    from qts.models.terrain import MarketTerrain
    from qts.world.episode import SimulatedEpisode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsObjectiveContext:
    """Everything the objective needs. `classifier` cache must be pre-warmed."""

    episodes: list[SimulatedEpisode]
    classifier: Any  # NewsClassifier or a stub exposing .classify(event) -> NewsSignal
    risk_limits: RiskLimits
    venue_config: VenueConfig = field(default_factory=VenueConfig)
    cvar_quantile: float = 0.25


def _build_inner_params(entry_threshold: float) -> StrategyParams:
    """Frozen momentum base; exit derived as 40% of entry so exit < entry always."""
    return StrategyParams(
        version="news-sweep",
        weights=SignalWeights(w_rsi=0.20, w_macd=0.20, w_bb=0.15, w_mom=0.15, w_sentiment=0.30),
        entry_threshold=entry_threshold,
        exit_threshold=round(entry_threshold * 0.4, 4),
        max_hold_bars=24,
        sentiment_fusion_weights=SentimentFusionWeights(news=0.5, social=0.3, geopolitical=0.2),
    )


def _hold_return(terrain: MarketTerrain) -> float:
    bars = terrain.bars
    if len(bars) < 2 or bars[0].close == 0:
        return 0.0
    return (bars[-1].close - bars[0].close) / bars[0].close


def _cvar_of_excess(excesses: list[float], quantile: float) -> float:
    if not excesses:
        return 0.0
    return float(np.quantile(excesses, quantile))


def make_news_objective(ctx: NewsObjectiveContext) -> Callable[[optuna.Trial], float]:
    """Build the Optuna objective closure (maximise the lower-quantile excess)."""

    def _objective(trial: optuna.Trial) -> float:
        from nautilus_trader.model.data import CustomData, DataType  # noqa: PLC0415

        from qts.nautilus.converters import text_event_to_news_data  # noqa: PLC0415
        from qts.nautilus.news_data import NewsDataPoint  # noqa: PLC0415
        from qts.nautilus.runner import run_terrain_backtest  # noqa: PLC0415
        from qts.optimisation.search_space import sample_news_params  # noqa: PLC0415
        from qts.strategies.momentum import MomentumStrategy  # noqa: PLC0415
        from qts.strategies.news_reactive import NewsReactiveMomentum  # noqa: PLC0415

        params = sample_news_params(trial)
        data_type = DataType(NewsDataPoint)
        excesses: list[float] = []

        for idx, ep in enumerate(ctx.episodes):
            strategy = NewsReactiveMomentum(
                inner=MomentumStrategy(
                    params=_build_inner_params(params.entry_threshold),
                    risk_limits=ctx.risk_limits,
                ),
                classifier=ctx.classifier,
                belief_half_life=timedelta(minutes=params.belief_half_life_minutes),
                news_signal_weight=params.news_signal_weight,
            )
            custom_data = [
                CustomData(data_type=data_type, data=text_event_to_news_data(e))
                for e in sorted(ep.text_events, key=lambda e: e.timestamp)
            ]
            try:
                result = run_terrain_backtest(
                    ep.terrain,
                    strategy,
                    venue_config=ctx.venue_config,
                    log_level="ERROR",
                    custom_data=custom_data,
                )
                excess = result.total_return - _hold_return(ep.terrain)
            except Exception:  # noqa: BLE001
                logger.exception("news_objective: backtest failed on episode %d", idx)
                excess = -1.0
            excesses.append(excess)

            running = _cvar_of_excess(excesses, ctx.cvar_quantile)
            trial.report(running, step=idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        score = _cvar_of_excess(excesses, ctx.cvar_quantile)
        trial.set_user_attr("cvar_excess", score)
        trial.set_user_attr("mean_excess", float(np.mean(excesses)) if excesses else 0.0)
        trial.set_user_attr("min_excess", float(np.min(excesses)) if excesses else 0.0)
        return score

    return _objective
