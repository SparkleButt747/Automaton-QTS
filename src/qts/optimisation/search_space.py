"""Strategy parameter search spaces for Optuna.

Maps StrategyParams fields to Optuna trial suggestion calls.
Analogous to racing lab's optimize/space.py (sample_mpcc_cfg, sample_rrhc_cfg).
"""

from __future__ import annotations

from typing import Any

import optuna

from qts.config import SignalWeights, SentimentFusionWeights, StrategyParams


def sample_momentum_params(trial: optuna.Trial) -> StrategyParams:
    """Sample MomentumStrategy parameters from Optuna search space.

    The search space covers:
    - Signal weights (normalised to sum=1)
    - Entry/exit thresholds
    - Max hold bars
    - Sentiment fusion weights (normalised to sum=1)
    """
    # Signal weights — sample raw, then normalise
    raw_rsi = trial.suggest_float("w_rsi", 0.05, 0.5)
    raw_macd = trial.suggest_float("w_macd", 0.05, 0.5)
    raw_bb = trial.suggest_float("w_bb", 0.05, 0.5)
    raw_mom = trial.suggest_float("w_mom", 0.05, 0.5)
    raw_sentiment = trial.suggest_float("w_sentiment", 0.0, 0.4)

    total = raw_rsi + raw_macd + raw_bb + raw_mom + raw_sentiment
    weights = SignalWeights(
        w_rsi=raw_rsi / total,
        w_macd=raw_macd / total,
        w_bb=raw_bb / total,
        w_mom=raw_mom / total,
        w_sentiment=raw_sentiment / total,
    )

    # Thresholds
    entry_threshold = trial.suggest_float("entry_threshold", 0.05, 0.5)
    exit_threshold = trial.suggest_float("exit_threshold", -0.3, entry_threshold - 0.01)
    max_hold_bars = trial.suggest_int("max_hold_bars", 5, 100)

    # Sentiment fusion weights
    raw_news = trial.suggest_float("fusion_news", 0.1, 0.7)
    raw_social = trial.suggest_float("fusion_social", 0.1, 0.5)
    raw_geo = trial.suggest_float("fusion_geo", 0.1, 0.5)
    fusion_total = raw_news + raw_social + raw_geo
    fusion_weights = SentimentFusionWeights(
        news=raw_news / fusion_total,
        social=raw_social / fusion_total,
        geopolitical=raw_geo / fusion_total,
    )

    return StrategyParams(
        version="optuna",
        weights=weights,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        max_hold_bars=max_hold_bars,
        sentiment_fusion_weights=fusion_weights,
    )


def sample_mean_reversion_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample MeanReversionStrategy parameters from Optuna search space."""
    bb_oversold = trial.suggest_float("bb_oversold", 0.05, 0.3)
    bb_overbought = trial.suggest_float("bb_overbought", 0.7, 0.95)
    bb_target = trial.suggest_float("bb_target", 0.35, 0.65)
    sentiment_threshold = trial.suggest_float("sentiment_confirm_threshold", 0.0, 0.5)
    max_hold_bars = trial.suggest_int("max_hold_bars", 5, 50)
    position_size = trial.suggest_float("position_size", 0.01, 0.1, log=True)

    return {
        "bb_oversold": bb_oversold,
        "bb_overbought": bb_overbought,
        "bb_target": bb_target,
        "sentiment_confirm_threshold": sentiment_threshold,
        "max_hold_bars": max_hold_bars,
        "position_size": position_size,
    }
