"""Unit tests for qts.optimisation.search_space.

Covers:
- T-OSPC-1: sample_momentum_params returns a StrategyParams with normalised signal weights
- T-OSPC-2: sample_momentum_params returns sentiment_fusion_weights that sum to 1
- T-OSPC-3: entry/exit threshold relationship (exit < entry)
- T-OSPC-4: max_hold_bars stays in [5, 100]
- T-OSPC-5: sample_mean_reversion_params returns a dict with all expected keys
- T-OSPC-6: mean reversion bounds (bb_oversold < bb_overbought, position_size in range)
"""

from __future__ import annotations

import math

import optuna

from qts.config import SentimentFusionWeights, SignalWeights, StrategyParams
from qts.optimisation.search_space import (
    sample_mean_reversion_params,
    sample_momentum_params,
)


def _make_study() -> optuna.Study:
    """Build a deterministic in-memory study for sampling."""
    return optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        direction="maximize",
    )


class TestSampleMomentumParams:
    def test_returns_strategy_params(self) -> None:  # T-OSPC-1a
        study = _make_study()
        params: StrategyParams | None = None

        def obj(trial: optuna.Trial) -> float:
            nonlocal params
            params = sample_momentum_params(trial)
            return 0.0

        study.optimize(obj, n_trials=1)

        assert params is not None
        assert isinstance(params, StrategyParams)
        assert params.version == "optuna"

    def test_signal_weights_normalised(self) -> None:  # T-OSPC-1b
        study = _make_study()
        weights_holder: list[SignalWeights] = []

        def obj(trial: optuna.Trial) -> float:
            params = sample_momentum_params(trial)
            weights_holder.append(params.weights)
            return 0.0

        study.optimize(obj, n_trials=5)

        for w in weights_holder:
            total = w.w_rsi + w.w_macd + w.w_bb + w.w_mom + w.w_sentiment
            assert math.isclose(total, 1.0, abs_tol=1e-9)

    def test_fusion_weights_normalised(self) -> None:  # T-OSPC-2
        study = _make_study()
        fusion_holder: list[SentimentFusionWeights] = []

        def obj(trial: optuna.Trial) -> float:
            params = sample_momentum_params(trial)
            fusion_holder.append(params.sentiment_fusion_weights)
            return 0.0

        study.optimize(obj, n_trials=5)

        for f in fusion_holder:
            total = f.news + f.social + f.geopolitical
            assert math.isclose(total, 1.0, abs_tol=1e-9)

    def test_exit_below_entry(self) -> None:  # T-OSPC-3
        study = _make_study()
        thresholds: list[tuple[float, float]] = []

        def obj(trial: optuna.Trial) -> float:
            params = sample_momentum_params(trial)
            thresholds.append((params.entry_threshold, params.exit_threshold))
            return 0.0

        study.optimize(obj, n_trials=10)

        for entry, exit_ in thresholds:
            assert exit_ < entry

    def test_max_hold_bars_in_range(self) -> None:  # T-OSPC-4
        study = _make_study()
        hold_bars: list[int] = []

        def obj(trial: optuna.Trial) -> float:
            params = sample_momentum_params(trial)
            hold_bars.append(params.max_hold_bars)
            return 0.0

        study.optimize(obj, n_trials=10)

        for h in hold_bars:
            assert 5 <= h <= 100


class TestSampleMeanReversionParams:
    def test_returns_dict_with_all_keys(self) -> None:  # T-OSPC-5
        study = _make_study()
        captured: dict[str, float] | None = None

        def obj(trial: optuna.Trial) -> float:
            nonlocal captured
            captured = sample_mean_reversion_params(trial)
            return 0.0

        study.optimize(obj, n_trials=1)

        assert captured is not None
        expected_keys = {
            "bb_oversold",
            "bb_overbought",
            "bb_target",
            "sentiment_confirm_threshold",
            "max_hold_bars",
            "position_size",
        }
        assert set(captured.keys()) == expected_keys

    def test_bounds_respected(self) -> None:  # T-OSPC-6
        study = _make_study()
        samples: list[dict[str, float]] = []

        def obj(trial: optuna.Trial) -> float:
            samples.append(sample_mean_reversion_params(trial))
            return 0.0

        study.optimize(obj, n_trials=10)

        for s in samples:
            assert 0.05 <= s["bb_oversold"] <= 0.3
            assert 0.7 <= s["bb_overbought"] <= 0.95
            assert s["bb_oversold"] < s["bb_overbought"]
            assert 0.35 <= s["bb_target"] <= 0.65
            assert 0.0 <= s["sentiment_confirm_threshold"] <= 0.5
            assert 5 <= s["max_hold_bars"] <= 50
            assert 0.01 <= s["position_size"] <= 0.1
