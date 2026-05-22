"""T-NSPACE-1..2: news param sampler respects bounds."""

from __future__ import annotations

import optuna

from qts.optimisation.search_space import NewsParams, sample_news_params


def test_sampled_params_within_bounds() -> None:  # T-NSPACE-1
    def objective(trial: optuna.Trial) -> float:
        p = sample_news_params(trial)
        assert isinstance(p, NewsParams)
        assert 0.1 <= p.news_signal_weight <= 0.95
        assert 15.0 <= p.belief_half_life_minutes <= 240.0
        assert 0.02 <= p.entry_threshold <= 0.4
        return 0.0

    study = optuna.create_study()
    study.optimize(objective, n_trials=25)


def test_weight_never_exceeds_constructor_limit() -> None:  # T-NSPACE-2
    def objective(trial: optuna.Trial) -> float:
        p = sample_news_params(trial)
        assert p.news_signal_weight <= 1.0  # NewsReactiveMomentum hard limit
        return 0.0

    optuna.create_study().optimize(objective, n_trials=25)
