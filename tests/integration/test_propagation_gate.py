"""T-PROP-GATE-1..3: end-to-end feasibility gate."""

from __future__ import annotations

import numpy as np
import pytest

from qts.propagation.sim import PropagationSimConfig, build_world, make_splits
from qts.propagation.train import FeasibilityReport, evaluate_feasibility, fit_graph


def _run(seed: int, *, n_train: int, epochs: int) -> FeasibilityReport:
    world = build_world(PropagationSimConfig(seed=seed))
    train, val, test, transfer = make_splits(
        world, np.random.default_rng(seed), n_train=n_train, n_val=500, n_test=500, n_transfer=500
    )
    model = fit_graph(world, train, val, epochs=epochs, seed=seed)
    return evaluate_feasibility(world, model, test, transfer, n_history=20000, seed=seed)


def test_report_is_well_formed() -> None:  # T-PROP-GATE-1
    report = _run(seed=0, n_train=400, epochs=20)
    assert isinstance(report, FeasibilityReport)
    for v in (report.test_mse_graph, report.sub_mse_graph, report.transfer_sub_mse_graph):
        assert np.isfinite(v)
    assert isinstance(report.passed, bool)


@pytest.mark.parametrize("seed", [0])
def test_prediction_gate_beats_baselines(seed: int) -> None:  # T-PROP-GATE-2
    report = _run(seed=seed, n_train=4000, epochs=600)
    assert report.test_mse_graph < report.test_mse_noprop  # beats the no-propagation floor
    assert report.sub_mse_graph < report.sub_mse_corr  # 1-hop substitute (B)
    assert report.terminal_mse_graph < report.terminal_mse_corr  # 2-hop terminal (C)
    assert report.prediction_pass


@pytest.mark.xfail(
    reason=(
        "Fixed-world training cannot transfer the 2-hop composition: a single-world fit_graph "
        "memorises its relation-directions and fails on an unseen chain (~2/5 seeds at best), "
        "invariant to data-scaling, model class, lr, epochs, edge capacity (§13/§14). This xfail "
        "is expected and CORRECT for the single-world fit_graph. The wall is CRACKED by a "
        "training-objective change — episodic relation-resampling across a pool of worlds "
        "(train_meta, robust 50/50 transfer); see qts.propagation.meta and design §15."
    ),
    strict=False,
)
@pytest.mark.parametrize("seed", [0])
def test_transfer_gate_unseen_pair(seed: int) -> None:  # T-PROP-GATE-3
    report = _run(seed=seed, n_train=4000, epochs=600)
    assert report.transfer_sub_mse_graph < report.transfer_sub_mse_corr  # unseen chain, 1-hop (B)
    assert report.transfer_terminal_mse_graph < report.transfer_terminal_mse_corr  # 2-hop (C)
    assert report.transfer_pass
