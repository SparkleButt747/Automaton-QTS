"""T-PROP-GATE-1..3: end-to-end feasibility gate."""

from __future__ import annotations

import numpy as np

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
