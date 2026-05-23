"""T-PATHA-GATE-*: train loop + capture/win report on held-out links."""

from __future__ import annotations

from datetime import date

import numpy as np

from qts.propagation.equity.gate import PathAReport, evaluate_path_a_gate, fit_typed_propagation
from qts.propagation.equity.model import RELATIONS
from qts.propagation.equity.samples import EventSample


def _samples(n: int, adj_type: np.ndarray, seed: int) -> list[EventSample]:
    rng = np.random.default_rng(seed)
    feats = np.eye(adj_type.shape[0], dtype=float)
    out = []
    for _ in range(n):
        named = int(rng.integers(adj_type.shape[0]))
        merit = float(rng.normal())
        reactions = np.zeros(adj_type.shape[0])
        reactions[named] = merit
        for j in range(adj_type.shape[0]):
            if adj_type[named, j] == RELATIONS.index("competitor"):
                reactions[j] = -0.8 * merit
        out.append(
            EventSample(
                named_idx=named,
                merit=merit,
                event_date=date(2022, 1, 1),
                features=feats,
                reactions=reactions,
            )
        )
    return out


def test_fit_then_report_is_well_formed() -> None:  # T-PATHA-GATE-1
    adj = np.full((3, 3), -1)
    adj[0, 1] = RELATIONS.index("competitor")
    train = _samples(64, adj, seed=0)
    model = fit_typed_propagation(train, adj_type=adj, feature_dim=3, steps=200, seed=0)
    report = evaluate_path_a_gate(model, _samples(32, adj, seed=1), adj_type=adj)
    assert isinstance(report, PathAReport)
    assert np.isfinite(report.graph_mse) and np.isfinite(report.baseline_mse)
    assert isinstance(report.beats_baseline, bool)
