"""T-PATHA-GATE-INT-1: with a real typed propagation in the data, the operator beats correlation."""

from __future__ import annotations

from datetime import date

import numpy as np

from qts.propagation.equity.gate import evaluate_path_a_gate, fit_typed_propagation
from qts.propagation.equity.model import RELATIONS
from qts.propagation.equity.samples import EventSample


def _world(n_nodes: int, seed: int):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    adj = np.full((n_nodes, n_nodes), -1)
    for i in range(n_nodes - 1):
        adj[i, i + 1] = RELATIONS.index("competitor")
    feats = np.eye(n_nodes, dtype=float)
    return adj, feats, rng


def _make(adj, feats, rng, n):  # type: ignore[no-untyped-def]
    out = []
    nn_ = adj.shape[0]
    for _ in range(n):
        named = int(rng.integers(nn_))
        merit = float(rng.normal())
        r = np.zeros(nn_)
        r[named] = merit
        for j in range(nn_):
            if adj[named, j] == RELATIONS.index("competitor"):
                r[j] = -0.8 * merit
        r += rng.normal(0, 0.05, nn_)
        out.append(
            EventSample(
                named_idx=named,
                merit=merit,
                event_date=date(2022, 1, 1),
                features=feats,
                reactions=r,
            )
        )
    return out


def test_operator_beats_baseline_on_negative_transmission() -> None:  # T-PATHA-GATE-INT-1
    adj, feats, rng = _world(5, seed=0)
    model = fit_typed_propagation(
        _make(adj, feats, rng, 400), adj_type=adj, feature_dim=5, steps=400, seed=0
    )
    report = evaluate_path_a_gate(model, _make(adj, feats, rng, 200), adj_type=adj)
    assert report.beats_baseline
    assert report.graph_mse < report.baseline_mse
