"""T-CRYPTO-GATE-*: crypto operator fit + gate (operator vs pairwise on linked peers)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
from qts.propagation.crypto.gate import (
    evaluate_crypto_gate,
    fit_crypto_propagation,
)

from qts.propagation.crypto.links import CRYPTO_RELATIONS
from qts.propagation.crypto.samples import ContagionSample


def _toy_samples(n_events: int = 8) -> list[ContagionSample]:
    rng = np.random.default_rng(0)
    n_nodes, dim = 3, len(CRYPTO_RELATIONS) + 2
    feats = rng.normal(size=(n_nodes, dim))
    out = []
    for _ in range(n_events):
        seed = float(rng.normal())
        reactions = np.zeros(n_nodes)
        reactions[0] = seed
        reactions[1] = 0.5 * seed  # peer 1 is a linked transmitter of the source
        out.append(
            ContagionSample(
                named_idx=0,
                merit=seed,
                event_ts=datetime(2023, 1, 1, tzinfo=UTC),
                features=feats,
                reactions=reactions,
            )
        )
    return out


def test_fit_uses_seven_relation_channels() -> None:  # T-CRYPTO-GATE-1
    samples = _toy_samples()
    adj = np.full((3, 3), -1, dtype=np.int64)
    adj[0, 1] = CRYPTO_RELATIONS.index("entity_exposure")  # relation index 4 (>3) must be valid
    model = fit_crypto_propagation(
        samples, adj_type=adj, feature_dim=samples[0].features.shape[1], steps=50
    )
    assert model.n_relations == len(CRYPTO_RELATIONS)  # 7, not the equity default of 4
    pred = model.predict_np(samples[0].features, adj, np.array([0]), np.array([samples[0].merit]))
    assert pred.shape == (1, 3) and np.isfinite(pred).all()  # index-4 relation did not crash


def test_gate_reports_and_perfect_pred_beats_pairwise() -> None:  # T-CRYPTO-GATE-2
    samples = _toy_samples()
    adj = np.full((3, 3), -1, dtype=np.int64)
    adj[0, 1] = CRYPTO_RELATIONS.index("entity_exposure")
    react = np.stack([s.reactions for s in samples])
    perfect = react.copy()  # predictions == truth -> graph_mse 0
    rep = evaluate_crypto_gate(perfect, samples, adj_type=adj)
    assert rep.n_linked_obs == len(samples)  # one linked peer (node 1) per event
    assert rep.graph_mse == 0.0 and rep.beats_pairwise is True
    assert rep.graph_hit == 1.0
