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


def test_event_study_detects_linked_more_negative() -> None:  # T-CRYPTO-GATE-3
    from qts.propagation.crypto.gate import event_study_linked_vs_unlinked

    # node 0 = source; 1 = linked (drops hard); 2 = unlinked (flat). 12 events.
    rng = np.random.default_rng(1)
    samples = []
    adj = np.full((3, 3), -1, dtype=np.int64)
    adj[0, 1] = CRYPTO_RELATIONS.index("entity_exposure")
    feats = rng.normal(size=(3, len(CRYPTO_RELATIONS) + 2))
    for _ in range(12):
        reactions = np.array([-0.10, -0.08 + rng.normal(0, 0.01), rng.normal(0, 0.01)])
        samples.append(
            ContagionSample(0, -0.10, datetime(2023, 1, 1, tzinfo=UTC), feats, reactions)
        )
    rep = event_study_linked_vs_unlinked(samples, adj_type=adj)
    assert rep.mean_linked_car < rep.mean_unlinked_car  # linked peers drop more
    assert rep.significant is True and rep.mann_whitney_p < 0.05


def test_backtest_pnl_both_legs() -> None:  # T-CRYPTO-GATE-4
    from datetime import timedelta

    from qts.propagation.crypto.gate import contagion_backtest

    # 4 events; node 1 is the linked peer we short; it drops -10% idiosyncratic each event.
    n_nodes = 3
    adj = np.full((n_nodes, n_nodes), -1, dtype=np.int64)
    adj[0, 1] = CRYPTO_RELATIONS.index("entity_exposure")
    base = datetime(2023, 1, 1, tzinfo=UTC)
    grid = [base + timedelta(hours=h) for h in range(50)]
    closes = {
        "BTC": np.full(50, 100.0),
        "SRC": np.full(50, 100.0),
        "PEER": np.concatenate([np.full(20, 100.0), np.linspace(100, 90, 30)]),
    }
    samples, preds = [], []
    feats = np.zeros((n_nodes, len(CRYPTO_RELATIONS) + 2))
    for k in range(4):
        reactions = np.array([-0.10, -0.10, 0.0])  # node1 abnormal -10%
        samples.append(ContagionSample(0, -0.10, grid[20], feats, reactions))
        preds.append([0.0, -0.09, 0.0])  # operator predicts node1 drops most
    from qts.propagation.crypto.dataset import ContagionDataset

    ds = ContagionDataset(
        samples=samples,
        adj_type=adj,
        feature_dim=feats.shape[1],
        graph=None,
        grid=grid,
        closes=closes,
    )
    res = contagion_backtest(
        np.array(preds), ds, token_names=("SRC", "PEER", "BTC"), top_k=1, cost_bps=0.0, horizon=3
    )
    assert res.n_trades == 4
    assert res.market_neutral_mean > 0.09  # shorting a -10% idiosyncratic move earns ~+10%
    assert res.outright_mean > 0.0  # peer raw price fell over the hold -> short profits
