"""T-PATHA-MODEL-*: relation-typed, LinkGraph-masked propagation."""

from __future__ import annotations

import numpy as np
import torch

from qts.propagation.equity.economic_links import EconomicLink
from qts.propagation.equity.graph import LinkGraph
from qts.propagation.equity.model import (
    RELATIONS,
    RelationTypedPropagation,
    build_typed_adjacency,
)
from qts.propagation.equity.universe import EquityUniverse


def _uni() -> EquityUniverse:
    return EquityUniverse(
        tickers=("A", "B", "C"),
        sectors=("S", "S", "S"),
        _aliases=(("a",), ("b",), ("c",)),
    )


def test_build_typed_adjacency_directed_and_typed() -> None:  # T-PATHA-MODEL-1
    links = [
        EconomicLink("A", "B", "competitor", "negative", 0.9),
        EconomicLink("B", "C", "supplier", "positive", 0.9),
    ]
    g = LinkGraph.from_links(links, min_confidence=0.5)
    adj = build_typed_adjacency(g, _uni())
    assert adj.shape == (3, 3)
    assert adj[0, 1] == RELATIONS.index("competitor")  # A->B
    assert adj[1, 2] == RELATIONS.index("supplier")  # B->C
    assert adj[0, 2] == -1  # no A->C edge


def test_forward_clamps_named_and_zero_off_graph() -> None:  # T-PATHA-MODEL-2
    adj = np.full((3, 3), -1)
    adj[0, 1] = RELATIONS.index("competitor")  # only A->B is an edge
    feats = torch.eye(3, dtype=torch.float32)  # 3 nodes, feature_dim 3
    m = RelationTypedPropagation(feature_dim=3, prop_steps=2)
    pred = m(
        feats,
        torch.as_tensor(adj),
        torch.tensor([0]),  # named = A
        torch.tensor([1.0]),  # merit
    )
    assert pred.shape == (1, 3)
    assert abs(float(pred[0, 0]) - 1.0) < 1e-6  # named clamped to merit
    assert abs(float(pred[0, 2])) < 1e-6  # C has no incoming edge -> 0


def test_sign_is_learnable_per_relation_type() -> None:  # T-PATHA-MODEL-3
    adj = np.full((2, 2), -1)
    adj[0, 1] = RELATIONS.index("competitor")
    feats = torch.eye(2, dtype=torch.float32)
    m = RelationTypedPropagation(feature_dim=2, prop_steps=1)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    adj_t = torch.as_tensor(adj)
    for _ in range(300):
        merit = torch.randn(64)
        pred = m(feats, adj_t, torch.zeros(64, dtype=torch.long), merit)
        target = torch.stack([merit, -0.8 * merit], dim=1)  # B = -0.8 * A's merit
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = m(feats, adj_t, torch.tensor([0]), torch.tensor([1.0]))
    assert p[0, 1] < -0.5  # learned the negative competitor transmission
