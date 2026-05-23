"""Relation-typed, LinkGraph-masked propagation: the operator learns transmission per link type.

Edges come from the LinkGraph (typed adjacency); the per-relation-type bilinear ``M_r`` learns
propagation sign+magnitude for type ``r`` from the endpoints' features. Meta-trained across links,
the generic per-type rule transfers to UNSEEN links (design §15 mechanism, real equities).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from qts.propagation.equity.graph import LinkGraph
from qts.propagation.equity.universe import EquityUniverse

RELATIONS: tuple[str, ...] = ("competitor", "supplier", "customer", "partner")
PROP_STEPS = 3


def build_typed_adjacency(graph: LinkGraph, universe: EquityUniverse) -> np.ndarray:
    """(N, N) int matrix: ``adj[i, j]`` = relation index of directed link i->j, or -1 if none."""
    n = len(universe.tickers)
    adj = np.full((n, n), -1, dtype=np.int64)
    rel_index = {r: i for i, r in enumerate(RELATIONS)}
    for e in graph.edges:
        if e.relation not in rel_index:
            continue
        i, j = universe.index_of(e.source), universe.index_of(e.peer)
        adj[i, j] = rel_index[e.relation]
    return adj


class RelationTypedPropagation(nn.Module):
    """``W[i,j] = xi_i^T M_{adj[i,j]} xi_j`` on graph edges; do()-clamp + propagate."""

    def __init__(
        self, feature_dim: int, n_relations: int = len(RELATIONS), prop_steps: int = PROP_STEPS
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.n_relations = n_relations
        self.prop_steps = prop_steps
        self.M = nn.Parameter(0.01 * torch.randn(n_relations, feature_dim, feature_dim))

    def edge_weights(self, feats: Tensor, adj_type: Tensor) -> Tensor:
        bil = torch.einsum("if,rfg,jg->rij", feats, self.M, feats)  # (R, N, N)
        mask = (adj_type >= 0).float()  # (N, N)
        safe = adj_type.clamp(min=0)  # (N, N)
        w = bil.gather(0, safe.unsqueeze(0)).squeeze(0)  # (N, N): pick M_{adj[i,j]}
        return w * mask

    def forward(self, feats: Tensor, adj_type: Tensor, named_idx: Tensor, merit: Tensor) -> Tensor:
        w = self.edge_weights(feats, adj_type)  # (N, N): edge i->j weight
        n = feats.shape[0]
        clamp = torch.nn.functional.one_hot(named_idx, n).float()  # (B, N)
        source = clamp * merit[:, None]
        clamp_mask = clamp.bool()
        x = source.clone()
        for _ in range(self.prop_steps):
            x = x @ w  # x_new[b, j] = sum_i x[b, i] * w[i, j]
            x = torch.where(clamp_mask, source, x)  # re-pin the do() source
        return x

    @torch.no_grad()
    def predict_np(
        self, feats: np.ndarray, adj_type: np.ndarray, named_idx: np.ndarray, merit: np.ndarray
    ) -> np.ndarray:
        self.eval()
        out = self(
            torch.as_tensor(feats, dtype=torch.float32),
            torch.as_tensor(adj_type, dtype=torch.long),
            torch.as_tensor(named_idx, dtype=torch.long),
            torch.as_tensor(merit, dtype=torch.float32),
        )
        return out.cpu().numpy()  # type: ignore[no-any-return]
