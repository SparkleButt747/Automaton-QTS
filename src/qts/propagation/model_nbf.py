"""NBFNet-style path-aggregation operator for the propagation graph (vector node states).

Graduation from the linear ``GatedPropagationGraph`` and the continuous ``GraphNeuralODE``: instead
of propagating a SCALAR reaction through a single feature-conditioned bilinear matrix, this carries
a per-node VECTOR hidden state through ``L`` LEARNABLE message-passing layers (NBFNet, NeurIPS
2021). The do() intervention seeds the named node with a query-conditioned encoding of its merit;
each layer applies a distinct relation-message linear, so a node accumulates a representation of
*which relation chain reached it*. The hypothesis: that path-representation transfers the 2-hop
composition R1 ∘ R2 to a held-out chain where the single bilinear matrix could only memorise seen
(entity) pairs.

Edges are feature-conditioned and regime-gated exactly as in the bilinear model
(``a[b,u,v] = Σ_r π_r(regime_b) · xi_u^T M_r xi_v``) so propagation depends on FEATURES, not entity
identity — the structural prerequisite for transfer. ``M`` keeps shape ``(n_regimes, F, F)`` to
honour the ``model.M`` sparsity contract used by ``fit_graph``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

K_CONCEPTS = 0  # no concept nodes needed; the 2-hop chain lives entirely on the asset graph
N_REGIMES = 2
HIDDEN_DIM = 16
N_LAYERS = 3  # message-passing depth; 3 covers the 2-hop chain plus the source pin


class NBFPropagation(nn.Module):
    """Query-conditioned, vector-state, feature-conditioned relational GNN.

    Same call signature as ``GatedPropagationGraph`` so it drops into ``fit_graph`` /
    ``evaluate_feasibility`` unchanged.
    """

    def __init__(
        self,
        asset_features: np.ndarray | Tensor,
        k_concepts: int = K_CONCEPTS,
        n_regimes: int = N_REGIMES,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = N_LAYERS,
    ) -> None:
        super().__init__()
        af = torch.as_tensor(np.asarray(asset_features), dtype=torch.float32)
        self.n_assets, self.feature_dim = af.shape
        self.k_concepts = k_concepts
        self.n_regimes = n_regimes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.register_buffer("asset_features", af)
        self.concept_features = (
            nn.Parameter(0.1 * torch.randn(k_concepts, self.feature_dim))
            if k_concepts > 0
            else None
        )
        # Feature-conditioned edge bilinear (the .M sparsity contract):
        # a[b,u,v] = Σ_r π_r · xi_u^T M_r xi_v.
        self.M = nn.Parameter(0.01 * torch.randn(n_regimes, self.feature_dim, self.feature_dim))
        self.gate = nn.Linear(n_regimes, n_regimes)
        # Query-conditioned init: scalar merit -> R^d encoding pinned at the do() source.
        self.v_query = nn.Parameter(0.5 * torch.randn(hidden_dim))
        # NBFNet per-layer relation message + self update (distinct weights per hop).
        scale = hidden_dim**-0.5
        self.w_msg = nn.ParameterList(
            [nn.Parameter(scale * torch.randn(hidden_dim, hidden_dim)) for _ in range(n_layers)]
        )
        self.w_self = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.w_out = nn.Linear(hidden_dim, 1, bias=False)

    @property
    def n_nodes(self) -> int:
        return self.n_assets + self.k_concepts

    def _xi(self) -> Tensor:
        af: Tensor = self.asset_features  # type: ignore[assignment]
        if self.concept_features is None:
            return af
        return torch.cat([af, self.concept_features], dim=0)

    def edge_weights(self, regime: Tensor) -> Tensor:
        xi = self._xi()  # (n_nodes, F)
        a = torch.einsum("nf,rfg,mg->rnm", xi, self.M, xi)  # (R, n_nodes, n_nodes)
        onehot = torch.nn.functional.one_hot(regime, self.n_regimes).float()  # (B, R)
        pi = torch.softmax(self.gate(onehot), dim=-1)  # (B, R)
        return torch.einsum("br,rnm->bnm", pi, a)  # (B, n_nodes, n_nodes)

    def forward(self, named_idx: Tensor, merit: Tensor, regime: Tensor) -> Tensor:
        a = self.edge_weights(regime)  # (B, n_nodes, n_nodes)
        clamp = torch.nn.functional.one_hot(named_idx, self.n_nodes).float()  # (B, n_nodes)
        # Query-conditioned init: h[named] = merit * v_query, all other nodes zero.
        h = clamp[:, :, None] * merit[:, None, None] * self.v_query[None, None, :]
        for layer in range(self.n_layers):
            msg_in = h @ self.w_msg[layer]  # (B, n_nodes, d) — NBFNet relation message
            msg = torch.einsum("bum,buv->bvm", msg_in, a)  # aggregate over incoming edges u->v
            h = torch.relu(self.w_self[layer](h) + msg)
        out: Tensor = self.w_out(h).squeeze(-1)  # (B, n_nodes)
        return out[:, : self.n_assets]

    @torch.no_grad()
    def predict_np(
        self, named_idx: np.ndarray, merit: np.ndarray, regime: np.ndarray
    ) -> np.ndarray:
        self.eval()
        out = self(
            torch.as_tensor(named_idx, dtype=torch.long),
            torch.as_tensor(merit, dtype=torch.float32),
            torch.as_tensor(regime, dtype=torch.long),
        )
        return out.cpu().numpy()  # type: ignore[no-any-return]
