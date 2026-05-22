"""Regime-gated, feature-conditioned bilinear propagation graph (linear dynamics, v0)."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

K_CONCEPTS = 3
N_REGIMES = 2
PROP_STEPS = 8


class GatedPropagationGraph(nn.Module):
    """W[i,j] = sum_r pi_r * (xi_i^T M_r xi_j); do()-clamp the named node, unroll PROP_STEPS."""

    def __init__(
        self,
        asset_features: np.ndarray | Tensor,
        k_concepts: int = K_CONCEPTS,
        n_regimes: int = N_REGIMES,
        prop_steps: int = PROP_STEPS,
    ) -> None:
        super().__init__()
        af = torch.as_tensor(np.asarray(asset_features), dtype=torch.float32)
        self.n_assets, self.feature_dim = af.shape
        self.k_concepts = k_concepts
        self.n_regimes = n_regimes
        self.prop_steps = prop_steps
        self.register_buffer("asset_features", af)
        self.concept_features = nn.Parameter(0.1 * torch.randn(k_concepts, self.feature_dim))
        self.M = nn.Parameter(0.01 * torch.randn(n_regimes, self.feature_dim, self.feature_dim))
        self.gate = nn.Linear(n_regimes, n_regimes)

    @property
    def n_nodes(self) -> int:
        return self.n_assets + self.k_concepts

    def _xi(self) -> Tensor:
        af: Tensor = self.asset_features  # type: ignore[assignment]
        return torch.cat([af, self.concept_features], dim=0)

    def edge_weights(self, regime: Tensor) -> Tensor:
        xi = self._xi()  # (n_nodes, F)
        a = torch.einsum("nf,rfg,mg->rnm", xi, self.M, xi)  # (R, n_nodes, n_nodes)
        onehot = torch.nn.functional.one_hot(regime, self.n_regimes).float()  # (B, R)
        pi = torch.softmax(self.gate(onehot), dim=-1)  # (B, R)
        return torch.einsum("br,rnm->bnm", pi, a)  # (B, n_nodes, n_nodes)

    def forward(self, named_idx: Tensor, merit: Tensor, regime: Tensor) -> Tensor:
        w = self.edge_weights(regime)  # (B, n_nodes, n_nodes)
        clamp = torch.nn.functional.one_hot(named_idx, self.n_nodes).float()  # (B, n_nodes)
        source = clamp * merit[:, None]  # merit at named node, 0 elsewhere
        clamp_mask = clamp.bool()
        x = source.clone()
        for _ in range(self.prop_steps):
            x = torch.einsum("bnm,bm->bn", w, x)
            x = torch.where(clamp_mask, source, x)  # re-pin the do() source
        return x[:, : self.n_assets]

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
