"""Graph neural-ODE for the propagation graph (continuous-time, nonlinear message passing).

Graduation from the linear ``GatedPropagationGraph``: each node carries a hidden state that evolves
under a continuous ODE with NONLINEAR (tanh) message passing over regime-gated, feature-conditioned
edges. The ``do()`` intervention enters as a forcing term pinning the named node to an encoding of
its merit; the system is integrated with ``torchdiffeq.odeint``. Edges read the FULL node features —
the model must still LEARN to ignore the factor/confound block (no cheating), so transfer is a fair
test of whether the richer dynamics generalise the 2-hop composition.

Outcome (2026-05-23): the richer dynamics fit the 2-hop chain in-sample but do NOT improve transfer
to an unseen chain over the linear model (~2/5 seeds; adding heads raises in-sample fit, not
transfer). Kept as documented exploration — not the gated model. See design doc §13.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torchdiffeq import odeint

K_CONCEPTS = 3
N_REGIMES = 2
N_HEADS = 1  # message-passing channels; >1 lets distinct relations (R1, R2) use separate subspaces
HIDDEN_DIM = 8
INTEGRATION_TIME = 4.0
ODE_STEPS = 8
FORCING = 5.0  # strength of the do() pull holding the named node at its merit encoding


class _Dynamics(nn.Module):
    """dh/dt = -h + tanh(Σ_h A_h·(h W_h)) + forcing·mask·(target - h).

    ``A``/``target``/``mask`` are fixed per batch. Multi-head: each head ``h`` carries its own
    regime-gated adjacency ``A_h`` and message weight ``W_h``, so different relations can propagate
    through different subspaces instead of interfering in a single message function. ``a`` is
    (n_heads, B, n_nodes, n_nodes); ``w_msg`` is (n_heads, hidden, hidden).
    """

    def __init__(
        self, w_msg: Tensor, a: Tensor, target: Tensor, mask: Tensor, forcing: float
    ) -> None:
        super().__init__()
        self.w_msg = w_msg  # (n_heads, hidden, hidden)
        self.a = a  # (n_heads, B, n_nodes, n_nodes)
        self.target = target  # (B, n_nodes, hidden)
        self.mask = mask  # (B, n_nodes, 1)
        self.forcing = forcing

    def forward(self, t: Tensor, h: Tensor) -> Tensor:
        msg = torch.einsum("bnd,hde->hbne", h, self.w_msg)  # (n_heads, B, n_nodes, hidden)
        agg = torch.einsum("hbnm,hbme->hbne", self.a, msg).sum(0)  # (B, n_nodes, hidden)
        dh = -h + torch.tanh(agg)
        return dh + self.forcing * self.mask * (self.target - h)


class GraphNeuralODE(nn.Module):
    """Continuous-time graph propagation. Same call signature as ``GatedPropagationGraph``."""

    def __init__(
        self,
        asset_features: np.ndarray | Tensor,
        k_concepts: int = K_CONCEPTS,
        n_regimes: int = N_REGIMES,
        n_heads: int = N_HEADS,
        hidden_dim: int = HIDDEN_DIM,
        integration_time: float = INTEGRATION_TIME,
        ode_steps: int = ODE_STEPS,
        forcing: float = FORCING,
    ) -> None:
        super().__init__()
        af = torch.as_tensor(np.asarray(asset_features), dtype=torch.float32)
        self.n_assets, self.feature_dim = af.shape
        self.k_concepts = k_concepts
        self.n_regimes = n_regimes
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.integration_time = integration_time
        self.ode_steps = ode_steps
        self.forcing = forcing
        self.register_buffer("asset_features", af)
        self.concept_features = nn.Parameter(0.1 * torch.randn(k_concepts, self.feature_dim))
        self.M = nn.Parameter(
            0.01 * torch.randn(n_heads, n_regimes, self.feature_dim, self.feature_dim)
        )
        self.gate = nn.Linear(n_regimes, n_regimes)
        self.w_msg = nn.Parameter(hidden_dim**-0.5 * torch.randn(n_heads, hidden_dim, hidden_dim))
        self.v_merit = nn.Parameter(0.5 * torch.randn(hidden_dim))
        self.w_out = nn.Linear(hidden_dim, 1, bias=False)

    @property
    def n_nodes(self) -> int:
        return self.n_assets + self.k_concepts

    def _xi(self) -> Tensor:
        af: Tensor = self.asset_features  # type: ignore[assignment]
        return torch.cat([af, self.concept_features], dim=0)

    def _edges(self, regime: Tensor) -> Tensor:
        xi = self._xi()
        a = torch.einsum("nf,hrfg,mg->hrnm", xi, self.M, xi)  # (n_heads, R, n_nodes, n_nodes)
        onehot = torch.nn.functional.one_hot(regime, self.n_regimes).float()  # (B, R)
        pi = torch.softmax(self.gate(onehot), dim=-1)  # (B, R)
        return torch.einsum("br,hrnm->hbnm", pi, a)  # (n_heads, B, n_nodes, n_nodes)

    def forward(self, named_idx: Tensor, merit: Tensor, regime: Tensor) -> Tensor:
        b = named_idx.shape[0]
        a = self._edges(regime)  # (n_heads, B, n_nodes, n_nodes)
        mask = torch.nn.functional.one_hot(named_idx, self.n_nodes).float().unsqueeze(-1)
        target = mask * merit[:, None, None] * self.v_merit[None, None, :]  # named -> merit·v_merit
        h0 = torch.zeros(b, self.n_nodes, self.hidden_dim, dtype=a.dtype, device=a.device)
        dyn = _Dynamics(self.w_msg, a, target, mask, self.forcing)
        t = torch.linspace(0.0, self.integration_time, self.ode_steps + 1, dtype=a.dtype)
        h_t = odeint(
            dyn, h0, t, method="rk4", options={"step_size": self.integration_time / self.ode_steps}
        )[-1]
        out: Tensor = self.w_out(h_t).squeeze(-1)
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
