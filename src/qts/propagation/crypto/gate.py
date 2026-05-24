"""Fit the relation-typed operator on contagion events; score against the nulls (spec §7, §12).

Null A = event-study linked-vs-unlinked; Null B = operator beats per-pair correlation.
Gate/backtest take a PREDICTIONS array (not the model) for deterministic testing;
the CLI computes predictions once via ``model.predict_np``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from qts.propagation.crypto.links import CRYPTO_RELATIONS
from qts.propagation.crypto.samples import ContagionSample
from qts.propagation.equity.baseline import EquityCorrelationalBaseline
from qts.propagation.equity.model import RelationTypedPropagation


def fit_crypto_propagation(
    samples: list[ContagionSample],
    *,
    adj_type: np.ndarray,
    feature_dim: int,
    steps: int = 2000,
    lr: float = 5e-3,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    seed: int = 0,
) -> RelationTypedPropagation:
    """Meta-train the per-relation-type operator over events.

    Uses ``n_relations=len(CRYPTO_RELATIONS)`` (7) — the equity ``fit_typed_propagation``
    hardcodes 4 and would index out of range on crypto edges.
    """
    torch.manual_seed(seed)
    feats = torch.as_tensor(samples[0].features, dtype=torch.float32)
    adj = torch.as_tensor(adj_type, dtype=torch.long)
    named = torch.as_tensor([s.named_idx for s in samples], dtype=torch.long)
    merit = torch.as_tensor([s.merit for s in samples], dtype=torch.float32)
    y = torch.as_tensor(np.stack([s.reactions for s in samples]), dtype=torch.float32)
    model = RelationTypedPropagation(feature_dim=feature_dim, n_relations=len(CRYPTO_RELATIONS))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        loss = torch.nn.functional.mse_loss(model(feats, adj, named, merit), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
    return model


@dataclass(frozen=True)
class CryptoGateReport:
    n_linked_obs: int
    graph_mse: float
    pairwise_mse: float
    graph_hit: float
    pairwise_hit: float
    beats_pairwise: bool


def evaluate_crypto_gate(
    predictions: np.ndarray, samples: list[ContagionSample], *, adj_type: np.ndarray
) -> CryptoGateReport:
    """Operator vs per-pair correlational baseline on LINKED-peer idiosyncratic CARs (Null B)."""
    react = np.stack([s.reactions for s in samples])
    named = np.array([s.named_idx for s in samples])
    rows = np.arange(len(samples))
    gerr, berr, ghit, bhit = [], [], [], []
    for b in range(len(samples)):
        src = named[b]
        for j in np.where(adj_type[src] >= 0)[0]:
            actual = react[b, j]
            gerr.append((predictions[b, j] - actual) ** 2)
            base = EquityCorrelationalBaseline.from_history(
                named_returns=react[rows, src], peer_returns=react[rows, j]
            )
            bp = base.predict(named_reaction=react[b, src])
            berr.append((bp - actual) ** 2)
            if actual != 0:
                ghit.append(float(np.sign(predictions[b, j]) == np.sign(actual)))
                bhit.append(float(np.sign(bp) == np.sign(actual)))
    gm, bm = float(np.mean(gerr)), float(np.mean(berr))
    return CryptoGateReport(
        n_linked_obs=len(gerr),
        graph_mse=gm,
        pairwise_mse=bm,
        graph_hit=float(np.mean(ghit)) if ghit else float("nan"),
        pairwise_hit=float(np.mean(bhit)) if bhit else float("nan"),
        beats_pairwise=gm < bm,
    )
