"""Train the relation-typed operator and score it vs the correlational baseline (spec §8 gate)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from qts.propagation.equity.baseline import EquityCorrelationalBaseline
from qts.propagation.equity.model import RelationTypedPropagation
from qts.propagation.equity.samples import EventSample


@dataclass(frozen=True)
class PathAReport:
    graph_mse: float
    baseline_mse: float
    beats_baseline: bool


def _stack(samples: list[EventSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    named = np.array([s.named_idx for s in samples])
    merit = np.array([s.merit for s in samples], dtype=float)
    reactions = np.stack([s.reactions for s in samples])
    return named, merit, reactions


def fit_typed_propagation(
    samples: list[EventSample],
    *,
    adj_type: np.ndarray,
    feature_dim: int,
    steps: int = 2000,
    lr: float = 5e-3,
    grad_clip: float = 1.0,
    seed: int = 0,
) -> RelationTypedPropagation:
    """Meta-train the per-relation-type operator over events (shared universe graph + features)."""
    torch.manual_seed(seed)
    feats = torch.as_tensor(samples[0].features, dtype=torch.float32)
    adj = torch.as_tensor(adj_type, dtype=torch.long)
    named, merit, reactions = _stack(samples)
    ni = torch.as_tensor(named, dtype=torch.long)
    me = torch.as_tensor(merit, dtype=torch.float32)
    y = torch.as_tensor(reactions, dtype=torch.float32)
    model = RelationTypedPropagation(feature_dim=feature_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = torch.nn.functional.mse_loss(model(feats, adj, ni, me), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
    return model


def evaluate_path_a_gate(
    model: RelationTypedPropagation,
    samples: list[EventSample],
    *,
    adj_type: np.ndarray,
) -> PathAReport:
    """Operator vs correlational baseline on LINKED-peer reactions (the non-correlational test)."""
    named, merit, reactions = _stack(samples)
    pred = model.predict_np(samples[0].features, adj_type, named, merit)
    rows, gerr, berr = np.arange(len(samples)), [], []
    for b in range(len(samples)):
        src = named[b]
        peers = np.where(adj_type[src] >= 0)[0]
        for j in peers:
            gerr.append((pred[b, j] - reactions[b, j]) ** 2)
            base = EquityCorrelationalBaseline.from_history(
                named_returns=reactions[rows, src], peer_returns=reactions[rows, j]
            )
            berr.append((base.predict(named_reaction=reactions[b, src]) - reactions[b, j]) ** 2)
    graph_mse, baseline_mse = float(np.mean(gerr)), float(np.mean(berr))
    return PathAReport(
        graph_mse=graph_mse, baseline_mse=baseline_mse, beats_baseline=graph_mse < baseline_mse
    )
