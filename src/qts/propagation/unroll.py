"""Compose the transferable 1-hop operator at inference (Path C). See spec 2026-05-23-...-unroll.

End-to-end 2-hop training does not transfer (design doc §13). Here the operator is trained on 1-hop
relations only; the 2-hop terminal is reached by UNROLLING — predict B from A, re-inject as merit,
predict C. Each hop is a bona-fide 1-hop prediction, the operation v0 proved transfers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qts.propagation.baselines import CorrelationalBaseline, no_propagation_predict
from qts.propagation.sim import EventBatch, GroundTruthWorld


class Operator(Protocol):
    """Anything with the propagation operator's call signature (linear graph or neural-ODE)."""

    def predict_np(
        self, named_idx: np.ndarray, merit: np.ndarray, regime: np.ndarray
    ) -> np.ndarray: ...


def unroll_predict(
    model: Operator,
    named_idx: np.ndarray,
    merit: np.ndarray,
    regime: np.ndarray,
    hop_successors: list[np.ndarray],
) -> np.ndarray:
    """Iterate the 1-hop operator along known successor indices; return the terminal reaction.

    ``hop_successors[h]`` is the per-row index of hop h's successor (known indices, spec §3).
    Hop h names the previous successor and injects its predicted reaction as the next merit.
    """
    rows = np.arange(len(named_idx))
    src = np.asarray(named_idx)
    m = np.asarray(merit, dtype=float)
    r_succ = m
    for succ in hop_successors:
        out = model.predict_np(src, m, regime)
        r_succ = out[rows, succ]
        src, m = succ, r_succ
    return r_succ


@dataclass(frozen=True)
class UnrollReport:
    hop1_mse_graph: float
    hop1_mse_corr: float
    terminal_mse_graph: float
    terminal_mse_corr: float
    terminal_mse_noprop: float
    hop1_pass: bool
    transfer_pass: bool


def evaluate_unroll_transfer(
    world: GroundTruthWorld,
    model: Operator,
    transfer: EventBatch,
    *,
    n_history: int = 20000,
    seed: int = 0,
) -> UnrollReport:
    """Score the unrolled A->B->C against the correlational baseline + no-prop floor."""
    corr = CorrelationalBaseline.from_history(world, n_samples=n_history, seed=seed)
    rows = np.arange(len(transfer))
    sub = world.substitute_indices(transfer.event_type)
    term = world.terminal_indices(transfer.event_type)

    r_b = unroll_predict(model, transfer.named_idx, transfer.merit, transfer.regime, [sub])
    r_c = unroll_predict(model, transfer.named_idx, transfer.merit, transfer.regime, [sub, term])

    truth_b = transfer.reactions[rows, sub]
    truth_c = transfer.reactions[rows, term]
    pc = corr.predict(transfer)
    pn = no_propagation_predict(transfer, world.config.n_assets)

    hop1_mse_graph = float(np.mean((r_b - truth_b) ** 2))
    hop1_mse_corr = float(np.mean((pc[rows, sub] - truth_b) ** 2))
    terminal_mse_graph = float(np.mean((r_c - truth_c) ** 2))
    terminal_mse_corr = float(np.mean((pc[rows, term] - truth_c) ** 2))
    terminal_mse_noprop = float(np.mean((pn[rows, term] - truth_c) ** 2))

    hop1_pass = hop1_mse_graph < hop1_mse_corr
    transfer_pass = (
        terminal_mse_graph < terminal_mse_corr and terminal_mse_graph < terminal_mse_noprop
    )
    return UnrollReport(
        hop1_mse_graph=hop1_mse_graph,
        hop1_mse_corr=hop1_mse_corr,
        terminal_mse_graph=terminal_mse_graph,
        terminal_mse_corr=terminal_mse_corr,
        terminal_mse_noprop=terminal_mse_noprop,
        hop1_pass=hop1_pass,
        transfer_pass=transfer_pass,
    )
