"""Compose the transferable 1-hop operator at inference (Path C). See spec 2026-05-23-...-unroll.

End-to-end 2-hop training does not transfer (design doc §13). Here the operator is trained on 1-hop
relations only; the 2-hop terminal is reached by UNROLLING — predict B from A, re-inject as merit,
predict C. Each hop is a bona-fide 1-hop prediction, the operation v0 proved transfers.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


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
