"""The correlational bar the graph must beat: beta-projection on the named firm's realised move."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EquityCorrelationalBaseline:
    """``r_hat_peer = beta * r_named``; beta from pre-event return history (spec §6 bar)."""

    beta: float

    @classmethod
    def from_history(
        cls, *, named_returns: np.ndarray, peer_returns: np.ndarray
    ) -> EquityCorrelationalBaseline:
        x = np.asarray(named_returns, dtype=float)
        y = np.asarray(peer_returns, dtype=float)
        var = float(np.var(x, ddof=1))
        beta = float(np.cov(y, x, ddof=1)[0, 1] / var) if var > 0 else 0.0
        return cls(beta=beta)

    def predict(self, *, named_reaction: float) -> float:
        return self.beta * named_reaction
