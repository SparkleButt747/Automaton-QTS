"""T-PATHA-BASE-*: correlational baseline (beta-projection on the named firm's move)."""

from __future__ import annotations

import numpy as np

from qts.propagation.equity.baseline import EquityCorrelationalBaseline


def test_beta_projection_predicts_peer_from_named_move() -> None:  # T-PATHA-BASE-1
    rng = np.random.default_rng(0)
    named_hist = rng.normal(0, 0.01, 500)
    peer_hist = 1.5 * named_hist
    base = EquityCorrelationalBaseline.from_history(
        named_returns=named_hist, peer_returns=peer_hist
    )
    pred = base.predict(named_reaction=0.02)
    assert abs(pred - 0.03) < 1e-6


def test_zero_beta_when_uncorrelated() -> None:  # T-PATHA-BASE-2
    rng = np.random.default_rng(1)
    base = EquityCorrelationalBaseline.from_history(
        named_returns=rng.normal(0, 0.01, 500), peer_returns=rng.normal(0, 0.01, 500)
    )
    assert abs(base.beta) < 0.2
