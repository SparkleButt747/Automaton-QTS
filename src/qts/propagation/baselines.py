"""Baselines the graph must beat: a no-propagation floor and a correlational bar."""

from __future__ import annotations

import numpy as np

from qts.propagation.sim import EventBatch, GroundTruthWorld


def no_propagation_predict(batch: EventBatch, n_assets: int) -> np.ndarray:
    """Only the named asset reacts (= its merit); everything else is zero."""
    out = np.zeros((len(batch), n_assets))
    out[np.arange(len(batch)), batch.named_idx] = batch.merit
    return out


class CorrelationalBaseline:
    """beta-projection on the named asset's observed move: r_hat[i] = (cov[i,n]/cov[n,n]) * r_n."""

    def __init__(self, cov: np.ndarray) -> None:
        self.cov = cov

    @classmethod
    def from_history(
        cls, world: GroundTruthWorld, *, n_samples: int = 5000, seed: int = 0
    ) -> CorrelationalBaseline:
        cfg = world.config
        rng = np.random.default_rng(seed)
        g = rng.normal(0.0, cfg.factor_vol, (n_samples, cfg.n_factors))
        eps = rng.normal(0.0, cfg.idiosyncratic_vol, (n_samples, cfg.n_assets))
        returns = g @ world.loadings.T + eps  # event-free history
        return cls(np.cov(returns, rowvar=False))

    def predict(self, batch: EventBatch) -> np.ndarray:
        named = batch.named_idx
        r_named = batch.reactions[np.arange(len(batch)), named]
        beta = self.cov[:, named] / self.cov[named, named]  # (n_assets, B)
        result: np.ndarray = beta.T * r_named[:, None]  # (B, n_assets)
        return result
