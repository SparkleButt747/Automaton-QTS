"""T-PROP-SIM-1..3: adversarial sim — confound, determinism, causal edge."""

from __future__ import annotations

import numpy as np
from qts.propagation.sim import PropagationSimConfig, build_world


def _corr_matrix(world) -> np.ndarray:
    cfg = world.config
    cov = world.loadings @ world.loadings.T + cfg.idiosyncratic_vol**2 * np.eye(cfg.n_assets)
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)


def test_confound_bounds_hold() -> None:  # T-PROP-SIM-1
    world = build_world(PropagationSimConfig(seed=0))
    corr = _corr_matrix(world)
    for tri in world.triples:
        assert corr[tri.named, tri.decoy] >= 0.5, "decoy must be correlated with named"
        assert abs(corr[tri.named, tri.substitute]) <= 0.2, "substitute must be ~uncorrelated"
        sn = world.features[tri.named, 2:]
        ss = world.features[tri.substitute, 2:]
        cos = float(sn @ ss / (np.linalg.norm(sn) * np.linalg.norm(ss)))
        assert cos >= 0.5, "substitute must share named's sector code"
