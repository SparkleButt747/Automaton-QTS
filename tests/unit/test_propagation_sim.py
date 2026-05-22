"""T-PROP-SIM-1..3: adversarial sim — confound, determinism, causal edge."""

from __future__ import annotations

import numpy as np

from qts.propagation.sim import (
    EventBatch,
    PropagationSimConfig,
    build_world,
    generate_events,
    make_splits,
)


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


def test_generate_is_seed_deterministic() -> None:  # T-PROP-SIM-2
    world = build_world(PropagationSimConfig(seed=0))
    a = generate_events(world, 64, np.random.default_rng(1))
    b = generate_events(world, 64, np.random.default_rng(1))
    assert isinstance(a, EventBatch)
    assert a.reactions.shape == (64, world.config.n_assets)
    assert np.array_equal(a.named_idx, b.named_idx)
    assert np.allclose(a.reactions, b.reactions)


def test_causal_edge_hits_substitute_not_decoy() -> None:  # T-PROP-SIM-3
    world = build_world(PropagationSimConfig(seed=0, idiosyncratic_vol=0.0, factor_vol=0.0))
    batch = generate_events(world, 2000, np.random.default_rng(2), allowed_types=(0,))
    sub = world.triples[0].substitute
    decoy = world.triples[0].decoy
    sign = world.regime_signs[batch.regime]
    expected_sub = sign * world.config.propagation_gain * batch.merit
    assert np.allclose(batch.reactions[:, sub], expected_sub, atol=1e-9)
    assert np.allclose(batch.reactions[:, decoy], 0.0, atol=1e-9)


def test_splits_partition_event_types() -> None:  # T-PROP-SIM-2b
    world = build_world(PropagationSimConfig(seed=0))
    n = world.config.n_event_types
    train, val, test, transfer = make_splits(
        world, np.random.default_rng(3), n_train=200, n_val=50, n_test=50, n_transfer=50
    )
    train_types = set(range(n - 1))
    assert set(np.unique(train.event_type)).issubset(train_types)
    assert set(np.unique(test.event_type)).issubset(train_types)
    assert set(np.unique(transfer.event_type)) == {n - 1}
