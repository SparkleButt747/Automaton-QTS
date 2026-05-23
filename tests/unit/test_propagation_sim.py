"""T-PROP-SIM-1..3: adversarial 2-hop-chain sim — confound, determinism, causal edges."""

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


def _cos(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def test_confound_bounds_hold() -> None:  # T-PROP-SIM-1
    world = build_world(PropagationSimConfig(seed=0))
    corr = _corr_matrix(world)
    nf = world.config.n_factors
    r1_dim = (world.config.feature_dim - nf) // 2
    r1 = slice(nf, nf + r1_dim)
    r2 = slice(nf + r1_dim, world.config.feature_dim)
    feats = world.features
    for c in world.chains:
        # both unnamed targets are factor-decorrelated from the named entity; the decoy is the bait
        assert corr[c.named, c.decoy] >= 0.5, "decoy must be correlated with named"
        assert abs(corr[c.named, c.substitute]) <= 0.2, "1-hop substitute must be ~uncorrelated"
        assert abs(corr[c.named, c.terminal]) <= 0.2, "2-hop terminal must be ~uncorrelated"
        # A->B shares the R1 ("substitution") code; B->C shares the R2 ("supply") code
        assert _cos(feats[c.named, r1], feats[c.substitute, r1]) >= 0.5, "A,B share R1 code"
        assert _cos(feats[c.substitute, r2], feats[c.terminal, r2]) >= 0.5, "B,C share R2 code"


def test_generate_is_seed_deterministic() -> None:  # T-PROP-SIM-2
    world = build_world(PropagationSimConfig(seed=0))
    a = generate_events(world, 64, np.random.default_rng(1))
    b = generate_events(world, 64, np.random.default_rng(1))
    assert isinstance(a, EventBatch)
    assert a.reactions.shape == (64, world.config.n_assets)
    assert np.array_equal(a.named_idx, b.named_idx)
    assert np.allclose(a.reactions, b.reactions)


def test_two_hop_causal_edges() -> None:  # T-PROP-SIM-3
    world = build_world(PropagationSimConfig(seed=0, idiosyncratic_vol=0.0, factor_vol=0.0))
    batch = generate_events(world, 2000, np.random.default_rng(2), allowed_types=(0,))
    cfg = world.config
    chain = world.chains[0]
    sign = world.regime_signs[batch.regime]
    expected_b = sign * cfg.propagation_gain * batch.merit
    expected_c = sign * cfg.propagation_gain * cfg.propagation_gain2 * batch.merit
    assert np.allclose(batch.reactions[:, chain.substitute], expected_b, atol=1e-9)
    assert np.allclose(batch.reactions[:, chain.terminal], expected_c, atol=1e-9)
    assert np.allclose(batch.reactions[:, chain.decoy], 0.0, atol=1e-9)


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
