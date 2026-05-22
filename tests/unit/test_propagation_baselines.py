"""T-PROP-BASE-1..2: correlational and no-propagation baselines."""

from __future__ import annotations

import numpy as np

from qts.propagation.baselines import CorrelationalBaseline, no_propagation_predict
from qts.propagation.sim import PropagationSimConfig, build_world, generate_events


def test_no_propagation_only_named_moves() -> None:  # T-PROP-BASE-2
    world = build_world(PropagationSimConfig(seed=0))
    batch = generate_events(world, 16, np.random.default_rng(0), allowed_types=(0,))
    pred = no_propagation_predict(batch, world.config.n_assets)
    rows = np.arange(len(batch))
    assert np.allclose(pred[rows, batch.named_idx], batch.merit)
    pred[rows, batch.named_idx] = 0.0
    assert np.allclose(pred, 0.0)


def test_correlational_chases_decoy_misses_substitute() -> None:  # T-PROP-BASE-1
    world = build_world(PropagationSimConfig(seed=0))
    base = CorrelationalBaseline.from_history(world, n_samples=20000, seed=0)
    batch = generate_events(world, 4000, np.random.default_rng(1), allowed_types=(0,))
    pred = base.predict(batch)
    sub, decoy = world.triples[0].substitute, world.triples[0].decoy
    truth = batch.reactions
    mse_sub = float(np.mean((pred[:, sub] - truth[:, sub]) ** 2))
    mse_decoy = float(np.mean((pred[:, decoy] - truth[:, decoy]) ** 2))
    assert mse_decoy < mse_sub
