"""T-PROP-UNROLL-*: unroll-composition (Path C)."""

from __future__ import annotations

import numpy as np

from qts.propagation.sim import (
    PropagationSimConfig,
    build_world,
    generate_chain_eval,
    generate_hop_events,
)


def test_hop_events_name_only_sources_and_fire_one_edge() -> None:  # T-PROP-UNROLL-1
    world = build_world(PropagationSimConfig(seed=0))
    n = world.config.n_event_types
    rng = np.random.default_rng(0)
    batch = generate_hop_events(world, 4000, rng)
    # named nodes are only A-role [0:n] or B-role [n:2n] — never terminal/decoy
    assert np.all(batch.named_idx < 2 * n)
    # both A and B are actually used as sources (so R1 and R2 are both trained)
    assert np.any(batch.named_idx < n) and np.any(batch.named_idx >= n)
    # the named node carries its own merit bump — a substantial component of its reaction, though
    # factor + idiosyncratic noise (present on every asset, named included) cap the correlation
    rows = np.arange(len(batch))
    named_react = batch.reactions[rows, batch.named_idx]
    assert np.corrcoef(named_react, batch.merit)[0, 1] > 0.5


def test_chain_eval_terminal_is_iterated_one_hop() -> None:  # T-PROP-UNROLL-2
    """Per-hop signing => r_C == gain1*gain2*merit (sign cancels), matching the unroll."""
    world = build_world(PropagationSimConfig(seed=1, idiosyncratic_vol=0.0, factor_vol=0.0))
    n = world.config.n_event_types
    rng = np.random.default_rng(1)
    batch = generate_chain_eval(world, 500, rng)
    rows = np.arange(len(batch))
    term_idx = np.array([2 * n + k for k in batch.event_type])
    r_c = batch.reactions[rows, term_idx]
    g1, g2 = world.config.propagation_gain, world.config.propagation_gain2
    np.testing.assert_allclose(r_c, g1 * g2 * batch.merit, atol=1e-6)
