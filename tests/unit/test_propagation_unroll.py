"""T-PROP-UNROLL-*: unroll-composition (Path C)."""

from __future__ import annotations

import numpy as np

from qts.propagation.sim import (
    PropagationSimConfig,
    build_world,
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
