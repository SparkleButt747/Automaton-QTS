"""T-PROP-UNROLL-*: unroll-composition (Path C)."""

from __future__ import annotations

import numpy as np

from qts.propagation.sim import (
    PropagationSimConfig,
    build_world,
    generate_chain_eval,
    generate_hop_events,
    make_unroll_splits,
)
from qts.propagation.unroll import unroll_predict


class _DoublingOperator:
    """Stub: predict_np returns array where index (named+1) gets 2x the merit, rest zero."""

    def __init__(self, n_assets: int) -> None:
        self.n_assets = n_assets

    def predict_np(self, named_idx, merit, regime):  # type: ignore[no-untyped-def]
        out = np.zeros((len(named_idx), self.n_assets))
        nxt = (np.asarray(named_idx) + 1) % self.n_assets
        out[np.arange(len(named_idx)), nxt] = 2.0 * np.asarray(merit, dtype=float)
        return out


def test_unroll_predict_composes_hops() -> None:  # T-PROP-UNROLL-4
    op = _DoublingOperator(n_assets=5)
    named = np.array([0, 0])
    merit = np.array([1.0, 3.0])
    regime = np.array([0, 0])
    # hop1 successor index 1 (=2*merit), hop2 successor index 2 (=2*r_b = 4*merit)
    r_term = unroll_predict(op, named, merit, regime, [np.array([1, 1]), np.array([2, 2])])
    np.testing.assert_allclose(r_term, 4.0 * merit)


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


def test_unroll_splits_hold_out_last_chain() -> None:  # T-PROP-UNROLL-3
    world = build_world(PropagationSimConfig(seed=0))
    n = world.config.n_event_types
    hop_train, hop_val, chain_test, chain_transfer = make_unroll_splits(
        world, np.random.default_rng(0), n_train=2000, n_val=500, n_test=500, n_transfer=500
    )
    # training/val/in-sample-test never see the held-out chain n-1
    for b in (hop_train, hop_val, chain_test):
        assert np.all(b.event_type < n - 1)
    # transfer is exactly the held-out chain
    assert np.all(chain_transfer.event_type == n - 1)
    # training is 1-hop (A or B sources only); chain_transfer is a full chain (named is A-role only)
    assert np.all(hop_train.named_idx < 2 * n)
    assert np.all(chain_transfer.named_idx < n)
