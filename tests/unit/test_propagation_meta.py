"""T-PROP-META-*: meta-learned propagation (episodic relation-resampling, design §15)."""

from __future__ import annotations

import numpy as np

from qts.propagation.meta import (
    MetaPropagationGraph,
    MetaTransferReport,
    evaluate_meta_transfer,
    few_shot_adapt,
    train_meta,
)
from qts.propagation.sim import PropagationSimConfig, build_world, generate_events


def test_forward_clamps_named_node_to_merit() -> None:  # T-PROP-META-1
    world = build_world(PropagationSimConfig(seed=0))
    model = MetaPropagationGraph(world.features.shape[1])
    rng = np.random.default_rng(0)
    batch = generate_events(world, 16, rng)
    pred = model.predict_np(world.features, batch.named_idx, batch.merit, batch.regime)
    assert pred.shape == (16, world.config.n_assets)
    # the do()-clamp re-pins the named node every step => its output is exactly its merit
    rows = np.arange(len(batch))
    np.testing.assert_allclose(pred[rows, batch.named_idx], batch.merit, atol=1e-5)


def test_train_meta_returns_well_formed_report() -> None:  # T-PROP-META-2
    model = train_meta(range(4), steps=200, batch_size=64, seed=0)
    assert isinstance(model, MetaPropagationGraph)
    held_out = build_world(PropagationSimConfig(seed=999))
    eval_batch = generate_events(held_out, 500, np.random.default_rng(7))
    report = evaluate_meta_transfer(model, held_out, eval_batch, n_history=4000, seed=0)
    assert isinstance(report, MetaTransferReport)
    for v in (
        report.sub_mse_graph,
        report.sub_mse_corr,
        report.terminal_mse_graph,
        report.terminal_mse_corr,
        report.sub_capture,
        report.terminal_capture,
    ):
        assert np.isfinite(v)
    assert isinstance(report.sub_win, bool) and isinstance(report.terminal_win, bool)


def test_few_shot_adapt_fits_support_without_mutating_base() -> None:  # T-PROP-META-3
    world = build_world(PropagationSimConfig(seed=1))
    base = train_meta(range(3), steps=100, batch_size=64, seed=1)
    base_m_before = base.M.detach().clone()
    support = generate_events(world, 128, np.random.default_rng(3))

    def _support_mse(model: MetaPropagationGraph) -> float:
        pred = model.predict_np(world.features, support.named_idx, support.merit, support.regime)
        rows = np.arange(len(support))
        # supervise on the substitute (the 1-hop target the adapter should fit)
        sub = world.substitute_indices(support.event_type)
        return float(np.mean((pred[rows, sub] - support.reactions[rows, sub]) ** 2))

    adapted = few_shot_adapt(base, world.features, support, steps=60)
    assert adapted is not base
    # adaptation reduces error on the support world's substitute
    assert _support_mse(adapted) < _support_mse(base)
    # the base operator is untouched (few_shot_adapt copies into a fresh module)
    np.testing.assert_allclose(base.M.detach().numpy(), base_m_before.numpy())


def test_meta_public_api_exported() -> None:  # T-PROP-META-4
    import qts.propagation as p

    for name in (
        "MetaPropagationGraph",
        "MetaTransferReport",
        "train_meta",
        "few_shot_adapt",
        "evaluate_meta_transfer",
    ):
        assert hasattr(p, name), name
