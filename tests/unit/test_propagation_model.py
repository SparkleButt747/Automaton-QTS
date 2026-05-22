"""T-PROP-MODEL-1..2: forward shapes, do()-clamp, gradient flow."""

from __future__ import annotations

import numpy as np
import torch

from qts.propagation.model import GatedPropagationGraph
from qts.propagation.sim import PropagationSimConfig, build_world, generate_events


def _model(world) -> GatedPropagationGraph:
    torch.manual_seed(0)
    return GatedPropagationGraph(world.features)


def test_forward_shape_and_clamp() -> None:  # T-PROP-MODEL-1
    world = build_world(PropagationSimConfig(seed=0))
    model = _model(world)
    batch = generate_events(world, 16, np.random.default_rng(0))
    named = torch.as_tensor(batch.named_idx, dtype=torch.long)
    merit = torch.as_tensor(batch.merit, dtype=torch.float32)
    regime = torch.as_tensor(batch.regime, dtype=torch.long)
    out = model(named, merit, regime)
    assert out.shape == (16, world.config.n_assets)
    rows = torch.arange(16)
    assert torch.allclose(out[rows, named], merit, atol=1e-5)  # do() clamp holds


def test_gradients_flow() -> None:  # T-PROP-MODEL-2
    world = build_world(PropagationSimConfig(seed=0))
    model = _model(world)
    batch = generate_events(world, 16, np.random.default_rng(0))
    out = model(
        torch.as_tensor(batch.named_idx, dtype=torch.long),
        torch.as_tensor(batch.merit, dtype=torch.float32),
        torch.as_tensor(batch.regime, dtype=torch.long),
    )
    target = torch.as_tensor(batch.reactions, dtype=torch.float32)
    torch.nn.functional.mse_loss(out, target).backward()
    assert model.M.grad is not None and torch.isfinite(model.M.grad).all()
    assert model.concept_features.grad is not None
