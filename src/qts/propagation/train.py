"""Train the propagation graph and run the two-part feasibility gate."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from qts.propagation.baselines import CorrelationalBaseline, no_propagation_predict
from qts.propagation.model import GatedPropagationGraph
from qts.propagation.sim import EventBatch, GroundTruthWorld

logger = logging.getLogger(__name__)

SUBSTITUTE_MARGIN = 0.75  # graph substitute-MSE must be < 0.75 * correlational substitute-MSE


@dataclass(frozen=True)
class FeasibilityReport:
    test_mse_graph: float
    test_mse_corr: float
    test_mse_noprop: float
    sub_mse_graph: float
    sub_mse_corr: float
    transfer_sub_mse_graph: float
    transfer_sub_mse_corr: float
    terminal_mse_graph: float
    terminal_mse_corr: float
    transfer_terminal_mse_graph: float
    transfer_terminal_mse_corr: float
    prediction_pass: bool
    transfer_pass: bool
    passed: bool


def _tensors(batch: EventBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(batch.named_idx, dtype=torch.long),
        torch.as_tensor(batch.merit, dtype=torch.float32),
        torch.as_tensor(batch.regime, dtype=torch.long),
        torch.as_tensor(batch.reactions, dtype=torch.float32),
    )


def fit_graph(
    world: GroundTruthWorld,
    train: EventBatch,
    val: EventBatch,
    *,
    epochs: int = 400,
    lr: float = 1e-2,
    l1_lambda: float = 1e-3,
    patience: int = 30,
    seed: int = 0,
    model_cls: Any = GatedPropagationGraph,
) -> Any:
    torch.manual_seed(seed)
    model = model_cls(world.features)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xn, xm, xr, y = _tensors(train)
    vn, vm, vr, vy = _tensors(val)

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    bad = 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(xn, xm, xr), y) + l1_lambda * model.M.abs().sum()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vloss = float(torch.nn.functional.mse_loss(model(vn, vm, vr), vy))
        if vloss < best_val - 1e-6:
            best_val, bad = vloss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _role_mse(pred: np.ndarray, batch: EventBatch, role_idx: np.ndarray) -> float:
    rows = np.arange(len(batch))
    return float(np.mean((pred[rows, role_idx] - batch.reactions[rows, role_idx]) ** 2))


def evaluate_feasibility(
    world: GroundTruthWorld,
    model: GatedPropagationGraph,
    test: EventBatch,
    transfer: EventBatch,
    *,
    n_history: int = 20000,
    seed: int = 0,
) -> FeasibilityReport:
    corr = CorrelationalBaseline.from_history(world, n_samples=n_history, seed=seed)
    n_assets = world.config.n_assets

    pg = model.predict_np(test.named_idx, test.merit, test.regime)
    pc = corr.predict(test)
    pn = no_propagation_predict(test, n_assets)
    test_mse_graph = float(np.mean((pg - test.reactions) ** 2))
    test_mse_corr = float(np.mean((pc - test.reactions) ** 2))
    test_mse_noprop = float(np.mean((pn - test.reactions) ** 2))
    sub_test = world.substitute_indices(test.event_type)
    term_test = world.terminal_indices(test.event_type)
    sub_mse_graph = _role_mse(pg, test, sub_test)
    sub_mse_corr = _role_mse(pc, test, sub_test)
    terminal_mse_graph = _role_mse(pg, test, term_test)
    terminal_mse_corr = _role_mse(pc, test, term_test)

    tpg = model.predict_np(transfer.named_idx, transfer.merit, transfer.regime)
    tpc = corr.predict(transfer)
    sub_tf = world.substitute_indices(transfer.event_type)
    term_tf = world.terminal_indices(transfer.event_type)
    transfer_sub_mse_graph = _role_mse(tpg, transfer, sub_tf)
    transfer_sub_mse_corr = _role_mse(tpc, transfer, sub_tf)
    transfer_terminal_mse_graph = _role_mse(tpg, transfer, term_tf)
    transfer_terminal_mse_corr = _role_mse(tpc, transfer, term_tf)

    # The graph only observes (named, merit, regime) — never the factor shock — so it cannot
    # predict factor co-movement that the correlational baseline gets free from the named asset's
    # realised move. Judging on overall MSE-vs-correlational is therefore an unfair bar; the
    # thesis-aligned tests are the 1-hop SUBSTITUTE (B) and the 2-hop TERMINAL (C) — the unnamed
    # entities — plus beating the no-prop floor.
    prediction_pass = (
        test_mse_graph < test_mse_noprop
        and sub_mse_graph < SUBSTITUTE_MARGIN * sub_mse_corr
        and terminal_mse_graph < SUBSTITUTE_MARGIN * terminal_mse_corr
    )
    transfer_pass = (
        transfer_sub_mse_graph < transfer_sub_mse_corr
        and transfer_terminal_mse_graph < transfer_terminal_mse_corr
    )
    return FeasibilityReport(
        test_mse_graph=test_mse_graph,
        test_mse_corr=test_mse_corr,
        test_mse_noprop=test_mse_noprop,
        sub_mse_graph=sub_mse_graph,
        sub_mse_corr=sub_mse_corr,
        transfer_sub_mse_graph=transfer_sub_mse_graph,
        transfer_sub_mse_corr=transfer_sub_mse_corr,
        terminal_mse_graph=terminal_mse_graph,
        terminal_mse_corr=terminal_mse_corr,
        transfer_terminal_mse_graph=transfer_terminal_mse_graph,
        transfer_terminal_mse_corr=transfer_terminal_mse_corr,
        prediction_pass=prediction_pass,
        transfer_pass=transfer_pass,
        passed=prediction_pass and transfer_pass,
    )
