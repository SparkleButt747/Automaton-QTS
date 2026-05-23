"""T-PROP-UNROLL-GATE-*: end-to-end unroll-composition gate.

Path C (compose a 1-hop operator at inference) is a DOCUMENTED NEGATIVE RESULT (design doc §14.2).
The unroll's second hop (R2/B->C) does not fire on an unseen chain, so 2-hop transfer (<=2/6 seeds)
is dominated by end-to-end and does not hold. GATE-1 asserts the report is well-formed; GATE-2 is
``xfail`` recording that the unroll does not achieve robust composition transfer.
"""

from __future__ import annotations

import numpy as np
import pytest

from qts.propagation.model import GatedPropagationGraph
from qts.propagation.sim import PropagationSimConfig, build_world, make_unroll_splits
from qts.propagation.train import fit_graph
from qts.propagation.unroll import UnrollReport, evaluate_unroll_transfer


def _fit_and_eval(model_cls, seed: int, *, n_train: int, epochs: int) -> UnrollReport:  # type: ignore[no-untyped-def]
    world = build_world(PropagationSimConfig(seed=seed))
    hop_train, hop_val, _, chain_transfer = make_unroll_splits(
        world, np.random.default_rng(seed), n_train=n_train, n_val=500, n_transfer=500
    )
    model = fit_graph(world, hop_train, hop_val, epochs=epochs, seed=seed, model_cls=model_cls)
    return evaluate_unroll_transfer(world, model, chain_transfer, n_history=20000, seed=seed)


def test_unroll_report_well_formed() -> None:  # T-PROP-UNROLL-GATE-1
    report = _fit_and_eval(GatedPropagationGraph, seed=0, n_train=400, epochs=20)
    assert isinstance(report, UnrollReport)
    for v in (report.hop1_mse_graph, report.terminal_mse_graph, report.terminal_mse_corr):
        assert np.isfinite(v)


@pytest.mark.xfail(
    reason=(
        "Path C negative result (design doc §14.2): composing the 1-hop operator by unrolling does "
        "NOT achieve robust 2-hop transfer (<=2/6 seeds; dominated by end-to-end). The 2nd hop "
        "(R2/B->C) does not fire on an unseen chain. Wall is inductive bias, not the mechanism."
    ),
    strict=False,
)
def test_unroll_composition_transfers() -> None:  # T-PROP-UNROLL-GATE-2
    report = _fit_and_eval(GatedPropagationGraph, seed=0, n_train=8000, epochs=600)
    assert report.transfer_pass
