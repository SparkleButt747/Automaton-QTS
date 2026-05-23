# Propagation Unroll-Composition (Path C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reach the 2-hop terminal C on a held-out chain by composing the transferable 1-hop operator at inference (unroll), instead of training a 2-hop A→C mapping (which doesn't transfer — design doc §13).

**Architecture:** Train the operator on **1-hop events only** (naming both A and B so R1 and R2 are each learned as local maps), holding out chain `n-1`. At inference, **unroll**: predict B from A, re-inject the prediction as merit, predict C. Eval reads **known successor indices** and uses a **per-hop-signed** ground truth so `r_C = gain1·gain2·merit` matches what iterated 1-hop produces. Compared across **both** operators (linear `GatedPropagationGraph` and `GraphNeuralODE`).

**Tech Stack:** numpy + plain PyTorch (both already deps), pytest with `--no-cov` locally. Spec: `docs/specs/2026-05-23-propagation-unroll-composition.md`.

**Run prefix (all commands):** `.venv/bin/python -m pytest <path> -v --no-cov -p no:cacheprovider`

---

## File Structure

- `src/qts/propagation/sim.py` — ADD `generate_hop_events`, `generate_chain_eval`, `make_unroll_splits`. **Do NOT touch** `generate_events`/`make_splits` (they back the committed §13 negative result + its pinned tests).
- `src/qts/propagation/unroll.py` — NEW. `unroll_predict` (model-agnostic), `UnrollReport`, `evaluate_unroll_transfer`.
- `src/qts/propagation/__init__.py` — export the new public names.
- `scripts/run_propagation_unroll.py` — NEW. Fit both operators on 1-hop data, run the unroll eval across `--seeds`, print reports.
- `tests/unit/test_propagation_unroll.py` — NEW. T-PROP-UNROLL-1..5.
- `tests/integration/test_propagation_unroll_gate.py` — NEW. T-PROP-UNROLL-GATE-1..3.

**Role-index invariant (used by the vectorised generators):** for `n = n_event_types`, chain `k` has `named=k`, `substitute=n+k`, `terminal=2n+k`, `decoy=3n+k` (see `EventChain` construction in `sim.py`).

---

### Task 1: `generate_hop_events` — 1-hop training events

**Files:**
- Modify: `src/qts/propagation/sim.py` (add after `generate_events`)
- Test: `tests/unit/test_propagation_unroll.py`

- [ ] **Step 1: Write the failing test**

```python
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
    # the named node always carries its own merit bump (within factor noise it correlates with merit)
    rows = np.arange(len(batch))
    named_react = batch.reactions[rows, batch.named_idx]
    assert np.corrcoef(named_react, batch.merit)[0, 1] > 0.8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_propagation_unroll.py::test_hop_events_name_only_sources_and_fire_one_edge -v --no-cov -p no:cacheprovider`
Expected: FAIL with `ImportError: cannot import name 'generate_hop_events'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/qts/propagation/sim.py`:

```python
def generate_hop_events(
    world: GroundTruthWorld,
    n: int,
    rng: np.random.Generator,
    allowed_types: tuple[int, ...] | None = None,
) -> EventBatch:
    """1-hop training events: name a relation-bearing source (A or B), supervise its direct successor.

    Each event fires exactly one edge — A->B (gain1) or B->C (gain2). Naming B is what lets R2 be
    learned at all; without it the unroll's second hop fires an untrained edge (spec §2).
    """
    cfg = world.config
    nt = cfg.n_event_types
    types = np.arange(nt) if allowed_types is None else np.array(allowed_types)
    event_type = rng.choice(types, size=n)
    hop = rng.integers(0, 2, size=n)  # 0: A->B (gain1), 1: B->C (gain2)
    regime = rng.integers(0, cfg.n_regimes, size=n)
    g = rng.normal(0.0, cfg.factor_vol, (n, cfg.n_factors))
    merit = rng.normal(0.0, cfg.merit_vol, n)
    eps = rng.normal(0.0, cfg.idiosyncratic_vol, (n, cfg.n_assets))
    reactions = g @ world.loadings.T + eps

    named = np.where(hop == 0, event_type, nt + event_type)  # A_k or B_k
    succ = np.where(hop == 0, nt + event_type, 2 * nt + event_type)  # B_k or C_k
    gain = np.where(hop == 0, cfg.propagation_gain, cfg.propagation_gain2)
    rows = np.arange(n)
    sign = world.regime_signs[regime]
    reactions[rows, named] += merit
    reactions[rows, succ] += sign * gain * merit
    return EventBatch(
        named_idx=named, merit=merit, regime=regime, reactions=reactions, event_type=event_type
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_propagation_unroll.py::test_hop_events_name_only_sources_and_fire_one_edge -v --no-cov -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qts/propagation/sim.py tests/unit/test_propagation_unroll.py
git commit -m "feat(propagation): 1-hop event generator for unroll-composition (Path C)"
```

---

### Task 2: `generate_chain_eval` — per-hop-signed 2-hop ground truth

**Files:**
- Modify: `src/qts/propagation/sim.py` (add after `generate_hop_events`)
- Test: `tests/unit/test_propagation_unroll.py`

- [ ] **Step 1: Write the failing test**

```python
from qts.propagation.sim import generate_chain_eval  # add to imports at top of file


def test_chain_eval_terminal_is_iterated_one_hop() -> None:  # T-PROP-UNROLL-2
    """Per-hop signing => r_C == gain1*gain2*merit (sign cancels), matching what the unroll produces."""
    world = build_world(PropagationSimConfig(seed=1, idiosyncratic_vol=0.0, factor_vol=0.0))
    n = world.config.n_event_types
    rng = np.random.default_rng(1)
    batch = generate_chain_eval(world, 500, rng)
    rows = np.arange(len(batch))
    term_idx = np.array([2 * n + k for k in batch.event_type])
    r_c = batch.reactions[rows, term_idx]
    g1, g2 = world.config.propagation_gain, world.config.propagation_gain2
    np.testing.assert_allclose(r_c, g1 * g2 * batch.merit, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_propagation_unroll.py::test_chain_eval_terminal_is_iterated_one_hop -v --no-cov -p no:cacheprovider`
Expected: FAIL with `ImportError: cannot import name 'generate_chain_eval'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/qts/propagation/sim.py`:

```python
def generate_chain_eval(
    world: GroundTruthWorld,
    n: int,
    rng: np.random.Generator,
    allowed_types: tuple[int, ...] | None = None,
) -> EventBatch:
    """Full 2-hop chains with PER-HOP regime signing, so ground truth == iterated 1-hop (spec §3).

    r_B = sign*gain1*merit ; r_C = sign*gain2*r_B = gain1*gain2*merit. Used only to score the unroll;
    this does NOT replace ``generate_events`` (which backs the committed §13 result).
    """
    cfg = world.config
    types = np.arange(cfg.n_event_types) if allowed_types is None else np.array(allowed_types)
    event_type = rng.choice(types, size=n)
    regime = rng.integers(0, cfg.n_regimes, size=n)
    g = rng.normal(0.0, cfg.factor_vol, (n, cfg.n_factors))
    merit = rng.normal(0.0, cfg.merit_vol, n)
    eps = rng.normal(0.0, cfg.idiosyncratic_vol, (n, cfg.n_assets))
    reactions = g @ world.loadings.T + eps
    named = np.array([world.chains[k].named for k in event_type])
    sub = np.array([world.chains[k].substitute for k in event_type])
    term = np.array([world.chains[k].terminal for k in event_type])
    rows = np.arange(n)
    sign = world.regime_signs[regime]
    r_b = sign * cfg.propagation_gain * merit
    reactions[rows, named] += merit
    reactions[rows, sub] += r_b
    reactions[rows, term] += sign * cfg.propagation_gain2 * r_b  # per-hop sign -> gain1*gain2*merit
    return EventBatch(
        named_idx=named, merit=merit, regime=regime, reactions=reactions, event_type=event_type
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_propagation_unroll.py::test_chain_eval_terminal_is_iterated_one_hop -v --no-cov -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qts/propagation/sim.py tests/unit/test_propagation_unroll.py
git commit -m "feat(propagation): per-hop-signed chain eval generator (composition-consistent ground truth)"
```

---

### Task 3: `make_unroll_splits` — assemble train/val/test/transfer

**Files:**
- Modify: `src/qts/propagation/sim.py` (add after `generate_chain_eval`)
- Test: `tests/unit/test_propagation_unroll.py`

- [ ] **Step 1: Write the failing test**

```python
from qts.propagation.sim import make_unroll_splits  # add to imports


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
    # training is 1-hop (sources are A or B only); chain_* are full chains (named is A-role only)
    assert np.all(hop_train.named_idx < 2 * n)
    assert np.all(chain_transfer.named_idx < n)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_propagation_unroll.py::test_unroll_splits_hold_out_last_chain -v --no-cov -p no:cacheprovider`
Expected: FAIL with `ImportError: cannot import name 'make_unroll_splits'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/qts/propagation/sim.py`:

```python
def make_unroll_splits(
    world: GroundTruthWorld,
    rng: np.random.Generator,
    *,
    n_train: int = 8000,
    n_val: int = 1000,
    n_test: int = 1000,
    n_transfer: int = 1000,
) -> tuple[EventBatch, EventBatch, EventBatch, EventBatch]:
    """1-hop training (chains 0..n-2), per-hop-signed chain eval (in-sample on trained, transfer on held-out).

    Returns (hop_train, hop_val, chain_test, chain_transfer). ``hop_*`` are 1-hop events naming A and
    B; ``chain_*`` are full 2-hop chains for scoring the unroll. n_train defaults higher than the
    end-to-end splits because there are 2x the (source, relation) combinations to cover.
    """
    n = world.config.n_event_types
    train_types = tuple(range(n - 1))
    transfer_types = (n - 1,)
    hop_train = generate_hop_events(world, n_train, rng, allowed_types=train_types)
    hop_val = generate_hop_events(world, n_val, rng, allowed_types=train_types)
    chain_test = generate_chain_eval(world, n_test, rng, allowed_types=train_types)
    chain_transfer = generate_chain_eval(world, n_transfer, rng, allowed_types=transfer_types)
    return hop_train, hop_val, chain_test, chain_transfer
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_propagation_unroll.py::test_unroll_splits_hold_out_last_chain -v --no-cov -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qts/propagation/sim.py tests/unit/test_propagation_unroll.py
git commit -m "feat(propagation): make_unroll_splits (1-hop train, per-hop-signed eval, held-out chain)"
```

---

### Task 4: `unroll_predict` — model-agnostic external unroll

**Files:**
- Create: `src/qts/propagation/unroll.py`
- Test: `tests/unit/test_propagation_unroll.py`

- [ ] **Step 1: Write the failing test**

```python
from qts.propagation.unroll import unroll_predict  # add to imports


class _DoublingOperator:
    """Stub: predict_np returns an array whose value at index (i+1) is 2x the merit (rest zero)."""

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_propagation_unroll.py::test_unroll_predict_composes_hops -v --no-cov -p no:cacheprovider`
Expected: FAIL with `ModuleNotFoundError: No module named 'qts.propagation.unroll'`

- [ ] **Step 3: Write minimal implementation**

Create `src/qts/propagation/unroll.py`:

```python
"""Compose the transferable 1-hop operator at inference (Path C). See spec 2026-05-23-...-unroll.

End-to-end 2-hop training does not transfer (design doc §13). Here the operator is trained on 1-hop
relations only; the 2-hop terminal is reached by UNROLLING — predict B from A, re-inject as merit,
predict C. Each hop is a bona-fide 1-hop prediction, the operation v0 proved transfers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from qts.propagation.baselines import CorrelationalBaseline, no_propagation_predict
from qts.propagation.sim import EventBatch, GroundTruthWorld


class Operator(Protocol):
    """Anything with the propagation operator's call signature (linear graph or neural-ODE)."""

    def predict_np(
        self, named_idx: np.ndarray, merit: np.ndarray, regime: np.ndarray
    ) -> np.ndarray: ...


def unroll_predict(
    model: Operator,
    named_idx: np.ndarray,
    merit: np.ndarray,
    regime: np.ndarray,
    hop_successors: list[np.ndarray],
) -> np.ndarray:
    """Iterate the 1-hop operator along known successor indices; return the terminal reaction.

    ``hop_successors[h]`` is the per-row index of hop h's successor (read with known indices, spec §3).
    Hop h names the previous successor and injects its predicted reaction as the next merit.
    """
    rows = np.arange(len(named_idx))
    src = np.asarray(named_idx)
    m = np.asarray(merit, dtype=float)
    r_succ = m
    for succ in hop_successors:
        out = model.predict_np(src, m, regime)
        r_succ = out[rows, succ]
        src, m = succ, r_succ
    return r_succ
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_propagation_unroll.py::test_unroll_predict_composes_hops -v --no-cov -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qts/propagation/unroll.py tests/unit/test_propagation_unroll.py
git commit -m "feat(propagation): model-agnostic external unroll operator"
```

---

### Task 5: `UnrollReport` + `evaluate_unroll_transfer`

**Files:**
- Modify: `src/qts/propagation/unroll.py`
- Test: `tests/unit/test_propagation_unroll.py`

- [ ] **Step 1: Write the failing test**

```python
from qts.propagation.unroll import UnrollReport, evaluate_unroll_transfer  # add to imports


class _OracleOperator:
    """Stub operator that reproduces per-hop-signed ground truth exactly: successor = sign*gain*merit."""

    def __init__(self, world) -> None:  # type: ignore[no-untyped-def]
        self.world = world

    def predict_np(self, named_idx, merit, regime):  # type: ignore[no-untyped-def]
        w = self.world
        cfg = w.config
        out = np.zeros((len(named_idx), cfg.n_assets))
        sign = w.regime_signs[regime]
        nt = cfg.n_event_types
        named = np.asarray(named_idx)
        is_a = named < nt
        succ = np.where(is_a, nt + named % nt, 2 * nt + (named - nt) % nt)
        gain = np.where(is_a, cfg.propagation_gain, cfg.propagation_gain2)
        rows = np.arange(len(named))
        out[rows, succ] = sign * gain * np.asarray(merit, dtype=float)
        return out


def test_evaluate_unroll_transfer_oracle_passes() -> None:  # T-PROP-UNROLL-5
    world = build_world(PropagationSimConfig(seed=0))
    _, _, _, chain_transfer = make_unroll_splits(world, np.random.default_rng(0), n_transfer=500)
    report = evaluate_unroll_transfer(world, _OracleOperator(world), chain_transfer, seed=0)
    assert isinstance(report, UnrollReport)
    assert report.terminal_mse_graph < 1e-6  # oracle reproduces C exactly
    assert report.transfer_pass and report.hop1_pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_propagation_unroll.py::test_evaluate_unroll_transfer_oracle_passes -v --no-cov -p no:cacheprovider`
Expected: FAIL with `ImportError: cannot import name 'UnrollReport'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/qts/propagation/unroll.py`:

```python
@dataclass(frozen=True)
class UnrollReport:
    hop1_mse_graph: float
    hop1_mse_corr: float
    terminal_mse_graph: float
    terminal_mse_corr: float
    terminal_mse_noprop: float
    hop1_pass: bool
    transfer_pass: bool


def evaluate_unroll_transfer(
    world: GroundTruthWorld,
    model: Operator,
    transfer: EventBatch,
    *,
    n_history: int = 20000,
    seed: int = 0,
) -> UnrollReport:
    """Score the unrolled A->B->C against the correlational baseline + no-prop floor on a chain batch."""
    corr = CorrelationalBaseline.from_history(world, n_samples=n_history, seed=seed)
    rows = np.arange(len(transfer))
    sub = world.substitute_indices(transfer.event_type)
    term = world.terminal_indices(transfer.event_type)

    r_b = unroll_predict(model, transfer.named_idx, transfer.merit, transfer.regime, [sub])
    r_c = unroll_predict(model, transfer.named_idx, transfer.merit, transfer.regime, [sub, term])

    truth_b = transfer.reactions[rows, sub]
    truth_c = transfer.reactions[rows, term]
    pc = corr.predict(transfer)
    pn = no_propagation_predict(transfer, world.config.n_assets)

    hop1_mse_graph = float(np.mean((r_b - truth_b) ** 2))
    hop1_mse_corr = float(np.mean((pc[rows, sub] - truth_b) ** 2))
    terminal_mse_graph = float(np.mean((r_c - truth_c) ** 2))
    terminal_mse_corr = float(np.mean((pc[rows, term] - truth_c) ** 2))
    terminal_mse_noprop = float(np.mean((pn[rows, term] - truth_c) ** 2))

    hop1_pass = hop1_mse_graph < hop1_mse_corr
    transfer_pass = (
        terminal_mse_graph < terminal_mse_corr and terminal_mse_graph < terminal_mse_noprop
    )
    return UnrollReport(
        hop1_mse_graph=hop1_mse_graph,
        hop1_mse_corr=hop1_mse_corr,
        terminal_mse_graph=terminal_mse_graph,
        terminal_mse_corr=terminal_mse_corr,
        terminal_mse_noprop=terminal_mse_noprop,
        hop1_pass=hop1_pass,
        transfer_pass=transfer_pass,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_propagation_unroll.py::test_evaluate_unroll_transfer_oracle_passes -v --no-cov -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qts/propagation/unroll.py tests/unit/test_propagation_unroll.py
git commit -m "feat(propagation): UnrollReport + evaluate_unroll_transfer"
```

---

### Task 6: Package exports

**Files:**
- Modify: `src/qts/propagation/__init__.py`
- Test: `tests/unit/test_propagation_unroll.py`

- [ ] **Step 1: Write the failing test**

```python
def test_unroll_public_api_exported() -> None:  # T-PROP-UNROLL-6
    import qts.propagation as p

    for name in ("unroll_predict", "evaluate_unroll_transfer", "UnrollReport",
                 "generate_hop_events", "generate_chain_eval", "make_unroll_splits"):
        assert hasattr(p, name), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_propagation_unroll.py::test_unroll_public_api_exported -v --no-cov -p no:cacheprovider`
Expected: FAIL (AssertionError on the first missing name)

- [ ] **Step 3: Write minimal implementation**

Edit `src/qts/propagation/__init__.py` — add imports and `__all__` entries (keep alphabetical within each group):

```python
from qts.propagation.sim import (
    EventBatch,
    EventChain,
    GroundTruthWorld,
    PropagationSimConfig,
    build_world,
    generate_chain_eval,
    generate_events,
    generate_hop_events,
    make_splits,
    make_unroll_splits,
)
from qts.propagation.train import FeasibilityReport, evaluate_feasibility, fit_graph
from qts.propagation.unroll import UnrollReport, evaluate_unroll_transfer, unroll_predict
```

And add to `__all__`: `"UnrollReport"`, `"evaluate_unroll_transfer"`, `"generate_chain_eval"`, `"generate_hop_events"`, `"make_unroll_splits"`, `"unroll_predict"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_propagation_unroll.py::test_unroll_public_api_exported -v --no-cov -p no:cacheprovider`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qts/propagation/__init__.py tests/unit/test_propagation_unroll.py
git commit -m "feat(propagation): export unroll-composition public API"
```

---

### Task 7: Integration gate — confident assertions only (hop-1 transfer + well-formed)

**Files:**
- Create: `tests/integration/test_propagation_unroll_gate.py`

This gate asserts only what we are confident about: report well-formedness and **hop-1 transfer** (which is just the v0 1-hop, known robust). The **2-hop composition outcome is NOT asserted here** — it is the experimental question, decided by the Task 9 sweep.

- [ ] **Step 1: Write the test (expected to pass once the operator trains)**

```python
"""T-PROP-UNROLL-GATE-*: end-to-end unroll-composition gate (hop-1 confident; 2-hop measured in sweep)."""

from __future__ import annotations

import numpy as np
import pytest

from qts.propagation.model import GatedPropagationGraph
from qts.propagation.model_ode import GraphNeuralODE
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


@pytest.mark.parametrize("model_cls", [GatedPropagationGraph, GraphNeuralODE])
def test_unroll_hop1_transfers(model_cls) -> None:  # T-PROP-UNROLL-GATE-2  # type: ignore[no-untyped-def]
    """Hop-1 (A->B on the held-out chain) must beat correlational — this is the v0 1-hop, known robust."""
    report = _fit_and_eval(model_cls, seed=0, n_train=8000, epochs=600)
    assert report.hop1_mse_graph < report.hop1_mse_corr
```

- [ ] **Step 2: Run the gate**

Run: `.venv/bin/python -m pytest tests/integration/test_propagation_unroll_gate.py -v --no-cov -p no:cacheprovider`
Expected: GATE-1 PASS. GATE-2 PASS for both models **if hop-1 transfers as expected**. If GATE-2 fails for a model, that is itself a finding (the 1-hop regressed under the hop-event training regime) — STOP and report to the controller before proceeding; do not weaken the assertion to force green.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_propagation_unroll_gate.py
git commit -m "test(propagation): unroll-composition gate (hop-1 transfer + well-formed)"
```

---

### Task 8: CLI — fit both operators, sweep seeds, print reports

**Files:**
- Create: `scripts/run_propagation_unroll.py`

- [ ] **Step 1: Write the script**

```python
"""Run the unroll-composition (Path C) experiment: fit 1-hop, unroll to C, score vs baselines.

Usage:
    .venv/bin/python -m scripts.run_propagation_unroll --seeds 6 --epochs 600
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from qts.propagation.model import GatedPropagationGraph
from qts.propagation.model_ode import GraphNeuralODE
from qts.propagation.sim import PropagationSimConfig, build_world, make_unroll_splits
from qts.propagation.train import fit_graph
from qts.propagation.unroll import evaluate_unroll_transfer

MODELS = {"linear": GatedPropagationGraph, "ode": GraphNeuralODE}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--n-train", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for name, cls in MODELS.items():
        hop1, transfer = 0, 0
        print(f"\n=== {name} ===")
        for seed in range(args.seeds):
            world = build_world(PropagationSimConfig(seed=seed))
            ht, hv, _, tf = make_unroll_splits(
                world, np.random.default_rng(seed), n_train=args.n_train, n_transfer=500
            )
            model = fit_graph(world, ht, hv, epochs=args.epochs, seed=seed, model_cls=cls)
            r = evaluate_unroll_transfer(world, model, tf, seed=seed)
            hop1 += r.hop1_pass
            transfer += r.transfer_pass
            print(
                f"  seed {seed}: hop1={r.hop1_pass} transfer={r.transfer_pass} "
                f"| termC graph={r.terminal_mse_graph:.3f} corr={r.terminal_mse_corr:.3f} "
                f"noprop={r.terminal_mse_noprop:.3f}",
                flush=True,
            )
        print(f"  {name}: hop1 {hop1}/{args.seeds}, TRANSFER {transfer}/{args.seeds}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run (cheap) to verify it executes**

Run: `.venv/bin/python -m scripts.run_propagation_unroll --seeds 1 --epochs 20`
Expected: prints a `linear` and an `ode` block with one seed line each; no exceptions.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_propagation_unroll.py
git commit -m "feat(propagation): unroll-composition experiment CLI (both operators, seed sweep)"
```

---

### Task 9: Run the sweep, interpret, finalise

This task has a **branch**: the 2-hop transfer result is unknown until measured.

- [ ] **Step 1: Run the full sweep (background; ~minutes for the ODE)**

Run: `.venv/bin/python -m scripts.run_propagation_unroll --seeds 6 --epochs 600`
Record the `hop1 X/6` and `TRANSFER X/6` lines for both `linear` and `ode`.

- [ ] **Step 2: Branch on the result**

- **If `TRANSFER ≥ 5/6` for an operator (PASS — Path C works):** add a strict transfer assertion to the gate for that operator at a seed that passed, e.g. append to `tests/integration/test_propagation_unroll_gate.py`:

```python
@pytest.mark.parametrize("model_cls,seed", [(GatedPropagationGraph, 0)])  # set to the operator+seed observed to pass
def test_unroll_composition_transfers(model_cls, seed) -> None:  # T-PROP-UNROLL-GATE-3  # type: ignore[no-untyped-def]
    report = _fit_and_eval(model_cls, seed=seed, n_train=8000, epochs=600)
    assert report.transfer_pass
```

  Then run `.venv/bin/python -m pytest tests/integration/test_propagation_unroll_gate.py -v --no-cov -p no:cacheprovider` (expect all PASS) and add a **§14 Findings (Path C — PASS)** entry to `docs/research/2026-05-22-event-propagation-graph-design.md` stating that nth-order propagation is reachable by iterating the transferable 1-hop operator, with the per-operator pass rates.

- **If `TRANSFER < 5/6` (FAIL — does not transfer either):** mark GATE-3 `@pytest.mark.xfail(strict=False, reason=...)` citing the sweep numbers, and write a **§14 Findings (Path C — negative)** entry that localises the wall: did hop-1 hold while hop-2 collapsed (error-compounding) or did hop-1 also fail (naming-B regression)? Quote the `hop1 X/6` vs `TRANSFER X/6` split — that split is the diagnostic.

- [ ] **Step 3: Update the spec status line**

Edit `docs/specs/2026-05-23-propagation-unroll-composition.md` `**Status:**` to `IMPLEMENTED — <PASS|negative result>`, pointing to design-doc §14.

- [ ] **Step 4: Final verification + commit**

Run the whole propagation suite: `.venv/bin/python -m pytest tests/unit/test_propagation_unroll.py tests/integration/test_propagation_unroll_gate.py -v --no-cov -p no:cacheprovider`
Run `.venv/bin/ruff check src/qts/propagation/ scripts/run_propagation_unroll.py tests/unit/test_propagation_unroll.py tests/integration/test_propagation_unroll_gate.py` and `.venv/bin/mypy src/qts/propagation/` — both must be clean.

```bash
git add docs/research/2026-05-22-event-propagation-graph-design.md docs/specs/2026-05-23-propagation-unroll-composition.md tests/integration/test_propagation_unroll_gate.py
git commit -m "docs(propagation): record Path C unroll-composition result (design doc §14)"
```

---

## Self-Review notes

- **Spec coverage:** sim generators (Tasks 1–3), unroll + eval (Tasks 4–5), exports (6), gate (7), CLI/sweep (8), interpret/finalise (9) — every spec §6 file is covered.
- **Type consistency:** `Operator` Protocol (unroll.py) matches the `predict_np(named_idx, merit, regime)` signature implemented by both `GatedPropagationGraph` and `GraphNeuralODE`. `fit_graph(..., model_cls=...)` already accepts a model class. `world.substitute_indices`/`terminal_indices` exist on `GroundTruthWorld`.
- **No mutation of the committed §13 artefacts:** new generators are added alongside `generate_events`; the multi-hop gate/tests are untouched.
- **Honest gate:** the experimental 2-hop assertion is gated behind the Task 9 sweep, not assumed — mirrors how the §13 negative result was handled.
```
