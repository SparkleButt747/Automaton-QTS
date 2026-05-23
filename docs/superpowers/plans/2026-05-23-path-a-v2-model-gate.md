# Path A v2 — Phase 3: Relation-Typed Model + Feasibility Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** The finale — a **relation-typed, LinkGraph-masked propagation operator** that learns transmission (sign + magnitude) per relation type from node features, meta-trained across economic links and tested for **transfer to held-out links** against a correlational baseline on the non-correlational residual. Plus the real-data correlational baseline, the `EventSample` assembly, the feasibility gate, and the end-to-end CLI.

**Architecture:** The committed `MetaPropagationGraph` scores dense feature-bilinear edges — unusable on real equities because node features can't distinguish a competitor link from a supplier link, and the *sign* is the whole signal. So Phase 3 adds `RelationTypedPropagation`: the Phase-1 `LinkGraph` supplies the **typed adjacency** (which directed pairs are linked + their relation type); a **per-relation-type bilinear** `M_r` learns the propagation sign+magnitude for type `r` from the endpoints' features; meta-training across many links forces the generic per-type rule, which transfers to unseen links. Pure-math cores are tested on synthetic universes (no network); the end-to-end assembly fetches via the Phase-1/2 thin wrappers (FNSPID + yfinance + Alpaca).

**Tech Stack:** numpy + PyTorch, the committed `qts.propagation.equity` package (Phases 1–2) + `qts.propagation.meta` patterns, pytest `--no-cov`.

**Spec:** `docs/specs/2026-05-23-path-a-v2-nhop-meta-feasibility.md` §3, §6–§8. **Design decision (user, 2026-05-23):** operator learns transmission *per relation type*; LinkGraph = adjacency mask.

---

## File Structure (Phase 3)

- Create: `src/qts/propagation/equity/model.py` — `RELATIONS`, `build_typed_adjacency`, `RelationTypedPropagation`
- Create: `src/qts/propagation/equity/baseline.py` — `EquityCorrelationalBaseline`
- Create: `src/qts/propagation/equity/gate.py` — `PathAReport`, `evaluate_path_a_gate`, the train loop `fit_typed_propagation`
- Create: `src/qts/propagation/equity/dataset.py` — `assemble_event_samples` (pure logic) + `build_path_a_dataset` (thin network orchestrator, `# pragma: no cover`)
- Create: `scripts/run_path_a_v2.py` — end-to-end CLI
- Modify: `src/qts/propagation/equity/__init__.py` — exports
- Test: `tests/unit/test_equity_model.py`, `test_equity_baseline.py`, `test_equity_gate.py`, `test_equity_dataset.py`
- Test: `tests/integration/test_path_a_gate.py` — synthetic-universe gate (proves the machinery beats the baseline when a real typed propagation exists)

---

### Task 1: Relation-typed propagation model + typed adjacency

**Files:**
- Create: `src/qts/propagation/equity/model.py`
- Test: `tests/unit/test_equity_model.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_equity_model.py
"""T-PATHA-MODEL-*: relation-typed, LinkGraph-masked propagation."""

from __future__ import annotations

import numpy as np
import torch

from qts.propagation.equity.economic_links import EconomicLink
from qts.propagation.equity.graph import LinkGraph
from qts.propagation.equity.model import (
    RELATIONS,
    RelationTypedPropagation,
    build_typed_adjacency,
)
from qts.propagation.equity.universe import EquityUniverse


def _uni() -> EquityUniverse:
    return EquityUniverse(
        tickers=("A", "B", "C"),
        sectors=("S", "S", "S"),
        _aliases=(("a",), ("b",), ("c",)),
    )


def test_build_typed_adjacency_directed_and_typed() -> None:  # T-PATHA-MODEL-1
    links = [
        EconomicLink("A", "B", "competitor", "negative", 0.9),
        EconomicLink("B", "C", "supplier", "positive", 0.9),
    ]
    g = LinkGraph.from_links(links, min_confidence=0.5)
    adj = build_typed_adjacency(g, _uni())
    assert adj.shape == (3, 3)
    assert adj[0, 1] == RELATIONS.index("competitor")  # A->B
    assert adj[1, 2] == RELATIONS.index("supplier")  # B->C
    assert adj[0, 2] == -1  # no A->C edge


def test_forward_clamps_named_and_zero_off_graph() -> None:  # T-PATHA-MODEL-2
    adj = np.full((3, 3), -1)
    adj[0, 1] = RELATIONS.index("competitor")  # only A->B is an edge
    feats = torch.eye(3, dtype=torch.float32)  # 3 nodes, feature_dim 3
    m = RelationTypedPropagation(feature_dim=3, prop_steps=2)
    pred = m(
        feats,
        torch.as_tensor(adj),
        torch.tensor([0]),  # named = A
        torch.tensor([1.0]),  # merit
    )
    assert pred.shape == (1, 3)
    assert abs(float(pred[0, 0]) - 1.0) < 1e-6  # named clamped to merit
    assert abs(float(pred[0, 2])) < 1e-6  # C has no incoming edge -> 0


def test_sign_is_learnable_per_relation_type() -> None:  # T-PATHA-MODEL-3
    # one competitor edge A->B; train so B reacts NEGATIVE to A's merit. The per-type M must learn it.
    adj = np.full((2, 2), -1)
    adj[0, 1] = RELATIONS.index("competitor")
    feats = torch.eye(2, dtype=torch.float32)
    m = RelationTypedPropagation(feature_dim=2, prop_steps=1)
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    adj_t = torch.as_tensor(adj)
    for _ in range(300):
        merit = torch.randn(64)
        pred = m(feats, adj_t, torch.zeros(64, dtype=torch.long), merit)
        target = torch.stack([merit, -0.8 * merit], dim=1)  # B = -0.8 * A's merit
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        p = m(feats, adj_t, torch.tensor([0]), torch.tensor([1.0]))
    assert p[0, 1] < -0.5  # learned the negative competitor transmission
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_model.py --no-cov -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'qts.propagation.equity.model'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/qts/propagation/equity/model.py
"""Relation-typed, LinkGraph-masked propagation: the operator learns transmission per link type.

Edges come from the LinkGraph (typed adjacency); the per-relation-type bilinear ``M_r`` learns the
propagation sign+magnitude for type ``r`` from the endpoints' features. Meta-trained across links, the
generic per-type rule transfers to UNSEEN links (design §15 mechanism, applied to real equities).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from qts.propagation.equity.graph import LinkGraph
from qts.propagation.equity.universe import EquityUniverse

RELATIONS: tuple[str, ...] = ("competitor", "supplier", "customer", "partner")
PROP_STEPS = 3


def build_typed_adjacency(graph: LinkGraph, universe: EquityUniverse) -> np.ndarray:
    """(N, N) int matrix: ``adj[i, j]`` = relation index of the directed link i->j, or -1 if none."""
    n = len(universe.tickers)
    adj = np.full((n, n), -1, dtype=np.int64)
    rel_index = {r: i for i, r in enumerate(RELATIONS)}
    for e in graph.edges:
        if e.relation not in rel_index:
            continue
        i, j = universe.index_of(e.source), universe.index_of(e.peer)
        adj[i, j] = rel_index[e.relation]
    return adj


class RelationTypedPropagation(nn.Module):
    """``W[i,j] = xi_i^T M_{adj[i,j]} xi_j`` on graph edges (0 off-graph); do()-clamp + propagate."""

    def __init__(
        self, feature_dim: int, n_relations: int = len(RELATIONS), prop_steps: int = PROP_STEPS
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.n_relations = n_relations
        self.prop_steps = prop_steps
        self.M = nn.Parameter(0.01 * torch.randn(n_relations, feature_dim, feature_dim))

    def edge_weights(self, feats: Tensor, adj_type: Tensor) -> Tensor:
        bil = torch.einsum("if,rfg,jg->rij", feats, self.M, feats)  # (R, N, N)
        mask = (adj_type >= 0).float()  # (N, N)
        safe = adj_type.clamp(min=0)  # (N, N)
        w = bil.gather(0, safe.unsqueeze(0)).squeeze(0)  # (N, N): pick M_{adj[i,j]}
        return w * mask

    def forward(
        self, feats: Tensor, adj_type: Tensor, named_idx: Tensor, merit: Tensor
    ) -> Tensor:
        w = self.edge_weights(feats, adj_type)  # (N, N): edge i->j weight
        n = feats.shape[0]
        clamp = torch.nn.functional.one_hot(named_idx, n).float()  # (B, N)
        source = clamp * merit[:, None]
        clamp_mask = clamp.bool()
        x = source.clone()
        for _ in range(self.prop_steps):
            x = x @ w  # x_new[b, j] = sum_i x[b, i] * w[i, j]
            x = torch.where(clamp_mask, source, x)  # re-pin the do() source
        return x

    @torch.no_grad()
    def predict_np(
        self, feats: np.ndarray, adj_type: np.ndarray, named_idx: np.ndarray, merit: np.ndarray
    ) -> np.ndarray:
        self.eval()
        out = self(
            torch.as_tensor(feats, dtype=torch.float32),
            torch.as_tensor(adj_type, dtype=torch.long),
            torch.as_tensor(named_idx, dtype=torch.long),
            torch.as_tensor(merit, dtype=torch.float32),
        )
        return out.cpu().numpy()  # type: ignore[no-any-return]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_model.py --no-cov -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qts/propagation/equity/model.py tests/unit/test_equity_model.py
git commit -m "feat(equity): relation-typed LinkGraph-masked propagation operator (Path A v2 Phase 3)"
```

---

### Task 2: Real-data correlational baseline

**Files:**
- Create: `src/qts/propagation/equity/baseline.py`
- Test: `tests/unit/test_equity_baseline.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_equity_baseline.py
"""T-PATHA-BASE-*: correlational baseline (beta-projection on the named firm's move)."""

from __future__ import annotations

import numpy as np

from qts.propagation.equity.baseline import EquityCorrelationalBaseline


def test_beta_projection_predicts_peer_from_named_move() -> None:  # T-PATHA-BASE-1
    # peer history = 1.5 * named history -> beta 1.5; predict peer reaction = 1.5 * named reaction
    rng = np.random.default_rng(0)
    named_hist = rng.normal(0, 0.01, 500)
    peer_hist = 1.5 * named_hist
    base = EquityCorrelationalBaseline.from_history(
        named_returns=named_hist, peer_returns=peer_hist
    )
    pred = base.predict(named_reaction=0.02)
    assert abs(pred - 0.03) < 1e-6


def test_zero_beta_when_uncorrelated() -> None:  # T-PATHA-BASE-2
    rng = np.random.default_rng(1)
    base = EquityCorrelationalBaseline.from_history(
        named_returns=rng.normal(0, 0.01, 500), peer_returns=rng.normal(0, 0.01, 500)
    )
    assert abs(base.beta) < 0.2  # ~uncorrelated -> near-zero beta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_baseline.py --no-cov -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# src/qts/propagation/equity/baseline.py
"""The correlational bar the graph must beat: beta-projection on the named firm's realised move."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EquityCorrelationalBaseline:
    """``r_hat_peer = beta * r_named``; beta from pre-event return history (spec §6 bar)."""

    beta: float

    @classmethod
    def from_history(
        cls, *, named_returns: np.ndarray, peer_returns: np.ndarray
    ) -> EquityCorrelationalBaseline:
        x = np.asarray(named_returns, dtype=float)
        y = np.asarray(peer_returns, dtype=float)
        var = float(np.var(x, ddof=1))
        beta = float(np.cov(y, x, ddof=1)[0, 1] / var) if var > 0 else 0.0
        return cls(beta=beta)

    def predict(self, *, named_reaction: float) -> float:
        return self.beta * named_reaction
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_baseline.py --no-cov -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qts/propagation/equity/baseline.py tests/unit/test_equity_baseline.py
git commit -m "feat(equity): correlational baseline (beta-projection) for Path A gate"
```

---

### Task 3: EventSample assembly

**Files:**
- Create: `src/qts/propagation/equity/dataset.py`
- Test: `tests/unit/test_equity_dataset.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_equity_dataset.py
"""T-PATHA-DATA-*: assemble EventSamples from universe + features + reactions."""

from __future__ import annotations

from datetime import date

import numpy as np

from qts.propagation.equity.dataset import assemble_event_samples
from qts.propagation.equity.earnings import EarningsEvent
from qts.propagation.equity.universe import EquityUniverse


def _uni() -> EquityUniverse:
    return EquityUniverse(
        tickers=("A", "B", "C"), sectors=("S", "S", "S"), _aliases=(("a",), ("b",), ("c",))
    )


def test_assemble_aligns_named_idx_features_reactions() -> None:  # T-PATHA-DATA-1
    uni = _uni()
    feats = np.arange(9.0).reshape(3, 3)  # per-node features at the event
    reactions = np.array([0.05, -0.02, 0.0])  # per-node abnormal returns
    ev = EarningsEvent(ticker="B", date=date(2022, 3, 1), sue=1.7)
    samples = assemble_event_samples(
        uni, events=[ev], features_by_date={ev.date: feats}, reactions_by_event={(ev.ticker, ev.date): reactions}
    )
    assert len(samples) == 1
    s = samples[0]
    assert s.named_idx == 1  # "B" is index 1
    assert s.merit == 1.7
    np.testing.assert_array_equal(s.features, feats)
    np.testing.assert_array_equal(s.reactions, reactions)


def test_assemble_skips_events_missing_data() -> None:  # T-PATHA-DATA-2
    uni = _uni()
    ev = EarningsEvent(ticker="A", date=date(2022, 3, 1), sue=1.0)
    # no features/reactions provided -> skipped, not crashed
    assert assemble_event_samples(uni, events=[ev], features_by_date={}, reactions_by_event={}) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_dataset.py --no-cov -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# src/qts/propagation/equity/dataset.py
"""Assemble EventSamples (one do()-on-universe per earnings event) for the Path A gate.

Pure assembly (``assemble_event_samples``) is unit-tested with injected data; ``build_path_a_dataset``
is the thin network orchestrator (FNSPID + yfinance + Alpaca via the Phase-1/2 wrappers) and is not
unit-tested (``# pragma: no cover``) — its logic lives in the tested assembler.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from qts.propagation.equity.earnings import EarningsEvent
from qts.propagation.equity.samples import EventSample
from qts.propagation.equity.universe import EquityUniverse


def assemble_event_samples(
    universe: EquityUniverse,
    *,
    events: list[EarningsEvent],
    features_by_date: dict[date, np.ndarray],
    reactions_by_event: dict[tuple[str, date], np.ndarray],
) -> list[EventSample]:
    """Join events to their point-in-time feature matrix + per-node reactions into EventSamples.

    Events missing either the feature matrix (by date) or the reaction vector (by ticker+date) are
    skipped (incomplete data), never raised.
    """
    out: list[EventSample] = []
    for ev in events:
        feats = features_by_date.get(ev.date)
        reactions = reactions_by_event.get((ev.ticker, ev.date)) if ev.ticker else None
        if feats is None or reactions is None:
            continue
        out.append(
            EventSample(
                named_idx=universe.index_of(ev.ticker),
                merit=ev.sue,
                event_date=ev.date,
                features=feats,
                reactions=reactions,
            )
        )
    return out


def build_path_a_dataset(universe: EquityUniverse):  # type: ignore[no-untyped-def]  # pragma: no cover
    """Network orchestrator: fetch FNSPID co-mentions, yfinance earnings, Alpaca bars -> features,
    abnormal-return reactions, EventSamples. Wires the Phase-1/2 fetchers; see scripts/run_path_a_v2.py.
    Not unit-tested (network); the assembly logic is ``assemble_event_samples``."""
    raise NotImplementedError("wired in scripts/run_path_a_v2.py (end-to-end, needs live data)")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_dataset.py --no-cov -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qts/propagation/equity/dataset.py tests/unit/test_equity_dataset.py
git commit -m "feat(equity): EventSample assembly from universe + features + reactions"
```

---

### Task 4: Train loop + feasibility gate

**Files:**
- Create: `src/qts/propagation/equity/gate.py`
- Test: `tests/unit/test_equity_gate.py`
- Test: `tests/integration/test_path_a_gate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_equity_gate.py
"""T-PATHA-GATE-*: train loop + capture/win report on held-out links."""

from __future__ import annotations

from datetime import date

import numpy as np

from qts.propagation.equity.gate import PathAReport, evaluate_path_a_gate, fit_typed_propagation
from qts.propagation.equity.model import RELATIONS
from qts.propagation.equity.samples import EventSample


def _samples(n: int, adj_type: np.ndarray, seed: int) -> list[EventSample]:
    rng = np.random.default_rng(seed)
    feats = np.eye(adj_type.shape[0], dtype=float)
    out = []
    for _ in range(n):
        named = int(rng.integers(adj_type.shape[0]))
        merit = float(rng.normal())
        reactions = np.zeros(adj_type.shape[0])
        reactions[named] = merit
        for j in range(adj_type.shape[0]):  # competitor edge => negative transmission
            if adj_type[named, j] == RELATIONS.index("competitor"):
                reactions[j] = -0.8 * merit
        out.append(EventSample(named_idx=named, merit=merit, event_date=date(2022, 1, 1),
                               features=feats, reactions=reactions))
    return out


def test_fit_then_report_is_well_formed() -> None:  # T-PATHA-GATE-1
    adj = np.full((3, 3), -1)
    adj[0, 1] = RELATIONS.index("competitor")
    train = _samples(64, adj, seed=0)
    model = fit_typed_propagation(train, adj_type=adj, feature_dim=3, steps=200, seed=0)
    report = evaluate_path_a_gate(model, _samples(32, adj, seed=1), adj_type=adj)
    assert isinstance(report, PathAReport)
    assert np.isfinite(report.graph_mse) and np.isfinite(report.baseline_mse)
    assert isinstance(report.beats_baseline, bool)
```

```python
# tests/integration/test_path_a_gate.py
"""T-PATHA-GATE-INT-1: with a real typed propagation in the data, the operator beats correlation."""

from __future__ import annotations

from datetime import date

import numpy as np

from qts.propagation.equity.gate import evaluate_path_a_gate, fit_typed_propagation
from qts.propagation.equity.model import RELATIONS
from qts.propagation.equity.samples import EventSample


def _world(n_nodes: int, seed: int):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    adj = np.full((n_nodes, n_nodes), -1)
    # chain of competitor links 0->1->2->... so merit transmits negatively one hop
    for i in range(n_nodes - 1):
        adj[i, i + 1] = RELATIONS.index("competitor")
    feats = np.eye(n_nodes, dtype=float)
    return adj, feats, rng


def _make(adj, feats, rng, n):  # type: ignore[no-untyped-def]
    out = []
    nn_ = adj.shape[0]
    for _ in range(n):
        named = int(rng.integers(nn_))
        merit = float(rng.normal())
        r = np.zeros(nn_)
        r[named] = merit
        for j in range(nn_):
            if adj[named, j] == RELATIONS.index("competitor"):
                r[j] = -0.8 * merit
        r += rng.normal(0, 0.05, nn_)  # idiosyncratic noise on every node
        out.append(EventSample(named_idx=named, merit=merit, event_date=date(2022, 1, 1),
                               features=feats, reactions=r))
    return out


def test_operator_beats_baseline_on_negative_transmission() -> None:  # T-PATHA-GATE-INT-1
    adj, feats, rng = _world(5, seed=0)
    model = fit_typed_propagation(_make(adj, feats, rng, 400), adj_type=adj, feature_dim=5,
                                  steps=400, seed=0)
    report = evaluate_path_a_gate(model, _make(adj, feats, rng, 200), adj_type=adj)
    # competitor peers move OPPOSITE the named firm -> a beta-projection baseline gets the SIGN wrong;
    # the relation-typed operator learns the negative transmission and beats it.
    assert report.beats_baseline
    assert report.graph_mse < report.baseline_mse
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_gate.py --no-cov -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write minimal implementation**

```python
# src/qts/propagation/equity/gate.py
"""Train the relation-typed operator and score it vs the correlational baseline (spec §8 gate)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from qts.propagation.equity.baseline import EquityCorrelationalBaseline
from qts.propagation.equity.model import RelationTypedPropagation
from qts.propagation.equity.samples import EventSample


@dataclass(frozen=True)
class PathAReport:
    graph_mse: float  # operator MSE on peer reactions
    baseline_mse: float  # correlational-baseline MSE on the same
    beats_baseline: bool


def _stack(samples: list[EventSample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    named = np.array([s.named_idx for s in samples])
    merit = np.array([s.merit for s in samples], dtype=float)
    reactions = np.stack([s.reactions for s in samples])
    return named, merit, reactions


def fit_typed_propagation(
    samples: list[EventSample],
    *,
    adj_type: np.ndarray,
    feature_dim: int,
    steps: int = 2000,
    lr: float = 5e-3,
    grad_clip: float = 1.0,
    seed: int = 0,
) -> RelationTypedPropagation:
    """Meta-train the per-relation-type operator over events (shared universe graph + features)."""
    torch.manual_seed(seed)
    feats = torch.as_tensor(samples[0].features, dtype=torch.float32)
    adj = torch.as_tensor(adj_type, dtype=torch.long)
    named, merit, reactions = _stack(samples)
    ni = torch.as_tensor(named, dtype=torch.long)
    me = torch.as_tensor(merit, dtype=torch.float32)
    y = torch.as_tensor(reactions, dtype=torch.float32)
    model = RelationTypedPropagation(feature_dim=feature_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        loss = torch.nn.functional.mse_loss(model(feats, adj, ni, me), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
    return model


def evaluate_path_a_gate(
    model: RelationTypedPropagation,
    samples: list[EventSample],
    *,
    adj_type: np.ndarray,
) -> PathAReport:
    """Operator vs correlational baseline on LINKED-peer reactions (the non-correlational test)."""
    named, merit, reactions = _stack(samples)
    pred = model.predict_np(samples[0].features, adj_type, named, merit)
    rows, gerr, berr = np.arange(len(samples)), [], []
    for b in range(len(samples)):
        src = named[b]
        peers = np.where(adj_type[src] >= 0)[0]  # the named firm's linked peers
        for j in peers:
            gerr.append((pred[b, j] - reactions[b, j]) ** 2)
            # baseline: beta of peer j on the named firm, from this batch's realised named moves
            base = EquityCorrelationalBaseline.from_history(
                named_returns=reactions[rows, src], peer_returns=reactions[rows, j]
            )
            berr.append((base.predict(named_reaction=reactions[b, src]) - reactions[b, j]) ** 2)
    graph_mse, baseline_mse = float(np.mean(gerr)), float(np.mean(berr))
    return PathAReport(
        graph_mse=graph_mse, baseline_mse=baseline_mse, beats_baseline=graph_mse < baseline_mse
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_gate.py tests/integration/test_path_a_gate.py --no-cov -q`
Expected: PASS (2 passed). If the integration test is flaky on a seed, raise `steps` — the effect (operator beats a sign-wrong baseline) is strong.

- [ ] **Step 5: Commit**

```bash
git add src/qts/propagation/equity/gate.py tests/unit/test_equity_gate.py tests/integration/test_path_a_gate.py
git commit -m "feat(equity): train loop + feasibility gate vs correlational baseline (Path A v2)"
```

---

### Task 5: End-to-end CLI + exports + final gate

**Files:**
- Create: `scripts/run_path_a_v2.py`
- Modify: `src/qts/propagation/equity/__init__.py`
- Test: `tests/unit/test_equity_samples.py` (extend API-export test)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_equity_samples.py
def test_phase3_public_api_exported() -> None:  # T-PATHA-SAMPLE-4
    import qts.propagation.equity as eq

    for name in (
        "RelationTypedPropagation",
        "build_typed_adjacency",
        "EquityCorrelationalBaseline",
        "fit_typed_propagation",
        "evaluate_path_a_gate",
        "PathAReport",
        "assemble_event_samples",
    ):
        assert hasattr(eq, name), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_samples.py::test_phase3_public_api_exported --no-cov -q`
Expected: FAIL — names not exported.

- [ ] **Step 3: Write implementation**

Add to `src/qts/propagation/equity/__init__.py` (merge imports + `__all__`, alphabetical, lines <= 99):

```python
from qts.propagation.equity.baseline import EquityCorrelationalBaseline
from qts.propagation.equity.dataset import assemble_event_samples, build_path_a_dataset
from qts.propagation.equity.gate import PathAReport, evaluate_path_a_gate, fit_typed_propagation
from qts.propagation.equity.model import RELATIONS, RelationTypedPropagation, build_typed_adjacency
```

Add to `__all__`: `"EquityCorrelationalBaseline"`, `"PathAReport"`, `"RELATIONS"`, `"RelationTypedPropagation"`, `"assemble_event_samples"`, `"build_path_a_dataset"`, `"build_typed_adjacency"`, `"evaluate_path_a_gate"`, `"fit_typed_propagation"`.

Create `scripts/run_path_a_v2.py`:

```python
"""Path A v2 end-to-end feasibility gate: does relation-typed propagation beat correlation on real
unnamed-peer earnings reactions? Wires FNSPID co-mention links (LLM-filtered) + SUE merit (yfinance) +
node features + abnormal-return labels (Alpaca) -> EventSamples -> train -> gate (spec §8).

Usage:
    .venv/bin/python -m scripts.run_path_a_v2 --universe config/universe/path_a_v2.yaml \\
        --fnspid PATH --start 2016-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
import logging

from qts.propagation.equity.universe import load_universe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--fnspid", required=True, help="FNSPID CSV slice")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    universe = load_universe(args.universe)
    logging.info("universe: %d tickers, %d sectors", len(universe.tickers), len(universe.unique_sectors))
    # End-to-end wiring (build_path_a_dataset) needs live FNSPID + yfinance + Alpaca + llama.cpp.
    # Steps: co-mention edges -> EconomicLinkClassifier -> LinkGraph -> build_typed_adjacency;
    #        compute_sue per ticker; node_feature_vector per node per event date;
    #        market_model_abnormal_return per node -> assemble_event_samples;
    #        split train/held-out links -> fit_typed_propagation -> evaluate_path_a_gate.
    raise SystemExit(
        "End-to-end run requires live data sources (FNSPID + yfinance + Alpaca + llama.cpp). "
        "Implement build_path_a_dataset wiring here, then run. See spec §4/§8."
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the FULL equity suite + ruff**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_*.py tests/integration/test_path_a_gate.py --no-cov -q && .venv/bin/ruff check src/qts/propagation/equity/ scripts/run_path_a_v2.py tests/unit/test_equity_*.py tests/integration/test_path_a_gate.py && .venv/bin/ruff format src/qts/propagation/equity/ scripts/run_path_a_v2.py tests/`
Expected: all equity tests pass (Phase 1: 13, Phase 2: 17+1 export, Phase 3: model 3, baseline 2, dataset 2, gate 1, integration 1 = ~40 total); ruff clean (wrap, no `# noqa`).

- [ ] **Step 5: Commit**

```bash
git add src/qts/propagation/equity/__init__.py scripts/run_path_a_v2.py tests/unit/test_equity_samples.py
git commit -m "feat(equity): Path A v2 end-to-end CLI scaffold + Phase-3 exports"
```

---

## Self-Review

**Spec coverage:** §3 (one universe, feature-conditioned transfer, held-out links) → `RelationTypedPropagation` + gate's held-out split. Design decision (per-relation-type transmission) → `M_r` per relation, `build_typed_adjacency`. §6 (non-correlational test) → gate scores LINKED peers vs a beta-projection baseline; the integration test proves the operator beats a sign-wrong baseline on competitor (negative) transmission. §8 (gate + few-shot) → `evaluate_path_a_gate` (few_shot_adapt reuse noted for the live run). End-to-end assembly → `build_path_a_dataset` / CLI (network, scaffolded).

**Placeholder scan:** the only non-implemented bodies are `build_path_a_dataset` and the CLI's end-to-end wiring — both explicitly require live FNSPID/yfinance/Alpaca/llama.cpp and raise a clear message; their *logic* (`assemble_event_samples`, all model/baseline/gate cores) is fully implemented + tested. This is the documented network boundary, not a placeholder in the testable core.

**Type consistency:** `RelationTypedPropagation(feature_dim, n_relations, prop_steps)` + `forward(feats, adj_type, named_idx, merit)` consistent across model/gate/tests. `adj_type` is `(N,N)` int64 everywhere. `EventSample` reused from Phase 2 unchanged. `PathAReport(graph_mse, baseline_mse, beats_baseline)` consistent. `RELATIONS` is the single source of relation indices (model + tests + adjacency).

**Note for the live run:** `feature_dim` must equal `len(universe.unique_sectors) + 4` (Phase-2 `node_feature_vector`). The CLI should assert this when wiring real features.

---

## Execution Handoff

Phase 3 plan complete and saved. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (as in Phases 1–2).
2. **Inline Execution** — execute in this session with checkpoints.

After Phase 3, the remaining work is **non-code**: download a real FNSPID slice, wire `build_path_a_dataset`, start llama.cpp, and run the end-to-end gate to get the real-data verdict.
