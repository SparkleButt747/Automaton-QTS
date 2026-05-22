# Spec: Event-Propagation Graph v0 (feasibility cut)

**Status:** spec (design approved). Ready for `writing-plans`.
**Date:** 2026-05-22
**Design doc:** `docs/research/2026-05-22-event-propagation-graph-design.md`
**Grill log:** `.grill/event-propagation-graph.md`

> **Amendments (post-build, 2026-05-22).** The v0 run revised three things from this spec; the
> committed code under `src/qts/propagation/` is authoritative where it differs. (1) **Scale:**
> 6 event types / 18 assets / 6-dim sector codes (was 3 / 8 / 2-dim) — 2 training pairs were too few
> to determine a transferable mechanism. (2) **Prediction gate:** judged on the substitute (≥25%) +
> the no-prop floor; the overall-MSE-vs-correlational term was dropped as unfair (the graph never
> observes the factor shock). (3) **Split:** hold out the LAST event type. Result: feasibility PASS
> (robust prediction AND transfer). See design doc §12 Findings.

---

## 1. Goal

Build a self-contained, sim-only v0 of the event-propagation graph and run its feasibility gate.
Deliverable = a reproducible experiment that answers one question: **can a state-gated graph predict
cross-asset reactions to `do()` events better than a strong correlational baseline — including on an
asset pair it never saw coupled?**

No real-market data, no LLM, no live trading. Everything is numpy/torch + pytest.

## 2. Scope / non-goals

**In scope:** the adversarial sim, the bilinear gated-graph model, both baselines, the training loop,
the two-part feasibility gate, and the tests that assert the gate's machinery is correct.

**Out of scope (v0):** real data, the LLM merit front-end, the neural-ODE variant, interpretability
tooling, multi-hop chains, anticipation/leakage/heavy-sparsity/drift obstacles, any integration with
the live QTS trading path. These are §9 of the design doc — later layers.

## 3. Design refinement vs the design doc (read this first)

The design doc §5.2 sketched **free per-regime adjacency matrices** `W_r`. That parameterisation
**cannot pass the chosen transfer gate**: a free weight `W_r[named, substitute]` for a pair held out
of training receives no gradient, so there is nothing to transfer. v0 therefore conditions edges on
**per-asset feature vectors** via a regime-gated **bilinear** form:

```
 W_eff[i, j] = Σ_r π_r · (ξ_i^T M_r ξ_j)
```

where `ξ_i` is asset/concept `i`'s feature vector and `M_r ∈ ℝ^{F×F}` is the learnable per-regime
interaction matrix. Because an edge entry is a *function of features* (not of node identity), the
mechanism generalises to an unseen pair with the same feature relationship — which is exactly what
"the substitution mechanism transferred" means. This is the minimal change that makes the gate
meaningful; propagation stays linear.

## 4. Module layout

New package `src/qts/propagation/` (greenfield — confirmed no existing graph/causal code):

| File | Responsibility |
|---|---|
| `src/qts/propagation/__init__.py` | package exports |
| `src/qts/propagation/sim.py` | `PropagationSimConfig`, `GroundTruthWorld`, event + dataset generation |
| `src/qts/propagation/model.py` | `GatedPropagationGraph` (`nn.Module`) |
| `src/qts/propagation/baselines.py` | `no_propagation_predict`, `CorrelationalBaseline` |
| `src/qts/propagation/train.py` | training loop + `evaluate_feasibility` (the gate) |
| `tests/unit/test_propagation_sim.py` | sim invariants |
| `tests/unit/test_propagation_model.py` | model shape/grad/forward |
| `tests/unit/test_propagation_baselines.py` | baseline correctness |
| `tests/integration/test_propagation_gate.py` | end-to-end feasibility gate |

## 5. Data model

All dataclasses live in `sim.py`. Use `@dataclass(frozen=True)` for config/ground-truth, plain
tensors/arrays for batched data.

```python
N_ASSETS = 8
N_FACTORS = 2
N_EVENT_TYPES = 3
FEATURE_DIM = 4          # ξ ∈ ℝ⁴: dims[0:2] drive factor loadings, dims[2:4] = "sector" tags

@dataclass(frozen=True)
class EventTriple:
    """One event type's causal triple. All indices into 0..N_ASSETS-1, distinct."""
    named: int
    substitute: int       # true causal target; corr(named, substitute) ≈ 0
    decoy: int            # bait; corr(named, decoy) high, NO causal edge

@dataclass(frozen=True)
class PropagationSimConfig:
    n_assets: int = N_ASSETS
    n_factors: int = N_FACTORS
    feature_dim: int = FEATURE_DIM
    n_regimes: int = 2
    factor_vol: float = 1.0
    idiosyncratic_vol: float = 0.3
    merit_vol: float = 1.0
    propagation_gain: float = 1.5      # |w_{n→s}| magnitude
    seed: int = 0

@dataclass(frozen=True)
class GroundTruthWorld:
    config: PropagationSimConfig
    features: np.ndarray               # (n_assets, feature_dim), fixed per seed
    loadings: np.ndarray               # (n_assets, n_factors) = features[:, :n_factors]
    triples: tuple[EventTriple, ...]   # length n_event_types
    regime_signs: np.ndarray           # (n_regimes,) ∈ {+1,-1}: substitution vs complement

@dataclass(frozen=True)
class EventBatch:
    """A batch of generated events ready for model/baseline consumption."""
    named_idx: np.ndarray              # (B,) which asset is intervened on
    merit: np.ndarray                  # (B,) observed magnitude·direction of the named shock
    regime: np.ndarray                 # (B,) regime id
    reactions: np.ndarray              # (B, n_assets) ground-truth per-asset window return (TARGET)
    event_type: np.ndarray             # (B,) which triple
```

## 6. The sim (`sim.py`)

### 6.1 Building the world — `build_world(config) -> GroundTruthWorld`

1. Seed numpy from `config.seed`.
2. Draw `features` `(n_assets, feature_dim)` ~ N(0,1). `loadings = features[:, :n_factors]`.
3. Construct `n_event_types` **disjoint** `EventTriple`s satisfying the confound (§6.2). Use a
   rejection loop over candidate (named, substitute, decoy) index choices; assert the properties hold.
4. `regime_signs = [+1, -1]` (regime 0 = substitution, regime 1 = complement).

### 6.2 The confound (must hold — asserted in tests)

For every triple `(n, s, d)`, using the asset return covariance `Σ = loadings @ loadings.T +
idiosyncratic_vol² · I`:

- **Decoy correlated:** `corr(n, d) ≥ 0.5` (achieved by `features[d, :2] ≈ features[n, :2]`).
- **Substitute uncorrelated:** `|corr(n, s)| ≤ 0.2` (achieved by `features[s, :2] ⊥ features[n, :2]`).
- **Substitute feature-linked:** `cos(features[n, 2:], features[s, 2:]) ≥ 0.5` (shared "sector" — the
  signal a feature-conditioned model can learn and transfer; identity-free).

### 6.3 Generating events — `generate_events(world, n, rng) -> EventBatch`

Per event:
1. Sample `event_type ∈ {0..n_event_types-1}`; let `(n, s, d) = world.triples[event_type]`.
2. Sample `regime ∈ {0,1}`; `regime_sign = world.regime_signs[regime]`.
3. Sample factor shock `g ~ N(0, factor_vol²)` size `(n_factors,)` — **independent of the causal edge**.
4. Sample `merit ~ N(0, merit_vol²)` (the named asset's observed shock).
5. Reactions `r ∈ ℝ^{n_assets}`:
   - `r[i] = loadings[i] · g + ε_i`, `ε_i ~ N(0, idiosyncratic_vol²)` for all i (the confound/noise).
   - `r[n] += merit` (named asset's own merit).
   - `r[s] += regime_sign · propagation_gain · merit` (the **causal edge**, gated by regime).
   - decoy `d` gets **no** causal term — it moves only via shared factor loading.

Return arrays stacked over the `n` events.

### 6.4 Dataset split — `make_splits(world, rng) -> (train, val, test, transfer)`

- `train`: events drawn from event types `{0, 1}` only. (≈4000 events.)
- `val`: same distribution as train, disjoint draw. (≈1000.)
- `test`: same distribution as train, disjoint draw. (≈1000.) — the **prediction** gate.
- `transfer`: events from the **held-out event type `{2}`** — the named→substitute pair never seen in
  training. (≈1000.) — the **transfer** gate.

## 7. The model (`model.py`)

`GatedPropagationGraph(nn.Module)` — bilinear, regime-gated, linear propagation, K latent concepts.

```python
K_CONCEPTS = 3
N_REGIMES  = 2
PROP_STEPS = 8            # unrolled linear steps (fixed-point solve is the deferred upgrade)

class GatedPropagationGraph(nn.Module):
    def __init__(self, asset_features: Tensor,   # (n_assets, F) — frozen sim features
                 k_concepts: int = K_CONCEPTS,
                 n_regimes: int = N_REGIMES,
                 prop_steps: int = PROP_STEPS): ...
        # frozen buffer: asset_features (n_assets, F)
        # learnable: concept_features ψ (k_concepts, F)
        # learnable: M_r (n_regimes, F, F)  — bilinear interaction per regime
        # learnable: gate g_φ : regime_onehot -> logits over regimes (small Linear)

    def edge_weights(self, regime: Tensor) -> Tensor:
        # ξ = cat([asset_features, concept_features])            -> (n_assets+K, F)
        # π = softmax(gate(onehot(regime)))                      -> (B, n_regimes)
        # W_eff[b] = Σ_r π[b,r] * (ξ @ M_r @ ξ.T)                -> (B, n_assets+K, n_assets+K)
        ...

    def forward(self, named_idx: Tensor, merit: Tensor, regime: Tensor) -> Tensor:
        # x0: zeros (n_assets+K); clamp x[named_idx] = merit   (the do() intervention)
        # for _ in range(prop_steps): x = clamp_named( W_eff @ x )
        # return x[:, :n_assets]                                 -> (B, n_assets) predicted reactions
        ...
```

**Loss** (`train.py`): `loss = mse(pred, target) + l1_lambda * sum(|M_r|)`, `l1_lambda = 1e-3`.
`do()` clamp is re-applied every step so the named node stays pinned to observed merit.

## 8. Baselines (`baselines.py`)

| Function / class | Prediction | Notes |
|---|---|---|
| `no_propagation_predict(batch) -> (B, n_assets)` | `merit` at `named_idx`, else 0 | floor |
| `CorrelationalBaseline(Σ).predict(batch)` | `r̂[i] = (Σ[i,named]/Σ[named,named]) · r_named_observed` | β-projection on the named asset's observed reaction; nails decoy, misses substitute |

`Σ` is estimated from a **pre-event return sample** (draw factor+idiosyncratic returns with no events),
NOT from the event reactions — the baseline only knows historical co-movement. `r_named_observed` is
`batch.reactions[:, named]` (the named asset's realised move is observable at prediction time).

## 9. Feasibility gate (`train.py::evaluate_feasibility`)

Train once (Adam, lr 1e-2, ≤500 epochs, early-stop on `val` MSE). Then both must pass:

1. **Prediction** (on `test`):
   - `mse(graph) < mse(correlational)` AND `mse(graph) < mse(no_prop)`, AND
   - per-substitute: `mse_substitute(graph) < mse_substitute(correlational)` by ≥ 25% relative.
2. **Transfer** (on `transfer`, the unseen pair):
   - `mse_substitute(graph) < mse_substitute(correlational)` on the held-out event type's substitute.

Return a `FeasibilityReport` dataclass (frozen): per-split MSEs, per-substitute MSEs, `prediction_pass`,
`transfer_pass`, `passed = prediction_pass and transfer_pass`.

## 10. Acceptance criteria (test IDs follow `T-PROP-*`)

| ID | Test | Asserts |
|---|---|---|
| T-PROP-SIM-1 | `build_world` confound | every triple meets the §6.2 corr/feature bounds |
| T-PROP-SIM-2 | `generate_events` shapes & determinism | shapes correct; same seed ⇒ identical batch |
| T-PROP-SIM-3 | causal edge present | substitute reaction correlates with `regime_sign·merit`; decoy does not |
| T-PROP-MODEL-1 | forward shapes | `(B, n_assets)` out; `do()` clamp holds at named idx |
| T-PROP-MODEL-2 | gradients flow | loss.backward populates grads on `M_r`, `concept_features`, gate |
| T-PROP-BASE-1 | correlational baseline | recovers β on a hand-built Σ; predicts decoy, ~0 on substitute |
| T-PROP-BASE-2 | no-prop baseline | named=merit, others 0 |
| T-PROP-GATE-1 | end-to-end (small, seeded) | `evaluate_feasibility` returns a well-formed report; machinery runs |
| T-PROP-GATE-2 | prediction gate passes | on a seed where the sim is learnable, `prediction_pass is True` |
| T-PROP-GATE-3 | transfer gate passes | `transfer_pass is True` on that seed |

T-PROP-GATE-2/3 are the real feasibility claim; keep their event counts/epochs modest so the test runs
in CI time, but large enough to actually train (use a fixed seed proven to converge during dev).

## 11. Determinism & repro

- Every entry point takes an explicit seed; seed numpy AND torch (`torch.manual_seed`).
- Tests use small fixed seeds and assert exact reproducibility (T-PROP-SIM-2).
- A `scripts/run_propagation_feasibility.py` CLI (thin, like `scripts/run_news_sweep.py`) runs the full
  gate at full scale and prints the `FeasibilityReport`. (Plan task; mirrors existing script style.)

## 12. Deferred to the plan (not design questions — implementation choices)

- Exact event counts / epochs / early-stop patience that make T-PROP-GATE-2/3 reliably green in CI.
- Whether `Σ` uses sample covariance or the analytic `loadings@loadings.T + σ²I` (start: sample, n≈5000).
- Whether the gate's regime context is the true label (v0: yes, clean) or inferred (deferred).
- numerical guard if `PROP_STEPS` unrolling diverges (clip W spectral norm, or reduce gain).
