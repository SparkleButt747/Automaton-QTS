# Spec: Propagation Graph v0.1 — Multi-hop chains (realism layer 1)

**Status:** IMPLEMENTED — outcome is a documented **negative result** for transfer. In-sample 2-hop
composition works (GATE-2 passes); 2-hop transfer to an unseen chain does **not** generalise (GATE-3
`xfail`), invariant to five tuning levers. Full write-up: design doc §13.
**Design doc:** `docs/research/2026-05-22-event-propagation-graph-design.md` (§8 build order; §13 findings).

## 1. Goal

Extend the sim from a single hop (named → substitute) to a genuine **2-hop chain A → B → C**, where
the 2-hop terminal **C is invisible to the correlational baseline** (factor-decorrelated from the
named A) and reachable only by composing two distinct learned relations. Re-run the feasibility gate
with added 2-hop prediction + transfer metrics. Tests the project's core "nth-order propagation to an
unnamed entity" thesis.

## 2. Why two relations (locked design decision)

One repeated relation collapses: a chain of "sector-match" edges keeps all nodes in one cluster, and
in 2 factor dims the 2-hop target is forced to ±A (correlated → baseline finds it). So:

| hop | relation | feature rule |
|---|---|---|
| A → B | R1 ("substitution") | B matches A's **R1-code**, factor ⊥ A |
| B → C | R2 ("supply") | C matches B's **R2-code**, factor ⊥ {A, B} |

A→C exists ONLY as the composition R1∘R2 — no direct A→C feature match. The model learns
`M ≈ blockdiag(0_factor, I_R1, I_R2)` and the unrolled propagation composes the two hops.

## 3. Sim changes (`sim.py`)

**Constants:**
```python
N_EVENT_TYPES = 6          # 5 chains trained, 1 held out for transfer
N_ASSETS = 4 * N_EVENT_TYPES   # named[0:6], B[6:12], C[12:18], decoy[18:24]
N_FACTORS = 3              # 3 so C can be factor-orthogonal to BOTH A and B
R1_DIM = 6
R2_DIM = 6
FEATURE_DIM = N_FACTORS + R1_DIM + R2_DIM   # 15 — [0:3] factor, [3:9] R1, [9:15] R2
```

**`EventTriple` → `EventChain`** (rename; update all references incl. `__init__` export + tests):
```python
@dataclass(frozen=True)
class EventChain:
    named: int        # A
    substitute: int   # B (1-hop, R1 of A)
    terminal: int     # C (2-hop, R2 of B)
    decoy: int        # factor-correlated with A, no causal edge
```

**`PropagationSimConfig`** gains `propagation_gain2: float = 1.0` (hop-2 gain). Keep
`propagation_gain` for hop 1.

**`GroundTruthWorld`** gains `terminal_indices(event_type) -> np.ndarray` (mirrors
`substitute_indices`). `loadings = features[:, :n_factors]` (now 3-dim) — unchanged logic.

**`build_world`** — per chain k (n = n_event_types):
- Draw a random **orthonormal triplet** `a_A, a_B, a_C` in `n_factors`-D (QR of a random matrix) so
  the three are mutually factor-orthogonal (`corr ≈ 0`).
- R1-code `r1 = unit(R1_DIM)`, R2-code `r2 = unit(R2_DIM)`.
- `named` (idx k): factor `a_A`, R1 `r1`, R2 `0`.
- `substitute` (n+k): factor `a_B`, R1 `r1` (matches A → A→B), R2 `r2` (B is R2-source).
- `terminal` (2n+k): factor `a_C`, R1 `0`, R2 `r2` (matches B → B→C).
- `decoy` (3n+k): factor `a_A + 0.05·noise` (corr high), R1 `0`, R2 `0`.

Replace the 2-D-only `_rot90` with `_orthonormal_basis(rng, dim, k_vectors)` (QR-based). Keep helper
`_unit`.

**`generate_events`** — add the 2-hop causal edge:
```python
reactions[rows, named] += merit
reactions[rows, B]     += sign * cfg.propagation_gain * merit                       # hop 1
reactions[rows, C]     += sign * cfg.propagation_gain * cfg.propagation_gain2 * merit  # hop 2 (through B)
```
`make_splits` unchanged (still holds out the last event type).

## 4. Model (`model.py`) — UNCHANGED

Auto-adapts to `feature_dim=15` / `n_assets=24` via `world.features.shape`. No edits.

## 5. Gate changes (`train.py`)

`FeasibilityReport` gains: `terminal_mse_graph`, `terminal_mse_corr`, `transfer_terminal_mse_graph`,
`transfer_terminal_mse_corr` (the 2-hop analogues of the substitute fields).

`evaluate_feasibility` computes terminal (C) MSEs for `test` and `transfer` via `terminal_indices`.

Revised gate (both hops must hold):
```python
prediction_pass = (
    test_mse_graph < test_mse_noprop
    and sub_mse_graph < SUBSTITUTE_MARGIN * sub_mse_corr        # 1-hop B (v0 claim)
    and terminal_mse_graph < SUBSTITUTE_MARGIN * terminal_mse_corr  # 2-hop C (new claim)
)
transfer_pass = (
    transfer_sub_mse_graph < transfer_sub_mse_corr             # 1-hop transfer
    and transfer_terminal_mse_graph < transfer_terminal_mse_corr  # 2-hop transfer
)
```

## 6. Tests (`T-PROP-*`, update + add)

| ID | Test | Asserts |
|---|---|---|
| T-PROP-SIM-1 | chain confound | per chain: `corr(A,decoy)≥0.5`; `\|corr(A,B)\|≤0.2`; `\|corr(A,C)\|≤0.2`; R1 match A·B, R2 match B·C (cos≥0.5) |
| T-PROP-SIM-3 | 2-hop causal edge | with zero factor/idio vol, `r_C ≈ sign·gain·gain2·merit`; decoy ≈ 0 |
| T-PROP-SIM-2/2b | determinism + split | shapes `(B, 24)`; transfer == `{n-1}` (unchanged logic) |
| T-PROP-MODEL-1/2 | unchanged | forward shape `(B, 24)`, clamp, grads |
| T-PROP-GATE-2 | 2-hop prediction | `prediction_pass` (incl. terminal bar) at a pinned seed |
| T-PROP-GATE-3 | 2-hop transfer | `transfer_pass` (incl. terminal) at a pinned seed |

## 7. CLI + docs

- `scripts/run_propagation_feasibility.py`: print terminal (2-hop) MSEs alongside the substitute.
- After the run: record results in design doc (§12 → add a multi-hop subsection) and confirm the
  feasibility verdict for 2-hop.

## 8. Acceptance

A seed where the full gate (1-hop AND 2-hop, prediction AND transfer) passes; confirm robustness with
a 0-5 seed sweep. If 2-hop transfer can't pass at any seed, that is a real finding — escalate (do not
weaken the gate); the honest fallback is "1-hop transfers, 2-hop needs the neural-ODE / richer model".

## 9. Out of scope (this layer)

Merit-noise, anticipation/leakage, sparsity, regime-drift, >2 hops, neural-ODE — all later layers.
