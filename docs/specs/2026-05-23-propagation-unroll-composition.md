# Spec: Propagation Graph — Unroll-Composition (Path C)

**Status:** IMPLEMENTED — **negative result.** Composing the 1-hop operator by unrolling does NOT
achieve robust 2-hop transfer (≤2/6 seeds; dominated by end-to-end). Full write-up + the data-scaling
flood finding + related work (NBFNet / MLC / meta-transfer) + novelty read: design doc §14.
**Design doc:** `docs/research/2026-05-22-event-propagation-graph-design.md` (§14 findings).

## 1. Goal

Reach the 2-hop terminal **C on a held-out chain** by **composing the transferable 1-hop operator at
inference** (unrolling), instead of training a 2-hop A→C mapping end-to-end (which overfits and does
not transfer — see §13). The bet: each individual hop is a 1-hop transfer, which v0 proved robust
(6/6); chaining two of them should inherit that robustness.

## 2. Why this can work where end-to-end failed (locked rationale)

End-to-end 2-hop training puts C in the loss, so the model finds chain-specific shortcuts that fit the
5 training chains but don't generalise. Path C **never makes A→C a training target**. It trains the
*local relations* R1 (A→B) and R2 (B→C) as independent 1-hop maps, then composes them at inference.
Each hop is exactly the operation that already transfers.

**Critical design point:** to learn R2 at all, **B-role nodes must be named during training** (an
event whose source is B and whose supervised target is C). "Train only A→B" would never exercise R2,
so naming B at inference would fire an untrained edge. Therefore the 1-hop training set names *every
relation-bearing source* (A and B) across chains 0–4; chain 5 is excluded entirely.

## 3. Locked decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Base operator | **Both** `GatedPropagationGraph` (linear) **and** `GraphNeuralODE` — compared | linear is the 6/6 1-hop performer; ODE comparison shows whether continuous dynamics help the *composed* case even though they didn't help end-to-end |
| Eval readout | **Known successor index** | isolates "does the composition transfer?" from "can it localise the reactor?" — the latter is deferred |
| Composition | **Explicit external unroll** (predict B → re-inject as merit → predict C) | each hop is a bona-fide 1-hop prediction; cleaner than relying on the model's internal multi-step diffusion |
| Eval ground truth | **per-hop signed** (`r_C = gain1·gain2·merit`) | the unroll applies the regime sign once per hop (sign²=1); eval truth must match iterated 1-hop or it mismatches at regime=−1 |
| Training target | **1-hop only**, A→C never supervised | the whole point — see §2 |

## 4. Mechanism (concrete)

**1-hop training events** (`generate_hop_events`): pick chain `k ∈ allowed_types`, pick hop
`∈ {0: A→B via gain1, 1: B→C via gain2}`, name the source (`A_k` or `B_k`), draw merit + regime.
Reactions = factor noise on all assets + `merit` at the source + `sign·gain·merit` at the direct
successor. Returns a standard `EventBatch` (named_idx = source). The full reaction vector carries the
supervised hop, so the existing MSE loss in `fit_graph` trains the local relation unchanged.

**Per-hop-signed eval chains** (`generate_chain_eval`): name `A_k`, set
`r_B = sign·gain1·merit`, `r_C = sign·gain2·r_B = gain1·gain2·merit` (per-hop sign), plus the named
bump and factor noise. Used only to produce ground-truth C for the transfer comparison.

**External unroll** (`unroll_predict`, model-agnostic — both models share the
`(named_idx, merit, regime)` signature): given named source indices + merit + regime + an ordered list
of successor indices per hop:
```
m = merit
for hop, succ_idx in enumerate(hop_successors):     # e.g. [B_idx, C_idx]
    out = model.predict_np(named=current_source, merit=m, regime=regime)
    r_succ = out[rows, succ_idx]
    current_source, m = succ_idx, r_succ
return r_succ   # reaction at the terminal
```

**Transfer eval** (`evaluate_unroll_transfer`): on the held-out chain, unroll A₅→B₅→C₅, compare the
predicted r_C₅ to the per-hop-signed ground truth and to the correlational baseline
(`CorrelationalBaseline.predict` — its β-projection on the named A₅'s move predicts ≈0 for the
factor-decorrelated C₅) and the no-prop floor. Report terminal-MSE for graph vs corr vs noprop, plus a
1-hop B₅ check (sanity: hop-1 alone must still transfer).

## 5. Success criterion (the gate)

For **each** operator (linear, ODE), across **≥6 seeds**:

- **Per-hop sanity:** unrolled 1-hop B₅ beats correlational (this is just the v0 1-hop, must hold).
- **Composition transfer (the result):** unrolled 2-hop C₅ MSE **< correlational** C₅ MSE **and
  < no-prop floor**, robust in **≥5/6 seeds**.

A PASS means nth-order propagation is reachable by iterating the transferable operator. A FAIL (e.g.
hop-1 holds but hop-2 collapses) localises the wall to error-compounding vs. naming-B, which is itself
a sharp finding.

## 6. Files

| File | Change |
|------|--------|
| `src/qts/propagation/sim.py` | ADD `generate_hop_events`, `generate_chain_eval`, `make_unroll_splits` (1-hop train/val on chains 0–4; per-hop-signed test on 0–4 and transfer on chain 5). **Do not touch** `generate_events`/`make_splits`. |
| `src/qts/propagation/unroll.py` | NEW — `unroll_predict`, `UnrollReport`, `evaluate_unroll_transfer`. |
| `src/qts/propagation/__init__.py` | export the new public functions/dataclass. |
| `scripts/run_propagation_unroll.py` | NEW — fit 1-hop, run the unroll gate for both models, print the report. |
| `tests/unit/test_propagation_unroll.py` | NEW — T-PROP-UNROLL-1.. (hop-event generation, unroll-predict on a hand-checkable case, eval well-formedness). |
| `tests/integration/test_propagation_unroll_gate.py` | NEW — T-PROP-UNROLL-GATE-1.. (per-hop-signed truth == iterated 1-hop; composition transfer at a pinned seed for both models). |

## 7. Out of scope (v0 of Path C)

- Reactor **localisation** (argmax over predicted reactions) — eval reads known indices.
- **k>2 hops** (A→B→C→D) — prove 2-hop first; the unroll loop is written general so k>2 is a sim
  extension, not a rewrite.
- Real-data / LLM merit — sim only.
- The internal-vs-external unroll comparison — external only.
- Mutating the committed multi-hop sim or its gate (the §13 negative result stays intact).

## 8. Test IDs

`T-PROP-UNROLL-1..n` (unit), `T-PROP-UNROLL-GATE-1..n` (integration). pytest `--no-cov` locally.
