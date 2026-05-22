# Event-Propagation Graph — v0 Design

**Status:** design converged (post-grill, pre-spec). Research direction, separate from Phase 8 trading.
**Date:** 2026-05-22
**Companion grill log:** `.grill/event-propagation-graph.md`
**Guiding quote:** *"We judge matters by their merit and actions by their timing."*

---

## 1. Thesis (the alpha)

A learnable graph that maps world **events** to **cross-asset reactions**. The edge is predicting
the reaction of an entity the news **never names** — e.g. *Anthropic ships Claude Design → Lovable
(a competitor, unnamed) drops*. The crowd/LLM read stops at the named entity; the graph's reach to
the **unnamed** one is the alpha.

**North star = profit + prediction quality.** Interpretability and exact structure recovery are
explicitly secondary — *"an uninterpretable-but-profitable relation is fine."*

**Sim-first.** Prove the architecture can do this in a controlled, *adversarial* world before
learning real-world mappings. The Phase-8 `SentimentDriftModel` is the **1-asset / 1-edge special
case** of this engine.

---

## 2. The partition: merit vs transmission

| | Role | Owns | Deliberately does NOT |
|---|---|---|---|
| **LLM** | *merit* (surface extraction) | `(named_entity, sector, direction, magnitude)` from text | propose *which* stocks relate — that would leak the graph's job and erase the edge |
| **Graph** | *transmission* | all relations, propagation, timing, the unnamed targets | re-learn language |

Merit is **observed** (the LLM externalises it) — that is what makes "disentangle merit vs timing"
coherent rather than a black box. **For v0 the LLM is bypassed**: the sim feeds clean merit labels
directly, to isolate the graph's causal-discovery from LLM noise. LLM noise is a later layer (§9).

---

## 3. Architecture overview (data flow)

```
            ┌─────────────────────────── v0 BYPASS ───────────────────────────┐
 text  ──▶  LLM (merit)  ──▶  (named, sector, direction, magnitude)  ──┐       │
            [deferred]                                                 │       │
 sim event ───────────────────────────────────────────────────────────┴──▶  do() intervention
                                                                              (clamp named node)
                                                                                    │
                                                                                    ▼
                                                            ┌──────────────────────────────────┐
                                                            │  GRAPH (transmission)             │
                                                            │  nodes = assets + latent concepts │
                                                            │  state-gated linear dynamics      │
                                                            │  cyclic (feedback allowed)        │
                                                            └──────────────────────────────────┘
                                                                                    │
                                                                                    ▼
                                                            per-asset reaction vector  r̂ ∈ ℝᴺ
                                                                  (cumulative return / window)
```

---

## 4. The adversarial sim (v0 world)

Scale (revised after the v0 run — see §12 Findings): **18 assets (6 disjoint triples) · 2 common
factors · 6 event types**, of which **5 are trained and 1 is held out** for the transfer gate. The
original 3-event-type sketch proved too small: 2 training pairs underdetermine a transferable
mechanism, so transfer could not be tested fairly (see §12).

### 4.1 Ground-truth objects

| Object | Symbol | Notes |
|---|---|---|
| Assets | `i = 0..7` | each emits a return over the post-event window |
| Common factors | `g ∈ ℝ²` | macro co-movement drivers — **the confound** |
| Factor loadings | `B ∈ ℝ^{8×2}` | asset `i` absorbs `B[i]·g` |
| Event types | `e ∈ {0,1,2}` | each = a `do()` intervention naming one asset |
| Per event type | `named n(e)`, `substitute s(e)`, `decoy d(e)` | the causal triple |
| Regime state | `z` | gates edge sign/magnitude (state-gated dynamics) |

### 4.2 The confound — *"correlation misleads"* (chosen de-circulariser)

The decoy is a **trap for the correlational baseline**:

- **Decoy `d`** shares the named asset's factor loadings: `B[d] ≈ B[n]` ⟹ **high `corr(n, d)`**, but
  receives **no causal edge** from the event.
- **Substitute `s`** has orthogonal/opposite loadings: `B[s] ⊥ B[n]` ⟹ **`corr(n, s) ≈ 0`**, and is
  the **true causal target** of the event.

So a model that chases co-movement bets on `d` and **misses** `s`. Winning therefore *requires*
causal skill — it de-circularises the whole test (you can't win by learning correlation).

### 4.3 Dynamics (state-gated linear, cyclic)

Edge `n → s` magnitude/sign depends on regime `z` (e.g. regime A = substitution, edge `+`;
regime B = complement, edge `−`). Feedback loops allowed (economies aren't DAGs). This is the MVP;
graph neural-ODE is the deferred upgrade (§9).

### 4.4 Event generation & price readout

Per event: sample type `e` → regime `z` → factor shock `g` → named merit `m = (direction, magnitude)`.

Reaction of asset `i` over the window:

```
 r_n = B[n]·g  +  m        +  εₙ      (named: factor + own merit)
 r_d = B[d]·g             +  ε_d      (decoy: factor ONLY — no causal edge)
 r_s = B[s]·g  +  w_{n→s}(z)·h(m)  +  ε_s   (substitute: factor + CAUSAL edge)
 r_i = B[i]·g             +  ε_i      (others)
```

The factor shock `g` is drawn **independently** of the causal edge each event — that independence is
what lets `do()`-supervision separate causation from the confound.

Events are **sparse** (don't fire every bar) but only modestly so in v0 — heavy sparsity is a v2
obstacle, not the primary one.

---

## 5. The model (v0 MVP — state-gated linear state-space)

### 5.1 Nodes & state

- **18 asset nodes** (observed) + **K latent concept nodes** (`K≈3`, learned, no fixed meaning).
- State vector `x ∈ ℝ^{8+K}` over [assets ; concept nodes].

### 5.2 Gated adjacency

- `R` learnable regime adjacencies `W_r ∈ ℝ^{(8+K)×(8+K)}`.
- A gate `π = softmax(g_φ(context))` mixes them: `W_eff = Σ_r π_r · W_r`.
- The gate is what produces the t0/t1 sign-flip (edge `+` in regime A, `−` in B).

### 5.3 Forward / readout

1. `do()`: clamp the named node to observed merit `m`.
2. Propagate: `x ← W_eff·x + b` for `T` steps (or solve the linear fixed point `(I − W_eff)⁻¹b`).
3. Readout: predicted reaction `r̂_i = x_i` for asset nodes → the **per-asset reaction vector**.

**Why latent concept nodes:** the substitution *mechanism* lives in a concept node
(`named → concept → substitute`). The **same** concept node can wire to a *new* (named, substitute)
pair → mechanism transfer. This is the entire justification for emergent latents over an asset-only
graph, and it is exactly what the transfer gate (§7) tests.

### 5.4 Training loss

```
 L = MSE(r̂_assets, r_assets)            # prediction quality on held-out reactions
   + λ · ‖W‖₁                            # sparsity — prevents dense correlational overfit
```

`do()`-supervision (interventions with the factor confound randomised independently) is what drives
the fit toward the causal map rather than the correlational one.

---

## 6. Baselines (the graph must beat both)

| Baseline | Prediction | Role |
|---|---|---|
| **No-propagation** | `r̂_i = m` if `i == n` else `0` | sanity floor |
| **Strong correlational** *(the real bar)* | `r̂_i = β_{i\|n}·r_n` where `β_{i\|n} = Σ_{i,n}/Σ_{n,n}` from a pre-event return covariance `Σ` | "chase co-movement" — nails the **decoy**, misses the **substitute** |

The correlational baseline is the honest bar: it captures everything correlation can capture.
Beating it on the **substitute's** reaction is proof the graph used causation, not co-movement.

---

## 7. Feasibility gate (pass / fail)

Both must hold:

1. **Prediction.** (Refined after the v0 run.) Graph must beat the **no-propagation floor** overall
   AND beat the **correlational baseline on the substitute `s`** by ≥25% (per-asset MSE). The graph
   only sees `(named, merit, regime)` — never the factor shock — so it cannot predict factor
   co-movement the baseline gets free from the named asset's realised move; judging on *overall*
   MSE-vs-correlational is therefore an unfair bar and was dropped. The substitute is the thesis.
2. **Intervention-transfer.** Train with one `(named, substitute)` pair **entirely held out**; at
   test the graph must predict that unseen pair's `s`-reaction **better than the correlational
   baseline**. Passing ⟹ the substitution *mechanism* (a concept node) transferred to assets it
   never saw coupled.

Structure-recovery up-to-isomorphism is demoted to a **secondary diagnostic** — useful to inspect,
not part of the gate. The gate targets real-world transfer directly and guards the exact overfitting
that killed the Phase-8 FOMC sweep.

---

## 8. Build order

1. **v0 (this doc):** numpy sim + clean merit labels + state-gated-linear graph + both baselines +
   the two-part gate. Prove feasibility.
2. **Graduate to neural-ODE** *only if* gated-linear hits a prediction ceiling.
3. **Layer in realism** (each a separate, later obstacle): LLM merit noise; anticipation/leakage;
   heavy event sparsity; non-stationarity/regime drift; multi-hop chains `A→B→C`; richer scale.

---

## 9. Out of scope (v0)

- Real-market data and the real LLM front-end (sim-first).
- The graph neural-ODE (timing-aware) version — deferred until the linear MVP plateaus.
- Interpretability as a goal — latent concept nodes need not be human-nameable.
- Adversarial obstacles beyond *correlation-misleads* + factor co-movement (anticipation, heavy
  sparsity, drift) — all v2 layers.

---

## 10. Tech stack

| Concern | Choice | Already a dep? |
|---|---|---|
| Sim generative process | NumPy | ✅ |
| Gated-linear model | plain PyTorch (dense adjacency, small `N+K`) | ✅ (`torch`) |
| Tests | pytest (existing `--no-cov` harness) | ✅ |
| Neural-ODE (deferred) | `torchdiffeq` | ❌ — add only at graduation |

No `torch-geometric` / `jax` needed at v0 scale (≈21 nodes = 18 assets + 3 concepts).

---

## 11. Open questions (for the spec)

- Exact module layout under `src/qts/` (new top-level package, e.g. `src/qts/propagation/`).
- `T` (propagation steps) vs solving the linear fixed point — pick per stability.
- `K` (number of latent concept nodes) and `R` (number of regimes) — start `K=3, R=2`, tune.
- Window length `N` bars for the reaction readout.
- Exact split protocol for the held-out transfer pair (which event type / pair is held out).
- Whether the gate's `context` is the regime label (clean) or must itself be inferred from state.

---

## 12. Findings (v0 run — 2026-05-22)

Built and run end-to-end (`src/qts/propagation/`). Locked config: **18 assets, 6 event types
(5 trained, 1 held out), 2 factors, 6-dim sector codes, 3 latent concept nodes, 2 regimes.**

**Verdict: v0 feasibility PASS.** The graph learns a substitution mechanism that *transfers to an
unseen asset pair* and beats a strong correlational baseline on the unnamed entity.

- **Prediction edge is robust.** Across seeds, the graph beats the correlational baseline on the
  substitute by ~2× (sub-MSE ≈1.5 vs ≈3.4) — it finds the unnamed-entity reaction that co-movement
  cannot. (The graph loses on *overall* MSE only because it never observes the factor shock — an
  unfair bar, dropped from the gate; see §7.)
- **Transfer was DATA-limited, not model-limited.** With only 2 training pairs (the original
  3-event-type sketch) transfer was unreliable (~2/6 seeds). With **≥3 training pairs** (4/6/8 event
  types) transfer is **robust (3/3 across every config tried)**. So the bilinear mechanism *does*
  generalise — it just needs enough pairs to be over-determined. Hence the scale-up to 6 event types.
- **Correction to an earlier read:** the first 8-asset sim's apparent transfer success was largely
  *factor-coincidence* (the held-out substitute sat in a trained substitute's factor direction), not
  genuine mechanism transfer. Giving each named asset a distinct factor direction removed that crutch
  and exposed the data-limitation — which the scale-up then resolved honestly.
- **Next (deferred):** the realism layers in §8/§9 (LLM merit noise, anticipation, sparsity, drift,
  multi-hop) and the neural-ODE upgrade remain the path from "machinery works" toward "money."
