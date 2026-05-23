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

## 13. Findings (multi-hop / 2-hop run — 2026-05-23)

Built the 2-hop chain sim (`EventChain`): each event is `named(A) -> substitute(B, via relation R1)
-> terminal(C, via relation R2)`, with **24 assets (6 disjoint chains, 5 trained + 1 held out),
3 factors** (so C is factor-orthogonal to *both* A and B), and **disjoint R1/R2 code blocks**. A->C
has no direct feature match — reaching C requires composing two learned relations. Spec:
`docs/specs/2026-05-23-propagation-multihop.md`.

**Verdict: in-sample composition works; 2-hop transfer does not robustly generalise.** *(The "hard
wall / invariant to data-scaling" framing below is partly revised by §14: scaling to E≈25 chains DOES
lift transfer into a noisy ~2–4/6 band — it is data-limited, not flat. The bottom line — no robust
≥5/6 transfer by any lever — still stands.)*

- **In-sample 2-hop is learned.** The graph beats the correlational baseline on *both* the 1-hop
  substitute (B) and the 2-hop terminal (C) by ≥25% (GATE-2 passes at seed 0; in-sample prediction
  ~4/5 across seeds). The model genuinely composes R1 ∘ R2 when it has seen the chain.
- **Transfer does NOT generalise to an unseen chain.** GATE-3 fails (marked `xfail`): on the held-out
  chain the graph is no better than — sometimes worse than — correlational on B and C. Best result
  ~2/5 seeds.
- **Levers tried (transfer stays low — but see §14 for the data-scaling revision):** data-scaling
  (tested only to E=12 *here* — §14 shows E=25 lifts it), model class
  (linear bilinear → continuous-time neural-ODE via `torchdiffeq`), learning rate, epochs, and edge
  capacity (1–3 message-passing heads). Adding capacity raises *in-sample* fit (H=2 → prediction 5/5)
  but not transfer (still 2/5; H=3 overfits worse) — the signature of a **generalisation wall, not a
  capacity wall**.
- **Even 1-hop transfer regressed** vs the v0 result (was robust 6/6). The richer 15-dim feature
  space gives the free bilinear `M` cross-block freedom to fit the 5 training chains with
  chain-specific solutions that don't transfer. The R1/R2 codes are cleanly disjoint by construction,
  so this is an inductive-bias / optimisation limit, **not a sim-design flaw**.
- **Untested lever (deferred, no longer "bounded"):** bias `M` toward a block-structured /
  similarity-kernel form so generic relation-matching is favoured over memorisation — but this edges
  toward hard-coding the very structure the model was meant to discover, so it was not pursued.
- **Status:** the v0 **1-hop** result remains the shipped feasibility claim. The multi-hop machinery
  (sim + gate + `GraphNeuralODE`) is committed as a **documented negative result**, with the transfer
  gate `xfail`. Next realism work should build on the proven 1-hop substrate.

## 14. Findings (data-scaling flood + unroll-composition "Path C" + related work — 2026-05-23)

### 14.1 Data-scaling revises §13: data-limited, not a flat wall

Flooding the sim with chains (constant model, end-to-end linear) moves 2-hop transfer:

| chains (E) | end-to-end 2-hop transfer (of 6 seeds) |
|------------|----------------------------------------|
| 6–12 | ~1/6 |
| 25 | **2/6–4/6** (draw-sensitive — same E gave 4/6 under one split RNG, 2/6 under another) |
| 40 | 1/6–3/6 |
| 50 | 0/6 (data-*per-chain* starved at fixed total n_train, plus optimiser divergence) |

So §13's "invariant to data-scaling" was wrong (we only tested to E=12): more chains **do** lift
transfer — into a noisy ~2–4/6 band — but it **plateaus well short of robust ≥5/6**. In most seeds the
graph is statistically tied with the correlational baseline on the unseen chain. 2-hop transfer is
**data-responsive but not data-solvable**; the residual gap is an inductive-bias problem.

### 14.2 Unroll-composition (Path C) — dominated by end-to-end

Trained a **1-hop-only** operator (naming both A and B so R1 *and* R2 are learned as local maps; chain
held out), then composed externally at inference (predict B → re-inject as merit → predict C). Code:
`src/qts/propagation/{sim.py:generate_hop_events,generate_chain_eval,make_unroll_splits, unroll.py}`,
tests `T-PROP-UNROLL-*`. Result: unroll transfer **≤2/6**, *worse* than end-to-end — the second hop
(R2/B→C) fails to fire on an unseen chain (terminal ≈ no-prop floor). `prop_steps=1` did not help.
Its value was **diagnostic**: it localises the wall to the *second* relation not transferring.

### 14.3 Related work — this is a KNOWN problem with KNOWN (untried) fixes

The failure is textbook on three fronts; we used a *known-weak* scorer (our bilinear `M` is a
feature-conditioned DistMult/RESCAL variant, pre-2017):

- **Systematic / compositional generalisation** — learn primitives, fail novel compositions. SCAN
  (Lake & Baroni 2018), COGS (Kim & Linzen 2020), MLC (Lake & Baroni, *Nature* 2023). Hard *without*
  the right inductive bias, routinely solved *with* it — matches our data plateau exactly.
- **Inductive KG link prediction** — bilinear/embedding models memorise endpoint statistics and fail
  multi-hop composition; the stated motivation for path-based GNNs: GraIL (Teru et al., ICML 2020),
  **NBFNet** (Zhu et al., NeurIPS 2021), A*Net (Zhu et al. 2022), RED-GNN (2022).
- **Causal mechanism transfer** — ICM (Schölkopf et al. 2021) explains *why* (entangled, non-modular
  mechanisms don't compose); transportability (Bareinboim & Pearl 2014) gives the identifiability
  language; the meta-transfer sparse-gradient objective (Bengio et al. 2020) is an actionable
  modularity penalty.

**Three convergent untried levers** (in rough order of leverage):
1. **NBFNet-style query-conditioned path aggregation** — threads the composition through the forward
   pass (relation-typed, entity-agnostic), so multi-hop chains transfer to unseen entities. The most
   direct fix; our bilinear scorer only scores endpoints and gives no gradient signal for chains.
2. **MLC meta-learning the composition** — training episodes that reward *inferring* R1∘R2, not fitting
   it as one more pattern.
3. **ICM meta-transfer sparse-gradient penalty** — enforce modular, reusable A→B / B→C mechanisms.

### 14.4 Novelty & profit read

- **Core ML: not novel.** It is inductive relational reasoning / compositional generalisation, both
  with mature literatures and SOTA solutions.
- **Application: novel.** No single paper combines {LLM-extracted event *merit* + a *learned*
  propagation graph + multi-hop prediction of *unnamed* entities + adversarial-vs-correlational eval}.
  Closest prior art is **FinRipple** (ACL Findings 2025) — but its knowledge graph is pre-specified
  and static-within-window, multi-hop depth is not demonstrated, and it has no propagating "merit"
  signal. The economic-link alpha itself is well-established: Cohen & Frazzini (2008, *JF* — 2.7%/mo
  for high-inattention names), Menzly & Ozbas (2010), Hou (2007), Lo & MacKinlay (1990).
- **Bottom line:** a credible profit *and* publishable story exists — *if* the ML is upgraded
  (NBFNet-style path aggregation), taken to real data, and positioned explicitly against FinRipple.

### 14.5 Decision

2-hop / n-hop transfer is **banked for now** — no robust transfer with the bilinear model — but it is
**not a dead end**: there are documented, theoretically-grounded untried levers (§14.3) and a genuinely
novel application (§14.4). Immediate direction: **Path A** — take the proven 1-hop graph + LLM merit
extractor to **real events** (where profit is decided). The prime future bet was an **NBFNet-style
path-aggregation** operator — now **tested, see §14.6 (it did not help)**.

## 14.6 NBFNet-style path aggregation — tested, does NOT crack it (2026-05-23)

Implemented a faithful *minimal* NBFNet-style operator (`src/qts/propagation/model_nbf.py`,
`NBFPropagation`): per-node **vector** hidden state (d=16), query-conditioned merit seed at the do()
source, feature-conditioned regime-gated edges (reusing the bilinear `M`, so transfer is via features
not entity IDs), and **L=3 learnable message-passing layers** with distinct per-hop relation-message +
self-update weights. Drops into `fit_graph`/`evaluate_feasibility` unchanged.

| lr | E=12 transfer | E=25 transfer | E=25 sub-win (R1) | E=25 terminal-win (R1∘R2) |
|----|---------------|---------------|-------------------|----------------------------|
| 3e-3 | 2/6 | 2/6 | 4/6 | 3/6 |
| 1e-2 | 2/6 | **3/6** | **5/6** | 3/6 |

**Verdict: NO** — path aggregation lands in the *same noisy 2–4/6 band* as the bilinear (best 3/6),
not robust ≥5/6. The decisive **sub-finding** (consistent across lr): the **1-hop substitute (R1)
transfers well (4–5/6)** but the **2-hop terminal composition (R1∘R2) does not (3/6)** — on seeds where
R1 transfers strongly, the terminal actively *diverges*. So the failure is **specifically the
composition**, now **consistent across all three architectures** (bilinear, neural-ODE, NBFNet
path-aggregation): richer models improve *in-sample* fit, never *transfer*.

**Caveat:** this is a minimal adaptation (continuous feature-conditioned edges, unnormalised sum
aggregation) — a full SOTA NBFNet (discrete relation embeddings, PNA/normalised aggregation, more
depth/width) is untested. But the cross-architecture consistency points to the **few-training-chains /
inductive-bias regime of the sim** as the bottleneck, not the scorer. Remaining untried levers:
**MLC-style meta-learning of the composition** and the **ICM meta-transfer sparse-gradient penalty**
(§14.3) — both are training-objective changes, not architecture swaps, which is the more likely lever
given three architectures have now failed identically. `model_nbf.py` committed as documented
exploration.

## 15. Findings — meta-resampling CRACKS n-hop transfer (2026-05-23)

**The §14.5 "banked" verdict is OVERTURNED.** The missing lever was the **training objective, not the
architecture** — exactly the call §14.6 made. The *same* feature-conditioned bilinear operator that
"failed" in §13/§14, trained with **MLC-style episodic relation-resampling** (§14.3 lever #2; Lake &
Baroni, *Nature* 2023), transfers its learned composition **robustly to entirely unseen worlds**. The
bilinear was never the bottleneck; the fixed-world training regime was. Productionised in
`src/qts/propagation/meta.py` (`MetaPropagationGraph`, `train_meta`, `few_shot_adapt`,
`evaluate_meta_transfer`), CLI `scripts/run_propagation_meta.py`, tests `T-PROP-META-*` +
`T-PROP-META-GATE-1`. The sweep scripts below are throwaway (`/tmp/meta_*.py`); the committed module +
CLI reproduce the headline.

### 15.1 The mechanism

Resample the world's relations **every training step** from a pool of distinct worlds, with ONE shared
operator `M` (features passed at forward time, so `M` binds to feature-codes, never entity IDs). Unable
to memorise per-world directions, it is forced into the generic composable rule. Stability sweet spot:
`prop_steps=3` (reaches a 2-hop terminal; **`prop_steps=8` diverges** under diverse worlds), gradient
clipping (1.0), `lr=3e-3`. Eval = held-out worlds with seeds disjoint from the training pool.

### 15.2 Robust zero-shot transfer (2000-world pool, 50 held-out worlds, 6 seeds)

| role | win-rate | graph MSE | corr MSE |
|------|----------|-----------|----------|
| SUB (1-hop B) | **50/50** every seed | ~2.15 | ~3.35 |
| TERM (2-hop C) | **50/50** every seed | ~2.65 | ~3.34 |

Both hops transfer to unseen worlds on every seed — where the single-world `fit_graph` (T-PROP-GATE-3)
fails. This is the result that cracks the wall.

### 15.3 Diversity, NOT count, is the lever

Two controls isolate the axis. **(a) Count** (fixed world, more chains E, constant per-chain data,
bilinear, 8 seeds): 1-hop becomes robust (`sub-win 8/8` at E=25) but **2-hop stays a coin-flip**
(`term-win` 3/8→5/8 across E=6..75; FULL-transfer 2–4/8). More data of the *same* relations does not
compose. **(b) Diversity-dosage** (pool size 1→2000, **constant 20k steps**, 50 held-out worlds):

| pool size | SUB capture (win) | TERM capture (win) |
|-----------|-------------------|--------------------|
| 1 | −536% (4/50) | −507% (3/50) |
| 3 | −276% (8/50) | −266% (4/50) |
| **10** | **64% (50/50)** | **42% (49/50)** |
| 30 | 61% (50/50) | 39% (50/50) |
| 100 | 55% (50/50) | 32% (50/50) |
| 2000 | 52% (50/50) | 31% (50/50) |

A **sharp phase change between 3 and 10 worlds**: at pool ≤3 the operator memorises and predicts
*worse than zero* on held-out worlds; at **pool ≥10 it snaps to robust 50/50** and stays robust.
Capture is *highest* at pool=10 (an easier averaging job) and eases slightly toward 2000. **The
diversity floor is ≈10 distinct relations** — real markets (hundreds–thousands of economic links)
clear it by 1–2 orders of magnitude.

### 15.4 Zero-shot capture decays with depth — a structural ceiling

`capture = (corr_mse − graph_mse) / signal_var` (corr ≈ predicts-zero for the factor-orthogonal roles;
1.0 = per-world oracle). Zero-shot: **1-hop ~52%, 2-hop ~31%, 3-hop ~3%**. This ceiling is **structural**
— unmovable by any of:

| lever | effect on capture |
|-------|-------------------|
| operator capacity (learned encoder, 32/64-d) | **hurts** (1-hop 52%→30%) |
| training length (20k→40k) | no change |
| propagation depth (`prop_steps` 3→5→8→12) | **strictly hurts** (2-hop 31%→21%→15%→12%) |

It is the price of *one* shared operator generalising across all worlds with zero observations. A 3-hop
world (A→B→C→D) confirms the decay: QUAD wins ~90% of the time but captures only ~3% — winning by a
hair, near-noise, and not recoverable by deeper propagation.

### 15.5 Few-shot adaptation breaks the ceiling at every depth

Test-time adaptation (`few_shot_adapt`: fine-tune a copy on K support events, L2-anchored to the
meta-init) recovers most of the signal. 2-hop world (anchored, 30 worlds):

| K-shot | SUB capture | TERM capture |
|--------|-------------|--------------|
| 0 | 51% | 30% |
| 32 | 56% | 37% |
| 128 | 79% | 67% |
| 256 | **83%** | **69%** |

3-hop world — **QUAD is rescued** (the decisive result):

| K-shot | SUB | TERM | QUAD |
|--------|-----|------|------|
| 0 | 43% | 20% | **3%** |
| 128 | 81% | 85% | 63% |
| 512 | 87% | 91% | **72%** |

So **no hop is fundamentally lost** — the 3-hop "structural decay" is a *zero-shot* phenomenon, not
unlearnability. Deeper hops just need more data. Small-K (<~32) overfits the support set's
factor/idiosyncratic noise without the anchor (an aggressive un-anchored recipe hit 91%/87% at K=128 on
2-hop but blew up at K=8/32 — the optimal adaptation strength scales with K).

### 15.6 What this means for real data (Path A)

- **Zero-shot** = a relation never seen fire → still beats correlation, more so for shallow links.
- **Few-shot** = a relation with some history → recovers most causal signal at *all* depths.
- Real markets have huge relation diversity (≫ the ~10 floor) and history per relation (the few-shot
  regime). **Path A is upgradeable** from 1-hop-only to the full n-hop meta-mechanism.

### 15.7 Novelty & honest caveats

- **Core ML not novel** (MLC meta-learning is the established mechanism); the **application + this
  characterisation** (diversity-not-count, structural zero-shot depth-ceiling, few-shot rescue on
  event-propagation graphs) is the novel contribution. Closest prior art FinRipple (ACL 2025): fixed
  KG, no demonstrated multi-hop transfer.
- **Caveats:** (1) synthetic sim — relation codes are orthogonal *by construction*; real features are
  noisier and correlated, so the real diversity floor may be higher. (2) Zero-shot capture is *partial*
  (~52%/31%) — profit on deep/novel links needs the few-shot regime or shallow hops. (3) Only the core
  mechanism (`meta.py` + CLI + tests) is committed; the capacity/depth/dosage/3-hop sweep scripts are
  throwaway and live only in the run logs.
