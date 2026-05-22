# Grill: Event-propagation graph (symbolic event → market-reaction engine)
Date: 2026-05-22
Status: design converged to a full v0; written up at docs/research/2026-05-22-event-propagation-graph-design.md
(pre-spec; research direction, separate from Phase 8 trading)

## Intent
A custom ML architecture: a learnable graph mapping world events to cross-asset reactions.
The alpha is predicting *unnamed* 2nd/nth-order propagation — e.g. Anthropic ships Claude Design
→ Lovable (a competitor NOT named in the news) drops. North star = PROFIT + prediction quality.
Sim-first to prove feasibility before learning real-world mappings. Guiding quote: "judge matters
by their merit and actions by their timing."

## Constraints / priorities
- **Profit & prediction quality are THE lever.** Interpretability and exact structure-recovery are
  explicitly secondary ("an uninterpretable-but-profitable relation is fine").
- **Sim-first**: validate the architecture in a controlled known world before real markets.
- The LLM does ONLY language work; the graph does ALL relational discovery.

## Key decisions
- **Alpha = cross-asset propagation to UN-named entities.** The crowd/LLM-grade read stops at the
  named entity; the graph's reach to the unnamed one is the edge. Heir to the decode-gap thesis.
- **The partition ("reasoning & transmission"):** LLM = *merit* = surface extraction
  `(named entity, sector, direction)` from text, nothing relational. Graph = *transmission* = all
  relations + propagation + timing. Reason: keep the LLM to language (don't make the graph re-learn
  language) AND avoid leakage — if the LLM names the related stock, there's no edge to discover and
  no alpha. Rejected: LLM proposing "what stocks relate" (cannibalises the graph's whole job).
- **Merit is OBSERVED** (the LLM externalises it) — that's what makes "disentangle merit vs timing"
  coherent instead of a black box.
- **Nodes = assets (observed) + EMERGENT latent concept nodes** (graph invents hidden mechanisms).
  Chosen over asset-only / LLM-named for expressiveness + generalisation (a learned mechanism like
  "substitution" transfers to unseen asset pairs).
- **Events = `do()` interventions at known nodes.** Originally to make latent recovery identifiable;
  after the profit-priority pivot, justified instead by out-of-sample/transfer generalisation
  (interventional >> correlational for not-overfitting).
- **Dynamics = cyclic dynamical system on the graph** (feedback allowed; economies aren't DAGs).
  Build order: **state-gated linear (regime-switching) MVP → graph neural-ODE.** State-gating
  produces the t0/t1 sign-flip (edge + in regime A, - in regime B). Rejected static DAG (no feedback).
- **Feasibility gate = PREDICTION + INTERVENTION-TRANSFER:** beat a no-propagation predictor on
  held-out cross-asset reactions AND survive a held-out perturbation (new event type / shifted
  regime). Structure-recovery up-to-isomorphism demoted to a secondary diagnostic. Reason: targets
  real-world transfer directly and guards the exact overfitting that killed the Phase 8 FOMC sweep.
- **Test philosophy = ADVERSARIAL SIM + beat a STRONG CORRELATIONAL baseline** (not just the
  no-propagation floor). De-circularises the feasibility test: if the model wins, it can't be because
  it learned co-movement. Rejected "sim validates machinery, accept near-circularity" as too weak.
- **Confound = "correlation MISLEADS"** (chosen over "correlation insufficient" / "hidden common
  factor"). Each event type has a **decoy** asset highly correlated with the named entity (shared
  factor loadings, NO causal edge — bait for the baseline) and a **substitute** asset that is
  anti-/un-correlated with the named entity but IS the true causal target. Directly models
  Anthropic→Lovable. A win is unambiguous causal skill. (Common-factor confound demoted to v2.)
- **v0 scale = 18 assets · 2 common factors · 6 event types** (REVISED up from 8/3 during the v0 run:
  3 event types left only 2 training pairs, too few to determine a transferable mechanism — see design
  doc §12 Findings). Big enough to foil memorisation, small
  enough to eyeball the learned adjacency vs ground truth. (4/1/1 too memorisable; 20/4/multi-hop = v2.)
- **Readout = per-asset reaction VECTOR** (cumulative return per asset over an N-bar post-event
  window), scored by MSE. Tradeable + directly comparable to the correlational baseline. (Direction-only
  throws away sizing; full-path/timing deferred to v2 despite the "timing" in the guiding quote.)
- **Transfer gate = UNSEEN ASSET PAIR.** Train on some (named→substitute) couplings, test on a NEW
  pair never coupled. Passing proves the substitution *mechanism* (a latent concept node) transferred —
  the whole point of emergent latents. (Regime-flip / new-event-type are narrower; demoted.)
- **v0 BYPASSES the LLM** (feeds clean merit labels) to isolate the graph's causal-discovery for the
  feasibility test. LLM merit-noise is a later realism layer. (Resolves the prior open question — yes.)

## Surfaced assumptions / insights
- Emergent latents are unidentifiable observationally; **interventions break the symmetry** —
  load-bearing for both recovery (diagnostic) and transfer (the gate).
- **Profit-priority RELAXES the identifiability constraint**, freeing the most expressive dynamics
  (neural ODE), judged on prediction not interpretability.
- The drift_model just built for Phase 8 is the **1-asset / 1-edge special case** of this engine.

## Open questions (now scoped to the spec — see design doc §11)
The big design questions above are resolved and written up. What remains is spec-level detail:
- Module layout under `src/qts/` (new package, e.g. `src/qts/propagation/`).
- `T` propagation steps vs solving the linear fixed point `(I − W_eff)⁻¹b`.
- `K` latent concept nodes & `R` regimes (start K=3, R=2), window length `N`.
- Exact held-out split protocol for the transfer pair.
- Whether the gate context is the clean regime label or must be inferred from state.

## Out of scope (for now)
- Real-market data / the real LLM front-end (sim-first).
- Interpretability as a goal (secondary diagnostic only).
- The neural-ODE version (graduate only if gated-linear hits a prediction ceiling).
