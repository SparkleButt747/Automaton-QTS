# Grill: Phase 8 v2.1 — News-Belief Mechanic + Dispatch Causality Redesign
Date: 2026-05-21

## Intent
The first real Phase 8 v2 run (Qwen3.6-35B-A3B Q6 on llama.cpp, real 2023-12-13 FOMC
data) proved the LLM decodes Powell correctly (net bullish on the dovish pivot:
11 bull / 2 bear / 12 neutral across 25 events) — but `NewsReactiveMomentum` made
**0 trades** and did not beat buy-and-hold (+3.32% on the day). v2.1 fixes the
*mechanic* so the correctly-decoded signal actually reaches the strategy and evolves
causally through the day. Tuning it to beat the market is explicitly Optuna's job, not
v2.1's.

## Constraints
- Do NOT re-litigate what works: `LlamaCppClient`, classifier quality, the content-hash
  disk cache, and the overall classify→belief→strategy→backtest plumbing are solid.
- Backtest classification is cache-only (`classifier.classify()` sync); the 2023-12-13
  cache is already warmed (~25 entries, gitignored, ~10 min to regenerate).
- Same model as ONI: Qwen3.6-35B-A3B Q6 MoE via llama.cpp, default sampling (custom
  sampling params break — confirmed).
- v2.1 must not scope-creep into Optuna's tuning job.

## Key decisions
- **Decision: Belief becomes a multi-axis accumulator**, not a single decaying scalar.
  Reason: the old `BeliefAxis.update()` REPLACED its value, so the final neutral Q&A
  paragraph (alpha 0.0) wiped the entire dovish signal to zero. A single scalar cannot
  hold per-axis structure. Alternative rejected: single-scalar EMA / decaying-sum — still
  collapses direction/confidence/relevance/magnitude into one number.
- **Decision: Combine rule = "relevance gates, confidence drives, magnitude scales":**
  `belief = conviction × relevance_gate × magnitude_scale`, where
  `conviction = decaying signed sum of (direction_sign × confidence)` over events,
  `relevance_gate ∈ [0,1]` (recent relevance), `magnitude_scale` sets amplitude.
  Reason: low-relevance Q&A should be *ignored* (gate→0), not *cancel* a strong prior;
  confident directional reads should *accumulate* conviction; magnitude sets size. Each
  axis is independently tunable for the Optuna sweep. Alternative rejected: naive average
  of axes (lets a neutral read cancel a strong one), single signed product (magnitude /
  relevance must never flip the sign).
- **Decision: Causality via Nautilus-native custom data.** Text events become a Nautilus
  custom data type; `engine.add_data()` + `subscribe_data` let the engine interleave them
  with bars by timestamp. Reason: events for `ts ≤ T` are delivered before bar `T`, so the
  ordering bug is solved at the engine level with no hand-maintained pending list.
  Alternatives rejected: (a) recompute-from-log pure function — simplest and bug-proof, but
  the user wants engine-native fidelity for the eventual live path; (b) incremental +
  manual per-bar drain in the actor — requires keeping dispatch ordering correct forever
  by hand.
- **Decision: Classification happens in the actor handler; custom data carries RAW text.**
  Actor's `on_data` unwraps the text event → `inner.on_text` → strategy classifies via the
  cache + folds the signal into the accumulator. Reason: mirrors live trading exactly
  (text arrives → strategy decodes on receipt); classifier stays in the strategy; the
  existing `on_text` path barely changes. Alternative rejected: pre-classify at data-build
  time (diverges from live, bakes the decode into fixture construction), carry-both (two
  code paths, YAGNI for a single-day backtest).
- **Decision: v2.1 acceptance = non-zero trades + belief evolves causally bar-by-bar.**
  Beat-buy-and-hold is deferred to the Optuna grill. Reason: v2.1's job is a *correct
  mechanic*; tuning weight/half-life/threshold to beat the market needs that correct
  mechanic to exist first. Alternative rejected: require beat-hold now (would scope-creep
  hand-tuning into v2.1).
- **Decision: Decay guard.** Clamp negative elapsed (`now < last_update`) → return the
  current (undecayed) value, never amplify. Reason: kills the latent look-ahead
  amplification bug (`0.5 ** negative > 1`). Native interleaving should make `now <
  last_update` unreachable, but guard defensively anyway.

## Surfaced assumptions
- The warmed 2023-12-13 cache stays valid (content-hash keyed) — cache-only `classify()`
  in the backtest will hit, no live LLM call needed for the re-run.
- Nautilus `add_data` + `subscribe_data(DataType)` delivers custom data in `ts_init` order
  interleaved with bars, with the actor receiving it via `on_data` (subscribe in
  `on_start`). The implementer must confirm the exact 1.221 API.
- Per-axis decay split: **conviction decays** over a tunable half-life; **relevance_gate**
  and **magnitude_scale** are "recent" snapshots (latest event, possibly lightly decayed).
  Exact decay-vs-snapshot per axis to be pinned concretely in the spec.

## Open questions
- Exact decay model per axis (decaying signed-sum vs EMA for conviction; pure snapshot vs
  light decay for the relevance/magnitude gates) — propose concretely in the spec.
- Optuna search ranges for `news_signal_weight`, conviction half-life, and the entry
  threshold — deferred to the Optuna grill. (Recall: old defaults gave weight 0.5 × peak
  alpha ~0.26 = 0.13, below the 0.25 entry threshold — Optuna territory.)

## Out of scope
- Beating buy-and-hold (→ Optuna grill).
- Multiple FOMC days beyond 2023-12-13 (→ deferred grill).
- Live Binance integration, testnet or real (→ deferred grill).
- The classification-stream / belief / PnL dashboard (→ deferred, user-flagged).
- Any change to `LlamaCppClient`, the classifier prompt/quality, the cache, or the overall
  plumbing — all confirmed solid.
