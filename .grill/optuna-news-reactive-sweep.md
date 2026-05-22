# Grill: Optuna news-reactive parameter sweep
Date: 2026-05-22

## Intent
Tune NewsReactiveMomentum's news params to find **robust** parameters that generalise
to unseen FOMC days — not curve-fit to the single real day we have (2023-12-13). The
deferred v2.1 goal was "beat buy-and-hold"; reframed here as "robust params, with the
real day as a held-out validity check."

## Constraints
- Exactly ONE real FOMC day available (2023-12-13). More real days is a separate
  deferred item, not a prerequisite for this sweep.
- Reuse existing Optuna infra (tuner.py: TPE + MedianPruner; objective.py;
  search_space.py) rather than building a parallel optimiser.
- LLM classification is expensive → frozen episode bank + warm cache **once**. All three
  tuned params are downstream of classification, so the cache stays valid across all trials.
- No leakage: the held-out test day must not calibrate the training data.

## Key decisions
- **Train on simulated episodes, hold out the real day.** Reason: only one real day exists;
  sim provides the diversity robustness needs. Rejected: source 8-12 real days (too
  expensive now), walk-forward within one day (no real diversity, one regime).
- **Make sim tradeable by wiring the MM's existing `sentiment_drift_bps` hook**
  (market_maker.py:33/51/77 — declared + consumed in `_build_quote` but never set) from the
  surprise bucket, WITH lag + noise. Reason: today sim has zero news→price coupling (regime
  `expected_drift` is attached post-hoc in runner.py, never fed back into the bars), so there
  is no alpha to tune toward. Rejected: clean deterministic coupling (trivial optimum),
  post-hoc bar injection (bypasses microstructure), abandon sim (expensive).
- **Randomise (onset-lag, drift-magnitude, noise) PER EPISODE** across plausible ranges.
  Reason: robust-by-construction — Optuna must find params that work across the whole range,
  not memorise one coupling; no test-day leakage. Validity check: the real day's actual
  reaction should fall INSIDE the randomised range. Rejected: calibrate to the real day
  (leaks test into training), single fixed value (overfits belief half-life to one lag).
- **Objective = maximise the 25th-percentile / CVaR of excess-return-vs-buy-and-hold across
  episodes.** Reason: operationalises "robust" (reward params that rarely do badly, not ones
  that occasionally moonshot) AND self-guards against the 0-trade degenerate (a do-nothing
  strat scores negative excess in rally episodes). Rejected: mean Sharpe (hides tails, noisy
  with few trades), mean beat-hold (averages away tail risk).
- **Tune ONLY news_signal_weight, belief_half_life, entry_threshold; freeze the momentum
  base.** Reason: smallest space = most robust + fastest + cleanest attribution of the
  decode-gap edge; these are exactly the 3 params the v2 0-trade bug implicated. Rejected:
  co-tune everything (confounds news-edge vs momentum-tuning, bigger overfit surface),
  two-stage (doubles runs).
- **Real-day pass bar = trades fire + correct direction (long into the dovish/up day) +
  beats buy-and-hold**; logged as directional evidence, explicitly NOT statistical proof
  (n=1). Rejected: bare beat-hold (noisy on one day), fraction-of-move (arbitrary X),
  positive-PnL-only (proves little about the edge).
- **Start 50 episodes × 150 trials; expand if CVaR is jumpy or TPE hasn't converged.**
  Reason: enough episode diversity for a stable lower-quantile + enough trials for TPE on 3
  params, without over-committing compute. Frozen bank, cache warmed once.

## Surfaced assumptions
- The whole sim-train plan silently assumed simulated episodes had tradeable news→price
  structure. They do NOT — the biggest excavated finding; it reshaped scope to include
  building the coupling itself.
- The classification cache stays valid across all trials BECAUSE all three tuned params are
  downstream of text→NewsSignal classification. (If we ever tuned a classification-time
  param, the cache would invalidate per trial and the frozen-bank economics break.)
- The decode-gap alpha in sim comes from the LAG: news → drift onsets over L bars → a
  strategy reading the news at t=0 positions before the price fully moves. If lag→0, there
  is no edge to capture and the tune is meaningless.

## Out of scope
- Sourcing additional real FOMC days (separate deferred item).
- Tuning momentum-base params (frozen at current defaults).
- Live Binance integration, live dashboard (deferred).
- Statistical proof of edge — impossible with n=1 real day; this sweep produces robust
  params + a single validity datapoint, nothing stronger.
