# Grill: Phase 8 v2 — News-Reactive Strategy + Real-Data Acceptance
Date: 2026-05-21

## Intent
v1 ships the news-reactive sim (FOMC + Powell + 3 anons + MM); now build the strategy that proves the alpha hypothesis (Qwen reads news smarter than VADER-grade anons) and validate it against **real** market data on a curated historical FOMC day. End-state: sim + live bot platform, both evolving together. This slice closes the loop from sim infrastructure → working news-reactive strategy → real-data validation.

## Constraints
- **Spot-only** — strategy can go long but cannot short (Phase 8 v1 inherited constraint; out of scope to remove here).
- **Local LLMs only** — Qwen via Ollama backend of the existing LLMClient; no paid APIs.
- **One curated day for acceptance** — 2023-12-13 dovish pivot. Real BTC bars + real Powell text. No bulk historical pipeline yet.
- **Vanilla MomentumStrategy must keep working** — it's the A/B baseline and existing tests pin it. NewsReactiveMomentum is additive.

## Key decisions

- **Decision:** The next slice is the **news-reactive strategy + real-data acceptance**, not v1.5 calibration as the original spec scaling path suggested. **Reason:** Strategy is the alpha hypothesis under test; calibrating a strategy that ignores news would be calibrating the wrong thing. Prove the news layer works on real data first, then revisit calibration with that working strategy. **Alternative rejected:** v1.5 (sim calibration against real Powell/FOMC reactions) — deferred until we have a strategy worth calibrating.

- **Decision:** **NewsReactiveMomentum** as a new strategy that **composes with MomentumStrategy** rather than modifying it in place. **Reason:** Vanilla MomentumStrategy stays as the A/B baseline; both strategies coexist. **Alternative rejected:** Modify MomentumStrategy in place (loses the baseline) and net-new Qwen-only strategy (too much work, can't isolate Qwen's value-add vs momentum).

- **Decision:** Qwen produces **multi-axis structured output** — `NewsSignal(direction, confidence, relevance, magnitude)` — not a single scalar. **Reason:** Direct expression of the decode-gap alpha hypothesis. A single scalar makes Qwen barely smarter than VADER; structured reasoning is where the edge lives. Strategy can gate on relevance, size by confidence × magnitude, and bias direction. **Alternative rejected:** Single scalar in [-1, 1] (lazy; collapses Qwen advantage), free-form text + parser (fragile to Qwen output drift).

- **Decision:** **Decaying belief state**, one input to every bar, with **exponential decay** (half-life ~4h, configurable). **Reason:** Markets metabolise news over hours, not instants. A fire-and-forget reaction underweights slow-burn news cycles. Belief state composes cleanly with MomentumStrategy's existing per-bar signal stack. **Alternative rejected:** Event-triggered fire-and-forget (forgets FOMC after one tick), sticky no-decay (week-old reads carry equal weight as fresh ones).

- **Decision:** Acceptance test is **NewsReactiveMomentum day-end equity > buy-and-hold day-end equity** on real 2023-12-13 BTC bars + real Powell text. **Reason:** Beats-baseline-strategy is too weak (could just track momentum). Beats buy-and-hold on a +7% rally day is honest — strategy must add value via timing AND re-entries on top of market beta. Hard bar; clean signal if passed. **Alternative rejected:** Head-to-head vs vanilla MomentumStrategy (cleaner attribution but lower bar), earlier-entry timing test (measures timing not PnL), positive-PnL-only (trivial on an up day).

- **Decision:** Test day is **2023-12-13 (Powell pivot)** — dovish, BTC +7% in 24h. **Reason:** Clean dovish narrative, large move, spot-only strategy can go long and capture the upside. Hawkish days would limit strategy to "avoided drawdown" which is messier PnL signal. **Alternative rejected:** 2024-09-18 first cut (messier narrative — 50bp vs 25bp ambiguity), 2022-11-02 hawkish 75bp, 2024-12-18 hawkish dot plot (both spot-shortable issues).

- **Decision:** Real data lives at `data/real/fomc/2023-12-13/{bars.csv, statement.txt, press_conf.json}`, populated by **a one-off fetcher script** (`scripts/fetch_fomc_data.py`). Fetched data **committed to repo** for reproducibility. **Reason:** federalreserve.gov URLs can change; committing the dataset isolates the test from upstream drift. **Alternative rejected:** Live-fetch every test run (flaky), gitignored data (loses reproducibility across machines).

- **Decision:** New **`RealEpisode`** type in `qts.data.real_episode` that wraps real bars + real text events into a MarketTerrain-shape, mirroring `SimulatedEpisode`. **Reason:** Strategy and runner don't know the difference between sim and real — same interface, swappable backend. Matches the spec's live-mirroring boundary intent from the original Phase 8 grill. **Alternative rejected:** Standalone test harness (doubles the integration surface), extending SimulatedEpisode (couples real/sim concepts).

- **Decision:** **Formalise `Strategy.on_text`** in the protocol with a no-op default. **Reason:** Phase 8 v1 used a duck-typed forward; relying on duck-typing across the codebase is fragile. Formal protocol with default makes the contract explicit; existing strategies stay text-blind via the default. **Alternative rejected:** Keep duck-typing (works but lossier).

- **Decision:** Qwen invocation is **cached by content hash** on disk. **Reason:** Local Qwen is slow (~5-15s/call); the same FOMC press release text shouldn't be re-classified across runs. Same key shape as Phase 8 v1's corpus cache. **Alternative rejected:** No cache (prohibitively slow under Optuna sweeps later), in-memory only (loses cache across runs).

## Surfaced assumptions

- **Local Qwen via Ollama is reachable and well-prompted enough to produce useful multi-axis output.** Validated indirectly by the existing macro classifier using the same LLMClient. Failure mode: Qwen produces garbage structured output → strategy reads noise. Mitigation: hand-validate Qwen outputs on the corpus before running the real-data test.
- **The 2023-12-13 FOMC statement + press conference text contains enough hawkish/dovish vocabulary for Qwen to classify confidently.** Plausible — Powell's pivot was a clear language shift. Risk: real Fed-speak is more ambiguous than the hand-tuned v1 corpus.
- **Binance kline API returns clean 1m bars for 2023-12-13 without authentication.** True for `klines` endpoint; spot-checked previously.
- **`run_terrain_backtest` can be extended to dispatch TextEvents at their timestamps without breaking the existing test suite.** Phase 8 v1 already proved this works for SimulatedEpisode → MarketTerrain → run_terrain_backtest; same pattern.
- **Beating buy-and-hold on a +7% day is achievable with spot-only + Qwen-grade entry timing + re-entries on intraday pullbacks.** Unvalidated. Risk: even a well-built strategy might tie buy-and-hold. Mitigation: track this as the headline metric; if missed, investigate whether spot-only is the binding constraint.

## Open questions

- **Optimal Qwen prompt** for the multi-axis output. Will be hand-tuned during implementation; revisit during Optuna grill (deferred).
- **Belief half-life default** — 4h is a guess. Should be exposed as a config knob; tuning is for the Optuna grill.
- **What `relevance` means concretely** for FOMC text. Probably hardcoded to 1.0 for FOMC sources in v1; future event types (CPI, Trump tweets) need relevance scoring to discount low-impact items.
- **Caching key invalidation** when Qwen prompt changes. Same problem as Phase 8 v1 corpus cache; defer to the future bulk-data slice.

## Out of scope (explicit)

- **Optuna sweep over the new strategy's params** — deferred to its own grill; flagged by user as important.
- **Multiple historical FOMC days / bulk-data pipeline** — deferred; user flagged as important. Single curated day is the v2 acceptance.
- **Live Binance integration** (testnet or real) — deferred; user flagged as important. Closes the platform loop but is a separate slice with its own data plumbing and risk surface.
- **Other event types** (CPI, NFP, geopolitical, USDT depeg) — architecture should be event-agnostic but only FOMC is tested in v2. CPI/NFP corpora come later.
- **Multimodal events** (Gemma video/audio of press conferences) — v4 territory per the original Phase 8 scaling path.
- **Avellaneda-Stoikov MM upgrade / additional MMs** — orthogonal to the news strategy; deferred.
- **RL adversaries** — research-grade; deferred indefinitely per original grill.
- **Removing the spot-only constraint** (margin / futures) — would change risk surface dramatically; deferred to its own design discussion.
