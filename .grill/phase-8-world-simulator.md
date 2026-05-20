# Grill: Phase 8 — World Simulator
Date: 2026-05-20

## Intent
This is a money-making system, not coursework. The strategy reacts to news + world events + watched accounts (US senators, Fed, CEOs, anon trader sentiment). Phase 8 is an **internal simulation of the world**, not a price generator: a venue where the strategy can be developed and stress-tested before pointing it at real flows on the web. Pre-paper-trading requirement — they want a credible internal sim BEFORE risking a Binance testnet.

Secondary intent: produce training data for future ML/LLM-based strategies.

## Constraints
- **Local LLMs only**: Qwen 3.6 for text, Gemma for multimodal (Gemma deferred to later — v1 is text-only).
- **No paid APIs**: anything that needs an API key gets dropped. Scrapling handles bot detection for everything else (Twitter, Truth Social, Reddit, Fed RSS, congressional hearings).
- **Live-mirroring interface**: same shape of data the sim produces is what scrapling will produce live. Sim is just one backend behind a swappable boundary.
- **The strategy already has Qwen-backed regime classification**: anons must NOT share that signal — the strategy's alpha hypothesis depends on it being smarter at signal extraction than the crowd.
- **Realistic price formation**: spreads, slippage, liquidity withdrawal under stress must be modelled. Real Nautilus matching engine, not a toy fill model.

## Key decisions

- **Decision**: Multi-agent co-simulation with order-matching price formation, not pre-baked narratives. **Reason**: news-reactive strategies need to see realistic order flow + liquidity dynamics, not scripted reactions; same matching engine as live execution closes the sim-to-real gap. **Alternative rejected**: pre-baked narrative scripts (faster but loses emergent dynamics + microstructure realism).

- **Decision**: v1 is a vertical slice — BTCUSDT, Fed rate decision scenario, 1 Powell persona (LLM-cached), 3 configurable anon retail agents (default: sentiment-chaser + trend-follower + mean-reverter), 1 inventory-aware market maker. **Reason**: ~1-2 weeks of work, builds confidence in the architecture before sinking 4+ weeks into the full roster. **Alternative rejected**: full agent roster from day one (too much to debug if architecture is wrong).

- **Decision**: LLM strategy has two modes — pre-generated persona-reaction corpus (offline batch, sim samples) + live LLM with aggressive caching by (persona, event, regime) tuple. **Reason**: corpus mode for fast Optuna iteration / training data generation; cache mode for high-fidelity one-off scenarios. **Alternative rejected**: pure-runtime LLM (28h per training set is prohibitive).

- **Decision**: Anons use VADER + keyword regex; strategy uses Qwen via the existing macro/classifier. **Reason**: the alpha hypothesis is "we extract signal better than the crowd" — the sim must enforce that asymmetry to be a meaningful test. **Alternative rejected**: shared LLM with corrupted prompts (expensive, hand-tuned corruption is fragile).

- **Decision**: Output type is `SimulatedEpisode` that **wraps** a `MarketTerrain` — `SimulatedEpisode.terrain` is the existing MarketTerrain, plus agent_trace, llm_corpus_refs, order_log. **Reason**: existing Optuna/perturbator/runner consume the wrapped terrain unchanged; debug tooling consumes the full episode. **Alternative rejected**: standalone SimulatedEpisode type (doubles the integration surface for adapters in Optuna + runner).

- **Decision**: Market maker is an inventory-aware naive quoter — bid/ask around theoretical mid (last + sentiment-weighted drift), width = base_spread + k × recent_realised_vol, leans quotes when inventory one-sided. **Reason**: captures the three things that matter for strategy testing (spreads exist, spreads widen in vol, MM defends against adverse flow) at ~150 LOC. **Alternative rejected**: Avellaneda-Stoikov (academic-grade but overkill for v1 scope).

- **Decision**: Calibration is hand-tuned in v1, empirically calibrated in v2. **Reason**: don't block v1 on collecting real Powell tweet datasets. But: instrument every output (event, persona-text, anon-flow, MM-state, resulting-price-move) so a v2 calibration job has the dataset waiting. **Alternative rejected**: calibrate first (3-5 days of data collection before the first end-to-end run).

- **Decision**: v1 episode is a 24h trading day with FOMC roughly mid-day, 1m bars (1440 bars per episode). **Reason**: captures pre-positioning + event + drift back to normal — the realistic trading shape. **Alternative rejected**: 4h compact window (faster but misses pre/post dynamics) and multi-day FOMC week (1000s of bars, hard to debug for v1).

- **Decision**: v1 acceptance test is round-trip reproducibility — declare scenario → generate N episodes by seed → run strategy → get SimulatedEpisode with metrics → same seed reproduces exactly. **No realism claim**. **Reason**: realism is v2 (after empirical calibration). v1 proves the pipeline exists, is debuggable, and is reproducible — pre-conditions for anything more ambitious.

- **Decision**: Strategy ingestion via a new optional `on_text(event: TextEvent)` callback on the Strategy protocol. **Reason**: clean split — existing strategies (MomentumStrategy, MeanReversion) ignore text and stay on the on_bar path unchanged; news-reactive strategies opt in and consume Qwen-grade signals. **Alternative rejected**: bake sentiment into SignalSnapshot (sim would run Qwen on the strategy's behalf, killing the decode-gap asymmetry).

- **Decision**: Code lives in `src/qts/world/`. **Reason**: clean separation from `qts/macro/` (classifier) and `qts/terrain/` (data layer). Reads as "the world generates the terrain". **Alternative rejected**: `qts/synth/` (undersells the multi-agent angle) and extending `qts/macro/` (already 4 modules, would overcrowd).

## Scaling path (post v1)

The vertical slice is explicitly v1. Documented scale-out path so we know where we're heading:

1. **v1**: Fed rate decision, BTCUSDT, 1 Powell + 3 anons + 1 MM, round-trip reproducible — **THE CURRENT TARGET**.
2. **v1.5**: Empirical calibration against real Powell/FOMC reactions; verify sim price moves match real magnitude/timing.
3. **v2**: Additional scenarios (CPI prints, NFP, geopolitical shocks, USDT depeg, regulatory action); additional personas (Trump, Musk, CEOs, congressional figures); larger anon pool (10-100 configurable agents); 2-3 competing MMs.
4. **v3**: News-reactive strategies that consume the `on_text` callback directly; Optuna runs over agent-roster permutations.
5. **v4**: Multimodal events via Gemma (Fed press conference video → speech-to-text → persona reactions); Avellaneda-Stoikov MMs; multi-asset cross-correlation.
6. **v5**: RL agents trained on real data deployed in the sim as adversaries (research-grade).

## Surfaced assumptions

- **Qwen and the local LLM stack are reliable enough to host as a service**: corpus pre-generation and live caching both depend on this. Failure mode: Qwen produces low-quality persona reactions and the sim teaches the strategy to react to noise. Mitigation: spot-check generated corpus; ship with a corpus-only mode as the default for v1.
- **Persona behaviours can be hand-tuned to produce plausible price reactions**: not validated yet. v2 calibration is the only path to confidence here.
- **Nautilus's matching engine can be driven by multiple actors emitting orders**: actor.py + QTSStrategy already use this pattern; the new MM/anon agents are just more actors. Assumption is consistent with Nautilus's design but unverified for the specific multi-actor pattern we want.
- **The 'configurable mix' for anon agents will produce realistic order flow without per-event tuning**: assumed; v2 calibration will validate.
- **Scrapling can reach all live sources we eventually need**: Twitter especially is hostile to scrapers; might force fallback to alternative aggregators or Bluesky.

## Open questions

- **Reproducibility under multi-threading**: Nautilus's BacktestEngine may execute agents non-deterministically. Need to verify a fixed seed across all RNGs + LLM cache + Nautilus event clock produces identical episodes. Defer to implementation; test as part of v1 acceptance.
- **Persona text → anon order size translation**: VADER score → buy/sell size needs a mapping. Hand-tune for v1; calibrate v2.
- **Bootstrap problem**: at episode start, Nautilus has no bars/quotes. How does the first bar get formed? Likely: seed with last_known_price + initial MM quote, then let agents trade. Detail belongs in the implementation plan.
- **LLM corpus invalidation**: when persona prompts change, corpus is stale. Need a cache-key versioning scheme.

## Out of scope (explicit)

- **Multimodal events** (Gemma video/audio) — deferred to v4.
- **Equities and FX** — v1 is BTCUSDT only. Other assets in v2+.
- **Empirical calibration against real history** — v2 deliverable.
- **Avellaneda-Stoikov MM** — too academic for v1.
- **RL-based agents** — research-level, deferred indefinitely.
- **Live deployment plumbing** — the swap to scrapling-fed real flows is a separate project (the sim's whole purpose is to validate the strategy BEFORE that step).
- **Twitter API**: dropped per scrapling-only constraint.
