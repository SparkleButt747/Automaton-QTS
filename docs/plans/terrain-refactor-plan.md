# Terrain Architecture Refactor — Implementation Plan

**Created**: 2026-04-05
**Status**: Phases 1-8 v1 COMPLETE
**Reference**: docs/research/market-terrain-architecture.md

---

## Phase 1: Foundation — Data Models + NautilusTrader ✓

### 1A: New enums and terrain data models ✓
- [x] Add to `models/base.py`: `Trend`, `VolLevel`, `LiquidityLevel`, `SentimentLevel`, `Catalyst` enums
- [x] Remove `VolRegime` enum — replaced by `VolLevel` (HIGH/LOW/TRANSITIONING)
- [x] Create `models/terrain.py`: `MacroRegime`, `MarketTerrain`, `MarketEvent`, `Level`, `RegimeSpan`
- [x] `MarketTerrain` query methods: `regime_at()`, `liquidity_at()`, `volatility_at()`, `nearest_event()`, `events_in_range()`

### 1B: Migrate all VolRegime references ✓
- [x] All source files, tests, demos, notebooks migrated to `VolLevel`/`vol_level`
- [x] Zero `VolRegime` references remain in src/ or tests/

### 1C: NautilusTrader setup ✓
- [x] `nautilus/__init__.py` — package docstring
- [x] `nautilus/actor.py` — `QTSStrategy(NtStrategy)` actor adapter
- [x] `nautilus/converters.py` — all conversion helpers (bar, fill, order, position)
- [x] `nautilus/runner.py` — `run_terrain_backtest()` → `BacktestResult`
- [x] `nautilus/config.py` — `QTSStrategyConfig`, `VenueConfig`, `BacktestRunConfig`, `BacktestResult`
- [x] `nautilus/catalog.py` — Parquet read/write, CSV→Parquet, `ensure_catalog()`

### 1D: Data pipeline (Parquet canonical storage) ✓
- [x] `nautilus/catalog.py` handles CSV→Parquet and bars→Parquet
- [x] `ensure_catalog()` for idempotent Parquet persistence

### 1E: Move legacy code ✓
- [x] `_legacy/backtest.py` — full copy of old BacktestEngine
- [x] `_legacy/execution_engine.py` — full copy of old ExecutionEngine
- [x] `_legacy/nautilus_adapter.py` — full copy of old adapter
- [x] Originals replaced with re-export shims (all import paths still work)

---

## Phase 2: MarketTerrain as First-Class Object ✓

- [x] `terrain/__init__.py`
- [x] `terrain/builder.py` — `TerrainBuilder` with fluent API, auto-compute features
- [x] `terrain/library.py` — `TerrainLibrary` with train/test split, regime filtering, YAML loading

---

## Phase 3: Historical Regime Annotation ✓

- [x] `terrain/annotator.py` — `HistoricalAnnotator` with price-based regime classification
- [x] 25 YAML terrain definitions in `config/terrains/` (20 train + 5 test)
- [x] Covers: bull, bear, crisis, sideways, high-vol, low-vol, pandemic, geopolitical, regulatory

---

## Phase 4: Macro Engine ✓

- [x] `macro/__init__.py`
- [x] `macro/engine.py` — `MacroEngine` orchestrator (classify mode)
- [x] `macro/indicators.py` — `MacroSnapshot`, `MacroIndicatorFetcher` (placeholder for FRED/AV)
- [x] `macro/classifier.py` — `MacroRegimeClassifier` (LLM-based, produces MacroRegime)

---

## Phase 5: Optimisation Loop ✓

- [x] `optimisation/__init__.py`
- [x] `optimisation/objective.py` — `ObjectiveContext`, `make_objective()` (multi-terrain Sharpe)
- [x] `optimisation/search_space.py` — `sample_momentum_params()`, `sample_mean_reversion_params()`
- [x] `optimisation/tuner.py` — `run_strategy_study()` with TPE, MedianPruner, multi-worker

---

## Phase 6: Perturbation Testing ✓

- [x] `terrain/perturbation.py` — `TerrainPerturbator`, `PerturbationConfig`
- [x] Perturbations: flash_crash, liquidity_shock, data_gap, correlation_breakdown
- [x] `slippage_venue_config()` for Nautilus fill model perturbation
- [x] `generate_all_perturbations()` for batch variant generation

---

## Phase 7: Live Trading Path ✓

- [x] `nautilus/live.py` — `run_live_strategy()` via TradingNode
- [x] `LiveVenueConfig` with testnet support
- [x] Binance adapter integration
- [x] CLI: `qts live --symbol X --venue BINANCE --testnet`

---

## Cross-cutting ✓

- [x] `pyproject.toml` — nautilus_trader, optuna, pyyaml as required deps
- [x] CLI updated: `terrain-backtest`, `terrain-list`, `optimise`, `live` commands
- [x] `models/__init__.py` exports all new types
- [x] All new modules pass syntax validation
- [x] VolRegime→VolLevel migration complete across entire codebase

---

## Phase 8: World Simulator — v1 vertical slice (DELIVERED)

See `docs/specs/2026-05-20-phase-8-world-simulator.md` for the full spec
and `.grill/phase-8-world-simulator.md` for the design log.

v1 delivered:
- [x] `src/qts/world/` package — events, episode, clock, scenario, corpus, sentiment, runner, agent_sim, bar_aggregator
- [x] Agent roster: SchedulerAgent + PersonaAgent (Powell) + 3 configurable AnonRetailAgents + InventoryAwareMM
- [x] SimpleOrderBook stage-1 matching engine
- [x] PersonaCorpus with seeded Powell FOMC YAML (hawkish/dovish/neutral × 4 statements)
- [x] FOMC v1 scenario config (`config/scenarios/fomc_btcusdt_v1.yaml`)
- [x] Acceptance suite — 6 tests covering reproducibility, seed variation, terrain compatibility, JSON serialisation, event_calendar population, text-blind-strategy compat
- [x] Strategy protocol unchanged; QTSStrategy actor forwards text events via duck-typed `on_text`

Deferred (per spec scaling path):
- v1.5: empirical calibration against real Powell/FOMC reactions on BTC
- v2: more scenarios (CPI, NFP, geopolitical, USDT depeg), more personas (Trump, Musk, CEOs, congressional), 2-3 competing MMs, larger anon pool
- v3: news-reactive strategies via `on_text`; Optuna sweeps over `(scenario, agent_roster)` space
- v4: multimodal events via Gemma (Fed press-conference video / audio); Avellaneda-Stoikov MMs; multi-asset cross-correlation
- v5: RL agents trained on real data deployed as adversaries (research-grade)

### Phase 8 v2 — News-Reactive Strategy + Real-Data Acceptance (IN-PROGRESS)

See `docs/specs/2026-05-21-phase-8-news-reactive-strategy.md` for the spec and `.grill/phase-8-news-reactive-strategy.md` for the design log.

v2 architecture delivered:
- [x] `Strategy.on_text(event)` formal protocol method (no-op default on existing strategies)
- [x] `qts.macro.NewsSignal` — multi-axis structured output (direction + confidence + relevance + magnitude)
- [x] `qts.macro.NewsClassifier` — Qwen-backed multi-axis classifier with content-hash disk cache + sync/async paths
- [x] `qts.strategies.BeliefAxis` — exponentially-decaying belief primitive
- [x] `qts.strategies.NewsReactiveMomentum` — composes MomentumStrategy with a Qwen-driven belief overlay
- [x] `qts.data.RealEpisode` — real-data analogue of SimulatedEpisode
- [x] `qts.nautilus.real_runner.run_real_backtest` — pre-dispatches text events, then runs terrain backtest
- [x] Curated `data/real/fomc/2023-12-13/` dataset (1440 bars + statement + 24 press-conf paragraphs)
- [x] Hand-validation tool `scripts/validate_news_classifier.py`
- [x] Acceptance test `tests/integration/test_news_reactive_2023_12_13.py` wired end-to-end

v2 alpha gate: NOT YET PROVEN — pending a working local LLM with a chat-completion model. The acceptance test is committed as a failing regression target. Once Ollama is configured with a working Qwen/GLM/equivalent model, run `scripts/validate_news_classifier.py data/real/fomc/2023-12-13` to warm the cache, then re-run the acceptance test.

Deferred (next-grill candidates, per project memory `project_deferred_grills.md`):
- Optuna sweep over news params (belief half-life, news_signal_weight, prompt variants)
- Multiple historical FOMC days / bulk-data pipeline
- Live Binance integration (scrapling + testnet)
- Other event types (CPI, NFP, geopolitical, USDT depeg)

Original Phase 8 entry (regime-switching GARCH, MarS-style generative models):
explicitly rejected during the grill. The strategy reacts to news+events, not
just price/volume, so a pure price-generative model would not test what
the strategy actually consumes. v1 is news-reactive instead.
