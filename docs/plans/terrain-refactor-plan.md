# Terrain Architecture Refactor — Implementation Plan

**Created**: 2026-04-05
**Status**: Phases 1-7 COMPLETE (Phase 8 deferred)
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

## Phase 8: Synthetic Market Generation (DEFERRED)

- [ ] Regime-switching GARCH generator
- [ ] MarS-style generative models (if Phases 1-7 demand it)
- [ ] Macro Engine generator mode
- [ ] `terrain/synthetic.py`
