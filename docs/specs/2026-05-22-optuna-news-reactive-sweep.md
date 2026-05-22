# Optuna News-Reactive Parameter Sweep — Spec

**Date:** 2026-05-22
**Status:** approved design, ready for planning
**Companion grill log:** `.grill/optuna-news-reactive-sweep.md`
**Depends on:** Phase 8 v2.1 (NewsBelief + Nautilus-native NewsDataPoint dispatch, already shipped)

## Goal

Find **robust** `(news_signal_weight, belief_half_life, entry_threshold)` for
`NewsReactiveMomentum` that generalise across diverse synthetic FOMC episodes, then validate
the best params on the held-out real 2023-12-13 FOMC day.

This is **not** a curve-fit to one day. Robustness is the target; the real day is a single
held-out validity datapoint, not statistical proof of edge.

## Background — why this exists

The first real v2 run (Qwen3.6-35B on real 2023-12-13 data) proved the LLM decodes Powell
correctly but `NewsReactiveMomentum` made **0 trades**. v2.1 fixed the belief-accumulation and
dispatch-causality bugs; the remaining question is whether *tuned* params can beat buy-and-hold.
The three implicated params:

- `news_signal_weight` (`w`) — convex blend weight in `blended = (1-w)·base_alpha + w·news_alpha`
  (news_reactive.py:71). **Hard-constrained to [0, 1]** (news_reactive.py:44).
- `belief_half_life` — `timedelta` controlling conviction decay (news_reactive.py:41, default 4h).
- `entry_threshold` — lives on the **inner** `MomentumStrategy`'s `StrategyParams`, applied to the
  blended alpha. Too high → news contribution never clears it (the v2 bug).

## Tech stack

Python 3.11, Optuna (TPE + MedianPruner — existing `tuner.py`), NautilusTrader backtest via
`run_terrain_backtest` (v2.1 custom-data path with `NewsDataPoint`), numpy for quantiles.

## Design decisions (from grill)

| Decision | Rationale | Rejected |
|---|---|---|
| Train on sim episodes, hold out real day | Only 1 real day exists; sim gives diversity | source more real days (expensive), walk-forward in 1 day (no diversity) |
| Mark sim market to drift-adjusted fair (off fixed pre-event ref) | Sim has zero news→price coupling today; the raw `sentiment_drift_bps` multiply is invisible without flow AND compounds with it | clean coupling (trivial optimum), post-hoc bars (no microstructure) |
| Randomise (lag, magnitude, noise) per episode | Robust-by-construction, no leakage | calibrate to real day (leakage), fixed value (overfits half-life) |
| Objective = 25th-pct/CVaR of excess-vs-hold | Operationalises "robust", self-guards 0-trade | mean Sharpe (hides tails), mean beat-hold (averages tails) |
| Tune only the 3 news params, freeze momentum | Smallest space = robust + clean attribution | co-tune all (confounds edge), two-stage (doubles runs) |
| Real-day bar: trades + direction + beat-hold | Strongest n=1 signal without overclaiming | bare beat-hold (noisy), positive-PnL-only (proves little) |
| 50 episodes × 150 trials, expand if unstable | Stable lower-quantile + TPE convergence | 20×80 (noisy), 100×300 (heavy) |

## Components

### 1. `src/qts/world/drift_model.py` (new)

```python
@dataclass(frozen=True)
class SentimentDriftModel:
    direction: float        # +1 dovish, -1 hawkish, 0 neutral
    onset_lag: timedelta    # time from event to peak drift
    peak_bps: float         # peak drift magnitude in bps
    decay_halflife: timedelta
    noise_std_bps: float
    event_time: datetime
    seed: int

    def value_at(self, now: datetime) -> float:
        """bps drift at `now`. 0 before event_time; ramps to peak over onset_lag;
        decays with decay_halflife afterward; plus seeded gaussian noise."""
```

Shape: `0` for `now < event_time`; linear ramp `0 → direction·peak_bps` over `[event_time,
event_time + onset_lag]`; exponential decay `0.5 ** (Δ/decay_halflife)` after the peak; additive
seeded gaussian noise (`noise_std_bps`). Deterministic given `seed`.

### 2. `src/qts/world/agent_sim.py` (modify)

Add optional `drift_model: SentimentDriftModel | None = None` to `run_agent_sim`. **Mark-to-fair**
(NOT the raw `sentiment_drift_bps` multiply — that is invisible without trade flow and compounds
*with* it). As the first statement inside the `for now in clock.iter_ticks():` loop:
```python
if drift_model is not None and now >= drift_model.event_time:
    if drift_ref_price is None:
        drift_ref_price = last_price          # fixed pre-event reference
    fair = drift_ref_price * (1.0 + drift_model.value_at(now) / 1e4)
    last_price = fair                          # agents perceive the move
    aggregator.add_trade(now, price=fair, qty=0.0)  # empty-tick bar tracks fair
```
`drift_ref_price` (a `float | None`) is captured once at the first on/after-event tick, so the
drift is applied off a fixed reference and never compounds. The MM still quotes its spread around
`last_price` (= fair), so fills/inventory microstructure stays intact. Gated on `drift_model is not
None` → baseline sims are byte-identical (zero regression). The `sentiment_drift_bps` field is left
unused.

### 3. `src/qts/world/runner.py` (modify)

Thread `drift_model` through `run_simulation(..., drift_model=None)` into `run_agent_sim`.
The drift direction must match the surprise bucket already computed at runner.py:111.

### 4. `src/qts/world/episode.py` (modify)

Add `text_events: list[TextEvent] = field(default_factory=list)` to `SimulatedEpisode` (symmetry
with `RealEpisode`, which already carries it). Populate from `sim.text_events` in `run_simulation`.
Needed so the bank carries the events for `NewsDataPoint` conversion at backtest time.

### 5. `src/qts/optimisation/episode_bank.py` (new)

```python
@dataclass(frozen=True)
class CouplingRanges:
    onset_lag_bars: tuple[int, int] = (3, 20)
    peak_bps: tuple[float, float] = (20.0, 120.0)
    noise_std_bps: tuple[float, float] = (5.0, 30.0)
    decay_halflife_bars: tuple[int, int] = (10, 60)

def generate_episode_bank(
    n: int, seed: int, base_scenario: ScenarioConfig,
    ranges: CouplingRanges = CouplingRanges(),
) -> list[SimulatedEpisode]: ...
```

For each of `n` episodes: draw a surprise bucket (mix of dovish/hawkish/neutral) → derive
`fomc_actual_rate` relative to `base_scenario.fomc_expected_rate`; draw `(onset_lag, peak_bps,
noise_std, decay_halflife)` from `ranges` with a per-episode seed; build the `SentimentDriftModel`
(direction from the bucket); call `run_simulation(base_scenario, strategy=object(), seed=ep_seed,
fomc_actual_rate=..., drift_model=...)`. Passing a bare `object()` skips stage-2 `on_text`
forwarding (guarded by `getattr(strategy, "on_text", None)` at runner.py:147). Reproducible from
the top-level `seed`.

### 6. `src/qts/optimisation/search_space.py` (modify — add)

```python
@dataclass(frozen=True)
class NewsParams:
    news_signal_weight: float
    belief_half_life_minutes: float
    entry_threshold: float

def sample_news_params(trial: optuna.Trial) -> NewsParams:
    return NewsParams(
        news_signal_weight=trial.suggest_float("news_signal_weight", 0.1, 0.95),
        belief_half_life_minutes=trial.suggest_float("belief_half_life_minutes", 15, 240, log=True),
        entry_threshold=trial.suggest_float("entry_threshold", 0.02, 0.4),
    )
```

Ranges grounded in code: `news_signal_weight` within the constructor's hard [0,1] limit;
`entry_threshold` floor lowered to 0.02 (vs momentum's 0.05) so small news signals can trigger;
`belief_half_life` upper = 240min = the current 4h default.

### 7. `src/qts/optimisation/news_objective.py` (new)

Cannot reuse `make_objective` (it returns **mean** of `primary_metric` and runs
`run_terrain_backtest` **without** custom_data — news events would never dispatch). New objective:

```python
@dataclass(frozen=True)
class NewsObjectiveContext:
    episodes: list[SimulatedEpisode]
    classifier: NewsClassifier      # cache pre-warmed for all episodes' text
    risk_limits: RiskLimits
    venue_config: VenueConfig = field(default_factory=VenueConfig)
    cvar_quantile: float = 0.25
    min_trades_floor: int = 1       # diagnostic only; objective self-guards via excess

def make_news_objective(ctx: NewsObjectiveContext) -> Callable[[optuna.Trial], float]:
    # per trial:
    #   params = sample_news_params(trial)
    #   strategy_factory: StrategyParams(entry_threshold=..., <frozen momentum defaults>)
    #     -> MomentumStrategy -> NewsReactiveMomentum(inner, ctx.classifier,
    #          belief_half_life=timedelta(minutes=...), news_signal_weight=...)
    #   for each episode:
    #     custom_data = [CustomData(DataType(NewsDataPoint),
    #                    text_event_to_news_data(e)) for e in episode.text_events]
    #     result = run_terrain_backtest(episode.terrain, strategy,
    #                  venue_config, custom_data=custom_data, log_level="ERROR")
    #     excess = strat_return(result) - hold_return(episode.terrain)
    #     report running quantile for pruning
    #   return float(np.quantile(excesses, ctx.cvar_quantile))
```

`hold_return(terrain) = (bars[-1].close - bars[0].close) / bars[0].close`.
`strat_return(result)`: **planning must pin the exact `BacktestResult` field** (likely
`total_return`; else derive from final equity / starting capital). See Open Items.

### 8. `src/qts/optimisation/run_news_sweep.py` (new)

Orchestrator: `generate_episode_bank` → `classifier.warm_cache_for(all text_events)` **once**
(async, before trials) → build `NewsObjectiveContext` → `run_strategy_study` (reuse `tuner.py`,
50 episodes × 150 trials, `direction="maximize"`) wired with `make_news_objective` → take
`best_params` → run the strategy with best params on the **real** 2023-12-13 `RealEpisode` →
evaluate the real-day pass bar → log a structured report (best params, train CVaR, real-day
excess/trades/direction).

## Data flow

```
generate_episode_bank(50) ──► frozen list[SimulatedEpisode]   (each: coupled bars + text_events)
        │
        ├─► classifier.warm_cache_for(all text)   [LLM called once per unique text]
        │
        ▼
Optuna trial (×150): sample 3 params ─► NewsReactiveMomentum
        │   for each episode: run_terrain_backtest(custom_data=NewsDataPoint[...])  [cache hits]
        │   excess_i = strat_return_i − hold_return_i
        ▼
   score = 25th-percentile(excesses)  ──► maximise
        │
        ▼
best_params ─► real 2023-12-13 backtest ─► pass bar: trades>0 ∧ long-into-up ∧ excess>0
```

All three tuned params are downstream of text→`NewsSignal` classification, so the warmed cache
stays valid across **every** trial — that is what makes 150 trials cheap.

## Testing strategy

**Unit:**
- `SentimentDriftModel`: 0 before event; reaches `±peak_bps` at `event+onset_lag`; halves after
  `decay_halflife`; noise is seed-deterministic; neutral direction → ~0.
- `generate_episode_bank`: reproducible from seed; couplings vary across episodes; bucket mix
  present; real 2023-12-13 reaction falls inside the default `CouplingRanges` (validity check).
- `sample_news_params`: respects bounds; `news_signal_weight ≤ 1.0` always.
- `news_objective`: CVaR math correct on a hand-built excess list; a 0-trade strategy scores
  **negative** on a rally episode (degenerate-guard test).

**Integration (T-NEWS-SWEEP-ACCEPT):** a tiny 5-episode × 10-trial sweep runs end-to-end with a
**stub/deterministic classifier** (no live LLM in CI) and returns a `TunerResult` with
`best_params` populated and `n_completed == 10`.

## Acceptance criteria

1. Full sweep (50×150) completes and yields `best_params` for the 3 news params.
2. On the held-out real 2023-12-13 day, best params: **(a)** fire ≥1 trade, **(b)** take a net
   long position into the dovish/up day, **(c)** beat buy-and-hold (positive excess). Logged as
   evidence, explicitly annotated n=1 (not proof).
3. Full test suite stays green (baseline 1180 passed / 4 skipped); new unit + integration tests pass.
4. CI green (ruff check, ruff format --check, mypy, pytest).

## Open items for planning

- **Pin `strat_return` extraction** from `BacktestResult` (confirm `total_return` vs deriving from
  equity). Read `src/qts/nautilus/config.py::BacktestResult` during task 1 of planning.
- Confirm `NewsClassifier.warm_cache_for` is async and the orchestrator awaits it before trials.
- Confirm `text_event_to_news_data` + `CustomData`/`DataType` import paths from the v2.1 work.
- Decide the frozen momentum `StrategyParams` defaults used for the inner strategy (use current
  `MomentumStrategy` defaults; do not tune).

## Out of scope

- Sourcing more real FOMC days (separate deferred item).
- Tuning momentum-base params (frozen).
- Live Binance integration, live dashboard (deferred).
- Multi-worker Optuna (single-worker is fine at 150 trials; `tuner.py` supports workers if needed).
- Statistical proof of edge (impossible at n=1 real day).
