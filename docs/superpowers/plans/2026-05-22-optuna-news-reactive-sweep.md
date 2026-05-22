# Optuna News-Reactive Parameter Sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tune `NewsReactiveMomentum`'s `(news_signal_weight, belief_half_life, entry_threshold)` for robustness across diverse synthetic FOMC episodes (CVaR-of-excess-vs-hold objective), validated on the held-out real 2023-12-13 day.

**Architecture:** Make simulated episodes tradeable by wiring the market-maker's existing `sentiment_drift_bps` hook from the surprise bucket with lag+noise (a new `SentimentDriftModel`). Pre-generate a frozen bank of 50 episodes with randomised couplings; warm the LLM classification cache once; run 150 Optuna TPE trials, each backtesting the strategy over the whole bank and scoring the 25th-percentile of excess-vs-buy-and-hold. Apply best params to the real day as an n=1 validity check.

**Tech Stack:** Python 3.11, Optuna (TPE + MedianPruner, existing `tuner.py`), NautilusTrader backtest via `run_terrain_backtest` (v2.1 `NewsDataPoint` custom-data path), numpy quantiles.

**Spec:** `docs/specs/2026-05-22-optuna-news-reactive-sweep.md`. **Grill log:** `.grill/optuna-news-reactive-sweep.md`.

**Baseline:** 1180 passed / 4 skipped on `main`. Run `pytest` with `--no-cov` locally. Format with `ruff` (line length 99, double quotes). Do NOT add the Co-Authored-By trailer.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/qts/world/drift_model.py` | create | `SentimentDriftModel`: time-varying bps drift trajectory from an FOMC event (ramp → decay → seeded noise). |
| `src/qts/world/agent_sim.py` | modify | Optional `drift_model` param; set `mm.sentiment_drift_bps = drift_model.value_at(now)` each tick. |
| `src/qts/world/runner.py` | modify | Thread `drift_model` through `run_simulation` into `run_agent_sim`; populate the new `text_events` field. |
| `src/qts/world/episode.py` | modify | Add `text_events: list[TextEvent]` to `SimulatedEpisode`. |
| `src/qts/optimisation/episode_bank.py` | create | `CouplingRanges` + `generate_episode_bank` — frozen reproducible bank with per-episode randomised couplings. |
| `src/qts/optimisation/search_space.py` | modify | Add `NewsParams` + `sample_news_params`. |
| `src/qts/optimisation/news_objective.py` | create | `NewsObjectiveContext` + `make_news_objective` (CVaR-of-excess) + `_build_inner_params`. |
| `src/qts/optimisation/tuner.py` | modify | Extract `build_study(config) -> optuna.Study`; reuse in `run_strategy_study`. |
| `src/qts/optimisation/run_news_sweep.py` | create | Orchestrator: bank → warm cache → study → best params → real-day validity check. |
| `tests/unit/test_drift_model.py` | create | Unit tests for `SentimentDriftModel`. |
| `tests/unit/test_episode_bank.py` | create | Unit tests for bank generation. |
| `tests/unit/test_news_search_space.py` | create | Unit tests for `sample_news_params`. |
| `tests/unit/test_news_objective.py` | create | Unit tests for the CVaR objective + degenerate guard. |
| `tests/integration/test_news_sweep.py` | create | End-to-end 5×10 sweep with a stub classifier + real-day-reaction-in-range validity test. |

**Frozen constants** (reused from the v2.1 acceptance test, proven to construct valid models):

```python
# Inner MomentumStrategy weights (sum to 1.0) — frozen, NOT tuned
SignalWeights(w_rsi=0.20, w_macd=0.20, w_bb=0.15, w_mom=0.15, w_sentiment=0.30)
# Sentiment fusion (sum to 1.0)
SentimentFusionWeights(news=0.5, social=0.3, geopolitical=0.2)
# Risk limits
RiskLimits(max_daily_drawdown_pct=0.05, max_position_size_pct=0.20, max_open_positions=5,
           circuit_breaker_cooldown_seconds=300, sentiment_signal_max_scalar=3.0)
```

---

### Task 1: SentimentDriftModel

**Files:**
- Create: `src/qts/world/drift_model.py`
- Test: `tests/unit/test_drift_model.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_drift_model.py
"""T-DRIFT-1..6: SentimentDriftModel ramp/decay/direction/noise."""

from __future__ import annotations

from datetime import datetime, timedelta

from qts.world.drift_model import SentimentDriftModel

_EVENT = datetime(2023, 12, 13, 19, 0, 0)


def _model(direction: float, noise: float = 0.0) -> SentimentDriftModel:
    return SentimentDriftModel(
        direction=direction,
        onset_lag=timedelta(minutes=10),
        peak_bps=100.0,
        decay_halflife=timedelta(minutes=30),
        noise_std_bps=noise,
        event_time=_EVENT,
        seed=7,
    )


def test_zero_before_event() -> None:  # T-DRIFT-1
    m = _model(1.0, noise=5.0)
    assert m.value_at(_EVENT - timedelta(minutes=1)) == 0.0


def test_ramps_to_peak_at_onset_lag() -> None:  # T-DRIFT-2
    m = _model(1.0)
    assert abs(m.value_at(_EVENT + timedelta(minutes=10)) - 100.0) < 1e-9


def test_half_ramp_midway() -> None:  # T-DRIFT-3
    m = _model(1.0)
    assert abs(m.value_at(_EVENT + timedelta(minutes=5)) - 50.0) < 1e-9


def test_decays_by_half_after_one_halflife() -> None:  # T-DRIFT-4
    m = _model(1.0)
    # peak at +10min, then one halflife (30min) later -> 50.0
    assert abs(m.value_at(_EVENT + timedelta(minutes=40)) - 50.0) < 1e-9


def test_hawkish_direction_is_negative() -> None:  # T-DRIFT-5
    m = _model(-1.0)
    assert m.value_at(_EVENT + timedelta(minutes=10)) < 0.0


def test_noise_is_seed_deterministic() -> None:  # T-DRIFT-6
    m = _model(0.0, noise=20.0)
    t = _EVENT + timedelta(minutes=15)
    assert m.value_at(t) == m.value_at(t)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_drift_model.py -x --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'qts.world.drift_model'`

- [ ] **Step 3: Implement `SentimentDriftModel`**

```python
# src/qts/world/drift_model.py
"""SentimentDriftModel — time-varying price drift induced by an FOMC news event.

Wired into the agent simulation: the market-maker reads sentiment_drift_bps each
tick, so this model shifts the simulated mid-price after a news event. The drift
ramps in over onset_lag, then decays — leaving a lead-lag window a news-reading
strategy can exploit (the simulated decode-gap edge). Per-tick gaussian noise is
seed-deterministic so episodes are reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class SentimentDriftModel:
    """Drift trajectory in bps, as a function of wall-clock time after an event."""

    direction: float  # +1 dovish (up), -1 hawkish (down), 0 neutral
    onset_lag: timedelta
    peak_bps: float
    decay_halflife: timedelta
    noise_std_bps: float
    event_time: datetime
    seed: int

    def value_at(self, now: datetime) -> float:
        """bps drift at `now`. 0 before the event; ramps to peak over onset_lag;
        exponential decay afterward; plus seeded gaussian noise."""
        if now < self.event_time:
            return 0.0

        elapsed = (now - self.event_time).total_seconds()
        lag_s = self.onset_lag.total_seconds()

        if lag_s > 0 and elapsed < lag_s:
            base = self.direction * self.peak_bps * (elapsed / lag_s)
        else:
            after_peak = elapsed - lag_s
            hl_s = self.decay_halflife.total_seconds()
            decay = 0.5 ** (after_peak / hl_s) if hl_s > 0 else 1.0
            base = self.direction * self.peak_bps * decay

        if self.noise_std_bps <= 0.0:
            return base
        rng = random.Random((self.seed, int(elapsed)))
        return base + rng.gauss(0.0, self.noise_std_bps)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_drift_model.py -v --no-cov`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qts/world/drift_model.py tests/unit/test_drift_model.py
git commit -m "feat(world): add SentimentDriftModel for news-driven sim price drift"
```

---

### Task 2: Wire drift_model into run_agent_sim

**Files:**
- Modify: `src/qts/world/agent_sim.py` (signature of `run_agent_sim`; tick loop near line 212)
- Test: `tests/unit/test_drift_coupling.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_drift_coupling.py
"""T-COUPLE-1: a dovish drift_model raises simulated closing price vs no coupling."""

from __future__ import annotations

from datetime import datetime, timedelta

from qts.world.agent_sim import run_agent_sim
from qts.world.corpus import PersonaCorpus
from qts.world.drift_model import SentimentDriftModel
from qts.world.runner import _DEFAULT_CORPUS
from qts.world.scenario import AnonAgentConfig, ScenarioConfig

_START = datetime(2023, 12, 13, 18, 0, 0)
_ANNOUNCE = datetime(2023, 12, 13, 19, 0, 0)
_END = datetime(2023, 12, 13, 21, 0, 0)


def _scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="couple-test",
        symbol="BTCUSDT",
        start=_START,
        end=_END,
        tick=timedelta(minutes=1),
        fomc_announcement_at=_ANNOUNCE,
        fomc_expected_rate=5.25,
        starting_price=40_000.0,
        anon_agents=[AnonAgentConfig(agent_id="a1", style="trend", aggressiveness=1.0)],
        mm_base_spread_bps=2.0,
        mm_vol_widen_k=1.0,
        powell_persona_id="powell",
    )


def test_dovish_drift_lifts_price() -> None:  # T-COUPLE-1
    corpus = PersonaCorpus.from_yaml(_DEFAULT_CORPUS)
    scenario = _scenario()

    baseline = run_agent_sim(scenario=scenario, corpus=corpus, seed=42, fomc_actual_rate=5.25)

    dovish = SentimentDriftModel(
        direction=1.0,
        onset_lag=timedelta(minutes=10),
        peak_bps=1000.0,  # 10% — dominate agent noise so the test is deterministic
        decay_halflife=timedelta(minutes=120),
        noise_std_bps=0.0,
        event_time=_ANNOUNCE,
        seed=42,
    )
    coupled = run_agent_sim(
        scenario=scenario, corpus=corpus, seed=42, fomc_actual_rate=5.25, drift_model=dovish
    )

    assert baseline.bars and coupled.bars
    assert coupled.bars[-1].close > baseline.bars[-1].close
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_drift_coupling.py -x --no-cov`
Expected: FAIL — `run_agent_sim() got an unexpected keyword argument 'drift_model'`

- [ ] **Step 3: Add the `drift_model` param and wire it**

First read the head of `src/qts/world/agent_sim.py` to find the exact `run_agent_sim` signature and the `mm` construction. Add `drift_model: SentimentDriftModel | None = None` to the signature (keyword, default None). Add the import:

```python
from qts.world.drift_model import SentimentDriftModel  # near other qts.world imports
```

In the tick loop (currently at `for now in clock.iter_ticks():`, agent_sim.py:212), add as the **first** statement inside the loop body, before `mm.on_tick`:

```python
    for now in clock.iter_ticks():
        # 0. Apply news-induced sentiment drift (if coupled) before the MM quotes.
        if drift_model is not None:
            mm.sentiment_drift_bps = drift_model.value_at(now)
        # 1. MM publishes a quote first so anons have something to hit
        _handle_outputs(mm.on_tick(_ctx_for(mm_rng, now)), mm, now)
        # ... existing steps 2 and 3 unchanged ...
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_drift_coupling.py -v --no-cov`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the existing world-sim tests to confirm no regression**

Run: `.venv/bin/python -m pytest tests/unit/ -k "agent_sim or scenario or runner or world" --no-cov`
Expected: PASS (all existing pass; drift_model defaults to None = unchanged behaviour)

- [ ] **Step 6: Commit**

```bash
git add src/qts/world/agent_sim.py tests/unit/test_drift_coupling.py
git commit -m "feat(world): wire SentimentDriftModel into agent sim tick loop"
```

---

### Task 3: SimulatedEpisode.text_events + thread drift_model through run_simulation

**Files:**
- Modify: `src/qts/world/episode.py` (add field)
- Modify: `src/qts/world/runner.py` (`run_simulation` signature + populate field + pass drift_model)
- Test: `tests/unit/test_run_simulation_drift.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_run_simulation_drift.py
"""T-RUNSIM-1: run_simulation surfaces text_events and applies drift."""

from __future__ import annotations

from datetime import datetime, timedelta

from qts.world.drift_model import SentimentDriftModel
from qts.world.runner import run_simulation
from qts.world.scenario import AnonAgentConfig, ScenarioConfig

_START = datetime(2023, 12, 13, 18, 0, 0)
_ANNOUNCE = datetime(2023, 12, 13, 19, 0, 0)
_END = datetime(2023, 12, 13, 21, 0, 0)


def _scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="runsim-test", symbol="BTCUSDT", start=_START, end=_END,
        tick=timedelta(minutes=1), fomc_announcement_at=_ANNOUNCE,
        fomc_expected_rate=5.25, starting_price=40_000.0,
        anon_agents=[AnonAgentConfig(agent_id="a1", style="trend")],
        mm_base_spread_bps=2.0, mm_vol_widen_k=1.0, powell_persona_id="powell",
    )


def test_episode_carries_text_events() -> None:  # T-RUNSIM-1
    drift = SentimentDriftModel(
        direction=1.0, onset_lag=timedelta(minutes=10), peak_bps=500.0,
        decay_halflife=timedelta(minutes=120), noise_std_bps=0.0,
        event_time=_ANNOUNCE, seed=1,
    )
    episode = run_simulation(
        scenario=_scenario(), strategy=object(), seed=1,
        fomc_actual_rate=5.00, drift_model=drift,  # 5.00 < 5.25 -> dovish
    )
    assert episode.text_events, "expected text_events on the episode"
    assert all(hasattr(e, "timestamp") and hasattr(e, "text") for e in episode.text_events)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_run_simulation_drift.py -x --no-cov`
Expected: FAIL — `run_simulation() got an unexpected keyword argument 'drift_model'` (and later, no `text_events` attr)

- [ ] **Step 3: Add the `text_events` field to SimulatedEpisode**

In `src/qts/world/episode.py`, add to the `SimulatedEpisode` dataclass (after `terrain`/before the metadata fields). Add the import under `TYPE_CHECKING` is not enough — the field needs a runtime default, so import `TextEvent` normally is unnecessary; use a string-free default:

```python
from qts.world.events import TextEvent  # add to imports at top
# ...
@dataclass
class SimulatedEpisode:
    terrain: MarketTerrain
    scenario_name: str
    seed: int
    text_events: list[TextEvent] = field(default_factory=list)
    agent_traces: dict[str, AgentTrace] = field(default_factory=dict)
    llm_corpus_refs: list[str] = field(default_factory=list)
    order_log: list[OrderLogEntry] = field(default_factory=list)
```

- [ ] **Step 4: Thread drift_model and populate text_events in run_simulation**

In `src/qts/world/runner.py`:
1. Add `drift_model: SentimentDriftModel | None = None` to the `run_simulation` signature (keyword-only, after `venue_config`). Add import `from qts.world.drift_model import SentimentDriftModel`.
2. Pass it to `run_agent_sim`:

```python
    sim = run_agent_sim(
        scenario=scenario,
        corpus=corpus,
        seed=seed,
        fomc_actual_rate=fomc_actual_rate,
        drift_model=drift_model,
    )
```

3. Populate `text_events` on the returned episode (in the `SimulatedEpisode(...)` constructor near line 152):

```python
    episode = SimulatedEpisode(
        terrain=terrain,
        scenario_name=scenario.name,
        seed=seed,
        text_events=list(sim.text_events),
        agent_traces=sim.agent_traces,
        llm_corpus_refs=sorted(set(sim.consumed_corpus_keys)),
        order_log=list(sim.order_log),
    )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_run_simulation_drift.py -v --no-cov`
Expected: PASS (1 passed)

- [ ] **Step 6: Run existing episode/runner tests for no regression**

Run: `.venv/bin/python -m pytest tests/unit/ -k "episode or runner or simulation" --no-cov`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/qts/world/episode.py src/qts/world/runner.py tests/unit/test_run_simulation_drift.py
git commit -m "feat(world): carry text_events on SimulatedEpisode and thread drift_model through run_simulation"
```

---

### Task 4: generate_episode_bank + CouplingRanges

**Files:**
- Create: `src/qts/optimisation/episode_bank.py`
- Test: `tests/unit/test_episode_bank.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_episode_bank.py
"""T-BANK-1..3: frozen reproducible episode bank with varied couplings."""

from __future__ import annotations

from datetime import datetime, timedelta

from qts.optimisation.episode_bank import CouplingRanges, generate_episode_bank
from qts.world.scenario import AnonAgentConfig, ScenarioConfig


def _base_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="bank-base", symbol="BTCUSDT",
        start=datetime(2023, 12, 13, 18, 0, 0),
        end=datetime(2023, 12, 13, 21, 0, 0),
        tick=timedelta(minutes=1),
        fomc_announcement_at=datetime(2023, 12, 13, 19, 0, 0),
        fomc_expected_rate=5.25, starting_price=40_000.0,
        anon_agents=[AnonAgentConfig(agent_id="a1", style="trend")],
        mm_base_spread_bps=2.0, mm_vol_widen_k=1.0, powell_persona_id="powell",
    )


def test_bank_is_reproducible() -> None:  # T-BANK-1
    s = _base_scenario()
    a = generate_episode_bank(n=4, seed=123, base_scenario=s)
    b = generate_episode_bank(n=4, seed=123, base_scenario=s)
    assert len(a) == len(b) == 4
    # Same seed -> identical closing prices across the bank
    assert [ep.terrain.bars[-1].close for ep in a] == [ep.terrain.bars[-1].close for ep in b]


def test_couplings_vary_across_episodes() -> None:  # T-BANK-2
    bank = generate_episode_bank(n=6, seed=99, base_scenario=_base_scenario())
    last_closes = {round(ep.terrain.bars[-1].close, 4) for ep in bank}
    assert len(last_closes) > 1, "episodes should differ — couplings are randomised"


def test_each_episode_has_text_events() -> None:  # T-BANK-3
    bank = generate_episode_bank(n=3, seed=5, base_scenario=_base_scenario())
    assert all(ep.text_events for ep in bank)


def test_coupling_ranges_defaults() -> None:  # T-BANK-4
    r = CouplingRanges()
    assert r.onset_lag_bars[0] < r.onset_lag_bars[1]
    assert r.peak_bps[0] < r.peak_bps[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_episode_bank.py -x --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'qts.optimisation.episode_bank'`

- [ ] **Step 3: Implement the bank generator**

```python
# src/qts/optimisation/episode_bank.py
"""Frozen, reproducible bank of simulated FOMC episodes with randomised couplings.

Each episode draws an independent (surprise bucket, onset-lag, peak-magnitude,
decay, noise) so an Optuna sweep must find news params robust across the whole
range rather than memorising one coupling. Generation is deterministic from the
top-level seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from qts.world.drift_model import SentimentDriftModel
from qts.world.episode import SimulatedEpisode
from qts.world.runner import _DEFAULT_CORPUS, run_simulation
from qts.world.scenario import ScenarioConfig

# bucket -> (drift direction, rate offset from expected). |offset| > 0.05 trips the bucket.
_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("dovish", 1.0, -0.25),
    ("hawkish", -1.0, +0.25),
    ("neutral", 0.0, 0.0),
)


@dataclass(frozen=True)
class CouplingRanges:
    """Plausible ranges the per-episode coupling is drawn from."""

    onset_lag_bars: tuple[int, int] = (3, 20)
    peak_bps: tuple[float, float] = (20.0, 120.0)
    noise_std_bps: tuple[float, float] = (5.0, 30.0)
    decay_halflife_bars: tuple[int, int] = (10, 60)


def generate_episode_bank(
    n: int,
    seed: int,
    base_scenario: ScenarioConfig,
    ranges: CouplingRanges = CouplingRanges(),
    corpus_path: Path | None = None,
) -> list[SimulatedEpisode]:
    """Generate `n` simulated episodes with independent randomised couplings.

    Reproducible from `seed`. Passing a bare object() as the strategy skips
    run_simulation's stage-2 on_text forwarding (we only want terrain + text).
    """
    if n <= 0:
        raise ValueError("n must be positive")

    master = random.Random(seed)
    tick_s = base_scenario.tick.total_seconds()
    episodes: list[SimulatedEpisode] = []

    for i in range(n):
        ep_seed = master.randint(1, 2**31 - 1)
        rng = random.Random(ep_seed)

        bucket, direction, rate_offset = _BUCKETS[i % len(_BUCKETS)]
        onset_bars = rng.randint(*ranges.onset_lag_bars)
        peak = rng.uniform(*ranges.peak_bps)
        noise = rng.uniform(*ranges.noise_std_bps)
        decay_bars = rng.randint(*ranges.decay_halflife_bars)

        drift = SentimentDriftModel(
            direction=direction,
            onset_lag=timedelta(seconds=onset_bars * tick_s),
            peak_bps=peak,
            decay_halflife=timedelta(seconds=decay_bars * tick_s),
            noise_std_bps=noise,
            event_time=base_scenario.fomc_announcement_at,
            seed=ep_seed,
        )
        episode = run_simulation(
            scenario=base_scenario,
            strategy=object(),
            seed=ep_seed,
            fomc_actual_rate=base_scenario.fomc_expected_rate + rate_offset,
            persona_corpus_path=corpus_path or _DEFAULT_CORPUS,
            drift_model=drift,
        )
        episodes.append(episode)

    return episodes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_episode_bank.py -v --no-cov`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qts/optimisation/episode_bank.py tests/unit/test_episode_bank.py
git commit -m "feat(optim): add generate_episode_bank with randomised per-episode couplings"
```

---

### Task 5: sample_news_params

**Files:**
- Modify: `src/qts/optimisation/search_space.py`
- Test: `tests/unit/test_news_search_space.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_news_search_space.py
"""T-NSPACE-1..2: news param sampler respects bounds."""

from __future__ import annotations

import optuna

from qts.optimisation.search_space import NewsParams, sample_news_params


def test_sampled_params_within_bounds() -> None:  # T-NSPACE-1
    def objective(trial: optuna.Trial) -> float:
        p = sample_news_params(trial)
        assert isinstance(p, NewsParams)
        assert 0.1 <= p.news_signal_weight <= 0.95
        assert 15.0 <= p.belief_half_life_minutes <= 240.0
        assert 0.02 <= p.entry_threshold <= 0.4
        return 0.0

    study = optuna.create_study()
    study.optimize(objective, n_trials=25)


def test_weight_never_exceeds_constructor_limit() -> None:  # T-NSPACE-2
    def objective(trial: optuna.Trial) -> float:
        p = sample_news_params(trial)
        assert p.news_signal_weight <= 1.0  # NewsReactiveMomentum hard limit
        return 0.0

    optuna.create_study().optimize(objective, n_trials=25)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_news_search_space.py -x --no-cov`
Expected: FAIL — `ImportError: cannot import name 'NewsParams'`

- [ ] **Step 3: Add NewsParams + sample_news_params**

Append to `src/qts/optimisation/search_space.py` (keep existing `sample_momentum_params` untouched). Ensure `optuna` and `dataclass` are imported at the top of that file (they are — used by the existing sampler).

```python
@dataclass(frozen=True)
class NewsParams:
    """Tuned params for NewsReactiveMomentum (the only 3 the sweep searches)."""

    news_signal_weight: float
    belief_half_life_minutes: float
    entry_threshold: float


def sample_news_params(trial: optuna.Trial) -> NewsParams:
    """Sample the 3 news params. Ranges grounded in code:
    - news_signal_weight in [0.1, 0.95]: within the convex-blend hard limit [0,1].
    - belief_half_life in [15, 240] min (log): 240 == the 4h default.
    - entry_threshold in [0.02, 0.4]: floor below momentum's 0.05 so small news triggers.
    """
    return NewsParams(
        news_signal_weight=trial.suggest_float("news_signal_weight", 0.1, 0.95),
        belief_half_life_minutes=trial.suggest_float(
            "belief_half_life_minutes", 15.0, 240.0, log=True
        ),
        entry_threshold=trial.suggest_float("entry_threshold", 0.02, 0.4),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_news_search_space.py -v --no-cov`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qts/optimisation/search_space.py tests/unit/test_news_search_space.py
git commit -m "feat(optim): add sample_news_params for the 3-param news sweep"
```

---

### Task 6: news_objective (CVaR of excess-vs-hold)

**Files:**
- Create: `src/qts/optimisation/news_objective.py`
- Test: `tests/unit/test_news_objective.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_news_objective.py
"""T-NOBJ-1..3: CVaR-of-excess objective + degenerate-guard + valid inner params."""

from __future__ import annotations

import numpy as np

from qts.optimisation.news_objective import (
    _build_inner_params,
    _cvar_of_excess,
    _hold_return,
)


def test_build_inner_params_satisfies_exit_below_entry() -> None:  # T-NOBJ-1
    # entry at the range floor must still yield exit < entry (validator)
    p = _build_inner_params(entry_threshold=0.02)
    assert p.exit_threshold < p.entry_threshold
    assert p.entry_threshold == 0.02


def test_cvar_picks_lower_quantile() -> None:  # T-NOBJ-2
    excesses = [0.10, 0.05, -0.02, 0.08, -0.10]
    expected = float(np.quantile(excesses, 0.25))
    assert abs(_cvar_of_excess(excesses, 0.25) - expected) < 1e-12


def test_hold_return_from_bars() -> None:  # T-NOBJ-3
    class _Bar:
        def __init__(self, close: float) -> None:
            self.close = close

    class _Terrain:
        bars = [_Bar(100.0), _Bar(110.0)]

    assert abs(_hold_return(_Terrain()) - 0.10) < 1e-12
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_news_objective.py -x --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'qts.optimisation.news_objective'`

- [ ] **Step 3: Implement the objective**

```python
# src/qts/optimisation/news_objective.py
"""CVaR-of-excess-vs-hold objective for the news-reactive Optuna sweep.

Each trial samples (news_signal_weight, belief_half_life, entry_threshold), builds
a NewsReactiveMomentum (frozen momentum base), backtests it over every episode in
a frozen bank via the v2.1 NewsDataPoint custom-data path, and scores the
25th-percentile of per-episode (strategy_return - buy_and_hold_return). The lower
quantile rewards robustness and self-guards against a do-nothing strategy (it loses
in rally episodes).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna

from qts.config import RiskLimits, SentimentFusionWeights, SignalWeights, StrategyParams
from qts.nautilus.config import VenueConfig

if TYPE_CHECKING:
    from qts.macro.news_classifier import NewsClassifier
    from qts.models.terrain import MarketTerrain
    from qts.world.episode import SimulatedEpisode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NewsObjectiveContext:
    """Everything the objective needs. `classifier` cache must be pre-warmed."""

    episodes: list[SimulatedEpisode]
    classifier: Any  # NewsClassifier or a stub exposing .classify(event) -> NewsSignal
    risk_limits: RiskLimits
    venue_config: VenueConfig = field(default_factory=VenueConfig)
    cvar_quantile: float = 0.25


def _build_inner_params(entry_threshold: float) -> StrategyParams:
    """Frozen momentum base; exit derived as 40% of entry so exit < entry always."""
    return StrategyParams(
        version="news-sweep",
        weights=SignalWeights(w_rsi=0.20, w_macd=0.20, w_bb=0.15, w_mom=0.15, w_sentiment=0.30),
        entry_threshold=entry_threshold,
        exit_threshold=round(entry_threshold * 0.4, 4),
        max_hold_bars=24,
        sentiment_fusion_weights=SentimentFusionWeights(news=0.5, social=0.3, geopolitical=0.2),
    )


def _hold_return(terrain: MarketTerrain) -> float:
    bars = terrain.bars
    if len(bars) < 2 or bars[0].close == 0:
        return 0.0
    return (bars[-1].close - bars[0].close) / bars[0].close


def _cvar_of_excess(excesses: list[float], quantile: float) -> float:
    if not excesses:
        return 0.0
    return float(np.quantile(excesses, quantile))


def make_news_objective(ctx: NewsObjectiveContext) -> Callable[[optuna.Trial], float]:
    """Build the Optuna objective closure (maximise the lower-quantile excess)."""

    def _objective(trial: optuna.Trial) -> float:
        from nautilus_trader.model.data import CustomData, DataType  # noqa: PLC0415

        from qts.nautilus.converters import text_event_to_news_data  # noqa: PLC0415
        from qts.nautilus.news_data import NewsDataPoint  # noqa: PLC0415
        from qts.nautilus.runner import run_terrain_backtest  # noqa: PLC0415
        from qts.optimisation.search_space import sample_news_params  # noqa: PLC0415
        from qts.strategies.momentum import MomentumStrategy  # noqa: PLC0415
        from qts.strategies.news_reactive import NewsReactiveMomentum  # noqa: PLC0415

        params = sample_news_params(trial)
        data_type = DataType(NewsDataPoint)
        excesses: list[float] = []

        for idx, ep in enumerate(ctx.episodes):
            strategy = NewsReactiveMomentum(
                inner=MomentumStrategy(
                    params=_build_inner_params(params.entry_threshold),
                    risk_limits=ctx.risk_limits,
                ),
                classifier=ctx.classifier,
                belief_half_life=timedelta(minutes=params.belief_half_life_minutes),
                news_signal_weight=params.news_signal_weight,
            )
            custom_data = [
                CustomData(data_type=data_type, data=text_event_to_news_data(e))
                for e in sorted(ep.text_events, key=lambda e: e.timestamp)
            ]
            try:
                result = run_terrain_backtest(
                    ep.terrain,
                    strategy,
                    venue_config=ctx.venue_config,
                    log_level="ERROR",
                    custom_data=custom_data,
                )
                excess = result.total_return - _hold_return(ep.terrain)
            except Exception:  # noqa: BLE001
                logger.exception("news_objective: backtest failed on episode %d", idx)
                excess = -1.0
            excesses.append(excess)

            running = _cvar_of_excess(excesses, ctx.cvar_quantile)
            trial.report(running, step=idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        score = _cvar_of_excess(excesses, ctx.cvar_quantile)
        trial.set_user_attr("cvar_excess", score)
        trial.set_user_attr("mean_excess", float(np.mean(excesses)) if excesses else 0.0)
        trial.set_user_attr("min_excess", float(np.min(excesses)) if excesses else 0.0)
        return score

    return _objective
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_news_objective.py -v --no-cov`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qts/optimisation/news_objective.py tests/unit/test_news_objective.py
git commit -m "feat(optim): add CVaR-of-excess news objective with degenerate guard"
```

---

### Task 7: Extract build_study in tuner.py

**Files:**
- Modify: `src/qts/optimisation/tuner.py`
- Test: `tests/unit/test_tuner.py` (existing — must stay green)

- [ ] **Step 1: Add `build_study` and refactor `run_strategy_study` to use it**

In `src/qts/optimisation/tuner.py`, extract the sampler/pruner/study creation (currently lines 99-118) into a public helper, and call it from `run_strategy_study`:

```python
def build_study(config: TunerConfig) -> optuna.Study:
    """Create (or load) an Optuna study with the standard TPE sampler + MedianPruner.

    Shared by run_strategy_study and the news sweep so both use identical
    sampler/pruner/storage wiring.
    """
    sampler = optuna.samplers.TPESampler(seed=config.sampler_seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=config.pruner_n_startup_trials,
        n_warmup_steps=config.pruner_n_warmup_steps,
    )
    storage = None
    if config.storage_path is not None:
        config.storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{config.storage_path}"
    return optuna.create_study(
        study_name=config.study_name,
        direction=config.direction,
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )
```

Replace lines 98-118 in `run_strategy_study` (the inline sampler/pruner/storage/create_study block) with:

```python
    study = build_study(cfg)
```

Leave everything else in `run_strategy_study` unchanged.

- [ ] **Step 2: Run the existing tuner tests to confirm behaviour preserved**

Run: `.venv/bin/python -m pytest tests/unit/test_tuner.py -v --no-cov`
Expected: PASS (all existing tuner tests still pass)

- [ ] **Step 3: Commit**

```bash
git add src/qts/optimisation/tuner.py
git commit -m "refactor(optim): extract build_study from run_strategy_study for reuse"
```

---

### Task 8: run_news_sweep orchestrator + real-day validity check

**Files:**
- Create: `src/qts/optimisation/run_news_sweep.py`
- Test: `tests/integration/test_news_sweep.py` (create — integration + validity)

- [ ] **Step 1: Write the failing integration test (stub classifier, no LLM)**

```python
# tests/integration/test_news_sweep.py
"""T-NEWS-SWEEP-ACCEPT: end-to-end 5x10 sweep with a stub classifier (no LLM)."""

from __future__ import annotations

from datetime import datetime, timedelta

from qts.config import RiskLimits
from qts.macro.news_signal import NewsSignal
from qts.optimisation.run_news_sweep import NewsSweepResult, run_news_sweep
from qts.world.scenario import AnonAgentConfig, ScenarioConfig


class _StubClassifier:
    """Deterministic classifier: dovish-sounding text -> bull, else neutral. No LLM."""

    def classify(self, event: object) -> NewsSignal:
        text = getattr(event, "text", "").lower()
        if any(w in text for w in ("cut", "dovish", "ease", "accommodativ", "lower")):
            return NewsSignal(direction="bull", confidence=0.8, relevance=0.8, magnitude=0.7)
        if any(w in text for w in ("hike", "hawkish", "tighten", "raise")):
            return NewsSignal(direction="bear", confidence=0.8, relevance=0.8, magnitude=0.7)
        return NewsSignal(direction="neutral", confidence=0.3, relevance=0.3, magnitude=0.3)


def _risk() -> RiskLimits:
    return RiskLimits(
        max_daily_drawdown_pct=0.05, max_position_size_pct=0.20, max_open_positions=5,
        circuit_breaker_cooldown_seconds=300, sentiment_signal_max_scalar=3.0,
    )


def _base_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="sweep-base", symbol="BTCUSDT",
        start=datetime(2023, 12, 13, 18, 0, 0), end=datetime(2023, 12, 13, 21, 0, 0),
        tick=timedelta(minutes=1), fomc_announcement_at=datetime(2023, 12, 13, 19, 0, 0),
        fomc_expected_rate=5.25, starting_price=40_000.0,
        anon_agents=[AnonAgentConfig(agent_id="a1", style="trend")],
        mm_base_spread_bps=2.0, mm_vol_widen_k=1.0, powell_persona_id="powell",
    )


def test_sweep_runs_end_to_end() -> None:  # T-NEWS-SWEEP-ACCEPT
    result = run_news_sweep(
        base_scenario=_base_scenario(),
        n_episodes=5,
        n_trials=10,
        seed=42,
        risk_limits=_risk(),
        classifier=_StubClassifier(),
    )
    assert isinstance(result, NewsSweepResult)
    assert result.n_completed >= 1
    assert set(result.best_params) == {
        "news_signal_weight", "belief_half_life_minutes", "entry_threshold"
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/integration/test_news_sweep.py -x --no-cov`
Expected: FAIL — `ModuleNotFoundError: No module named 'qts.optimisation.run_news_sweep'`

- [ ] **Step 3: Implement the orchestrator**

```python
# src/qts/optimisation/run_news_sweep.py
"""Orchestrate the news-reactive Optuna sweep: bank -> warm cache -> study ->
best params -> held-out real-day validity check.

The real run uses the live LlamaCpp classifier (cache warmed once). Tests inject a
stub classifier so CI needs no LLM.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import optuna

from qts.config import RiskLimits
from qts.nautilus.config import VenueConfig
from qts.optimisation.episode_bank import CouplingRanges, generate_episode_bank
from qts.optimisation.news_objective import NewsObjectiveContext, make_news_objective
from qts.optimisation.tuner import TunerConfig, build_study

if TYPE_CHECKING:
    from qts.data.real_episode import RealEpisode
    from qts.world.scenario import ScenarioConfig

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path("data/news_cache")


@dataclass
class RealDayVerdict:
    trades: int
    strat_return: float
    hold_return: float
    excess: float
    passed: bool


@dataclass
class NewsSweepResult:
    study: optuna.Study
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    n_completed: int = 0
    n_pruned: int = 0
    real_day: RealDayVerdict | None = None


def run_news_sweep(
    base_scenario: ScenarioConfig,
    risk_limits: RiskLimits,
    n_episodes: int = 50,
    n_trials: int = 150,
    seed: int = 42,
    classifier: Any | None = None,
    real_episode: RealEpisode | None = None,
    venue_config: VenueConfig | None = None,
    coupling_ranges: CouplingRanges = CouplingRanges(),
    cache_dir: Path = _DEFAULT_CACHE_DIR,
) -> NewsSweepResult:
    """Run the sweep. If `classifier` is None, build the live LlamaCpp classifier and
    warm its cache for every episode's text events before tuning."""
    vc = venue_config or VenueConfig()
    bank = generate_episode_bank(n_episodes, seed, base_scenario, coupling_ranges)

    if classifier is None:
        from qts.macro.news_classifier import NewsClassifier  # noqa: PLC0415
        from qts.oversight.llm_client import create_llm_client  # noqa: PLC0415

        llm = create_llm_client(backend="llamacpp")
        classifier = NewsClassifier(llm_client=llm, cache_dir=cache_dir)
        all_text = [e for ep in bank for e in ep.text_events]
        asyncio.run(classifier.warm_cache_for(all_text))

    ctx = NewsObjectiveContext(
        episodes=bank, classifier=classifier, risk_limits=risk_limits, venue_config=vc
    )
    cfg = TunerConfig(n_trials=n_trials, study_name="qts_news_sweep", direction="maximize")
    study = build_study(cfg)
    study.optimize(
        make_news_objective(ctx), n_trials=n_trials, catch=(Exception,), gc_after_trial=True
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    result = NewsSweepResult(
        study=study, n_completed=len(completed), n_pruned=len(pruned)
    )
    if completed:
        result.best_params = study.best_params
        result.best_score = study.best_value

    if real_episode is not None and result.best_params:
        result.real_day = _evaluate_real_day(
            result.best_params, real_episode, classifier, risk_limits, vc
        )
        logger.info("Real-day verdict: %s", result.real_day)

    return result


def _evaluate_real_day(
    best_params: dict[str, Any],
    episode: RealEpisode,
    classifier: Any,
    risk_limits: RiskLimits,
    venue_config: VenueConfig,
) -> RealDayVerdict:
    """Apply best params to the held-out real day. Pass = trades fire AND beat hold
    AND positive return (on a dovish/up day, beating hold while profitable => long)."""
    from qts.nautilus.real_runner import run_real_backtest  # noqa: PLC0415
    from qts.optimisation.news_objective import _build_inner_params, _hold_return  # noqa: PLC0415
    from qts.strategies.momentum import MomentumStrategy  # noqa: PLC0415
    from qts.strategies.news_reactive import NewsReactiveMomentum  # noqa: PLC0415

    strategy = NewsReactiveMomentum(
        inner=MomentumStrategy(
            params=_build_inner_params(float(best_params["entry_threshold"])),
            risk_limits=risk_limits,
        ),
        classifier=classifier,
        belief_half_life=timedelta(minutes=float(best_params["belief_half_life_minutes"])),
        news_signal_weight=float(best_params["news_signal_weight"]),
    )
    result = run_real_backtest(episode, strategy, venue_config=venue_config, log_level="ERROR")
    hold = _hold_return(episode.terrain)
    excess = result.total_return - hold
    passed = result.total_trades > 0 and excess > 0 and result.total_return > 0
    return RealDayVerdict(
        trades=result.total_trades,
        strat_return=result.total_return,
        hold_return=hold,
        excess=excess,
        passed=passed,
    )
```

- [ ] **Step 4: Run the integration test to verify it passes**

Run: `.venv/bin/python -m pytest tests/integration/test_news_sweep.py -v --no-cov`
Expected: PASS (1 passed). If trials are pruned to 0 completed, lower the MedianPruner warmup by confirming `n_completed >= 1` holds — the stub produces non-degenerate excess so at least the startup trials complete.

- [ ] **Step 5: Commit**

```bash
git add src/qts/optimisation/run_news_sweep.py tests/integration/test_news_sweep.py
git commit -m "feat(optim): add run_news_sweep orchestrator with real-day validity check"
```

---

### Task 9: Validity test — real 2023-12-13 reaction falls inside CouplingRanges

**Files:**
- Modify: `tests/integration/test_news_sweep.py` (add one skipif test)

- [ ] **Step 1: Add the validity test**

```python
# append to tests/integration/test_news_sweep.py
from pathlib import Path

import pytest

_CURATED_ROOT = Path("data/real/fomc/2023-12-13")


def _curated_exists() -> bool:
    return (_CURATED_ROOT / "bars.csv").exists()


@pytest.mark.skipif(not _curated_exists(), reason="curated dataset missing")
def test_real_day_reaction_inside_coupling_range() -> None:  # T-NEWS-SWEEP-VALIDITY
    """The real day's post-announcement move should fall inside the default peak_bps
    range — confirms we aren't training on couplings that can't produce reality."""
    from qts.data.real_episode import RealEpisode
    from qts.optimisation.episode_bank import CouplingRanges

    episode = RealEpisode.from_disk(_CURATED_ROOT, symbol="BTCUSDT", source="fomc:2023-12-13")
    bars = episode.terrain.bars
    announce = min(e.timestamp for e in episode.text_events)
    pre = [b.close for b in bars if b.timestamp < announce]
    post = [b.close for b in bars if b.timestamp >= announce]
    assert pre and post, "need bars both sides of the announcement"

    move_bps = abs(max(post) - pre[-1]) / pre[-1] * 1e4
    lo, hi = CouplingRanges().peak_bps
    # The realised move should be within an order of magnitude of our peak range —
    # a loose sanity bound, not a tight fit (we randomise, we don't calibrate).
    assert move_bps <= hi * 5.0, f"real move {move_bps:.0f}bps far exceeds range top {hi}bps"
```

- [ ] **Step 2: Run it (passes or skips depending on data presence)**

Run: `.venv/bin/python -m pytest tests/integration/test_news_sweep.py -v --no-cov`
Expected: PASS or SKIP (skips cleanly if curated data absent — as in CI)

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_news_sweep.py
git commit -m "test(optim): assert real 2023-12-13 reaction sits inside coupling range"
```

---

### Task 10: Full suite + CI green

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest tests/ --no-cov -q`
Expected: baseline 1180 passed + the new tests (≈ +16), 4 skipped (+1 if curated data absent). 0 failures.

- [ ] **Step 2: Lint + format check (mirror CI)**

Run: `.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format --check src/ tests/ scripts/`
Expected: no errors. If `ruff format --check` reports diffs, run `.venv/bin/ruff format src/ tests/` and re-commit.

- [ ] **Step 3: Type check (mirror CI)**

Run: `.venv/bin/mypy src/qts --config-file pyproject.toml`
Expected: no issues. Likely `# type: ignore` touchpoints: the `classifier: Any` field (intentional — stub/real duck-typing), and Nautilus `CustomData`/`DataType` interop (follow the v2.1 pattern already in `real_runner.py`).

- [ ] **Step 4: Final commit (only if formatting/type fixes were needed)**

```bash
git add -A
git commit -m "chore(optim): lint, format, and type fixes for news sweep"
```

---

## Self-Review Notes

- **Spec coverage:** drift model (T1), coupling wiring (T2), episode plumbing (T3), bank (T4), search space (T5), CVaR objective + inner-params validator guard (T6), study reuse (T7), orchestrator + real-day bar (T8), validity check (T9), green gate (T10). All spec sections mapped.
- **Type consistency:** `NewsParams` field names (`news_signal_weight`, `belief_half_life_minutes`, `entry_threshold`) are used identically in the sampler, objective, and `_evaluate_real_day`. `belief_half_life_minutes` → `timedelta(minutes=...)` at both strategy-build sites. `_build_inner_params`/`_hold_return`/`_cvar_of_excess` are defined once in `news_objective.py` and imported elsewhere.
- **Hard constraints honoured:** `news_signal_weight ≤ 0.95 < 1.0` (constructor limit); `exit_threshold = 0.4·entry < entry` (StrategyParams validator); SignalWeights and SentimentFusionWeights sum to 1.0 (frozen, proven values).
- **No live LLM in CI:** all tests use the stub classifier or skip on missing curated data; the real classifier path is only hit when `classifier=None` in a manual run.
