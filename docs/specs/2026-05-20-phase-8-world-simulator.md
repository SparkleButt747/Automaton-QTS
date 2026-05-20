# Phase 8 — World Simulator (v1)

**Status**: spec draft
**Date**: 2026-05-20
**Predecessor**: `docs/plans/terrain-refactor-plan.md` (Phase 8 entry, was deferred)
**Companion log**: `.grill/phase-8-world-simulator.md`

## Why

The trading strategy reacts to news + world events + watched accounts (Fed, US senators, CEOs, anon trader sentiment), not just price/volume. Before pointing it at real flows via Binance testnet + scrapling, it needs an internal **world simulator** to develop and stress-test against. This is the deferred Phase 8.

Same swappable boundary: sim emits the same shape of events scrapling will emit live, so flipping the source is a config change.

## What v1 ships

A vertical slice: one regime, one scenario, the minimum agent roster to produce realistic order flow, integrated end-to-end.

### Concrete scenario

- **Asset**: BTCUSDT on a `BINANCE` venue (Nautilus matching engine, `CASH` multi-currency account — reuses the post-#2 fix from the prior session).
- **Episode**: 24h trading day. FOMC announcement at ~14:00 UTC (mid-day). 1-minute bars → 1440 bars.
- **Information events**:
  - 1 scheduled FOMC release (rate decision + statement) with a `surprise` axis (hawkish/dovish/neutral, sampled per episode).
  - Powell tweets/Q&A statements during a 1h window after the decision (5-10 events).
  - Anon retail posts throughout the day.

### Agent roster

| Agent | Count | Behaviour |
|---|---|---|
| Scheduler | 1 | Clock-driven; emits the FOMC event at the scheduled tick + triggers Powell's post-event statements |
| Powell persona | 1 | Local LLM (Qwen 3.6); cached by `(regime, event, surprise_bucket)` tuple |
| Anon retail | 3 (configurable) | VADER + keyword regex on Powell's text + price action. Defaults: sentiment-chaser, trend-follower, mean-reverter |
| Market maker | 1 | Inventory-aware naive quoter. Quotes around `last + sentiment_drift`, width = `base_spread + k × recent_realised_vol`, leans quotes when inventory one-sided |
| Strategy under test | 1 | Existing QTSStrategy actor. v1 uses `MomentumStrategy`; later strategies opt into the new `on_text` callback |

### LLM mode (two)

1. **Corpus mode** (default for v1): pre-generated persona-reaction corpus on disk. Sim samples by `(persona, event, regime, surprise_bucket)` key. Zero LLM at runtime.
2. **Live cache mode**: Qwen called for any cache miss; result cached. Same key structure. Used when the corpus is being grown or for high-fidelity one-off scenarios.

Cache directory: `data/world/persona_corpus/`.

## Architecture

### Package layout

```
src/qts/world/
├── __init__.py
├── scenario.py        # Scenario, ScenarioConfig — declarative description of an episode
├── episode.py         # SimulatedEpisode (wraps MarketTerrain + agent metadata)
├── runner.py          # run_simulation(scenario, strategy, seed) -> SimulatedEpisode
├── clock.py           # Simulated clock + tick scheduling
├── corpus.py          # PersonaCorpus: load/save/sample/generate the LLM cache
├── agents/
│   ├── __init__.py
│   ├── base.py        # WorldAgent protocol — on_tick, on_event, on_fill
│   ├── persona.py     # PersonaAgent (Powell)
│   ├── anon.py        # AnonRetailAgent (configurable style)
│   ├── market_maker.py # InventoryAwareMM
│   └── scheduler.py   # SchedulerAgent — fires calendar events
├── events.py          # TextEvent, MacroEvent, EventBus
└── sentiment.py       # VADER + keyword regex helpers for anon decoding
```

### Data model

```python
# qts/world/events.py
@dataclass(frozen=True)
class TextEvent:
    timestamp: datetime
    source: str          # "powell", "anon_42", "fed_press_release"
    persona: str | None  # name if attributed
    text: str
    metadata: dict[str, Any]   # surprise_bucket, regime tags, etc.

@dataclass(frozen=True)
class MacroEvent:
    timestamp: datetime
    kind: str           # "FOMC", "CPI", ...
    expected: float
    actual: float
    surprise: float     # actual - expected, normalised
    text_event: TextEvent | None   # accompanying release text

# qts/world/episode.py
@dataclass
class SimulatedEpisode:
    terrain: MarketTerrain          # the existing type; bars + event_calendar populated
    scenario_name: str
    seed: int
    agent_trace: AgentTrace         # per-agent state timeline (orders, positions, sentiment readings)
    llm_corpus_refs: list[str]      # which corpus keys were consumed
    order_log: list[OrderLogEntry]  # every submitted order + fate
```

### Runtime

```python
def run_simulation(
    scenario: ScenarioConfig,
    strategy: Strategy,
    seed: int,
    *,
    llm_mode: Literal["corpus", "live_cached"] = "corpus",
    persona_corpus_path: Path | None = None,
) -> SimulatedEpisode:
    """
    1. Build a Nautilus BacktestEngine with multi-currency CASH account on BINANCE.
    2. Construct agents from scenario.agent_roster (scheduler, persona, anons, MM).
    3. Attach the strategy-under-test (existing QTSStrategy actor) as another agent.
    4. Step the simulated clock; agents react to events + bars on their schedules.
    5. MM quotes -> matching engine -> emitted bars feed back to all agents.
    6. Strategy.on_text() forwards persona/event text to news-aware strategies.
    7. On completion, collect bars + events into a MarketTerrain; bundle metadata into SimulatedEpisode.
    """
```

### Strategy protocol extension

```python
# qts/strategies/base.py
class Strategy(Protocol):
    def on_bar(self, bar: Bar, snapshot: SignalSnapshot, positions: list[Position]) -> list[Order]: ...
    def on_fill(self, fill: Fill) -> None: ...
    def on_text(self, event: TextEvent) -> None: ...   # NEW, optional via default no-op
```

Default implementation in `BaseStrategy.on_text` is a no-op. Existing strategies (`MomentumStrategy`, `MeanReversionStrategy`) inherit unchanged.

QTSStrategy (the Nautilus actor) subscribes to the world's text-event bus and forwards each TextEvent to `inner_strategy.on_text(event)`.

### Determinism

- One master `seed` per episode. All agent RNGs, scenario sampling, and Nautilus's fill model derive from it.
- LLM cache keys are deterministic functions of (persona, event_kind, regime, surprise_bucket). Corpus mode is fully reproducible; live-cached mode is reproducible after the first miss.
- Reproducibility check is in the v1 acceptance suite: two runs with identical seed must produce identical bars, orders, and PnL.

## Acceptance criteria for v1

1. `pytest tests/unit/test_world_runner.py` passes a round-trip test:
   - Build a `ScenarioConfig` for "FOMC, BTCUSDT, hawkish surprise".
   - Run it twice with `seed=42`.
   - Assert: identical bar count, identical close-price series, identical strategy PnL.
2. The same scenario with a different seed produces a **different** but valid episode (bars exist, all agents emitted, no Nautilus halt).
3. `SimulatedEpisode.terrain` is a valid `MarketTerrain` and can be passed to `run_terrain_backtest()` for a second strategy run.
4. `agent_trace` and `order_log` are non-empty and JSON-serialisable.
5. `SimulatedEpisode.terrain.event_calendar` contains the FOMC `MacroEvent` and each `PersonaAgent` emission.
6. Strategies that don't implement `on_text` run successfully (the default no-op kicks in).

Out of scope for v1 acceptance: realism of price moves, statistical match against real FOMC days, multi-day dynamics, Optuna integration.

## Plan for scaling beyond v1

Per the grill, this is explicitly v1. The scaling path (so we don't lose it):

- **v1.5**: empirical calibration. Collect real Powell/FOMC reactions on BTC; fit agent params so sim distributions match real (vol, spread, max move). Validation suite plots sim vs real distributions.
- **v2**: more scenarios (CPI, NFP, geopolitical, USDT depeg, regulatory); more personas (Trump, Musk, CEOs, congressional); larger anon pool (10-100 configurable); 2-3 competing MMs.
- **v3**: news-reactive strategies via `on_text`; Optuna sweeps over `(scenario, agent_roster)` space.
- **v4**: multimodal via Gemma (press conference video/audio); Avellaneda-Stoikov MMs; multi-asset cross-correlation.
- **v5**: RL agents trained on real data deployed as adversaries (research-grade, deferred).

## Out of scope (explicit)

- Multimodal events (deferred to v4).
- Assets other than BTCUSDT (v2+).
- Empirical calibration against real history (v1.5).
- Avellaneda-Stoikov MM (v2+).
- RL-based agents (deferred indefinitely).
- Live deployment / scrapling wiring (separate project; this sim's whole purpose is the staging ground).

## Risks

- **Reproducibility under Nautilus's event loop**: agents emit orders into a shared engine; non-determinism could leak through internal Nautilus threading or HashMap iteration. Plan: verify with the acceptance test #1 above; if it fails, identify and seed the source.
- **Persona realism**: hand-tuned v1 may produce off-distribution price reactions that teach the strategy bad lessons. Mitigation: corpus mode lets us hand-inspect generated text before running; v2 calibration is the real fix.
- **LLM corpus bloat**: keys grow as new persona prompts / regimes are added. Plan: corpus is content-addressed; old entries stay valid until prompt or regime taxonomy changes; ship a small invalidation tool.
- **Anon decode gap holds up only if anons stay dumber than the strategy**: the moment a user wires a real LLM into AnonRetailAgent, the alpha collapses. Guardrail: AnonRetailAgent uses VADER explicitly; opting into LLM-decoded anons is a v3 thing requiring a deliberate constructor change.

## File-level impact estimate

- New: ~15 files under `src/qts/world/`, ~1500-2000 LOC total
- Modified: `src/qts/strategies/base.py` (one new method on Strategy protocol with default no-op)
- Modified: `src/qts/nautilus/actor.py` (forward text events to inner strategy)
- New tests: `tests/unit/test_world_*.py`, ~8-10 files, ~800 LOC

Total: ~2-3 weeks of focused work for v1. Mid-effort, contained blast radius.
