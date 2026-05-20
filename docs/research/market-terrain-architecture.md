# Market Terrain Architecture — Research Notes

**Date**: April 2026
**Context**: Exploring how to redesign automaton-qts using architectural patterns from the racing-lab-tyd project, informed by the MarS paper and MiroFish assessment.

---

## 1. The Core Insight: Racing Controllers and Trading Strategies Are the Same Thing

The racing-lab-tyd project and automaton-qts share an identical architectural skeleton. The domain changes (curvature to volatility, steering to position sizing, lap time to Sharpe ratio), but the structure — environment generation, simulation loop, controller interface, safety layer, scoring, optimisation — maps directly.

### 1.1 Component Mapping

| Racing Lab | Automaton QTS | Shared Concept |
|---|---|---|
| **Track** (centerline, curvature, width) | **Market** (price history, bars, order book) | The **environment** being navigated |
| **Car Physics** (bicycle model, 6-state ODE) | **NautilusTrader Engine** (exchange sim, LOB matching, fills) | The **plant** — how actions change state |
| `simulate_episode()` | `BacktestNode.run()` / live `TradingNode.run()` | The **simulation loop** |
| **Controller** (RRHC / MPCC) | **QTSStrategy Actor** wrapping `MomentumStrategy` etc. | The **controller** — decisions each step |
| `controller.act(t, state, track)` -> `[delta, Fx]` | `strategy.on_bar(bar, snapshot, positions)` -> `[Order]` | The **control law** |
| `PlantState [x, y, psi, vx, vy, r]` | `SignalSnapshot + Nautilus Portfolio State` | **Observable state** |
| `ControllerOutput [delta, Fx]` (steer, throttle) | `Order (side, quantity, type, price)` (direction, size, execution) | **Action space** |
| Track curvature, width, tangents | RSI, MACD, BB, ATR, momentum, sentiment | **Feature extraction** |
| **Racing line weight** (where to position the car) | **Combined alpha** (which direction to trade) | The **guidance signal** |
| Track limits, spin detection | Nautilus pre-trade risk + drawdown limits + circuit breaker | **Safety constraints** |
| `LapTimeStrategy.score()` | Sharpe ratio, total P&L, max drawdown | **Objective function** |
| Optuna TPE (offline parameter tuning) | LLM proposals + human approval | **Hyperparameter optimisation** |
| `FairnessConfig` | `BacktestRunConfig` + `MarketTerrain` | **Experiment contract** |

### 1.2 The Control Loop Analogy

In racing, the RRHC controller does this every 0.02s:

```
state [x, y, psi, vx, vy, r]  ->  project onto track  ->  racing line weight  ->  [delta, Fx]
         current car pose           where am I?            where should I be?      steer + throttle
```

The trading strategy does the same thing every bar:

```
state [bars, positions, pnl]  ->  compute signals  ->  combined alpha  ->  [Order]
       current market pose        where am I?          where should I be?    direction + size
```

Both are reactive controllers navigating a 1D path through time.

---

## 2. Gaps in Current Automaton Architecture

Compared to the racing lab's clean separation, automaton has three structural gaps:

### 2.1 The "Track" is Underspecified

In racing, `Track` is a first-class frozen dataclass with geometry, curvature, annotations, and query methods. It is generated *before* the simulation runs.

In automaton, "the market" is just a stream of bars. There is no equivalent `Market` object that pre-computes the terrain. We need a `MarketTerrain` — the trading equivalent of `Track` + `TrackAnnotations`.

### 2.2 The Simulation Loop Isn't Separated from Execution

In racing, `simulate_episode()` is a pure loop: step physics, step controller, check events. It doesn't care if it's headless evaluation or live HUD.

Automaton's `ExecutionEngine` mixes the simulation concern (stepping through bars) with execution concerns (paper fills, commissions). **Resolution: NautilusTrader.** Nautilus provides backtest-live equivalence out of the box — the same strategy actor code runs against a `BacktestNode` (simulated exchange) or a `TradingNode` (live venue). The event loop, fill simulation, and portfolio accounting are all handled by Nautilus's Rust core. Our custom `BacktestEngine` and `ExecutionEngine` become legacy code to retire.

### 2.3 The Optimisation Loop is Missing

Racing lab's Optuna loop tunes controller parameters across 5 different track geometries, preventing overfitting to one circuit. Automaton has no equivalent. We need parameter optimisation across diverse market regimes.

---

## 3. Proposed Three-Layer Architecture

```
RACING LAB                         AUTOMATON (target)
────────────────────────────────────────────────────────
Track Generator                    Macro Engine
  (Fourier paths, curvature,         (earnings, news, bank statements,
   corner placement)                   macro indicators, NLP)
       |                                   |
       | Track geometry                    | Regime + scenario conditioning
       v                                   v
Car Physics / Plant                NautilusTrader Engine
  (bicycle model ODE,                (exchange simulation, LOB matching,
   tire forces, friction)              realistic fills, multi-venue)
       |                                   |
       | state each dt                     | events (bars, fills, positions)
       v                                   v
Controller (RRHC/MPCC)             Strategy (NautilusTrader Actor)
  .act(state, track) -> [d, Fx]      .on_bar() -> submit_order()
```

> **Design decision (April 2026):** NautilusTrader is the execution/simulation
> engine from day one. The rationale is that if this system graduates to a
> product or fund, strategies must never be developed against a toy fill model
> and then "ported" to production execution. The car physics must be
> production-grade throughout — strategies that work in dev work in prod.
> NautilusTrader gives us exchange-grade matching, realistic queue-position
> fills, multi-venue routing, and a zero-cost backtest-to-live path (same
> strategy code, swap the venue config).

### 3.1 Layer 1: Macro Engine (the "Track Generator")

Decides *what kind of market environment* we are in. Ingests fundamentals, news, and macro data to produce conditioning signals.

```
                 +----------------------------+
                 |       MACRO ENGINE          |
                 +-------------+--------------+
                               |
            +------------------+------------------+
            v                  v                   v
    +---------------+  +---------------+   +------------------+
    | Fundamentals  |  | News / NLP    |   | Macro Indicators |
    |               |  |               |   |                  |
    | - Earnings    |  | - FinBERT     |   | - Interest rates |
    | - Balance     |  | - Headlines   |   | - CPI / PPI      |
    |   sheets      |  | - SEC filings |   | - Unemployment   |
    | - Cash flow   |  | - Analyst     |   | - GDP            |
    | - Revenue     |  |   reports     |   | - Yield curve    |
    |   guidance    |  | - Social      |   | - VIX            |
    | - Debt ratios |  |   sentiment   |   | - Credit spreads |
    +-------+------+  +-------+-------+   +--------+---------+
            |                  |                     |
            v                  v                     v
    +-----------------------------------------------------+
    |              Regime Classifier                        |
    |                                                       |
    |  Outputs MacroRegime:                                |
    |    - trend: BULL / BEAR / SIDEWAYS                   |
    |    - volatility: HIGH / LOW / TRANSITIONING          |
    |    - liquidity: ABUNDANT / TIGHT / CRISIS            |
    |    - sentiment: EUPHORIC / FEARFUL / NEUTRAL         |
    |    - catalyst: EARNINGS / MACRO_EVENT / NONE         |
    |                                                       |
    |  + scenario_description: str                         |
    |    "Post-earnings beat, sector rotation into          |
    |     tech, rising yields creating headwind"            |
    +---------------------------+--------------------------+
                                |
                                v
                        Market Simulator
```

### 3.2 Layer 2: NautilusTrader Engine (the "Car Physics")

[NautilusTrader](https://nautilustrader.io/) is the execution and simulation engine — the equivalent of the bicycle model ODE in the racing lab. It owns:

- **Order matching**: exchange-grade FIFO matching with queue-position modelling
- **Fill simulation**: probabilistic fills on limits/stops, latency modelling, realistic slippage
- **Portfolio state**: positions, P&L, margin, multi-currency accounting
- **Risk at execution level**: pre-trade risk checks, order throttling
- **Data ingestion**: bar, tick, quote, and L2 order book data
- **Venue abstraction**: same code runs against simulated exchange (backtest) or live venue (Binance, IB)

NautilusTrader replaces both our custom `BacktestEngine` and `ExecutionEngine`. The critical property is **backtest-live equivalence**: the strategy actor, its event handlers, and its order submission code are identical in both modes. The only thing that changes is the venue configuration.

```
                    +─────────────────────────────────+
                    │      NautilusTrader Node         │
                    │                                  │
                    │  ┌──────────┐  ┌──────────────┐ │
  MarketTerrain ──► │  │ Data     │  │ Execution    │ │
  (bars, events)    │  │ Engine   │──│ Engine       │ │
                    │  └────┬─────┘  └──────┬───────┘ │
                    │       │               │         │
                    │       ▼               ▼         │
                    │  ┌──────────────────────────┐   │
                    │  │  Strategy Actor           │   │
                    │  │  (on_bar, on_data,        │   │
                    │  │   on_event, on_fill)      │   │
                    │  └──────────────────────────┘   │
                    │                                  │
                    │  Venue: SimulatedExchange (test) │
                    │     or: Binance / IB (live)      │
                    +──────────────────────────────────+
```

**What this means for our architecture:**

| Concern | Before (custom engine) | After (NautilusTrader) |
|---|---|---|
| Fill model | Flat slippage % at bar close | Queue-position, latency, probabilistic limit fills |
| Order types | Market only | Market, limit, stop, stop-limit, bracket, trailing, IOC, FOK |
| Multi-asset | Single symbol per run | Multi-venue, multi-instrument, cross-margin |
| Live trading | Separate `ExecutionEngine` (paper only) | Same strategy, swap venue config |
| Performance | Pure Python loop | Rust core + Cython — orders of magnitude faster |
| Portfolio | Manual cash + position tracking | Built-in accounting, margin, multi-currency |

### 3.3 Layer 3: Strategy Actor (the "Controller")

Strategies are NautilusTrader actors that receive events from the engine. Our QTS signal pipeline integrates as a custom data source — the actor receives bar events, computes signals via our pipeline, and submits orders through Nautilus's order management.

```python
from nautilus_trader.trading.strategy import Strategy as NtStrategy
from nautilus_trader.model.data import Bar as NtBar

class QTSStrategy(NtStrategy):
    """Bridge: NautilusTrader actor wrapping our signal + decision logic."""

    def __init__(self, config: QTSStrategyConfig) -> None:
        super().__init__(config)
        self._signal_pipeline = SignalPipeline(symbol=config.symbol)
        self._inner_strategy = config.qts_strategy  # our Strategy protocol impl
        self._bar_buffer: list[Bar] = []

    def on_bar(self, bar: NtBar) -> None:
        """Event handler: new bar received from Nautilus data engine."""
        qts_bar = nautilus_bar_to_qts(bar)
        self._bar_buffer.append(qts_bar)

        # Compute signals via our pipeline
        snapshot = self._signal_pipeline.compute(self._bar_buffer[-50:])
        if snapshot is None:
            return

        # Get positions from Nautilus portfolio (single source of truth)
        positions = self._get_qts_positions()

        # Decision logic stays in our Strategy protocol
        orders = self._inner_strategy.on_bar(qts_bar, snapshot, positions)

        # Submit through Nautilus order management
        for order in orders:
            nt_order = qts_order_to_nautilus(order, self.instrument_id)
            self.submit_order(nt_order)

    def on_order_filled(self, event: OrderFilled) -> None:
        """Nautilus fill event — forward to our strategy for bookkeeping."""
        fill = nautilus_fill_to_qts(event)
        self._inner_strategy.on_fill(fill)
```

This preserves our `Strategy` protocol as the decision interface — strategies don't need to know about NautilusTrader. The `QTSStrategy` actor is the adapter between worlds. This means:

1. **Strategy authors write pure decision logic** against `on_bar(bar, snapshot, positions) -> list[Order]`
2. **The adapter handles Nautilus plumbing** — data conversion, order submission, position queries
3. **Testing strategies in isolation** is still easy — mock the adapter, feed bars directly
4. **The same strategy code runs in backtest and live** — the adapter is the same, only the venue changes

---

## 4. Proposed Data Models

### 4.1 MacroRegime

```python
@dataclass(frozen=True)
class MacroRegime:
    """The 'track geometry' — what kind of market are we in?"""
    trend: Trend                    # like overall track direction
    volatility: VolLevel            # like track width (narrow = high vol)
    liquidity: LiquidityLevel       # like track grip
    sentiment: SentimentLevel       # like track surface condition
    catalyst: Catalyst | None       # like a chicane — event-driven

    # Conditioning signals for the market simulator
    expected_drift: float           # annualised, like track gradient
    expected_vol: float             # like curvature intensity
    correlation_regime: float       # cross-asset

    # Natural language description
    scenario_description: str
```

### 4.2 MarketTerrain

```python
@dataclass(frozen=True)
class MarketTerrain:
    """The complete 'track' — pre-computed environment for a strategy to navigate."""
    regime: MacroRegime
    bars: Sequence[Bar]                      # generated or historical price data
    volatility_surface: Sequence[float]      # like curvature profile
    support_resistance: Sequence[Level]      # like track boundaries
    liquidity_profile: Sequence[float]       # like grip at each point
    event_calendar: Sequence[MarketEvent]    # like corner annotations

    # Query methods (like Track.get_curvature_at(s))
    def regime_at(self, t: datetime) -> MacroRegime: ...
    def liquidity_at(self, t: datetime) -> float: ...
    def nearest_event(self, t: datetime) -> MarketEvent | None: ...
```

### 4.3 Simulation via NautilusTrader BacktestNode

The simulation loop is no longer a hand-rolled `for bar in bars` — it's a NautilusTrader `BacktestNode` configured with our terrain data and strategy actor.

```python
from nautilus_trader.backtest.node import BacktestNode, BacktestRunConfig
from nautilus_trader.config import BacktestVenueConfig, BacktestDataConfig

def run_terrain_backtest(
    terrain: MarketTerrain,
    strategy_config: QTSStrategyConfig,
    venue: str = "BINANCE",
) -> BacktestResult:
    """Run a strategy against a MarketTerrain via NautilusTrader.

    This is the equivalent of simulate_episode() in the racing lab.
    The terrain provides the data; Nautilus provides the physics.
    """
    # 1. Configure venue (the "physics engine")
    venue_config = BacktestVenueConfig(
        name=venue,
        oms_type="NETTING",
        account_type="MARGIN",
        base_currency="USD",
        starting_balances=["100_000 USD"],
        fill_model=RealisticFillModel(),  # queue-position, latency
    )

    # 2. Load terrain bars as Nautilus data
    data_config = BacktestDataConfig(
        catalog_path=terrain.catalog_path,  # persisted Parquet catalog
        data_cls="Bar",
        instrument_id=f"{terrain.symbol}.{venue}",
    )

    # 3. Configure the run
    run_config = BacktestRunConfig(
        engine=BacktestEngineConfig(logging_level="WARNING"),
        venues=[venue_config],
        data=[data_config],
        strategies=[strategy_config],
    )

    # 4. Run — Nautilus handles the event loop
    node = BacktestNode(configs=[run_config])
    results = node.run()

    # 5. Extract results into our domain
    return extract_backtest_result(results, terrain)
```

**Key difference from the old pure loop:** We don't manually step through bars, simulate fills, or track portfolio state. NautilusTrader's event engine does all of this with production-grade fidelity. Our code configures the *what* (terrain data, strategy, venue) and Nautilus handles the *how* (matching, fills, accounting).

---

## 5. Three Approaches to "Generating the Real World"

### 5.1 Approach A: Historical Replay + Macro Annotation

Use real market data but annotate it with the Macro Engine — loading a real-world circuit into the racing sim rather than generating one procedurally.

```
Real historical bars (2020-2024)
        |
        v
Macro Engine annotates each period:
  - Jan 2020: LOW_VOL, BULL, ABUNDANT liquidity
  - Mar 2020: CRISIS, BEAR, vol spike, COVID catalyst
  - Nov 2020: RECOVERY, BULL, stimulus catalyst
  - 2022: TIGHTENING, BEAR, rate hikes
        |
        v
MarketTerrain objects per period
  (each one is a "track" the strategy must navigate)
        |
        v
Train strategy across ALL terrains
  (like training RRHC on 5 different track geometries)
```

**Pros**: Real data, no generation artefacts, ground truth exists.
**Cons**: Limited to regimes we have seen. Can't test tail risk scenarios.

### 5.2 Approach B: Synthetic Generation (MarS-style)

Use the Macro Engine to condition a generative model that produces synthetic but realistic market data.

```
Macro Engine generates scenario:
  "Earnings miss + rising yields + sector rotation"
        |
        v
MarS-style generator (conditioned on scenario)
  -> Produces realistic order flow / price trajectories
        |
        v
MarketTerrain (synthetic but realistic)
```

**Pros**: Unlimited scenarios, test regimes that haven't happened yet.
**Cons**: Validation is hard. MarS is a massive project (16B tokens, Microsoft Research scale).

### 5.3 Approach C: Hybrid (Recommended)

```
TRAINING SET:
+-- Historical terrains (Approach A) — 20-30 annotated real periods
|     covers regimes you've actually seen
|
+-- Synthetic terrains (Approach B) — unlimited generated scenarios
|     covers regimes you HAVEN'T seen (tail risk, black swans)
|
+-- Perturbation terrains — like P4a-P4e in racing lab
      take a real period, inject: liquidity shock, flash crash,
      correlation breakdown, news front-run
```

Directly analogous to how racing lab uses 5 procedural train tracks + 10 unseen test tracks + perturbation sweeps.

---

## 6. The Optimisation Loop

Direct translation of racing lab's Optuna pipeline:

```python
for trial in range(N):
    params = sampler.sample(strategy_search_space)
    scores = []
    for terrain in train_terrains:       # different date ranges / regimes
        config = QTSStrategyConfig(
            qts_strategy=MomentumStrategy(params),
            symbol=terrain.symbol,
        )
        result = run_terrain_backtest(terrain, config)  # Nautilus BacktestNode
        scores.append(result.sharpe_ratio)
    objective = mean(scores)             # must work in ALL regimes
    sampler.update(params, objective)
```

Train on multiple market regimes (bull, bear, sideways, high-vol, low-vol) the same way racing trains on multiple track geometries. This prevents overfitting to one market condition. NautilusTrader's Rust core makes each trial fast enough for Optuna's iterative sampling — hundreds of backtests per hour across diverse terrains.

---

## 7. MarS Paper — Key Findings

**Paper**: "MarS: a Financial Market Simulation Engine Powered by Generative Foundation Model" (Li et al., Microsoft Research Asia, 2024). [arXiv:2409.07486v2](https://arxiv.org/html/2409.07486v2)

### 7.1 What MarS Does

MarS is an order-level market microstructure simulator using a LLaMA2-based transformer (the "Large Market Model") trained on 16 billion order tokens from 500 top-liquidity Chinese stocks (2017-2023).

It generates realistic order-by-order trading activity — think of it as the **car physics engine** (how prices move tick-by-tick), not the track generator (what macro context drives them).

### 7.2 Architecture

| Component | Function |
|---|---|
| **Order Sequence Model** | Causal transformer predicting next individual order from recent orders + LOB state |
| **Order-Batch Model** | Auto-regressive transformer + VQ-VAE generating next minute's order distribution |
| **Ensemble Model** | Balances immediate market impact with control signal matching |
| **Signal Interface** | Maps natural language scenarios to numerical control signals |
| **Simulated Clearing House** | Real-time order matching, LOB updates, closed-loop dynamics |

### 7.3 Order Tokenisation

```
Emb_i = emb(order_i) + linear_proj(LOB_i^volumes) + emb(LOB_i^mid_price)
```

- Order type: {Ask, Bid, Cancel}
- Price: discretised to [0, 32) relative to mid-price
- Volume: discretised to [0, 32)
- Interval: discretised to [0, 16) seconds between orders
- LOB: 10-level bid/ask volumes + mid-price (ticks since market open)

### 7.4 Key Results

- Reproduces Cont's 11 stylized facts of financial markets (aggregational Gaussianity, absence of autocorrelations, volatility clustering)
- Market impact follows Square-Root-Law: `delta ~ sigma * sqrt(Q/V)`
- Scaling laws validated: performance improves with model size (0.22B to 1.02B params) and data
- Supports interactive simulation — user-injected orders create realistic market impact responses

### 7.5 Discovered Market Impact Factors (via Symbolic Regression)

Beyond the classic Square-Root-Law, MarS discovered three additional factors:
1. **Resiliency**: Market's ability to recover after trades
2. **LOB pressure**: Bid/ask volume imbalance
3. **LOB depth**: Available liquidity at multiple price levels

### 7.6 Long-Term Market Impact ODE

```
dY(t)/dt = sum_i sum_j W_{i,j} * X_i * F_j^decay(t)
```

Where X includes (volume, price, resiliency, LOB_pressure, LOB_depth) and F^decay are decay functions [1/t, ..., 1/sqrt(t)].

### 7.7 Limitations

- **No macro context**: Purely microstructure. No news, earnings, fundamentals, or macro indicators.
- **Chinese market only**: Trained on Chinese stock data.
- **Massive scale**: 16B tokens, up to 3B parameters — not something we can replicate.
- **No open weights**: Code at github.com/microsoft/MarS but model weights are not public.

### 7.8 Relevance to Automaton

MarS validates the idea that generative models can produce realistic market dynamics. Its Signal Interface (natural language -> control signals -> conditioned generation) is the bridge where our Macro Engine would connect. However, building a full MarS-scale model is out of scope. The architecture pattern is what matters — not the specific implementation.

---

## 8. MiroFish Assessment

**Repository**: [github.com/666ghj/MiroFish](https://github.com/666ghj/MiroFish)

### 8.1 What MiroFish Actually Is

MiroFish is a **social media simulation engine**, not a financial simulator. Built on OASIS (CAMEL-AI), it simulates thousands of LLM-powered agents interacting on Twitter and Reddit.

Pipeline:
1. Seed documents (news, reports) -> Zep GraphRAG (knowledge graph)
2. Agent Persona Generator (LLM creates personalities with memory + behavioural logic)
3. OASIS Social Simulation (agents post, reply, retweet on simulated Twitter/Reddit)
4. ReportAgent summarises emergent outcomes

### 8.2 Tech Stack

- Python 3.11-3.12 backend (Flask)
- Vue.js frontend
- Zep Cloud for agent memory / knowledge graphs
- OpenAI-compatible LLM API (recommends Alibaba Qwen-plus)
- CAMEL-AI OASIS for social simulation

### 8.3 Where It Could Fit

The only plausible integration point is **synthetic sentiment generation** within the Macro Engine:

```
Event: "Fed raises rates 50bp"
        |
        v
MiroFish simulates social media response
  -> thousands of agent personas react
  -> sentiment trajectory over simulated hours/days
        |
        v
SentimentFusion receives synthetic social sentiment
```

### 8.4 Why It's Not Worth Integrating

| Factor | Assessment |
|--------|-----------|
| **Complexity** | Massive — Zep Cloud, LLM costs for thousands of agents, OASIS framework |
| **Cost** | README warns about high token consumption; recommends <40 simulation rounds |
| **Financial validation** | Zero. "Financial Prediction" listed as *coming soon* |
| **Value over current system** | Simulating what people *would* say vs measuring what they *did* say |
| **License** | AGPL-3.0 — viral copyleft, any derivative must also be AGPL |
| **Verdict** | Cool research project, wrong problem domain |

---

## 9. Recommended Implementation Path

Priority-ordered steps. NautilusTrader is the execution foundation from Phase 1 — everything builds on it.

### Phase 1: NautilusTrader as Execution Foundation

- Set up NautilusTrader `BacktestNode` with a simulated exchange venue
- Build the `QTSStrategy` actor adapter (Section 3.3) that bridges our `Strategy` protocol to Nautilus's actor model
- Port data ingestion: load historical bars into a Nautilus `ParquetDataCatalog`
- Wire up conversion helpers: `qts_bar_to_nautilus`, `nautilus_fill_to_qts`, `qts_order_to_nautilus` (some already exist in `nautilus_adapter.py`)
- Validate: run existing `MomentumStrategy` through Nautilus and compare results against old `BacktestEngine` to quantify fill model differences
- Retire custom `BacktestEngine` and `ExecutionEngine` once Nautilus path is validated

### Phase 2: MarketTerrain as First-Class Object

- Define `MacroRegime` and `MarketTerrain` frozen dataclasses
- Add query methods (like `Track.get_curvature_at(s)`)
- Wrap existing historical bar data into `MarketTerrain` objects
- Each terrain maps to a Nautilus `BacktestDataConfig` — the terrain provides the *what*, Nautilus provides the *how*

### Phase 3: Historical Regime Annotation

- Label past market periods with macro context (manually + LLM-assisted)
- Build a library of 20-30 annotated `MarketTerrain` instances covering bull, bear, crisis, sideways, high-vol, low-vol regimes
- Persist terrains as Parquet catalogs that Nautilus can ingest directly
- This is the equivalent of the 5 procedural train tracks

### Phase 4: Macro Engine (NLP + Fundamentals)

- Ingest earnings data, SEC filings, macro indicators
- FinBERT / LLM classify regime from text
- Produce `MacroRegime` objects automatically from raw data
- Feed into `MarketTerrain` construction

### Phase 5: Optimisation Loop

- Optuna (or similar) tuning strategy params across diverse terrains
- Each trial spins up a Nautilus `BacktestNode` per terrain — these are fast (Rust core) and can run in parallel
- Same pattern as racing lab: `mean(scores)` across train terrains forces generalisation
- Pruning for bad parameter sets (like racing lab's three-layer pruner)

### Phase 6: Perturbation Testing

- Like racing lab's P4a-P4e phases
- Take real market periods, inject: liquidity shocks, flash crashes, correlation breakdowns, sentiment reversals, slippage spikes
- NautilusTrader's `FillModel` can be configured per-perturbation (e.g., higher slippage, lower fill probability, wider spreads) — the physics engine itself becomes part of the perturbation
- Verify strategy degrades gracefully

### Phase 7: Live Trading Path

- NautilusTrader `TradingNode` with live venue adapter (Binance, Interactive Brokers)
- Same `QTSStrategy` actor, same signal pipeline, same risk logic — zero code changes
- Add monitoring: Nautilus emits events for every order, fill, and position change — wire these to our existing monitoring/alerting stack
- Paper trade first via Binance testnet, then graduate to live

### Phase 8 (Optional): Synthetic Market Generation

- Simpler statistical models first (regime-switching, GARCH)
- MarS-style generative models only if Phase 1-7 results demand it
- Conditioned on Macro Engine output
- Synthetic data feeds into Nautilus via the same `ParquetDataCatalog` path

---

## 10. Key Takeaways

1. **The racing lab architecture maps 1:1 to trading.** Track -> MarketTerrain, Controller -> QTSStrategy Actor, Physics -> NautilusTrader Engine, Optuna -> Parameter Tuning.

2. **NautilusTrader is the car physics engine.** It owns execution simulation with production-grade fidelity — exchange matching, queue-position fills, multi-venue routing, portfolio accounting. Strategies developed against Nautilus work identically in live trading. No "graduation" step, no fill model surprises.

3. **The missing piece is a first-class `MarketTerrain` object** that pre-computes the environment before the strategy navigates it. This is the single most important *data* architecture change (Nautilus is the execution architecture change).

4. **MarS validates generative market simulation** but is out of scope to replicate. The architecture pattern (conditioning + generation + matching engine) is the takeaway, not the 16B-token model.

5. **MiroFish is the wrong tool.** Social media simulation, not market simulation. AGPL license. No financial validation.

6. **Start with historical replay + macro annotation** (Approach A). It gives 80% of the value immediately. Synthetic generation (Approach B) is a Phase 8 stretch goal.

7. **Train across diverse regimes** the same way RRHC trains across diverse track geometries. This is the key to preventing overfitting.

8. **The Macro Engine is the unique value-add.** MarS has microstructure but no macro awareness. NautilusTrader has execution but no macro awareness. Our system layers macro context on top — MarketTerrain + Macro Engine is what makes the strategies actually useful. The stack is: Macro Engine (why) -> MarketTerrain (what) -> NautilusTrader (how) -> Strategy (decisions).

9. **The backtest-to-live path is now trivial.** Phase 7 (live trading) requires zero strategy code changes — swap `BacktestNode` for `TradingNode`, configure venue credentials, and go. This is the hedge fund readiness that a custom backtest engine could never provide.
