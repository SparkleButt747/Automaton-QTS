# QTS Architecture

## System Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────┐
│                          QTS System                              │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐   │
│  │  Data Layer  │   │ Signal Layer │   │  Strategy Layer   │   │
│  │              │   │              │   │                   │   │
│  │ Binance Feed │──▶│ RSI / MACD   │──▶│  Alpha Fusion     │   │
│  │ News API     │   │ Bollinger    │   │  Entry/Exit Logic │   │
│  │ Reddit/PRAW  │──▶│ ATR / Mom.   │   │  Regime Scaling   │   │
│  │ GDELT        │   │ HMM Regime   │   │                   │   │
│  │ Alpha Vantage│──▶│ FinBERT/VADER│──▶│  Combined Alpha   │   │
│  └──────────────┘   └──────────────┘   └─────────┬─────────┘   │
│                                                   │             │
│  ┌──────────────────────────────────────────────▼────────────┐ │
│  │                    Execution Layer                         │ │
│  │  ExecutionEngine ──▶ RiskManager ──▶ OrderManager         │ │
│  │        │                 │                │                │ │
│  │        │           CircuitBreaker    Fills/Cancels         │ │
│  │        ▼                                  ▼                │ │
│  │   TradeLogger                       SignalLogger           │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐   │
│  │ Oversight    │   │  Monitoring  │   │  Analytics        │   │
│  │              │   │              │   │                   │   │
│  │ Proposal Q   │   │ Dashboard    │   │ Performance Calc  │   │
│  │ LLM Debrief  │   │ HealthCheck  │   │ Attribution       │   │
│  │ Audit Trail  │   │ AlertManager │   │ Backtester        │   │
│  └──────────────┘   └──────────────┘   └───────────────────┘   │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Database Layer                          │  │
│  │  PostgreSQL/TimescaleDB (prod)  |  SQLite (dev/test)       │  │
│  │  Trade Records  |  Signal Snapshots  |  Proposals          │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Description (Per Phase)

### Phase 1 – Data Ingestion

Raw market data, news, social, and geopolitical signals arrive from external
sources and are normalised into internal domain objects:

- `Tick` – single trade from exchange
- `Bar` – OHLCV aggregation (1 h default)
- Sentiment payloads – raw text scores before fusion

**Hot path:** Binance WebSocket → `TickAggregator` → `Bar` buffer
**Cold path:** News/Reddit polling → NLP pipeline → `SentimentScore`

### Phase 2 – Signal Computation

The `SignalPipeline` is called once per bar for each symbol:

1. Appends the new `Bar` to a rolling window.
2. Computes technical indicators (RSI, MACD, Bollinger Bands, ATR, Momentum).
3. Runs the HMM `RegimeDetector` on ATR history → `VolRegime`.
4. Injects the current `sentiment_score`.
5. Assembles a frozen `SignalSnapshot`.

### Phase 3 – Alpha Fusion

`combined_alpha(snapshot, params)`:

- Normalises each signal component to `[-1, 1]`.
- Applies per-signal weights from `StrategyParams.weights`.
- Scales by `_regime_scalar` (1.0 HIGH / 0.5 LOW).
- Clips to `[-1, 1]`.

### Phase 4 – Sentiment Processing

Multi-source sentiment pipeline:

1. `FinBERTSentimentScorer` processes news headlines.
2. `VADERScorer` processes Reddit posts.
3. `GDELTGeopoliticalScorer` converts GDELT intensity to `[-1, 1]`.
4. `SentimentFusion` computes `news×0.4 + social×0.3 + geopolitical×0.3`.

### Phase 5 – Execution

`ExecutionEngine.process_bar(bar, strategy, signal_pipeline)`:

1. Extend bar buffer per symbol.
2. `SignalPipeline.compute(bars)` → `SignalSnapshot` or `None`.
3. `strategy.generate_order(snapshot, positions)` → `Order` or `None`.
4. `RiskManager.approve_order(...)` → bool / raises `CircuitBreakerError`.
5. `OrderManager.submit_order(order)` → fills.
6. Fills persisted via `TradeLogger`.

### Phase 6 – Oversight

A dedicated `ProposalQueue` accumulates high-impact decisions that require
human review before execution:

- LLM `RegimeClassifier` annotates each bar with a macro label.
- `LLMDebriefClient` generates daily post-session debrief.
- Human approves/rejects proposals via CLI or dashboard.
- All decisions are written to the immutable audit trail.

### Phase 7 – Monitoring Dashboard

`TradingDashboard` renders a Rich-powered multi-panel CLI display:

- **Portfolio:** positions, unrealised PnL, daily drawdown vs limit.
- **Recent Trades:** last 10 trades coloured green/red by outcome.
- **Signals:** per-symbol RSI, MACD, sentiment, combined alpha.
- **Regime:** HMM vol regime + LLM narrative.
- **System Health:** feed freshness, DB connectivity, Redis ping.
- **Alerts:** active alerts from `AlertManager`.

`HealthChecker` probes database, market feed, and Redis.
`AlertManager` raises `Alert` objects on threshold breaches.

---

## Component Interactions

```
CLI (click)
  └─▶ AppSettings (Pydantic Settings)
        ├─▶ RiskLimits  (config/risk_limits.json)
        ├─▶ StrategyParams  (config/params.json)
        ├─▶ DatabaseSettings  (env vars)
        ├─▶ ExchangeSettings  (env vars)
        └─▶ LLMSettings  (env vars)

ExecutionEngine
  ├─▶ SignalPipeline
  │     ├─▶ indicators  (RSI, MACD, BB, ATR, Mom)
  │     └─▶ RegimeDetector  (HMM)
  ├─▶ SentimentFusion
  │     ├─▶ FinBERTScorer
  │     ├─▶ VADERScorer
  │     └─▶ GDELTScorer
  ├─▶ RiskManager  (enforces RiskLimits)
  ├─▶ OrderManager  (paper/live fills)
  ├─▶ TradeLogger  (write-through to DB)
  └─▶ SignalLogger  (write-through to DB)

TradingDashboard
  ├─▶ DashboardState  (data transfer object)
  ├─▶ HealthChecker  (check_database, check_market_feed, check_redis)
  └─▶ AlertManager  (check_drawdown, check_feed_staleness, check_position_concentration)
```

---

## Hot Path vs Cold Path

| Path   | Components                                          | Latency Target |
|--------|-----------------------------------------------------|----------------|
| Hot    | Binance WS → TickAggregator → Bar → SignalPipeline → RiskManager → OrderManager | < 100 ms |
| Warm   | Sentiment fusion update (news/Reddit polling)       | < 5 s          |
| Cold   | LLM debrief, regime classification, GDELT           | < 60 s         |
| Async  | ProposalQueue, audit trail writes                   | best-effort    |

The hot path **never** blocks on LLM calls, DB writes, or Redis.  All
persistence is fire-and-forget via `TradeLogger`/`SignalLogger`, which fall
back to an in-memory buffer on failure.

---

## Database Schema Overview

### `trade_records`

| Column                       | Type        | Notes                         |
|------------------------------|-------------|-------------------------------|
| trade_id                     | UUID PK     |                               |
| symbol                       | VARCHAR     | e.g. BTCUSDT                  |
| entry_time / exit_time       | TIMESTAMPTZ | TimescaleDB hypertable key    |
| direction                    | ENUM        | LONG / SHORT                  |
| entry_price / exit_price     | NUMERIC     |                               |
| size / pnl_usd / pnl_pct     | NUMERIC     |                               |
| outcome                      | ENUM        | WIN / LOSS / BREAKEVEN        |
| exit_reason                  | ENUM        | ALPHA_FLIP / TIME_STOP / …    |
| failure_mode                 | ENUM NULL   | Attribution taxonomy          |
| params_version               | VARCHAR     | Links to config snapshot      |

### `signal_snapshots`

| Column              | Type        | Notes                             |
|---------------------|-------------|-----------------------------------|
| symbol              | VARCHAR     |                                   |
| timestamp           | TIMESTAMPTZ | TimescaleDB hypertable key        |
| rsi / macd_*        | NUMERIC     | Technical indicator values        |
| bb_position         | NUMERIC     | [0, 1]                            |
| atr / momentum_5    | NUMERIC     |                                   |
| vol_regime          | ENUM        | HIGH / LOW                        |
| sentiment_score     | NUMERIC     | [-1, 1]                           |
| combined_alpha      | NUMERIC     | [-1, 1]                           |

### `proposals`

| Column              | Type        | Notes                             |
|---------------------|-------------|-----------------------------------|
| proposal_id         | UUID PK     |                                   |
| created_at          | TIMESTAMPTZ |                                   |
| kind                | ENUM        | REGIME_OVERRIDE / LIMIT_CHANGE / … |
| status              | ENUM        | PENDING / APPROVED / REJECTED     |
| payload             | JSONB       | Proposal details                  |
| reviewed_at         | TIMESTAMPTZ NULL |                              |
| reviewer            | VARCHAR NULL |                                  |
