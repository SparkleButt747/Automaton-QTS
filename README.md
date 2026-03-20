# Automaton QTS — Quantitative Trading System

> A multi-signal, LLM-assisted trading engine where every trade is traceable to exact signal values, parameter versions, and regime state — and a human always has the final word.

![CI](https://img.shields.io/badge/CI-passing-brightgreen) ![Coverage](https://img.shields.io/badge/coverage-346%20tests-blue) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-green)

---

## Overview

Automaton QTS fuses technical indicators (RSI, MACD, Bollinger Bands, ATR, Momentum), multi-source NLP sentiment (FinBERT, VADER, GDELT), and Hidden Markov Model volatility regime detection into a single composite alpha score that drives position management. An LLM layer (Ollama locally, or Anthropic Claude in the cloud) acts as an analyst — generating daily debriefs, classifying macro regimes, and proposing parameter adjustments — but **never** executes trades autonomously. Every proposed parameter change is queued, reviewed in a Rich terminal UI, and only applied after explicit human approval, with a full Git audit trail.

Key differentiators:
- **Full traceability**: each trade record links to the exact `SignalSnapshot`, `params_version`, and regime state at entry.
- **Immutable risk controls**: `config/risk_limits.json` is protected by a pre-commit hook; the LLM cannot modify it under any circumstances.
- **Human-in-the-loop**: all parameter changes go through a proposal → review → approve/reject workflow before taking effect.

---

## Features

| Feature | Description |
|---|---|
| **Multi-signal alpha fusion** | Configurable weighted combination of RSI, MACD, Bollinger Bands, ATR, and Momentum signals |
| **NLP sentiment pipeline** | FinBERT transformer inference on news, VADER on social posts, GDELT geopolitical conflict scoring — fused into a single `[-1, 1]` sentiment score |
| **HMM volatility regime** | Hidden Markov Model classifies each bar as HIGH or LOW volatility; regime scalar adjusts position sizing accordingly |
| **Immutable risk controls** | Hard circuit breaker, max drawdown, position size, and open-position limits; protected by pre-commit hook |
| **LLM oversight** | Daily session debrief, per-bar regime classification, parameter change proposals — LLM is analyst only, never executor |
| **Human approval CLI** | Rich terminal UI for reviewing and approving/rejecting LLM proposals; every decision appended to Git audit trail |
| **Event-driven backtest** | Deterministic bar-by-bar backtest using identical signal logic as live trading; realistic fill simulation |
| **Stress testing** | Parameterised scenarios: flash crash, sentiment spike, data gap — run against any strategy |
| **Trade attribution** | Failure-mode taxonomy (alpha flip, time stop, risk limit, circuit breaker) with per-trade attribution reports |
| **Rich monitoring dashboard** | Multi-panel terminal dashboard: portfolio, recent trades, signals, regime, system health, active alerts |
| **Dual LLM backends** | Ollama (local, no API key required) and Anthropic Claude (cloud); switchable via a single env var |

---

## Architecture

### System Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Automaton QTS                                │
│                                                                       │
│  ┌────────────────┐   ┌─────────────────┐   ┌─────────────────────┐ │
│  │  Data Layer    │   │  Signal Layer   │   │   Strategy Layer    │ │
│  │                │   │                 │   │                     │ │
│  │ Binance Feed   │──▶│ RSI / MACD      │──▶│  Alpha Fusion       │ │
│  │ News API       │   │ Bollinger / ATR  │   │  Entry/Exit Logic   │ │
│  │ Reddit/PRAW    │──▶│ Momentum        │   │  Regime Scaling     │ │
│  │ GDELT          │   │ HMM Regime      │   │                     │ │
│  │ Alpha Vantage  │──▶│ FinBERT / VADER │──▶│  Combined Alpha     │ │
│  └────────────────┘   └─────────────────┘   └──────────┬──────────┘ │
│                                                         │            │
│  ┌──────────────────────────────────────────────────────▼──────────┐ │
│  │                       Execution Layer                            │ │
│  │   ExecutionEngine ──▶ RiskManager ──▶ OrderManager              │ │
│  │         │                  │               │                    │ │
│  │         │            CircuitBreaker   Fills / Cancels           │ │
│  │         ▼                                  ▼                    │ │
│  │    TradeLogger                       SignalLogger               │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌────────────────┐   ┌─────────────────┐   ┌─────────────────────┐ │
│  │  Oversight     │   │  Monitoring     │   │  Analytics          │ │
│  │                │   │                 │   │                     │ │
│  │ ProposalQueue  │   │ Dashboard       │   │ Performance Calc    │ │
│  │ LLM Debrief    │   │ HealthChecker   │   │ Attribution         │ │
│  │ Audit Trail    │   │ AlertManager    │   │ Backtester          │ │
│  └────────────────┘   └─────────────────┘   └─────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                       Database Layer                             │ │
│  │   PostgreSQL / TimescaleDB (prod)  |  SQLite (dev / test)       │ │
│  │   Trade Records  |  Signal Snapshots  |  Proposals              │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### Hot Path vs Cold Path

| Path | Components | Latency Target |
|---|---|---|
| **Hot** | Binance WebSocket → TickAggregator → Bar → SignalPipeline → RiskManager → OrderManager | < 100 ms |
| **Warm** | Sentiment fusion update (news/Reddit polling) | < 5 s |
| **Cold** | LLM debrief, regime classification, GDELT ingestion | < 60 s |
| **Async** | ProposalQueue writes, audit trail, DB persistence | best-effort |

The hot path **never** blocks on LLM calls, database writes, or Redis. All persistence is fire-and-forget via `TradeLogger` and `SignalLogger`, which fall back to an in-memory buffer on failure.

---

## Project Structure

```
quant-trading-system/
├── config/
│   ├── params.json            # Strategy parameters (LLM-proposable, human-approved)
│   ├── risk_limits.json       # IMMUTABLE hard limits — never LLM-modifiable
│   └── data_sources.json      # Feed configuration
├── src/qts/
│   ├── config.py              # Pydantic settings: RiskLimits, StrategyParams, LLMSettings, etc.
│   ├── models/base.py         # Domain models: Tick, Bar, Order, Fill, Position, SignalSnapshot, TradeRecord
│   ├── data/                  # Market data adapters and protocols
│   │   ├── protocols.py       # TickProvider, BarProvider, OrderBookProvider interfaces
│   │   ├── market/            # CSV adapter (backtest), mock adapter (testing)
│   │   ├── news/              # Alpha Vantage news client
│   │   ├── social/            # Reddit (PRAW), StockTwits clients
│   │   └── geopolitical/      # GDELT event ingestion
│   ├── signals/               # Technical indicators and signal pipeline
│   │   ├── indicators.py      # RSI, MACD, Bollinger, ATR, Momentum (pure numpy)
│   │   ├── regime.py          # HMM volatility regime detector
│   │   ├── alpha.py           # Combined alpha signal fusion
│   │   └── pipeline.py        # Orchestrates all signal computation per bar
│   ├── nlp/                   # NLP and sentiment analysis
│   │   ├── finbert.py         # FinBERT transformer inference with time decay
│   │   ├── vader.py           # VADER social media sentiment
│   │   ├── gdelt.py           # GDELT conflict intensity scoring
│   │   └── fusion.py          # Multi-source sentiment fusion (news×0.4 + social×0.3 + geo×0.3)
│   ├── strategies/            # Trading strategies
│   │   ├── base.py            # Strategy protocol (interface)
│   │   ├── momentum.py        # Momentum strategy with sentiment overlay
│   │   └── sma_crossover.py   # SMA crossover strategy (integration tests)
│   ├── execution/             # Order execution and risk management
│   │   ├── engine.py          # ExecutionEngine — bar processing loop
│   │   ├── risk.py            # RiskManager and circuit breaker
│   │   └── order_manager.py   # Order lifecycle management
│   ├── simulation/            # Backtesting and stress testing
│   │   ├── backtest.py        # Deterministic backtest engine
│   │   └── scenario.py        # Flash crash, sentiment spike, data gap scenarios
│   ├── oversight/             # LLM oversight layer
│   │   ├── llm_client.py      # OllamaClient + AnthropicClient (LLMClientProtocol)
│   │   ├── debrief.py         # Daily session debrief generation via LLM
│   │   ├── regime_classifier.py  # LLM-assisted geopolitical regime classification
│   │   ├── proposals.py       # Proposal management: save, validate, approve, reject
│   │   └── approval_cli.py    # Rich terminal approval UI
│   ├── analytics/             # Trade attribution
│   │   └── attribution.py     # Failure mode classification and attribution reports
│   ├── trade_logging/         # Structured trade and signal event logging
│   ├── monitoring/            # Live monitoring
│   │   ├── dashboard.py       # Rich terminal dashboard (multi-panel)
│   │   ├── health.py          # Component health checks (DB, feed, Redis)
│   │   └── alerts.py          # Drawdown, feed staleness, concentration alerts
│   ├── db/                    # Database layer
│   │   ├── engine.py          # Async SQLAlchemy engine factory
│   │   ├── tables.py          # ORM table definitions
│   │   └── repository.py      # Repository pattern (CRUD operations)
│   └── cli/                   # Click CLI entry points
├── tests/
│   ├── unit/                  # 346 unit tests
│   └── integration/           # Backtest and end-to-end integration tests
├── scripts/
│   ├── test_ollama.py         # Functional test for the Ollama LLM integration
│   ├── run_backtest.py        # Backtest runner (wraps qts backtest CLI)
│   ├── run_debrief.py         # Trigger LLM daily debrief
│   └── approve_proposals.py   # Launch the proposal approval UI
├── docs/
│   ├── architecture.md        # Full system architecture and data flow diagrams
│   ├── signal_spec.md         # Every signal: formula, range, staleness policy
│   ├── risk_controls.md       # Hard limits, rationale, and change procedures
│   ├── llm_prompt_templates.md # All LLM prompt templates and expected schemas
│   └── incident_runbook.md    # Six incident response procedures
├── docker/
│   ├── Dockerfile             # Multi-stage Python 3.11 build
│   └── docker-compose.yml     # TimescaleDB + Redis + application stack
├── pyproject.toml             # Project metadata, dependencies, and tool configuration
├── Makefile                   # Dev workflow commands
└── .env.example               # Environment variable template
```

---

## Prerequisites

- **Python 3.11+**
- **Git**
- **Ollama** — for local LLM inference (recommended for development, no API key required)
  - Install: https://ollama.com/download
- **Docker & Docker Compose** — optional, for TimescaleDB + Redis in development

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/SparkleButt747/Automaton_QTS.git
cd Automaton_QTS

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install the package with dev dependencies
pip install -e ".[dev]"

# 4. Configure environment variables
cp .env.example .env
# Open .env and fill in any API keys you want to use.
# For basic usage (backtest + Ollama), no keys are required.

# 5. Run the full test suite
make test
# Or directly: pytest tests/ -v

# 6. (Optional) Start infrastructure services
make docker-up                   # Starts TimescaleDB and Redis via Docker Compose

# 7. (Optional) Test the Ollama integration
ollama pull llama3               # Pull a local model (or any supported model)
python scripts/test_ollama.py    # Verify end-to-end LLM connectivity
```

---

## Configuration Guide

### Environment Variables (`.env`)

Copy `.env.example` to `.env` and set the values you need. All variables have sensible defaults for local development.

| Variable | Default | Description |
|---|---|---|
| `QTS_ENV` | `development` | Runtime environment: `development`, `staging`, or `production` |
| `QTS_LOG_LEVEL` | `INFO` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `QTS_DRY_RUN` | `true` | Paper trading mode — no real orders are submitted when `true` |
| `DATABASE_URL` | `sqlite:///data/trading.db` | SQLAlchemy connection URL (use TimescaleDB URL in production) |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL for background task queue |
| `LLM_BACKEND` | `ollama` | LLM provider: `ollama` (local) or `anthropic` (cloud) |
| `OLLAMA_MODEL` | `llama3` | Ollama model tag to use (e.g. `llama3`, `mistral`, `codellama`) |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL for the local Ollama API server |
| `ANTHROPIC_API_KEY` | _(empty)_ | Anthropic API key — only required when `LLM_BACKEND=anthropic` |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Anthropic model ID |
| `LLM_MAX_TOKENS` | `4096` | Maximum tokens in LLM responses |
| `LLM_TEMPERATURE` | `0.0` | LLM sampling temperature (0 = deterministic) |
| `BINANCE_API_KEY` | _(empty)_ | Binance API key — only needed for live trading |
| `BINANCE_API_SECRET` | _(empty)_ | Binance API secret |
| `ALPHA_VANTAGE_API_KEY` | _(empty)_ | Alpha Vantage key for equity news and market data |
| `REDDIT_CLIENT_ID` | _(empty)_ | Reddit OAuth2 client ID for social sentiment |
| `REDDIT_CLIENT_SECRET` | _(empty)_ | Reddit OAuth2 client secret |
| `REDDIT_USER_AGENT` | `QTS/1.0` | Reddit API user agent string |

### Strategy Parameters (`config/params.json`)

Controls how signals are weighted and when positions are entered or exited.

```json
{
  "version": "1.0.0",
  "weights": {
    "w_rsi": 0.20,
    "w_macd": 0.20,
    "w_bb": 0.15,
    "w_mom": 0.15,
    "w_sentiment": 0.30
  },
  "entry_threshold": 0.25,
  "exit_threshold": -0.10,
  "max_hold_bars": 48,
  "sentiment_fusion_weights": {
    "news": 0.40,
    "social": 0.30,
    "geopolitical": 0.30
  }
}
```

| Parameter | Description |
|---|---|
| `weights.w_*` | Per-signal contribution to combined alpha. **Must sum to exactly 1.0.** |
| `entry_threshold` | Minimum composite alpha score `[0, 1]` required to open a new position |
| `exit_threshold` | Alpha score below which an existing position is closed |
| `max_hold_bars` | Maximum number of 1-hour bars before a position is force-closed |
| `sentiment_fusion_weights` | How to combine news, social, and geopolitical scores. **Must sum to 1.0.** |

The LLM may propose changes to this file, but they only take effect after human approval via `qts approve` or `python scripts/approve_proposals.py`.

### Risk Limits (`config/risk_limits.json`)

```json
{
  "max_daily_drawdown_pct": 0.02,
  "max_position_size_pct": 0.05,
  "max_open_positions": 5,
  "circuit_breaker_cooldown_seconds": 3600,
  "sentiment_signal_max_scalar": 2.0
}
```

| Limit | Default | Description |
|---|---|---|
| `max_daily_drawdown_pct` | 2% | Portfolio loss ceiling per calendar day; triggers circuit breaker |
| `max_position_size_pct` | 5% | Maximum single-position size as a fraction of total portfolio |
| `max_open_positions` | 5 | Hard cap on simultaneously open positions |
| `circuit_breaker_cooldown_seconds` | 3600 | Lockout period (1 hour) after the circuit breaker trips |
| `sentiment_signal_max_scalar` | 2.0 | Maximum multiplier the sentiment signal can apply to position sizing |

**These limits are immutable at runtime.** The pre-commit hook (`.pre-commit-hooks/protect-risk-limits.sh`) rejects any commit that modifies `config/risk_limits.json` without the `--force` flag. The LLM layer is architecturally prohibited from proposing changes to this file.

To change a risk limit: edit the file, run `git commit --no-verify` (requires manual review), and document the change in `docs/risk_controls.md`.

### LLM Backend Selection

**Ollama (default — recommended for development)**

Ollama runs models locally with no API key required.

```bash
# Install Ollama: https://ollama.com/download
ollama pull llama3           # Pull a model
ollama serve                 # Start the API server (usually starts automatically)
```

Set in `.env`:
```
LLM_BACKEND=ollama
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
```

Test the connection:
```bash
python scripts/test_ollama.py
```

**Anthropic Claude (cloud)**

```
LLM_BACKEND=anthropic
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

Supported Ollama models (tested): `llama3`, `mistral`, `codellama`, `glm-5:cloud`  
Supported Anthropic models: any Claude 3 / Claude 4 model ID from the Anthropic API.

---

## Usage Examples

### Run a Backtest

```python
import asyncio
from qts.simulation.backtest import BacktestEngine
from qts.strategies.sma_crossover import SMACrossoverStrategy
from qts.data.market.mock_adapter import MockBarAdapter
from qts.config import get_settings

async def main():
    settings = get_settings()
    adapter = MockBarAdapter(symbol="BTCUSDT", num_bars=500)
    strategy = SMACrossoverStrategy(fast=10, slow=30)
    engine = BacktestEngine(
        data_adapter=adapter,
        strategy=strategy,
        risk_limits=settings.risk,
        strategy_params=settings.strategy,
    )
    result = await engine.run()
    print(f"Total return:   {result.total_return_pct:.2f}%")
    print(f"Sharpe ratio:   {result.sharpe_ratio:.3f}")
    print(f"Max drawdown:   {result.max_drawdown_pct:.2f}%")
    print(f"Win rate:       {result.win_rate:.1%}")
    print(f"Total trades:   {result.total_trades}")

asyncio.run(main())
```

Or use the CLI:

```bash
qts backtest --symbol BTCUSDT --start 2023-01-01 --end 2023-12-31
# Alternatively:
python scripts/run_backtest.py --symbol BTCUSDT --start 2023-01-01 --end 2023-12-31
```

### Compute Signals for a Bar Series

```python
from qts.signals.pipeline import SignalPipeline
from qts.signals.alpha import combined_alpha
from qts.config import get_settings

settings = get_settings()
pipeline = SignalPipeline(params=settings.strategy)

# `bars` is a list of qts.models.base.Bar objects
snapshot = pipeline.compute(bars=bars, sentiment_score=0.15)
if snapshot is not None:
    alpha = combined_alpha(snapshot, settings.strategy)
    print(f"Combined alpha: {alpha:.4f}")
    print(f"RSI:            {snapshot.rsi:.2f}")
    print(f"MACD:           {snapshot.macd_line:.4f}")
    print(f"Vol regime:     {snapshot.vol_regime}")
    print(f"Sentiment:      {snapshot.sentiment_score:.4f}")
```

### Review and Approve LLM Proposals

```bash
# Interactive Rich terminal UI — lists all pending proposals with full detail
python scripts/approve_proposals.py
# Or:
qts approve
```

Proposals show the LLM's reasoning, the current parameter value, the proposed value, and confidence score. Approving writes the change to `config/params.json` and appends an entry to the audit log.

### Trigger a Post-Session Debrief

```bash
python scripts/run_debrief.py
# Or:
qts debrief
```

The debrief queries the LLM with a summary of the session's trades, signal values, and attribution data. The resulting report is printed to the terminal and saved to the database.

### Monitor Live Trading

```bash
qts monitor
```

The Rich dashboard displays six panels refreshed every second:

- **Portfolio** — open positions, unrealised PnL, daily drawdown vs limit
- **Recent Trades** — last 10 trades coloured by outcome (green/red)
- **Signals** — per-symbol RSI, MACD, sentiment, and combined alpha
- **Regime** — HMM vol regime and LLM macro narrative
- **System Health** — feed freshness, database connectivity, Redis ping
- **Alerts** — active alerts from `AlertManager`

---

## Development

### Make Targets

| Target | Description |
|---|---|
| `make install` | Install production dependencies only |
| `make dev-install` | Install all dependencies including dev tools and pre-commit hooks |
| `make fmt` | Auto-format code with Black and Ruff (`--fix`) |
| `make lint` | Run Ruff linter and Black format check |
| `make typecheck` | Run mypy in strict mode |
| `make check` | `lint` + `typecheck` |
| `make test` | Full test suite with coverage report (HTML + terminal) |
| `make test-unit` | Unit tests only |
| `make test-integration` | Integration tests only |
| `make test-fast` | Quick run, stops on first failure, no coverage |
| `make docker-up` | Start TimescaleDB and Redis via Docker Compose |
| `make docker-down` | Stop Docker services |
| `make docker-logs` | Stream Docker service logs |
| `make clean` | Remove build artifacts, caches, and coverage files |
| `make all` | `fmt` + `lint` + `typecheck` + `test` |

### Running Tests

```bash
make test              # Full suite with coverage (346 unit + integration tests)
make test-unit         # Unit tests only
make test-fast         # Quick run, stop on first failure, skip coverage
make test-integration  # Integration tests only

# Run a specific test file directly
pytest tests/unit/test_signals.py -v
```

### Code Quality

```bash
make fmt           # Auto-format with black and ruff (modifies files)
make lint          # Check formatting and style (read-only)
make typecheck     # mypy --strict over src/qts/
make check         # lint + typecheck (no file modifications)
make all           # fmt + lint + typecheck + test (full CI equivalent)
```

---

## Documentation

| Document | Description |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Full system architecture, data flow per phase, database schema, component interaction diagram |
| [`docs/signal_spec.md`](docs/signal_spec.md) | Every signal: formula, output range, staleness policy, and edge cases |
| [`docs/risk_controls.md`](docs/risk_controls.md) | Hard risk limits, rationale for each, and the procedure for changing them |
| [`docs/llm_prompt_templates.md`](docs/llm_prompt_templates.md) | All LLM prompt templates, expected JSON schemas, and example responses |
| [`docs/incident_runbook.md`](docs/incident_runbook.md) | Six incident response procedures (circuit breaker trip, data feed outage, LLM failure, etc.) |

---

## Risk Notice

**This software is for research and educational purposes only.**

Algorithmic trading carries significant financial risk, including the total loss of invested capital. Backtested performance does not guarantee future results. Always run with `QTS_DRY_RUN=true` until the system has been thoroughly validated in your environment. The authors accept no liability for financial losses incurred through the use of this software. Consult a qualified financial advisor before deploying any trading system with real capital.

---

## License

CC-BY-NC-SA-4.0 — see [LICENSE](LICENSE) for details.
