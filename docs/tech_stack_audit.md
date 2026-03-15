# Tech Stack Audit & Implementation TODO

> **Generated:** 2026-03-14  
> **Status:** Post-Phase 7 scaffold — functional test suite, partial integrations  
> **Tests:** 346 passing | Coverage: 56%

---

## Executive Summary

The project has a **solid core**: config management, signal engine, backtest engine, NLP
pipeline, LLM oversight, and monitoring dashboard are all implemented with real code and
passing tests. However, several **planned integrations exist only as config fields or
dependency listings** — notably NautilusTrader, hftbacktest, Binance WebSocket, Alpaca,
Celery tasks, Alembic migrations, and Prometheus/Grafana. This document maps every
planned technology to its actual implementation status, API key requirements, and what
work remains.

---

## Tech Stack Status Matrix

| # | Technology | Planned Role | Status | API Key? | Free Tier? | Functional Now? |
|---|-----------|-------------|--------|----------|------------|-----------------|
| 1 | **Python 3.11+** | Language | ✅ Fully implemented | No | — | ✅ Yes |
| 2 | **NautilusTrader** | Trading engine | ❌ Not implemented | No | OSS (LGPL) | ❌ No |
| 3 | **hftbacktest** | HFT backtesting | ❌ Not implemented | No | OSS (Apache-2.0) | ❌ No |
| 4 | **Binance WS+REST** | Crypto market data | ❌ Not implemented | Yes (for private) | Yes (public data free) | ❌ No |
| 5 | **Alpaca** | Equity market data | ❌ Not implemented | Yes | Yes (paper trading free) | ❌ No |
| 6 | **FinBERT** | News sentiment | ✅ Implemented | No | Free HF model | ✅ Yes (lazy load) |
| 7 | **VADER** | Social sentiment | ✅ Implemented | No | Free package | ✅ Yes |
| 8 | **GDELT** | Geopolitical events | ⚠️ Processor only | No | Free, no key | ⚠️ Partial |
| 9 | **Alpha Vantage** | News feed | ✅ Client implemented | Yes | 25 req/day (free) | ⚠️ Needs key |
| 10 | **Reddit / PRAW** | Social data | ✅ Client implemented | Yes (OAuth2) | 60 req/min | ⚠️ Needs key |
| 11 | **StockTwits** | Social sentiment | ✅ Client implemented | No (public endpoint) | 1000 req/hr | ✅ Yes |
| 12 | **Claude API** | LLM oversight | ✅ Client implemented | Yes | No free tier | ⚠️ Needs key |
| 13 | **Ollama** | Local LLM | ✅ Fully implemented | No | Free (local) | ✅ Yes |
| 14 | **SQLite** | Dev database | ✅ Configured | No | — | ✅ Yes |
| 15 | **PostgreSQL/TimescaleDB** | Prod database | ⚠️ Docker only | No | OSS | ⚠️ Docker req'd |
| 16 | **Celery + Redis** | Task queue | ❌ No tasks defined | No | OSS | ❌ No |
| 17 | **Pydantic** | Config management | ✅ Fully implemented | No | — | ✅ Yes |
| 18 | **pytest + pytest-asyncio** | Testing | ✅ 346 tests | No | — | ✅ Yes |
| 19 | **hypothesis** | Property testing | ❌ No tests written | No | — | ❌ No |
| 20 | **Grafana + Prometheus** | Monitoring | ❌ Not implemented | No | OSS | ❌ No |
| 21 | **Rich CLI dashboard** | Monitoring | ✅ Implemented | No | — | ✅ Yes |
| 22 | **Docker Compose** | Containerisation | ⚠️ Config exists | No | — | ⚠️ Untested |
| 23 | **Alembic** | DB migrations | ❌ Not configured | No | — | ❌ No |

**Legend:** ✅ Working now | ⚠️ Partial/needs config | ❌ Not implemented

---

## Detailed Assessment Per Technology

### 1. Python 3.11+ ✅
- **Status:** Fully functional. Project runs on Python 3.14 (tested).
- **Nothing to do.**

### 2. NautilusTrader ❌
- **What it is:** Event-driven trading framework with a Rust core and Python API. Supports live trading with Binance, Interactive Brokers, etc.
- **Current state:** Listed as optional dependency (`pip install qts[trading]`). **Zero imports** in `src/qts/`. Not used anywhere.
- **What exists instead:** Custom `BacktestEngine` in `src/qts/simulation/backtest.py` and custom `ExecutionEngine` in `src/qts/execution/engine.py`.
- **API key:** No (open source, LGPL-2.1 license).
- **Cost:** Free.
- **What's needed:** Either integrate NautilusTrader as the execution backbone (replace custom engine) or remove it from the plan and keep the custom engine. The custom engine is simpler and works — NautilusTrader adds realistic exchange simulation, multi-venue support, and live broker connectivity.
- **Effort:** Large (2-3 weeks) — NautilusTrader has a complex API with Cython internals.

### 3. hftbacktest ❌
- **What it is:** Rust-based HFT backtesting library with queue-position-aware, tick-level simulation.
- **Current state:** **Not even listed in pyproject.toml dependencies.** Zero presence in codebase.
- **API key:** No (open source, Apache-2.0).
- **Cost:** Free.
- **What's needed:** Add dependency, create `src/qts/simulation/hft_backtest.py` adapter, convert tick data to hftbacktest format. Only needed if pursuing HFT (sub-second) strategies.
- **Effort:** Medium (1-2 weeks) — requires understanding hftbacktest's Rust data format.

### 4. Binance WebSocket + REST ❌
- **What it is:** Real-time crypto market data and order execution.
- **Current state:** Config fields exist (`BINANCE_API_KEY`, `BINANCE_API_SECRET` in `ExchangeSettings`). **No WebSocket code exists.** Market data directory only has `csv_adapter.py` and `mock_adapter.py`.
- **API key:** Required for private endpoints (trading, account). **Public market data (ticks, orderbook, klines) is free without a key.**
- **Free tier:** 1,200 request weight/min REST, 5 WS connections/IP. More than enough for a single-strategy system.
- **What's needed:** Create `src/qts/data/market/binance_adapter.py` implementing `BarProvider` and `TickProvider` protocols. Use `httpx` for REST (historical klines) and `websockets` for real-time streams.
- **Effort:** Medium (1 week) — the protocol interfaces already exist, just need the concrete Binance adapter.
- **Can start without API key:** Yes, for public market data only.

### 5. Alpaca ❌
- **What it is:** Commission-free equity/crypto trading API with market data.
- **Current state:** **Zero presence in codebase.** Not in dependencies, not in config, not imported anywhere.
- **API key:** Required (free account provides paper trading + limited market data).
- **Free tier:** Unlimited paper trading, 200 market data requests/day (REST), 5 WS connections.
- **What's needed:** Add `alpaca-trade-api` dependency, create `src/qts/data/market/alpaca_adapter.py`, add `AlpacaSettings` to config.
- **Effort:** Medium (1 week).

### 6. FinBERT ✅
- **What it is:** Fine-tuned BERT model for financial sentiment analysis (HuggingFace).
- **Current state:** `src/qts/nlp/finbert.py` — **fully implemented** with lazy model loading, batch inference, exponential decay function, and `FinBERTProtocol` for DI. Falls back gracefully if `transformers`/`torch` not installed.
- **API key:** None. Model downloads from HuggingFace Hub on first use (~400MB).
- **Cost:** Free. Runs on CPU (~400ms/article).
- **What's needed:** Nothing for core functionality. Consider GPU support for batch processing.
- **Works now:** ✅ Yes (model auto-downloads on first `analyze()` call).

### 7. VADER ✅
- **What it is:** Rule-based sentiment analysis, good for social media text.
- **Current state:** `src/qts/nlp/vader.py` — **fully implemented** with lazy loading and `VaderProtocol`.
- **API key:** None.
- **Cost:** Free.
- **Works now:** ✅ Yes.

### 8. GDELT ⚠️ Partial
- **What it is:** Open database of global events, language, and tone. Updated every 15 minutes.
- **Current state:** `src/qts/nlp/gdelt.py` has `GDELTProcessor` that processes `GeopoliticalEvent` objects and computes conflict intensity scores. **But there is no HTTP client to actually fetch data from GDELT.** The processor works on pre-fetched data only.
- **API key:** None required. GDELT is fully free and open.
- **What's needed:** Create `src/qts/data/geopolitical/gdelt_client.py` that fetches from GDELT GKG API (HTTP GET, returns CSV/JSON). Wire it to feed `GeopoliticalEvent` objects into `GDELTProcessor`.
- **Effort:** Small (2-3 days).
- **Works now:** ⚠️ Processor works, but needs a data fetcher.

### 9. Alpha Vantage ⚠️ Needs Key
- **What it is:** Financial data API with a news sentiment endpoint.
- **Current state:** `src/qts/data/news/alpha_vantage.py` — **real httpx async client** that hits `https://www.alphavantage.co/query?function=NEWS_SENTIMENT`. Fully implemented.
- **API key:** Required. Free key from [alphavantage.co](https://www.alphavantage.co/support/#api-key).
- **Free tier:** 25 requests/day. Paid starts at $49.99/mo for 30 req/min.
- **What's needed:** Get a free API key, set `ALPHA_VANTAGE_API_KEY` in `.env`.
- **Works now:** ⚠️ Code works, needs API key.

### 10. Reddit / PRAW ⚠️ Needs Key
- **What it is:** Reddit API client for scraping subreddit posts.
- **Current state:** `src/qts/data/social/reddit.py` — **real PRAW client** using OAuth2, runs in thread executor for async compatibility.
- **API key:** Required (Reddit OAuth2 app credentials).
- **Free tier:** 60 requests/minute. Create app at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps).
- **What's needed:** Register a Reddit app, set `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` in `.env`.
- **Works now:** ⚠️ Code works, needs OAuth credentials.

### 11. StockTwits ✅
- **What it is:** Social platform for traders with sentiment data.
- **Current state:** `src/qts/data/social/stocktwits.py` — **real httpx client** hitting the public API.
- **API key:** Not required for public streams.
- **Free tier:** 1,000 requests/hour.
- **Works now:** ✅ Yes, no key needed for basic usage.

### 12. Claude API (Anthropic) ⚠️ Needs Key
- **What it is:** Anthropic's Claude LLM for async analysis jobs.
- **Current state:** `src/qts/oversight/llm_client.py` — **fully implemented** `LLMClient` using `anthropic.AsyncAnthropic` with exponential backoff retry.
- **API key:** Required. No free tier. Pay-per-token pricing.
- **What's needed:** Set `ANTHROPIC_API_KEY` in `.env` and `LLM_BACKEND=anthropic`.
- **Works now:** ⚠️ Code works, needs API key + billing.

### 13. Ollama (Local LLM) ✅
- **What it is:** Local LLM inference server.
- **Current state:** `OllamaClient` in `llm_client.py` — **fully implemented and tested**. Factory function `create_llm_client(backend="ollama")` selects it.
- **API key:** None.
- **Cost:** Free (runs locally).
- **Works now:** ✅ Yes. Tested with `glm-5:cloud`. Use any Ollama model.

### 14. SQLite ✅
- **Current state:** Default database backend. Used in all tests (in-memory). Configured via `DATABASE_URL=sqlite:///data/trading.db`.
- **Works now:** ✅ Yes.

### 15. PostgreSQL / TimescaleDB ⚠️ Docker Required
- **Current state:** `docker-compose.yml` defines a `timescaledb` service (`timescale/timescaledb:latest-pg16`). DB engine in `src/qts/db/engine.py` supports PostgreSQL URLs. **No Alembic migrations exist** to create tables.
- **What's needed:** Set up Alembic, create initial migration, test with Docker Compose.
- **Works now:** ⚠️ Docker service defined but tables aren't auto-created.

### 16. Celery + Redis ❌
- **What it is:** Distributed task queue for async background jobs (sentiment updates, LLM calls, GDELT polling).
- **Current state:** `celery` and `redis` are in `pyproject.toml` dependencies. `docker-compose.yml` defines both `redis` service and a `celery-worker` service that runs `celery -A qts.tasks worker`. **But `qts.tasks` module does not exist.** There are zero `@task` decorators anywhere in the codebase.
- **API key:** None (self-hosted).
- **What's needed:** Create `src/qts/tasks.py` with Celery app instance and task definitions for: sentiment pipeline refresh, GDELT polling, LLM debrief, nightly attribution.
- **Effort:** Medium (3-5 days).

### 17. Pydantic Config ✅
- **Current state:** Fully implemented. `RiskLimits` (frozen, from JSON), `StrategyParams` (with weight sum validators), `AppSettings` (master class with lazy sub-settings). All tested.
- **Works now:** ✅ Yes.

### 18. pytest + pytest-asyncio ✅
- **Current state:** 346 tests passing. Coverage at 56%. Async tests use `asyncio_mode = "auto"`.
- **Works now:** ✅ Yes.

### 19. hypothesis (Property Testing) ❌
- **What it is:** Property-based testing library for generating random test cases.
- **Current state:** Listed in dev dependencies. **Zero `@given` decorators or hypothesis imports in any test file.**
- **What's needed:** Add property tests for:
  - `indicators.py`: RSI always in [0, 100], MACD histogram = line - signal, etc.
  - `alpha.py`: combined_alpha always in [-1, 1] for any input
  - `fusion.py`: fused sentiment always in [-1, 1]
  - `risk.py`: circuit breaker always triggers on drawdown > limit
  - Fill model: never fills outside OHLC range (from spec)
- **Effort:** Small (2-3 days).

### 20. Grafana + Prometheus ❌
- **What it is:** Metrics collection (Prometheus) + dashboarding (Grafana).
- **Current state:** **Zero presence.** No Prometheus client, no metrics endpoints, no Grafana dashboards, no Docker services.
- **The spec says:** "(optional)" — the Rich CLI dashboard is the required monitoring.
- **What's needed (if desired):** Add `prometheus-client` dependency, instrument key functions with counters/histograms, expose `/metrics` endpoint, add Grafana + Prometheus services to docker-compose, create dashboard JSON.
- **Effort:** Medium (1 week).

### 21. Rich CLI Dashboard ✅
- **Current state:** `src/qts/monitoring/dashboard.py` — implemented with Rich `Live` layout, multiple panels (portfolio, trades, signals, regime, health, alerts).
- **Works now:** ✅ Yes.

### 22. Docker Compose ⚠️ Untested
- **Current state:** `docker/docker-compose.yml` exists with 4 services (timescaledb, redis, app, celery-worker). `docker/Dockerfile` has multi-stage build. **Not tested end-to-end.** The celery-worker references `qts.tasks` which doesn't exist.
- **What's needed:** Fix celery-worker reference, test full `docker-compose up`, verify inter-service connectivity.
- **Effort:** Small (1-2 days).

### 23. Alembic (DB Migrations) ❌
- **Current state:** `alembic` is in dependencies. `src/qts/db/__init__.py` mentions it in a docstring. **No `alembic.ini`, no `migrations/` directory, no migration files.**
- **What's needed:** `alembic init`, configure `env.py` to use our `Base.metadata`, create initial migration from existing SQLAlchemy models, test up/down migrations.
- **Effort:** Small (1-2 days).

---

## What Works Right Now (No API Keys Needed)

These features are fully functional today with zero external dependencies:

| Feature | How to use |
|---------|-----------|
| **Config management** | `from qts.config import get_settings` |
| **Technical indicators** | `from qts.signals.indicators import compute_rsi, compute_macd, ...` |
| **HMM regime detection** | `from qts.signals.regime import RegimeDetector` |
| **Signal pipeline** | `from qts.signals.pipeline import SignalPipeline` |
| **Combined alpha** | `from qts.signals.alpha import combined_alpha` |
| **SMA crossover backtest** | `python scripts/run_backtest.py` |
| **Custom backtest engine** | `from qts.simulation.backtest import BacktestEngine` |
| **Stress scenarios** | `from qts.simulation.scenario import StressScenarioRunner` |
| **Risk management** | `from qts.execution.risk import RiskManager` |
| **Order management** | `from qts.execution.order_manager import OrderManager` |
| **Trade attribution** | `from qts.analytics.attribution import AttributionEngine` |
| **VADER sentiment** | `from qts.nlp.vader import VaderAnalyzer` |
| **FinBERT sentiment** | `from qts.nlp.finbert import FinBERTAnalyzer` (auto-downloads model) |
| **GDELT processor** | `from qts.nlp.gdelt import GDELTProcessor` (needs pre-fetched data) |
| **Sentiment fusion** | `from qts.nlp.fusion import SentimentFusion` |
| **Ollama LLM queries** | `python scripts/test_ollama.py` (needs Ollama running) |
| **LLM debrief engine** | `from qts.oversight.debrief import DebriefEngine` |
| **Proposal management** | `from qts.oversight.proposals import ProposalManager` |
| **Monitoring dashboard** | `from qts.monitoring.dashboard import TradingDashboard` |
| **Health checks** | `from qts.monitoring.health import HealthChecker` |
| **Alerts** | `from qts.monitoring.alerts import AlertManager` |
| **Full test suite** | `make test` (346 tests, ~3.5 seconds) |

---

## What Needs API Keys

| Service | Key Type | How to Get | Cost | Set In `.env` As |
|---------|----------|-----------|------|-----------------|
| **Alpha Vantage** | API key | [alphavantage.co/support](https://www.alphavantage.co/support/#api-key) | Free (25 req/day) | `ALPHA_VANTAGE_API_KEY` |
| **Reddit** | OAuth2 app | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) | Free (60 req/min) | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` |
| **Binance** | API key+secret | [binance.com/en/my/settings/api-management](https://www.binance.com/en/my/settings/api-management) | Free (public data) | `BINANCE_API_KEY`, `BINANCE_API_SECRET` |
| **Alpaca** | API key+secret | [app.alpaca.markets](https://app.alpaca.markets/) | Free (paper trading) | Not yet in config |
| **Anthropic** | API key | [console.anthropic.com](https://console.anthropic.com/) | Pay-per-token | `ANTHROPIC_API_KEY` |

**Note:** StockTwits public endpoint and GDELT require no keys at all.

---

## Implementation TODO — Priority Order

### 🔴 P0 — Critical Path (needed for basic live operation)

- [ ] **Binance market data adapter** — `src/qts/data/market/binance_adapter.py`
  - REST client for historical klines (backfill)
  - WebSocket client for real-time trade stream + L2 orderbook
  - Implement `BarProvider` and `TickProvider` protocols
  - No API key needed for public data; key needed for trading
  - _Effort: 1 week_

- [ ] **Alembic migration setup**
  - `alembic init migrations`
  - Configure `env.py` to use existing `Base.metadata` from `db/tables.py`
  - Generate initial migration: `alembic revision --autogenerate -m "initial tables"`
  - Test `alembic upgrade head` on SQLite and PostgreSQL
  - _Effort: 1-2 days_

- [ ] **Celery task definitions** — `src/qts/tasks.py`
  - Create Celery app instance configured via `AppSettings.redis_url`
  - Define periodic tasks:
    - `refresh_sentiment` (every 5 min): run FinBERT on new headlines
    - `poll_gdelt` (every 15 min): fetch GDELT GKG, feed to processor
    - `poll_reddit` (every 15 min): fetch subreddit mentions
    - `run_nightly_attribution` (daily): classify failure modes
    - `run_debrief` (daily): LLM session analysis
  - Wire to `celery-worker` service in docker-compose
  - _Effort: 3-5 days_

- [ ] **GDELT HTTP client** — `src/qts/data/geopolitical/gdelt_client.py`
  - Fetch from GDELT GKG API (free, no key)
  - Parse response into `GeopoliticalEvent` objects
  - Feed into existing `GDELTProcessor`
  - _Effort: 2-3 days_

### 🟡 P1 — Important (needed for production quality)

- [ ] **hypothesis property tests**
  - `tests/unit/test_indicators_property.py`:
    - RSI output always in [0, 100] for any positive price series
    - MACD histogram == MACD line - signal line
    - BB position in [0, 1] when price between bands
    - ATR always non-negative
  - `tests/unit/test_alpha_property.py`:
    - combined_alpha always in [-1, 1] for any valid SignalSnapshot
  - `tests/unit/test_fusion_property.py`:
    - fused_sentiment always in [-1, 1]
  - `tests/unit/test_risk_property.py`:
    - circuit breaker always triggers when drawdown > limit
    - fill model never fills outside OHLC range
  - _Effort: 2-3 days_

- [ ] **Docker Compose end-to-end test**
  - Fix celery-worker service (needs `qts.tasks` module)
  - Test `docker-compose up` builds and starts all services
  - Verify app can connect to TimescaleDB and Redis
  - Run migrations via entrypoint
  - _Effort: 1-2 days_

- [ ] **Walk-forward backtesting** — `src/qts/simulation/walk_forward.py`
  - Train/validate/test splitting (70/15/15)
  - Rolling window parameter optimisation
  - Out-of-sample performance metrics
  - Monte Carlo permutation test for Sharpe significance
  - _Effort: 1 week_

- [ ] **Database schema auto-creation on startup**
  - Wire `Base.metadata.create_all()` or Alembic `upgrade head` into app startup
  - Ensure tables exist before first write
  - _Effort: 1 day_

### 🟢 P2 — Nice to Have (enhances the system)

- [ ] **NautilusTrader integration**
  - Create adapter wrapping NautilusTrader's `BacktestEngine` with `FillModel.REALISTIC`
  - Use NautilusTrader's built-in Binance integration for live trading
  - Map NautilusTrader events to our domain model (Bar, Fill, Position)
  - Consider: is the custom engine sufficient? NautilusTrader adds complexity.
  - _Effort: 2-3 weeks_

- [ ] **Alpaca equities adapter** — `src/qts/data/market/alpaca_adapter.py`
  - Add `alpaca-trade-api` to dependencies
  - Implement `BarProvider` for equity market data
  - Add `AlpacaSettings` to config
  - _Effort: 1 week_

- [ ] **hftbacktest integration** — `src/qts/simulation/hft_backtest.py`
  - Add to dependencies
  - Create adapter for tick-level backtesting
  - Only needed for sub-second HFT strategies
  - _Effort: 1-2 weeks_

- [ ] **Prometheus + Grafana monitoring**
  - Add `prometheus-client` dependency
  - Instrument: trade count, PnL, latency histograms, signal values
  - Expose `/metrics` endpoint
  - Add Prometheus + Grafana services to docker-compose
  - Create dashboard JSON (trades, equity curve, signal panel, health)
  - _Effort: 1 week_

- [ ] **Mean reversion strategy** — `src/qts/strategies/mean_reversion.py`
  - Planned in repo structure but never implemented
  - Entry on BB position < 0.2 (oversold) with sentiment confirmation
  - Exit on BB position > 0.5 (mean reversion target)
  - _Effort: 3-5 days_

### 🔵 P3 — Polish

- [ ] **Increase test coverage to 80%+**
  - Priority gaps: `execution/engine.py` (21%), `trade_logging/` (29-39%),
    `monitoring/dashboard.py` (32%), `strategies/momentum.py` (0%),
    `simulation/scenario.py` (0%), CLI modules (0%)
  - _Effort: 1 week_

- [ ] **mypy --strict pass**
  - Currently untested. Likely many type errors due to third-party libs.
  - Fix with type: ignore comments and stub files where needed.
  - _Effort: 2-3 days_

- [ ] **CI/CD pipeline** (GitHub Actions)
  - Lint (ruff) + format check (black) + type check (mypy) + test (pytest)
  - Coverage badge on README
  - Docker build test
  - _Effort: 1-2 days_

- [ ] **Binance historical data backfill script**
  - Download 2 years of 1m/5m klines via REST
  - Store as Parquet files for backtest consumption
  - _Effort: 2-3 days_

---

## Recommended Next Steps (First Sprint)

If you want to get to a **paper trading state** as fast as possible:

```
Week 1:  Binance adapter (public WS + REST) + Alembic migrations
Week 2:  Celery tasks + GDELT client + Docker Compose testing
Week 3:  Walk-forward backtesting + hypothesis property tests
Week 4:  Paper trading on Binance testnet (simulated fills, real prices)
```

If you want to **improve what already works** first:

```
Week 1:  hypothesis property tests + mypy --strict pass
Week 2:  Test coverage to 80% + CI/CD pipeline
Week 3:  GDELT client + Celery tasks
Week 4:  Binance adapter + Docker Compose
```

---

## API Key Quick-Start Checklist

Get these (all free) to unlock the full sentiment pipeline:

- [ ] Alpha Vantage: [Get free key](https://www.alphavantage.co/support/#api-key) → `ALPHA_VANTAGE_API_KEY`
- [ ] Reddit: [Create app](https://www.reddit.com/prefs/apps) (type: script) → `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET`
- [ ] Binance: [API management](https://www.binance.com/en/my/settings/api-management) → `BINANCE_API_KEY` + `BINANCE_API_SECRET`

Optional (paid):
- [ ] Anthropic: [Console](https://console.anthropic.com/) → `ANTHROPIC_API_KEY` (or use Ollama for free)
- [ ] Alpaca: [Sign up](https://app.alpaca.markets/) → not yet in config

No key needed:
- StockTwits (public API)
- GDELT (open data)
- FinBERT (HuggingFace, auto-downloads)
- Ollama (local)
