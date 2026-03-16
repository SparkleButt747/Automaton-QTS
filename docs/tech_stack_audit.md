# Tech Stack Audit & Implementation TODO

> **Generated:** 2026-03-14  
> **Status:** Post-Phase 8 — walk-forward backtesting, Celery tasks, Alembic, CI, property tests  
> **Tests:** 605 passing | Coverage: 61%

---

## Executive Summary

The project has a **solid core**: config management, signal engine, backtest engine, NLP
pipeline, LLM oversight, and monitoring dashboard are all implemented with real code and
passing tests. Several **planned integrations exist only as config fields or
dependency listings** — notably NautilusTrader, hftbacktest, Binance WebSocket, and Alpaca.
However, major infrastructure items that were previously missing are now implemented:
Celery task queue (6 tasks + beat schedule), Alembic migrations (configured, initial
migration tested up/down/up), hypothesis property tests (indicators, alpha, fusion, risk),
Prometheus/Grafana metrics instrumentation, walk-forward backtesting with Monte Carlo
significance testing, mean reversion strategy, and a GitHub Actions CI pipeline.

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
| 8 | **GDELT** | Geopolitical events | ✅ Client + processor | No | Free, no key | ✅ Yes |
| 9 | **Alpha Vantage** | News feed | ✅ Client implemented | Yes | 25 req/day (free) | ⚠️ Needs key |
| 10 | **Reddit / PRAW** | Social data | ✅ Client implemented | Yes (OAuth2) | 60 req/min | ⚠️ Needs key |
| 11 | **StockTwits** | Social sentiment | ✅ Client implemented | No (public endpoint) | 1000 req/hr | ✅ Yes |
| 12 | **Claude API** | LLM oversight | ✅ Client implemented | Yes | No free tier | ⚠️ Needs key |
| 13 | **Ollama** | Local LLM | ✅ Fully implemented | No | Free (local) | ✅ Yes |
| 14 | **SQLite** | Dev database | ✅ Configured | No | — | ✅ Yes |
| 15 | **PostgreSQL/TimescaleDB** | Prod database | ⚠️ Docker only | No | OSS | ⚠️ Docker req'd |
| 16 | **Celery + Redis** | Task queue | ✅ 6 tasks + beat schedule | No | OSS | ✅ Yes |
| 17 | **Pydantic** | Config management | ✅ Fully implemented | No | — | ✅ Yes |
| 18 | **pytest + pytest-asyncio** | Testing | ✅ 605 tests | No | — | ✅ Yes |
| 19 | **hypothesis** | Property testing | ✅ Property tests written | No | — | ✅ Yes |
| 20 | **Grafana + Prometheus** | Monitoring | ⚠️ Instrumented, not live-tested | No | OSS | ⚠️ Docker req'd |
| 21 | **Rich CLI dashboard** | Monitoring | ✅ Implemented | No | — | ✅ Yes |
| 22 | **Docker Compose** | Containerisation | ⚠️ Includes Prometheus + Grafana | No | — | ⚠️ Untested e2e |
| 23 | **Alembic** | DB migrations | ✅ Configured + migration tested | No | — | ✅ Yes |
| 24 | **Walk-forward backtesting** | Strategy validation | ✅ Engine + Monte Carlo test | No | — | ✅ Yes |
| 25 | **Mean reversion strategy** | Trading strategy | ✅ Implemented + tested | No | — | ✅ Yes |
| 26 | **GitHub Actions CI** | CI pipeline | ✅ Lint + typecheck + test | No | Free | ✅ Yes |

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

### 8. GDELT ✅
- **What it is:** Open database of global events, language, and tone. Updated every 15 minutes.
- **Current state:** `src/qts/data/geopolitical/gdelt_client.py` fetches from the GDELT GKG API (HTTP GET, CSV/JSON response) and parses into `GeopoliticalEvent` objects. `src/qts/nlp/gdelt.py` (`GDELTProcessor`) processes those events into conflict intensity scores. Both client and processor are fully implemented.
- **API key:** None required. GDELT is fully free and open.
- **Works now:** ✅ Yes — full pipeline from HTTP fetch to processed events.

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
- **Current state:** `docker-compose.yml` defines a `timescaledb` service (`timescale/timescaledb:latest-pg16`). DB engine in `src/qts/db/engine.py` supports PostgreSQL URLs. Alembic is configured and the initial migration has been tested up/down/up on SQLite.
- **What's needed:** Test Alembic `upgrade head` end-to-end against the Docker TimescaleDB service.
- **Works now:** ⚠️ Docker service defined; migrations work on SQLite, PostgreSQL path untested e2e.

### 16. Celery + Redis ✅
- **What it is:** Distributed task queue for async background jobs (sentiment updates, LLM calls, GDELT polling).
- **Current state:** `src/qts/tasks.py` — **fully implemented** with Celery app instance (configured via `AppSettings.redis_url`) and 6 task definitions with a beat schedule:
  - `refresh_sentiment` (every 5 min)
  - `poll_gdelt` (every 15 min)
  - `poll_reddit` (every 15 min)
  - `run_nightly_attribution` (daily)
  - `run_debrief` (daily)
  - `run_health_check` (every 1 min)
- **API key:** None (self-hosted).
- **Works now:** ✅ Yes — `celery-worker` service in docker-compose points to `qts.tasks`.

### 17. Pydantic Config ✅
- **Current state:** Fully implemented. `RiskLimits` (frozen, from JSON), `StrategyParams` (with weight sum validators), `AppSettings` (master class with lazy sub-settings). All tested.
- **Works now:** ✅ Yes.

### 18. pytest + pytest-asyncio ✅
- **Current state:** 605 tests passing. Coverage at 61%. Async tests use `asyncio_mode = "auto"`.
- **Works now:** ✅ Yes.

### 19. hypothesis (Property Testing) ✅
- **What it is:** Property-based testing library for generating random test cases.
- **Current state:** Property tests implemented across four test modules:
  - `tests/unit/test_indicators_property.py`: RSI always in [0, 100], MACD histogram = line - signal, BB position in [0, 1], ATR always non-negative.
  - `tests/unit/test_alpha_property.py`: combined_alpha always in [-1, 1] for any valid `SignalSnapshot`.
  - `tests/unit/test_fusion_property.py`: fused_sentiment always in [-1, 1].
  - `tests/unit/test_risk_property.py`: circuit breaker always triggers when drawdown > limit.
- **Works now:** ✅ Yes.

### 20. Grafana + Prometheus ⚠️ Instrumented, Not Live-Tested
- **What it is:** Metrics collection (Prometheus) + dashboarding (Grafana).
- **Current state:** `prometheus-client` dependency added. Key functions instrumented with counters and histograms. `/metrics` endpoint exposed. Prometheus and Grafana services added to `docker-compose.yml`. Dashboard JSON created. **Not yet verified end-to-end with live Docker services.**
- **The spec says:** "(optional)" — the Rich CLI dashboard is the required monitoring.
- **What's needed:** Run `docker-compose up` and verify Grafana scrapes Prometheus and dashboards load.
- **Works now:** ⚠️ Metrics code is in place; live-test against Docker stack pending.

### 21. Rich CLI Dashboard ✅
- **Current state:** `src/qts/monitoring/dashboard.py` — implemented with Rich `Live` layout, multiple panels (portfolio, trades, signals, regime, health, alerts).
- **Works now:** ✅ Yes.

### 22. Docker Compose ⚠️ Untested E2E
- **Current state:** `docker/docker-compose.yml` now includes 6 services: timescaledb, redis, app, celery-worker, prometheus, and grafana. `docker/Dockerfile` has multi-stage build. The celery-worker correctly references `qts.tasks`. **Not yet tested end-to-end.**
- **What's needed:** Run `docker-compose up` and verify inter-service connectivity (app ↔ TimescaleDB, app ↔ Redis, Prometheus ↔ app, Grafana ↔ Prometheus).
- **Effort:** Small (1-2 days).

### 23. Alembic (DB Migrations) ✅
- **Current state:** `alembic.ini` and `migrations/` directory created. `env.py` configured to use `Base.metadata` from `db/tables.py`. Initial migration generated (`alembic revision --autogenerate -m "initial tables"`). Tested `alembic upgrade head` → `alembic downgrade base` → `alembic upgrade head` cycle successfully on SQLite.
- **Works now:** ✅ Yes on SQLite. PostgreSQL e2e pending (see #15).

### 24. Walk-Forward Backtesting ✅
- **What it is:** Out-of-sample strategy validation with rolling train/validate/test windows.
- **Current state:** `src/qts/simulation/walk_forward.py` — **fully implemented** with:
  - Configurable train/validate/test splits (default 70/15/15)
  - Rolling window parameter optimisation
  - Out-of-sample performance metrics aggregation
  - Monte Carlo permutation test for Sharpe ratio significance
- **Works now:** ✅ Yes.

### 25. Mean Reversion Strategy ✅
- **What it is:** Counter-trend strategy using Bollinger Band position + sentiment confirmation.
- **Current state:** `src/qts/strategies/mean_reversion.py` — **fully implemented and tested**:
  - Entry on BB position < 0.2 (oversold) with sentiment confirmation
  - Exit on BB position > 0.5 (mean reversion target)
  - Full test suite in `tests/unit/test_mean_reversion.py`
- **Works now:** ✅ Yes.

### 26. GitHub Actions CI ✅
- **What it is:** Continuous integration pipeline for automated quality checks.
- **Current state:** `.github/workflows/ci.yml` — **fully implemented** with:
  - Lint (ruff) + format check (black)
  - Type check (mypy)
  - Full test suite (pytest) with coverage reporting
  - Runs on every push and pull request
- **Works now:** ✅ Yes.

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
| **Walk-forward backtesting** | `from qts.simulation.walk_forward import WalkForwardEngine` |
| **Mean reversion strategy** | `from qts.strategies.mean_reversion import MeanReversionStrategy` |
| **Stress scenarios** | `from qts.simulation.scenario import StressScenarioRunner` |
| **Risk management** | `from qts.execution.risk import RiskManager` |
| **Order management** | `from qts.execution.order_manager import OrderManager` |
| **Trade attribution** | `from qts.analytics.attribution import AttributionEngine` |
| **VADER sentiment** | `from qts.nlp.vader import VaderAnalyzer` |
| **FinBERT sentiment** | `from qts.nlp.finbert import FinBERTAnalyzer` (auto-downloads model) |
| **GDELT client + processor** | `from qts.data.geopolitical.gdelt_client import GDELTClient` |
| **Sentiment fusion** | `from qts.nlp.fusion import SentimentFusion` |
| **Celery task queue** | `celery -A qts.tasks worker` (requires Redis) |
| **DB migrations** | `alembic upgrade head` |
| **Ollama LLM queries** | `python scripts/test_ollama.py` (needs Ollama running) |
| **LLM debrief engine** | `from qts.oversight.debrief import DebriefEngine` |
| **Proposal management** | `from qts.oversight.proposals import ProposalManager` |
| **Monitoring dashboard** | `from qts.monitoring.dashboard import TradingDashboard` |
| **Health checks** | `from qts.monitoring.health import HealthChecker` |
| **Alerts** | `from qts.monitoring.alerts import AlertManager` |
| **Prometheus metrics** | Instrumented — expose via `/metrics` endpoint |
| **CI pipeline** | Runs automatically on push/PR via GitHub Actions |
| **Full test suite** | `make test` (605 tests, ~5 seconds) |

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

- [x] **Alembic migration setup** — ✅ Done
  - `alembic init migrations`
  - Configured `env.py` to use existing `Base.metadata` from `db/tables.py`
  - Generated initial migration: `alembic revision --autogenerate -m "initial tables"`
  - Tested `alembic upgrade head` / downgrade / upgrade on SQLite

- [x] **Celery task definitions** — ✅ Done (`src/qts/tasks.py`)
  - Celery app instance configured via `AppSettings.redis_url`
  - 6 periodic tasks defined with beat schedule
  - Wired to `celery-worker` service in docker-compose

- [x] **GDELT HTTP client** — ✅ Done (`src/qts/data/geopolitical/gdelt_client.py`)
  - Fetches from GDELT GKG API (free, no key)
  - Parses response into `GeopoliticalEvent` objects
  - Feeds into existing `GDELTProcessor`

### 🟡 P1 — Important (needed for production quality)

- [x] **hypothesis property tests** — ✅ Done
  - `tests/unit/test_indicators_property.py`, `test_alpha_property.py`, `test_fusion_property.py`, `test_risk_property.py`

- [ ] **Docker Compose end-to-end test**
  - Run `docker-compose up` and verify all 6 services start cleanly
  - Verify app can connect to TimescaleDB and Redis
  - Run Alembic migrations via entrypoint
  - Verify Prometheus scrapes `/metrics` and Grafana dashboards load
  - _Effort: 1-2 days_

- [x] **Walk-forward backtesting** — ✅ Done (`src/qts/simulation/walk_forward.py`)
  - Train/validate/test splitting (70/15/15)
  - Rolling window parameter optimisation
  - Monte Carlo permutation test for Sharpe significance

- [ ] **Database schema auto-creation on startup**
  - Wire `alembic upgrade head` (or `Base.metadata.create_all()`) into app startup
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

- [x] **Prometheus + Grafana monitoring** — ✅ Instrumented (live-test pending)
  - `prometheus-client` added; key functions instrumented
  - `/metrics` endpoint exposed
  - Prometheus + Grafana services added to docker-compose
  - Dashboard JSON created
  - _Remaining: verify end-to-end with live Docker stack_

- [x] **Mean reversion strategy** — ✅ Done (`src/qts/strategies/mean_reversion.py`)
  - Entry on BB position < 0.2 (oversold) with sentiment confirmation
  - Exit on BB position > 0.5 (mean reversion target)

### 🔵 P3 — Polish

- [ ] **Increase test coverage to 80%+**
  - Currently at 61%. Priority gaps: `execution/engine.py`, `trade_logging/`,
    `monitoring/dashboard.py`, `strategies/momentum.py`, `simulation/scenario.py`, CLI modules
  - _Effort: 1 week_

- [ ] **mypy --strict pass**
  - Currently untested under `--strict`. Likely many type errors due to third-party libs.
  - Fix with `type: ignore` comments and stub files where needed.
  - _Effort: 2-3 days_

- [x] **CI/CD pipeline** — ✅ Done (`.github/workflows/ci.yml`)
  - Lint (ruff) + format check (black) + type check (mypy) + test (pytest)
  - Runs on push and pull request

- [ ] **Binance historical data backfill script**
  - Download 2 years of 1m/5m klines via REST
  - Store as Parquet files for backtest consumption
  - _Effort: 2-3 days_

---

## Recommended Next Steps (First Sprint)

If you want to get to a **paper trading state** as fast as possible:

```
Week 1:  Binance adapter (public WS + REST) + Docker Compose e2e test
Week 2:  DB schema auto-creation on startup + Grafana live-test
Week 3:  Paper trading on Binance testnet (simulated fills, real prices)
Week 4:  Coverage to 80% + mypy --strict pass
```

If you want to **improve what already works** first:

```
Week 1:  mypy --strict pass + test coverage to 80%
Week 2:  Docker Compose e2e test + Grafana/Prometheus live verification
Week 3:  Binance adapter + DB startup wiring
Week 4:  Binance historical data backfill script
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
- GDELT (open data — client now implemented)
- FinBERT (HuggingFace, auto-downloads)
- Ollama (local)
