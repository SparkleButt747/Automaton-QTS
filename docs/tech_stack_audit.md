# Tech Stack Audit & Implementation TODO

> **Generated:** 2026-03-14
> **Updated:** 2026-03-16
> **Status:** Post-Phase 9 — all planned integrations complete, Docker e2e verified
> **Tests:** 944 passing (4 skipped) | Coverage: 81% | mypy --strict: 0 errors

---

## Executive Summary

The project is **feature-complete** against the original tech stack plan. All previously
unimplemented integrations — Binance WebSocket+REST, Alpaca equities, NautilusTrader,
hftbacktest — now have working adapter code with tests. The Docker Compose stack has been
fixed and verified end-to-end (6 services: TimescaleDB, Redis, app, Celery worker,
Prometheus, Grafana). Database schema auto-creation is wired into the CLI startup.
Test coverage reached 81% (up from 61%) and mypy --strict passes with 0 errors across
67 source files.

---

## Tech Stack Status Matrix

| # | Technology | Planned Role | Status | API Key? | Free Tier? | Functional Now? |
|---|-----------|-------------|--------|----------|------------|-----------------|
| 1 | **Python 3.11+** | Language | ✅ Fully implemented | No | — | ✅ Yes |
| 2 | **NautilusTrader** | Trading engine | ✅ Adapter implemented | No | OSS (LGPL) | ✅ Yes (`pip install qts[trading]`) |
| 3 | **hftbacktest** | HFT backtesting | ✅ Adapter implemented | No | OSS (Apache-2.0) | ✅ Yes (`pip install qts[hft]`) |
| 4 | **Binance WS+REST** | Crypto market data | ✅ Full adapter | Yes (for private) | Yes (public data free) | ✅ Yes |
| 5 | **Alpaca** | Equity market data | ✅ Full adapter | Yes | Yes (paper trading free) | ✅ Yes (needs key) |
| 6 | **FinBERT** | News sentiment | ✅ Implemented | No | Free HF model | ✅ Yes (lazy load) |
| 7 | **VADER** | Social sentiment | ✅ Implemented | No | Free package | ✅ Yes |
| 8 | **GDELT** | Geopolitical events | ✅ Client + processor | No | Free, no key | ✅ Yes |
| 9 | **Alpha Vantage** | News feed | ✅ Client implemented | Yes | 25 req/day (free) | ⚠️ Needs key |
| 10 | **Reddit / PRAW** | Social data | ✅ Client implemented | Yes (OAuth2) | 60 req/min | ⚠️ Needs key |
| 11 | **StockTwits** | Social sentiment | ✅ Client implemented | No (public endpoint) | 1000 req/hr | ✅ Yes |
| 12 | **Claude API** | LLM oversight | ✅ Client implemented | Yes | No free tier | ⚠️ Needs key |
| 13 | **Ollama** | Local LLM | ✅ Fully implemented | No | Free (local) | ✅ Yes |
| 14 | **SQLite** | Dev database | ✅ Configured | No | — | ✅ Yes |
| 15 | **PostgreSQL/TimescaleDB** | Prod database | ✅ Docker verified | No | OSS | ✅ Yes (Docker) |
| 16 | **Celery + Redis** | Task queue | ✅ 6 tasks + beat schedule | No | OSS | ✅ Yes |
| 17 | **Pydantic** | Config management | ✅ Fully implemented | No | — | ✅ Yes |
| 18 | **pytest + pytest-asyncio** | Testing | ✅ 944 tests | No | — | ✅ Yes |
| 19 | **hypothesis** | Property testing | ✅ Property tests written | No | — | ✅ Yes |
| 20 | **Grafana + Prometheus** | Monitoring | ✅ Docker verified e2e | No | OSS | ✅ Yes (Docker) |
| 21 | **Rich CLI dashboard** | Monitoring | ✅ Implemented | No | — | ✅ Yes |
| 22 | **Docker Compose** | Containerisation | ✅ 6 services, e2e tested | No | — | ✅ Yes |
| 23 | **Alembic** | DB migrations | ✅ Works on SQLite + PostgreSQL | No | — | ✅ Yes |
| 24 | **Walk-forward backtesting** | Strategy validation | ✅ Engine + Monte Carlo test | No | — | ✅ Yes |
| 25 | **Mean reversion strategy** | Trading strategy | ✅ Implemented + tested | No | — | ✅ Yes |
| 26 | **GitHub Actions CI** | CI pipeline | ✅ Lint + typecheck + test | No | Free | ✅ Yes |

**Legend:** ✅ Working now | ⚠️ Partial/needs config | ❌ Not implemented

---

## Detailed Assessment Per Technology

### 1. Python 3.11+ ✅
- **Status:** Fully functional. Project runs on Python 3.14 (tested).
- **Nothing to do.**

### 2. NautilusTrader ✅
- **What it is:** Event-driven trading framework with a Rust core and Python API. Supports live trading with Binance, Interactive Brokers, etc.
- **Current state:** `src/qts/simulation/nautilus_adapter.py` — **bridge adapter implemented** with:
  - `NautilusBacktestAdapter` wrapping NautilusTrader's `BacktestEngine` with configurable fill model (`REALISTIC` or `IMMEDIATE`)
  - Bar/Fill/Position conversion functions between QTS and NautilusTrader domain models
  - `_NautilusStrategyBridge` for wrapping QTS strategies in NautilusTrader's strategy interface
  - Lazy import guard — module works without `nautilus_trader` installed, raises clear `ImportError` on use
  - 14 tests (11 pass, 3 skipped without `nautilus_trader`)
- **Install:** `pip install qts[trading]`
- **Works now:** ✅ Yes (with optional dependency).

### 3. hftbacktest ✅
- **What it is:** Rust-based HFT backtesting library with queue-position-aware, tick-level simulation.
- **Current state:** `src/qts/simulation/hft_backtest.py` — **adapter implemented** with:
  - `HFTBacktestAdapter` bridging to `HashMapMarketDepthBacktest`
  - `convert_ticks_to_hft_format()` converting QTS `Tick` objects to hftbacktest's 6-column NumPy format
  - `load_hft_data()` for `.npy`/`.npz` files
  - `scripts/convert_ticks_to_hft.py` CLI for CSV/Parquet → hftbacktest format
  - Lazy import guard with clear install instructions
  - 15 tests (14 pass, 1 skipped without `hftbacktest`)
- **Install:** `pip install qts[hft]`
- **Works now:** ✅ Yes (with optional dependency).

### 4. Binance WebSocket + REST ✅
- **What it is:** Real-time crypto market data and order execution.
- **Current state:** `src/qts/data/market/binance_adapter.py` — **fully implemented** with:
  - `BinanceBarAdapter` — REST `/api/v3/klines` with pagination + WebSocket `@kline_<interval>` streams
  - `BinanceTickAdapter` — WebSocket `@trade` streams
  - `BinanceOrderBookAdapter` — REST `/api/v3/depth` snapshots + WebSocket `@depth<levels>@100ms`
  - Exponential backoff on HTTP 429 rate limits
  - 37 unit tests covering REST parsing, error handling, and edge cases
- **Backfill script:** `scripts/backfill_binance.py` downloads historical klines to Parquet files
- **Can start without API key:** Yes, for public market data only.
- **Works now:** ✅ Yes.

### 5. Alpaca ✅
- **What it is:** Commission-free equity/crypto trading API with market data.
- **Current state:** `src/qts/data/market/alpaca_adapter.py` — **fully implemented** with:
  - `AlpacaBarAdapter` implementing `BarProvider` protocol via raw `httpx` (no heavy `alpaca-trade-api` dependency)
  - REST historical bars with automatic pagination via `next_page_token`
  - WebSocket subscription for real-time bars
  - `AlpacaSettings` added to config with `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ALPACA_BASE_URL`, `ALPACA_DATA_URL`
  - 29 unit tests
- **API key:** Required (free account for paper trading).
- **Works now:** ✅ Yes (needs API key).

### 6. FinBERT ✅
- **What it is:** Fine-tuned BERT model for financial sentiment analysis (HuggingFace).
- **Current state:** `src/qts/nlp/finbert.py` — **fully implemented** with lazy model loading, batch inference, exponential decay function, and `FinBERTProtocol` for DI. Falls back gracefully if `transformers`/`torch` not installed.
- **API key:** None. Model downloads from HuggingFace Hub on first use (~400MB).
- **Cost:** Free. Runs on CPU (~400ms/article).
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

### 15. PostgreSQL / TimescaleDB ✅
- **Current state:** Docker Compose stack verified end-to-end. `docker/docker-compose.yml` defines a `timescaledb` service (`timescale/timescaledb:latest-pg15`). DB engine in `src/qts/db/engine.py` supports PostgreSQL URLs with `asyncpg`. Alembic `env.py` handles async driver URL conversion. `ensure_tables()` auto-creates schema on startup.
- **Works now:** ✅ Yes (via Docker).

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
- **Current state:** Fully implemented. `RiskLimits` (frozen, from JSON), `StrategyParams` (with weight sum validators), `AppSettings` (master class with lazy sub-settings including `AlpacaSettings`). All tested.
- **Works now:** ✅ Yes.

### 18. pytest + pytest-asyncio ✅
- **Current state:** 944 tests passing (4 skipped). Coverage at 81%. Async tests use `asyncio_mode = "auto"`.
- **Works now:** ✅ Yes.

### 19. hypothesis (Property Testing) ✅
- **What it is:** Property-based testing library for generating random test cases.
- **Current state:** Property tests implemented across four test modules:
  - `tests/unit/test_indicators_property.py`: RSI always in [0, 100], MACD histogram = line - signal, BB position in [0, 1], ATR always non-negative.
  - `tests/unit/test_alpha_property.py`: combined_alpha always in [-1, 1] for any valid `SignalSnapshot`.
  - `tests/unit/test_fusion_property.py`: fused_sentiment always in [-1, 1].
  - `tests/unit/test_risk_property.py`: circuit breaker always triggers when drawdown > limit.
- **Works now:** ✅ Yes.

### 20. Grafana + Prometheus ✅
- **What it is:** Metrics collection (Prometheus) + dashboarding (Grafana).
- **Current state:** `prometheus-client` dependency added. Key functions instrumented with counters and histograms. `/metrics` endpoint exposed on port 8080. Prometheus and Grafana services in `docker-compose.yml`. **Docker e2e verified** — Prometheus scrapes app, Grafana loads.
- **Works now:** ✅ Yes (Docker).

### 21. Rich CLI Dashboard ✅
- **Current state:** `src/qts/monitoring/dashboard.py` — implemented with Rich `Live` layout, multiple panels (portfolio, trades, signals, regime, health, alerts).
- **Works now:** ✅ Yes.

### 22. Docker Compose ✅
- **Current state:** `docker/docker-compose.yml` includes 6 services: timescaledb, redis, app, celery-worker, prometheus, and grafana. `docker/Dockerfile` has multi-stage build with entrypoint that auto-runs `ensure_tables()`. **Verified end-to-end** — all services start, inter-service connectivity confirmed (app ↔ TimescaleDB, app ↔ Redis, Prometheus ↔ app, Grafana ↔ Prometheus).
- **E2E test script:** `scripts/docker_e2e_test.sh`
- **Works now:** ✅ Yes.

### 23. Alembic (DB Migrations) ✅
- **Current state:** `alembic.ini` and `migrations/` directory created. `env.py` configured to use `Base.metadata` from `db/tables.py` with `DATABASE_URL` environment variable override. Handles async driver URL conversion (`+asyncpg` → sync). Tested on both SQLite and PostgreSQL (via Docker).
- **Works now:** ✅ Yes.

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
| **NautilusTrader backtest** | `from qts.simulation.nautilus_adapter import NautilusBacktestAdapter` (needs `qts[trading]`) |
| **HFT tick-level backtest** | `from qts.simulation.hft_backtest import HFTBacktestAdapter` (needs `qts[hft]`) |
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
| **Binance market data** | `from qts.data.market.binance_adapter import BinanceBarAdapter` (public data, no key) |
| **Binance data backfill** | `python scripts/backfill_binance.py --symbol BTCUSDT --start 2024-01-01` |
| **HFT data conversion** | `python scripts/convert_ticks_to_hft.py --input ticks.csv --output ticks.npz` |
| **Celery task queue** | `celery -A qts.tasks worker` (requires Redis) |
| **DB migrations** | `alembic upgrade head` |
| **DB auto-creation** | Built into CLI startup (`qts` command auto-runs `ensure_tables()`) |
| **Ollama LLM queries** | `python scripts/test_ollama.py` (needs Ollama running) |
| **LLM debrief engine** | `from qts.oversight.debrief import DebriefEngine` |
| **Proposal management** | `from qts.oversight.proposals import ProposalManager` |
| **Monitoring dashboard** | `from qts.monitoring.dashboard import TradingDashboard` |
| **Health checks** | `from qts.monitoring.health import HealthChecker` |
| **Alerts** | `from qts.monitoring.alerts import AlertManager` |
| **Prometheus metrics** | Instrumented — expose via `/metrics` endpoint on port 8080 |
| **Docker full stack** | `docker compose -f docker/docker-compose.yml up -d` (6 services) |
| **Docker e2e test** | `bash scripts/docker_e2e_test.sh` |
| **CI pipeline** | Runs automatically on push/PR via GitHub Actions |
| **Full test suite** | `pytest` (944 tests, ~15 seconds) |

---

## What Needs API Keys

| Service | Key Type | How to Get | Cost | Set In `.env` As |
|---------|----------|-----------|------|-----------------|
| **Alpha Vantage** | API key | [alphavantage.co/support](https://www.alphavantage.co/support/#api-key) | Free (25 req/day) | `ALPHA_VANTAGE_API_KEY` |
| **Reddit** | OAuth2 app | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) | Free (60 req/min) | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET` |
| **Binance** | API key+secret | [binance.com/en/my/settings/api-management](https://www.binance.com/en/my/settings/api-management) | Free (public data) | `BINANCE_API_KEY`, `BINANCE_API_SECRET` |
| **Alpaca** | API key+secret | [app.alpaca.markets](https://app.alpaca.markets/) | Free (paper trading) | `ALPACA_API_KEY`, `ALPACA_API_SECRET` |
| **Anthropic** | API key | [console.anthropic.com](https://console.anthropic.com/) | Pay-per-token | `ANTHROPIC_API_KEY` |

**Note:** StockTwits public endpoint and GDELT require no keys at all.

---

## Implementation TODO — All Complete ✅

### 🔴 P0 — Critical Path

- [x] **Binance market data adapter** — ✅ Done (`src/qts/data/market/binance_adapter.py`)
  - REST + WebSocket for klines, trades, and order book
  - Implements `BarProvider`, `TickProvider`, and `OrderBookProvider` protocols
  - 37 unit tests

- [x] **Alembic migration setup** — ✅ Done
  - Configured `env.py` with `DATABASE_URL` override and async driver conversion
  - Tested on both SQLite and PostgreSQL

- [x] **Celery task definitions** — ✅ Done (`src/qts/tasks.py`)
  - 6 periodic tasks with beat schedule

- [x] **GDELT HTTP client** — ✅ Done (`src/qts/data/geopolitical/gdelt_client.py`)

### 🟡 P1 — Important

- [x] **hypothesis property tests** — ✅ Done

- [x] **Docker Compose end-to-end test** — ✅ Done
  - All 6 services verified: TimescaleDB, Redis, app, Celery worker, Prometheus, Grafana
  - E2E test script: `scripts/docker_e2e_test.sh`
  - Fixed: Prometheus volume path, scrape target port, Dockerfile entrypoint, Alembic env.py

- [x] **Walk-forward backtesting** — ✅ Done (`src/qts/simulation/walk_forward.py`)

- [x] **Database schema auto-creation on startup** — ✅ Done
  - `ensure_tables()` in `src/qts/db/engine.py`
  - Wired into CLI group callback with `--skip-db-init` opt-out
  - Handles SQLite → aiosqlite URL conversion

### 🟢 P2 — Nice to Have

- [x] **NautilusTrader integration** — ✅ Done (`src/qts/simulation/nautilus_adapter.py`)
  - Bridge adapter with `NautilusBacktestAdapter`
  - Bar/Fill/Position conversion functions
  - Lazy import guard — works without dependency
  - 14 tests (11 pass, 3 skipped without nautilus_trader)

- [x] **Alpaca equities adapter** — ✅ Done (`src/qts/data/market/alpaca_adapter.py`)
  - `AlpacaBarAdapter` via raw httpx (no heavy alpaca-trade-api dependency)
  - `AlpacaSettings` in config with lazy loading
  - 29 unit tests

- [x] **hftbacktest integration** — ✅ Done (`src/qts/simulation/hft_backtest.py`)
  - `HFTBacktestAdapter` with tick-to-numpy conversion
  - `scripts/convert_ticks_to_hft.py` data conversion CLI
  - 15 tests (14 pass, 1 skipped without hftbacktest)

- [x] **Prometheus + Grafana monitoring** — ✅ Done and verified via Docker e2e

- [x] **Mean reversion strategy** — ✅ Done (`src/qts/strategies/mean_reversion.py`)

### 🔵 P3 — Polish

- [x] **Increase test coverage to 80%+** — ✅ Done (81%)
  - 944 tests (up from 605)
  - Key modules now covered: scenario, momentum, approval_cli, trade_logger, signal_logger, execution engine, order_manager, db repository, db engine, CLI, alpha_vantage, stocktwits, reddit

- [x] **mypy --strict pass** — ✅ Done (0 errors, 67 files)
  - Added type annotations across all source files
  - Fixed stale type: ignore comments
  - Added third-party module overrides

- [x] **CI/CD pipeline** — ✅ Done (`.github/workflows/ci.yml`)

- [x] **Binance historical data backfill script** — ✅ Done (`scripts/backfill_binance.py`)
  - Click CLI with progress bars (rich)
  - Paginates Binance klines API
  - Saves to Parquet with correct dtypes

---

## API Key Quick-Start Checklist

Get these (all free) to unlock the full sentiment pipeline:

- [ ] Alpha Vantage: [Get free key](https://www.alphavantage.co/support/#api-key) → `ALPHA_VANTAGE_API_KEY`
- [ ] Reddit: [Create app](https://www.reddit.com/prefs/apps) (type: script) → `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET`
- [ ] Binance: [API management](https://www.binance.com/en/my/settings/api-management) → `BINANCE_API_KEY` + `BINANCE_API_SECRET`
- [ ] Alpaca: [Sign up](https://app.alpaca.markets/) → `ALPACA_API_KEY` + `ALPACA_API_SECRET`

Optional (paid):
- [ ] Anthropic: [Console](https://console.anthropic.com/) → `ANTHROPIC_API_KEY` (or use Ollama for free)

No key needed:
- StockTwits (public API)
- GDELT (open data)
- FinBERT (HuggingFace, auto-downloads)
- Ollama (local)
