# Quant Trading System (QTS)

A multi-signal crypto/equity strategy engine combining technical analysis, sentiment fusion, and LLM-assisted decision support with human-in-the-loop oversight.

## Overview

QTS is a modular quantitative trading system designed for production-grade crypto and equity trading. It fuses signals from multiple sources — technical indicators (RSI, MACD, Bollinger Bands, Momentum), multi-source sentiment analysis (news, social media, geopolitical events), and LLM-based macro reasoning — into a composite score that drives position management.

### Key Features

- **Multi-signal fusion**: Configurable weighted combination of 4 technical + 1 sentiment signal
- **Sentiment pipeline**: VADER, transformer-based NLP, and LLM (Claude) for news/Reddit/GDELT
- **Risk controls**: Immutable risk limits with circuit breaker and drawdown protection
- **Human oversight**: All non-trivial trade proposals require approval before execution
- **Backtesting**: Historical simulation with identical signal logic to live trading
- **Async-first**: Built on asyncio with Celery for background tasks

## Architecture

```
src/qts/
├── config.py           # Pydantic Settings models (risk limits, strategy params)
├── data/               # Market, news, social, geopolitical data ingestion
├── models/             # SQLAlchemy ORM + Pydantic domain schemas
├── signals/            # RSI, MACD, BB, momentum, sentiment signal generators
├── strategies/         # Strategy base class + concrete implementations
├── execution/          # Order management, broker adapters, position tracking
├── nlp/                # Sentiment classifiers, LLM integration
├── simulation/         # Backtesting engine, paper trading
├── oversight/          # Approval workflows, human-in-the-loop controls
├── db/                 # SQLAlchemy engine, sessions, Alembic migrations
├── analytics/          # Performance metrics, Sharpe ratio, drawdown analysis
├── trade_logging/      # Structured trade and event logging
├── monitoring/         # Health checks, metrics, alerting
└── cli/                # Click-based CLI
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for TimescaleDB + Redis)
- (Optional) Binance API keys, Anthropic API key

### Setup

```bash
# 1. Clone and enter the project
cd quant-trading-system

# 2. Install dependencies
make dev-install

# 3. Configure environment
cp .env.example .env
# Edit .env and add your API keys

# 4. Start backing services
make docker-up

# 5. Check system status
qts status
```

### Development

```bash
make fmt        # Format code
make lint       # Run linter
make typecheck  # Run mypy
make test       # Run full test suite
```

### Running a Backtest

```bash
qts backtest --symbol BTCUSDT --start 2023-01-01 --end 2023-12-31
```

## Configuration

### Strategy Parameters (`config/params.json`)

Controls signal weights and entry/exit thresholds. Weights must sum to 1.0.

### Risk Limits (`config/risk_limits.json`)

**IMMUTABLE** — changes are rejected by the pre-commit hook unless you use `--force`. This file controls hard circuit breakers:
- `max_daily_drawdown_pct`: 2% maximum daily portfolio loss
- `max_position_size_pct`: 5% maximum single-position size
- `max_open_positions`: Maximum 5 concurrent open positions
- `circuit_breaker_cooldown_seconds`: 1 hour lockout after circuit break

## Risk Notice

This software is for research and educational purposes. Quantitative strategies carry significant financial risk. Always use the dry-run mode (`QTS_DRY_RUN=true`) until thoroughly validated. Past backtested performance does not guarantee future results.

## License

MIT
