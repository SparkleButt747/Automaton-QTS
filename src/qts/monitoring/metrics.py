"""Prometheus metrics instrumentation for QTS."""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Info

from qts.models.base import TradeRecord

# ── Trade counters & histograms ───────────────────────────────────────────────

TRADES_TOTAL = Counter(
    "qts_trades_total",
    "Total trades executed",
    ["symbol", "direction", "outcome"],
)

TRADE_PNL = Histogram(
    "qts_trade_pnl_usd",
    "Trade PnL in USD",
    buckets=[-1000, -500, -100, -50, 0, 50, 100, 500, 1000],
)

# ── Portfolio gauges ──────────────────────────────────────────────────────────

PORTFOLIO_VALUE = Gauge(
    "qts_portfolio_value_usd",
    "Current portfolio value",
)

DAILY_DRAWDOWN = Gauge(
    "qts_daily_drawdown_pct",
    "Current daily drawdown percentage",
)

# ── Signal / sentiment gauges ─────────────────────────────────────────────────

SIGNAL_VALUE = Gauge(
    "qts_signal_value",
    "Current signal value",
    ["symbol", "signal_name"],
)

SENTIMENT_SCORE = Gauge(
    "qts_sentiment_score",
    "Current sentiment score",
    ["symbol", "source"],
)

# ── Regime info ───────────────────────────────────────────────────────────────

REGIME = Info(
    "qts_regime",
    "Current volatility regime",
)

# ── Latency histograms ────────────────────────────────────────────────────────

FEED_LATENCY = Histogram(
    "qts_feed_latency_seconds",
    "Market data feed latency",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

LLM_REQUEST_DURATION = Histogram(
    "qts_llm_request_duration_seconds",
    "LLM API call duration",
)

# ── Operational counters / gauges ─────────────────────────────────────────────

CIRCUIT_BREAKER_TRIPS = Counter(
    "qts_circuit_breaker_trips_total",
    "Circuit breaker trip count",
)

ACTIVE_POSITIONS = Gauge(
    "qts_active_positions",
    "Number of open positions",
)

PENDING_PROPOSALS = Gauge(
    "qts_pending_proposals",
    "Number of pending LLM proposals",
)

# ── Helper functions ──────────────────────────────────────────────────────────


def record_trade(trade: TradeRecord) -> None:
    """Record a completed trade in Prometheus metrics.

    Increments the trades counter (labelled by symbol, direction and outcome)
    and observes the trade PnL in the histogram.

    Args:
        trade: The completed :class:`~qts.models.base.TradeRecord` to record.
    """
    TRADES_TOTAL.labels(
        symbol=trade.symbol,
        direction=str(trade.direction),
        outcome=str(trade.outcome),
    ).inc()
    TRADE_PNL.observe(trade.pnl_usd)


def update_portfolio(value: float, drawdown: float) -> None:
    """Update portfolio-level Prometheus gauges.

    Args:
        value: Current total portfolio value in USD.
        drawdown: Current daily drawdown as a positive percentage (e.g. 1.5 for 1.5%).
    """
    PORTFOLIO_VALUE.set(value)
    DAILY_DRAWDOWN.set(drawdown)


def update_signal(symbol: str, name: str, value: float) -> None:
    """Update a labelled signal gauge.

    Args:
        symbol: Ticker symbol the signal belongs to.
        name: Human-readable name of the signal (e.g. ``"rsi"``).
        value: Current signal value.
    """
    SIGNAL_VALUE.labels(symbol=symbol, signal_name=name).set(value)


def update_sentiment(symbol: str, source: str, value: float) -> None:
    """Update a labelled sentiment gauge.

    Args:
        symbol: Ticker symbol the sentiment score belongs to.
        source: Data source (e.g. ``"twitter"``, ``"reddit"``).
        value: Current sentiment score.
    """
    SENTIMENT_SCORE.labels(symbol=symbol, source=source).set(value)


# ── HTTP server ───────────────────────────────────────────────────────────────


def start_metrics_server(port: int = 9090) -> None:
    """Start Prometheus metrics HTTP server on given port.

    Args:
        port: TCP port to bind.  Defaults to ``9090``.
    """
    from prometheus_client import start_http_server  # noqa: PLC0415

    start_http_server(port)
