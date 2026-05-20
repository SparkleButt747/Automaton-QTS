"""Unit tests for Prometheus metrics instrumentation."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from prometheus_client import REGISTRY

from qts.models.base import ExitReason, TradeDirection, TradeOutcome, TradeRecord, VolLevel
from qts.monitoring.metrics import (
    ACTIVE_POSITIONS,
    CIRCUIT_BREAKER_TRIPS,
    DAILY_DRAWDOWN,
    FEED_LATENCY,
    LLM_REQUEST_DURATION,
    PENDING_PROPOSALS,
    PORTFOLIO_VALUE,
    REGIME,
    SENTIMENT_SCORE,
    SIGNAL_VALUE,
    TRADE_PNL,
    TRADES_TOTAL,
    record_trade,
    start_metrics_server,
    update_portfolio,
    update_sentiment,
    update_signal,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_trade(
    symbol: str = "BTC-USD",
    direction: TradeDirection = TradeDirection.LONG,
    outcome: TradeOutcome = TradeOutcome.WIN,
    pnl_usd: float = 250.0,
) -> TradeRecord:
    now = datetime.now(UTC)
    return TradeRecord(
        trade_id=str(uuid.uuid4()),
        symbol=symbol,
        entry_time=now,
        exit_time=now,
        direction=direction,
        entry_price=50_000.0,
        exit_price=50_500.0,
        size=0.1,
        pnl_usd=pnl_usd,
        pnl_pct=0.01,
        slippage_usd=0.5,
        fees_usd=1.0,
        rsi_at_entry=55.0,
        macd_at_entry=0.0012,
        sentiment_at_entry=0.3,
        social_velocity_at_entry=0.5,
        gdelt_intensity_at_entry=0.2,
        vol_level_at_entry=VolLevel.LOW,
        combined_alpha_at_entry=0.4,
        params_version="v1",
        outcome=outcome,
        exit_reason=ExitReason.ALPHA_FLIP,
        failure_mode=None,
    )


# ── Metric existence ──────────────────────────────────────────────────────────


class TestMetricDefinitions:
    """All metric objects should be importable and non-None."""

    def test_trades_total_defined(self) -> None:
        assert TRADES_TOTAL is not None

    def test_trade_pnl_defined(self) -> None:
        assert TRADE_PNL is not None

    def test_portfolio_value_defined(self) -> None:
        assert PORTFOLIO_VALUE is not None

    def test_daily_drawdown_defined(self) -> None:
        assert DAILY_DRAWDOWN is not None

    def test_signal_value_defined(self) -> None:
        assert SIGNAL_VALUE is not None

    def test_sentiment_score_defined(self) -> None:
        assert SENTIMENT_SCORE is not None

    def test_regime_defined(self) -> None:
        assert REGIME is not None

    def test_feed_latency_defined(self) -> None:
        assert FEED_LATENCY is not None

    def test_llm_request_duration_defined(self) -> None:
        assert LLM_REQUEST_DURATION is not None

    def test_circuit_breaker_trips_defined(self) -> None:
        assert CIRCUIT_BREAKER_TRIPS is not None

    def test_active_positions_defined(self) -> None:
        assert ACTIVE_POSITIONS is not None

    def test_pending_proposals_defined(self) -> None:
        assert PENDING_PROPOSALS is not None

    def test_all_metrics_registered(self) -> None:
        """Verify the metric names are present in the Prometheus registry.

        Note: prometheus_client strips the ``_total`` suffix from Counter metric
        names when storing them in the registry (the suffix is re-appended at
        exposition time), so we check the base names for Counter metrics.
        """
        registered_names = {m.name for m in REGISTRY.collect()}
        # Counters are registered under their base name (without _total)
        expected = {
            "qts_trades",  # Counter — registered without _total
            "qts_trade_pnl_usd",
            "qts_portfolio_value_usd",
            "qts_daily_drawdown_pct",
            "qts_signal_value",
            "qts_sentiment_score",
            "qts_regime",
            "qts_feed_latency_seconds",
            "qts_llm_request_duration_seconds",
            "qts_circuit_breaker_trips",  # Counter — registered without _total
            "qts_active_positions",
            "qts_pending_proposals",
        }
        for name in expected:
            assert name in registered_names, f"Metric '{name}' not found in registry"


# ── record_trade ──────────────────────────────────────────────────────────────


class TestRecordTrade:
    def test_increments_counter(self) -> None:
        trade = _make_trade(
            symbol="ETH-USD", direction=TradeDirection.LONG, outcome=TradeOutcome.WIN
        )

        before = TRADES_TOTAL.labels(
            symbol="ETH-USD",
            direction="LONG",
            outcome="WIN",
        )._value.get()

        record_trade(trade)

        after = TRADES_TOTAL.labels(
            symbol="ETH-USD",
            direction="LONG",
            outcome="WIN",
        )._value.get()

        assert after == before + 1.0

    def test_increments_loss_counter(self) -> None:
        trade = _make_trade(
            symbol="SOL-USD",
            direction=TradeDirection.SHORT,
            outcome=TradeOutcome.LOSS,
            pnl_usd=-120.0,
        )

        before = TRADES_TOTAL.labels(
            symbol="SOL-USD",
            direction="SHORT",
            outcome="LOSS",
        )._value.get()

        record_trade(trade)

        after = TRADES_TOTAL.labels(
            symbol="SOL-USD",
            direction="SHORT",
            outcome="LOSS",
        )._value.get()

        assert after == before + 1.0

    def test_observes_pnl_histogram(self) -> None:
        """PnL histogram sum should increase after recording a trade."""
        trade = _make_trade(pnl_usd=99.0)

        before_sum = TRADE_PNL._sum.get()
        record_trade(trade)
        after_sum = TRADE_PNL._sum.get()

        assert after_sum == pytest.approx(before_sum + 99.0)


# ── update_portfolio ──────────────────────────────────────────────────────────


class TestUpdatePortfolio:
    def test_sets_portfolio_value(self) -> None:
        update_portfolio(value=100_000.0, drawdown=1.5)
        assert PORTFOLIO_VALUE._value.get() == pytest.approx(100_000.0)

    def test_sets_drawdown(self) -> None:
        update_portfolio(value=50_000.0, drawdown=2.75)
        assert DAILY_DRAWDOWN._value.get() == pytest.approx(2.75)

    def test_overwrites_previous_value(self) -> None:
        update_portfolio(value=200_000.0, drawdown=0.5)
        update_portfolio(value=195_000.0, drawdown=1.0)
        assert PORTFOLIO_VALUE._value.get() == pytest.approx(195_000.0)


# ── update_signal ─────────────────────────────────────────────────────────────


class TestUpdateSignal:
    def test_sets_labeled_gauge(self) -> None:
        update_signal(symbol="BTC-USD", name="rsi", value=67.3)
        gauge_val = SIGNAL_VALUE.labels(symbol="BTC-USD", signal_name="rsi")._value.get()
        assert gauge_val == pytest.approx(67.3)

    def test_different_labels_are_independent(self) -> None:
        update_signal(symbol="BTC-USD", name="macd", value=0.0042)
        update_signal(symbol="ETH-USD", name="macd", value=-0.0012)

        btc_val = SIGNAL_VALUE.labels(symbol="BTC-USD", signal_name="macd")._value.get()
        eth_val = SIGNAL_VALUE.labels(symbol="ETH-USD", signal_name="macd")._value.get()

        assert btc_val == pytest.approx(0.0042)
        assert eth_val == pytest.approx(-0.0012)


# ── update_sentiment ──────────────────────────────────────────────────────────


class TestUpdateSentiment:
    def test_sets_labeled_gauge(self) -> None:
        update_sentiment(symbol="BTC-USD", source="twitter", value=0.65)
        gauge_val = SENTIMENT_SCORE.labels(symbol="BTC-USD", source="twitter")._value.get()
        assert gauge_val == pytest.approx(0.65)

    def test_multiple_sources_independent(self) -> None:
        update_sentiment(symbol="BTC-USD", source="reddit", value=0.3)
        update_sentiment(symbol="BTC-USD", source="news", value=-0.1)

        reddit_val = SENTIMENT_SCORE.labels(symbol="BTC-USD", source="reddit")._value.get()
        news_val = SENTIMENT_SCORE.labels(symbol="BTC-USD", source="news")._value.get()

        assert reddit_val == pytest.approx(0.3)
        assert news_val == pytest.approx(-0.1)


# ── start_metrics_server ──────────────────────────────────────────────────────


class TestStartMetricsServer:
    def test_function_exists(self) -> None:
        """Verify the server helper is importable and callable."""
        assert callable(start_metrics_server)

    def test_accepts_port_argument(self) -> None:
        """Verify the function accepts a port keyword argument without error
        (we do NOT actually start the server in tests)."""
        import inspect

        sig = inspect.signature(start_metrics_server)
        assert "port" in sig.parameters
        assert sig.parameters["port"].default == 9090
