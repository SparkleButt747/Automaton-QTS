"""Unit tests for qts.execution.engine.ExecutionEngine.

Verifies:
- process_bar buffers bars and calls signal pipeline
- process_bar returns [] when pipeline returns None (insufficient bars)
- process_bar returns [] when strategy returns None
- process_bar runs risk gate and returns [] when rejected
- process_bar returns a Fill when all gates pass
- process_bar handles strategy exceptions gracefully
- process_bar handles risk manager exceptions gracefully
- _execute_order generates a fill with correct commission
- run_live_loop iterates over an iterable provider
- reset_daily_state resets _daily_pnl and delegates to risk manager
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from qts.execution.engine import ExecutionEngine
from qts.models.base import Bar, Fill, Order, OrderSide, OrderType

# ── Helpers ───────────────────────────────────────────────────────────────────

_TS = datetime(2024, 1, 1, tzinfo=UTC)


def _make_bar(close=42000.0, symbol="BTCUSDT") -> Bar:
    return Bar(
        timestamp=_TS,
        symbol=symbol,
        open=close - 100,
        high=close + 200,
        low=close - 200,
        close=close,
        volume=100.0,
        bar_count=10,
    )


def _make_order(qty=0.5, price=42000.0) -> Order:
    return Order(
        order_id="ord-1",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=qty,
        price=price,
    )


def _mock_engine(approve=True):
    """Return an ExecutionEngine with mocked loggers and risk manager."""
    risk = MagicMock()
    risk.approve_order.return_value = approve

    tl = MagicMock()
    sl = MagicMock()

    engine = ExecutionEngine(
        risk_manager=risk,
        trade_logger=tl,
        signal_logger=sl,
        portfolio_value=100_000.0,
        paper_mode=True,
    )
    return engine, risk


# ── T-EXEC-1 ──────────────────────────────────────────────────────────────────


class TestProcessBarSignalPipeline:
    def test_returns_empty_when_pipeline_returns_none(self):
        """process_bar returns [] when the signal pipeline has insufficient data."""
        engine, _ = _mock_engine()

        pipeline = MagicMock()
        pipeline.compute.return_value = None

        strategy = MagicMock()

        fills = engine.process_bar(_make_bar(), strategy, pipeline)
        assert fills == []

    def test_buffers_bars_per_symbol(self):
        """process_bar accumulates bars in the buffer keyed by symbol."""
        engine, _ = _mock_engine()

        pipeline = MagicMock()
        pipeline.compute.return_value = None

        bar_btc = _make_bar(symbol="BTCUSDT")
        bar_eth = _make_bar(symbol="ETHUSDT")

        engine.process_bar(bar_btc, MagicMock(), pipeline)
        engine.process_bar(bar_eth, MagicMock(), pipeline)
        engine.process_bar(bar_btc, MagicMock(), pipeline)

        assert len(engine._bar_buffer["BTCUSDT"]) == 2
        assert len(engine._bar_buffer["ETHUSDT"]) == 1

    def test_logs_signal_snapshot(self):
        """process_bar logs the snapshot when pipeline returns one."""
        engine, _ = _mock_engine()

        snapshot = MagicMock()
        pipeline = MagicMock()
        pipeline.compute.return_value = snapshot

        strategy = MagicMock()
        strategy.generate_order.return_value = None

        engine.process_bar(_make_bar(), strategy, pipeline)
        engine.signal_logger.log_snapshot.assert_called_once_with(snapshot)


# ── T-EXEC-2 ──────────────────────────────────────────────────────────────────


class TestProcessBarStrategy:
    def test_returns_empty_when_strategy_returns_none(self):
        """process_bar returns [] when the strategy does not generate an order."""
        engine, _ = _mock_engine()

        pipeline = MagicMock()
        pipeline.compute.return_value = MagicMock()

        strategy = MagicMock()
        strategy.generate_order.return_value = None

        fills = engine.process_bar(_make_bar(), strategy, pipeline)
        assert fills == []

    def test_returns_empty_when_strategy_raises(self):
        """process_bar swallows strategy exceptions and returns []."""
        engine, _ = _mock_engine()

        pipeline = MagicMock()
        pipeline.compute.return_value = MagicMock()

        strategy = MagicMock()
        strategy.generate_order.side_effect = RuntimeError("strategy blew up")

        fills = engine.process_bar(_make_bar(), strategy, pipeline)
        assert fills == []

    def test_strategy_without_generate_order_returns_empty(self):
        """process_bar handles strategy objects without generate_order gracefully."""
        engine, _ = _mock_engine()

        pipeline = MagicMock()
        pipeline.compute.return_value = MagicMock()

        # No generate_order attribute
        strategy = object()

        fills = engine.process_bar(_make_bar(), strategy, pipeline)
        assert fills == []


# ── T-EXEC-3 ──────────────────────────────────────────────────────────────────


class TestProcessBarRiskGate:
    def test_returns_empty_when_risk_rejects(self):
        """process_bar returns [] when the risk manager rejects the order."""
        engine, risk = _mock_engine(approve=False)

        pipeline = MagicMock()
        pipeline.compute.return_value = MagicMock()

        strategy = MagicMock()
        strategy.generate_order.return_value = _make_order()

        fills = engine.process_bar(_make_bar(), strategy, pipeline)
        assert fills == []
        risk.approve_order.assert_called_once()

    def test_returns_empty_when_risk_raises(self):
        """process_bar returns [] when the risk manager raises an exception."""
        engine, risk = _mock_engine()
        risk.approve_order.side_effect = RuntimeError("risk error")

        pipeline = MagicMock()
        pipeline.compute.return_value = MagicMock()

        strategy = MagicMock()
        strategy.generate_order.return_value = _make_order()

        fills = engine.process_bar(_make_bar(), strategy, pipeline)
        assert fills == []

    def test_returns_fill_when_approved(self):
        """process_bar returns a Fill when risk approves the order."""
        engine, _ = _mock_engine(approve=True)

        pipeline = MagicMock()
        pipeline.compute.return_value = MagicMock()

        strategy = MagicMock()
        strategy.generate_order.return_value = _make_order(qty=0.5, price=42000.0)

        bar = _make_bar(close=42000.0)
        fills = engine.process_bar(bar, strategy, pipeline)

        assert len(fills) == 1
        assert isinstance(fills[0], Fill)
        assert fills[0].symbol == "BTCUSDT"
        assert fills[0].price == 42000.0
        assert fills[0].quantity == 0.5


# ── T-EXEC-4 ──────────────────────────────────────────────────────────────────


class TestExecuteOrder:
    def test_fill_has_correct_commission(self):
        """_execute_order charges 0.1% taker fee on fill value."""
        engine, _ = _mock_engine()
        bar = _make_bar(close=1000.0)
        order = _make_order(qty=2.0, price=1000.0)

        fills = engine._execute_order(order, bar)
        assert len(fills) == 1
        expected_commission = 1000.0 * 2.0 * 0.001  # 2.0
        assert fills[0].commission == pytest.approx(expected_commission)

    def test_fill_price_equals_bar_close(self):
        """_execute_order uses bar.close as the fill price."""
        engine, _ = _mock_engine()
        bar = _make_bar(close=50000.0)
        order = _make_order(qty=0.1, price=49000.0)

        fills = engine._execute_order(order, bar)
        assert fills[0].price == 50000.0

    def test_fill_has_zero_slippage(self):
        """_execute_order sets slippage=0.0 in paper mode."""
        engine, _ = _mock_engine()
        fills = engine._execute_order(_make_order(), _make_bar())
        assert fills[0].slippage == 0.0


# ── T-EXEC-5 ──────────────────────────────────────────────────────────────────


class TestRunLiveLoop:
    def test_iterates_over_iterable_provider(self):
        """run_live_loop processes each bar from an iterable provider."""
        engine, _ = _mock_engine()

        pipeline = MagicMock()
        pipeline.compute.return_value = None  # no fills needed

        bars = [_make_bar(), _make_bar(), _make_bar()]
        engine.run_live_loop(bars, MagicMock(), pipeline)

        # All three bars should be in the buffer
        assert len(engine._bar_buffer["BTCUSDT"]) == 3

    def test_stops_gracefully_on_empty_iterable(self):
        """run_live_loop handles an empty iterable without raising."""
        engine, _ = _mock_engine()
        engine.run_live_loop([], MagicMock(), MagicMock())


# ── T-EXEC-6 ──────────────────────────────────────────────────────────────────


class TestResetDailyState:
    def test_resets_daily_pnl(self):
        """reset_daily_state sets _daily_pnl back to zero."""
        engine, _ = _mock_engine()
        engine._daily_pnl = 500.0
        engine.reset_daily_state()
        assert engine._daily_pnl == 0.0

    def test_delegates_to_risk_manager_if_method_present(self):
        """reset_daily_state calls risk_manager.reset_daily_state() when available."""
        engine, risk = _mock_engine()
        engine.reset_daily_state()
        risk.reset_daily_state.assert_called_once()
