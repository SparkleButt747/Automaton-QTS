"""Unit tests for qts.execution.risk.

Verifies:
- Position size enforcement
- Max open positions enforcement
- Daily drawdown + circuit breaker logic
- Cooldown expiry
- approve_order combines all checks
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from qts.config import RiskLimits
from qts.execution.risk import CircuitBreakerError, RiskManager
from qts.models.base import Position, TradeDirection


def _make_limits(
    max_drawdown: float = 0.02,
    max_position_pct: float = 0.05,
    max_positions: int = 3,
    cooldown: int = 3600,
) -> RiskLimits:
    return RiskLimits(
        max_daily_drawdown_pct=max_drawdown,
        max_position_size_pct=max_position_pct,
        max_open_positions=max_positions,
        circuit_breaker_cooldown_seconds=cooldown,
        sentiment_signal_max_scalar=2.0,
    )


def _make_position(symbol: str = "BTCUSDT") -> Position:
    return Position(
        symbol=symbol,
        direction=TradeDirection.LONG,
        entry_price=100.0,
        quantity=1.0,
        entry_time=datetime(2024, 1, 1),
    )


# ── Position size ─────────────────────────────────────────────────────────────


class TestCheckPositionSize:
    def test_within_limit_allowed(self) -> None:
        rm = RiskManager(_make_limits(max_position_pct=0.05))
        assert rm.check_position_size(4_000.0, 100_000.0) is True  # 4% < 5%

    def test_exactly_at_limit_allowed(self) -> None:
        rm = RiskManager(_make_limits(max_position_pct=0.05))
        assert rm.check_position_size(5_000.0, 100_000.0) is True  # exactly 5%

    def test_above_limit_rejected(self) -> None:
        rm = RiskManager(_make_limits(max_position_pct=0.05))
        assert rm.check_position_size(6_000.0, 100_000.0) is False  # 6% > 5%

    def test_zero_portfolio_value_rejected(self) -> None:
        rm = RiskManager(_make_limits())
        assert rm.check_position_size(1_000.0, 0.0) is False


# ── Max positions ─────────────────────────────────────────────────────────────


class TestCheckMaxPositions:
    def test_empty_positions_allowed(self) -> None:
        rm = RiskManager(_make_limits(max_positions=3))
        assert rm.check_max_positions([]) is True

    def test_below_limit_allowed(self) -> None:
        rm = RiskManager(_make_limits(max_positions=3))
        positions = [_make_position("A"), _make_position("B")]
        assert rm.check_max_positions(positions) is True

    def test_at_limit_rejected(self) -> None:
        rm = RiskManager(_make_limits(max_positions=3))
        positions = [_make_position("A"), _make_position("B"), _make_position("C")]
        assert rm.check_max_positions(positions) is False

    def test_above_limit_rejected(self) -> None:
        rm = RiskManager(_make_limits(max_positions=2))
        positions = [_make_position("A"), _make_position("B"), _make_position("C")]
        assert rm.check_max_positions(positions) is False


# ── Daily drawdown & circuit breaker ─────────────────────────────────────────


class TestCheckDailyDrawdown:
    def test_no_loss_allowed(self) -> None:
        rm = RiskManager(_make_limits(max_drawdown=0.02))
        assert rm.check_daily_drawdown(0.0, 100_000.0) is True

    def test_small_loss_allowed(self) -> None:
        rm = RiskManager(_make_limits(max_drawdown=0.02))
        # 1% loss, limit is 2%
        assert rm.check_daily_drawdown(-1_000.0, 100_000.0) is True

    def test_loss_at_limit_trips_breaker(self) -> None:
        rm = RiskManager(_make_limits(max_drawdown=0.02))
        # Exactly 2% loss
        result = rm.check_daily_drawdown(-2_000.0, 100_000.0)
        assert result is False
        assert rm._halted is True

    def test_loss_above_limit_trips_breaker(self) -> None:
        rm = RiskManager(_make_limits(max_drawdown=0.02))
        result = rm.check_daily_drawdown(-5_000.0, 100_000.0)
        assert result is False
        assert rm._halted is True

    def test_tripped_once_stays_tripped(self) -> None:
        rm = RiskManager(_make_limits(max_drawdown=0.02, cooldown=3600))
        rm.check_daily_drawdown(-3_000.0, 100_000.0)  # trip
        # Second call with small loss should still be blocked (halted)
        rm.check_daily_drawdown(-100.0, 100_000.0)
        assert rm._halted is True


class TestCircuitBreaker:
    def test_halted_after_trip(self) -> None:
        rm = RiskManager(_make_limits(max_drawdown=0.02, cooldown=3600))
        rm.check_daily_drawdown(-3_000.0, 100_000.0)
        assert rm.is_halted() is True

    def test_cooldown_lifts_halt(self) -> None:
        rm = RiskManager(_make_limits(max_drawdown=0.02, cooldown=60))
        rm.check_daily_drawdown(-3_000.0, 100_000.0)
        assert rm._halted is True
        # Simulate 61 seconds later
        future = datetime.now(UTC) + timedelta(seconds=61)
        assert rm.is_halted(future) is False

    def test_cooldown_not_yet_expired(self) -> None:
        rm = RiskManager(_make_limits(max_drawdown=0.02, cooldown=3600))
        rm.check_daily_drawdown(-3_000.0, 100_000.0)
        # 10 seconds after trip
        future = datetime.now(UTC) + timedelta(seconds=10)
        assert rm.is_halted(future) is True

    def test_reset_lifts_halt(self) -> None:
        rm = RiskManager(_make_limits(max_drawdown=0.02, cooldown=3600))
        rm.check_daily_drawdown(-3_000.0, 100_000.0)
        rm.reset_daily_state()
        assert rm.is_halted() is False


# ── approve_order ─────────────────────────────────────────────────────────────


class TestApproveOrder:
    def test_all_checks_pass(self) -> None:
        rm = RiskManager(_make_limits())
        result = rm.approve_order(
            proposed_size_usd=3_000.0,
            portfolio_value=100_000.0,
            current_positions=[],
            daily_pnl=0.0,
        )
        assert result is True

    def test_circuit_breaker_raises(self) -> None:
        rm = RiskManager(_make_limits(max_drawdown=0.02))
        rm.check_daily_drawdown(-5_000.0, 100_000.0)  # trip
        with pytest.raises(CircuitBreakerError):
            rm.approve_order(
                proposed_size_usd=1_000.0,
                portfolio_value=100_000.0,
                current_positions=[],
                daily_pnl=0.0,
            )

    def test_position_size_blocks_order(self) -> None:
        rm = RiskManager(_make_limits(max_position_pct=0.05))
        result = rm.approve_order(
            proposed_size_usd=10_000.0,  # 10% > 5%
            portfolio_value=100_000.0,
            current_positions=[],
            daily_pnl=0.0,
        )
        assert result is False

    def test_max_positions_blocks_order(self) -> None:
        rm = RiskManager(_make_limits(max_positions=2))
        positions = [_make_position("A"), _make_position("B")]
        result = rm.approve_order(
            proposed_size_usd=1_000.0,
            portfolio_value=100_000.0,
            current_positions=positions,
            daily_pnl=0.0,
        )
        assert result is False
