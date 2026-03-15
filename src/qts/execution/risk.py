"""Risk management module.

Provides the RiskManager class that enforces position size limits,
maximum open position counts, daily drawdown limits, and circuit breaker
logic with configurable cooldown periods.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from qts.config import RiskLimits
from qts.models.base import Position

logger = logging.getLogger(__name__)


class CircuitBreakerError(Exception):
    """Raised when a trade is blocked by an active circuit breaker."""


class RiskManager:
    """Enforces risk limits and circuit-breaker logic.

    The RiskManager is stateful: it tracks the circuit breaker trip time
    and daily PnL across calls. Reset between trading sessions with
    `reset_daily_state()`.

    Attributes:
        limits: Immutable RiskLimits configuration.
        _circuit_breaker_tripped_at: Timestamp when circuit breaker was triggered.
        _halted: Whether trading is currently halted.
    """

    def __init__(self, limits: RiskLimits) -> None:
        """Initialise the RiskManager.

        Args:
            limits: Loaded RiskLimits configuration.
        """
        self.limits = limits
        self._circuit_breaker_tripped_at: Optional[datetime] = None
        self._halted: bool = False

    # ------------------------------------------------------------------
    # Position size check
    # ------------------------------------------------------------------

    def check_position_size(
        self, proposed_size: float, portfolio_value: float
    ) -> bool:
        """Return True if the proposed position size is within limits.

        Args:
            proposed_size: Notional USD value of the proposed position.
            portfolio_value: Current total portfolio value.

        Returns:
            True if the position is allowed; False if it exceeds the limit.
        """
        if portfolio_value <= 0.0:
            logger.warning("check_position_size: portfolio_value is non-positive")
            return False
        fraction = proposed_size / portfolio_value
        allowed = fraction <= self.limits.max_position_size_pct
        if not allowed:
            logger.warning(
                "Position size rejected: %.2f%% > max %.2f%%",
                fraction * 100,
                self.limits.max_position_size_pct * 100,
            )
        return allowed

    # ------------------------------------------------------------------
    # Max open positions
    # ------------------------------------------------------------------

    def check_max_positions(self, current_positions: list[Position]) -> bool:
        """Return True if opening another position is within the limit.

        Args:
            current_positions: List of currently open Position objects.

        Returns:
            True if a new position can be opened; False if at the limit.
        """
        current_count = len(current_positions)
        allowed = current_count < self.limits.max_open_positions
        if not allowed:
            logger.warning(
                "Max positions reached: %d >= %d",
                current_count,
                self.limits.max_open_positions,
            )
        return allowed

    # ------------------------------------------------------------------
    # Daily drawdown check
    # ------------------------------------------------------------------

    def check_daily_drawdown(
        self, daily_pnl: float, portfolio_value: float
    ) -> bool:
        """Return True if daily drawdown is within the allowed limit.

        If the drawdown exceeds the limit, the circuit breaker is tripped
        and trading is halted for `circuit_breaker_cooldown_seconds`.

        Args:
            daily_pnl: Realised + unrealised PnL for today (negative = loss).
            portfolio_value: Portfolio value at start of day (denominator).

        Returns:
            True if within limits; False if circuit breaker was tripped.
        """
        if portfolio_value <= 0.0:
            return True

        drawdown_pct = -daily_pnl / portfolio_value  # positive = loss
        if drawdown_pct >= self.limits.max_daily_drawdown_pct:
            if not self._halted:
                self._trip_circuit_breaker()
            return False
        return True

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------

    def _trip_circuit_breaker(self) -> None:
        """Trip the circuit breaker and halt trading."""
        now = datetime.now(timezone.utc)
        self._circuit_breaker_tripped_at = now
        self._halted = True
        logger.error(
            "CIRCUIT BREAKER TRIPPED at %s — trading halted for %d seconds",
            now.isoformat(),
            self.limits.circuit_breaker_cooldown_seconds,
        )

    def is_halted(self, current_time: Optional[datetime] = None) -> bool:
        """Return True if trading is currently halted by the circuit breaker.

        Automatically lifts the halt after the cooldown period has elapsed.

        Args:
            current_time: Timestamp to check against cooldown. Defaults to
                          datetime.utcnow().

        Returns:
            True if trading is halted; False if the cooldown has passed.
        """
        if not self._halted:
            return False

        now = current_time or datetime.now(timezone.utc)
        if self._circuit_breaker_tripped_at is None:
            self._halted = False
            return False

        elapsed = (now - self._circuit_breaker_tripped_at).total_seconds()
        if elapsed >= self.limits.circuit_breaker_cooldown_seconds:
            logger.info(
                "Circuit breaker cooldown elapsed (%.1f s >= %d s) — trading resumed",
                elapsed,
                self.limits.circuit_breaker_cooldown_seconds,
            )
            self._halted = False
            self._circuit_breaker_tripped_at = None

        return self._halted

    def reset_daily_state(self) -> None:
        """Reset daily PnL tracking and lift any expired circuit breaker.

        Call at the start of each trading session.
        """
        self._halted = False
        self._circuit_breaker_tripped_at = None
        logger.debug("RiskManager: daily state reset")

    # ------------------------------------------------------------------
    # Combined guard
    # ------------------------------------------------------------------

    def approve_order(
        self,
        proposed_size_usd: float,
        portfolio_value: float,
        current_positions: list[Position],
        daily_pnl: float,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """Run all risk checks and return True only if all pass.

        Args:
            proposed_size_usd: Notional USD value of the proposed trade.
            portfolio_value: Current total portfolio value.
            current_positions: Currently open positions.
            daily_pnl: Today's running PnL.
            current_time: Timestamp for circuit-breaker check.

        Returns:
            True if the order is approved; False if blocked.

        Raises:
            CircuitBreakerError: If trading is halted.
        """
        if self.is_halted(current_time):
            raise CircuitBreakerError(
                "Trading halted by circuit breaker. "
                f"Cooldown: {self.limits.circuit_breaker_cooldown_seconds}s"
            )

        if not self.check_daily_drawdown(daily_pnl, portfolio_value):
            return False

        if not self.check_max_positions(current_positions):
            return False

        if not self.check_position_size(proposed_size_usd, portfolio_value):
            return False

        return True
