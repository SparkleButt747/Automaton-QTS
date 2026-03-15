"""Momentum + sentiment overlay strategy.

The main production strategy. Uses the combined_alpha signal for entry/exit
decisions with Kelly-inspired position sizing and risk-limit enforcement.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

import numpy as np

from qts.config import RiskLimits, StrategyParams
from qts.models.base import (
    Bar,
    Fill,
    Order,
    OrderSide,
    OrderType,
    Position,
    SignalSnapshot,
    TradeDirection,
)

logger = logging.getLogger(__name__)


class MomentumStrategy:
    """Combined momentum + sentiment strategy.

    Entry rules:
        - LONG  when combined_alpha > entry_threshold
        - SHORT when combined_alpha < -entry_threshold

    Exit rules:
        - Exit when combined_alpha falls below exit_threshold (for longs)
          or rises above -exit_threshold (for shorts)
        - Exit when bars_held >= max_hold_bars (time stop)

    Position sizing:
        - Kelly fraction: f* = alpha / (1 + alpha)   (simplified)
        - Capped by risk_limits.max_position_size_pct * portfolio_value

    Attributes:
        params: StrategyParams containing weights and thresholds.
        risk_limits: RiskLimits for position size and drawdown caps.
        portfolio_value: Notional portfolio size for sizing calculations.
    """

    def __init__(
        self,
        params: StrategyParams,
        risk_limits: RiskLimits,
        portfolio_value: float = 100_000.0,
    ) -> None:
        """Initialise the MomentumStrategy.

        Args:
            params: Strategy parameters (thresholds, weights).
            risk_limits: Risk limit configuration.
            portfolio_value: Current portfolio value for position sizing.
        """
        self.params = params
        self.risk_limits = risk_limits
        self.portfolio_value = portfolio_value

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "MomentumSentiment"

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def _kelly_quantity(self, alpha: float, price: float) -> float:
        """Compute a Kelly-inspired order quantity.

        Formula:
            f* = |alpha| / (1 + |alpha|)      # simplified half-Kelly
            max_usd = portfolio_value * max_position_size_pct
            quantity = min(f* * portfolio_value, max_usd) / price

        Args:
            alpha: Combined alpha value.
            price: Current asset price.

        Returns:
            Order quantity (shares / contracts) as a positive float.
        """
        abs_alpha = abs(alpha)
        kelly_frac = abs_alpha / (1.0 + abs_alpha)
        max_usd = self.portfolio_value * self.risk_limits.max_position_size_pct
        notional = min(kelly_frac * self.portfolio_value, max_usd)
        quantity = notional / price if price > 0.0 else 0.0
        return max(quantity, 0.0)

    def on_bar(
        self,
        bar: Bar,
        snapshot: SignalSnapshot,
        positions: list[Position],
    ) -> list[Order]:
        """Process a bar and return any orders to execute.

        Args:
            bar: The newly closed OHLCV bar.
            snapshot: Latest signal snapshot with combined_alpha set.
            positions: Currently open positions.

        Returns:
            List of Order objects; may be empty.
        """
        alpha = snapshot.combined_alpha
        orders: list[Order] = []

        existing_long: Optional[Position] = next(
            (p for p in positions if p.direction == TradeDirection.LONG), None
        )
        existing_short: Optional[Position] = next(
            (p for p in positions if p.direction == TradeDirection.SHORT), None
        )

        # ── Exit logic ───────────────────────────────────────────────────────
        if existing_long is not None:
            time_stop = existing_long.bars_held >= self.params.max_hold_bars
            alpha_stop = alpha < self.params.exit_threshold
            if time_stop or alpha_stop:
                reason = "time_stop" if time_stop else "alpha_exit"
                logger.debug("Closing LONG: reason=%s, alpha=%.3f", reason, alpha)
                orders.append(
                    Order(
                        order_id=str(uuid.uuid4()),
                        symbol=bar.symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=existing_long.quantity,
                        timestamp=bar.timestamp,
                    )
                )
                existing_long = None

        if existing_short is not None:
            time_stop = existing_short.bars_held >= self.params.max_hold_bars
            alpha_stop = alpha > -self.params.exit_threshold
            if time_stop or alpha_stop:
                reason = "time_stop" if time_stop else "alpha_exit"
                logger.debug("Closing SHORT: reason=%s, alpha=%.3f", reason, alpha)
                orders.append(
                    Order(
                        order_id=str(uuid.uuid4()),
                        symbol=bar.symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=existing_short.quantity,
                        timestamp=bar.timestamp,
                    )
                )
                existing_short = None

        # ── Entry logic ──────────────────────────────────────────────────────
        if alpha > self.params.entry_threshold and existing_long is None:
            qty = self._kelly_quantity(alpha, bar.close)
            if qty > 0.0:
                logger.debug("Opening LONG: alpha=%.3f, qty=%.4f", alpha, qty)
                orders.append(
                    Order(
                        order_id=str(uuid.uuid4()),
                        symbol=bar.symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=qty,
                        timestamp=bar.timestamp,
                    )
                )

        elif alpha < -self.params.entry_threshold and existing_short is None:
            qty = self._kelly_quantity(alpha, bar.close)
            if qty > 0.0:
                logger.debug("Opening SHORT: alpha=%.3f, qty=%.4f", alpha, qty)
                orders.append(
                    Order(
                        order_id=str(uuid.uuid4()),
                        symbol=bar.symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=qty,
                        timestamp=bar.timestamp,
                    )
                )

        return orders

    def on_fill(self, fill: Fill) -> None:
        """Handle fill notification.

        Args:
            fill: Execution fill report.
        """
        logger.debug(
            "Fill received: order_id=%s, side=%s, qty=%.4f, price=%.4f",
            fill.order_id,
            fill.side,
            fill.quantity,
            fill.price,
        )
