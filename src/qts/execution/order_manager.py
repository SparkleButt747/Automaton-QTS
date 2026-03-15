"""Order lifecycle management."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from qts.models.base import Fill, Order, OrderSide, OrderStatus, OrderType

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages the full lifecycle of trading orders.

    In paper-trading mode (``paper_mode=True``) all orders receive an
    immediate simulated fill at the requested price (or mid-market if no
    price is set).  In live mode the manager would interface with a broker
    API; that is left as a future extension point.

    Args:
        paper_mode: When True, fills are simulated immediately.
        default_fill_price: Fallback fill price when no price is attached to
                            a market order (useful for tests).
    """

    def __init__(
        self,
        paper_mode: bool = True,
        default_fill_price: float = 100.0,
    ) -> None:
        self.paper_mode = paper_mode
        self.default_fill_price = default_fill_price
        # order_id -> Order
        self._orders: dict[str, Order] = {}
        # order_id -> OrderStatus
        self._statuses: dict[str, OrderStatus] = {}
        # order_id -> list[Fill]
        self._fills: dict[str, list[Fill]] = {}

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(self, order: Order) -> str:
        """Submit an order and return the assigned order ID.

        In paper mode the order is immediately simulated as a full fill.

        Args:
            order: :class:`~qts.models.base.Order` to submit.

        Returns:
            The order ID string.
        """
        order_id = order.order_id or str(uuid.uuid4())
        self._orders[order_id] = order
        self._statuses[order_id] = OrderStatus.SUBMITTED

        logger.info(
            "OrderManager: submitted %s | %s %s qty=%.4f",
            order_id,
            order.side,
            order.symbol,
            order.quantity,
        )

        if self.paper_mode:
            self._simulate_fill(order_id, order)

        return order_id

    def _simulate_fill(self, order_id: str, order: Order) -> None:
        """Immediately simulate a full fill for paper trading.

        Args:
            order_id: ID of the order to fill.
            order: The order to simulate.
        """
        fill_price = order.price or self.default_fill_price
        commission = fill_price * order.quantity * 0.001  # 0.1% taker fee

        fill = Fill(
            order_id=order_id,
            fill_id=str(uuid.uuid4()),
            symbol=order.symbol,
            side=order.side,
            price=fill_price,
            quantity=order.quantity,
            commission=commission,
            timestamp=datetime.now(timezone.utc),
            slippage=0.0,
        )

        self._fills.setdefault(order_id, []).append(fill)
        self._statuses[order_id] = OrderStatus.FILLED

        logger.debug(
            "OrderManager: simulated fill %s @ %.4f for order %s",
            fill.fill_id,
            fill.price,
            order_id,
        )

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order.

        Args:
            order_id: ID of the order to cancel.

        Returns:
            True if the cancellation was accepted; False if the order is not
            in a cancellable state (e.g. already filled or unknown).
        """
        status = self._statuses.get(order_id)
        if status is None:
            logger.warning("OrderManager: cancel_order — unknown order %s", order_id)
            return False

        cancellable = {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}
        if status not in cancellable:
            logger.info(
                "OrderManager: cannot cancel order %s in status %s", order_id, status
            )
            return False

        self._statuses[order_id] = OrderStatus.CANCELLED
        logger.info("OrderManager: cancelled order %s", order_id)
        return True

    def get_order_status(self, order_id: str) -> OrderStatus:
        """Return the current status of an order.

        Args:
            order_id: ID of the order to query.

        Returns:
            :class:`~qts.models.base.OrderStatus`.

        Raises:
            KeyError: If the order ID is not found.
        """
        if order_id not in self._statuses:
            raise KeyError(f"Unknown order ID: {order_id}")
        return self._statuses[order_id]

    def get_open_orders(self) -> list[Order]:
        """Return all orders that are currently open (not terminal).

        Returns:
            List of :class:`~qts.models.base.Order` objects.
        """
        terminal = {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}
        return [
            order
            for order_id, order in self._orders.items()
            if self._statuses.get(order_id) not in terminal
        ]

    def get_fills(self, order_id: str) -> list[Fill]:
        """Return all fills associated with a given order.

        Args:
            order_id: Order ID to look up.

        Returns:
            List of :class:`~qts.models.base.Fill` objects (may be empty).
        """
        return list(self._fills.get(order_id, []))
