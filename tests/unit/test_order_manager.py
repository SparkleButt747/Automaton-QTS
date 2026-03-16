"""Unit tests for qts.execution.order_manager.OrderManager.

Verifies:
- submit_order registers the order and returns an order ID
- submit_order immediately simulates a fill in paper_mode=True
- submit_order uses order.price as fill price when set
- submit_order falls back to default_fill_price when order.price is None
- get_order_status returns FILLED after paper fill
- get_order_status raises KeyError for unknown IDs
- cancel_order cancels a submitted order (not yet filled)
- cancel_order returns False for already-filled orders
- cancel_order returns False for unknown order IDs
- get_open_orders excludes terminal orders
- get_fills returns fills for a given order ID
- get_fills returns [] for an order with no fills
"""

from __future__ import annotations

import pytest

from qts.execution.order_manager import OrderManager
from qts.models.base import Fill, Order, OrderSide, OrderStatus, OrderType

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_order(order_id="ord-1", price: float | None = 42000.0, qty=0.5) -> Order:
    return Order(
        order_id=order_id,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=qty,
        price=price,
    )


# ── T-OM-1 ────────────────────────────────────────────────────────────────────


class TestSubmitOrder:
    def test_returns_order_id(self):
        """submit_order returns the order's own ID when one is provided."""
        om = OrderManager(paper_mode=True)
        order_id = om.submit_order(_make_order(order_id="my-id"))
        assert order_id == "my-id"

    def test_generates_id_when_none(self):
        """submit_order generates a UUID when order.order_id is falsy."""
        om = OrderManager(paper_mode=True)
        order = Order(
            order_id="",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=1.0,
        )
        order_id = om.submit_order(order)
        assert order_id  # non-empty string

    def test_paper_mode_fills_immediately(self):
        """In paper_mode=True, the order is filled immediately after submission."""
        om = OrderManager(paper_mode=True)
        order_id = om.submit_order(_make_order())
        assert om.get_order_status(order_id) == OrderStatus.FILLED

    def test_non_paper_mode_stays_submitted(self):
        """In paper_mode=False, the order stays in SUBMITTED status."""
        om = OrderManager(paper_mode=False)
        order_id = om.submit_order(_make_order())
        assert om.get_order_status(order_id) == OrderStatus.SUBMITTED

    def test_fill_uses_order_price(self):
        """Paper fill price equals order.price when explicitly set."""
        om = OrderManager(paper_mode=True)
        order_id = om.submit_order(_make_order(price=50000.0))
        fills = om.get_fills(order_id)
        assert fills[0].price == 50000.0

    def test_fill_falls_back_to_default_price(self):
        """Paper fill uses default_fill_price when order.price is None."""
        om = OrderManager(paper_mode=True, default_fill_price=99.0)
        order = _make_order(price=None)
        order_id = om.submit_order(order)
        fills = om.get_fills(order_id)
        assert fills[0].price == 99.0

    def test_commission_is_01_percent_of_fill_value(self):
        """Paper fill commission is 0.1% of price * quantity."""
        om = OrderManager(paper_mode=True)
        order_id = om.submit_order(_make_order(price=1000.0, qty=2.0))
        fills = om.get_fills(order_id)
        assert fills[0].commission == pytest.approx(1000.0 * 2.0 * 0.001)


# ── T-OM-2 ────────────────────────────────────────────────────────────────────


class TestGetOrderStatus:
    def test_returns_status_for_known_order(self):
        """get_order_status returns the current status of a known order."""
        om = OrderManager(paper_mode=False)
        order_id = om.submit_order(_make_order())
        status = om.get_order_status(order_id)
        assert status == OrderStatus.SUBMITTED

    def test_raises_key_error_for_unknown_id(self):
        """get_order_status raises KeyError when the order ID is not found."""
        om = OrderManager()
        with pytest.raises(KeyError, match="Unknown order ID"):
            om.get_order_status("ghost-id")


# ── T-OM-3 ────────────────────────────────────────────────────────────────────


class TestCancelOrder:
    def test_cancels_submitted_order(self):
        """cancel_order succeeds and marks a SUBMITTED order as CANCELLED."""
        om = OrderManager(paper_mode=False)
        order_id = om.submit_order(_make_order())
        result = om.cancel_order(order_id)
        assert result is True
        assert om.get_order_status(order_id) == OrderStatus.CANCELLED

    def test_returns_false_for_filled_order(self):
        """cancel_order returns False for an already-filled order."""
        om = OrderManager(paper_mode=True)
        order_id = om.submit_order(_make_order())
        # order is already FILLED by paper mode
        result = om.cancel_order(order_id)
        assert result is False
        assert om.get_order_status(order_id) == OrderStatus.FILLED

    def test_returns_false_for_unknown_order(self):
        """cancel_order returns False when the order ID is not found."""
        om = OrderManager()
        result = om.cancel_order("no-such-order")
        assert result is False


# ── T-OM-4 ────────────────────────────────────────────────────────────────────


class TestGetOpenOrders:
    def test_returns_submitted_order(self):
        """get_open_orders includes orders in SUBMITTED status."""
        om = OrderManager(paper_mode=False)
        om.submit_order(_make_order(order_id="ord-1"))
        open_orders = om.get_open_orders()
        assert len(open_orders) == 1

    def test_excludes_filled_orders(self):
        """get_open_orders excludes orders in FILLED status."""
        om = OrderManager(paper_mode=True)
        om.submit_order(_make_order())
        open_orders = om.get_open_orders()
        assert open_orders == []

    def test_excludes_cancelled_orders(self):
        """get_open_orders excludes orders in CANCELLED status."""
        om = OrderManager(paper_mode=False)
        order_id = om.submit_order(_make_order())
        om.cancel_order(order_id)
        open_orders = om.get_open_orders()
        assert open_orders == []


# ── T-OM-5 ────────────────────────────────────────────────────────────────────


class TestGetFills:
    def test_returns_fills_after_paper_fill(self):
        """get_fills returns the simulated fill for a paper order."""
        om = OrderManager(paper_mode=True)
        order_id = om.submit_order(_make_order())
        fills = om.get_fills(order_id)
        assert len(fills) == 1
        assert isinstance(fills[0], Fill)

    def test_returns_empty_list_for_unfilled_order(self):
        """get_fills returns [] for a SUBMITTED order with no fills yet."""
        om = OrderManager(paper_mode=False)
        order_id = om.submit_order(_make_order())
        fills = om.get_fills(order_id)
        assert fills == []

    def test_returns_empty_list_for_unknown_order(self):
        """get_fills returns [] when the order ID is not known."""
        om = OrderManager()
        assert om.get_fills("ghost") == []
