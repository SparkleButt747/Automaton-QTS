"""Simple SMA(10)/SMA(30) crossover strategy.

Used as an integration-test reference strategy that exercises the
full pipeline without any complex signal logic.
"""

from __future__ import annotations

import uuid
from collections import deque

import numpy as np

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


class SMACrossoverStrategy:
    """SMA crossover strategy using SMA(fast) / SMA(slow).

    Generates a BUY market order when fast SMA crosses above slow SMA,
    and a SELL market order when fast SMA crosses below slow SMA.
    Maintains at most one position.

    Attributes:
        fast_period: Look-back for the fast SMA.
        slow_period: Look-back for the slow SMA.
        quantity: Fixed order quantity per trade.
    """

    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 30,
        quantity: float = 1.0,
    ) -> None:
        """Initialise the SMA crossover strategy.

        Args:
            fast_period: Fast SMA window (default 10).
            slow_period: Slow SMA window (default 30).
            quantity: Fixed quantity for each order (default 1.0).
        """
        if fast_period >= slow_period:
            raise ValueError(f"fast_period ({fast_period}) must be < slow_period ({slow_period})")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.quantity = quantity

        self._closes: deque[float] = deque(maxlen=slow_period)
        self._prev_fast_above: bool | None = None

    @property
    def name(self) -> str:
        """Return strategy name."""
        return f"SMA_Crossover({self.fast_period},{self.slow_period})"

    def on_bar(
        self,
        bar: Bar,
        snapshot: SignalSnapshot,
        positions: list[Position],
    ) -> list[Order]:
        """Process a new bar and emit a crossover order if applicable.

        Args:
            bar: The newly closed OHLCV bar.
            snapshot: Signal snapshot (not used by this strategy).
            positions: Currently open positions.

        Returns:
            List of at most one Order.
        """
        self._closes.append(bar.close)

        if len(self._closes) < self.slow_period:
            return []

        closes_arr = np.array(self._closes, dtype=np.float64)
        fast_sma = float(np.mean(closes_arr[-self.fast_period :]))
        slow_sma = float(np.mean(closes_arr))

        fast_above = fast_sma > slow_sma
        orders: list[Order] = []

        if self._prev_fast_above is not None:
            has_long = any(p.direction == TradeDirection.LONG for p in positions)
            has_short = any(p.direction == TradeDirection.SHORT for p in positions)

            if fast_above and not self._prev_fast_above:
                # Golden cross: close short if any, open long
                if has_short:
                    orders.append(
                        Order(
                            order_id=str(uuid.uuid4()),
                            symbol=bar.symbol,
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            quantity=self.quantity,
                            timestamp=bar.timestamp,
                        )
                    )
                if not has_long:
                    orders.append(
                        Order(
                            order_id=str(uuid.uuid4()),
                            symbol=bar.symbol,
                            side=OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            quantity=self.quantity,
                            timestamp=bar.timestamp,
                        )
                    )

            elif not fast_above and self._prev_fast_above:
                # Death cross: close long if any, open short
                if has_long:
                    orders.append(
                        Order(
                            order_id=str(uuid.uuid4()),
                            symbol=bar.symbol,
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=self.quantity,
                            timestamp=bar.timestamp,
                        )
                    )
                if not has_short:
                    orders.append(
                        Order(
                            order_id=str(uuid.uuid4()),
                            symbol=bar.symbol,
                            side=OrderSide.SELL,
                            order_type=OrderType.MARKET,
                            quantity=self.quantity,
                            timestamp=bar.timestamp,
                        )
                    )

        self._prev_fast_above = fast_above
        return orders

    def on_fill(self, fill: Fill) -> None:
        """Handle fill notification (no-op for this strategy).

        Args:
            fill: Execution fill report.
        """
