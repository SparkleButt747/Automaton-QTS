"""Base strategy interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from qts.models.base import Bar, Fill, Order, Position, SignalSnapshot
from qts.world.events import TextEvent


@runtime_checkable
class Strategy(Protocol):
    """Strategy interface. All strategies must implement this."""

    @property
    def name(self) -> str:
        """Return the human-readable strategy name."""
        ...

    def on_bar(
        self,
        bar: Bar,
        snapshot: SignalSnapshot,
        positions: list[Position],
    ) -> list[Order]:
        """Process a new bar and return orders to execute.

        Args:
            bar: The newly closed OHLCV bar.
            snapshot: Latest signal snapshot containing all indicators.
            positions: List of currently open positions.

        Returns:
            List of Order objects to submit; may be empty.
        """
        ...

    def on_fill(self, fill: Fill) -> None:
        """Handle fill notification.

        Args:
            fill: The execution fill report from the broker.
        """
        ...

    def on_text(self, event: TextEvent) -> None:
        """Handle a text event (news, persona statement, macro release).

        Default implementation in concrete strategies is a no-op; news-reactive
        strategies override to update internal belief state.

        Args:
            event: The TextEvent to consume.
        """
        ...
