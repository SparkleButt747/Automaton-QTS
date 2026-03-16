"""Market data provider protocols for dependency injection."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from qts.models.base import Bar, OrderBookSnapshot, Tick


@runtime_checkable
class TickProvider(Protocol):
    """Provides real-time tick data."""

    async def subscribe(self, symbol: str) -> AsyncIterator[Tick]:
        """Subscribe to real-time tick data for a symbol.

        Args:
            symbol: The ticker symbol to subscribe to.

        Returns:
            An async iterator yielding Tick objects as they arrive.
        """
        ...

    async def close(self) -> None:
        """Close the provider and release any held resources."""
        ...


@runtime_checkable
class BarProvider(Protocol):
    """Provides OHLCV bar data."""

    async def get_historical_bars(
        self, symbol: str, start: str, end: str, interval: str = "1m"
    ) -> list[Bar]:
        """Fetch historical OHLCV bars for a symbol.

        Args:
            symbol: The ticker symbol to query.
            start: Start datetime string (ISO format recommended).
            end: End datetime string (ISO format recommended).
            interval: Bar interval (e.g. '1m', '5m', '1h').

        Returns:
            List of Bar objects ordered by timestamp ascending.
        """
        ...

    async def subscribe(self, symbol: str, interval: str = "1m") -> AsyncIterator[Bar]:
        """Subscribe to real-time bar updates for a symbol.

        Args:
            symbol: The ticker symbol to subscribe to.
            interval: Bar interval (e.g. '1m', '5m', '1h').

        Returns:
            An async iterator yielding Bar objects as each interval closes.
        """
        ...

    async def close(self) -> None:
        """Close the provider and release any held resources."""
        ...


@runtime_checkable
class OrderBookProvider(Protocol):
    """Provides order book snapshots."""

    async def get_snapshot(self, symbol: str, depth: int = 20) -> OrderBookSnapshot:
        """Fetch a current order book snapshot for a symbol.

        Args:
            symbol: The ticker symbol to query.
            depth: Number of price levels to include per side.

        Returns:
            An OrderBookSnapshot with bids and asks up to the requested depth.
        """
        ...

    async def subscribe(
        self, symbol: str, depth: int = 20
    ) -> AsyncIterator[OrderBookSnapshot]:
        """Subscribe to real-time order book updates for a symbol.

        Args:
            symbol: The ticker symbol to subscribe to.
            depth: Number of price levels to include per side.

        Returns:
            An async iterator yielding OrderBookSnapshot objects as the book updates.
        """
        ...

    async def close(self) -> None:
        """Close the provider and release any held resources."""
        ...
