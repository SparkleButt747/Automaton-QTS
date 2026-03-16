"""Mock market data providers for testing.

Implements TickProvider, BarProvider, and OrderBookProvider protocols using
deterministic synthetic data generated with a seeded numpy random number
generator. All outputs are reproducible for a given seed, making these
adapters suitable for unit tests and integration tests.

Synthetic data model:
- Prices follow a geometric random walk (log-normal increments).
- Volume is drawn from a log-normal distribution.
- Order book spreads are proportional to price.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import numpy as np

from qts.models.base import Bar, OrderBookLevel, OrderBookSnapshot, OrderSide, Tick

logger = logging.getLogger(__name__)

# Default parameters for synthetic data generation
_DEFAULT_SEED = 42
_DEFAULT_INITIAL_PRICE = 50_000.0  # USD, appropriate for BTC-like instruments
_DEFAULT_DRIFT = 0.0  # Annualised log-return drift (0 = no trend)
_DEFAULT_VOLATILITY = 0.02  # Per-bar standard deviation of log-returns
_DEFAULT_VOLUME_MEAN = 1.5  # Mean of log-normal volume distribution
_DEFAULT_VOLUME_STD = 0.5  # Std of log-normal volume distribution
_DEFAULT_BAR_INTERVAL_SECONDS = 60  # 1-minute bars


def _make_rng(seed: int) -> np.random.Generator:
    """Create a seeded NumPy random generator.

    Args:
        seed: Integer seed for reproducibility.

    Returns:
        A seeded numpy.random.Generator.
    """
    return np.random.default_rng(seed)


class MockTickAdapter:
    """Mock TickProvider yielding deterministic synthetic tick data.

    Generates a stream of Tick objects for any requested symbol using a
    geometric random walk price model. The symbol name is embedded in each
    tick but does not affect the synthetic data generation (only the seed does).

    Attributes:
        seed: RNG seed used for reproducibility.
        initial_price: Starting price for the random walk.
        n_ticks: Number of ticks to generate per subscription.
        tick_interval_ms: Interval between ticks in milliseconds.
    """

    def __init__(
        self,
        seed: int = _DEFAULT_SEED,
        initial_price: float = _DEFAULT_INITIAL_PRICE,
        n_ticks: int = 100,
        tick_interval_ms: int = 500,
        volatility: float = _DEFAULT_VOLATILITY,
    ) -> None:
        """Initialise the mock tick adapter.

        Args:
            seed: Integer seed for the RNG.
            initial_price: Starting price for the geometric random walk.
            n_ticks: Number of ticks to emit per subscribe call.
            tick_interval_ms: Simulated interval between ticks (milliseconds).
            volatility: Per-tick price volatility (log-return std dev).
        """
        self._seed = seed
        self._initial_price = initial_price
        self._n_ticks = n_ticks
        self._tick_interval_ms = tick_interval_ms
        self._volatility = volatility
        self._closed = False

    async def subscribe(self, symbol: str) -> AsyncIterator[Tick]:
        """Yield synthetic tick data for a symbol.

        Args:
            symbol: Ticker symbol to simulate (embedded in Tick objects).

        Yields:
            Deterministic Tick objects simulating a price random walk.
        """
        rng = _make_rng(self._seed)
        price = self._initial_price
        start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)

        for i in range(self._n_ticks):
            if self._closed:
                break

            # Geometric random walk step
            log_return = rng.normal(0.0, self._volatility)
            price = price * np.exp(log_return)

            # Quantity: log-normal
            quantity = float(np.exp(rng.normal(_DEFAULT_VOLUME_MEAN, _DEFAULT_VOLUME_STD)))

            # Alternate buy/sell with slight random noise
            side = OrderSide.BUY if rng.random() > 0.5 else OrderSide.SELL  # noqa: S311

            timestamp = start_time + timedelta(milliseconds=i * self._tick_interval_ms)

            yield Tick(
                timestamp=timestamp,
                symbol=symbol,
                price=float(price),
                quantity=quantity,
                side=side,
            )
            await asyncio.sleep(0)

    async def close(self) -> None:
        """Mark the adapter as closed; subsequent subscriptions will emit no ticks."""
        self._closed = True
        logger.debug("MockTickAdapter closed")


class MockBarAdapter:
    """Mock BarProvider yielding deterministic synthetic OHLCV bars.

    Generates bars using a geometric random walk for close prices, with OHLC
    values synthesised from intra-bar noise. Volume is log-normal.

    Attributes:
        seed: RNG seed used for reproducibility.
        initial_price: Starting price for the random walk.
        n_bars: Number of bars to generate per request/subscription.
    """

    def __init__(
        self,
        seed: int = _DEFAULT_SEED,
        initial_price: float = _DEFAULT_INITIAL_PRICE,
        n_bars: int = 200,
        bar_interval_seconds: int = _DEFAULT_BAR_INTERVAL_SECONDS,
        volatility: float = _DEFAULT_VOLATILITY,
    ) -> None:
        """Initialise the mock bar adapter.

        Args:
            seed: Integer seed for the RNG.
            initial_price: Starting price for the geometric random walk.
            n_bars: Number of bars to emit per get_historical_bars / subscribe call.
            bar_interval_seconds: Simulated interval between bars (seconds).
            volatility: Per-bar price volatility (log-return std dev).
        """
        self._seed = seed
        self._initial_price = initial_price
        self._n_bars = n_bars
        self._bar_interval_seconds = bar_interval_seconds
        self._volatility = volatility
        self._closed = False

    def _generate_bars(self, symbol: str, n: int) -> list[Bar]:
        """Generate a deterministic list of OHLCV bars.

        Args:
            symbol: Ticker symbol to embed in each bar.
            n: Number of bars to generate.

        Returns:
            List of Bar objects in chronological order.
        """
        rng = _make_rng(self._seed)
        price = self._initial_price
        start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        bars: list[Bar] = []

        for i in range(n):
            open_price = price

            # Close is a random walk step from open
            log_return = rng.normal(0.0, self._volatility)
            close_price = open_price * float(np.exp(log_return))

            # High/low: add intra-bar noise around the open/close range
            intra_noise = abs(float(rng.normal(0.0, self._volatility * 0.5)))
            high_price = max(open_price, close_price) * (1.0 + intra_noise)
            low_price = min(open_price, close_price) * (1.0 - intra_noise)

            # Volume: log-normal
            volume = float(np.exp(rng.normal(_DEFAULT_VOLUME_MEAN, _DEFAULT_VOLUME_STD)) * 10.0)

            timestamp = start_time + timedelta(seconds=i * self._bar_interval_seconds)

            bars.append(
                Bar(
                    timestamp=timestamp,
                    symbol=symbol,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                    volume=volume,
                    bar_count=int(rng.integers(1, 50)),
                )
            )

            price = close_price  # Continue the random walk from close

        return bars

    async def get_historical_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1m",
    ) -> list[Bar]:
        """Return deterministic historical bars for a symbol.

        The start and end parameters are accepted for protocol compatibility.
        The actual bars generated are determined solely by the seed and n_bars.

        Args:
            symbol: Ticker symbol to simulate.
            start: Start datetime string (accepted but not used for filtering).
            end: End datetime string (accepted but not used for filtering).
            interval: Bar interval string (accepted but not used).

        Returns:
            List of synthetic Bar objects.
        """
        bars = self._generate_bars(symbol, self._n_bars)
        logger.debug("MockBarAdapter: returning %d historical bars for %s", len(bars), symbol)
        return bars

    async def subscribe(self, symbol: str, interval: str = "1m") -> AsyncIterator[Bar]:
        """Yield synthetic bar data as an async stream.

        Args:
            symbol: Ticker symbol to simulate.
            interval: Bar interval string (accepted for protocol compatibility).

        Yields:
            Deterministic Bar objects in chronological order.
        """
        bars = self._generate_bars(symbol, self._n_bars)
        logger.info("MockBarAdapter: subscribing to %d bars for symbol %s", len(bars), symbol)
        for bar in bars:
            if self._closed:
                break
            yield bar
            await asyncio.sleep(0)

    async def close(self) -> None:
        """Mark the adapter as closed; subsequent subscriptions will emit no bars."""
        self._closed = True
        logger.debug("MockBarAdapter closed")


class MockOrderBookAdapter:
    """Mock OrderBookProvider yielding deterministic synthetic order book snapshots.

    Generates L2 order book snapshots using synthetic bid/ask spreads
    around a random-walk mid price.

    Attributes:
        seed: RNG seed used for reproducibility.
        initial_price: Starting mid price for the random walk.
        depth: Default number of price levels per side.
    """

    def __init__(
        self,
        seed: int = _DEFAULT_SEED,
        initial_price: float = _DEFAULT_INITIAL_PRICE,
        depth: int = 20,
        n_snapshots: int = 50,
        spread_bps: float = 2.0,
    ) -> None:
        """Initialise the mock order book adapter.

        Args:
            seed: Integer seed for the RNG.
            initial_price: Starting mid price.
            depth: Default number of price levels per side.
            n_snapshots: Number of snapshots to emit per subscription.
            spread_bps: Bid/ask spread in basis points around the mid price.
        """
        self._seed = seed
        self._initial_price = initial_price
        self._depth = depth
        self._n_snapshots = n_snapshots
        self._spread_bps = spread_bps
        self._closed = False

    def _make_snapshot(
        self,
        symbol: str,
        mid_price: float,
        depth: int,
        rng: np.random.Generator,
        timestamp: datetime,
    ) -> OrderBookSnapshot:
        """Build a single synthetic order book snapshot.

        Args:
            symbol: Ticker symbol to embed.
            mid_price: Mid price around which to construct the book.
            depth: Number of levels per side.
            rng: Seeded RNG for reproducibility.
            timestamp: Timestamp to embed in the snapshot.

        Returns:
            An OrderBookSnapshot with synthetic bids and asks.
        """
        half_spread = mid_price * self._spread_bps / 10_000.0

        bids: list[OrderBookLevel] = []
        asks: list[OrderBookLevel] = []

        for level in range(depth):
            # Each level is spaced by the spread; quantities are log-normal
            bid_price = mid_price - half_spread - level * half_spread * 0.5
            ask_price = mid_price + half_spread + level * half_spread * 0.5
            bid_qty = float(np.exp(rng.normal(0.5, 0.3)))
            ask_qty = float(np.exp(rng.normal(0.5, 0.3)))

            bids.append(OrderBookLevel(price=bid_price, quantity=bid_qty))
            asks.append(OrderBookLevel(price=ask_price, quantity=ask_qty))

        return OrderBookSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            bids=tuple(bids),
            asks=tuple(asks),
        )

    async def get_snapshot(self, symbol: str, depth: int = 20) -> OrderBookSnapshot:
        """Return a single synthetic order book snapshot.

        Args:
            symbol: Ticker symbol to simulate.
            depth: Number of price levels to include per side.

        Returns:
            A deterministic OrderBookSnapshot.
        """
        rng = _make_rng(self._seed)
        timestamp = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        snapshot = self._make_snapshot(symbol, self._initial_price, depth, rng, timestamp)
        logger.debug("MockOrderBookAdapter: snapshot for %s, mid=%.2f", symbol, self._initial_price)
        return snapshot

    async def subscribe(self, symbol: str, depth: int = 20) -> AsyncIterator[OrderBookSnapshot]:
        """Yield synthetic order book snapshots as an async stream.

        Args:
            symbol: Ticker symbol to simulate.
            depth: Number of price levels per side.

        Yields:
            Deterministic OrderBookSnapshot objects in chronological order.
        """
        rng = _make_rng(self._seed)
        price = self._initial_price
        start_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)

        logger.info(
            "MockOrderBookAdapter: subscribing to %d snapshots for %s",
            self._n_snapshots,
            symbol,
        )

        for i in range(self._n_snapshots):
            if self._closed:
                break

            # Evolve mid price with a small random walk
            log_return = rng.normal(0.0, _DEFAULT_VOLATILITY * 0.1)
            price = price * float(np.exp(log_return))
            timestamp = start_time + timedelta(milliseconds=i * 500)

            yield self._make_snapshot(symbol, price, depth, rng, timestamp)
            await asyncio.sleep(0)

    async def close(self) -> None:
        """Mark the adapter as closed; subsequent subscriptions will emit no snapshots."""
        self._closed = True
        logger.debug("MockOrderBookAdapter closed")
