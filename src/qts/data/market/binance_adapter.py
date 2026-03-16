"""Binance public market data adapters.

Implements BarProvider, TickProvider, and OrderBookProvider protocols using
the Binance REST and WebSocket APIs. No API key required (public endpoints only).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import httpx

from qts.models.base import Bar, OrderBookLevel, OrderBookSnapshot, OrderSide, Tick

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BINANCE_REST_BASE = "https://api.binance.com"
BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws/"

KLINES_ENDPOINT = "/api/v3/klines"
DEPTH_ENDPOINT = "/api/v3/depth"

MAX_KLINES_PER_REQUEST = 1000

# Rate-limit backoff
INITIAL_BACKOFF_SECS = 0.5
MAX_BACKOFF_SECS = 30.0
BACKOFF_FACTOR = 2.0
MAX_RETRIES = 5

# Default HTTP timeout
DEFAULT_TIMEOUT = 30.0


def _ms_to_datetime(ms: int | float) -> datetime:
    """Convert millisecond Unix timestamp to timezone-aware datetime."""
    return datetime.fromtimestamp(int(ms) / 1000, tz=UTC)


def _iso_to_ms(iso_str: str) -> int:
    """Convert ISO 8601 datetime string to millisecond Unix timestamp."""
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _parse_kline(symbol: str, kline: list[Any]) -> Bar:
    """Parse a Binance kline array into a Bar.

    Binance kline format: [open_time, open, high, low, close, volume,
    close_time, quote_volume, trades, taker_buy_base, taker_buy_quote, ignore]
    """
    return Bar(
        timestamp=_ms_to_datetime(kline[0]),
        symbol=symbol,
        open=float(kline[1]),
        high=float(kline[2]),
        low=float(kline[3]),
        close=float(kline[4]),
        volume=float(kline[5]),
        bar_count=int(kline[8]) if len(kline) > 8 else 0,
    )


def _parse_depth(
    symbol: str,
    data: dict[str, Any],
    timestamp: datetime | None = None,
) -> OrderBookSnapshot:
    """Parse a Binance depth response into an OrderBookSnapshot."""
    ts = timestamp or datetime.now(UTC)
    bids = tuple(
        OrderBookLevel(price=float(level[0]), quantity=float(level[1]))
        for level in data.get("bids", [])
    )
    asks = tuple(
        OrderBookLevel(price=float(level[0]), quantity=float(level[1]))
        for level in data.get("asks", [])
    )
    return OrderBookSnapshot(timestamp=ts, symbol=symbol, bids=bids, asks=asks)


def _parse_trade(symbol: str, data: dict[str, Any]) -> Tick:
    """Parse a Binance trade WebSocket message into a Tick."""
    return Tick(
        timestamp=_ms_to_datetime(data["T"]),
        symbol=symbol,
        price=float(data["p"]),
        quantity=float(data["q"]),
        side=OrderSide.SELL if data.get("m", False) else OrderSide.BUY,
    )


async def _request_with_backoff(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any],
) -> httpx.Response:
    """Make an HTTP GET request with exponential backoff on rate limits."""
    backoff = INITIAL_BACKOFF_SECS
    for attempt in range(MAX_RETRIES):
        response = await client.get(url, params=params)
        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", backoff))
            logger.warning(
                "Binance rate limited (attempt %d/%d), backing off %.1fs",
                attempt + 1,
                MAX_RETRIES,
                retry_after,
            )
            await asyncio.sleep(retry_after)
            backoff = min(backoff * BACKOFF_FACTOR, MAX_BACKOFF_SECS)
            continue
        response.raise_for_status()
        return response
    # Final attempt — let it raise
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response


# ── Bar Adapter ──────────────────────────────────────────────────────────────


class BinanceBarAdapter:
    """BarProvider implementation using Binance REST + WebSocket APIs.

    REST for historical klines, WebSocket for real-time kline streams.
    """

    def __init__(
        self,
        base_url: str = BINANCE_REST_BASE,
        ws_base_url: str = BINANCE_WS_BASE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = base_url
        self._ws_base_url = ws_base_url
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._ws_connections: list[Any] = []

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def get_historical_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1m",
    ) -> list[Bar]:
        """Fetch historical OHLCV bars from Binance REST API.

        Automatically paginates if the requested range exceeds 1000 bars.
        """
        client = self._get_client()
        url = f"{self._base_url}{KLINES_ENDPOINT}"
        start_ms = _iso_to_ms(start)
        end_ms = _iso_to_ms(end)

        all_bars: list[Bar] = []
        current_start = start_ms

        while current_start < end_ms:
            params: dict[str, Any] = {
                "symbol": symbol.upper(),
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ms,
                "limit": MAX_KLINES_PER_REQUEST,
            }
            try:
                response = await _request_with_backoff(client, url, params)
                data = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                logger.warning("Binance klines request failed: %s", exc)
                break

            if not data:
                break

            for kline in data:
                try:
                    all_bars.append(_parse_kline(symbol.upper(), kline))
                except (IndexError, ValueError, TypeError) as exc:
                    logger.debug("Skipping malformed kline: %s", exc)
                    continue

            # Advance past the last kline's open time to avoid duplicates
            last_open_time = int(data[-1][0])
            if last_open_time <= current_start:
                break  # No progress — avoid infinite loop
            current_start = last_open_time + 1

        logger.debug(
            "BinanceBarAdapter: fetched %d bars for %s [%s, %s]",
            len(all_bars),
            symbol,
            start,
            end,
        )
        return all_bars

    async def subscribe(self, symbol: str, interval: str = "1m") -> AsyncIterator[Bar]:
        """Subscribe to real-time kline updates via Binance WebSocket."""
        import websockets

        stream = f"{symbol.lower()}@kline_{interval}"
        uri = f"{self._ws_base_url}{stream}"
        logger.info("BinanceBarAdapter: connecting to %s", uri)

        ws = await websockets.connect(uri)
        self._ws_connections.append(ws)
        try:
            async for raw_msg in ws:
                try:
                    msg = json.loads(raw_msg)
                    kline_data = msg.get("k", {})
                    # Only yield closed bars
                    if not kline_data.get("x", False):
                        continue
                    bar = Bar(
                        timestamp=_ms_to_datetime(kline_data["t"]),
                        symbol=symbol.upper(),
                        open=float(kline_data["o"]),
                        high=float(kline_data["h"]),
                        low=float(kline_data["l"]),
                        close=float(kline_data["c"]),
                        volume=float(kline_data["v"]),
                        bar_count=int(kline_data.get("n", 0)),
                    )
                    yield bar
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                    logger.debug("Skipping malformed WS kline message: %s", exc)
                    continue
        except websockets.ConnectionClosed:
            logger.info("BinanceBarAdapter: WebSocket connection closed")
        finally:
            if ws in self._ws_connections:
                self._ws_connections.remove(ws)

    async def close(self) -> None:
        """Close HTTP client and all WebSocket connections."""
        for ws in self._ws_connections:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                logger.debug("BinanceBarAdapter: error closing WS", exc_info=True)
        self._ws_connections.clear()
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.debug("BinanceBarAdapter: closed")


# ── Tick Adapter ─────────────────────────────────────────────────────────────


class BinanceTickAdapter:
    """TickProvider implementation using Binance WebSocket trade stream."""

    def __init__(
        self,
        ws_base_url: str = BINANCE_WS_BASE,
    ) -> None:
        self._ws_base_url = ws_base_url
        self._ws_connections: list[Any] = []

    async def subscribe(self, symbol: str) -> AsyncIterator[Tick]:
        """Subscribe to real-time trade ticks via Binance WebSocket."""
        import websockets

        stream = f"{symbol.lower()}@trade"
        uri = f"{self._ws_base_url}{stream}"
        logger.info("BinanceTickAdapter: connecting to %s", uri)

        ws = await websockets.connect(uri)
        self._ws_connections.append(ws)
        try:
            async for raw_msg in ws:
                try:
                    data = json.loads(raw_msg)
                    yield _parse_trade(symbol.upper(), data)
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                    logger.debug("Skipping malformed WS trade message: %s", exc)
                    continue
        except websockets.ConnectionClosed:
            logger.info("BinanceTickAdapter: WebSocket connection closed")
        finally:
            if ws in self._ws_connections:
                self._ws_connections.remove(ws)

    async def close(self) -> None:
        """Close all WebSocket connections."""
        for ws in self._ws_connections:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                logger.debug("BinanceTickAdapter: error closing WS", exc_info=True)
        self._ws_connections.clear()
        logger.debug("BinanceTickAdapter: closed")


# ── Order Book Adapter ───────────────────────────────────────────────────────


class BinanceOrderBookAdapter:
    """OrderBookProvider implementation using Binance REST + WebSocket APIs.

    REST for point-in-time snapshots, WebSocket for streaming partial depth.
    """

    def __init__(
        self,
        base_url: str = BINANCE_REST_BASE,
        ws_base_url: str = BINANCE_WS_BASE,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._base_url = base_url
        self._ws_base_url = ws_base_url
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._ws_connections: list[Any] = []

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def get_snapshot(self, symbol: str, depth: int = 20) -> OrderBookSnapshot:
        """Fetch a current order book snapshot from Binance REST API."""
        client = self._get_client()
        url = f"{self._base_url}{DEPTH_ENDPOINT}"
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "limit": depth,
        }
        try:
            response = await _request_with_backoff(client, url, params)
            data = response.json()
            return _parse_depth(symbol.upper(), data)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Binance depth request failed: %s", exc)
            return OrderBookSnapshot(
                timestamp=datetime.now(UTC),
                symbol=symbol.upper(),
                bids=(),
                asks=(),
            )

    async def subscribe(self, symbol: str, depth: int = 20) -> AsyncIterator[OrderBookSnapshot]:
        """Subscribe to real-time partial depth updates via Binance WebSocket."""
        import websockets

        stream = f"{symbol.lower()}@depth{depth}@100ms"
        uri = f"{self._ws_base_url}{stream}"
        logger.info("BinanceOrderBookAdapter: connecting to %s", uri)

        ws = await websockets.connect(uri)
        self._ws_connections.append(ws)
        try:
            async for raw_msg in ws:
                try:
                    data = json.loads(raw_msg)
                    snapshot = _parse_depth(symbol.upper(), data)
                    yield snapshot
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                    logger.debug("Skipping malformed WS depth message: %s", exc)
                    continue
        except websockets.ConnectionClosed:
            logger.info("BinanceOrderBookAdapter: WebSocket connection closed")
        finally:
            if ws in self._ws_connections:
                self._ws_connections.remove(ws)

    async def close(self) -> None:
        """Close HTTP client and all WebSocket connections."""
        for ws in self._ws_connections:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                logger.debug("BinanceOrderBookAdapter: error closing WS", exc_info=True)
        self._ws_connections.clear()
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.debug("BinanceOrderBookAdapter: closed")
