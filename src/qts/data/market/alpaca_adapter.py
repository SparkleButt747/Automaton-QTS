"""Alpaca Markets equities bar data provider.

Implements the BarProvider protocol using raw HTTP (httpx) against the
Alpaca Data API v2. No alpaca-trade-api dependency required.

Historical bars are fetched via REST with automatic pagination.
Real-time bars are streamed via the Alpaca WebSocket endpoint.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any

import httpx

from qts.models.base import Bar

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_DATA_URL = "https://data.alpaca.markets"
DEFAULT_WS_URL = "wss://stream.data.alpaca.markets/v2/iex"
DEFAULT_TIMEOUT = 30.0
MAX_BARS_PER_PAGE = 10_000

# Alpaca uses its own timeframe strings
INTERVAL_MAP: dict[str, str] = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h": "1Hour",
    "1d": "1Day",
}


def _map_interval(interval: str) -> str:
    """Map a canonical interval string to the Alpaca timeframe parameter.

    Args:
        interval: Canonical interval (e.g. "1m", "5m", "1h", "1d").

    Returns:
        Alpaca-format timeframe string.

    Raises:
        ValueError: If the interval is not recognised.
    """
    mapped = INTERVAL_MAP.get(interval)
    if mapped is None:
        raise ValueError(
            f"Unsupported interval '{interval}'. Supported: {', '.join(sorted(INTERVAL_MAP))}"
        )
    return mapped


def _parse_bar(raw: dict[str, Any], symbol: str) -> Bar:
    """Parse a single Alpaca bar JSON object into a Bar dataclass.

    Args:
        raw: Dict with keys t, o, h, l, c, v, n.
        symbol: Ticker symbol to attach.

    Returns:
        Parsed Bar instance.
    """
    return Bar(
        timestamp=datetime.fromisoformat(raw["t"].replace("Z", "+00:00")),
        symbol=symbol,
        open=float(raw["o"]),
        high=float(raw["h"]),
        low=float(raw["l"]),
        close=float(raw["c"]),
        volume=float(raw["v"]),
        bar_count=int(raw.get("n", 0)),
    )


class AlpacaBarAdapter:
    """Bar provider backed by the Alpaca Data API v2 (REST + WebSocket).

    Uses raw httpx for HTTP requests — no alpaca-trade-api dependency.

    Args:
        api_key: Alpaca API key ID.
        api_secret: Alpaca API secret key.
        data_url: Base URL for the Alpaca data API.
        ws_url: WebSocket URL for real-time bar streaming.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        data_url: str = DEFAULT_DATA_URL,
        ws_url: str = DEFAULT_WS_URL,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._data_url = data_url.rstrip("/")
        self._ws_url = ws_url
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return (creating if necessary) a shared httpx.AsyncClient."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                headers={
                    "APCA-API-KEY-ID": self._api_key,
                    "APCA-API-SECRET-KEY": self._api_secret,
                },
            )
        return self._client

    async def get_historical_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1m",
    ) -> list[Bar]:
        """Fetch historical OHLCV bars from Alpaca with automatic pagination.

        Args:
            symbol: US equity ticker (e.g. "AAPL").
            start: ISO 8601 start datetime string.
            end: ISO 8601 end datetime string.
            interval: Bar interval ("1m", "5m", "15m", "30m", "1h", "1d").

        Returns:
            List of Bar objects ordered by timestamp ascending.

        Raises:
            httpx.HTTPStatusError: On 4xx/5xx responses.
            ValueError: If the interval is not supported.
        """
        timeframe = _map_interval(interval)
        client = self._get_client()
        url = f"{self._data_url}/v2/stocks/{symbol}/bars"

        bars: list[Bar] = []
        page_token: str | None = None

        while True:
            params: dict[str, str | int] = {
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "limit": MAX_BARS_PER_PAGE,
            }
            if page_token is not None:
                params["page_token"] = page_token

            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            raw_bars = data.get("bars") or []
            for raw in raw_bars:
                bars.append(_parse_bar(raw, symbol))

            page_token = data.get("next_page_token")
            if not page_token:
                break

            logger.debug(
                "Alpaca: fetched %d bars for %s, paginating (token=%s…)",
                len(raw_bars),
                symbol,
                page_token[:12] if len(page_token) > 12 else page_token,
            )

        logger.info(
            "Alpaca: fetched %d total bars for %s [%s, %s] @ %s",
            len(bars),
            symbol,
            start,
            end,
            interval,
        )
        return bars

    async def subscribe(self, symbol: str, interval: str = "1m") -> AsyncIterator[Bar]:
        """Stream real-time bars via the Alpaca WebSocket.

        Connects to the Alpaca streaming endpoint, authenticates, subscribes
        to bars for the given symbol, and yields Bar objects as they arrive.

        Args:
            symbol: US equity ticker (e.g. "AAPL").
            interval: Bar interval (used for subscription; Alpaca streams 1Min bars by default).

        Yields:
            Bar objects as they arrive from the WebSocket.
        """
        import websockets

        async with websockets.connect(self._ws_url) as ws:
            # Read the welcome message
            welcome = await ws.recv()
            logger.debug("Alpaca WS welcome: %s", welcome)

            # Authenticate
            auth_msg = json.dumps(
                {
                    "action": "auth",
                    "key": self._api_key,
                    "secret": self._api_secret,
                }
            )
            await ws.send(auth_msg)
            auth_resp = await ws.recv()
            logger.debug("Alpaca WS auth response: %s", auth_resp)

            # Subscribe to bars
            sub_msg = json.dumps(
                {
                    "action": "subscribe",
                    "bars": [symbol],
                }
            )
            await ws.send(sub_msg)
            sub_resp = await ws.recv()
            logger.debug("Alpaca WS subscribe response: %s", sub_resp)

            # Yield bars as they arrive
            async for message in ws:
                payload = json.loads(message)
                if not isinstance(payload, list):
                    payload = [payload]
                for item in payload:
                    msg_type = item.get("T")
                    if msg_type == "b":
                        yield _parse_bar(
                            {
                                "t": item["t"],
                                "o": item["o"],
                                "h": item["h"],
                                "l": item["l"],
                                "c": item["c"],
                                "v": item["v"],
                                "n": item.get("n", 0),
                            },
                            symbol=item.get("S", symbol),
                        )

    async def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.debug("AlpacaBarAdapter: HTTP client closed")
