"""StockTwits social sentiment provider."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ── Domain types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class StockTwitsSentiment:
    """Aggregated StockTwits sentiment for a single symbol.

    Attributes:
        bullish_count: Number of bullish messages in the snapshot window.
        bearish_count: Number of bearish messages in the snapshot window.
        sentiment_ratio: Fraction of bullish messages: bullish / (bullish + bearish).
            Returns 0.5 when neither count is available.
        timestamp: UTC datetime of the snapshot.
    """

    bullish_count: int
    bearish_count: int
    sentiment_ratio: float
    timestamp: datetime


# ── Protocol (for dependency injection) ──────────────────────────────────────


@runtime_checkable
class StockTwitsProtocol(Protocol):
    """Dependency-injection interface for StockTwits sentiment providers."""

    async def fetch_sentiment(self, symbol: str) -> StockTwitsSentiment:
        """Fetch the current crowd sentiment for *symbol*.

        Args:
            symbol: Ticker symbol (e.g. ``"AAPL"`` or ``"BTC.X"`` for crypto).

        Returns:
            :class:`StockTwitsSentiment` with current crowd sentiment metrics.
        """
        ...


# ── Implementation ────────────────────────────────────────────────────────────

_ST_SYMBOL_STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"


class StockTwitsSentimentProvider:
    """Async StockTwits sentiment provider using httpx.

    Fetches the latest 30 messages from the StockTwits stream for a symbol and
    counts bullish / bearish sentiment labels.  Messages without an explicit
    sentiment are ignored when computing the ratio.

    Args:
        base_url: Override the base URL for the stream endpoint (useful in tests).
        timeout: HTTP request timeout in seconds (default 10).
        access_token: Optional OAuth access token for authenticated requests
            (higher rate limits).
    """

    def __init__(
        self,
        base_url: str = _ST_SYMBOL_STREAM_URL,
        timeout: float = 10.0,
        access_token: str = "",
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._access_token = access_token
        self._client: Any = None

    def _get_client(self) -> Any:
        """Return (lazily creating) a shared ``httpx.AsyncClient``.

        Returns:
            Configured :class:`httpx.AsyncClient`.
        """
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _build_url(self, symbol: str) -> str:
        """Construct the stream URL for *symbol*.

        Args:
            symbol: StockTwits symbol identifier.

        Returns:
            Fully-qualified stream URL string.
        """
        return self._base_url.format(symbol=symbol)

    def _count_sentiments(self, messages: list[dict[str, Any]]) -> tuple[int, int]:
        """Count bullish and bearish messages in the stream response.

        Args:
            messages: List of message dicts from the StockTwits API.

        Returns:
            Tuple of (bullish_count, bearish_count).
        """
        bullish = 0
        bearish = 0
        for msg in messages:
            entities = msg.get("entities", {})
            sentiment_obj = entities.get("sentiment", None)
            if not sentiment_obj:
                # Try the top-level sentiment key (older API versions)
                sentiment_obj = msg.get("sentiment", None)
            if sentiment_obj and isinstance(sentiment_obj, dict):
                basic = sentiment_obj.get("basic", "")
                if basic == "Bullish":
                    bullish += 1
                elif basic == "Bearish":
                    bearish += 1
        return bullish, bearish

    async def fetch_sentiment(self, symbol: str) -> StockTwitsSentiment:
        """Fetch crowd sentiment for *symbol* from StockTwits.

        Args:
            symbol: Ticker symbol (e.g. ``"AAPL"`` or ``"BTC.X"``).

        Returns:
            :class:`StockTwitsSentiment` snapshot.

        Raises:
            httpx.HTTPStatusError: On non-2xx API responses.
            httpx.RequestError: On network-level errors.
        """
        client = self._get_client()
        url = self._build_url(symbol)
        params: dict[str, Any] = {}
        if self._access_token:
            params["access_token"] = self._access_token

        logger.debug("StockTwitsSentimentProvider: fetching sentiment for %s", symbol)
        response = await client.get(url, params=params)
        response.raise_for_status()

        data: dict[str, Any] = response.json()
        messages: list[dict[str, Any]] = data.get("messages", [])

        bullish_count, bearish_count = self._count_sentiments(messages)
        total = bullish_count + bearish_count
        sentiment_ratio = bullish_count / total if total > 0 else 0.5

        logger.info(
            "StockTwitsSentimentProvider: %s -> bullish=%d, bearish=%d, ratio=%.2f",
            symbol,
            bullish_count,
            bearish_count,
            sentiment_ratio,
        )

        return StockTwitsSentiment(
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            sentiment_ratio=sentiment_ratio,
            timestamp=datetime.now(tz=UTC),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.debug("StockTwitsSentimentProvider: HTTP client closed")
