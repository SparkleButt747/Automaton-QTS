"""Alpha Vantage news & sentiment client."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ── Domain types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class NewsArticle:
    """A single news article returned by Alpha Vantage.

    Attributes:
        headline: Article title / headline.
        summary: Short article summary.
        source: Publisher name.
        published_at: UTC publication datetime.
        ticker_sentiment: Per-ticker sentiment scores from Alpha Vantage
            (list of dicts with keys ``ticker``, ``relevance_score``,
            ``ticker_sentiment_score``, ``ticker_sentiment_label``).
        url: Link to the original article.
    """

    headline: str
    summary: str
    source: str
    published_at: datetime
    ticker_sentiment: list[dict[str, Any]]
    url: str


# ── Protocol (for dependency injection) ──────────────────────────────────────


@runtime_checkable
class AlphaVantageProtocol(Protocol):
    """Dependency-injection interface for Alpha Vantage news clients."""

    async def fetch_news(self, symbol: str, limit: int = 50) -> list[NewsArticle]:
        """Fetch news articles relevant to *symbol*.

        Args:
            symbol: Ticker symbol (e.g. ``"AAPL"``).
            limit: Maximum number of articles to return (default 50).

        Returns:
            List of :class:`NewsArticle` objects ordered newest-first.
        """
        ...


# ── Implementation ────────────────────────────────────────────────────────────

# Alpha Vantage NEWS_SENTIMENT endpoint
_AV_BASE_URL = "https://www.alphavantage.co/query"
_AV_FUNCTION = "NEWS_SENTIMENT"

# Alpha Vantage datetime format: "20240101T120000"
_AV_DT_FMT = "%Y%m%dT%H%M%S"


def _parse_av_datetime(raw: str) -> datetime:
    """Parse an Alpha Vantage ``time_published`` string into a UTC datetime.

    Args:
        raw: Datetime string in ``YYYYMMDDTHHMMSS`` format.

    Returns:
        UTC-aware :class:`datetime` object.
    """
    try:
        return datetime.strptime(raw, _AV_DT_FMT).replace(tzinfo=UTC)
    except (ValueError, TypeError):
        logger.warning("AlphaVantageClient: could not parse datetime %r, using epoch", raw)
        return datetime.fromtimestamp(0, tz=UTC)


class AlphaVantageClient:
    """Async HTTP client for the Alpha Vantage NEWS_SENTIMENT endpoint.

    Uses ``httpx.AsyncClient`` for non-blocking HTTP requests.  Instantiation
    does not make any network calls; the client is created lazily on the first
    :meth:`fetch_news` call.

    Args:
        api_key: Alpha Vantage API key.  Defaults to empty string (will
            fail at request time if not set).
        base_url: Override the API base URL (useful in tests).
        timeout: HTTP request timeout in seconds (default 10).
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = _AV_BASE_URL,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._client: Any = None

    def _get_client(self) -> Any:
        """Return (creating if necessary) a shared httpx.AsyncClient.

        Returns:
            Configured :class:`httpx.AsyncClient` instance.
        """
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    def _parse_article(self, raw: dict[str, Any]) -> NewsArticle:
        """Convert a raw Alpha Vantage feed item dict into a NewsArticle.

        Args:
            raw: Dictionary from the ``feed`` array in the API response.

        Returns:
            Populated :class:`NewsArticle` instance.
        """
        return NewsArticle(
            headline=str(raw.get("title", "")),
            summary=str(raw.get("summary", "")),
            source=str(raw.get("source", "")),
            published_at=_parse_av_datetime(str(raw.get("time_published", ""))),
            ticker_sentiment=list(raw.get("ticker_sentiment", [])),
            url=str(raw.get("url", "")),
        )

    async def fetch_news(self, symbol: str, limit: int = 50) -> list[NewsArticle]:
        """Fetch news articles for *symbol* from Alpha Vantage.

        Calls ``GET /query?function=NEWS_SENTIMENT&tickers=<symbol>&limit=<limit>``.

        Args:
            symbol: Ticker symbol (e.g. ``"AAPL"``).
            limit: Maximum number of articles to return (default 50).

        Returns:
            List of :class:`NewsArticle` objects, ordered newest-first.

        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status code.
            httpx.RequestError: On network-level errors.
        """

        client = self._get_client()
        params: dict[str, Any] = {
            "function": _AV_FUNCTION,
            "tickers": symbol,
            "limit": limit,
            "apikey": self._api_key,
        }

        logger.debug("AlphaVantageClient: fetching news for %s (limit=%d)", symbol, limit)
        response = await client.get(self._base_url, params=params)
        response.raise_for_status()

        data: dict[str, Any] = response.json()
        feed: list[dict[str, Any]] = data.get("feed", [])

        articles = [self._parse_article(item) for item in feed[:limit]]
        logger.info("AlphaVantageClient: received %d articles for %s", len(articles), symbol)
        return articles

    async def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.debug("AlphaVantageClient: HTTP client closed")
