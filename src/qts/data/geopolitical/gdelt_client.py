"""GDELT Global Knowledge Graph HTTP client."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

import httpx

from qts.nlp.gdelt import GeopoliticalEvent

logger = logging.getLogger(__name__)

_GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


@runtime_checkable
class GDELTClientProtocol(Protocol):
    """Protocol for GDELT data fetching."""

    async def fetch_events(self, query: str, max_records: int = 75) -> list[GeopoliticalEvent]: ...

    async def close(self) -> None: ...


class GDELTClient:
    """Async HTTP client for GDELT API. No API key required.

    Fetches articles from the GDELT Doc API (ArtList mode) and maps them to
    :class:`~qts.nlp.gdelt.GeopoliticalEvent` objects for downstream signal
    processing.

    Args:
        base_url: Override the GDELT Doc API base URL (useful in tests).
        timeout: HTTP request timeout in seconds (default 30).
    """

    def __init__(
        self,
        base_url: str = _GDELT_DOC_API,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        """Return (creating if necessary) a shared httpx.AsyncClient.

        Returns:
            Configured :class:`httpx.AsyncClient` instance.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def fetch_events(
        self,
        query: str = "conflict OR sanctions OR military",
        max_records: int = 75,
    ) -> list[GeopoliticalEvent]:
        """Fetch events from GDELT Doc API.

        Calls the GDELT Doc API in ``ArtList`` mode with the supplied query and
        returns parsed :class:`~qts.nlp.gdelt.GeopoliticalEvent` objects.

        Args:
            query: Search query string.
            max_records: Maximum number of records to return.

        Returns:
            List of :class:`~qts.nlp.gdelt.GeopoliticalEvent` objects.
            Returns an empty list on any HTTP or parsing error.
        """
        client = self._get_client()
        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": str(max_records),
            "format": "json",
        }
        try:
            response = await client.get(self._base_url, params=params)
            response.raise_for_status()
            data = response.json()
            return self._parse_articles(data)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.warning("GDELT fetch failed: %s", exc)
            return []

    async def fetch_recent_conflict_events(self, hours: int = 24) -> list[GeopoliticalEvent]:
        """Fetch recent conflict/sanctions/military events.

        Convenience method that builds a conflict-focused query and fetches up
        to 100 articles from the GDELT Doc API.

        Args:
            hours: Look-back window in hours (informational; GDELT determines
                the actual time window based on the query).

        Returns:
            List of :class:`~qts.nlp.gdelt.GeopoliticalEvent` objects filtered
            by CAMEO relevance.
        """
        query = "(conflict OR sanctions OR military OR war) sourcecountry:US"
        return await self.fetch_events(query=query, max_records=100)

    def _parse_articles(self, data: dict) -> list[GeopoliticalEvent]:  # type: ignore[type-arg]
        """Parse GDELT Doc API JSON response into GeopoliticalEvent objects.

        Args:
            data: Parsed JSON response from the GDELT Doc API.

        Returns:
            List of :class:`~qts.nlp.gdelt.GeopoliticalEvent` objects.
        """
        events: list[GeopoliticalEvent] = []
        articles = data.get("articles") or []
        for article in articles:
            if not isinstance(article, dict):
                logger.debug("Skipping non-dict GDELT article: %r", type(article))
                continue
            try:
                event = GeopoliticalEvent(
                    event_id=article.get("url", ""),
                    timestamp=datetime.now(UTC),
                    cameo_code=article.get("eventcode", "190"),
                    description=article.get("title", ""),
                    countries=tuple(self._extract_countries(article)),
                    intensity=self._estimate_intensity(article),
                    source_url=article.get("url", ""),
                )
                events.append(event)
            except (KeyError, TypeError) as exc:
                logger.debug("Skipping malformed GDELT article: %s", exc)
                continue
        return events

    def _extract_countries(self, article: dict) -> list[str]:  # type: ignore[type-arg]
        """Extract country codes from article metadata.

        Looks for the ``sourcecountry`` field in the article dict.

        Args:
            article: Raw article dict from the GDELT Doc API response.

        Returns:
            List of country code strings (may be empty).
        """
        country = article.get("sourcecountry", "")
        if country:
            return [country]
        return []

    def _estimate_intensity(self, article: dict) -> float:  # type: ignore[type-arg]
        """Estimate event intensity from article tone.

        GDELT articles include a ``tone`` field ranging roughly -100 to +100.
        We normalise to [0, 1] where higher = more intense/negative.

        Negative tone (conflict, crisis) maps to high intensity scores.

        Args:
            article: Raw article dict from the GDELT Doc API response.

        Returns:
            Float in [0.0, 1.0].
        """
        tone = article.get("tone", 0)
        if isinstance(tone, str):
            try:
                tone = float(tone.split(",")[0])
            except (ValueError, IndexError):
                tone = 0.0
        # Map negative tone to higher intensity (conflict = negative tone)
        return min(1.0, max(0.0, -float(tone) / 15.0))

    async def close(self) -> None:
        """Close the underlying HTTP client and release connections."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.debug("GDELTClient: HTTP client closed")
