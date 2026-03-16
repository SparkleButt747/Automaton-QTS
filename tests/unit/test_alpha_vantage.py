"""Unit tests for qts.data.news.alpha_vantage.

Verifies:
- _parse_av_datetime parses valid AV datetime strings
- _parse_av_datetime returns epoch on malformed input
- AlphaVantageClient._parse_article converts a raw dict to NewsArticle
- AlphaVantageClient._parse_article handles missing fields gracefully
- AlphaVantageClient.fetch_news returns a list of NewsArticle objects
- AlphaVantageClient.fetch_news handles empty feed response
- AlphaVantageClient.fetch_news raises on HTTP error
- AlphaVantageClient.fetch_news respects the limit parameter
- AlphaVantageClient.close releases the HTTP client
- AlphaVantageClient._get_client is lazily created and reused
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from qts.data.news.alpha_vantage import (
    AlphaVantageClient,
    NewsArticle,
    _parse_av_datetime,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _mock_response(json_data, status_code=200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}", request=MagicMock(), response=resp
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _sample_feed_item(title="Headline", published="20240101T120000") -> dict:
    return {
        "title": title,
        "summary": "A summary",
        "source": "Reuters",
        "time_published": published,
        "ticker_sentiment": [{"ticker": "AAPL", "ticker_sentiment_score": "0.8"}],
        "url": "https://example.com/article",
    }


# ── T-AV-1 ────────────────────────────────────────────────────────────────────


class TestParseAvDatetime:
    def test_parses_valid_datetime(self):
        """_parse_av_datetime correctly parses the AV YYYYMMDDTHHMMSS format."""
        dt = _parse_av_datetime("20240115T093000")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15
        assert dt.hour == 9
        assert dt.minute == 30
        assert dt.tzinfo == timezone.utc

    def test_returns_epoch_on_bad_input(self):
        """_parse_av_datetime returns epoch UTC for malformed strings."""
        dt = _parse_av_datetime("not-a-date")
        assert dt == datetime.fromtimestamp(0, tz=timezone.utc)

    def test_returns_epoch_on_empty_string(self):
        """_parse_av_datetime returns epoch UTC for an empty string."""
        dt = _parse_av_datetime("")
        assert dt == datetime.fromtimestamp(0, tz=timezone.utc)


# ── T-AV-2 ────────────────────────────────────────────────────────────────────


class TestParseArticle:
    def test_parses_full_feed_item(self):
        """_parse_article converts a well-formed feed dict to a NewsArticle."""
        client = AlphaVantageClient(api_key="test")
        item = _sample_feed_item()
        article = client._parse_article(item)

        assert isinstance(article, NewsArticle)
        assert article.headline == "Headline"
        assert article.summary == "A summary"
        assert article.source == "Reuters"
        assert article.url == "https://example.com/article"
        assert len(article.ticker_sentiment) == 1
        assert article.published_at.tzinfo == timezone.utc

    def test_handles_missing_optional_fields(self):
        """_parse_article uses sensible defaults when optional fields are absent."""
        client = AlphaVantageClient(api_key="test")
        article = client._parse_article({})

        assert article.headline == ""
        assert article.summary == ""
        assert article.source == ""
        assert article.ticker_sentiment == []
        assert article.url == ""


# ── T-AV-3 ────────────────────────────────────────────────────────────────────


class TestFetchNews:
    @pytest.mark.asyncio
    async def test_returns_list_of_articles(self):
        """fetch_news returns a list of NewsArticle objects for a valid response."""
        client = AlphaVantageClient(api_key="key")
        feed = [_sample_feed_item("A"), _sample_feed_item("B")]
        response = _mock_response({"feed": feed})

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get.return_value = mock_http

            articles = await client.fetch_news("AAPL")

        assert len(articles) == 2
        assert all(isinstance(a, NewsArticle) for a in articles)
        assert articles[0].headline == "A"

    @pytest.mark.asyncio
    async def test_returns_empty_list_for_empty_feed(self):
        """fetch_news returns [] when the API returns an empty feed."""
        client = AlphaVantageClient(api_key="key")
        response = _mock_response({"feed": []})

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get.return_value = mock_http

            articles = await client.fetch_news("AAPL")

        assert articles == []

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self):
        """fetch_news propagates httpx.HTTPStatusError on a non-2xx response."""
        client = AlphaVantageClient(api_key="key")
        response = _mock_response({}, status_code=429)

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get.return_value = mock_http

            with pytest.raises(httpx.HTTPStatusError):
                await client.fetch_news("AAPL")

    @pytest.mark.asyncio
    async def test_respects_limit_parameter(self):
        """fetch_news returns at most `limit` articles."""
        client = AlphaVantageClient(api_key="key")
        feed = [_sample_feed_item(f"Title {i}") for i in range(10)]
        response = _mock_response({"feed": feed})

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get.return_value = mock_http

            articles = await client.fetch_news("AAPL", limit=3)

        assert len(articles) == 3

    @pytest.mark.asyncio
    async def test_handles_missing_feed_key(self):
        """fetch_news returns [] when response has no 'feed' key."""
        client = AlphaVantageClient(api_key="key")
        response = _mock_response({"note": "API limit reached"})

        with patch.object(client, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get.return_value = mock_http

            articles = await client.fetch_news("AAPL")

        assert articles == []


# ── T-AV-4 ────────────────────────────────────────────────────────────────────


class TestClientLifecycle:
    def test_get_client_lazy_creation(self):
        """_get_client creates the httpx.AsyncClient on first call."""
        client = AlphaVantageClient(api_key="key")
        assert client._client is None
        http = client._get_client()
        assert http is not None
        assert isinstance(http, httpx.AsyncClient)

    def test_get_client_reuses_instance(self):
        """_get_client returns the same instance on repeated calls."""
        client = AlphaVantageClient(api_key="key")
        c1 = client._get_client()
        c2 = client._get_client()
        assert c1 is c2

    @pytest.mark.asyncio
    async def test_close_releases_client(self):
        """close() closes the underlying httpx.AsyncClient."""
        client = AlphaVantageClient(api_key="key")
        mock_http = MagicMock(spec=httpx.AsyncClient)
        mock_http.aclose = AsyncMock()
        client._client = mock_http

        await client.close()

        mock_http.aclose.assert_awaited_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_close_before_use_does_not_raise(self):
        """close() does nothing when called before the client is created."""
        client = AlphaVantageClient(api_key="key")
        await client.close()  # should not raise
