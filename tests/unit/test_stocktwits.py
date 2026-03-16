"""Unit tests for qts.data.social.stocktwits.

Verifies:
- StockTwitsSentimentProvider._build_url constructs the correct endpoint
- _count_sentiments correctly counts bullish and bearish messages
- _count_sentiments handles messages with no sentiment field
- _count_sentiments falls back to top-level sentiment key
- fetch_sentiment returns a StockTwitsSentiment for a valid response
- fetch_sentiment computes sentiment_ratio correctly
- fetch_sentiment returns ratio=0.5 when no labelled messages
- fetch_sentiment raises on HTTP error
- fetch_sentiment passes access_token as query param when set
- close() releases the HTTP client
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from qts.data.social.stocktwits import StockTwitsSentiment, StockTwitsSentimentProvider

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


def _make_msg(basic: str | None = None, use_entities: bool = True) -> dict:
    if basic is None:
        return {"body": "No sentiment here"}
    if use_entities:
        return {"entities": {"sentiment": {"basic": basic}}, "body": "msg"}
    return {"sentiment": {"basic": basic}, "body": "msg"}


# ── T-ST-1 ────────────────────────────────────────────────────────────────────


class TestBuildUrl:
    def test_builds_correct_url(self):
        """_build_url interpolates the symbol into the URL template."""
        provider = StockTwitsSentimentProvider()
        url = provider._build_url("AAPL")
        assert "AAPL" in url
        assert url.startswith("https://")


# ── T-ST-2 ────────────────────────────────────────────────────────────────────


class TestCountSentiments:
    def test_counts_bullish_and_bearish(self):
        """_count_sentiments correctly tallies Bullish and Bearish messages."""
        provider = StockTwitsSentimentProvider()
        messages = [
            _make_msg("Bullish"),
            _make_msg("Bullish"),
            _make_msg("Bearish"),
        ]
        bullish, bearish = provider._count_sentiments(messages)
        assert bullish == 2
        assert bearish == 1

    def test_ignores_messages_without_sentiment(self):
        """_count_sentiments ignores messages that have no sentiment field."""
        provider = StockTwitsSentimentProvider()
        messages = [_make_msg(None), _make_msg(None)]
        bullish, bearish = provider._count_sentiments(messages)
        assert bullish == 0
        assert bearish == 0

    def test_falls_back_to_top_level_sentiment_key(self):
        """_count_sentiments uses the top-level 'sentiment' key as a fallback."""
        provider = StockTwitsSentimentProvider()
        messages = [_make_msg("Bullish", use_entities=False)]
        bullish, bearish = provider._count_sentiments(messages)
        assert bullish == 1
        assert bearish == 0

    def test_empty_messages_returns_zeros(self):
        """_count_sentiments returns (0, 0) for an empty message list."""
        provider = StockTwitsSentimentProvider()
        b, be = provider._count_sentiments([])
        assert b == 0
        assert be == 0


# ── T-ST-3 ────────────────────────────────────────────────────────────────────


class TestFetchSentiment:
    @pytest.mark.asyncio
    async def test_returns_sentiment_object(self):
        """fetch_sentiment returns a StockTwitsSentiment for a valid response."""
        provider = StockTwitsSentimentProvider()
        messages = [_make_msg("Bullish"), _make_msg("Bullish"), _make_msg("Bearish")]
        response = _mock_response({"messages": messages})

        with patch.object(provider, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get.return_value = mock_http

            result = await provider.fetch_sentiment("AAPL")

        assert isinstance(result, StockTwitsSentiment)
        assert result.bullish_count == 2
        assert result.bearish_count == 1

    @pytest.mark.asyncio
    async def test_sentiment_ratio_computed_correctly(self):
        """fetch_sentiment computes sentiment_ratio as bullish / (bullish + bearish)."""
        provider = StockTwitsSentimentProvider()
        messages = [_make_msg("Bullish"), _make_msg("Bullish"), _make_msg("Bullish"), _make_msg("Bearish")]
        response = _mock_response({"messages": messages})

        with patch.object(provider, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get.return_value = mock_http

            result = await provider.fetch_sentiment("AAPL")

        assert result.sentiment_ratio == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_returns_half_ratio_when_no_labelled_messages(self):
        """fetch_sentiment returns ratio=0.5 when no messages have sentiment labels."""
        provider = StockTwitsSentimentProvider()
        response = _mock_response({"messages": [_make_msg(None), _make_msg(None)]})

        with patch.object(provider, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get.return_value = mock_http

            result = await provider.fetch_sentiment("AAPL")

        assert result.sentiment_ratio == 0.5

    @pytest.mark.asyncio
    async def test_returns_half_ratio_for_empty_messages(self):
        """fetch_sentiment returns ratio=0.5 when the messages list is empty."""
        provider = StockTwitsSentimentProvider()
        response = _mock_response({"messages": []})

        with patch.object(provider, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get.return_value = mock_http

            result = await provider.fetch_sentiment("AAPL")

        assert result.sentiment_ratio == 0.5
        assert result.bullish_count == 0
        assert result.bearish_count == 0

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self):
        """fetch_sentiment propagates httpx.HTTPStatusError on a non-2xx response."""
        provider = StockTwitsSentimentProvider()
        response = _mock_response({}, status_code=429)

        with patch.object(provider, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get.return_value = mock_http

            with pytest.raises(httpx.HTTPStatusError):
                await provider.fetch_sentiment("AAPL")

    @pytest.mark.asyncio
    async def test_passes_access_token_when_set(self):
        """fetch_sentiment includes access_token in the request params when configured."""
        provider = StockTwitsSentimentProvider(access_token="my-token")
        response = _mock_response({"messages": []})

        with patch.object(provider, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get.return_value = mock_http

            await provider.fetch_sentiment("AAPL")

        call_kwargs = mock_http.get.call_args
        params = call_kwargs.kwargs.get("params", {}) or call_kwargs.args[1] if len(call_kwargs.args) > 1 else {}
        # access_token must appear in the call arguments
        assert "access_token" in str(call_kwargs)


# ── T-ST-4 ────────────────────────────────────────────────────────────────────


class TestClientLifecycle:
    def test_get_client_lazy_creation(self):
        """_get_client creates the httpx.AsyncClient on first call."""
        provider = StockTwitsSentimentProvider()
        assert provider._client is None
        client = provider._get_client()
        assert isinstance(client, httpx.AsyncClient)

    def test_get_client_reused(self):
        """_get_client returns the same instance on repeated calls."""
        provider = StockTwitsSentimentProvider()
        c1 = provider._get_client()
        c2 = provider._get_client()
        assert c1 is c2

    @pytest.mark.asyncio
    async def test_close_releases_client(self):
        """close() closes the underlying httpx.AsyncClient."""
        provider = StockTwitsSentimentProvider()
        mock_http = MagicMock(spec=httpx.AsyncClient)
        mock_http.aclose = AsyncMock()
        provider._client = mock_http

        await provider.close()

        mock_http.aclose.assert_awaited_once()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_close_before_use_does_not_raise(self):
        """close() is a no-op when called before the client is created."""
        provider = StockTwitsSentimentProvider()
        await provider.close()  # should not raise
