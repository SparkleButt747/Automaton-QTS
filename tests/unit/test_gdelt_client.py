"""Unit tests for qts.data.geopolitical.gdelt_client.

Verifies:
- Parses valid GDELT JSON into GeopoliticalEvent objects
- Handles empty articles list
- Handles empty response / no "articles" key
- Handles HTTP error gracefully (returns empty list)
- Handles malformed JSON
- Protocol conformance
- close() works
- Intensity estimation (negative tone -> high intensity)
- Country extraction
"""

from __future__ import annotations

from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from qts.data.geopolitical.gdelt_client import GDELTClient, GDELTClientProtocol
from qts.nlp.gdelt import GeopoliticalEvent

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def client() -> GDELTClient:
    """A GDELTClient with a short timeout for tests."""
    return GDELTClient(base_url="https://api.gdeltproject.org/api/v2/doc/doc", timeout=5.0)


def _make_article(
    url: str = "https://example.com/article1",
    title: str = "Military conflict in region",
    eventcode: str = "190",
    tone: str = "-5.0,3.0,2.0",
    sourcecountry: str = "US",
) -> dict:  # type: ignore[type-arg]
    """Build a minimal GDELT Doc API article dict."""
    return {
        "url": url,
        "title": title,
        "eventcode": eventcode,
        "tone": tone,
        "sourcecountry": sourcecountry,
    }


def _make_gdelt_response(articles: list) -> dict:  # type: ignore[type-arg]
    """Wrap articles in a GDELT-style response envelope."""
    return {"articles": articles}


def _mock_httpx_response(json_data: object, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=mock_resp,
        )
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


# ── Protocol conformance ──────────────────────────────────────────────────────


class TestProtocolConformance:
    def test_gdelt_client_implements_protocol(self, client: GDELTClient) -> None:
        """GDELTClient must satisfy GDELTClientProtocol."""
        assert isinstance(client, GDELTClientProtocol)

    def test_protocol_has_fetch_events(self, client: GDELTClient) -> None:
        assert hasattr(client, "fetch_events")
        assert callable(client.fetch_events)

    def test_protocol_has_close(self, client: GDELTClient) -> None:
        assert hasattr(client, "close")
        assert callable(client.close)


# ── Parsing valid articles ────────────────────────────────────────────────────


class TestParseValidArticles:
    @pytest.mark.asyncio
    async def test_parses_single_article(self, client: GDELTClient) -> None:
        article = _make_article(
            url="https://example.com/news1",
            title="Sanctions imposed on country",
            eventcode="163",
            tone="-8.0,2.0,6.0",
            sourcecountry="RU",
        )
        response = _mock_httpx_response(_make_gdelt_response([article]))

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            events = await client.fetch_events(query="sanctions")

        assert len(events) == 1
        event = events[0]
        assert isinstance(event, GeopoliticalEvent)
        assert event.event_id == "https://example.com/news1"
        assert event.source_url == "https://example.com/news1"
        assert event.description == "Sanctions imposed on country"
        assert event.cameo_code == "163"
        assert "RU" in event.countries
        assert event.timestamp.tzinfo is not None
        assert event.timestamp.tzinfo == UTC

    @pytest.mark.asyncio
    async def test_parses_multiple_articles(self, client: GDELTClient) -> None:
        articles = [
            _make_article(url=f"https://example.com/{i}", title=f"Event {i}") for i in range(5)
        ]
        response = _mock_httpx_response(_make_gdelt_response(articles))

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            events = await client.fetch_events()

        assert len(events) == 5

    @pytest.mark.asyncio
    async def test_article_intensity_in_unit_interval(self, client: GDELTClient) -> None:
        article = _make_article(tone="-12.0,5.0,7.0")
        response = _mock_httpx_response(_make_gdelt_response([article]))

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            events = await client.fetch_events()

        assert len(events) == 1
        assert 0.0 <= events[0].intensity <= 1.0

    @pytest.mark.asyncio
    async def test_article_default_cameo_code(self, client: GDELTClient) -> None:
        """Article with no eventcode should default to '190'."""
        article = {"url": "https://example.com/x", "title": "Conflict event"}
        response = _mock_httpx_response(_make_gdelt_response([article]))

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            events = await client.fetch_events()

        assert events[0].cameo_code == "190"


# ── Empty / missing articles ──────────────────────────────────────────────────


class TestEmptyArticles:
    @pytest.mark.asyncio
    async def test_empty_articles_list(self, client: GDELTClient) -> None:
        response = _mock_httpx_response(_make_gdelt_response([]))

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            events = await client.fetch_events()

        assert events == []

    @pytest.mark.asyncio
    async def test_missing_articles_key(self, client: GDELTClient) -> None:
        """Response with no 'articles' key should return empty list."""
        response = _mock_httpx_response({})

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            events = await client.fetch_events()

        assert events == []

    @pytest.mark.asyncio
    async def test_null_articles_key(self, client: GDELTClient) -> None:
        """Response with articles=null should return empty list."""
        response = _mock_httpx_response({"articles": None})

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            events = await client.fetch_events()

        assert events == []


# ── Error handling ────────────────────────────────────────────────────────────


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_http_error_returns_empty_list(self, client: GDELTClient) -> None:
        """HTTP 500 response should log a warning and return []."""
        response = _mock_httpx_response({}, status_code=500)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            events = await client.fetch_events()

        assert events == []

    @pytest.mark.asyncio
    async def test_http_404_returns_empty_list(self, client: GDELTClient) -> None:
        response = _mock_httpx_response({}, status_code=404)

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            events = await client.fetch_events()

        assert events == []

    @pytest.mark.asyncio
    async def test_network_error_returns_empty_list(self, client: GDELTClient) -> None:
        """Network-level httpx.HTTPError should return empty list, not raise."""
        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_get_client.return_value = mock_http

            events = await client.fetch_events()

        assert events == []

    @pytest.mark.asyncio
    async def test_json_decode_error_returns_empty_list(self, client: GDELTClient) -> None:
        """ValueError from .json() should return empty list."""
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = ValueError("Invalid JSON")

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=mock_resp)
            mock_get_client.return_value = mock_http

            events = await client.fetch_events()

        assert events == []

    @pytest.mark.asyncio
    async def test_malformed_article_is_skipped(self, client: GDELTClient) -> None:
        """A malformed article dict should be skipped; valid ones still parsed."""
        articles = [
            _make_article(url="https://example.com/good"),
            None,  # malformed
            _make_article(url="https://example.com/also-good"),
        ]
        response = _mock_httpx_response({"articles": articles})

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            # None articles will raise TypeError and be skipped
            events = await client.fetch_events()

        # 2 valid articles should produce 2 events (None is skipped)
        assert len(events) == 2


# ── Intensity estimation ──────────────────────────────────────────────────────


class TestIntensityEstimation:
    def test_negative_tone_high_intensity(self, client: GDELTClient) -> None:
        """Strongly negative tone should produce high intensity (close to 1)."""
        article = {"tone": "-15.0,5.0,10.0"}
        intensity = client._estimate_intensity(article)
        assert intensity == pytest.approx(1.0)

    def test_zero_tone_zero_intensity(self, client: GDELTClient) -> None:
        """Neutral tone (0) should produce zero intensity."""
        article = {"tone": "0.0,0.0,0.0"}
        intensity = client._estimate_intensity(article)
        assert intensity == pytest.approx(0.0)

    def test_positive_tone_zero_intensity(self, client: GDELTClient) -> None:
        """Positive tone should clamp to zero intensity."""
        article = {"tone": "10.0,2.0,8.0"}
        intensity = client._estimate_intensity(article)
        assert intensity == pytest.approx(0.0)

    def test_moderate_negative_tone_partial_intensity(self, client: GDELTClient) -> None:
        """Tone of -7.5 should yield intensity of 0.5."""
        article = {"tone": "-7.5"}
        intensity = client._estimate_intensity(article)
        assert intensity == pytest.approx(0.5)

    def test_missing_tone_defaults_to_zero(self, client: GDELTClient) -> None:
        """Missing 'tone' key should default to 0.0 intensity."""
        article = {}  # type: ignore[var-annotated]
        intensity = client._estimate_intensity(article)
        assert intensity == pytest.approx(0.0)

    def test_tone_as_integer(self, client: GDELTClient) -> None:
        """Tone as an integer value should be handled correctly."""
        article = {"tone": -10}
        intensity = client._estimate_intensity(article)
        assert intensity == pytest.approx(min(1.0, 10.0 / 15.0))

    def test_intensity_always_in_unit_interval(self, client: GDELTClient) -> None:
        """Intensity must always be in [0, 1] regardless of tone value."""
        for tone_val in [-100.0, -50.0, -15.0, 0.0, 15.0, 100.0]:
            article = {"tone": str(tone_val)}
            intensity = client._estimate_intensity(article)
            assert 0.0 <= intensity <= 1.0, f"intensity out of range for tone={tone_val}"

    def test_tone_string_with_comma_uses_first_field(self, client: GDELTClient) -> None:
        """GDELT tone field is comma-separated; only the first value is tone."""
        article = {"tone": "-5.0,3.2,1.8,0.5"}
        intensity = client._estimate_intensity(article)
        expected = min(1.0, max(0.0, 5.0 / 15.0))
        assert intensity == pytest.approx(expected)

    def test_malformed_tone_string_defaults_zero(self, client: GDELTClient) -> None:
        """Non-parseable tone string should default to zero intensity."""
        article = {"tone": "not-a-number"}
        intensity = client._estimate_intensity(article)
        assert intensity == pytest.approx(0.0)


# ── Country extraction ────────────────────────────────────────────────────────


class TestCountryExtraction:
    def test_extracts_source_country(self, client: GDELTClient) -> None:
        article = {"sourcecountry": "US"}
        countries = client._extract_countries(article)
        assert countries == ["US"]

    def test_missing_source_country_returns_empty(self, client: GDELTClient) -> None:
        article = {"title": "Some event"}
        countries = client._extract_countries(article)
        assert countries == []

    def test_empty_source_country_returns_empty(self, client: GDELTClient) -> None:
        article = {"sourcecountry": ""}
        countries = client._extract_countries(article)
        assert countries == []

    def test_country_preserved_in_event(self, client: GDELTClient) -> None:
        article = _make_article(sourcecountry="UK")
        events = client._parse_articles({"articles": [article]})
        assert len(events) == 1
        assert "UK" in events[0].countries


# ── close() behaviour ─────────────────────────────────────────────────────────


class TestClose:
    @pytest.mark.asyncio
    async def test_close_before_any_request(self, client: GDELTClient) -> None:
        """close() with no client created should not raise."""
        await client.close()  # no-op — client never created

    @pytest.mark.asyncio
    async def test_close_after_request(self, client: GDELTClient) -> None:
        """close() after using the client should call aclose on the httpx client."""
        mock_http_client = MagicMock(spec=httpx.AsyncClient)
        mock_http_client.is_closed = False
        mock_http_client.aclose = AsyncMock()

        client._client = mock_http_client  # type: ignore[assignment]
        await client.close()

        mock_http_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_idempotent_when_already_closed(self, client: GDELTClient) -> None:
        """close() on an already-closed client should not raise."""
        mock_http_client = MagicMock(spec=httpx.AsyncClient)
        mock_http_client.is_closed = True
        mock_http_client.aclose = AsyncMock()

        client._client = mock_http_client  # type: ignore[assignment]
        await client.close()

        # aclose should NOT be called because is_closed is True
        mock_http_client.aclose.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_close_called_twice_no_error(self, client: GDELTClient) -> None:
        """Calling close() twice should not raise."""
        await client.close()
        await client.close()


# ── fetch_recent_conflict_events ─────────────────────────────────────────────


class TestFetchRecentConflictEvents:
    @pytest.mark.asyncio
    async def test_delegates_to_fetch_events(self, client: GDELTClient) -> None:
        """fetch_recent_conflict_events should call fetch_events internally."""
        articles = [_make_article()]
        response = _mock_httpx_response(_make_gdelt_response(articles))

        with patch.object(client, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            events = await client.fetch_recent_conflict_events(hours=48)

        assert len(events) == 1
        assert isinstance(events[0], GeopoliticalEvent)
