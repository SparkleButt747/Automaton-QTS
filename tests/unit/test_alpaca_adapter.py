"""Unit tests for qts.data.market.alpaca_adapter.

Verifies:
- Historical bars JSON parsing
- Interval mapping (1m -> 1Min, etc.)
- Empty response handling
- Pagination handling
- Error responses (401, 429, 500)
- close() behaviour
- _parse_bar edge cases
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from qts.data.market.alpaca_adapter import (
    INTERVAL_MAP,
    AlpacaBarAdapter,
    _map_interval,
    _parse_bar,
)
from qts.models.base import Bar

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def adapter() -> AlpacaBarAdapter:
    """An AlpacaBarAdapter with test credentials."""
    return AlpacaBarAdapter(
        api_key="test-key-id",
        api_secret="test-secret-key",
        data_url="https://data.alpaca.markets",
        timeout=5.0,
    )


def _make_alpaca_bar(
    t: str = "2024-06-15T14:30:00Z",
    o: float = 150.0,
    h: float = 151.5,
    low: float = 149.5,
    c: float = 151.0,
    v: int = 10_000,
    n: int = 42,
) -> dict:
    """Build a minimal Alpaca bar dict."""
    return {"t": t, "o": o, "h": h, "l": low, "c": c, "v": v, "n": n}


def _make_bars_response(
    bars: list[dict] | None = None,
    next_page_token: str | None = None,
) -> dict:
    """Wrap bars in an Alpaca-style response envelope."""
    result: dict = {"bars": bars if bars is not None else []}
    if next_page_token is not None:
        result["next_page_token"] = next_page_token
    else:
        result["next_page_token"] = None
    return result


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


# ── Interval mapping ─────────────────────────────────────────────────────────


class TestIntervalMapping:
    """T-ALP-1: Interval string mapping to Alpaca timeframe format."""

    def test_1m_maps_to_1min(self) -> None:
        assert _map_interval("1m") == "1Min"

    def test_5m_maps_to_5min(self) -> None:
        assert _map_interval("5m") == "5Min"

    def test_15m_maps_to_15min(self) -> None:
        assert _map_interval("15m") == "15Min"

    def test_30m_maps_to_30min(self) -> None:
        assert _map_interval("30m") == "30Min"

    def test_1h_maps_to_1hour(self) -> None:
        assert _map_interval("1h") == "1Hour"

    def test_1d_maps_to_1day(self) -> None:
        assert _map_interval("1d") == "1Day"

    def test_unsupported_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported interval '2h'"):
            _map_interval("2h")

    def test_all_interval_map_entries_covered(self) -> None:
        """Every key in INTERVAL_MAP must roundtrip through _map_interval."""
        for key, expected in INTERVAL_MAP.items():
            assert _map_interval(key) == expected


# ── Bar parsing ──────────────────────────────────────────────────────────────


class TestParseBar:
    """T-ALP-2: Parsing a single Alpaca bar JSON into a Bar dataclass."""

    def test_parses_valid_bar(self) -> None:
        raw = _make_alpaca_bar()
        bar = _parse_bar(raw, "AAPL")

        assert isinstance(bar, Bar)
        assert bar.symbol == "AAPL"
        assert bar.open == 150.0
        assert bar.high == 151.5
        assert bar.low == 149.5
        assert bar.close == 151.0
        assert bar.volume == 10_000.0
        assert bar.bar_count == 42

    def test_timestamp_is_utc(self) -> None:
        raw = _make_alpaca_bar(t="2024-01-01T00:00:00Z")
        bar = _parse_bar(raw, "TSLA")
        assert bar.timestamp.year == 2024
        assert bar.timestamp.month == 1
        assert bar.timestamp.day == 1

    def test_missing_n_defaults_to_zero(self) -> None:
        raw = {"t": "2024-06-15T14:30:00Z", "o": 100, "h": 101, "l": 99, "c": 100, "v": 500}
        bar = _parse_bar(raw, "SPY")
        assert bar.bar_count == 0

    def test_bar_is_frozen(self) -> None:
        raw = _make_alpaca_bar()
        bar = _parse_bar(raw, "AAPL")
        with pytest.raises(AttributeError):
            bar.close = 999.0  # type: ignore[misc]


# ── Historical bars (REST) ───────────────────────────────────────────────────


class TestGetHistoricalBars:
    """T-ALP-3: Fetching historical bars via the REST API."""

    @pytest.mark.asyncio
    async def test_parses_bars_from_response(self, adapter: AlpacaBarAdapter) -> None:
        bars_data = [
            _make_alpaca_bar(t="2024-06-15T14:30:00Z", o=150.0, c=151.0),
            _make_alpaca_bar(t="2024-06-15T14:31:00Z", o=151.0, c=152.0),
        ]
        response = _mock_httpx_response(_make_bars_response(bars_data))

        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            bars = await adapter.get_historical_bars(
                "AAPL", "2024-06-15", "2024-06-16", interval="1m"
            )

        assert len(bars) == 2
        assert all(isinstance(b, Bar) for b in bars)
        assert bars[0].open == 150.0
        assert bars[1].close == 152.0

    @pytest.mark.asyncio
    async def test_empty_bars_response(self, adapter: AlpacaBarAdapter) -> None:
        response = _mock_httpx_response(_make_bars_response([]))

        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            bars = await adapter.get_historical_bars("AAPL", "2024-06-15", "2024-06-16")

        assert bars == []

    @pytest.mark.asyncio
    async def test_null_bars_key(self, adapter: AlpacaBarAdapter) -> None:
        """Response with bars=null should return empty list."""
        response = _mock_httpx_response({"bars": None, "next_page_token": None})

        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            bars = await adapter.get_historical_bars("AAPL", "2024-06-15", "2024-06-16")

        assert bars == []

    @pytest.mark.asyncio
    async def test_pagination_fetches_all_pages(self, adapter: AlpacaBarAdapter) -> None:
        page1_bars = [_make_alpaca_bar(t="2024-06-15T14:30:00Z")]
        page2_bars = [_make_alpaca_bar(t="2024-06-15T14:31:00Z")]

        resp_page1 = _mock_httpx_response(_make_bars_response(page1_bars, next_page_token="abc123"))
        resp_page2 = _mock_httpx_response(_make_bars_response(page2_bars, next_page_token=None))

        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=[resp_page1, resp_page2])
            mock_get_client.return_value = mock_http

            bars = await adapter.get_historical_bars("AAPL", "2024-06-15", "2024-06-16")

        assert len(bars) == 2
        assert mock_http.get.await_count == 2

    @pytest.mark.asyncio
    async def test_passes_correct_params(self, adapter: AlpacaBarAdapter) -> None:
        response = _mock_httpx_response(_make_bars_response([]))

        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            await adapter.get_historical_bars("TSLA", "2024-01-01", "2024-01-31", interval="5m")

        call_kwargs = mock_http.get.call_args
        assert "TSLA" in call_kwargs.args[0]
        params = call_kwargs.kwargs["params"]
        assert params["timeframe"] == "5Min"
        assert params["start"] == "2024-01-01"
        assert params["end"] == "2024-01-31"


# ── Error handling ───────────────────────────────────────────────────────────


class TestErrorHandling:
    """T-ALP-4: HTTP error responses from the Alpaca API."""

    @pytest.mark.asyncio
    async def test_401_raises_http_error(self, adapter: AlpacaBarAdapter) -> None:
        """Unauthorised response should propagate as HTTPStatusError."""
        response = _mock_httpx_response({}, status_code=401)

        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            with pytest.raises(httpx.HTTPStatusError):
                await adapter.get_historical_bars("AAPL", "2024-06-15", "2024-06-16")

    @pytest.mark.asyncio
    async def test_429_raises_http_error(self, adapter: AlpacaBarAdapter) -> None:
        """Rate limited response should propagate as HTTPStatusError."""
        response = _mock_httpx_response({}, status_code=429)

        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            with pytest.raises(httpx.HTTPStatusError):
                await adapter.get_historical_bars("AAPL", "2024-06-15", "2024-06-16")

    @pytest.mark.asyncio
    async def test_500_raises_http_error(self, adapter: AlpacaBarAdapter) -> None:
        """Server error should propagate as HTTPStatusError."""
        response = _mock_httpx_response({}, status_code=500)

        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get_client.return_value = mock_http

            with pytest.raises(httpx.HTTPStatusError):
                await adapter.get_historical_bars("AAPL", "2024-06-15", "2024-06-16")

    @pytest.mark.asyncio
    async def test_network_error_propagates(self, adapter: AlpacaBarAdapter) -> None:
        """Network-level httpx error should propagate."""
        with patch.object(adapter, "_get_client") as mock_get_client:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            mock_get_client.return_value = mock_http

            with pytest.raises(httpx.ConnectError):
                await adapter.get_historical_bars("AAPL", "2024-06-15", "2024-06-16")

    @pytest.mark.asyncio
    async def test_invalid_interval_raises_value_error(self, adapter: AlpacaBarAdapter) -> None:
        with pytest.raises(ValueError, match="Unsupported interval"):
            await adapter.get_historical_bars("AAPL", "2024-06-15", "2024-06-16", interval="3m")


# ── close() behaviour ────────────────────────────────────────────────────────


class TestClose:
    """T-ALP-5: Adapter close behaviour."""

    @pytest.mark.asyncio
    async def test_close_before_any_request(self, adapter: AlpacaBarAdapter) -> None:
        """close() with no client created should not raise."""
        await adapter.close()

    @pytest.mark.asyncio
    async def test_close_after_request(self, adapter: AlpacaBarAdapter) -> None:
        mock_http_client = MagicMock(spec=httpx.AsyncClient)
        mock_http_client.is_closed = False
        mock_http_client.aclose = AsyncMock()

        adapter._client = mock_http_client
        await adapter.close()

        mock_http_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_idempotent_when_already_closed(self, adapter: AlpacaBarAdapter) -> None:
        mock_http_client = MagicMock(spec=httpx.AsyncClient)
        mock_http_client.is_closed = True
        mock_http_client.aclose = AsyncMock()

        adapter._client = mock_http_client
        await adapter.close()

        mock_http_client.aclose.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_close_called_twice_no_error(self, adapter: AlpacaBarAdapter) -> None:
        await adapter.close()
        await adapter.close()


# ── Client initialisation ────────────────────────────────────────────────────


class TestClientInit:
    """T-ALP-6: Lazy client creation and header injection."""

    def test_client_not_created_on_init(self, adapter: AlpacaBarAdapter) -> None:
        assert adapter._client is None

    def test_get_client_creates_client(self, adapter: AlpacaBarAdapter) -> None:
        client = adapter._get_client()
        assert isinstance(client, httpx.AsyncClient)
        assert client.headers["APCA-API-KEY-ID"] == "test-key-id"
        assert client.headers["APCA-API-SECRET-KEY"] == "test-secret-key"

    def test_get_client_returns_same_instance(self, adapter: AlpacaBarAdapter) -> None:
        c1 = adapter._get_client()
        c2 = adapter._get_client()
        assert c1 is c2
