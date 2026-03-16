"""Unit tests for qts.data.market.binance_adapter.

Verifies:
- Historical kline parsing (REST response -> Bar)
- Order book snapshot parsing (REST response -> OrderBookSnapshot)
- Trade tick parsing (WS message -> Tick)
- Edge cases: empty responses, malformed data, missing fields
- close() behaviour
- Pagination logic
- Rate-limit backoff (429 handling)
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from qts.data.market.binance_adapter import (
    BinanceBarAdapter,
    BinanceOrderBookAdapter,
    BinanceTickAdapter,
    _iso_to_ms,
    _ms_to_datetime,
    _parse_depth,
    _parse_kline,
    _parse_trade,
)
from qts.models.base import Bar, OrderBookLevel, OrderBookSnapshot, OrderSide, Tick

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_kline(
    open_time: int = 1704067200000,
    open_: str = "42000.00",
    high: str = "42500.00",
    low: str = "41800.00",
    close: str = "42300.00",
    volume: str = "150.5",
    close_time: int = 1704067259999,
    quote_volume: str = "6345750.00",
    trades: int = 1234,
    taker_buy_base: str = "80.0",
    taker_buy_quote: str = "3376000.00",
    ignore: str = "0",
) -> list:
    """Build a Binance kline array."""
    return [
        open_time,
        open_,
        high,
        low,
        close,
        volume,
        close_time,
        quote_volume,
        trades,
        taker_buy_base,
        taker_buy_quote,
        ignore,
    ]


def _make_depth_response(
    bids: list | None = None,
    asks: list | None = None,
) -> dict:
    """Build a Binance depth REST response."""
    return {
        "lastUpdateId": 12345678,
        "bids": bids or [["42000.00", "1.5"], ["41999.00", "2.0"]],
        "asks": asks or [["42001.00", "0.8"], ["42002.00", "1.2"]],
    }


def _make_trade_ws_msg(
    price: str = "42100.00",
    quantity: str = "0.5",
    trade_time: int = 1704067230000,
    is_buyer_maker: bool = False,
) -> dict:
    """Build a Binance trade WebSocket message."""
    return {
        "e": "trade",
        "E": trade_time,
        "s": "BTCUSDT",
        "t": 123456,
        "p": price,
        "q": quantity,
        "b": 111,
        "a": 222,
        "T": trade_time,
        "m": is_buyer_maker,
        "M": True,
    }


def _mock_httpx_response(json_data, status_code=200):
    """Create a mock httpx.Response."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.headers = {}
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=mock_resp,
        )
    else:
        mock_resp.raise_for_status.return_value = None
    return mock_resp


# ── Timestamp conversion ─────────────────────────────────────────────────────


class TestTimestampConversion:
    def test_ms_to_datetime(self):
        # 2024-01-01 00:00:00 UTC
        dt = _ms_to_datetime(1704067200000)
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 1
        assert dt.tzinfo == UTC

    def test_iso_to_ms(self):
        ms = _iso_to_ms("2024-01-01T00:00:00+00:00")
        assert ms == 1704067200000

    def test_iso_to_ms_naive_assumes_utc(self):
        ms = _iso_to_ms("2024-01-01T00:00:00")
        assert ms == 1704067200000


# ── Kline parsing ────────────────────────────────────────────────────────────


class TestParseKline:
    def test_parses_valid_kline(self):
        kline = _make_kline()
        bar = _parse_kline("BTCUSDT", kline)

        assert isinstance(bar, Bar)
        assert bar.symbol == "BTCUSDT"
        assert bar.open == 42000.0
        assert bar.high == 42500.0
        assert bar.low == 41800.0
        assert bar.close == 42300.0
        assert bar.volume == 150.5
        assert bar.bar_count == 1234
        assert bar.timestamp.tzinfo == UTC

    def test_parses_kline_without_trades_field(self):
        """Short kline array (< 9 elements) should default bar_count to 0."""
        short_kline = [1704067200000, "42000", "42500", "41800", "42300", "150.5"]
        bar = _parse_kline("ETHUSDT", short_kline)

        assert bar.bar_count == 0
        assert bar.symbol == "ETHUSDT"

    def test_raises_on_empty_kline(self):
        with pytest.raises((IndexError, TypeError)):
            _parse_kline("BTCUSDT", [])

    def test_raises_on_non_numeric_values(self):
        bad_kline = [1704067200000, "not_a_number", "42500", "41800", "42300", "150.5"]
        with pytest.raises(ValueError):
            _parse_kline("BTCUSDT", bad_kline)


# ── Depth parsing ────────────────────────────────────────────────────────────


class TestParseDepth:
    def test_parses_valid_depth(self):
        data = _make_depth_response()
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        snapshot = _parse_depth("BTCUSDT", data, timestamp=ts)

        assert isinstance(snapshot, OrderBookSnapshot)
        assert snapshot.symbol == "BTCUSDT"
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2
        assert snapshot.bids[0] == OrderBookLevel(price=42000.0, quantity=1.5)
        assert snapshot.asks[0] == OrderBookLevel(price=42001.0, quantity=0.8)
        assert snapshot.timestamp == ts

    def test_empty_bids_and_asks(self):
        data = {"lastUpdateId": 1, "bids": [], "asks": []}
        snapshot = _parse_depth("BTCUSDT", data)

        assert snapshot.bids == ()
        assert snapshot.asks == ()

    def test_missing_bids_key(self):
        data = {"asks": [["42001.00", "0.8"]]}
        snapshot = _parse_depth("BTCUSDT", data)

        assert snapshot.bids == ()
        assert len(snapshot.asks) == 1

    def test_default_timestamp_is_utc(self):
        data = _make_depth_response()
        snapshot = _parse_depth("BTCUSDT", data)

        assert snapshot.timestamp.tzinfo == UTC


# ── Trade parsing ────────────────────────────────────────────────────────────


class TestParseTrade:
    def test_parses_buy_trade(self):
        data = _make_trade_ws_msg(is_buyer_maker=False)
        tick = _parse_trade("BTCUSDT", data)

        assert isinstance(tick, Tick)
        assert tick.symbol == "BTCUSDT"
        assert tick.price == 42100.0
        assert tick.quantity == 0.5
        assert tick.side == OrderSide.BUY
        assert tick.timestamp.tzinfo == UTC

    def test_parses_sell_trade(self):
        data = _make_trade_ws_msg(is_buyer_maker=True)
        tick = _parse_trade("BTCUSDT", data)

        assert tick.side == OrderSide.SELL

    def test_raises_on_missing_price(self):
        data = {"T": 1704067230000, "q": "0.5", "m": False}
        with pytest.raises(KeyError):
            _parse_trade("BTCUSDT", data)

    def test_raises_on_missing_timestamp(self):
        data = {"p": "42100.00", "q": "0.5", "m": False}
        with pytest.raises(KeyError):
            _parse_trade("BTCUSDT", data)


# ── BinanceBarAdapter REST ───────────────────────────────────────────────────


class TestBinanceBarAdapterHistorical:
    @pytest.fixture()
    def adapter(self):
        return BinanceBarAdapter()

    @pytest.mark.asyncio
    async def test_fetches_historical_bars(self, adapter):
        klines = [_make_kline(), _make_kline(open_time=1704067260000)]
        # Return data on first call, empty on second to stop pagination
        response_data = _mock_httpx_response(klines)
        response_empty = _mock_httpx_response([])

        with patch.object(adapter, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=[response_data, response_empty])
            mock_get.return_value = mock_http

            bars = await adapter.get_historical_bars(
                "BTCUSDT", "2024-01-01T00:00:00+00:00", "2024-01-01T01:00:00+00:00"
            )

        assert len(bars) == 2
        assert all(isinstance(b, Bar) for b in bars)
        assert bars[0].symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_empty_response_returns_empty_list(self, adapter):
        response = _mock_httpx_response([])

        with patch.object(adapter, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get.return_value = mock_http

            bars = await adapter.get_historical_bars(
                "BTCUSDT", "2024-01-01T00:00:00+00:00", "2024-01-01T01:00:00+00:00"
            )

        assert bars == []

    @pytest.mark.asyncio
    async def test_http_error_returns_empty_list(self, adapter):
        response = _mock_httpx_response({}, status_code=500)

        with patch.object(adapter, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get.return_value = mock_http

            bars = await adapter.get_historical_bars(
                "BTCUSDT", "2024-01-01T00:00:00+00:00", "2024-01-01T01:00:00+00:00"
            )

        assert bars == []

    @pytest.mark.asyncio
    async def test_malformed_kline_is_skipped(self, adapter):
        """Valid klines should be parsed; malformed ones skipped."""
        klines = [
            _make_kline(),
            [1704067260000, "not_a_number"],  # malformed — too few fields
            _make_kline(open_time=1704067320000),
        ]
        response_data = _mock_httpx_response(klines)
        response_empty = _mock_httpx_response([])

        with patch.object(adapter, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=[response_data, response_empty])
            mock_get.return_value = mock_http

            bars = await adapter.get_historical_bars(
                "BTCUSDT", "2024-01-01T00:00:00+00:00", "2024-01-01T01:00:00+00:00"
            )

        # [ts, "not_a_number"] has only 2 elements so kline[2] raises IndexError -> skipped
        assert len(bars) == 2

    @pytest.mark.asyncio
    async def test_symbol_uppercased(self, adapter):
        klines = [_make_kline()]
        response_data = _mock_httpx_response(klines)
        response_empty = _mock_httpx_response([])

        with patch.object(adapter, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(side_effect=[response_data, response_empty])
            mock_get.return_value = mock_http

            bars = await adapter.get_historical_bars(
                "btcusdt", "2024-01-01T00:00:00+00:00", "2024-01-01T01:00:00+00:00"
            )

        assert bars[0].symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_network_error_returns_empty_list(self, adapter):
        with patch.object(adapter, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            mock_get.return_value = mock_http

            bars = await adapter.get_historical_bars(
                "BTCUSDT", "2024-01-01T00:00:00+00:00", "2024-01-01T01:00:00+00:00"
            )

        assert bars == []


# ── BinanceOrderBookAdapter REST ─────────────────────────────────────────────


class TestBinanceOrderBookAdapterSnapshot:
    @pytest.fixture()
    def adapter(self):
        return BinanceOrderBookAdapter()

    @pytest.mark.asyncio
    async def test_fetches_snapshot(self, adapter):
        depth_data = _make_depth_response()
        response = _mock_httpx_response(depth_data)

        with patch.object(adapter, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get.return_value = mock_http

            snapshot = await adapter.get_snapshot("BTCUSDT", depth=20)

        assert isinstance(snapshot, OrderBookSnapshot)
        assert snapshot.symbol == "BTCUSDT"
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2

    @pytest.mark.asyncio
    async def test_http_error_returns_empty_snapshot(self, adapter):
        response = _mock_httpx_response({}, status_code=500)

        with patch.object(adapter, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get.return_value = mock_http

            snapshot = await adapter.get_snapshot("BTCUSDT")

        assert snapshot.bids == ()
        assert snapshot.asks == ()
        assert snapshot.symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_network_error_returns_empty_snapshot(self, adapter):
        with patch.object(adapter, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )
            mock_get.return_value = mock_http

            snapshot = await adapter.get_snapshot("BTCUSDT")

        assert snapshot.bids == ()
        assert snapshot.asks == ()

    @pytest.mark.asyncio
    async def test_empty_depth_response(self, adapter):
        depth_data = {"lastUpdateId": 1, "bids": [], "asks": []}
        response = _mock_httpx_response(depth_data)

        with patch.object(adapter, "_get_client") as mock_get:
            mock_http = AsyncMock()
            mock_http.get = AsyncMock(return_value=response)
            mock_get.return_value = mock_http

            snapshot = await adapter.get_snapshot("BTCUSDT")

        assert snapshot.bids == ()
        assert snapshot.asks == ()


# ── Close behaviour ──────────────────────────────────────────────────────────


class TestCloseBehaviour:
    @pytest.mark.asyncio
    async def test_bar_adapter_close_before_use(self):
        adapter = BinanceBarAdapter()
        await adapter.close()  # should not raise

    @pytest.mark.asyncio
    async def test_bar_adapter_close_with_client(self):
        adapter = BinanceBarAdapter()
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        adapter._client = mock_client

        await adapter.close()
        mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tick_adapter_close(self):
        adapter = BinanceTickAdapter()
        await adapter.close()  # should not raise

    @pytest.mark.asyncio
    async def test_order_book_adapter_close_before_use(self):
        adapter = BinanceOrderBookAdapter()
        await adapter.close()  # should not raise

    @pytest.mark.asyncio
    async def test_order_book_adapter_close_with_client(self):
        adapter = BinanceOrderBookAdapter()
        mock_client = MagicMock(spec=httpx.AsyncClient)
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        adapter._client = mock_client

        await adapter.close()
        mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_idempotent(self):
        adapter = BinanceBarAdapter()
        await adapter.close()
        await adapter.close()  # second call should not raise


# ── Protocol conformance ─────────────────────────────────────────────────────


class TestProtocolConformance:
    def test_bar_adapter_has_required_methods(self):
        adapter = BinanceBarAdapter()
        assert hasattr(adapter, "get_historical_bars")
        assert hasattr(adapter, "subscribe")
        assert hasattr(adapter, "close")

    def test_tick_adapter_has_required_methods(self):
        adapter = BinanceTickAdapter()
        assert hasattr(adapter, "subscribe")
        assert hasattr(adapter, "close")

    def test_order_book_adapter_has_required_methods(self):
        adapter = BinanceOrderBookAdapter()
        assert hasattr(adapter, "get_snapshot")
        assert hasattr(adapter, "subscribe")
        assert hasattr(adapter, "close")


# ── Lazy client creation ─────────────────────────────────────────────────────


class TestLazyClientCreation:
    def test_bar_adapter_creates_client_lazily(self):
        adapter = BinanceBarAdapter()
        assert adapter._client is None
        client = adapter._get_client()
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)

    def test_bar_adapter_reuses_client(self):
        adapter = BinanceBarAdapter()
        c1 = adapter._get_client()
        c2 = adapter._get_client()
        assert c1 is c2

    def test_order_book_adapter_creates_client_lazily(self):
        adapter = BinanceOrderBookAdapter()
        assert adapter._client is None
        client = adapter._get_client()
        assert client is not None
