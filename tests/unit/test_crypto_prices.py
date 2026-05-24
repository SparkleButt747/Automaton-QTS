"""T-CRYPTO-PRICE-*: symbol mapping + aligned panel from a stub bar adapter (no network)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from qts.propagation.crypto.prices import binance_symbol, fetch_price_panel

from qts.models.base import Bar


class _StubAdapter:
    """Returns synthetic hourly bars; ``empty`` symbols (delisted) return []."""

    def __init__(self, empty: set[str] | None = None) -> None:
        self._empty = empty or set()

    async def get_historical_bars(
        self, symbol: str, start: str, end: str, interval: str = "1m"
    ) -> list[Bar]:
        if symbol in self._empty:
            return []
        base = datetime(2023, 1, 1, tzinfo=UTC)
        return [
            Bar(
                timestamp=base + timedelta(hours=h),
                symbol=symbol,
                open=1.0,
                high=1.0,
                low=1.0,
                close=float(100 + h),
                volume=1.0,
                bar_count=1,
            )
            for h in range(5)
        ]


def test_binance_symbol_default_and_override() -> None:  # T-CRYPTO-PRICE-1
    assert binance_symbol("SOL") == "SOLUSDT"
    assert binance_symbol("FTT", overrides={"FTT": "FTTBUSD"}) == "FTTBUSD"


def test_panel_aligns_and_drops_missing() -> None:  # T-CRYPTO-PRICE-2
    adapter = _StubAdapter(empty={"LUNAUSDT"})  # LUNA delisted -> no data
    dates, closes = asyncio.run(
        fetch_price_panel(
            ["BTC", "SOL", "LUNA"],
            bar_adapter=adapter,
            start="2023-01-01",
            end="2023-01-02",
            interval="1h",
        )
    )
    assert "BTC" in closes and "SOL" in closes
    assert "LUNA" not in closes  # dropped, no data
    assert len(dates) == 5 and len(closes["SOL"]) == 5
    assert closes["SOL"][0] == 100.0


def test_panel_requires_btc() -> None:  # T-CRYPTO-PRICE-3
    adapter = _StubAdapter(empty={"BTCUSDT"})
    with pytest.raises(ValueError, match="BTC"):
        asyncio.run(
            fetch_price_panel(
                ["BTC", "SOL"], bar_adapter=adapter, start="2023-01-01", end="2023-01-02"
            )
        )
