"""Aligned hourly close panel from Binance for the contagion study (spec §4). Delisted symbols
return no data and are dropped (logged); BTC (the market factor) is required. Alignment = the
timestamp intersection across tokens that returned data."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Protocol

import numpy as np

from qts.models.base import Bar

logger = logging.getLogger(__name__)


class _BarProvider(Protocol):
    async def get_historical_bars(
        self, symbol: str, start: str, end: str, interval: str = ...
    ) -> list[Bar]: ...


def binance_symbol(token: str, *, overrides: dict[str, str] | None = None) -> str:
    """Map a universe token to its Binance spot symbol (default ``{TOKEN}USDT``)."""
    overrides = overrides or {}
    return overrides.get(token, f"{token}USDT")


async def fetch_price_panel(
    tokens: list[str],
    *,
    bar_adapter: _BarProvider,
    start: str,
    end: str,
    interval: str = "1h",
    symbol_overrides: dict[str, str] | None = None,
) -> tuple[list[datetime], dict[str, np.ndarray]]:
    """Return (aligned_timestamps, {token: closes}) over the shared timestamp grid. ``BTC`` must be
    present in ``tokens`` and must return data."""
    per_token: dict[str, dict[datetime, float]] = {}
    for tok in tokens:
        bars = await bar_adapter.get_historical_bars(
            binance_symbol(tok, overrides=symbol_overrides), start, end, interval
        )
        if not bars:
            logger.warning("no klines for %s (delisted?) — dropping from panel", tok)
            continue
        per_token[tok] = {b.timestamp: b.close for b in bars}
    if "BTC" not in per_token:
        raise ValueError("BTC (market factor) has no price data — cannot build panel")
    common = set.intersection(*(set(ts.keys()) for ts in per_token.values()))
    grid = sorted(common)
    closes = {tok: np.array([ts[d] for d in grid], dtype=float) for tok, ts in per_token.items()}
    return grid, closes
