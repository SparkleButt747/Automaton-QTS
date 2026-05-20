"""Aggregate stage-1 trades into 1m Bars for the Nautilus backtest stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from qts.models.base import Bar


@dataclass
class _PartialBar:
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BarAggregator:
    symbol: str
    tick: timedelta
    _partials: dict[datetime, _PartialBar] = field(default_factory=dict)
    _last_close: float | None = None
    _last_bucket: datetime | None = None

    def _bucket(self, ts: datetime) -> datetime:
        offset_us = (ts - datetime.min.replace(tzinfo=ts.tzinfo)) // self.tick
        return datetime.min.replace(tzinfo=ts.tzinfo) + offset_us * self.tick

    def add_trade(self, ts: datetime, price: float, qty: float) -> None:
        bucket = self._bucket(ts)
        partial = self._partials.get(bucket)
        if partial is None:
            self._partials[bucket] = _PartialBar(
                open=price, high=price, low=price, close=price, volume=qty
            )
        else:
            partial.high = max(partial.high, price)
            partial.low = min(partial.low, price)
            partial.close = price
            partial.volume += qty
        self._last_close = price
        if self._last_bucket is None or bucket > self._last_bucket:
            self._last_bucket = bucket

    def flush(self, end: datetime) -> list[Bar]:
        """Emit a Bar for every bucket from the earliest seen up to (exclusive) end.

        Empty buckets are filled with a flat carry-bar at the last close.
        """
        if not self._partials and self._last_bucket is None:
            return []

        first = min(self._partials.keys()) if self._partials else self._last_bucket
        assert first is not None
        out: list[Bar] = []
        cursor = first
        while cursor < end:
            partial = self._partials.get(cursor)
            if partial is None:
                close = self._last_close if self._last_close is not None else 0.0
                out.append(
                    Bar(
                        symbol=self.symbol,
                        timestamp=cursor,
                        open=close,
                        high=close,
                        low=close,
                        close=close,
                        volume=0.0,
                    )
                )
            else:
                out.append(
                    Bar(
                        symbol=self.symbol,
                        timestamp=cursor,
                        open=partial.open,
                        high=partial.high,
                        low=partial.low,
                        close=partial.close,
                        volume=partial.volume,
                    )
                )
                self._last_close = partial.close
            cursor = cursor + self.tick
        return out
