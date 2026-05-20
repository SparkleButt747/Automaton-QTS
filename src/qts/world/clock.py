"""Simulated wall-clock for the world simulator.

The clock is deterministic, seedable-by-construction (no RNG of its own),
and yields tick timestamps inclusive of start, exclusive of end.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class SimulatedClock:
    """Deterministic wall-clock advancing in fixed-size ticks."""

    start: datetime
    end: datetime
    tick: timedelta

    def __post_init__(self) -> None:
        if self.tick.total_seconds() <= 0:
            raise ValueError("tick must be positive")
        if self.end <= self.start:
            raise ValueError("end must be after start")
        self._now = self.start

    @property
    def now(self) -> datetime:
        return self._now

    def advance(self) -> None:
        self._now = self._now + self.tick

    def finished(self) -> bool:
        return self._now >= self.end

    def iter_ticks(self) -> Iterator[datetime]:
        """Yield every tick from start (inclusive) to end (exclusive)."""
        t = self.start
        while t < self.end:
            yield t
            t = t + self.tick
