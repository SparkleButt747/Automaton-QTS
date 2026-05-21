"""Exponentially-decaying scalar belief primitive.

Used by NewsReactiveMomentum to hold a per-axis news read that fades
over time, so the strategy 'forgets' stale signals at a controlled rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class BeliefAxis:
    """A scalar belief that decays exponentially toward zero over time.

    Args:
        half_life: Time after which an updated value decays by 50%.
    """

    half_life: timedelta
    _value: float = field(default=0.0, init=False)
    _last_update: datetime | None = field(default=None, init=False)

    def update(self, value: float, now: datetime) -> None:
        """Replace the belief with `value` anchored at `now`."""
        self._value = value
        self._last_update = now

    def at(self, now: datetime) -> float:
        """Return the value at `now`, after exponential decay since last_update."""
        if self._last_update is None:
            return 0.0
        elapsed = (now - self._last_update).total_seconds()
        half_life_s = self.half_life.total_seconds()
        if half_life_s <= 0:
            return self._value
        decay = 0.5 ** (elapsed / half_life_s)
        return self._value * decay
