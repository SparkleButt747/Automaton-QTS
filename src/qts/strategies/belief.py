"""Multi-axis decaying news belief.

Replaces the v2 single-scalar BeliefAxis (which REPLACED its value, so a trailing
neutral read wiped the signal to zero). NewsBelief accumulates conviction across
events and combines three axes at read time:

    belief = conviction × relevance_gate × magnitude_scale

- conviction:      decaying signed sum of (direction_sign × confidence). Accumulates,
                   so a burst of same-direction confident reads builds the signal.
                   This is the ONLY time-decay in the model (one forgetting mechanism).
- relevance_gate:  running max of relevance (0-1). Low-relevance reads can't lower it
                   → 'ignored, not cancelling'.
- magnitude_scale: running max of magnitude → output amplitude.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from qts.macro.news_signal import NewsSignal


@dataclass
class NewsBelief:
    """Multi-axis decaying belief over a stream of NewsSignals.

    Args:
        half_life: Time after which the accumulated conviction decays by 50%.
    """

    half_life: timedelta
    _conviction: float = field(default=0.0, init=False)
    _relevance: float = field(default=0.0, init=False)
    _magnitude: float = field(default=0.0, init=False)
    _last_update: datetime | None = field(default=None, init=False)

    def _decay(self, now: datetime) -> float:
        """Conviction decay factor since last_update; clamps negative elapsed to 0."""
        if self._last_update is None:
            return 1.0
        elapsed = max(0.0, (now - self._last_update).total_seconds())  # never amplify
        half_life_s = self.half_life.total_seconds()
        if half_life_s <= 0:
            return 1.0
        return float(0.5 ** (elapsed / half_life_s))

    def observe(self, signal: NewsSignal, now: datetime) -> None:
        """Fold a classified signal into the belief at time `now`."""
        d = self._decay(now)
        self._conviction = self._conviction * d + signal.direction_sign * signal.confidence
        self._relevance = max(self._relevance, signal.relevance)
        self._magnitude = max(self._magnitude, signal.magnitude)
        self._last_update = now

    def value_at(self, now: datetime) -> float:
        """Return the combined belief at `now` (pure read; only conviction decays)."""
        if self._last_update is None:
            return 0.0
        d = self._decay(now)
        return (self._conviction * d) * self._relevance * self._magnitude
