"""SentimentDriftModel — time-varying price drift induced by an FOMC news event.

Wired into the agent simulation: each tick the sim marks the market to a drift-
adjusted fair price off a fixed pre-event reference, so this model shifts the
simulated price after a news event. The drift ramps in over onset_lag, then decays
— leaving a lead-lag window a news-reading strategy can exploit (the simulated
decode-gap edge). Per-tick gaussian noise is seed-deterministic so episodes are
reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class SentimentDriftModel:
    """Drift trajectory in bps, as a function of wall-clock time after an event."""

    direction: float  # +1 dovish (up), -1 hawkish (down), 0 neutral
    onset_lag: timedelta
    peak_bps: float
    decay_halflife: timedelta
    noise_std_bps: float
    event_time: datetime
    seed: int

    def value_at(self, now: datetime) -> float:
        """bps drift at `now`. 0 before the event; ramps to peak over onset_lag;
        exponential decay afterward; plus seeded gaussian noise."""
        if now < self.event_time:
            return 0.0

        elapsed = (now - self.event_time).total_seconds()
        lag_s = self.onset_lag.total_seconds()

        if lag_s > 0 and elapsed < lag_s:
            base = self.direction * self.peak_bps * (elapsed / lag_s)
        else:
            after_peak = elapsed - lag_s
            hl_s = self.decay_halflife.total_seconds()
            decay = 0.5 ** (after_peak / hl_s) if hl_s > 0 else 1.0
            base = self.direction * self.peak_bps * decay

        if self.noise_std_bps <= 0.0:
            return base
        rng = random.Random(hash((self.seed, int(elapsed))))  # noqa: S311
        return base + rng.gauss(0.0, self.noise_std_bps)
