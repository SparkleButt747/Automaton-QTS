"""Frozen, reproducible bank of simulated FOMC episodes with randomised couplings.

Each episode draws an independent (surprise bucket, onset-lag, peak-magnitude,
decay, noise) so an Optuna sweep must find news params robust across the whole
range rather than memorising one coupling. Generation is deterministic from the
top-level seed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from qts.world.drift_model import SentimentDriftModel
from qts.world.runner import _DEFAULT_CORPUS, run_simulation

if TYPE_CHECKING:
    from pathlib import Path

    from qts.world.episode import SimulatedEpisode
    from qts.world.scenario import ScenarioConfig

# bucket -> (drift direction, rate offset from expected). |offset| > 0.05 trips the bucket.
_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("dovish", 1.0, -0.25),
    ("hawkish", -1.0, +0.25),
    ("neutral", 0.0, 0.0),
)


@dataclass(frozen=True)
class CouplingRanges:
    """Plausible ranges the per-episode coupling is drawn from."""

    onset_lag_bars: tuple[int, int] = (3, 20)
    peak_bps: tuple[float, float] = (20.0, 120.0)
    noise_std_bps: tuple[float, float] = (5.0, 30.0)
    decay_halflife_bars: tuple[int, int] = (10, 60)


# Frozen, so a shared module-level default is safe (avoids a call in arg defaults).
_DEFAULT_COUPLING_RANGES = CouplingRanges()


def generate_episode_bank(
    n: int,
    seed: int,
    base_scenario: ScenarioConfig,
    ranges: CouplingRanges = _DEFAULT_COUPLING_RANGES,
    corpus_path: Path | None = None,
) -> list[SimulatedEpisode]:
    """Generate `n` simulated episodes with independent randomised couplings.

    Reproducible from `seed`. Passing a bare object() as the strategy skips
    run_simulation's stage-2 on_text forwarding (we only want terrain + text).
    """
    if n <= 0:
        raise ValueError("n must be positive")

    master = random.Random(seed)  # noqa: S311 - deterministic sim, not crypto
    tick_s = base_scenario.tick.total_seconds()
    episodes: list[SimulatedEpisode] = []

    for i in range(n):
        ep_seed = master.randint(1, 2**31 - 1)
        rng = random.Random(ep_seed)  # noqa: S311

        _bucket, direction, rate_offset = _BUCKETS[i % len(_BUCKETS)]
        onset_bars = rng.randint(*ranges.onset_lag_bars)
        peak = rng.uniform(*ranges.peak_bps)
        noise = rng.uniform(*ranges.noise_std_bps)
        decay_bars = rng.randint(*ranges.decay_halflife_bars)

        drift = SentimentDriftModel(
            direction=direction,
            onset_lag=timedelta(seconds=onset_bars * tick_s),
            peak_bps=peak,
            decay_halflife=timedelta(seconds=decay_bars * tick_s),
            noise_std_bps=noise,
            event_time=base_scenario.fomc_announcement_at,
            seed=ep_seed,
        )
        episode = run_simulation(
            scenario=base_scenario,
            strategy=object(),
            seed=ep_seed,
            fomc_actual_rate=base_scenario.fomc_expected_rate + rate_offset,
            persona_corpus_path=corpus_path or _DEFAULT_CORPUS,
            drift_model=drift,
        )
        episodes.append(episode)

    return episodes
