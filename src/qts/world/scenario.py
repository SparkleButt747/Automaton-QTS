"""Declarative scenario definition for the world simulator.

A ScenarioConfig is everything needed to deterministically reconstruct
an episode given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass(frozen=True, slots=True)
class AnonAgentConfig:
    """Configuration for one anonymous retail agent.

    style:
        - "sentiment": reads recent text, buys when sentiment positive
        - "trend":     buys on recent positive returns
        - "mean_revert": fades extreme recent moves
    aggressiveness:
        Multiplier on base order size; 0 = inactive, 1 = baseline.
    """

    agent_id: str
    style: str
    aggressiveness: float = 1.0
    reaction_lag_bars: int = 1


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    """Full v1 scenario specification — FOMC on BTCUSDT."""

    name: str
    symbol: str
    start: datetime
    end: datetime
    tick: timedelta

    # FOMC event
    fomc_announcement_at: datetime
    fomc_expected_rate: float

    # Market priming
    starting_price: float

    # Agent roster
    anon_agents: list[AnonAgentConfig]
    mm_base_spread_bps: float
    mm_vol_widen_k: float
    powell_persona_id: str

    # Optional persona schedule overrides (timestamps for forced statements)
    powell_q_and_a_times: list[datetime] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("end must be after start")
        if not (self.start <= self.fomc_announcement_at < self.end):
            raise ValueError("fomc_announcement_at must fall within [start, end)")
        if self.tick.total_seconds() <= 0:
            raise ValueError("tick must be positive")
