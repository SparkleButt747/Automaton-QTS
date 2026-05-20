"""Declarative scenario definition for the world simulator.

A ScenarioConfig is everything needed to deterministically reconstruct
an episode given a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import yaml


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


def load_scenario_yaml(path: Path) -> ScenarioConfig:
    """Load a ScenarioConfig from a YAML file.

    Expected schema (minimum fields):
        name: str
        symbol: str
        start: ISO datetime
        end: ISO datetime
        tick_minutes: int
        fomc_announcement_at: ISO datetime
        fomc_expected_rate: float
        starting_price: float
        mm_base_spread_bps: float
        mm_vol_widen_k: float
        powell_persona_id: str
        anon_agents:
          - agent_id: str
            style: str
            aggressiveness: float
    """
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    required = (
        "name",
        "symbol",
        "start",
        "end",
        "tick_minutes",
        "fomc_announcement_at",
        "fomc_expected_rate",
        "starting_price",
        "mm_base_spread_bps",
        "mm_vol_widen_k",
        "powell_persona_id",
        "anon_agents",
    )
    for f in required:
        if f not in raw:
            raise KeyError(f"scenario YAML missing required field: {f}")

    return ScenarioConfig(
        name=str(raw["name"]),
        symbol=str(raw["symbol"]),
        start=datetime.fromisoformat(str(raw["start"])),
        end=datetime.fromisoformat(str(raw["end"])),
        tick=timedelta(minutes=int(raw["tick_minutes"])),
        fomc_announcement_at=datetime.fromisoformat(str(raw["fomc_announcement_at"])),
        fomc_expected_rate=float(raw["fomc_expected_rate"]),
        starting_price=float(raw["starting_price"]),
        anon_agents=[
            AnonAgentConfig(
                agent_id=str(a["agent_id"]),
                style=str(a["style"]),
                aggressiveness=float(a.get("aggressiveness", 1.0)),
                reaction_lag_bars=int(a.get("reaction_lag_bars", 1)),
            )
            for a in raw["anon_agents"]
        ],
        mm_base_spread_bps=float(raw["mm_base_spread_bps"]),
        mm_vol_widen_k=float(raw["mm_vol_widen_k"]),
        powell_persona_id=str(raw["powell_persona_id"]),
    )
