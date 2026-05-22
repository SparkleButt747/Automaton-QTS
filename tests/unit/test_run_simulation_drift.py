"""T-RUNSIM-1: run_simulation surfaces text_events and applies drift."""

from __future__ import annotations

from datetime import datetime, timedelta

from qts.world.drift_model import SentimentDriftModel
from qts.world.runner import run_simulation
from qts.world.scenario import AnonAgentConfig, ScenarioConfig

_START = datetime(2023, 12, 13, 18, 0, 0)
_ANNOUNCE = datetime(2023, 12, 13, 19, 0, 0)
_END = datetime(2023, 12, 13, 21, 0, 0)


def _scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="runsim-test",
        symbol="BTCUSDT",
        start=_START,
        end=_END,
        tick=timedelta(minutes=1),
        fomc_announcement_at=_ANNOUNCE,
        fomc_expected_rate=5.25,
        starting_price=40_000.0,
        anon_agents=[AnonAgentConfig(agent_id="a1", style="trend")],
        mm_base_spread_bps=2.0,
        mm_vol_widen_k=1.0,
        powell_persona_id="powell",
    )


def test_episode_carries_text_events() -> None:  # T-RUNSIM-1
    drift = SentimentDriftModel(
        direction=1.0,
        onset_lag=timedelta(minutes=10),
        peak_bps=500.0,
        decay_halflife=timedelta(minutes=120),
        noise_std_bps=0.0,
        event_time=_ANNOUNCE,
        seed=1,
    )
    episode = run_simulation(
        scenario=_scenario(),
        strategy=object(),
        seed=1,
        fomc_actual_rate=5.00,
        drift_model=drift,  # 5.00 < 5.25 -> dovish
    )
    assert episode.text_events, "expected text_events on the episode"
    assert all(hasattr(e, "timestamp") and hasattr(e, "text") for e in episode.text_events)
