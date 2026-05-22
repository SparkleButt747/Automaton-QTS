"""T-COUPLE-1: a dovish drift_model raises simulated closing price vs no coupling."""

from __future__ import annotations

from datetime import datetime, timedelta

from qts.world.agent_sim import run_agent_sim
from qts.world.corpus import PersonaCorpus
from qts.world.drift_model import SentimentDriftModel
from qts.world.runner import _DEFAULT_CORPUS
from qts.world.scenario import AnonAgentConfig, ScenarioConfig

_START = datetime(2023, 12, 13, 18, 0, 0)
_ANNOUNCE = datetime(2023, 12, 13, 19, 0, 0)
_END = datetime(2023, 12, 13, 21, 0, 0)


def _scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="couple-test",
        symbol="BTCUSDT",
        start=_START,
        end=_END,
        tick=timedelta(minutes=1),
        fomc_announcement_at=_ANNOUNCE,
        fomc_expected_rate=5.25,
        starting_price=40_000.0,
        anon_agents=[AnonAgentConfig(agent_id="a1", style="trend", aggressiveness=1.0)],
        mm_base_spread_bps=2.0,
        mm_vol_widen_k=1.0,
        powell_persona_id="powell",
    )


def test_dovish_drift_lifts_price() -> None:  # T-COUPLE-1
    corpus = PersonaCorpus.from_yaml(_DEFAULT_CORPUS)
    scenario = _scenario()

    baseline = run_agent_sim(scenario=scenario, corpus=corpus, seed=42, fomc_actual_rate=5.25)

    dovish = SentimentDriftModel(
        direction=1.0,
        onset_lag=timedelta(minutes=10),
        peak_bps=1000.0,  # 10% — dominate agent noise so the test is deterministic
        decay_halflife=timedelta(minutes=120),
        noise_std_bps=0.0,
        event_time=_ANNOUNCE,
        seed=42,
    )
    coupled = run_agent_sim(
        scenario=scenario, corpus=corpus, seed=42, fomc_actual_rate=5.25, drift_model=dovish
    )

    assert baseline.bars and coupled.bars
    assert coupled.bars[-1].close > baseline.bars[-1].close
