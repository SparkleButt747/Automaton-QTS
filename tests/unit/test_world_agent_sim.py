"""Tests for qts.world.agent_sim — the stage-1 loop."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path


def _scenario():
    from qts.world.scenario import AnonAgentConfig, ScenarioConfig

    start = datetime(2025, 3, 19, 0, 0, tzinfo=UTC)
    return ScenarioConfig(
        name="fomc_test",
        symbol="BTCUSDT",
        start=start,
        end=start + timedelta(hours=1),  # 60 ticks for speed
        tick=timedelta(minutes=1),
        fomc_announcement_at=start + timedelta(minutes=30),
        fomc_expected_rate=5.25,
        starting_price=30000.0,
        anon_agents=[
            AnonAgentConfig(agent_id="anon_sent", style="sentiment"),
            AnonAgentConfig(agent_id="anon_trend", style="trend"),
            AnonAgentConfig(agent_id="anon_mr", style="mean_revert"),
        ],
        mm_base_spread_bps=10.0,
        mm_vol_widen_k=2.0,
        powell_persona_id="powell",
    )


def test_agent_sim_emits_bars_and_events() -> None:  # T-WAS-1
    from qts.world.agent_sim import run_agent_sim
    from qts.world.corpus import PersonaCorpus

    corpus = PersonaCorpus.from_yaml(Path("data/world/persona_corpus/powell_fomc.yaml"))
    result = run_agent_sim(
        scenario=_scenario(),
        corpus=corpus,
        seed=42,
        fomc_actual_rate=5.5,
    )

    assert len(result.bars) == 60  # 60 minutes at 1m bars
    assert any(e.kind == "fomc" for e in result.macro_events)
    assert len(result.text_events) > 0


def test_agent_sim_reproducible() -> None:  # T-WAS-2
    from qts.world.agent_sim import run_agent_sim
    from qts.world.corpus import PersonaCorpus

    corpus = PersonaCorpus.from_yaml(Path("data/world/persona_corpus/powell_fomc.yaml"))
    a = run_agent_sim(scenario=_scenario(), corpus=corpus, seed=42, fomc_actual_rate=5.5)
    b = run_agent_sim(scenario=_scenario(), corpus=corpus, seed=42, fomc_actual_rate=5.5)

    assert [bar.close for bar in a.bars] == [bar.close for bar in b.bars]
    assert [e.text for e in a.text_events] == [e.text for e in b.text_events]


def test_agent_sim_different_seed_different_bars() -> None:  # T-WAS-3
    from qts.world.agent_sim import run_agent_sim
    from qts.world.corpus import PersonaCorpus

    corpus = PersonaCorpus.from_yaml(Path("data/world/persona_corpus/powell_fomc.yaml"))
    a = run_agent_sim(scenario=_scenario(), corpus=corpus, seed=42, fomc_actual_rate=5.5)
    b = run_agent_sim(scenario=_scenario(), corpus=corpus, seed=99, fomc_actual_rate=5.5)

    # Either bars or text differ — not both identical
    bars_equal = [bar.close for bar in a.bars] == [bar.close for bar in b.bars]
    text_equal = [e.text for e in a.text_events] == [e.text for e in b.text_events]
    assert not (bars_equal and text_equal)
