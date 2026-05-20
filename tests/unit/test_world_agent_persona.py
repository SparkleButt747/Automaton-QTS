"""Tests for qts.world.agents.persona."""

from __future__ import annotations

import random
from datetime import UTC, datetime
from pathlib import Path

from qts.world.agents.base import AgentContext
from qts.world.corpus import PersonaCorpus
from qts.world.events import MacroEvent, TextEvent


def _ctx(now: datetime, rng: random.Random) -> AgentContext:
    return AgentContext(now=now, last_price=30000.0, rng=rng, recent_bars=[], recent_text=[])


def _seeded_corpus() -> PersonaCorpus:
    return PersonaCorpus.from_yaml(Path("data/world/persona_corpus/powell_fomc.yaml"))


def test_persona_reacts_to_fomc_event() -> None:  # T-WPER-1
    from qts.world.agents.persona import PersonaAgent

    agent = PersonaAgent(
        agent_id="powell",
        persona_id="powell",
        corpus=_seeded_corpus(),
        surprise_bucket="hawkish",
    )

    fomc = MacroEvent(
        timestamp=datetime(2025, 3, 19, 14, 0, tzinfo=UTC),
        kind="fomc",
        expected=5.25,
        actual=5.50,
        surprise=0.25,
        text_event=None,
    )
    out = agent.on_event(fomc, _ctx(fomc.timestamp, random.Random(42)))

    text_events = [e for e in out if isinstance(e, TextEvent)]
    assert len(text_events) == 1
    assert text_events[0].source == "powell"
    assert text_events[0].metadata["surprise_bucket"] == "hawkish"
    assert "powell:fomc:hawkish" in agent.consumed_corpus_keys


def test_persona_ignores_unrelated_events() -> None:  # T-WPER-2
    from qts.world.agents.persona import PersonaAgent

    agent = PersonaAgent(
        agent_id="powell",
        persona_id="powell",
        corpus=_seeded_corpus(),
        surprise_bucket="hawkish",
    )

    cpi = MacroEvent(
        timestamp=datetime(2025, 3, 19, 14, 0, tzinfo=UTC),
        kind="cpi",
        expected=3.2,
        actual=3.5,
        surprise=0.3,
        text_event=None,
    )
    out = agent.on_event(cpi, _ctx(cpi.timestamp, random.Random(42)))
    assert out == []


def test_persona_on_tick_is_silent_outside_window() -> None:  # T-WPER-3
    from qts.world.agents.persona import PersonaAgent

    agent = PersonaAgent(
        agent_id="powell",
        persona_id="powell",
        corpus=_seeded_corpus(),
        surprise_bucket="hawkish",
    )

    out = agent.on_tick(_ctx(datetime(2025, 3, 19, 8, 0, tzinfo=UTC), random.Random(0)))
    assert out == []


def test_persona_deterministic_under_seed() -> None:  # T-WPER-4
    from qts.world.agents.persona import PersonaAgent

    def run(seed: int) -> list[str]:
        agent = PersonaAgent(
            agent_id="powell",
            persona_id="powell",
            corpus=_seeded_corpus(),
            surprise_bucket="dovish",
        )
        fomc = MacroEvent(
            timestamp=datetime(2025, 3, 19, 14, 0, tzinfo=UTC),
            kind="fomc",
            expected=5.25,
            actual=5.0,
            surprise=-0.25,
            text_event=None,
        )
        out = agent.on_event(fomc, _ctx(fomc.timestamp, random.Random(seed)))
        return [e.text for e in out if isinstance(e, TextEvent)]

    assert run(42) == run(42)
