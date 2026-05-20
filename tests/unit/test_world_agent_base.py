"""Tests for qts.world.agents.base."""

from __future__ import annotations

import random
from datetime import UTC, datetime


def test_agent_context_holds_state() -> None:  # T-WAG-1
    from qts.world.agents.base import AgentContext

    ctx = AgentContext(
        now=datetime(2025, 3, 19, 14, 0, tzinfo=UTC),
        last_price=30000.0,
        rng=random.Random(42),
        recent_bars=[],
        recent_text=[],
    )
    assert ctx.last_price == 30000.0
    assert isinstance(ctx.rng, random.Random)


def test_world_agent_protocol_runtime_checkable() -> None:  # T-WAG-2
    from qts.world.agents.base import WorldAgent

    class _Stub:
        agent_id = "stub"

        def on_tick(self, ctx: object) -> list[object]:
            return []

        def on_event(self, event: object, ctx: object) -> list[object]:
            return []

        def on_fill(self, fill: object, ctx: object) -> None:
            return None

    assert isinstance(_Stub(), WorldAgent)


def test_world_agent_protocol_rejects_incomplete() -> None:  # T-WAG-3
    from qts.world.agents.base import WorldAgent

    class _MissingOnFill:
        agent_id = "x"

        def on_tick(self, ctx: object) -> list[object]:
            return []

        def on_event(self, event: object, ctx: object) -> list[object]:
            return []

    assert not isinstance(_MissingOnFill(), WorldAgent)
