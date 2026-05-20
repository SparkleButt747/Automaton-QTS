"""Tests for qts.world.agents.scheduler."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from qts.world.agents.base import AgentContext
from qts.world.events import MacroEvent


def _ctx(now: datetime) -> AgentContext:
    return AgentContext(
        now=now, last_price=30000.0, rng=random.Random(0), recent_bars=[], recent_text=[]
    )


def test_scheduler_fires_macro_at_scheduled_tick() -> None:  # T-WSCH-1
    from qts.world.agents.scheduler import SchedulerAgent

    fire = datetime(2025, 3, 19, 14, 0, tzinfo=UTC)
    sched = SchedulerAgent(
        agent_id="sched",
        fomc_at=fire,
        fomc_expected_rate=5.25,
        fomc_actual_rate=5.5,
    )

    pre = sched.on_tick(_ctx(fire - timedelta(minutes=1)))
    assert pre == []

    at = sched.on_tick(_ctx(fire))
    macros = [e for e in at if isinstance(e, MacroEvent)]
    assert len(macros) == 1
    assert macros[0].kind == "fomc"
    assert macros[0].surprise > 0  # hawkish surprise


def test_scheduler_fires_only_once() -> None:  # T-WSCH-2
    from qts.world.agents.scheduler import SchedulerAgent

    fire = datetime(2025, 3, 19, 14, 0, tzinfo=UTC)
    sched = SchedulerAgent(
        agent_id="sched",
        fomc_at=fire,
        fomc_expected_rate=5.25,
        fomc_actual_rate=5.5,
    )
    first = sched.on_tick(_ctx(fire))
    second = sched.on_tick(_ctx(fire + timedelta(minutes=1)))
    assert len(first) == 1
    assert second == []


def test_scheduler_surprise_bucket_hawkish() -> None:  # T-WSCH-3
    from qts.world.agents.scheduler import SchedulerAgent

    sched = SchedulerAgent(
        agent_id="sched",
        fomc_at=datetime(2025, 3, 19, 14, 0, tzinfo=UTC),
        fomc_expected_rate=5.25,
        fomc_actual_rate=5.5,
    )
    assert sched.surprise_bucket() == "hawkish"


def test_scheduler_surprise_bucket_dovish() -> None:  # T-WSCH-4
    from qts.world.agents.scheduler import SchedulerAgent

    sched = SchedulerAgent(
        agent_id="sched",
        fomc_at=datetime(2025, 3, 19, 14, 0, tzinfo=UTC),
        fomc_expected_rate=5.25,
        fomc_actual_rate=5.0,
    )
    assert sched.surprise_bucket() == "dovish"


def test_scheduler_surprise_bucket_neutral() -> None:  # T-WSCH-5
    from qts.world.agents.scheduler import SchedulerAgent

    sched = SchedulerAgent(
        agent_id="sched",
        fomc_at=datetime(2025, 3, 19, 14, 0, tzinfo=UTC),
        fomc_expected_rate=5.25,
        fomc_actual_rate=5.25,
    )
    assert sched.surprise_bucket() == "neutral"
