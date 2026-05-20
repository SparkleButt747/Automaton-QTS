"""Tests for qts.world.agents.anon."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from qts.models.base import Bar
from qts.world.agents.base import AgentContext, AgentOrder
from qts.world.events import TextEvent


def _bars(closes: list[float], start: datetime) -> list[Bar]:
    return [
        Bar(
            symbol="BTCUSDT",
            timestamp=start + timedelta(minutes=i),
            open=c,
            high=c,
            low=c,
            close=c,
            volume=1.0,
        )
        for i, c in enumerate(closes)
    ]


def test_sentiment_anon_buys_on_dovish_text() -> None:  # T-WAN-1
    from qts.world.agents.anon import AnonRetailAgent

    agent = AnonRetailAgent(
        agent_id="anon_0",
        style="sentiment",
        aggressiveness=1.0,
        base_order_size=0.05,
    )
    text = TextEvent(
        timestamp=datetime(2025, 3, 19, 14, 1, tzinfo=UTC),
        source="powell",
        persona="powell",
        text="Progress on inflation; we may be near the peak.",
        metadata={},
    )
    ctx = AgentContext(
        now=text.timestamp,
        last_price=30000.0,
        rng=random.Random(0),
        recent_bars=_bars([30000.0] * 5, datetime(2025, 3, 19, 14, 0, tzinfo=UTC)),
        recent_text=[text],
    )

    out = agent.on_event(text, ctx)
    orders = [o for o in out if isinstance(o, AgentOrder)]
    assert len(orders) == 1
    assert orders[0].side == "BUY"
    assert orders[0].quantity > 0


def test_sentiment_anon_sells_on_hawkish_text() -> None:  # T-WAN-2
    from qts.world.agents.anon import AnonRetailAgent

    agent = AnonRetailAgent(
        agent_id="anon_0",
        style="sentiment",
        aggressiveness=1.0,
        base_order_size=0.05,
    )
    text = TextEvent(
        timestamp=datetime(2025, 3, 19, 14, 1, tzinfo=UTC),
        source="powell",
        persona="powell",
        text="Inflation stubborn; further hikes warranted; restrictive policy.",
        metadata={},
    )
    ctx = AgentContext(
        now=text.timestamp,
        last_price=30000.0,
        rng=random.Random(0),
        recent_bars=_bars([30000.0] * 5, datetime(2025, 3, 19, 14, 0, tzinfo=UTC)),
        recent_text=[text],
    )

    out = agent.on_event(text, ctx)
    orders = [o for o in out if isinstance(o, AgentOrder)]
    assert len(orders) == 1
    assert orders[0].side == "SELL"


def test_trend_anon_buys_on_uptrend() -> None:  # T-WAN-3
    from qts.world.agents.anon import AnonRetailAgent

    agent = AnonRetailAgent(
        agent_id="anon_1",
        style="trend",
        aggressiveness=1.0,
        base_order_size=0.05,
        trend_lookback=5,
    )
    rising = _bars(
        [30000.0, 30030.0, 30060.0, 30090.0, 30120.0],
        datetime(2025, 3, 19, 14, 0, tzinfo=UTC),
    )
    ctx = AgentContext(
        now=rising[-1].timestamp,
        last_price=30120.0,
        rng=random.Random(0),
        recent_bars=rising,
        recent_text=[],
    )

    out = agent.on_tick(ctx)
    orders = [o for o in out if isinstance(o, AgentOrder)]
    assert orders and orders[0].side == "BUY"


def test_mean_revert_anon_sells_on_spike() -> None:  # T-WAN-4
    from qts.world.agents.anon import AnonRetailAgent

    agent = AnonRetailAgent(
        agent_id="anon_2",
        style="mean_revert",
        aggressiveness=1.0,
        base_order_size=0.05,
        revert_threshold=0.005,
        revert_lookback=5,
    )
    spike = _bars(
        [30000.0, 30000.0, 30000.0, 30000.0, 30500.0],  # +1.6% spike on last bar
        datetime(2025, 3, 19, 14, 0, tzinfo=UTC),
    )
    ctx = AgentContext(
        now=spike[-1].timestamp,
        last_price=30500.0,
        rng=random.Random(0),
        recent_bars=spike,
        recent_text=[],
    )

    out = agent.on_tick(ctx)
    orders = [o for o in out if isinstance(o, AgentOrder)]
    assert orders and orders[0].side == "SELL"


def test_anon_rejects_unknown_style() -> None:  # T-WAN-5
    import pytest
    from qts.world.agents.anon import AnonRetailAgent

    with pytest.raises(ValueError, match="unknown style"):
        AnonRetailAgent(
            agent_id="bad",
            style="cosmic",
            aggressiveness=1.0,
            base_order_size=0.05,
        )
