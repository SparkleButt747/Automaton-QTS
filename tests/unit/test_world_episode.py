"""Tests for qts.world.episode."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from qts.models.base import Bar, Catalyst, LiquidityLevel, SentimentLevel, Trend, VolLevel
from qts.models.terrain import MacroRegime, MarketTerrain


def _make_terrain() -> MarketTerrain:
    start = datetime(2025, 3, 19, tzinfo=UTC)
    bars = [
        Bar(
            symbol="BTCUSDT",
            timestamp=start + timedelta(minutes=i),
            open=30000.0,
            high=30001.0,
            low=29999.0,
            close=30000.5,
            volume=1.0,
        )
        for i in range(5)
    ]
    regime = MacroRegime(
        trend=Trend.SIDEWAYS,
        volatility=VolLevel.LOW,
        liquidity=LiquidityLevel.ABUNDANT,
        sentiment=SentimentLevel.NEUTRAL,
        catalyst=Catalyst.NONE,
        expected_drift=0.0,
        expected_vol=0.01,
        correlation_regime=0.4,
        scenario_description="test",
    )
    return MarketTerrain(
        name="test",
        symbol="BTCUSDT",
        start=bars[0].timestamp,
        end=bars[-1].timestamp,
        regime=regime,
        bars=bars,
    )


def test_order_log_entry_construction() -> None:  # T-WEP-1
    from qts.world.episode import OrderLogEntry

    entry = OrderLogEntry(
        timestamp=datetime(2025, 3, 19, tzinfo=UTC),
        agent_id="anon_0",
        side="BUY",
        quantity=0.1,
        price=30000.0,
        fate="FILLED",
    )
    assert entry.fate == "FILLED"


def test_agent_trace_record_sentiment_and_orders() -> None:  # T-WEP-2
    from qts.world.episode import AgentTrace, OrderLogEntry

    trace = AgentTrace(agent_id="anon_0", style="sentiment")
    trace.sentiment_readings.append((datetime(2025, 3, 19, tzinfo=UTC), 0.7))
    trace.orders.append(
        OrderLogEntry(
            timestamp=datetime(2025, 3, 19, tzinfo=UTC),
            agent_id="anon_0",
            side="BUY",
            quantity=0.1,
            price=30000.0,
            fate="FILLED",
        )
    )

    assert len(trace.sentiment_readings) == 1
    assert len(trace.orders) == 1


def test_simulated_episode_wraps_terrain() -> None:  # T-WEP-3
    from qts.world.episode import SimulatedEpisode

    terrain = _make_terrain()
    ep = SimulatedEpisode(
        terrain=terrain,
        scenario_name="fomc_btcusdt_v1",
        seed=42,
    )

    assert ep.terrain is terrain
    assert ep.scenario_name == "fomc_btcusdt_v1"
    assert ep.seed == 42
    assert ep.agent_traces == {}
    assert ep.llm_corpus_refs == []
    assert ep.order_log == []


def test_simulated_episode_to_json() -> None:  # T-WEP-4
    from qts.world.episode import OrderLogEntry, SimulatedEpisode

    terrain = _make_terrain()
    ep = SimulatedEpisode(
        terrain=terrain,
        scenario_name="fomc_btcusdt_v1",
        seed=42,
    )
    ep.order_log.append(
        OrderLogEntry(
            timestamp=datetime(2025, 3, 19, tzinfo=UTC),
            agent_id="anon_0",
            side="BUY",
            quantity=0.1,
            price=30000.0,
            fate="FILLED",
        )
    )
    ep.llm_corpus_refs.append("powell:fomc:hawkish:v1")

    blob = ep.to_json()
    parsed = json.loads(blob)
    assert parsed["scenario_name"] == "fomc_btcusdt_v1"
    assert parsed["seed"] == 42
    assert parsed["order_log"][0]["side"] == "BUY"
    assert parsed["llm_corpus_refs"] == ["powell:fomc:hawkish:v1"]
