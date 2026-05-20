"""Tests for qts.world.scenario."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def test_anon_config_construction() -> None:  # T-WSC-1
    from qts.world.scenario import AnonAgentConfig

    cfg = AnonAgentConfig(agent_id="anon_0", style="sentiment", aggressiveness=0.5)
    assert cfg.agent_id == "anon_0"
    assert cfg.style == "sentiment"


def test_scenario_config_full() -> None:  # T-WSC-2
    from qts.world.scenario import AnonAgentConfig, ScenarioConfig

    start = datetime(2025, 3, 19, 0, 0, tzinfo=UTC)
    end = datetime(2025, 3, 20, 0, 0, tzinfo=UTC)
    cfg = ScenarioConfig(
        name="fomc_btcusdt_v1",
        symbol="BTCUSDT",
        start=start,
        end=end,
        tick=timedelta(minutes=1),
        fomc_announcement_at=datetime(2025, 3, 19, 14, 0, tzinfo=UTC),
        fomc_expected_rate=5.25,
        starting_price=30000.0,
        anon_agents=[
            AnonAgentConfig(agent_id="anon_0", style="sentiment", aggressiveness=0.5),
            AnonAgentConfig(agent_id="anon_1", style="trend", aggressiveness=0.3),
            AnonAgentConfig(agent_id="anon_2", style="mean_revert", aggressiveness=0.4),
        ],
        mm_base_spread_bps=10.0,
        mm_vol_widen_k=2.0,
        powell_persona_id="jerome_powell",
    )

    assert cfg.name == "fomc_btcusdt_v1"
    assert len(cfg.anon_agents) == 3
    assert cfg.mm_base_spread_bps == 10.0


def test_scenario_rejects_invalid_window() -> None:  # T-WSC-3
    import pytest
    from qts.world.scenario import ScenarioConfig

    with pytest.raises(ValueError, match="end must be after start"):
        ScenarioConfig(
            name="bad",
            symbol="BTCUSDT",
            start=datetime(2025, 3, 20, tzinfo=UTC),
            end=datetime(2025, 3, 19, tzinfo=UTC),
            tick=timedelta(minutes=1),
            fomc_announcement_at=datetime(2025, 3, 19, 14, 0, tzinfo=UTC),
            fomc_expected_rate=5.25,
            starting_price=30000.0,
            anon_agents=[],
            mm_base_spread_bps=10.0,
            mm_vol_widen_k=2.0,
            powell_persona_id="jerome_powell",
        )


def test_scenario_rejects_fomc_outside_window() -> None:  # T-WSC-4
    import pytest
    from qts.world.scenario import ScenarioConfig

    with pytest.raises(ValueError, match="fomc_announcement_at must fall"):
        ScenarioConfig(
            name="bad",
            symbol="BTCUSDT",
            start=datetime(2025, 3, 19, tzinfo=UTC),
            end=datetime(2025, 3, 20, tzinfo=UTC),
            tick=timedelta(minutes=1),
            fomc_announcement_at=datetime(2025, 3, 21, tzinfo=UTC),
            fomc_expected_rate=5.25,
            starting_price=30000.0,
            anon_agents=[],
            mm_base_spread_bps=10.0,
            mm_vol_widen_k=2.0,
            powell_persona_id="jerome_powell",
        )
