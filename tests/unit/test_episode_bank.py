"""T-BANK-1..4: frozen reproducible episode bank with varied couplings."""

from __future__ import annotations

from datetime import datetime, timedelta

from qts.optimisation.episode_bank import CouplingRanges, generate_episode_bank
from qts.world.scenario import AnonAgentConfig, ScenarioConfig


def _base_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="bank-base",
        symbol="BTCUSDT",
        start=datetime(2023, 12, 13, 18, 0, 0),
        end=datetime(2023, 12, 13, 21, 0, 0),
        tick=timedelta(minutes=1),
        fomc_announcement_at=datetime(2023, 12, 13, 19, 0, 0),
        fomc_expected_rate=5.25,
        starting_price=40_000.0,
        anon_agents=[AnonAgentConfig(agent_id="a1", style="trend")],
        mm_base_spread_bps=2.0,
        mm_vol_widen_k=1.0,
        powell_persona_id="powell",
    )


def test_bank_is_reproducible() -> None:  # T-BANK-1
    s = _base_scenario()
    a = generate_episode_bank(n=4, seed=123, base_scenario=s)
    b = generate_episode_bank(n=4, seed=123, base_scenario=s)
    assert len(a) == len(b) == 4
    # Same seed -> identical closing prices across the bank
    assert [ep.terrain.bars[-1].close for ep in a] == [ep.terrain.bars[-1].close for ep in b]


def test_couplings_vary_across_episodes() -> None:  # T-BANK-2
    bank = generate_episode_bank(n=6, seed=99, base_scenario=_base_scenario())
    last_closes = {round(ep.terrain.bars[-1].close, 4) for ep in bank}
    assert len(last_closes) > 1, "episodes should differ — couplings are randomised"


def test_each_episode_has_text_events() -> None:  # T-BANK-3
    bank = generate_episode_bank(n=3, seed=5, base_scenario=_base_scenario())
    assert all(ep.text_events for ep in bank)


def test_coupling_ranges_defaults() -> None:  # T-BANK-4
    r = CouplingRanges()
    assert r.onset_lag_bars[0] < r.onset_lag_bars[1]
    assert r.peak_bps[0] < r.peak_bps[1]
