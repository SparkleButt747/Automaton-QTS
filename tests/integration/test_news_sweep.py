"""T-NEWS-SWEEP-ACCEPT: end-to-end 5x10 sweep with a stub classifier (no LLM)."""

from __future__ import annotations

from datetime import datetime, timedelta

from qts.config import RiskLimits
from qts.macro.news_signal import NewsSignal
from qts.optimisation.run_news_sweep import NewsSweepResult, run_news_sweep
from qts.world.scenario import AnonAgentConfig, ScenarioConfig


class _StubClassifier:
    """Deterministic classifier: dovish-sounding text -> bull, else neutral. No LLM."""

    def classify(self, event: object) -> NewsSignal:
        text = getattr(event, "text", "").lower()
        if any(w in text for w in ("cut", "dovish", "ease", "accommodativ", "lower")):
            return NewsSignal(direction="bull", confidence=0.8, relevance=0.8, magnitude=0.7)
        if any(w in text for w in ("hike", "hawkish", "tighten", "raise")):
            return NewsSignal(direction="bear", confidence=0.8, relevance=0.8, magnitude=0.7)
        return NewsSignal(direction="neutral", confidence=0.3, relevance=0.3, magnitude=0.3)


def _risk() -> RiskLimits:
    return RiskLimits(
        max_daily_drawdown_pct=0.05,
        max_position_size_pct=0.20,
        max_open_positions=5,
        circuit_breaker_cooldown_seconds=300,
        sentiment_signal_max_scalar=3.0,
    )


def _base_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        name="sweep-base",
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


def test_sweep_runs_end_to_end() -> None:  # T-NEWS-SWEEP-ACCEPT
    result = run_news_sweep(
        base_scenario=_base_scenario(),
        n_episodes=5,
        n_trials=10,
        seed=42,
        risk_limits=_risk(),
        classifier=_StubClassifier(),
    )
    assert isinstance(result, NewsSweepResult)
    assert result.n_completed >= 1
    assert set(result.best_params) == {
        "news_signal_weight",
        "belief_half_life_minutes",
        "entry_threshold",
    }
