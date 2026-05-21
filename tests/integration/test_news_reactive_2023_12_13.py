"""v2.1 acceptance: NewsReactiveMomentum produces trades + a causal belief on 2023-12-13.

Loads the curated dataset, classifies all text events (cache-first; fall back to
live LLM via warm_cache_for if cache is empty), runs NewsReactiveMomentum through
Nautilus, and asserts (1) the strategy makes at least one trade and (2) the news
belief is causal — zero before the first text event and time-varying after.

Beating buy-and-hold is deferred to the Optuna grill (it tunes news_signal_weight /
half-life / entry threshold); here we only prove the mechanic is correct and
look-ahead-free.

Skips if:
    - curated dataset is missing (run scripts/fetch_fomc_data.py)
    - cache is empty AND no working LLM (run scripts/validate_news_classifier.py)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

_CURATED_ROOT = Path("data/real/fomc/2023-12-13")
_CACHE_DIR = Path("data/news_cache")


def _curated_exists() -> bool:
    return (
        (_CURATED_ROOT / "bars.csv").exists()
        and (_CURATED_ROOT / "statement.txt").exists()
        and (_CURATED_ROOT / "press_conf.json").exists()
    )


def _cache_has_entries() -> bool:
    """Cache is considered usable if at least one .json entry exists."""
    if not _CACHE_DIR.exists():
        return False
    return any(_CACHE_DIR.glob("*.json"))


_LLAMACPP_BASE_URL = "http://localhost:8080"


def _llm_reachable() -> bool:
    """Probe whether a llama-server with a loaded model is available.

    Checks the OpenAI-compatible /v1/models endpoint — a non-empty model list
    is a stronger signal than a bare server-up check (it confirms a model is
    actually loaded, not just that the daemon answers).
    """
    try:
        import httpx

        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{_LLAMACPP_BASE_URL}/v1/models")
            response.raise_for_status()
            data = response.json()
        return bool(data.get("data"))
    except Exception:  # noqa: BLE001
        return False


def _build_params_and_risk():
    """Realistic StrategyParams + RiskLimits fixtures (pydantic models require fields)."""
    from qts.config import RiskLimits, SentimentFusionWeights, SignalWeights, StrategyParams

    risk = RiskLimits(
        max_daily_drawdown_pct=0.05,
        max_position_size_pct=0.20,
        max_open_positions=5,
        circuit_breaker_cooldown_seconds=300,
        sentiment_signal_max_scalar=3.0,
    )
    params = StrategyParams(
        version="1.0.0",
        weights=SignalWeights(
            w_rsi=0.20,
            w_macd=0.20,
            w_bb=0.15,
            w_mom=0.15,
            w_sentiment=0.30,
        ),
        entry_threshold=0.25,
        exit_threshold=0.05,
        max_hold_bars=24,
        sentiment_fusion_weights=SentimentFusionWeights(
            news=0.5,
            social=0.3,
            geopolitical=0.2,
        ),
    )
    return params, risk


@pytest.mark.skipif(
    not _curated_exists(), reason="curated dataset missing — run scripts/fetch_fomc_data.py"
)
@pytest.mark.skipif(
    not (_cache_has_entries() or _llm_reachable()),
    reason="no classifier cache and no reachable LLM — run validate_news_classifier.py first",
)
def test_produces_trades_with_causal_belief() -> None:  # T-V2-ACCEPT
    from qts.data.real_episode import RealEpisode
    from qts.macro.news_classifier import NewsClassifier
    from qts.nautilus.config import VenueConfig
    from qts.nautilus.real_runner import run_real_backtest
    from qts.oversight.llm_client import create_llm_client
    from qts.strategies.momentum import MomentumStrategy
    from qts.strategies.news_reactive import NewsReactiveMomentum

    episode = RealEpisode.from_disk(_CURATED_ROOT, symbol="BTCUSDT", source="fomc:2023-12-13")

    # Pre-warm classifier cache. If the cache is already populated, no LLM calls happen.
    llm = create_llm_client(backend="llamacpp")
    classifier = NewsClassifier(llm_client=llm, cache_dir=_CACHE_DIR)
    asyncio.run(classifier.warm_cache_for(episode.text_events))

    params, risk = _build_params_and_risk()

    # Strategy under test
    strat = NewsReactiveMomentum(
        inner=MomentumStrategy(params=params, risk_limits=risk),
        classifier=classifier,
        news_signal_weight=0.5,
    )
    result = run_real_backtest(episode, strat, log_level="ERROR")

    # (1) Trades fire — the mechanic produces decisions.
    assert result.total_trades > 0, "NewsReactiveMomentum made no trades"

    # (2) Belief is causal and evolves bar-by-bar.
    traj = strat.belief_trajectory
    assert traj, "no belief trajectory recorded"
    first_event_ts = min(e.timestamp for e in episode.text_events)
    before = [v for ts, v in traj if ts < first_event_ts]
    after = [v for ts, v in traj if ts >= first_event_ts]
    assert before, "expected bars before the first text event"
    assert all(v == 0.0 for v in before), "belief non-zero before any news — look-ahead!"
    assert any(v != 0.0 for v in after), "belief never activated after the news"
    assert max(after) != min(after), "belief frozen — not evolving bar-by-bar"

    # Informational only: beating buy-and-hold is the Optuna grill's gate, not v2.1's.
    vc = VenueConfig()
    bars = episode.terrain.bars
    hold_equity = vc.starting_balance * (bars[-1].close / bars[0].open)
    strat_equity = result.equity_curve[-1] if result.equity_curve else 0.0
    logging.getLogger(__name__).info(
        "v2.1 2023-12-13: trades=%d strat_equity=%.2f hold_equity=%.2f",
        result.total_trades,
        strat_equity,
        hold_equity,
    )
