"""v2 acceptance: NewsReactiveMomentum beats buy-and-hold on 2023-12-13.

Loads the curated dataset, classifies all text events (cache-first; fall back
to live LLM via warm_cache_for if cache is empty), runs NewsReactiveMomentum
through Nautilus, and asserts the strategy's day-end equity exceeds
buy-and-hold equity for the same notional.

Skips if:
    - curated dataset is missing (run scripts/fetch_fomc_data.py)
    - cache is empty AND no working LLM (run scripts/validate_news_classifier.py)
"""

from __future__ import annotations

import asyncio
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


def _llm_reachable() -> bool:
    """Probe whether a usable LLM is available for live cache warming."""
    try:
        import httpx

        with httpx.Client(timeout=2.0) as client:
            response = client.get("http://localhost:11434/api/tags")
            response.raise_for_status()
        return True
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
    reason="no classifier cache and no reachable LLM — run scripts/validate_news_classifier.py first",
)
def test_beats_buy_and_hold_on_dovish_pivot() -> None:  # T-V2-ACCEPT
    from qts.data.real_episode import RealEpisode
    from qts.macro.news_classifier import NewsClassifier
    from qts.nautilus.config import VenueConfig
    from qts.nautilus.real_runner import run_real_backtest
    from qts.oversight.llm_client import create_llm_client
    from qts.strategies.momentum import MomentumStrategy
    from qts.strategies.news_reactive import NewsReactiveMomentum

    episode = RealEpisode.from_disk(_CURATED_ROOT, symbol="BTCUSDT", source="fomc:2023-12-13")

    # Pre-warm classifier cache. If the cache is already populated, no LLM calls happen.
    llm = create_llm_client(backend="ollama")
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

    # Buy-and-hold benchmark: equity is starting_balance × (last_close / first_open).
    vc = VenueConfig()
    bars = episode.terrain.bars
    hold_multiplier = bars[-1].close / bars[0].open
    hold_equity = vc.starting_balance * hold_multiplier

    strat_equity = result.equity_curve[-1] if result.equity_curve else 0.0

    assert strat_equity > hold_equity, (
        f"NewsReactiveMomentum did NOT beat buy-and-hold:\n"
        f"  strategy day-end equity:  {strat_equity:,.2f}\n"
        f"  buy-and-hold equity:      {hold_equity:,.2f}\n"
        f"  shortfall:                {hold_equity - strat_equity:,.2f}\n"
    )
