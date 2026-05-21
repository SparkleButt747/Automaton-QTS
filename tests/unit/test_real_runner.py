"""Tests for run_real_backtest — terrain backtest with text-event dispatch."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from qts.models.base import Bar
from qts.models.terrain import MarketTerrain
from qts.world.events import TextEvent


def _terrain_with_n_bars(n: int) -> MarketTerrain:
    from qts.models.base import (
        Catalyst,
        LiquidityLevel,
        SentimentLevel,
        Trend,
        VolLevel,
    )
    from qts.models.terrain import MacroRegime

    start = datetime(2023, 12, 13, 18, 50, tzinfo=UTC)
    bars = [
        Bar(
            timestamp=start + timedelta(minutes=i),
            symbol="BTCUSDT",
            open=40_000.0 + i,
            high=40_000.0 + i + 5,
            low=40_000.0 + i - 5,
            close=40_000.0 + i,
            volume=1.0,
        )
        for i in range(n)
    ]
    regime = MacroRegime(
        trend=Trend.SIDEWAYS,
        volatility=VolLevel.LOW,
        liquidity=LiquidityLevel.ABUNDANT,
        sentiment=SentimentLevel.NEUTRAL,
        catalyst=Catalyst.MACRO_EVENT,
        expected_drift=0.0,
        expected_vol=0.01,
        correlation_regime=0.5,
        scenario_description="test",
    )
    return MarketTerrain(
        name="test",
        symbol="BTCUSDT",
        start=bars[0].timestamp,
        end=bars[-1].timestamp,
        regime=regime,
        bars=bars,
        event_calendar=[],
    )


def test_run_real_backtest_dispatches_text_events_in_order(tmp_path: Path) -> None:  # T-REAL-1
    from qts.data.real_episode import RealEpisode
    from qts.nautilus.real_runner import run_real_backtest

    seen: list[TextEvent] = []

    class _Strategy:
        params = None
        name = "spy"

        def on_bar(self, *a, **k):
            return []

        def on_fill(self, *a, **k):
            pass

        def on_text(self, event: TextEvent) -> None:
            seen.append(event)

    terrain = _terrain_with_n_bars(20)
    events = [
        TextEvent(
            timestamp=terrain.bars[5].timestamp,
            source="fed_press_release",
            persona=None,
            text="event-A",
            metadata={},
        ),
        TextEvent(
            timestamp=terrain.bars[15].timestamp,
            source="powell_press_conf",
            persona="powell",
            text="event-B",
            metadata={},
        ),
    ]
    episode = RealEpisode(terrain=terrain, text_events=events, source="test")

    run_real_backtest(episode, _Strategy(), log_level="ERROR")

    assert [e.text for e in seen] == ["event-A", "event-B"]


def test_run_real_backtest_returns_backtest_result(tmp_path: Path) -> None:  # T-REAL-2
    from qts.config import RiskLimits, StrategyParams
    from qts.data.real_episode import RealEpisode
    from qts.nautilus.real_runner import run_real_backtest
    from qts.strategies.momentum import MomentumStrategy

    risk_limits = RiskLimits(
        max_daily_drawdown_pct=0.05,
        max_position_size_pct=0.20,
        max_open_positions=5,
        circuit_breaker_cooldown_seconds=300,
        sentiment_signal_max_scalar=3.0,
    )
    params = StrategyParams(
        version="1.0.0",
        weights={
            "w_rsi": 0.20,
            "w_macd": 0.20,
            "w_bb": 0.15,
            "w_mom": 0.15,
            "w_sentiment": 0.30,
        },
        entry_threshold=0.25,
        exit_threshold=0.05,
        max_hold_bars=24,
        sentiment_fusion_weights={"news": 0.5, "social": 0.3, "geopolitical": 0.2},
    )

    terrain = _terrain_with_n_bars(60)
    episode = RealEpisode(terrain=terrain, text_events=[], source="test")
    strat = MomentumStrategy(params=params, risk_limits=risk_limits)

    result = run_real_backtest(episode, strat, log_level="ERROR")
    assert len(result.equity_curve) > 0


def test_run_real_backtest_interleaves_news_between_bars(tmp_path: Path) -> None:  # T-REAL-3
    from qts.data.real_episode import RealEpisode
    from qts.nautilus.real_runner import run_real_backtest

    sequence: list[tuple[str, object]] = []

    class _Strategy:
        params = None
        name = "spy"

        def on_bar(self, bar, *a, **k):
            sequence.append(("bar", bar.timestamp))
            return []

        def on_fill(self, *a, **k):
            pass

        def on_text(self, event) -> None:
            sequence.append(("news", event.timestamp))

    terrain = _terrain_with_n_bars(80)
    # Event mid-stream and strictly between two bars, well after indicator warmup
    # (the actor only forwards bars to the inner strategy once the pipeline has
    # enough history, so the event must land after that to prove interleaving).
    evt = TextEvent(
        timestamp=terrain.bars[65].timestamp + timedelta(seconds=30),
        source="fed_press_release",
        persona=None,
        text="mid-stream",
        metadata={},
    )
    episode = RealEpisode(terrain=terrain, text_events=[evt], source="test")
    run_real_backtest(episode, _Strategy(), log_level="ERROR")

    kinds = [k for k, _ in sequence]
    assert "news" in kinds  # delivered via the engine, not pre-dispatched
    news_idx = kinds.index("news")
    assert news_idx > 0  # bars recorded BEFORE the news (not pre-dispatched up front)
    assert "bar" in kinds[news_idx + 1 :]  # and bars AFTER → genuinely interleaved
