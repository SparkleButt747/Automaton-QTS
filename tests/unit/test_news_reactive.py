"""Tests for NewsReactiveMomentum — belief integration + alpha blend."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from qts.config import RiskLimits, StrategyParams
from qts.models.base import Bar, SignalSnapshot, VolLevel
from qts.world.events import TextEvent

# Realistic fixtures — see plan note about pydantic model field requirements.
_RISK_LIMITS = RiskLimits(
    max_daily_drawdown_pct=0.05,
    max_position_size_pct=0.20,
    max_open_positions=5,
    circuit_breaker_cooldown_seconds=300,
    sentiment_signal_max_scalar=3.0,
)

_PARAMS = StrategyParams(
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


def _bar(t: datetime, close: float = 30_000.0) -> Bar:
    return Bar(
        timestamp=t,
        symbol="BTCUSDT",
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def _snapshot(t: datetime, combined_alpha: float = 0.0) -> SignalSnapshot:
    return SignalSnapshot(
        timestamp=t,
        symbol="BTCUSDT",
        rsi=50.0,
        macd_histogram=0.0,
        macd_line=0.0,
        macd_signal=0.0,
        bb_position=0.5,
        bb_upper=31_000.0,
        bb_lower=29_000.0,
        atr=200.0,
        momentum_5=0.0,
        vol_level=VolLevel.LOW,
        vol_level_confidence=0.8,
        sentiment_score=0.0,
        combined_alpha=combined_alpha,
    )


class _FakeLLM:
    def __init__(self, response: dict) -> None:
        self._response = response
        self.calls = 0

    async def query(self, *_a: object, **_k: object) -> str:
        import json

        return json.dumps(self._response)

    async def query_json(self, *_a: object, **_k: object) -> dict:
        self.calls += 1
        return self._response


@pytest.mark.asyncio
async def test_news_reactive_updates_belief_on_text(tmp_path: Path) -> None:  # T-NRX-1
    from qts.macro.news_classifier import NewsClassifier
    from qts.strategies.momentum import MomentumStrategy
    from qts.strategies.news_reactive import NewsReactiveMomentum

    llm = _FakeLLM({"direction": "bull", "confidence": 0.8, "relevance": 1.0, "magnitude": 0.5})
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    t0 = datetime(2023, 12, 13, 19, 0, tzinfo=UTC)
    evt = TextEvent(
        timestamp=t0,
        source="fed_press_release",
        persona=None,
        text="Inflation has slowed; we expect cuts in 2024.",
        metadata={"event_kind": "FOMC"},
    )
    await clf.warm_cache_for([evt])

    inner = MomentumStrategy(params=_PARAMS, risk_limits=_RISK_LIMITS)
    strat = NewsReactiveMomentum(
        inner=inner,
        classifier=clf,
        belief_half_life=timedelta(hours=4),
        news_signal_weight=0.5,
    )

    strat.on_text(evt)

    # Belief is updated at the event timestamp.
    assert strat.news_alpha_at(t0) == pytest.approx(0.8 * 1.0 * 0.5)  # 0.4


@pytest.mark.asyncio
async def test_news_reactive_blends_into_snapshot_alpha(tmp_path: Path) -> None:  # T-NRX-2
    from qts.macro.news_classifier import NewsClassifier
    from qts.strategies.momentum import MomentumStrategy
    from qts.strategies.news_reactive import NewsReactiveMomentum

    llm = _FakeLLM({"direction": "bull", "confidence": 1.0, "relevance": 1.0, "magnitude": 1.0})
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    t0 = datetime(2023, 12, 13, 19, 0, tzinfo=UTC)
    evt = TextEvent(
        timestamp=t0,
        source="fed_press_release",
        persona=None,
        text="dovish surprise",
        metadata={},
    )
    await clf.warm_cache_for([evt])

    inner = MomentumStrategy(params=_PARAMS, risk_limits=_RISK_LIMITS)
    strat = NewsReactiveMomentum(
        inner=inner,
        classifier=clf,
        belief_half_life=timedelta(hours=4),
        news_signal_weight=0.5,
    )

    # Update belief.
    strat.on_text(evt)

    # Bar at the same timestamp; base combined_alpha = 0.
    bar = _bar(t0, close=30_000.0)
    snap = _snapshot(t0, combined_alpha=0.0)

    # Capture the snapshot that inner.on_bar sees.
    captured: list[SignalSnapshot] = []
    original_on_bar = inner.on_bar

    def _spy(b, s, pos):
        captured.append(s)
        return original_on_bar(b, s, pos)

    inner.on_bar = _spy  # type: ignore[method-assign]

    strat.on_bar(bar, snap, positions=[])

    assert len(captured) == 1
    blended = captured[0].combined_alpha
    # base=0, news_alpha=1.0, weight=0.5 → blended = 0*0.5 + 1.0*0.5 = 0.5
    assert blended == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_news_reactive_belief_decays_between_bars(tmp_path: Path) -> None:  # T-NRX-3
    from qts.macro.news_classifier import NewsClassifier
    from qts.strategies.momentum import MomentumStrategy
    from qts.strategies.news_reactive import NewsReactiveMomentum

    llm = _FakeLLM({"direction": "bear", "confidence": 1.0, "relevance": 1.0, "magnitude": 1.0})
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    t0 = datetime(2023, 12, 13, 19, 0, tzinfo=UTC)
    evt = TextEvent(
        timestamp=t0,
        source="fed_press_release",
        persona=None,
        text="hawkish surprise",
        metadata={},
    )
    await clf.warm_cache_for([evt])

    strat = NewsReactiveMomentum(
        inner=MomentumStrategy(params=_PARAMS, risk_limits=_RISK_LIMITS),
        classifier=clf,
        belief_half_life=timedelta(hours=2),
        news_signal_weight=1.0,  # all news, no base
    )
    strat.on_text(evt)

    # At t0: news_alpha = -1.0 (full hit).
    assert strat.news_alpha_at(t0) == pytest.approx(-1.0)
    # At t0 + half_life: decayed to half.
    assert strat.news_alpha_at(t0 + timedelta(hours=2)) == pytest.approx(-0.5)


def test_news_reactive_forwards_on_fill(tmp_path: Path) -> None:  # T-NRX-4
    from qts.macro.news_classifier import NewsClassifier
    from qts.models.base import Fill, OrderSide
    from qts.strategies.momentum import MomentumStrategy
    from qts.strategies.news_reactive import NewsReactiveMomentum

    clf = NewsClassifier(llm_client=_FakeLLM({}), cache_dir=tmp_path)
    inner = MomentumStrategy(params=_PARAMS, risk_limits=_RISK_LIMITS)
    strat = NewsReactiveMomentum(inner=inner, classifier=clf)

    captured: list[Fill] = []
    original = inner.on_fill

    def _spy(fill):
        captured.append(fill)
        return original(fill)

    inner.on_fill = _spy  # type: ignore[method-assign]

    f = Fill(
        order_id="x",
        fill_id="y",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        price=30_000.0,
        quantity=0.1,
        commission=0.0,
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )
    strat.on_fill(f)
    assert captured == [f]


@pytest.mark.asyncio
async def test_belief_trajectory_recorded_per_bar(tmp_path: Path) -> None:  # T-NRX-5
    from qts.macro.news_classifier import NewsClassifier
    from qts.strategies.momentum import MomentumStrategy
    from qts.strategies.news_reactive import NewsReactiveMomentum

    llm = _FakeLLM({"direction": "bull", "confidence": 1.0, "relevance": 1.0, "magnitude": 1.0})
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)
    t0 = datetime(2023, 12, 13, 19, 0, tzinfo=UTC)
    evt = TextEvent(timestamp=t0, source="fed", persona=None, text="dovish", metadata={})
    await clf.warm_cache_for([evt])

    strat = NewsReactiveMomentum(
        inner=MomentumStrategy(params=_PARAMS, risk_limits=_RISK_LIMITS),
        classifier=clf,
        news_signal_weight=0.5,
    )

    # One bar BEFORE any news → belief must be 0.0 (causal).
    strat.on_bar(
        _bar(t0 - timedelta(minutes=1)), _snapshot(t0 - timedelta(minutes=1)), positions=[]
    )
    strat.on_text(evt)  # news arrives at t0
    strat.on_bar(_bar(t0), _snapshot(t0), positions=[])

    traj = strat.belief_trajectory
    assert len(traj) == 2
    assert traj[0][1] == 0.0  # zero before news (no look-ahead)
    assert traj[1][1] > 0.0  # active after news
