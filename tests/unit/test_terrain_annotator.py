"""Unit tests for qts.terrain.annotator.HistoricalAnnotator.

Covers:
- T-ANN-1  to T-ANN-12
- classify() on obvious bull, bear, sideways price series
- Sentiment score mapping: >0.3 => EUPHORIC, <-0.3 => FEARFUL, otherwise NEUTRAL
- Liquidity: uniform volume => ABUNDANT; recently-drained volume => CRISIS
- Fewer than 10 bars returns the neutral default regime
- catalyst and scenario_description are passed through unchanged
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from qts.models.base import Bar, Catalyst, LiquidityLevel, SentimentLevel, Trend, VolLevel
from qts.terrain.annotator import HistoricalAnnotator

# ── Helpers ───────────────────────────────────────────────────────────────────

_T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _make_trending_bars(
    n: int,
    pct_per_bar: float,
    base: float = 100.0,
    volume: float = 1_000.0,
) -> list[Bar]:
    """Generate n bars with a constant per-bar percentage price change."""
    bars: list[Bar] = []
    price = base
    for i in range(n):
        price *= 1.0 + pct_per_bar
        bars.append(
            Bar(
                timestamp=_T0 + timedelta(days=i),
                symbol="SPY",
                open=price,
                high=price * 1.005,
                low=price * 0.995,
                close=price,
                volume=volume,
                bar_count=1,
            )
        )
    return bars


def _make_bars_with_volume_series(volumes: list[float], base: float = 100.0) -> list[Bar]:
    """Generate bars whose close is a gentle drift; volumes come from the supplied list."""
    bars: list[Bar] = []
    for i, vol in enumerate(volumes):
        price = base + i * 0.01
        bars.append(
            Bar(
                timestamp=_T0 + timedelta(days=i),
                symbol="SPY",
                open=price,
                high=price * 1.001,
                low=price * 0.999,
                close=price,
                volume=vol,
                bar_count=1,
            )
        )
    return bars


# ── Trend detection ───────────────────────────────────────────────────────────


class TestHistoricalAnnotatorTrend:
    def test_strong_uptrend_classified_as_bull(self) -> None:  # T-ANN-1
        """A series with >10% total return over the lookback must be classified BULL."""
        # 100 bars, +0.2% per bar => cumulative ~22% return
        bars = _make_trending_bars(100, pct_per_bar=0.002)
        ann = HistoricalAnnotator(trend_lookback=60)
        regime = ann.classify(bars)
        assert regime.trend == Trend.BULL

    def test_strong_downtrend_classified_as_bear(self) -> None:  # T-ANN-2
        """A series with >10% total loss over the lookback must be classified BEAR."""
        # 100 bars, -0.2% per bar => cumulative ~18% loss
        bars = _make_trending_bars(100, pct_per_bar=-0.002, base=200.0)
        ann = HistoricalAnnotator(trend_lookback=60)
        regime = ann.classify(bars)
        assert regime.trend == Trend.BEAR

    def test_flat_price_series_classified_as_sideways(self) -> None:  # T-ANN-3
        """A nearly flat series (<10% move) must be classified SIDEWAYS."""
        # +0.001% per bar over 100 bars => ~0.1% total — well below the 10% threshold
        bars = _make_trending_bars(100, pct_per_bar=0.00001)
        ann = HistoricalAnnotator(trend_lookback=100)
        regime = ann.classify(bars)
        assert regime.trend == Trend.SIDEWAYS


# ── Sentiment mapping ─────────────────────────────────────────────────────────


class TestHistoricalAnnotatorSentiment:
    def test_positive_score_maps_to_euphoric(self) -> None:  # T-ANN-4
        """sentiment_score > 0.3 must produce SentimentLevel.EUPHORIC."""
        bars = _make_trending_bars(50, pct_per_bar=0.001)
        ann = HistoricalAnnotator()
        regime = ann.classify(bars, sentiment_score=0.5)
        assert regime.sentiment == SentimentLevel.EUPHORIC

    def test_negative_score_maps_to_fearful(self) -> None:  # T-ANN-5
        """sentiment_score < -0.3 must produce SentimentLevel.FEARFUL."""
        bars = _make_trending_bars(50, pct_per_bar=0.001)
        ann = HistoricalAnnotator()
        regime = ann.classify(bars, sentiment_score=-0.5)
        assert regime.sentiment == SentimentLevel.FEARFUL

    def test_neutral_score_maps_to_neutral(self) -> None:  # T-ANN-6
        """sentiment_score in (-0.3, 0.3) must produce SentimentLevel.NEUTRAL."""
        bars = _make_trending_bars(50, pct_per_bar=0.001)
        ann = HistoricalAnnotator()
        regime = ann.classify(bars, sentiment_score=0.1)
        assert regime.sentiment == SentimentLevel.NEUTRAL


# ── Liquidity classification ──────────────────────────────────────────────────


class TestHistoricalAnnotatorLiquidity:
    def test_uniform_volume_classified_abundant(self) -> None:  # T-ANN-7
        """Uniform volume across all bars must produce LiquidityLevel.ABUNDANT."""
        bars = _make_bars_with_volume_series([1_000.0] * 100)
        ann = HistoricalAnnotator(liquidity_volume_threshold=0.3)
        regime = ann.classify(bars)
        assert regime.liquidity == LiquidityLevel.ABUNDANT

    def test_recently_drained_volume_classified_crisis(self) -> None:  # T-ANN-8
        """Recent bars with volume << median must produce LiquidityLevel.CRISIS."""
        # 80 bars at 1000, 20 bars at 10 => recent_vol/median = 10/1000 = 0.01 < 0.3
        volumes = [1_000.0] * 80 + [10.0] * 20
        bars = _make_bars_with_volume_series(volumes)
        ann = HistoricalAnnotator(liquidity_volume_threshold=0.3)
        regime = ann.classify(bars)
        assert regime.liquidity == LiquidityLevel.CRISIS


# ── Insufficient data ─────────────────────────────────────────────────────────


class TestHistoricalAnnotatorInsufficientData:
    def test_fewer_than_10_bars_returns_default_regime(self) -> None:  # T-ANN-9
        """classify() with fewer than 10 bars must return the neutral default regime."""
        bars = _make_trending_bars(5, pct_per_bar=0.01)
        ann = HistoricalAnnotator()
        regime = ann.classify(bars)
        assert regime.trend == Trend.SIDEWAYS
        assert regime.volatility == VolLevel.LOW
        assert regime.liquidity == LiquidityLevel.ABUNDANT

    def test_empty_bars_returns_default_regime(self) -> None:  # T-ANN-10
        """classify() with an empty sequence must return the neutral default regime."""
        ann = HistoricalAnnotator()
        regime = ann.classify([])
        assert regime.trend == Trend.SIDEWAYS


# ── Passthrough fields ────────────────────────────────────────────────────────


class TestHistoricalAnnotatorPassthrough:
    def test_catalyst_passed_through(self) -> None:  # T-ANN-11
        """The supplied catalyst must appear verbatim on the returned MacroRegime."""
        bars = _make_trending_bars(50, pct_per_bar=0.001)
        ann = HistoricalAnnotator()
        regime = ann.classify(bars, catalyst=Catalyst.PANDEMIC)
        assert regime.catalyst == Catalyst.PANDEMIC

    def test_scenario_description_passed_through(self) -> None:  # T-ANN-12
        """The supplied scenario_description must appear verbatim on the returned MacroRegime."""
        bars = _make_trending_bars(50, pct_per_bar=0.001)
        ann = HistoricalAnnotator()
        desc = "Custom scenario description for testing"
        regime = ann.classify(bars, scenario_description=desc)
        assert regime.scenario_description == desc
