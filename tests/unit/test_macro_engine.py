"""Unit tests for qts.macro.engine.MacroEngine.

Covers:
- T-MENG-1  to T-MENG-10
- classify(): delegates to HistoricalAnnotator for price-based regime
- VIX macro override: >30 => HIGH vol + FEARFUL sentiment; <15 => LOW vol + EUPHORIC
- Credit spread override: >5.0 => CRISIS liquidity
- classify_with_regime_override(): preserves categorical fields, recomputes drift/vol
- No LLM calls are made (llm_classifier is None by default)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from qts.macro.engine import MacroEngine
from qts.models.base import Bar, Catalyst, LiquidityLevel, SentimentLevel, Trend, VolLevel
from qts.models.terrain import MacroRegime

# ── Helpers ───────────────────────────────────────────────────────────────────

_T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _make_bars(n: int, pct_per_bar: float = 0.002, base: float = 100.0) -> list[Bar]:
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
                volume=1_000.0,
                bar_count=1,
            )
        )
    return bars


def _make_regime(
    trend: Trend = Trend.BEAR,
    vol: VolLevel = VolLevel.HIGH,
    liq: LiquidityLevel = LiquidityLevel.CRISIS,
) -> MacroRegime:
    return MacroRegime(
        trend=trend,
        volatility=vol,
        liquidity=liq,
        sentiment=SentimentLevel.FEARFUL,
        catalyst=Catalyst.PANDEMIC,
        expected_drift=-3.0,
        expected_vol=0.80,
        correlation_regime=0.90,
        scenario_description="test macro override",
    )


# ── Basic classification ──────────────────────────────────────────────────────


class TestMacroEngineClassify:
    def test_classify_bull_trend_detected(self) -> None:  # T-MENG-1
        """classify() on strongly trending-up bars must return Trend.BULL."""
        engine = MacroEngine()
        bars = _make_bars(100, pct_per_bar=0.002)
        regime = engine.classify(bars)
        assert regime.trend == Trend.BULL

    def test_classify_bear_trend_detected(self) -> None:  # T-MENG-2
        """classify() on strongly trending-down bars must return Trend.BEAR."""
        engine = MacroEngine()
        bars = _make_bars(100, pct_per_bar=-0.002, base=200.0)
        regime = engine.classify(bars)
        assert regime.trend == Trend.BEAR

    def test_classify_returns_macro_regime_instance(self) -> None:  # T-MENG-3
        """classify() must always return a MacroRegime dataclass instance."""
        engine = MacroEngine()
        result = engine.classify(_make_bars(50))
        assert isinstance(result, MacroRegime)

    def test_classify_passthrough_catalyst(self) -> None:  # T-MENG-4
        """Catalyst supplied to classify() must appear on the returned MacroRegime."""
        engine = MacroEngine()
        regime = engine.classify(_make_bars(50), catalyst=Catalyst.GEOPOLITICAL)
        assert regime.catalyst == Catalyst.GEOPOLITICAL


# ── VIX macro overrides ───────────────────────────────────────────────────────


class TestMacroEngineVixOverride:
    def test_high_vix_sets_high_volatility(self) -> None:  # T-MENG-5
        """VIX > 30 must force VolLevel.HIGH regardless of price-based classification."""
        engine = MacroEngine()
        # Use sideways bars to get a neutral price-based signal
        bars = _make_bars(100, pct_per_bar=0.00001)
        regime = engine.classify(bars, macro_indicators={"vix": 40.0})
        assert regime.volatility == VolLevel.HIGH

    def test_high_vix_sets_fearful_sentiment(self) -> None:  # T-MENG-6
        """VIX > 30 must force SentimentLevel.FEARFUL."""
        engine = MacroEngine()
        bars = _make_bars(100, pct_per_bar=0.00001)
        regime = engine.classify(bars, macro_indicators={"vix": 35.0})
        assert regime.sentiment == SentimentLevel.FEARFUL

    def test_low_vix_sets_low_volatility(self) -> None:  # T-MENG-7
        """VIX < 15 must force VolLevel.LOW."""
        engine = MacroEngine()
        bars = _make_bars(100, pct_per_bar=0.00001)
        regime = engine.classify(bars, macro_indicators={"vix": 10.0})
        assert regime.volatility == VolLevel.LOW

    def test_low_vix_sets_euphoric_sentiment(self) -> None:  # T-MENG-8
        """VIX < 15 must force SentimentLevel.EUPHORIC."""
        engine = MacroEngine()
        bars = _make_bars(100, pct_per_bar=0.00001)
        regime = engine.classify(bars, macro_indicators={"vix": 12.0})
        assert regime.sentiment == SentimentLevel.EUPHORIC


# ── Credit spread override ────────────────────────────────────────────────────


class TestMacroEngineCreditSpreadOverride:
    def test_high_credit_spread_sets_crisis_liquidity(self) -> None:  # T-MENG-9
        """credit_spread > 5.0 must force LiquidityLevel.CRISIS."""
        engine = MacroEngine()
        bars = _make_bars(100, pct_per_bar=0.001)
        regime = engine.classify(bars, macro_indicators={"credit_spread": 8.0})
        assert regime.liquidity == LiquidityLevel.CRISIS


# ── classify_with_regime_override ─────────────────────────────────────────────


class TestMacroEngineClassifyWithRegimeOverride:
    def test_categorical_fields_preserved(self) -> None:  # T-MENG-10
        """classify_with_regime_override must carry through all categorical regime fields."""
        engine = MacroEngine()
        bars = _make_bars(100, pct_per_bar=0.002)
        source_regime = _make_regime(trend=Trend.BEAR, vol=VolLevel.HIGH, liq=LiquidityLevel.CRISIS)
        result = engine.classify_with_regime_override(bars, source_regime)

        assert result.trend == Trend.BEAR
        assert result.volatility == VolLevel.HIGH
        assert result.liquidity == LiquidityLevel.CRISIS
        assert result.catalyst == source_regime.catalyst
        assert result.scenario_description == source_regime.scenario_description

    def test_drift_computed_from_bars(self) -> None:  # T-MENG-11
        """classify_with_regime_override must recompute expected_drift from bar returns."""
        engine = MacroEngine()
        # Strongly trending up bars — drift must be positive
        bars = _make_bars(100, pct_per_bar=0.003)
        source_regime = _make_regime()
        result = engine.classify_with_regime_override(bars, source_regime)
        assert result.expected_drift > 0.0

    def test_expected_vol_computed_from_bars(self) -> None:  # T-MENG-12
        """classify_with_regime_override must produce a positive expected_vol from ATR."""
        engine = MacroEngine()
        bars = _make_bars(100, pct_per_bar=0.002)
        source_regime = _make_regime()
        result = engine.classify_with_regime_override(bars, source_regime)
        assert result.expected_vol > 0.0
