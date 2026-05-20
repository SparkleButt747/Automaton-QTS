"""MacroEngine — orchestrates fundamentals, NLP, and macro indicators into MacroRegime.

The Macro Engine is the 'Track Generator': it decides what kind of market
environment we are in and produces a MacroRegime that conditions everything
downstream.

Two modes:
- classify(): Phase 3 — real data annotation ("this WAS a crisis")
- generate(): Phase 8 — hypothetical scenario ("this WOULD BE a crisis")
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from qts.models.base import (
    Bar,
    Catalyst,
    LiquidityLevel,
    SentimentLevel,
    VolLevel,
)
from qts.models.terrain import MacroRegime
from qts.terrain.annotator import HistoricalAnnotator

logger = logging.getLogger(__name__)


class MacroEngine:
    """Orchestrates regime classification from multiple data sources.

    Combines:
    - Price-based features (via HistoricalAnnotator)
    - News sentiment (via FinBERT / LLM)
    - Social sentiment (via VADER)
    - Geopolitical events (via GDELT)
    - Macro indicators (VIX, yield curve, CPI — future)

    The engine produces a MacroRegime that feeds into MarketTerrain
    construction or directly into strategy conditioning.
    """

    def __init__(
        self,
        annotator: HistoricalAnnotator | None = None,
        llm_classifier: Any | None = None,
        sentiment_fusion: Any | None = None,
    ) -> None:
        """Initialise the MacroEngine.

        Args:
            annotator: Price-based regime annotator.
            llm_classifier: LLM-based regime classifier (from oversight).
            sentiment_fusion: Multi-source sentiment fusion engine.
        """
        self._annotator = annotator or HistoricalAnnotator()
        self._llm_classifier = llm_classifier
        self._sentiment_fusion = sentiment_fusion

    def classify(
        self,
        bars: Sequence[Bar],
        headlines: list[str] | None = None,
        social_scores: list[float] | None = None,
        macro_indicators: dict[str, float] | None = None,
        catalyst: Catalyst = Catalyst.NONE,
        scenario_description: str = "",
    ) -> MacroRegime:
        """Classify the macro regime from real data (Phase 3 mode).

        Combines price-based analysis with optional NLP and macro data
        to produce a comprehensive regime classification.

        Args:
            bars: Historical bar data for the period.
            headlines: Recent news headlines for NLP sentiment.
            social_scores: Social media sentiment scores.
            macro_indicators: Dict of macro indicators (e.g., {"vix": 25.3}).
            catalyst: Known event catalyst.
            scenario_description: Human-readable description.

        Returns:
            MacroRegime classification.
        """
        # Start with price-based classification
        sentiment_score = 0.0

        # Compute sentiment from multiple sources if available
        if self._sentiment_fusion is not None:
            news_score = self._compute_news_sentiment(headlines or [])
            social_score = float(sum(social_scores) / len(social_scores)) if social_scores else 0.0
            geo_score = 0.0  # placeholder for GDELT integration
            sentiment_score = self._sentiment_fusion.fuse(news_score, social_score, geo_score)

        # Price-based regime from annotator
        regime = self._annotator.classify(
            bars=bars,
            sentiment_score=sentiment_score,
            catalyst=catalyst,
            scenario_description=scenario_description,
        )

        # Override with macro indicators if available
        if macro_indicators:
            regime = self._apply_macro_overrides(regime, macro_indicators)

        # Override with LLM classification if available
        if self._llm_classifier is not None and headlines:
            regime = self._apply_llm_override(regime, headlines)

        return regime

    def classify_with_regime_override(
        self,
        bars: Sequence[Bar],
        regime: MacroRegime,
    ) -> MacroRegime:
        """Return a regime with price-derived fields filled in from bars.

        Uses the provided regime's categorical fields (trend, sentiment, etc.)
        but computes expected_drift and expected_vol from the actual bar data.
        """
        import numpy as np  # noqa: PLC0415

        from qts.signals.indicators import compute_atr  # noqa: PLC0415

        closes = np.array([b.close for b in bars], dtype=np.float64)
        highs = np.array([b.high for b in bars], dtype=np.float64)
        lows = np.array([b.low for b in bars], dtype=np.float64)

        # Compute drift
        total_return = (closes[-1] / closes[0]) - 1.0 if len(closes) > 1 else 0.0
        n_bars = len(closes)
        expected_drift = total_return * (252 / n_bars) if n_bars > 0 else 0.0

        # Compute vol
        atr = compute_atr(highs, lows, closes)
        valid = atr[~np.isnan(atr)]
        mean_close = float(np.mean(closes[closes > 0])) if len(closes[closes > 0]) > 0 else 1.0
        expected_vol = float(np.mean(valid / mean_close) * np.sqrt(252)) if len(valid) > 0 else 0.15

        return MacroRegime(
            trend=regime.trend,
            volatility=regime.volatility,
            liquidity=regime.liquidity,
            sentiment=regime.sentiment,
            catalyst=regime.catalyst,
            expected_drift=expected_drift,
            expected_vol=expected_vol,
            correlation_regime=regime.correlation_regime,
            scenario_description=regime.scenario_description,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_news_sentiment(self, headlines: list[str]) -> float:
        """Compute aggregate news sentiment from headlines."""
        if not headlines:
            return 0.0
        # Placeholder — in production, FinBERT runs here
        return 0.0

    def _apply_macro_overrides(
        self, regime: MacroRegime, indicators: dict[str, float]
    ) -> MacroRegime:
        """Override regime fields based on macro indicator values."""
        volatility = regime.volatility
        liquidity = regime.liquidity
        sentiment = regime.sentiment

        vix = indicators.get("vix")
        if vix is not None:
            if vix > 30:
                volatility = VolLevel.HIGH
                sentiment = SentimentLevel.FEARFUL
            elif vix < 15:
                volatility = VolLevel.LOW
                sentiment = SentimentLevel.EUPHORIC

        # Credit spreads indicate liquidity stress
        credit_spread = indicators.get("credit_spread")
        if credit_spread is not None and credit_spread > 5.0:
            liquidity = LiquidityLevel.CRISIS

        if (
            volatility == regime.volatility
            and liquidity == regime.liquidity
            and sentiment == regime.sentiment
        ):
            return regime

        return MacroRegime(
            trend=regime.trend,
            volatility=volatility,
            liquidity=liquidity,
            sentiment=sentiment,
            catalyst=regime.catalyst,
            expected_drift=regime.expected_drift,
            expected_vol=regime.expected_vol,
            correlation_regime=regime.correlation_regime,
            scenario_description=regime.scenario_description,
        )

    def _apply_llm_override(self, regime: MacroRegime, headlines: list[str]) -> MacroRegime:
        """Override regime with LLM classification if confidence is high enough."""
        # Placeholder — in production, calls RegimeClassifierLLM
        return regime
