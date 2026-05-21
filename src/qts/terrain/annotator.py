"""HistoricalAnnotator — labels real market data with MacroRegime metadata.

Two modes:
1. Manual: regime labels come from YAML config files
2. Auto: uses RegimeDetector (vol) + SentimentFusion (sentiment) to auto-label

This is the Macro Engine operating in classifier mode (Phase 3):
real history → classify → "this WAS a crisis" → MarketTerrain with real bars + label.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from qts.models.base import (
    Bar,
    Catalyst,
    LiquidityLevel,
    SentimentLevel,
    Trend,
    VolLevel,
)
from qts.models.terrain import MacroRegime
from qts.signals.indicators import compute_atr

logger = logging.getLogger(__name__)


class HistoricalAnnotator:
    """Annotates historical bar data with macro regime classifications.

    The annotator uses statistical features of the bar data to classify
    the regime. It can be calibrated with external macro data or run
    in a purely price-based mode.
    """

    def __init__(
        self,
        vol_high_threshold: float = 0.7,
        vol_low_threshold: float = 0.3,
        trend_lookback: int = 60,
        liquidity_volume_threshold: float = 0.3,
    ) -> None:
        """Initialise the annotator.

        Args:
            vol_high_threshold: ATR percentile above which vol is HIGH.
            vol_low_threshold: ATR percentile below which vol is LOW.
            trend_lookback: Number of bars to look back for trend detection.
            liquidity_volume_threshold: Volume fraction below which liquidity is TIGHT.
        """
        self._vol_high = vol_high_threshold
        self._vol_low = vol_low_threshold
        self._trend_lookback = trend_lookback
        self._liq_threshold = liquidity_volume_threshold

    def classify(
        self,
        bars: Sequence[Bar],
        sentiment_score: float = 0.0,
        catalyst: Catalyst = Catalyst.NONE,
        scenario_description: str = "",
    ) -> MacroRegime:
        """Classify a bar sequence into a MacroRegime.

        Uses price-based heuristics to determine trend, volatility,
        and liquidity. Sentiment and catalyst are provided externally
        (from NLP pipeline or manual annotation).

        Args:
            bars: Bar data to classify.
            sentiment_score: External sentiment score in [-1, 1].
            catalyst: External catalyst classification.
            scenario_description: Human-readable scenario text.

        Returns:
            MacroRegime classification for the period.
        """
        if len(bars) < 10:
            return self._default_regime(scenario_description)

        closes = np.array([b.close for b in bars], dtype=np.float64)
        highs = np.array([b.high for b in bars], dtype=np.float64)
        lows = np.array([b.low for b in bars], dtype=np.float64)
        volumes = np.array([b.volume for b in bars], dtype=np.float64)

        # Trend detection
        trend = self._detect_trend(closes)

        # Volatility classification
        volatility, expected_vol = self._classify_volatility(highs, lows, closes)

        # Liquidity assessment
        liquidity = self._classify_liquidity(volumes)

        # Sentiment mapping
        sentiment = self._map_sentiment(sentiment_score)

        # Drift estimation (annualised return)
        total_return = (closes[-1] / closes[0]) - 1.0
        n_bars = len(closes)
        bars_per_year = 252
        expected_drift = total_return * (bars_per_year / n_bars) if n_bars > 0 else 0.0

        # Correlation regime (placeholder — requires multi-asset data)
        correlation_regime = 0.5

        return MacroRegime(
            trend=trend,
            volatility=volatility,
            liquidity=liquidity,
            sentiment=sentiment,
            catalyst=catalyst,
            expected_drift=expected_drift,
            expected_vol=expected_vol,
            correlation_regime=correlation_regime,
            scenario_description=scenario_description,
        )

    # ------------------------------------------------------------------
    # Feature extractors
    # ------------------------------------------------------------------

    def _detect_trend(self, closes: np.ndarray) -> Trend:
        """Detect trend from price momentum over the lookback window."""
        lookback = min(self._trend_lookback, len(closes))
        if lookback < 5:
            return Trend.SIDEWAYS

        recent = closes[-lookback:]
        total_return = (recent[-1] / recent[0]) - 1.0

        # Simple threshold: >10% up = BULL, >10% down = BEAR
        if total_return > 0.10:
            return Trend.BULL
        elif total_return < -0.10:
            return Trend.BEAR
        return Trend.SIDEWAYS

    def _classify_volatility(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
    ) -> tuple[VolLevel, float]:
        """Classify volatility from ATR percentile."""
        atr = compute_atr(highs, lows, closes)
        valid = atr[~np.isnan(atr)]

        if len(valid) < 5:
            return VolLevel.LOW, 0.15

        # Normalise ATR by price level
        mean_close = float(np.mean(closes[closes > 0]))
        normalised = valid / mean_close if mean_close > 0 else valid
        percentile = float(np.mean(normalised))

        # Expected vol (annualised)
        expected_vol = float(np.mean(normalised)) * np.sqrt(252)

        if percentile > self._vol_high * float(np.max(normalised)):
            return VolLevel.HIGH, expected_vol
        elif percentile < self._vol_low * float(np.max(normalised)):
            return VolLevel.LOW, expected_vol
        return VolLevel.TRANSITIONING, expected_vol

    def _classify_liquidity(self, volumes: np.ndarray) -> LiquidityLevel:
        """Classify liquidity from volume profile."""
        valid = volumes[volumes > 0]
        if len(valid) < 5:
            return LiquidityLevel.ABUNDANT

        median_vol = float(np.median(valid))
        recent_vol = float(np.mean(valid[-20:])) if len(valid) >= 20 else float(np.mean(valid))

        ratio = recent_vol / median_vol if median_vol > 0 else 1.0

        if ratio < self._liq_threshold:
            return LiquidityLevel.CRISIS
        elif ratio < 0.7:
            return LiquidityLevel.TIGHT
        return LiquidityLevel.ABUNDANT

    def _map_sentiment(self, score: float) -> SentimentLevel:
        """Map a [-1, 1] sentiment score to a SentimentLevel."""
        if score > 0.3:
            return SentimentLevel.EUPHORIC
        elif score < -0.3:
            return SentimentLevel.FEARFUL
        return SentimentLevel.NEUTRAL

    def _default_regime(self, description: str) -> MacroRegime:
        """Return a neutral default regime for insufficient data."""
        return MacroRegime(
            trend=Trend.SIDEWAYS,
            volatility=VolLevel.LOW,
            liquidity=LiquidityLevel.ABUNDANT,
            sentiment=SentimentLevel.NEUTRAL,
            catalyst=Catalyst.NONE,
            expected_drift=0.0,
            expected_vol=0.15,
            correlation_regime=0.5,
            scenario_description=description or "Insufficient data for classification",
        )
