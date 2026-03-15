"""Signal pipeline: computes all indicators, runs regime detection,
and assembles a SignalSnapshot from a sequence of Bar objects.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Sequence

import numpy as np

from qts.models.base import Bar, SignalSnapshot, VolRegime
from qts.signals.alpha import combined_alpha
from qts.signals.indicators import (
    compute_atr,
    compute_bb_position,
    compute_bollinger_bands,
    compute_macd,
    compute_momentum,
    compute_rsi,
    normalise_rsi,
)
from qts.signals.regime import RegimeDetector

logger = logging.getLogger(__name__)

# Minimum bars required before we can produce a meaningful snapshot
_MIN_BARS_FOR_SNAPSHOT = 30


class SignalPipeline:
    """Compute technical indicators, detect regime, and build SignalSnapshot.

    The pipeline is stateful: it maintains a RegimeDetector that is fit
    once sufficient history is available and re-used across calls.

    Usage::

        pipeline = SignalPipeline(symbol="BTCUSDT")
        snapshot = pipeline.compute(bars)

    Attributes:
        symbol: Ticker symbol this pipeline processes.
        _detector: HMM-based RegimeDetector instance.
        _detector_fitted: Whether the detector has been fitted.
    """

    def __init__(
        self,
        symbol: str,
        regime_detector: Optional[RegimeDetector] = None,
        sentiment_score: float = 0.0,
    ) -> None:
        """Initialise the signal pipeline.

        Args:
            symbol: Ticker symbol being processed.
            regime_detector: Optional pre-fitted RegimeDetector; a new one
                is created if not provided.
            sentiment_score: Default sentiment score to embed in snapshots
                (can be overridden per call).
        """
        self.symbol = symbol
        self._detector: RegimeDetector = regime_detector or RegimeDetector()
        self._detector_fitted: bool = False
        self._default_sentiment: float = sentiment_score

    def compute(
        self,
        bars: Sequence[Bar],
        sentiment_score: Optional[float] = None,
        params: Optional[object] = None,
    ) -> Optional[SignalSnapshot]:
        """Compute all indicators and return a SignalSnapshot for the last bar.

        Returns None if insufficient bars are available.

        Args:
            bars: Ordered sequence of Bar objects (oldest first).
            sentiment_score: Override for sentiment score; uses default if None.
            params: Optional StrategyParams; if provided, combined_alpha is
                    computed and embedded in the snapshot.

        Returns:
            SignalSnapshot for the most recent bar, or None.
        """
        if len(bars) < _MIN_BARS_FOR_SNAPSHOT:
            logger.debug(
                "Insufficient bars (%d < %d) for snapshot",
                len(bars),
                _MIN_BARS_FOR_SNAPSHOT,
            )
            return None

        closes = np.array([b.close for b in bars], dtype=np.float64)
        highs = np.array([b.high for b in bars], dtype=np.float64)
        lows = np.array([b.low for b in bars], dtype=np.float64)

        # ── Indicators ───────────────────────────────────────────────────────
        rsi_arr = compute_rsi(closes)
        macd_line, signal_line, histogram = compute_macd(closes)
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(closes)
        atr_arr = compute_atr(highs, lows, closes)
        mom_arr = compute_momentum(closes)

        # Latest valid values
        def _last_valid(arr: np.ndarray) -> float:
            valid = arr[~np.isnan(arr)]
            return float(valid[-1]) if len(valid) > 0 else float("nan")

        rsi_val = _last_valid(rsi_arr)
        macd_line_val = _last_valid(macd_line)
        macd_signal_val = _last_valid(signal_line)
        macd_hist_val = _last_valid(histogram)
        bb_upper_val = _last_valid(bb_upper)
        bb_lower_val = _last_valid(bb_lower)
        atr_val = _last_valid(atr_arr)
        mom_val = _last_valid(mom_arr)

        last_close = closes[-1]
        bb_pos = compute_bb_position(last_close, bb_upper_val, bb_lower_val)

        # ── Regime detection ─────────────────────────────────────────────────
        if not self._detector_fitted:
            try:
                self._detector.fit(atr_arr)
                self._detector_fitted = True
            except ValueError as exc:
                logger.warning("Regime detector not ready: %s", exc)

        if self._detector_fitted:
            regime, regime_confidence = self._detector.predict(atr_arr)
        else:
            regime = VolRegime.LOW
            regime_confidence = 0.0

        # ── Assemble snapshot ────────────────────────────────────────────────
        effective_sentiment = (
            sentiment_score if sentiment_score is not None else self._default_sentiment
        )

        last_bar = bars[-1]
        snapshot = SignalSnapshot(
            timestamp=last_bar.timestamp,
            symbol=self.symbol,
            rsi=rsi_val,
            macd_histogram=macd_hist_val,
            macd_line=macd_line_val,
            macd_signal=macd_signal_val,
            bb_position=bb_pos,
            bb_upper=bb_upper_val,
            bb_lower=bb_lower_val,
            atr=atr_val,
            momentum_5=mom_val,
            vol_regime=regime,
            vol_regime_confidence=regime_confidence,
            sentiment_score=effective_sentiment,
            combined_alpha=0.0,  # placeholder; updated below if params provided
        )

        # ── Combined alpha ───────────────────────────────────────────────────
        if params is not None:
            # Import here to avoid circular imports
            from qts.config import StrategyParams  # noqa: PLC0415

            if isinstance(params, StrategyParams):
                alpha = combined_alpha(snapshot, params)
                # SignalSnapshot is frozen; rebuild with alpha
                snapshot = SignalSnapshot(
                    timestamp=snapshot.timestamp,
                    symbol=snapshot.symbol,
                    rsi=snapshot.rsi,
                    macd_histogram=snapshot.macd_histogram,
                    macd_line=snapshot.macd_line,
                    macd_signal=snapshot.macd_signal,
                    bb_position=snapshot.bb_position,
                    bb_upper=snapshot.bb_upper,
                    bb_lower=snapshot.bb_lower,
                    atr=snapshot.atr,
                    momentum_5=snapshot.momentum_5,
                    vol_regime=snapshot.vol_regime,
                    vol_regime_confidence=snapshot.vol_regime_confidence,
                    sentiment_score=snapshot.sentiment_score,
                    combined_alpha=alpha,
                )

        return snapshot
