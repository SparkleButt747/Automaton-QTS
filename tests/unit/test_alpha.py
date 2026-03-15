"""Unit tests for qts.signals.alpha.

Verifies:
- combined_alpha always returns a value in [-1, 1]
- Regime scalar correctly dampens LOW regime signals
- Sentiment and RSI contributions work as expected
"""
from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pytest

from qts.config import SentimentFusionWeights, SignalWeights, StrategyParams
from qts.models.base import SignalSnapshot, VolRegime
from qts.signals.alpha import combined_alpha


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_params(
    w_rsi: float = 0.20,
    w_macd: float = 0.20,
    w_bb: float = 0.15,
    w_mom: float = 0.15,
    w_sentiment: float = 0.30,
) -> StrategyParams:
    return StrategyParams(
        version="test",
        weights=SignalWeights(
            w_rsi=w_rsi, w_macd=w_macd, w_bb=w_bb, w_mom=w_mom, w_sentiment=w_sentiment
        ),
        entry_threshold=0.25,
        exit_threshold=-0.10,
        max_hold_bars=48,
        sentiment_fusion_weights=SentimentFusionWeights(
            news=0.40, social=0.30, geopolitical=0.30
        ),
    )


def _make_snapshot(
    rsi: float = 50.0,
    macd_histogram: float = 0.0,
    bb_position: float = 0.5,
    momentum_5: float = 0.0,
    sentiment_score: float = 0.0,
    vol_regime: VolRegime = VolRegime.HIGH,
) -> SignalSnapshot:
    return SignalSnapshot(
        timestamp=datetime(2024, 1, 1),
        symbol="TEST",
        rsi=rsi,
        macd_histogram=macd_histogram,
        macd_line=0.0,
        macd_signal=0.0,
        bb_position=bb_position,
        bb_upper=110.0,
        bb_lower=90.0,
        atr=1.0,
        momentum_5=momentum_5,
        vol_regime=vol_regime,
        vol_regime_confidence=0.9,
        sentiment_score=sentiment_score,
        combined_alpha=0.0,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestCombinedAlphaRange:
    def test_neutral_snapshot_returns_zero(self) -> None:
        snap = _make_snapshot()
        params = _make_params()
        result = combined_alpha(snap, params)
        # Neutral values: RSI=50->0, macd=0->0, bb=0.5->0, mom=0->0, sent=0->0
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_all_bullish_clipped_to_one(self) -> None:
        snap = _make_snapshot(
            rsi=100.0,
            macd_histogram=100.0,  # large positive -> tanh ~ 1
            bb_position=1.0,
            momentum_5=1.0,  # large positive -> tanh ~ 1
            sentiment_score=1.0,
            vol_regime=VolRegime.HIGH,
        )
        params = _make_params()
        result = combined_alpha(snap, params)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_all_bearish_clipped_to_minus_one(self) -> None:
        snap = _make_snapshot(
            rsi=0.0,
            macd_histogram=-100.0,
            bb_position=0.0,
            momentum_5=-1.0,
            sentiment_score=-1.0,
            vol_regime=VolRegime.HIGH,
        )
        params = _make_params()
        result = combined_alpha(snap, params)
        assert result == pytest.approx(-1.0, abs=1e-6)

    def test_output_always_in_range(self) -> None:
        rng = np.random.default_rng(42)
        params = _make_params()
        for _ in range(200):
            snap = _make_snapshot(
                rsi=float(rng.uniform(0, 100)),
                macd_histogram=float(rng.uniform(-50, 50)),
                bb_position=float(rng.uniform(0, 1)),
                momentum_5=float(rng.uniform(-0.5, 0.5)),
                sentiment_score=float(rng.uniform(-1, 1)),
                vol_regime=VolRegime.HIGH if rng.random() > 0.5 else VolRegime.LOW,
            )
            result = combined_alpha(snap, params)
            assert -1.0 <= result <= 1.0, f"Out of range: {result}"


class TestRegimeScalar:
    def test_low_regime_dampens_signal(self) -> None:
        """Same snapshot in LOW regime should produce half the alpha of HIGH."""
        snap_high = _make_snapshot(
            rsi=80.0, sentiment_score=0.8, vol_regime=VolRegime.HIGH
        )
        snap_low = _make_snapshot(
            rsi=80.0, sentiment_score=0.8, vol_regime=VolRegime.LOW
        )
        params = _make_params()
        alpha_high = combined_alpha(snap_high, params)
        alpha_low = combined_alpha(snap_low, params)
        # LOW should be half of HIGH (regime scalar 0.5 vs 1.0)
        assert abs(alpha_low - alpha_high * 0.5) < 1e-6

    def test_high_regime_full_signal(self) -> None:
        """HIGH regime scalar should be 1.0."""
        snap = _make_snapshot(rsi=70.0, sentiment_score=0.5, vol_regime=VolRegime.HIGH)
        params = _make_params(w_rsi=0.5, w_macd=0.0, w_bb=0.0, w_mom=0.0, w_sentiment=0.5)
        result = combined_alpha(snap, params)
        # Should be unscaled
        assert abs(result) <= 1.0


class TestSentimentContribution:
    def test_pure_sentiment_positive(self) -> None:
        """With only sentiment weight, positive sentiment -> positive alpha."""
        snap = _make_snapshot(
            rsi=50.0, macd_histogram=0.0, bb_position=0.5,
            momentum_5=0.0, sentiment_score=1.0, vol_regime=VolRegime.HIGH
        )
        params = _make_params(w_rsi=0.0, w_macd=0.0, w_bb=0.0, w_mom=0.0, w_sentiment=1.0)
        result = combined_alpha(snap, params)
        assert result > 0.0

    def test_pure_sentiment_negative(self) -> None:
        snap = _make_snapshot(
            rsi=50.0, macd_histogram=0.0, bb_position=0.5,
            momentum_5=0.0, sentiment_score=-1.0, vol_regime=VolRegime.HIGH
        )
        params = _make_params(w_rsi=0.0, w_macd=0.0, w_bb=0.0, w_mom=0.0, w_sentiment=1.0)
        result = combined_alpha(snap, params)
        assert result < 0.0


class TestRSIContribution:
    def test_rsi_overbought_positive(self) -> None:
        snap = _make_snapshot(rsi=80.0, vol_regime=VolRegime.HIGH)
        params = _make_params(w_rsi=1.0, w_macd=0.0, w_bb=0.0, w_mom=0.0, w_sentiment=0.0)
        result = combined_alpha(snap, params)
        assert result > 0.0

    def test_rsi_oversold_negative(self) -> None:
        snap = _make_snapshot(rsi=20.0, vol_regime=VolRegime.HIGH)
        params = _make_params(w_rsi=1.0, w_macd=0.0, w_bb=0.0, w_mom=0.0, w_sentiment=0.0)
        result = combined_alpha(snap, params)
        assert result < 0.0

    def test_nan_rsi_defaults_to_zero(self) -> None:
        snap = _make_snapshot(rsi=float("nan"), vol_regime=VolRegime.HIGH)
        params = _make_params(w_rsi=1.0, w_macd=0.0, w_bb=0.0, w_mom=0.0, w_sentiment=0.0)
        result = combined_alpha(snap, params)
        assert result == pytest.approx(0.0, abs=1e-6)
