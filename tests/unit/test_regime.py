"""Unit tests for qts.signals.regime.

Verifies:
- RegimeDetector always outputs VolLevel.HIGH or VolLevel.LOW
- Confidence is always in [0, 1]
- Fit/predict lifecycle works correctly
- Edge cases (insufficient data, unfitted model)
"""

from __future__ import annotations

import numpy as np
import pytest

from qts.models.base import VolLevel
from qts.signals.regime import RegimeDetector, RegimeDetectorProtocol


class TestRegimeDetectorProtocol:
    def test_detector_implements_protocol(self) -> None:
        detector = RegimeDetector()
        assert isinstance(detector, RegimeDetectorProtocol)


class TestRegimeDetectorFit:
    def test_fit_requires_at_least_10_observations(self) -> None:
        detector = RegimeDetector()
        with pytest.raises(ValueError, match="at least 10"):
            detector.fit(np.array([1.0, 2.0, 3.0], dtype=np.float64))

    def test_fit_strips_nan(self) -> None:
        detector = RegimeDetector()
        atr = np.array([np.nan] * 5 + list(range(1, 21)), dtype=np.float64)
        # Should succeed (20 valid observations)
        detector.fit(atr)
        assert detector._fitted

    def test_fit_sets_fitted_flag(self) -> None:
        detector = RegimeDetector()
        atr = np.random.default_rng(42).uniform(0.5, 3.0, 100)
        detector.fit(atr.astype(np.float64))
        assert detector._fitted

    def test_fit_identifies_high_state(self) -> None:
        """High-volatility state should have higher mean ATR."""
        detector = RegimeDetector(random_state=42)
        # Create two clear clusters
        low_atr = np.random.default_rng(0).uniform(0.5, 1.0, 50)
        high_atr = np.random.default_rng(1).uniform(3.0, 5.0, 50)
        atr = np.concatenate([low_atr, high_atr]).astype(np.float64)
        detector.fit(atr)
        # High state should have higher mean
        means = detector._model.means_.flatten()
        assert means[detector._high_state] == max(means)


class TestRegimeDetectorPredict:
    def test_predict_before_fit_raises(self) -> None:
        detector = RegimeDetector()
        with pytest.raises(RuntimeError, match="fit()"):
            detector.predict(np.array([1.0, 2.0, 3.0], dtype=np.float64))

    def test_predict_returns_vol_regime(self) -> None:
        rng = np.random.default_rng(42)
        atr = rng.uniform(0.5, 3.0, 100).astype(np.float64)
        detector = RegimeDetector(random_state=42)
        detector.fit(atr)
        regime, confidence = detector.predict(atr)
        assert isinstance(regime, VolLevel)
        assert regime in (VolLevel.HIGH, VolLevel.LOW)

    def test_confidence_in_unit_interval(self) -> None:
        rng = np.random.default_rng(42)
        atr = rng.uniform(0.5, 3.0, 100).astype(np.float64)
        detector = RegimeDetector(random_state=42)
        detector.fit(atr)
        for _ in range(10):
            sub = atr[: rng.integers(15, 100)]
            _, confidence = detector.predict(sub)
            assert 0.0 <= confidence <= 1.0

    def test_high_atr_predicts_high_regime(self) -> None:
        """Consistently high ATR should predict HIGH regime."""
        rng = np.random.default_rng(42)
        # Training: clear bimodal
        low_atr = rng.uniform(0.5, 1.0, 60).astype(np.float64)
        high_atr = rng.uniform(4.0, 6.0, 60).astype(np.float64)
        train = np.concatenate([low_atr, high_atr])
        detector = RegimeDetector(random_state=42)
        detector.fit(train)

        # Prediction on purely high ATR
        test_high = rng.uniform(4.0, 6.0, 30).astype(np.float64)
        regime, _ = detector.predict(test_high)
        assert regime == VolLevel.HIGH

    def test_low_atr_predicts_low_regime(self) -> None:
        """Consistently low ATR should predict LOW regime."""
        rng = np.random.default_rng(42)
        low_atr = rng.uniform(0.5, 1.0, 60).astype(np.float64)
        high_atr = rng.uniform(4.0, 6.0, 60).astype(np.float64)
        train = np.concatenate([low_atr, high_atr])
        detector = RegimeDetector(random_state=42)
        detector.fit(train)

        test_low = rng.uniform(0.5, 1.0, 30).astype(np.float64)
        regime, _ = detector.predict(test_low)
        assert regime == VolLevel.LOW

    def test_empty_atr_returns_low_zero_confidence(self) -> None:
        rng = np.random.default_rng(42)
        atr = rng.uniform(0.5, 3.0, 100).astype(np.float64)
        detector = RegimeDetector(random_state=42)
        detector.fit(atr)
        regime, confidence = detector.predict(np.array([], dtype=np.float64))
        assert regime == VolLevel.LOW
        assert confidence == 0.0

    def test_predict_with_nan_prefixed_atr(self) -> None:
        """NaN values at the start (warm-up) should be stripped."""
        rng = np.random.default_rng(42)
        atr = rng.uniform(0.5, 3.0, 100).astype(np.float64)
        detector = RegimeDetector(random_state=42)
        detector.fit(atr)
        atr_with_nan = np.concatenate([[np.nan] * 14, atr])
        regime, confidence = detector.predict(atr_with_nan)
        assert regime in (VolLevel.HIGH, VolLevel.LOW)
        assert 0.0 <= confidence <= 1.0
