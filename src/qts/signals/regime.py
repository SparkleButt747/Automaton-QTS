"""HMM-based volatility regime detection.

Uses a 2-state Gaussian Hidden Markov Model fitted on ATR history to
classify the current market into HIGH or LOW volatility regimes.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from qts.models.base import VolRegime

logger = logging.getLogger(__name__)


@runtime_checkable
class RegimeDetectorProtocol(Protocol):
    """Protocol for volatility regime detectors (for dependency injection)."""

    def fit(self, atr_history: NDArray[np.float64]) -> None:
        """Fit the detector on historical ATR data.

        Args:
            atr_history: 1-D array of ATR values (NaN values are excluded).
        """
        ...

    def predict(
        self, atr_history: NDArray[np.float64]
    ) -> tuple[VolRegime, float]:
        """Predict the current volatility regime.

        Args:
            atr_history: 1-D array of ATR values (NaN values are excluded).

        Returns:
            Tuple of (regime, confidence) where confidence is in [0, 1].
        """
        ...


class RegimeDetector:
    """2-state HMM volatility regime detector wrapping hmmlearn.GaussianHMM.

    Fits a Gaussian HMM with 2 hidden states on ATR series. The state with
    the higher mean ATR is labelled HIGH; the other is labelled LOW.

    After fitting, `predict` returns the most likely current state and the
    posterior probability of that state as confidence.

    Attributes:
        n_components: Number of HMM states (always 2).
        _model: Underlying hmmlearn GaussianHMM instance.
        _high_state: Integer index (0 or 1) of the HIGH-volatility state.
        _fitted: Whether the model has been trained.
    """

    def __init__(self, random_state: int = 42) -> None:
        """Initialise a RegimeDetector.

        Args:
            random_state: Seed for reproducible HMM initialisation.
        """
        # Import here to keep the optional dependency isolated
        from hmmlearn.hmm import GaussianHMM

        self.n_components: int = 2
        self._model = GaussianHMM(
            n_components=self.n_components,
            covariance_type="diag",
            n_iter=100,
            random_state=random_state,
            verbose=False,
        )
        self._high_state: int = 1  # default; recalculated after fit
        self._fitted: bool = False

    def fit(self, atr_history: NDArray[np.float64]) -> None:
        """Fit the 2-state HMM on the provided ATR series.

        NaN values are stripped before fitting. The HIGH state is determined
        as the state with the greater mean ATR emission.

        Args:
            atr_history: 1-D float64 array of ATR values.

        Raises:
            ValueError: If fewer than 10 valid observations are provided.
        """
        clean = atr_history[~np.isnan(atr_history)].reshape(-1, 1)
        if len(clean) < 10:
            raise ValueError(
                f"Need at least 10 valid ATR observations to fit HMM, got {len(clean)}."
            )
        self._model.fit(clean)
        # Identify which state corresponds to higher ATR (HIGH regime)
        means = self._model.means_.flatten()
        self._high_state = int(np.argmax(means))
        self._fitted = True
        logger.debug(
            "RegimeDetector fitted: high_state=%d, means=%s",
            self._high_state,
            means,
        )

    def predict(
        self, atr_history: NDArray[np.float64]
    ) -> tuple[VolRegime, float]:
        """Predict the current volatility regime and confidence.

        Runs the Viterbi algorithm on the clean ATR history and reads the
        final state. Confidence is the posterior probability of that state
        at the last time-step.

        Args:
            atr_history: 1-D float64 array of ATR values (NaN stripped internally).

        Returns:
            Tuple (regime, confidence) where regime is VolRegime.HIGH or LOW
            and confidence is a float in [0.0, 1.0].

        Raises:
            RuntimeError: If the detector has not been fitted yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "RegimeDetector.predict() called before fit(). Call fit() first."
            )
        clean = atr_history[~np.isnan(atr_history)].reshape(-1, 1)
        if len(clean) == 0:
            # No data: fall back to LOW with low confidence
            return VolRegime.LOW, 0.0

        # Viterbi path for the most likely state sequence
        _logprob, state_sequence = self._model.decode(clean, algorithm="viterbi")
        current_state = int(state_sequence[-1])

        # Posterior probability at the last observation
        posteriors = self._model.predict_proba(clean)
        confidence = float(posteriors[-1, current_state])

        regime = VolRegime.HIGH if current_state == self._high_state else VolRegime.LOW
        logger.debug(
            "RegimeDetector predict: state=%d, regime=%s, confidence=%.3f",
            current_state,
            regime,
            confidence,
        )
        return regime, confidence
