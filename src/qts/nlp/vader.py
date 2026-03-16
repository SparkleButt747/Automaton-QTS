"""VADER sentiment analysis for social media text."""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from qts.nlp.finbert import SentimentResult

logger = logging.getLogger(__name__)

# ── Protocol (for dependency injection) ──────────────────────────────────────


@runtime_checkable
class VaderProtocol(Protocol):
    """Dependency-injection interface for VADER-style sentiment analysers."""

    def analyze(self, texts: list[str]) -> list[SentimentResult]:
        """Batch-analyse a list of texts and return one result per input.

        Args:
            texts: List of text strings to classify.

        Returns:
            List of SentimentResult objects, same length as *texts*.
        """
        ...


# ── Implementation ────────────────────────────────────────────────────────────


class VaderAnalyzer:
    """VADER-based sentiment analyser optimised for short social media text.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is a rule-based
    sentiment analysis tool that works well for tweets, Reddit posts, and other
    short informal text without requiring model training.

    The ``vaderSentiment`` package is imported lazily. If it is not installed,
    all calls to :meth:`analyze` return NEUTRAL results with score=0 and a
    warning is logged.

    The raw VADER compound score (range [-1, 1]) is mapped to a label:
    - compound >= 0.05  -> POSITIVE
    - compound <= -0.05 -> NEGATIVE
    - otherwise         -> NEUTRAL

    The ``score`` field of :class:`~qts.nlp.finbert.SentimentResult` is the
    absolute value of the compound score (in [0, 1]).  ``confidence`` is the
    same value (VADER does not produce multi-class probabilities).
    """

    _POSITIVE_THRESHOLD: float = 0.05
    _NEGATIVE_THRESHOLD: float = -0.05

    def __init__(self) -> None:
        """Initialise the VADER analyser (lazy import of vaderSentiment)."""
        self._sia: object | None = None
        self._available: bool | None = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_loaded(self) -> bool:
        """Load the SentimentIntensityAnalyzer if not already initialised.

        Returns:
            True if VADER is ready, False if the package is missing.
        """
        if self._available is not None:
            return self._available

        try:
            from vaderSentiment.vaderSentiment import (
                SentimentIntensityAnalyzer,
            )

            self._sia = SentimentIntensityAnalyzer()
            self._available = True
            logger.info("VaderAnalyzer: loaded SentimentIntensityAnalyzer")
        except ImportError:
            logger.warning(
                "VaderAnalyzer: 'vaderSentiment' not installed. "
                "All sentiments will be NEUTRAL with score=0. "
                "Install with: pip install vaderSentiment"
            )
            self._available = False

        return bool(self._available)

    def _polarity_scores(self, text: str) -> dict[str, float]:
        """Compute VADER polarity scores for a single text.

        Args:
            text: Input text string.

        Returns:
            Dict with keys 'neg', 'neu', 'pos', 'compound'.
        """
        return self._sia.polarity_scores(text)  # type: ignore[union-attr, no-any-return]

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, texts: list[str]) -> list[SentimentResult]:
        """Run VADER sentiment analysis on a list of texts.

        Args:
            texts: List of social-media or news text strings.

        Returns:
            List of :class:`~qts.nlp.finbert.SentimentResult` objects (one per
            input text). If ``vaderSentiment`` is not installed, every result
            has label ``NEUTRAL``, score ``0.0``, and confidence ``0.0``.
        """
        if not texts:
            return []

        if not self._ensure_loaded() or self._sia is None:
            return [
                SentimentResult(text=t, label="NEUTRAL", score=0.0, confidence=0.0)
                for t in texts
            ]

        results: list[SentimentResult] = []
        for text in texts:
            scores = self._polarity_scores(text)
            compound: float = scores["compound"]

            if compound >= self._POSITIVE_THRESHOLD:
                label = "POSITIVE"
            elif compound <= self._NEGATIVE_THRESHOLD:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"

            abs_score = abs(compound)

            results.append(
                SentimentResult(
                    text=text,
                    label=label,
                    score=abs_score,
                    confidence=abs_score,
                )
            )

        return results
