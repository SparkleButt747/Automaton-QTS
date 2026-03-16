"""FinBERT sentiment analysis for financial headlines."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ── Domain types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SentimentResult:
    """Sentiment analysis output for a single piece of text.

    Attributes:
        text: The input text that was analysed.
        label: Predicted class: POSITIVE, NEGATIVE, or NEUTRAL.
        score: Raw model score for the predicted label (in [0, 1]).
        confidence: Highest softmax probability across all classes (in [0, 1]).
    """

    text: str
    label: str  # "POSITIVE" | "NEGATIVE" | "NEUTRAL"
    score: float
    confidence: float


# ── Protocol (for dependency injection) ──────────────────────────────────────


@runtime_checkable
class FinBERTProtocol(Protocol):
    """Dependency-injection interface for FinBERT-style sentiment analysers."""

    def analyze(self, texts: list[str]) -> list[SentimentResult]:
        """Batch-analyse a list of texts and return one result per input.

        Args:
            texts: List of text strings to classify.

        Returns:
            List of SentimentResult objects, same length as *texts*.
        """
        ...

    def decayed_score(
        self,
        score: float,
        t_published: datetime,
        t_now: datetime,
        half_life_hours: float = 2.0,
    ) -> float:
        """Apply exponential time-decay to a sentiment score.

        Args:
            score: Original sentiment score.
            t_published: UTC datetime when the content was published.
            t_now: Current UTC datetime.
            half_life_hours: Score half-life in hours (default 2.0).

        Returns:
            Decayed score = score * exp(-ln(2) * elapsed_seconds / (half_life * 3600)).
        """
        ...


# ── Implementation ────────────────────────────────────────────────────────────


class FinBERTAnalyzer:
    """FinBERT-based sentiment classifier for financial text.

    The transformer model is loaded lazily on the first call to :meth:`analyze`
    to avoid importing heavy dependencies at import time and to allow the class
    to be instantiated even when ``transformers`` / ``torch`` are not installed.

    If the required packages are unavailable, every call to :meth:`analyze`
    returns NEUTRAL results with zero score and a warning is emitted.

    Attributes:
        model_name: HuggingFace model hub identifier.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert") -> None:
        """Initialise the analyzer.

        Args:
            model_name: HuggingFace model identifier to load (default ``ProsusAI/finbert``).
        """
        self.model_name: str = model_name
        self._pipeline: object | None = None
        self._available: bool | None = None  # tri-state: None = not yet checked

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_loaded(self) -> bool:
        """Attempt to load the transformers pipeline if not already loaded.

        Returns:
            True if the model is loaded and ready, False if dependencies are absent.
        """
        if self._available is not None:
            return self._available

        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                return_all_scores=True,
            )
            self._available = True
            logger.info("FinBERTAnalyzer: loaded model %s", self.model_name)
        except ImportError:
            logger.warning(
                "FinBERTAnalyzer: 'transformers' or 'torch' not installed. "
                "All sentiments will be NEUTRAL with score=0. "
                "Install with: pip install transformers torch"
            )
            self._available = False

        return bool(self._available)

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, texts: list[str]) -> list[SentimentResult]:
        """Run batch sentiment inference on a list of texts.

        Labels are normalised to uppercase: POSITIVE, NEGATIVE, or NEUTRAL.
        When the model is unavailable, all results are NEUTRAL with score=0
        and confidence=0.

        Args:
            texts: List of financial text strings to classify.

        Returns:
            List of :class:`SentimentResult` objects (one per input text).
        """
        if not texts:
            return []

        if not self._ensure_loaded() or self._pipeline is None:
            return [
                SentimentResult(text=t, label="NEUTRAL", score=0.0, confidence=0.0) for t in texts
            ]

        # The pipeline returns a list-of-lists when return_all_scores=True
        raw_outputs: list[list[dict[str, object]]] = self._pipeline(texts)  # type: ignore[operator]

        results: list[SentimentResult] = []
        for text, scores in zip(texts, raw_outputs, strict=True):
            # Find the label with the highest score
            best = max(scores, key=lambda s: float(s["score"]))  # type: ignore[arg-type]
            label = str(best["label"]).upper()
            score = float(best["score"])  # type: ignore[arg-type]
            confidence = score  # highest softmax probability

            results.append(
                SentimentResult(
                    text=text,
                    label=label,
                    score=score,
                    confidence=confidence,
                )
            )

        return results

    def decayed_score(
        self,
        score: float,
        t_published: datetime,
        t_now: datetime,
        half_life_hours: float = 2.0,
    ) -> float:
        """Apply exponential time-decay to a sentiment score.

        Formula::

            score * exp(-ln(2) * (t_now - t_published).total_seconds()
                        / (half_life_hours * 3600))

        Args:
            score: Original (undecayed) sentiment score.
            t_published: UTC datetime when the content was published.
            t_now: Current UTC datetime.
            half_life_hours: Half-life in hours (default 2.0).

        Returns:
            Decay-adjusted score.  Returns *score* unchanged if elapsed time
            is negative (i.e. *t_now* < *t_published*).
        """
        elapsed_seconds = (t_now - t_published).total_seconds()
        if elapsed_seconds <= 0.0:
            return score
        decay = math.exp(-math.log(2) * elapsed_seconds / (half_life_hours * 3600.0))
        return score * decay
