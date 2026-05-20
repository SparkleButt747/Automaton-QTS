"""Crowd-grade sentiment scoring for anon retail agents.

Intentionally cheap and noisy: VADER compound score plus a small
keyword-regex bump for FOMC-specific vocabulary the strategy could
spot but the crowd routinely misreads. The asymmetry between this
scorer and the strategy's Qwen-backed regime classifier is the alpha
hypothesis under test (see grill log).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Hawkish vocabulary subtracts; dovish adds.
# `disinflation` is intentionally NOT in DOVISH — it appears in both bullish
# ("disinflation underway") and bearish ("disinflation has slowed") contexts;
# leaving it neutral lets the surrounding vocabulary decide.
_HAWKISH = re.compile(
    r"\b("
    r"hawkish|hike|hikes|hiking|"
    r"tighten\w*|restrict\w*|stubborn\w*|persistent\w*|sticky?|"
    r"firm(?:er|ing|s)|warrant\w*|restore|stabilit\w*|"
    r"raise(?:s|d)?|raising|further|additional"
    r")",
    re.IGNORECASE,
)
_DOVISH = re.compile(
    r"\b(dovish|cut|cuts|cutting|easing|ease|progress|peak|paused?|patient|cooling|cooler|sustainably)",
    re.IGNORECASE,
)


@dataclass
class SentimentScorer:
    """VADER + keyword regex. Returns score in [-1, 1]."""

    keyword_weight: float = 0.15
    _analyzer: object | None = None

    def _ensure_vader(self) -> None:
        if self._analyzer is None:
            from vaderSentiment.vaderSentiment import (  # noqa: PLC0415
                SentimentIntensityAnalyzer,
            )

            self._analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        if not text:
            return 0.0
        self._ensure_vader()
        assert self._analyzer is not None
        # vader returns a dict with compound in [-1, 1]
        base = float(self._analyzer.polarity_scores(text)["compound"])

        keyword_bump = 0.0
        keyword_bump += self.keyword_weight * len(_DOVISH.findall(text))
        keyword_bump -= self.keyword_weight * len(_HAWKISH.findall(text))

        return max(-1.0, min(1.0, base + keyword_bump))
