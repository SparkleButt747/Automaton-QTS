"""Multi-axis structured output from the news classifier.

The decode-gap alpha hypothesis: Qwen reads news smarter than VADER-grade
crowd sentiment by producing a structured belief over multiple axes, not
a single scalar. The strategy combines these axes into an alpha bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

_VALID_DIRECTIONS: tuple[str, ...] = ("bull", "bear", "neutral")


@dataclass(frozen=True, slots=True)
class NewsSignal:
    """Structured news read.

    direction:  classification of price-impact direction
    confidence: how confident the classifier is in the direction call
    relevance:  how relevant this text is to the asset under analysis
    magnitude:  how large a move this implies
    """

    direction: Literal["bull", "bear", "neutral"]
    confidence: float
    relevance: float
    magnitude: float

    def __post_init__(self) -> None:
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {_VALID_DIRECTIONS}, got {self.direction!r}"
            )
        for field_name, value in (
            ("confidence", self.confidence),
            ("relevance", self.relevance),
            ("magnitude", self.magnitude),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1], got {value}")

    @property
    def direction_sign(self) -> float:
        """+1 for bull, -1 for bear, 0 for neutral."""
        if self.direction == "bull":
            return 1.0
        if self.direction == "bear":
            return -1.0
        return 0.0

    def alpha_contribution(self) -> float:
        """Combined scalar bias for the strategy's alpha blend.

        Returns direction_sign × confidence × relevance × magnitude, which
        is bounded in [-1, 1]. Used as one input to a weighted alpha blend
        in NewsReactiveMomentum.
        """
        return self.direction_sign * self.confidence * self.relevance * self.magnitude
