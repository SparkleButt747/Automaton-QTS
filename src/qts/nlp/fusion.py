"""Multi-source sentiment score fusion engine."""

from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Protocol, runtime_checkable

from qts.config import SentimentFusionWeights

logger = logging.getLogger(__name__)

# ── Protocol (for dependency injection) ──────────────────────────────────────


@runtime_checkable
class SentimentFusionProtocol(Protocol):
    """Dependency-injection interface for multi-source sentiment fusion."""

    def fuse(
        self,
        news_score: float,
        social_score: float,
        geopolitical_score: float,
    ) -> float:
        """Compute a weighted fusion of three sentiment scores.

        Args:
            news_score: News-derived sentiment in [-1, 1].
            social_score: Social-media sentiment in [-1, 1].
            geopolitical_score: Geopolitical sentiment in [-1, 1].

        Returns:
            Fused score clipped to [-1.0, 1.0].
        """
        ...

    def fuse_with_decay(
        self,
        scores_with_timestamps: list[tuple[float, float, float, datetime]],
    ) -> float:
        """Fuse scores after applying per-record exponential time decay.

        Args:
            scores_with_timestamps: List of (news_score, social_score,
                geopolitical_score, published_at) tuples. *published_at* is
                compared against the most recent timestamp in the list.

        Returns:
            Fused score after decay, clipped to [-1.0, 1.0].
        """
        ...


# ── Implementation ────────────────────────────────────────────────────────────


class SentimentFusion:
    """Combines news, social, and geopolitical sentiment scores.

    Weights are loaded from :class:`~qts.config.SentimentFusionWeights`.  The
    default configuration uses news=0.4, social=0.3, geopolitical=0.3.

    Attributes:
        weights: Immutable fusion weight configuration.
        half_life_hours: Half-life (in hours) used for exponential time decay
            in :meth:`fuse_with_decay`.
    """

    # Default weight values when no config object is supplied
    _DEFAULT_NEWS: float = 0.4
    _DEFAULT_SOCIAL: float = 0.3
    _DEFAULT_GEOPOLITICAL: float = 0.3

    def __init__(
        self,
        weights: SentimentFusionWeights | None = None,
        half_life_hours: float = 2.0,
    ) -> None:
        """Initialise the fusion engine.

        Args:
            weights: Optional :class:`~qts.config.SentimentFusionWeights` instance.
                If *None*, defaults to news=0.4, social=0.3, geopolitical=0.3.
            half_life_hours: Half-life for exponential decay (default 2.0 h).
        """
        if weights is None:
            self.weights = SentimentFusionWeights(
                news=self._DEFAULT_NEWS,
                social=self._DEFAULT_SOCIAL,
                geopolitical=self._DEFAULT_GEOPOLITICAL,
            )
        else:
            self.weights = weights

        self.half_life_hours: float = half_life_hours

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _decay_factor(self, t_published: datetime, t_now: datetime) -> float:
        """Compute exponential decay factor for a given age.

        Args:
            t_published: Publication timestamp.
            t_now: Current (reference) timestamp.

        Returns:
            Decay factor in (0, 1].  Returns 1.0 for non-positive elapsed time.
        """
        elapsed = (t_now - t_published).total_seconds()
        if elapsed <= 0.0:
            return 1.0
        return math.exp(-math.log(2) * elapsed / (self.half_life_hours * 3600.0))

    # ── Public API ────────────────────────────────────────────────────────────

    def fuse(
        self,
        news_score: float,
        social_score: float,
        geopolitical_score: float,
    ) -> float:
        """Compute a weighted fusion of three sentiment scores.

        All input scores are expected to be in the range [-1, 1], though
        values outside this range are clipped in the output.

        Formula::

            fused = w_news * news + w_social * social + w_geo * geo
            return clip(fused, -1.0, 1.0)

        Args:
            news_score: News-derived sentiment score in [-1, 1].
            social_score: Social-media sentiment score in [-1, 1].
            geopolitical_score: Geopolitical event score in [-1, 1].

        Returns:
            Fused sentiment score clipped to [-1.0, 1.0].
        """
        raw = (
            self.weights.news * news_score
            + self.weights.social * social_score
            + self.weights.geopolitical * geopolitical_score
        )
        return max(-1.0, min(1.0, raw))

    def fuse_with_decay(
        self,
        scores_with_timestamps: list[tuple[float, float, float, datetime]],
    ) -> float:
        """Fuse multiple score records after applying exponential time decay.

        Each record is a tuple of (news_score, social_score, geo_score, published_at).
        The most-recent timestamp in the list is used as *t_now* so that the
        freshest record has decay factor = 1.0 and older records have lower weight.

        If *scores_with_timestamps* is empty, returns 0.0.

        Args:
            scores_with_timestamps: List of (news, social, geo, published_at) tuples.

        Returns:
            Decay-weighted fused score, clipped to [-1.0, 1.0].
        """
        if not scores_with_timestamps:
            return 0.0

        # Reference time = most recent timestamp in the batch
        t_now = max(ts for _, _, _, ts in scores_with_timestamps)

        total_decay = 0.0
        weighted_sum = 0.0

        for news_s, social_s, geo_s, t_pub in scores_with_timestamps:
            decay = self._decay_factor(t_pub, t_now)
            fused = self.fuse(news_s, social_s, geo_s)
            weighted_sum += decay * fused
            total_decay += decay

        if total_decay == 0.0:
            return 0.0

        raw = weighted_sum / total_decay
        return max(-1.0, min(1.0, raw))
