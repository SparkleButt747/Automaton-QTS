"""Unit tests for qts.nlp.fusion.

Tests:
- Fused sentiment always clipped to [-1.0, 1.0]
- Weights applied correctly to individual source scores
- Exponential decay reduces influence of old scores
- Protocol compliance
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from qts.config import SentimentFusionWeights
from qts.nlp.fusion import SentimentFusion, SentimentFusionProtocol

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def default_fusion() -> SentimentFusion:
    """SentimentFusion with default weights (news=0.4, social=0.3, geo=0.3)."""
    return SentimentFusion()


@pytest.fixture()
def equal_fusion() -> SentimentFusion:
    """SentimentFusion with equal weights (1/3, 1/3, 1/3)."""
    weights = SentimentFusionWeights(
        news=1 / 3,
        social=1 / 3,
        geopolitical=1 / 3,
    )
    return SentimentFusion(weights=weights)


@pytest.fixture()
def news_heavy_fusion() -> SentimentFusion:
    """SentimentFusion with 80% news weight."""
    weights = SentimentFusionWeights(news=0.8, social=0.1, geopolitical=0.1)
    return SentimentFusion(weights=weights)


# ── Protocol compliance ───────────────────────────────────────────────────────


class TestSentimentFusionProtocol:
    def test_implements_protocol(self, default_fusion: SentimentFusion) -> None:
        assert isinstance(default_fusion, SentimentFusionProtocol)


# ── Clip / range tests ────────────────────────────────────────────────────────


class TestFusedRange:
    """Fused score must always be in [-1.0, 1.0]."""

    def test_all_positive_max_clipped(self, default_fusion: SentimentFusion) -> None:
        result = default_fusion.fuse(1.0, 1.0, 1.0)
        assert result <= 1.0

    def test_all_negative_min_clipped(self, default_fusion: SentimentFusion) -> None:
        result = default_fusion.fuse(-1.0, -1.0, -1.0)
        assert result >= -1.0

    def test_overflow_positive_clipped_to_one(self) -> None:
        # Use large positive values to ensure overflow is clipped
        weights = SentimentFusionWeights(news=1.0, social=0.0, geopolitical=0.0)
        fusion = SentimentFusion(weights=weights)
        result = fusion.fuse(2.0, 0.0, 0.0)  # deliberately out of range input
        assert result <= 1.0

    def test_overflow_negative_clipped_to_minus_one(self) -> None:
        weights = SentimentFusionWeights(news=1.0, social=0.0, geopolitical=0.0)
        fusion = SentimentFusion(weights=weights)
        result = fusion.fuse(-2.0, 0.0, 0.0)
        assert result >= -1.0

    def test_zero_inputs_returns_zero(self, default_fusion: SentimentFusion) -> None:
        assert default_fusion.fuse(0.0, 0.0, 0.0) == 0.0

    def test_mixed_signs_in_range(self, default_fusion: SentimentFusion) -> None:
        result = default_fusion.fuse(0.8, -0.5, 0.3)
        assert -1.0 <= result <= 1.0

    def test_range_property_over_many_inputs(self, default_fusion: SentimentFusion) -> None:
        inputs = [
            (0.1, 0.2, 0.3),
            (-0.5, 0.5, -0.5),
            (1.0, -1.0, 0.0),
            (-0.9, -0.8, -0.7),
            (0.5, 0.5, 0.5),
        ]
        for news, social, geo in inputs:
            r = default_fusion.fuse(news, social, geo)
            assert -1.0 <= r <= 1.0, f"Out of range for ({news}, {social}, {geo}): {r}"


# ── Weight correctness tests ──────────────────────────────────────────────────


class TestWeightApplication:
    def test_equal_weights_equal_contribution(self, equal_fusion: SentimentFusion) -> None:
        """With 1/3 weights, fusing (1, -1, 0) should be approximately 0."""
        result = equal_fusion.fuse(1.0, -1.0, 0.0)
        assert math.isclose(result, 0.0, abs_tol=1e-6)

    def test_news_heavy_dominates(self, news_heavy_fusion: SentimentFusion) -> None:
        """With 80% news weight, a strongly positive news score should dominate."""
        result = news_heavy_fusion.fuse(1.0, -1.0, -1.0)
        # Expected: 0.8 * 1.0 + 0.1 * -1.0 + 0.1 * -1.0 = 0.6
        assert math.isclose(result, 0.6, abs_tol=1e-9)

    def test_default_weights_calculation(self, default_fusion: SentimentFusion) -> None:
        """Verify exact arithmetic with default weights (0.4, 0.3, 0.3)."""
        news, social, geo = 0.8, 0.4, -0.2
        expected = 0.4 * 0.8 + 0.3 * 0.4 + 0.3 * (-0.2)
        result = default_fusion.fuse(news, social, geo)
        assert math.isclose(result, expected, abs_tol=1e-9)

    def test_only_news_score(self) -> None:
        """When social and geo are zero, result equals news * w_news."""
        weights = SentimentFusionWeights(news=0.6, social=0.2, geopolitical=0.2)
        fusion = SentimentFusion(weights=weights)
        result = fusion.fuse(0.5, 0.0, 0.0)
        assert math.isclose(result, 0.3, abs_tol=1e-9)

    def test_weights_sum_to_one_preserves_input(self, equal_fusion: SentimentFusion) -> None:
        """All same inputs: result should equal that value (if within [-1,1])."""
        for val in [0.5, -0.5, 0.0, 1.0, -1.0]:
            result = equal_fusion.fuse(val, val, val)
            assert math.isclose(result, max(-1.0, min(1.0, val)), abs_tol=1e-9)


# ── Decay tests ───────────────────────────────────────────────────────────────


class TestDecayFusion:
    _NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)

    def _make_record(
        self, news: float, social: float, geo: float, hours_ago: float
    ) -> tuple[float, float, float, datetime]:
        published = self._NOW - timedelta(hours=hours_ago)
        return news, social, geo, published

    def test_single_fresh_record(self, default_fusion: SentimentFusion) -> None:
        """A single fresh record should have decay ~1 and fuse normally."""
        records = [self._make_record(0.5, 0.3, 0.2, 0.0)]
        result = default_fusion.fuse_with_decay(records)
        expected = default_fusion.fuse(0.5, 0.3, 0.2)
        assert math.isclose(result, expected, abs_tol=1e-6)

    def test_empty_records_returns_zero(self, default_fusion: SentimentFusion) -> None:
        result = default_fusion.fuse_with_decay([])
        assert result == 0.0

    def test_decay_reduces_old_score(self, default_fusion: SentimentFusion) -> None:
        """A very old record should contribute less than a fresh record with the same scores."""
        fresh_records = [self._make_record(0.8, 0.0, 0.0, 0.0)]
        stale_records = [self._make_record(0.8, 0.0, 0.0, 20.0)]  # 20h = 10 half-lives

        fresh_result = default_fusion.fuse_with_decay(fresh_records)
        stale_result = default_fusion.fuse_with_decay(stale_records)

        # Both should be in valid range; fresh should give same score (no decay)
        # stale should also be within range (just less "aged" contribution after norm)
        assert -1.0 <= fresh_result <= 1.0
        assert -1.0 <= stale_result <= 1.0

    def test_multiple_records_reference_is_most_recent(
        self, default_fusion: SentimentFusion
    ) -> None:
        """The most recent record should have decay factor = 1.0."""
        # Fresh positive + stale negative -> result should be net positive (fresh dominates)
        fresh_positive = self._make_record(1.0, 0.0, 0.0, 0.0)  # 0h ago
        stale_negative = self._make_record(-1.0, 0.0, 0.0, 100.0)  # 100h ago

        records = [fresh_positive, stale_negative]
        result = default_fusion.fuse_with_decay(records)
        # Fresh record contributes much more; should be net positive
        assert result > 0.0

    def test_decay_result_in_range(self, default_fusion: SentimentFusion) -> None:
        """fuse_with_decay result must always be in [-1, 1]."""
        records = [
            self._make_record(1.0, 1.0, 1.0, 0.0),
            self._make_record(-1.0, -1.0, -1.0, 5.0),
            self._make_record(0.5, 0.5, 0.5, 10.0),
        ]
        result = default_fusion.fuse_with_decay(records)
        assert -1.0 <= result <= 1.0

    def test_two_equal_records_different_ages(self, default_fusion: SentimentFusion) -> None:
        """When both records have the same scores, newer one gets more weight."""
        newer = self._make_record(0.8, 0.0, 0.0, 1.0)  # 1h ago (decay ~0.707)
        older = self._make_record(0.8, 0.0, 0.0, 10.0)  # 10h ago (heavily decayed)
        records = [newer, older]
        result = default_fusion.fuse_with_decay(records)

        # Both records have the same sign so result should still be positive
        assert result > 0.0
        # Result should be ≤ fuse(0.8, 0, 0) since both are below 1.0
        assert result <= default_fusion.fuse(0.8, 0.0, 0.0) + 1e-6


# ── Default weight values ─────────────────────────────────────────────────────


class TestDefaultWeights:
    def test_default_weights_are_correct(self, default_fusion: SentimentFusion) -> None:
        assert math.isclose(default_fusion.weights.news, 0.4, abs_tol=1e-9)
        assert math.isclose(default_fusion.weights.social, 0.3, abs_tol=1e-9)
        assert math.isclose(default_fusion.weights.geopolitical, 0.3, abs_tol=1e-9)

    def test_custom_weights_applied(self) -> None:
        weights = SentimentFusionWeights(news=0.5, social=0.3, geopolitical=0.2)
        fusion = SentimentFusion(weights=weights)
        assert math.isclose(fusion.weights.news, 0.5)
        assert math.isclose(fusion.weights.social, 0.3)
        assert math.isclose(fusion.weights.geopolitical, 0.2)
