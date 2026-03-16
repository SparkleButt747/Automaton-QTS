"""Property-based tests for sentiment fusion."""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from qts.config import SentimentFusionWeights
from qts.nlp.fusion import SentimentFusion

# ── Strategies ────────────────────────────────────────────────────────────────

_SCORE_STRATEGY = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _fusion_weights_strategy() -> st.SearchStrategy:  # type: ignore[type-arg]
    """Generate valid SentimentFusionWeights (summing to 1.0)."""
    def build_weights(raw: list[float]) -> SentimentFusionWeights:
        total = sum(raw)
        if total < 1e-9:
            raw = [1.0 / 3, 1.0 / 3, 1.0 / 3]
            total = 1.0
        normed = [v / total for v in raw]
        # Clamp to 0.0 to prevent tiny negative values from floating-point rounding errors.
        normed[-1] = max(0.0, 1.0 - sum(normed[:-1]))
        return SentimentFusionWeights(
            news=normed[0],
            social=normed[1],
            geopolitical=normed[2],
        )

    return st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=3,
        max_size=3,
    ).map(build_weights)


def _fusion_strategy() -> st.SearchStrategy:  # type: ignore[type-arg]
    """Generate SentimentFusion with random valid weights."""
    return _fusion_weights_strategy().map(lambda w: SentimentFusion(weights=w))


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSentimentFusionProperty:
    @given(
        news=_SCORE_STRATEGY,
        social=_SCORE_STRATEGY,
        geopolitical=_SCORE_STRATEGY,
        fusion=_fusion_strategy(),
    )
    @settings(max_examples=500)
    def test_fused_sentiment_always_in_minus1_to_1(
        self,
        news: float,
        social: float,
        geopolitical: float,
        fusion: SentimentFusion,
    ) -> None:
        """Fused sentiment must always be in [-1, 1] for any inputs in [-1, 1]."""
        result = fusion.fuse(news, social, geopolitical)
        assert -1.0 <= result <= 1.0, (
            f"fused_sentiment out of [-1, 1]: {result} for inputs "
            f"(news={news}, social={social}, geo={geopolitical})"
        )

    @given(fusion=_fusion_strategy())
    @settings(max_examples=200)
    def test_fusion_all_zeros_returns_zero(self, fusion: SentimentFusion) -> None:
        """Fusion with all-zero inputs must return 0.0."""
        result = fusion.fuse(0.0, 0.0, 0.0)
        assert result == 0.0, f"Expected 0.0 for all-zero inputs, got {result}"

    def test_fusion_all_positive_one_returns_one(self) -> None:
        """Fusion with extreme inputs (+1, +1, +1) must return 1.0."""
        fusion = SentimentFusion()
        result = fusion.fuse(1.0, 1.0, 1.0)
        assert result == 1.0, f"Expected 1.0 for all +1 inputs, got {result}"

    def test_fusion_all_negative_one_returns_minus_one(self) -> None:
        """Fusion with extreme inputs (-1, -1, -1) must return -1.0."""
        fusion = SentimentFusion()
        result = fusion.fuse(-1.0, -1.0, -1.0)
        assert result == -1.0, f"Expected -1.0 for all -1 inputs, got {result}"

    @given(
        score=_SCORE_STRATEGY,
        fusion=_fusion_strategy(),
    )
    @settings(max_examples=300)
    def test_fusion_symmetric(self, score: float, fusion: SentimentFusion) -> None:
        """Fusion(-score, -score, -score) must equal -fusion(score, score, score)."""
        pos = fusion.fuse(score, score, score)
        neg = fusion.fuse(-score, -score, -score)
        assert abs(pos + neg) < 1e-9, (
            f"Fusion not symmetric: fuse({score})={pos}, fuse({-score})={neg}"
        )

    @given(
        news=_SCORE_STRATEGY,
        social=_SCORE_STRATEGY,
        geopolitical=_SCORE_STRATEGY,
        fusion=_fusion_strategy(),
    )
    @settings(max_examples=300)
    def test_fusion_is_finite(
        self,
        news: float,
        social: float,
        geopolitical: float,
        fusion: SentimentFusion,
    ) -> None:
        """Fused result must always be a finite number."""
        import math
        result = fusion.fuse(news, social, geopolitical)
        assert math.isfinite(result), f"fuse returned non-finite: {result}"

    @given(
        news=_SCORE_STRATEGY,
        social=_SCORE_STRATEGY,
        geopolitical=_SCORE_STRATEGY,
    )
    @settings(max_examples=200)
    def test_fusion_with_default_weights_in_range(
        self, news: float, social: float, geopolitical: float
    ) -> None:
        """Default SentimentFusion must return values in [-1, 1]."""
        fusion = SentimentFusion()
        result = fusion.fuse(news, social, geopolitical)
        assert -1.0 <= result <= 1.0, (
            f"Default fusion out of range: {result}"
        )
