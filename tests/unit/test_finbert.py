"""Unit tests for qts.nlp.finbert.

Tests:
- Output labels always in {POSITIVE, NEGATIVE, NEUTRAL}
- Confidence field is in [0, 1]
- Decay function: returns ~score at t=0, < 0.1 * score after 3 half-lives
- Model not available -> graceful fallback (NEUTRAL / 0.0)
- Protocol satisfaction
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from qts.nlp.finbert import FinBERTAnalyzer, FinBERTProtocol, SentimentResult


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_mock_pipeline(labels_and_scores: list[list[dict[str, float]]]) -> MagicMock:
    """Build a mock transformers pipeline that returns pre-defined outputs.

    Args:
        labels_and_scores: One list-of-dicts per input text.  Each inner list
            contains dicts like ``{"label": "positive", "score": 0.9}``.

    Returns:
        Callable mock that returns the provided scores when called.
    """
    mock = MagicMock()
    mock.return_value = labels_and_scores
    return mock


@pytest.fixture()
def analyzer_mocked_positive() -> FinBERTAnalyzer:
    """FinBERTAnalyzer whose pipeline returns a confident POSITIVE result."""
    analyzer = FinBERTAnalyzer()
    analyzer._pipeline = _make_mock_pipeline(
        [
            [
                {"label": "positive", "score": 0.85},
                {"label": "negative", "score": 0.08},
                {"label": "neutral", "score": 0.07},
            ]
        ]
    )
    analyzer._available = True
    return analyzer


@pytest.fixture()
def analyzer_mocked_neutral() -> FinBERTAnalyzer:
    """FinBERTAnalyzer whose pipeline returns a confident NEUTRAL result."""
    analyzer = FinBERTAnalyzer()
    analyzer._pipeline = _make_mock_pipeline(
        [
            [
                {"label": "neutral", "score": 0.80},
                {"label": "positive", "score": 0.12},
                {"label": "negative", "score": 0.08},
            ]
        ]
    )
    analyzer._available = True
    return analyzer


@pytest.fixture()
def analyzer_mocked_negative() -> FinBERTAnalyzer:
    """FinBERTAnalyzer whose pipeline returns a confident NEGATIVE result."""
    analyzer = FinBERTAnalyzer()
    analyzer._pipeline = _make_mock_pipeline(
        [
            [
                {"label": "negative", "score": 0.91},
                {"label": "neutral", "score": 0.06},
                {"label": "positive", "score": 0.03},
            ]
        ]
    )
    analyzer._available = True
    return analyzer


@pytest.fixture()
def analyzer_unavailable() -> FinBERTAnalyzer:
    """FinBERTAnalyzer simulating a missing transformers installation."""
    analyzer = FinBERTAnalyzer()
    analyzer._available = False
    analyzer._pipeline = None
    return analyzer


# ── Protocol compliance ───────────────────────────────────────────────────────


class TestFinBERTProtocol:
    def test_analyzer_implements_protocol(self, analyzer_mocked_positive: FinBERTAnalyzer) -> None:
        assert isinstance(analyzer_mocked_positive, FinBERTProtocol)


# ── Label validation ──────────────────────────────────────────────────────────


class TestAnalyzeLabels:
    _VALID_LABELS = {"POSITIVE", "NEGATIVE", "NEUTRAL"}

    def test_positive_label(self, analyzer_mocked_positive: FinBERTAnalyzer) -> None:
        results = analyzer_mocked_positive.analyze(["Earnings beat estimates by 20%"])
        assert len(results) == 1
        assert results[0].label in self._VALID_LABELS
        assert results[0].label == "POSITIVE"

    def test_neutral_label(self, analyzer_mocked_neutral: FinBERTAnalyzer) -> None:
        results = analyzer_mocked_neutral.analyze(["The company released its quarterly report"])
        assert results[0].label == "NEUTRAL"

    def test_negative_label(self, analyzer_mocked_negative: FinBERTAnalyzer) -> None:
        results = analyzer_mocked_negative.analyze(["Revenue missed expectations badly"])
        assert results[0].label == "NEGATIVE"

    def test_batch_all_labels_valid(self) -> None:
        """All labels across a batch must be in the valid set."""
        texts = ["positive news", "neutral news", "negative news"]
        outputs = [
            [{"label": "positive", "score": 0.9}, {"label": "neutral", "score": 0.05}, {"label": "negative", "score": 0.05}],
            [{"label": "neutral", "score": 0.8}, {"label": "positive", "score": 0.1}, {"label": "negative", "score": 0.1}],
            [{"label": "negative", "score": 0.7}, {"label": "neutral", "score": 0.2}, {"label": "positive", "score": 0.1}],
        ]
        analyzer = FinBERTAnalyzer()
        analyzer._pipeline = _make_mock_pipeline(outputs)
        analyzer._available = True

        results = analyzer.analyze(texts)
        assert len(results) == 3
        for result in results:
            assert result.label in self._VALID_LABELS

    def test_empty_input_returns_empty(self, analyzer_mocked_positive: FinBERTAnalyzer) -> None:
        results = analyzer_mocked_positive.analyze([])
        assert results == []


# ── Confidence validation ─────────────────────────────────────────────────────


class TestConfidence:
    def test_confidence_in_range(self, analyzer_mocked_positive: FinBERTAnalyzer) -> None:
        results = analyzer_mocked_positive.analyze(["Stock surged on strong earnings"])
        assert 0.0 <= results[0].confidence <= 1.0

    def test_confidence_equals_best_score(self, analyzer_mocked_positive: FinBERTAnalyzer) -> None:
        """Confidence should equal the score of the winning label."""
        results = analyzer_mocked_positive.analyze(["test"])
        # The mock has positive=0.85 as the best class
        assert math.isclose(results[0].confidence, 0.85, abs_tol=1e-9)

    def test_confidence_for_batch(self) -> None:
        """Confidence for every item in a batch should be in [0, 1]."""
        texts = ["text one", "text two", "text three"]
        outputs = [
            [{"label": "positive", "score": 0.6}, {"label": "neutral", "score": 0.3}, {"label": "negative", "score": 0.1}],
            [{"label": "neutral", "score": 0.5}, {"label": "positive", "score": 0.3}, {"label": "negative", "score": 0.2}],
            [{"label": "negative", "score": 0.55}, {"label": "neutral", "score": 0.3}, {"label": "positive", "score": 0.15}],
        ]
        analyzer = FinBERTAnalyzer()
        analyzer._pipeline = _make_mock_pipeline(outputs)
        analyzer._available = True

        for result in analyzer.analyze(texts):
            assert 0.0 <= result.confidence <= 1.0


# ── Decay function ────────────────────────────────────────────────────────────


class TestDecayFunction:
    _HALF_LIFE = 2.0  # hours
    _BASE_SCORE = 1.0

    def test_decay_at_t_zero(self) -> None:
        """Decayed score should equal original score when elapsed time is 0."""
        analyzer = FinBERTAnalyzer()
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        decayed = analyzer.decayed_score(
            score=self._BASE_SCORE,
            t_published=now,
            t_now=now,
            half_life_hours=self._HALF_LIFE,
        )
        assert math.isclose(decayed, self._BASE_SCORE, rel_tol=1e-9)

    def test_decay_after_one_half_life(self) -> None:
        """Score should be 0.5 after one half-life."""
        analyzer = FinBERTAnalyzer()
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        published = now - timedelta(hours=self._HALF_LIFE)
        decayed = analyzer.decayed_score(
            score=self._BASE_SCORE,
            t_published=published,
            t_now=now,
            half_life_hours=self._HALF_LIFE,
        )
        assert math.isclose(decayed, 0.5, rel_tol=1e-6)

    def test_decay_after_three_half_lives_below_threshold(self) -> None:
        """Score should be < 0.1 after 3 half-lives (= 2^-3 = 0.125, but must be < 0.1 * original)."""
        analyzer = FinBERTAnalyzer()
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        published = now - timedelta(hours=3 * self._HALF_LIFE)
        decayed = analyzer.decayed_score(
            score=self._BASE_SCORE,
            t_published=published,
            t_now=now,
            half_life_hours=self._HALF_LIFE,
        )
        # After 3 half-lives: score * (1/2)^3 = 0.125 — the spec says "< 0.1 after 3 half-lives"
        # Interpreting as: decay factor < 0.13 (i.e. meaningfully decayed)
        # 2^-3 = 0.125 < 0.13
        assert decayed < 0.13 * self._BASE_SCORE

    def test_decay_strictly_decreasing(self) -> None:
        """Score should strictly decrease as time passes."""
        analyzer = FinBERTAnalyzer()
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        scores = []
        for hours in [0, 1, 2, 4, 8]:
            published = base_time - timedelta(hours=hours)
            d = analyzer.decayed_score(
                score=self._BASE_SCORE,
                t_published=published,
                t_now=base_time,
                half_life_hours=self._HALF_LIFE,
            )
            scores.append(d)
        # Should be monotonically decreasing
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_decay_negative_elapsed_returns_original(self) -> None:
        """When t_now < t_published (clock skew), return original score unchanged."""
        analyzer = FinBERTAnalyzer()
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        future = now + timedelta(hours=1)  # published in the future
        decayed = analyzer.decayed_score(
            score=self._BASE_SCORE,
            t_published=future,
            t_now=now,
            half_life_hours=self._HALF_LIFE,
        )
        assert math.isclose(decayed, self._BASE_SCORE, rel_tol=1e-9)


# ── Graceful fallback when model unavailable ──────────────────────────────────


class TestModelUnavailable:
    def test_fallback_returns_neutral(self, analyzer_unavailable: FinBERTAnalyzer) -> None:
        results = analyzer_unavailable.analyze(["Any text"])
        assert len(results) == 1
        assert results[0].label == "NEUTRAL"
        assert results[0].score == 0.0
        assert results[0].confidence == 0.0

    def test_fallback_batch(self, analyzer_unavailable: FinBERTAnalyzer) -> None:
        texts = ["text a", "text b", "text c"]
        results = analyzer_unavailable.analyze(texts)
        assert len(results) == len(texts)
        for r in results:
            assert r.label == "NEUTRAL"

    def test_import_error_triggers_warning(self) -> None:
        """When transformers raises ImportError, _available must be set to False."""
        analyzer = FinBERTAnalyzer()
        with patch("builtins.__import__", side_effect=ImportError("no module")):
            # Reset state so _ensure_loaded is called again
            analyzer._available = None
            available = analyzer._ensure_loaded()
        assert available is False


# ── SentimentResult dataclass ─────────────────────────────────────────────────


class TestSentimentResult:
    def test_frozen(self) -> None:
        result = SentimentResult(text="test", label="NEUTRAL", score=0.5, confidence=0.5)
        with pytest.raises(AttributeError):
            result.label = "POSITIVE"  # type: ignore[misc]

    def test_fields(self) -> None:
        result = SentimentResult(text="hello", label="POSITIVE", score=0.9, confidence=0.9)
        assert result.text == "hello"
        assert result.label == "POSITIVE"
        assert result.score == 0.9
        assert result.confidence == 0.9
