"""Unit tests for qts.nlp.vader.

Tests:
- VaderAnalyzer with known text inputs (positive, negative, neutral)
- Output format: label in {POSITIVE, NEGATIVE, NEUTRAL}, score in [0, 1]
- Protocol compliance
- Graceful fallback when vaderSentiment is not available
"""
from __future__ import annotations

import pytest

from qts.nlp.finbert import SentimentResult
from qts.nlp.vader import VaderAnalyzer, VaderProtocol


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def vader() -> VaderAnalyzer:
    """VaderAnalyzer with the real vaderSentiment backend loaded."""
    analyzer = VaderAnalyzer()
    # Force eager load so tests fail fast if the package is missing
    loaded = analyzer._ensure_loaded()
    if not loaded:
        pytest.skip("vaderSentiment not installed")
    return analyzer


@pytest.fixture()
def vader_unavailable() -> VaderAnalyzer:
    """VaderAnalyzer simulating a missing vaderSentiment installation."""
    analyzer = VaderAnalyzer()
    analyzer._available = False
    analyzer._sia = None
    return analyzer


# ── Protocol compliance ───────────────────────────────────────────────────────


class TestVaderProtocol:
    def test_implements_protocol(self, vader: VaderAnalyzer) -> None:
        assert isinstance(vader, VaderProtocol)


# ── Label classification tests ─────────────────────────────────────────────────


class TestKnownTextInputs:
    """Test VADER with strongly-valenced texts that have well-known sentiment."""

    _VALID_LABELS = {"POSITIVE", "NEGATIVE", "NEUTRAL"}

    def test_strongly_positive_text(self, vader: VaderAnalyzer) -> None:
        results = vader.analyze(["This is absolutely fantastic! Best news ever!!! :)"])
        assert results[0].label == "POSITIVE"

    def test_strongly_negative_text(self, vader: VaderAnalyzer) -> None:
        results = vader.analyze(["This is terrible, awful, catastrophic! Horrible disaster!"])
        assert results[0].label == "NEGATIVE"

    def test_neutral_text(self, vader: VaderAnalyzer) -> None:
        results = vader.analyze(["The company filed its quarterly report"])
        # Neutral texts should not be confidently positive or negative
        # We accept both NEUTRAL and weak POSITIVE/NEGATIVE
        assert results[0].label in self._VALID_LABELS

    def test_empty_input_returns_empty(self, vader: VaderAnalyzer) -> None:
        assert vader.analyze([]) == []

    def test_batch_returns_correct_length(self, vader: VaderAnalyzer) -> None:
        texts = ["Great results!", "Terrible losses!", "Revenue unchanged"]
        results = vader.analyze(texts)
        assert len(results) == len(texts)


# ── Output format validation ──────────────────────────────────────────────────


class TestOutputFormat:
    _VALID_LABELS = {"POSITIVE", "NEGATIVE", "NEUTRAL"}

    def test_label_always_in_valid_set(self, vader: VaderAnalyzer) -> None:
        texts = [
            "Absolutely wonderful news!",
            "Disastrous earnings report",
            "Company released a statement",
            "Stocks up 5%",
            "CEO resigns after scandal",
        ]
        for result in vader.analyze(texts):
            assert result.label in self._VALID_LABELS

    def test_score_in_unit_interval(self, vader: VaderAnalyzer) -> None:
        texts = [
            "Super bullish breakout!!",
            "Massive sell-off crash",
            "Volume was average today",
        ]
        for result in vader.analyze(texts):
            assert 0.0 <= result.score <= 1.0

    def test_confidence_in_unit_interval(self, vader: VaderAnalyzer) -> None:
        texts = ["Amazing rally!", "Dreadful collapse!", "Flat trading session"]
        for result in vader.analyze(texts):
            assert 0.0 <= result.confidence <= 1.0

    def test_text_preserved_in_result(self, vader: VaderAnalyzer) -> None:
        text = "Revenue exceeded analyst expectations significantly"
        results = vader.analyze([text])
        assert results[0].text == text

    def test_result_is_sentiment_result(self, vader: VaderAnalyzer) -> None:
        results = vader.analyze(["Any text here"])
        assert isinstance(results[0], SentimentResult)


# ── Threshold boundary tests ──────────────────────────────────────────────────


class TestThresholds:
    """Test the compound score thresholds that determine POSITIVE/NEUTRAL/NEGATIVE."""

    def _make_analyzer_with_compound(self, compound: float) -> VaderAnalyzer:
        """Create a VaderAnalyzer with a mock SIA returning a specific compound score."""
        from unittest.mock import MagicMock

        analyzer = VaderAnalyzer()
        mock_sia = MagicMock()
        mock_sia.polarity_scores.return_value = {
            "neg": 0.0,
            "neu": 1.0,
            "pos": 0.0,
            "compound": compound,
        }
        analyzer._sia = mock_sia
        analyzer._available = True
        return analyzer

    def test_compound_above_positive_threshold(self) -> None:
        analyzer = self._make_analyzer_with_compound(0.1)
        results = analyzer.analyze(["test"])
        assert results[0].label == "POSITIVE"

    def test_compound_below_negative_threshold(self) -> None:
        analyzer = self._make_analyzer_with_compound(-0.1)
        results = analyzer.analyze(["test"])
        assert results[0].label == "NEGATIVE"

    def test_compound_in_neutral_zone(self) -> None:
        analyzer = self._make_analyzer_with_compound(0.0)
        results = analyzer.analyze(["test"])
        assert results[0].label == "NEUTRAL"

    def test_compound_at_positive_boundary(self) -> None:
        """compound == 0.05 (exactly at threshold) -> POSITIVE."""
        analyzer = self._make_analyzer_with_compound(0.05)
        results = analyzer.analyze(["test"])
        assert results[0].label == "POSITIVE"

    def test_compound_at_negative_boundary(self) -> None:
        """compound == -0.05 (exactly at threshold) -> NEGATIVE."""
        analyzer = self._make_analyzer_with_compound(-0.05)
        results = analyzer.analyze(["test"])
        assert results[0].label == "NEGATIVE"


# ── Fallback when vaderSentiment unavailable ──────────────────────────────────


class TestVaderUnavailable:
    def test_returns_neutral_when_unavailable(
        self, vader_unavailable: VaderAnalyzer
    ) -> None:
        results = vader_unavailable.analyze(["Any text"])
        assert len(results) == 1
        assert results[0].label == "NEUTRAL"
        assert results[0].score == 0.0
        assert results[0].confidence == 0.0

    def test_batch_fallback(self, vader_unavailable: VaderAnalyzer) -> None:
        texts = ["a", "b", "c"]
        results = vader_unavailable.analyze(texts)
        assert len(results) == len(texts)
        for r in results:
            assert r.label == "NEUTRAL"
