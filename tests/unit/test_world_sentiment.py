"""Tests for qts.world.sentiment."""

from __future__ import annotations


def test_scorer_neutral_on_empty_text() -> None:  # T-WSE-1
    from qts.world.sentiment import SentimentScorer

    scorer = SentimentScorer()
    assert scorer.score("") == 0.0


def test_scorer_positive_on_dovish_keywords() -> None:  # T-WSE-2
    from qts.world.sentiment import SentimentScorer

    scorer = SentimentScorer()
    score = scorer.score("Progress on inflation; we may be near the peak of this cycle.")
    assert score > 0.0


def test_scorer_negative_on_hawkish_keywords() -> None:  # T-WSE-3
    from qts.world.sentiment import SentimentScorer

    scorer = SentimentScorer()
    score = scorer.score("Inflation stubborn; further hikes warranted.")
    assert score < 0.0


def test_scorer_bounded_pm_one() -> None:  # T-WSE-4
    from qts.world.sentiment import SentimentScorer

    scorer = SentimentScorer()
    for text in (
        "great wonderful amazing dovish dovish dovish dovish dovish progress progress",
        "terrible awful hawk hawk hawk hawk hawk hawk hawk hawk",
        "neutral statement",
    ):
        s = scorer.score(text)
        assert -1.0 <= s <= 1.0, f"score {s} out of [-1, 1] for: {text}"


def test_scorer_deterministic() -> None:  # T-WSE-5
    from qts.world.sentiment import SentimentScorer

    scorer = SentimentScorer()
    text = "Inflation may be coming back to target."
    assert scorer.score(text) == scorer.score(text)
