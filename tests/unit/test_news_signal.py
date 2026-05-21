"""Tests for the NewsSignal multi-axis structured output."""

from __future__ import annotations

import pytest


def test_news_signal_shape() -> None:  # T-NSIG-1
    from qts.macro.news_signal import NewsSignal

    sig = NewsSignal(direction="bull", confidence=0.8, relevance=0.9, magnitude=0.6)
    assert sig.direction == "bull"
    assert sig.confidence == 0.8
    assert sig.relevance == 0.9
    assert sig.magnitude == 0.6


def test_news_signal_rejects_invalid_direction() -> None:  # T-NSIG-2
    from qts.macro.news_signal import NewsSignal

    with pytest.raises(ValueError, match="direction must be"):
        NewsSignal(direction="up", confidence=0.5, relevance=0.5, magnitude=0.5)


def test_news_signal_clamps_out_of_range() -> None:  # T-NSIG-3
    from qts.macro.news_signal import NewsSignal

    with pytest.raises(ValueError, match="must be in"):
        NewsSignal(direction="bull", confidence=1.5, relevance=0.5, magnitude=0.5)

    with pytest.raises(ValueError, match="must be in"):
        NewsSignal(direction="bull", confidence=0.5, relevance=-0.1, magnitude=0.5)


def test_news_signal_direction_sign() -> None:  # T-NSIG-4
    from qts.macro.news_signal import NewsSignal

    assert (
        NewsSignal(direction="bull", confidence=0.5, relevance=0.5, magnitude=0.5).direction_sign
        == 1.0
    )
    assert (
        NewsSignal(direction="bear", confidence=0.5, relevance=0.5, magnitude=0.5).direction_sign
        == -1.0
    )
    assert (
        NewsSignal(direction="neutral", confidence=0.5, relevance=0.5, magnitude=0.5).direction_sign
        == 0.0
    )


def test_news_signal_alpha_contribution() -> None:  # T-NSIG-5
    from qts.macro.news_signal import NewsSignal

    sig = NewsSignal(direction="bull", confidence=0.8, relevance=1.0, magnitude=0.5)
    # alpha = direction_sign × confidence × relevance × magnitude = 1 × 0.8 × 1.0 × 0.5 = 0.4
    assert sig.alpha_contribution() == pytest.approx(0.4)

    sig_bear = NewsSignal(direction="bear", confidence=0.6, relevance=0.5, magnitude=0.8)
    # -1 × 0.6 × 0.5 × 0.8 = -0.24
    assert sig_bear.alpha_contribution() == pytest.approx(-0.24)
