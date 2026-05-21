"""Unit tests for NewsBelief — multi-axis decaying news belief."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from qts.macro.news_signal import NewsSignal
from qts.strategies.belief import NewsBelief

_T0 = datetime(2023, 12, 13, 19, 0, tzinfo=UTC)


def _sig(direction="bull", confidence=1.0, relevance=1.0, magnitude=1.0):
    return NewsSignal(
        direction=direction, confidence=confidence, relevance=relevance, magnitude=magnitude
    )


def test_empty_belief_is_zero():  # T-V21-BELIEF-1
    assert NewsBelief(half_life=timedelta(hours=4)).value_at(_T0) == 0.0


def test_single_event_matches_alpha_contribution():  # T-V21-BELIEF-2
    belief = NewsBelief(half_life=timedelta(hours=4))
    sig = _sig(direction="bull", confidence=0.8, relevance=1.0, magnitude=0.5)
    belief.observe(sig, _T0)
    assert belief.value_at(_T0) == pytest.approx(0.4)
    assert belief.value_at(_T0) == pytest.approx(sig.alpha_contribution())


def test_conviction_accumulates():  # T-V21-BELIEF-3
    belief = NewsBelief(half_life=timedelta(hours=4))
    one = _sig(direction="bull", confidence=0.6, relevance=1.0, magnitude=1.0)
    belief.observe(one, _T0)
    single = belief.value_at(_T0)
    belief.observe(one, _T0)  # second same-direction read at the same instant
    assert belief.value_at(_T0) == pytest.approx(2 * single)


def test_neutral_read_is_ignored_not_cancelling():  # T-V21-BELIEF-4 (regression: v2 0-trades)
    belief = NewsBelief(half_life=timedelta(hours=4))
    belief.observe(_sig("bull", 0.9, 1.0, 0.8), _T0)
    before = belief.value_at(_T0)
    assert before > 0
    # A trailing neutral, low-relevance Q&A (FOMC event #25) must NOT wipe the signal.
    belief.observe(_sig("neutral", 0.0, 0.1, 0.1), _T0)
    assert belief.value_at(_T0) == pytest.approx(before)


def test_relevance_gate_is_running_max():  # T-V21-BELIEF-5
    belief = NewsBelief(half_life=timedelta(hours=4))
    belief.observe(_sig("bull", 1.0, 0.9, 1.0), _T0)
    high = belief.value_at(_T0)
    belief.observe(_sig("bull", 0.0, 0.1, 1.0), _T0)  # low-rel can't pull the gate down
    assert belief.value_at(_T0) == pytest.approx(high)


def test_single_event_halves_at_half_life():  # T-V21-BELIEF-6
    belief = NewsBelief(half_life=timedelta(hours=2))
    belief.observe(_sig("bull", 1.0, 1.0, 1.0), _T0)
    full = belief.value_at(_T0)
    assert belief.value_at(_T0 + timedelta(hours=2)) == pytest.approx(full * 0.5)


def test_decay_guard_never_amplifies():  # T-V21-BELIEF-7 (regression: negative-elapsed look-ahead)
    belief = NewsBelief(half_life=timedelta(hours=2))
    belief.observe(_sig("bull", 1.0, 1.0, 1.0), _T0)
    at_update = belief.value_at(_T0)
    before = belief.value_at(_T0 - timedelta(hours=1))  # reading BEFORE last_update
    assert before <= at_update
    assert before == pytest.approx(at_update)  # clamped elapsed=0 → no amplification
