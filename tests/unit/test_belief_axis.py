"""Tests for the exponential-decay belief primitive."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest


def test_belief_axis_default_zero() -> None:  # T-BEL-1
    from qts.strategies.belief import BeliefAxis

    b = BeliefAxis(half_life=timedelta(hours=4))
    now = datetime(2024, 1, 1, tzinfo=UTC)
    assert b.at(now) == 0.0


def test_belief_axis_update_then_at_returns_full_value() -> None:  # T-BEL-2
    from qts.strategies.belief import BeliefAxis

    b = BeliefAxis(half_life=timedelta(hours=4))
    t = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    b.update(value=0.6, now=t)

    # No time elapsed → no decay → full value.
    assert b.at(t) == pytest.approx(0.6)


def test_belief_axis_decays_to_half_after_half_life() -> None:  # T-BEL-3
    from qts.strategies.belief import BeliefAxis

    b = BeliefAxis(half_life=timedelta(hours=4))
    t0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    b.update(value=0.8, now=t0)

    t1 = t0 + timedelta(hours=4)
    # One half-life elapsed → value × 0.5.
    assert b.at(t1) == pytest.approx(0.4)

    t2 = t0 + timedelta(hours=8)
    # Two half-lives → value × 0.25.
    assert b.at(t2) == pytest.approx(0.2)


def test_belief_axis_signed_decay() -> None:  # T-BEL-4
    from qts.strategies.belief import BeliefAxis

    b = BeliefAxis(half_life=timedelta(hours=2))
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    b.update(value=-0.6, now=t0)

    t1 = t0 + timedelta(hours=2)
    assert b.at(t1) == pytest.approx(-0.3)


def test_belief_axis_update_replaces_value() -> None:  # T-BEL-5
    from qts.strategies.belief import BeliefAxis

    b = BeliefAxis(half_life=timedelta(hours=4))
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    b.update(value=0.5, now=t0)

    t1 = t0 + timedelta(hours=2)
    b.update(value=-0.4, now=t1)

    # New value replaces the old; decay timer resets to t1.
    assert b.at(t1) == pytest.approx(-0.4)
    assert b.at(t1 + timedelta(hours=4)) == pytest.approx(-0.2)
