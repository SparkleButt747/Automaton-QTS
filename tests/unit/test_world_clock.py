"""Tests for qts.world.clock."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def test_clock_starts_at_start() -> None:  # T-WCLK-1
    from qts.world.clock import SimulatedClock

    start = datetime(2025, 3, 19, 0, 0, tzinfo=UTC)
    end = datetime(2025, 3, 20, 0, 0, tzinfo=UTC)
    clock = SimulatedClock(start=start, end=end, tick=timedelta(minutes=1))

    assert clock.now == start


def test_clock_advances_by_tick() -> None:  # T-WCLK-2
    from qts.world.clock import SimulatedClock

    start = datetime(2025, 3, 19, 0, 0, tzinfo=UTC)
    end = datetime(2025, 3, 19, 0, 5, tzinfo=UTC)
    clock = SimulatedClock(start=start, end=end, tick=timedelta(minutes=1))

    clock.advance()
    assert clock.now == start + timedelta(minutes=1)
    clock.advance()
    assert clock.now == start + timedelta(minutes=2)


def test_clock_iter_yields_every_tick() -> None:  # T-WCLK-3
    from qts.world.clock import SimulatedClock

    start = datetime(2025, 3, 19, 0, 0, tzinfo=UTC)
    end = datetime(2025, 3, 19, 0, 3, tzinfo=UTC)
    clock = SimulatedClock(start=start, end=end, tick=timedelta(minutes=1))

    ts = list(clock.iter_ticks())
    # Inclusive of start, exclusive of end — same as range()
    assert ts == [
        datetime(2025, 3, 19, 0, 0, tzinfo=UTC),
        datetime(2025, 3, 19, 0, 1, tzinfo=UTC),
        datetime(2025, 3, 19, 0, 2, tzinfo=UTC),
    ]


def test_clock_finished_at_end() -> None:  # T-WCLK-4
    from qts.world.clock import SimulatedClock

    start = datetime(2025, 3, 19, 0, 0, tzinfo=UTC)
    end = datetime(2025, 3, 19, 0, 2, tzinfo=UTC)
    clock = SimulatedClock(start=start, end=end, tick=timedelta(minutes=1))

    assert not clock.finished()
    clock.advance()
    assert not clock.finished()
    clock.advance()
    assert clock.finished()
