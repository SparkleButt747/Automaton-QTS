"""Tests for qts.world.bar_aggregator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def test_aggregator_produces_one_bar_per_minute() -> None:  # T-WBA-1
    from qts.world.bar_aggregator import BarAggregator

    start = datetime(2025, 3, 19, 14, 0, tzinfo=UTC)
    agg = BarAggregator(symbol="BTCUSDT", tick=timedelta(minutes=1))

    # Three trades inside the first minute
    agg.add_trade(start + timedelta(seconds=10), price=30000.0, qty=0.1)
    agg.add_trade(start + timedelta(seconds=20), price=30050.0, qty=0.2)
    agg.add_trade(start + timedelta(seconds=50), price=30025.0, qty=0.1)

    # Roll over: one trade in next minute
    agg.add_trade(start + timedelta(minutes=1, seconds=10), price=30100.0, qty=0.1)

    bars = agg.flush(end=start + timedelta(minutes=2))
    assert len(bars) == 2

    first = bars[0]
    assert first.timestamp == start
    assert first.open == 30000.0
    assert first.high == 30050.0
    assert first.low == 30000.0
    assert first.close == 30025.0
    assert abs(first.volume - 0.4) < 1e-9


def test_aggregator_emits_carry_bar_when_no_trades() -> None:  # T-WBA-2
    """If a minute has no trades, emit a flat bar from the last close."""
    from qts.world.bar_aggregator import BarAggregator

    start = datetime(2025, 3, 19, 14, 0, tzinfo=UTC)
    agg = BarAggregator(symbol="BTCUSDT", tick=timedelta(minutes=1))
    agg.add_trade(start + timedelta(seconds=10), price=30000.0, qty=0.1)

    bars = agg.flush(end=start + timedelta(minutes=3))
    # Minute 0: real trade ; Minutes 1, 2: carry bars
    assert len(bars) == 3
    assert bars[1].open == bars[1].close == 30000.0
    assert bars[1].volume == 0.0
    assert bars[2].open == bars[2].close == 30000.0
