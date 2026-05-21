"""Tests for qts.world.events."""

from __future__ import annotations

from datetime import UTC, datetime


def test_text_event_construction() -> None:  # T-WEV-1
    from qts.world.events import TextEvent

    ts = datetime(2025, 3, 19, 14, 5, tzinfo=UTC)
    event = TextEvent(
        timestamp=ts,
        source="powell",
        persona="Jerome Powell",
        text="Inflation remains stubborn.",
        metadata={"surprise_bucket": "hawkish", "regime": "BEAR"},
    )

    assert event.timestamp == ts
    assert event.source == "powell"
    assert event.persona == "Jerome Powell"
    assert event.metadata["surprise_bucket"] == "hawkish"


def test_text_event_is_frozen() -> None:  # T-WEV-2
    from dataclasses import FrozenInstanceError

    import pytest

    from qts.world.events import TextEvent

    event = TextEvent(
        timestamp=datetime(2025, 3, 19, 14, 5, tzinfo=UTC),
        source="powell",
        persona=None,
        text="test",
        metadata={},
    )
    with pytest.raises(FrozenInstanceError):
        event.source = "other"  # type: ignore[misc]


def test_macro_event_construction() -> None:  # T-WEV-3
    from qts.world.events import MacroEvent, TextEvent

    ts = datetime(2025, 3, 19, 14, 0, tzinfo=UTC)
    text = TextEvent(
        timestamp=ts,
        source="fed_press_release",
        persona=None,
        text="The Committee decided to maintain the target range...",
        metadata={},
    )
    event = MacroEvent(
        timestamp=ts,
        kind="FOMC",
        expected=5.25,
        actual=5.50,
        surprise=0.25,
        text_event=text,
    )

    assert event.kind == "FOMC"
    assert event.surprise == 0.25
    assert event.text_event is text


def test_macro_event_optional_text() -> None:  # T-WEV-4
    from qts.world.events import MacroEvent

    event = MacroEvent(
        timestamp=datetime(2025, 3, 19, 14, 0, tzinfo=UTC),
        kind="CPI",
        expected=3.2,
        actual=3.5,
        surprise=0.3,
        text_event=None,
    )
    assert event.text_event is None
