"""Tests for NewsDataPoint custom data + TextEvent converters."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from qts.macro.news_classifier import NewsClassifier
from qts.nautilus.converters import news_data_to_text_event, text_event_to_news_data
from qts.nautilus.news_data import NewsDataPoint
from qts.world.events import TextEvent


def test_news_data_point_construction():  # T-V21-CONV-1
    p = NewsDataPoint(source="fomc", persona="powell", text="dovish", ts_event=123, ts_init=123)
    assert p.source == "fomc"
    assert p.text == "dovish"
    assert p.ts_event == 123
    assert p.ts_init == 123


def test_text_event_roundtrip_preserves_fields():  # T-V21-CONV-2
    evt = TextEvent(
        timestamp=datetime(2023, 12, 13, 19, 0, tzinfo=UTC),
        source="fed_press_release",
        persona=None,
        text="Inflation has slowed.",
        metadata={"event_kind": "FOMC"},
    )
    back = news_data_to_text_event(text_event_to_news_data(evt))
    assert back.source == evt.source
    assert back.text == evt.text
    assert back.persona == evt.persona  # None -> "" -> None


def test_roundtrip_preserves_classifier_cache_key(tmp_path: Path):  # T-V21-CONV-3
    evt = TextEvent(
        timestamp=datetime(2023, 12, 13, 19, 0, tzinfo=UTC),
        source="fed_press_release",
        persona=None,
        text="Inflation has slowed.",
        metadata={},
    )
    back = news_data_to_text_event(text_event_to_news_data(evt))
    clf = NewsClassifier(llm_client=object(), cache_dir=tmp_path)
    assert clf._cache_key(evt) == clf._cache_key(back)
