"""Tests for the FOMC data fetcher script's parsing helpers."""

from __future__ import annotations

from datetime import UTC, datetime


def test_parse_binance_klines() -> None:  # T-FETCH-1
    from scripts.fetch_fomc_data import parse_binance_klines

    raw = [
        # [open_time_ms, open, high, low, close, volume, close_time_ms, ...]
        [
            1702425600000,
            "40000.0",
            "40100.0",
            "39950.0",
            "40050.0",
            "12.3",
            1702425659999,
            "0",
            0,
            "0",
            "0",
            "0",
        ],
        [
            1702425660000,
            "40050.0",
            "40080.0",
            "40010.0",
            "40060.0",
            "9.1",
            1702425719999,
            "0",
            0,
            "0",
            "0",
            "0",
        ],
    ]
    bars = parse_binance_klines(raw, symbol="BTCUSDT")
    assert len(bars) == 2
    assert bars[0]["timestamp"] == datetime(2023, 12, 13, 0, 0, tzinfo=UTC)
    assert bars[0]["open"] == 40_000.0
    assert bars[0]["close"] == 40_050.0
    assert bars[1]["timestamp"] == datetime(2023, 12, 13, 0, 1, tzinfo=UTC)


def test_extract_paragraphs_from_html() -> None:  # T-FETCH-2
    from scripts.fetch_fomc_data import extract_paragraphs_from_html

    html = (
        "<html><body>"
        "<p>This is the first paragraph of the FOMC transcript document.</p>"
        "<p>Short.</p>"  # too short — filtered out
        "<p>Here is a longer paragraph that should survive the filter "
        "because it's discussion-worthy.</p>"
        "</body></html>"
    )
    paragraphs = extract_paragraphs_from_html(html, min_chars=50)
    assert len(paragraphs) == 2
    assert "first paragraph" in paragraphs[0]
    assert "longer paragraph" in paragraphs[1]


def test_assign_paragraph_timestamps() -> None:  # T-FETCH-3
    from scripts.fetch_fomc_data import assign_paragraph_timestamps

    start = datetime(2023, 12, 13, 19, 30, tzinfo=UTC)
    paragraphs = ["a", "b", "c", "d"]
    spaced = assign_paragraph_timestamps(paragraphs, start=start, span_minutes=60)

    assert len(spaced) == 4
    assert spaced[0]["timestamp"] == start
    # Equal spacing across 60 minutes -> 20-min gaps.
    assert (spaced[1]["timestamp"] - spaced[0]["timestamp"]).total_seconds() == 20 * 60
    assert (spaced[3]["timestamp"] - spaced[0]["timestamp"]).total_seconds() == 60 * 60
