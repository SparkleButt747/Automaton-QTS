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


def test_extract_paragraphs_from_html_skips_outside_article() -> None:  # T-FETCH-4
    """fed.gov wraps the real release in <div id='article'> — banner text must
    not leak into the corpus."""
    from scripts.fetch_fomc_data import extract_paragraphs_from_html

    html = (
        "<html><body>"
        "<p>Official websites use .gov A .gov website belongs to an official "
        "government organization in the United States and bla bla banner.</p>"
        "<p>Secure .gov websites use HTTPS A lock or https:// means you've "
        "safely connected to the .gov website. Boilerplate goes here.</p>"
        '<div id="article">'
        "<p>The Committee decided to maintain the target range for the federal "
        "funds rate at 5-1/4 to 5-1/2 percent at this meeting.</p>"
        "</div>"
        "</body></html>"
    )
    paragraphs = extract_paragraphs_from_html(html, min_chars=50)
    assert len(paragraphs) == 1
    assert "Committee" in paragraphs[0]
    assert "federal funds rate" in paragraphs[0]


def test_pdf_page_header_regex() -> None:  # T-FETCH-5
    """Page-header lines like 'Page 4 of 24' (with optional preceding header
    text on the same line) must be stripped so they don't appear in the
    extracted paragraphs corpus."""
    from scripts.fetch_fomc_data import _PDF_PAGE_HEADER_RE

    sample = (
        "Page 1 of 24\n"
        "CHAIR POWELL.  Good afternoon.\n"
        "Page 2 of 24\n"
        "Inflation has eased over the past year.\n"
    )
    cleaned = _PDF_PAGE_HEADER_RE.sub("", sample)
    assert "Page 1 of 24" not in cleaned
    assert "Page 2 of 24" not in cleaned
    assert "CHAIR POWELL" in cleaned
    assert "Inflation has eased" in cleaned


def test_pdf_running_footer_regex() -> None:  # T-FETCH-6
    """The per-page running footer ('December 13, 2023 Chair Powell's Press
    Conference FINAL') gets concatenated mid-paragraph by pypdf and must be
    scrubbed before paragraph splitting."""
    from scripts.fetch_fomc_data import _PDF_RUNNING_FOOTER_RE

    sample = (
        "Recent indicators suggest growth has slowed substantially. "
        "December 13, 2023 Chair Powell’s Press Conference FINAL "
        "Even so, GDP is on track to expand around 2.5 percent."
    )
    cleaned = _PDF_RUNNING_FOOTER_RE.sub("", sample)
    assert "FINAL" not in cleaned
    assert "Press Conference" not in cleaned
    assert "Recent indicators" in cleaned
    assert "GDP is on track" in cleaned
