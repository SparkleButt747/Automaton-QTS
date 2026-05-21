"""One-off fetcher for a curated FOMC dataset.

Pulls real BTCUSDT 1m bars from Binance + FOMC statement + Powell press
conference transcript from federalreserve.gov, into a single directory.

Usage:
    .venv/bin/python -m scripts.fetch_fomc_data 2023-12-13 data/real/fomc/2023-12-13

The output is committed to the repo so the integration test is reproducible
without re-fetching.
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import logging
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from pypdf import PdfReader

logger = logging.getLogger(__name__)

_BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
_BINANCE_MAX_LIMIT = 1000
_PARAGRAPH_MIN_CHARS = 80
_PRESS_CONF_START_UTC_HOUR = 19  # 14:00 ET — 30 min after press release
_PRESS_CONF_START_UTC_MINUTE = 30


def parse_binance_klines(raw: list[list[Any]], symbol: str) -> list[dict]:
    """Convert raw Binance klines payload into bar dicts."""
    bars: list[dict] = []
    for row in raw:
        open_ms = int(row[0])
        bars.append(
            {
                "timestamp": datetime.fromtimestamp(open_ms / 1000.0, tz=UTC),
                "symbol": symbol,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )
    return bars


def fetch_binance_bars(symbol: str, day: datetime) -> list[dict]:
    """Fetch all 1m bars for `day` (UTC) from Binance, chunked across multiple calls."""
    start_ms = int(day.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    end_ms = int((day + timedelta(days=1)).timestamp() * 1000) - 1

    bars: list[dict] = []
    cursor_ms = start_ms
    while cursor_ms < end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": "1m",
            "startTime": cursor_ms,
            "endTime": end_ms,
            "limit": _BINANCE_MAX_LIMIT,
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.get(_BINANCE_KLINES_URL, params=params)
            response.raise_for_status()
            raw = response.json()
        if not raw:
            break
        chunk = parse_binance_klines(raw, symbol)
        bars.extend(chunk)
        cursor_ms = int(raw[-1][0]) + 60_000  # advance past the last open_time
        if len(chunk) < _BINANCE_MAX_LIMIT:
            break
    return bars


def write_bars_csv(bars: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for b in bars:
            writer.writerow(
                [
                    b["timestamp"].isoformat(),
                    b["open"],
                    b["high"],
                    b["low"],
                    b["close"],
                    b["volume"],
                ]
            )


def fetch_fomc_statement(date_str: str) -> str:
    """Fetch the FOMC statement HTML from federalreserve.gov and extract its body text."""
    # Standard URL pattern for FOMC press releases.
    yyyymmdd = date_str.replace("-", "")
    url = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{yyyymmdd}a.htm"
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    body = response.text
    paragraphs = extract_paragraphs_from_html(body, min_chars=_PARAGRAPH_MIN_CHARS)
    return "\n\n".join(paragraphs)


def fetch_press_conf_paragraphs(date_str: str) -> list[str]:
    """Fetch the press conference transcript PDF and return its body paragraphs.

    federalreserve.gov publishes Powell's press-conference transcripts as PDFs
    only; the matching .htm page is just a wrapper around the video player.
    """
    yyyymmdd = date_str.replace("-", "")
    url = f"https://www.federalreserve.gov/mediacenter/files/FOMCpresconf{yyyymmdd}.pdf"
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    return extract_paragraphs_from_pdf(response.content, min_chars=_PARAGRAPH_MIN_CHARS)


_TAG_RE = re.compile(r"<[^>]+>")
_PARA_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)
# fed.gov wraps the real release body in <div id="article"> — everything outside
# is .gov banner / chrome boilerplate that we don't want in the corpus.
_ARTICLE_RE = re.compile(r'<div\b[^>]*\bid=["\']article["\'][^>]*>(.*)', re.IGNORECASE | re.DOTALL)
# Per-page chrome that pypdf concatenates into the extracted text. We strip
# the "Page N of M" line outright and also the running footer that fed.gov
# stamps on every transcript page (date + "Chair Powell's Press Conference
# FINAL"), which otherwise leaks into the end of each paragraph.
_PDF_PAGE_HEADER_RE = re.compile(r"^[^\n]*Page \d+ of \d+\s*$", re.MULTILINE)
# Anchored to non-whitespace boundaries so that the surrounding paragraph-break
# newlines are preserved when we delete the footer.
_PDF_RUNNING_FOOTER_RE = re.compile(
    r"[A-Z][a-z]+\s+\d{1,2},\s+\d{4}\s+Chair\s+\S+(?:\s+\S+)?\s+Press\s+Conference\s+FINAL"
)


def extract_paragraphs_from_html(content: str, min_chars: int = 80) -> list[str]:
    """Naive <p>-tag extractor. Filters paragraphs shorter than min_chars.

    If the document contains a <div id="article"> (fed.gov release layout), only
    paragraphs inside that container are returned — the surrounding .gov banner
    and security-notice boilerplate is dropped.
    """
    article_match = _ARTICLE_RE.search(content)
    scope = article_match.group(1) if article_match else content
    paragraphs: list[str] = []
    for match in _PARA_RE.finditer(scope):
        inner = _TAG_RE.sub("", match.group(1))
        text = html.unescape(inner).strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) >= min_chars:
            paragraphs.append(text)
    return paragraphs


def extract_paragraphs_from_pdf(pdf_bytes: bytes, min_chars: int = 80) -> list[str]:
    """Extract paragraphs from a press-conference transcript PDF.

    Strips per-page headers like "Page 4 of 24" and splits on blank lines.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    raw = "\n".join(page.extract_text() or "" for page in reader.pages)
    cleaned = _PDF_PAGE_HEADER_RE.sub("", raw)
    cleaned = _PDF_RUNNING_FOOTER_RE.sub("", cleaned)
    paragraphs: list[str] = []
    for chunk in re.split(r"\n\s*\n", cleaned):
        text = re.sub(r"\s+", " ", chunk).strip()
        if len(text) >= min_chars:
            paragraphs.append(text)
    return paragraphs


def assign_paragraph_timestamps(
    paragraphs: list[str],
    start: datetime,
    span_minutes: int = 60,
) -> list[dict]:
    """Spread N paragraphs evenly across `span_minutes` starting at `start`."""
    if not paragraphs:
        return []
    if len(paragraphs) == 1:
        return [{"timestamp": start, "text": paragraphs[0]}]
    step = timedelta(minutes=span_minutes / (len(paragraphs) - 1))
    return [{"timestamp": start + step * i, "text": text} for i, text in enumerate(paragraphs)]


def write_statement_txt(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def write_press_conf_json(paragraphs_with_ts: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "paragraphs": [
            {"timestamp": p["timestamp"].isoformat(), "text": p["text"]} for p in paragraphs_with_ts
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch curated FOMC data for a given date.")
    parser.add_argument("date", help="ISO date, e.g. 2023-12-13")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    day = datetime.fromisoformat(args.date).replace(tzinfo=UTC)
    out = Path(args.out_dir)

    logger.info("Fetching Binance bars for %s on %s", args.symbol, args.date)
    bars = fetch_binance_bars(args.symbol, day)
    logger.info("  got %d bars", len(bars))
    write_bars_csv(bars, out / "bars.csv")

    logger.info("Fetching FOMC statement")
    statement = fetch_fomc_statement(args.date)
    write_statement_txt(statement, out / "statement.txt")

    logger.info("Fetching press conference transcript")
    paragraphs = fetch_press_conf_paragraphs(args.date)
    logger.info("  got %d paragraphs", len(paragraphs))
    presser_start = day.replace(
        hour=_PRESS_CONF_START_UTC_HOUR,
        minute=_PRESS_CONF_START_UTC_MINUTE,
        second=0,
        microsecond=0,
    )
    timestamped = assign_paragraph_timestamps(paragraphs, presser_start, span_minutes=60)
    write_press_conf_json(timestamped, out / "press_conf.json")

    logger.info("Done. Output in %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
