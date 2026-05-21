"""Tests for RealEpisode and its disk loader."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def _write_curated(root: Path) -> None:
    """Write a minimal curated dataset (3 bars + 1 statement + 1 press-conf paragraph)."""
    root.mkdir(parents=True, exist_ok=True)
    bars_csv = "\n".join(
        [
            "timestamp,open,high,low,close,volume",
            "2023-12-13T00:00:00+00:00,40000.0,40100.0,39950.0,40050.0,12.3",
            "2023-12-13T00:01:00+00:00,40050.0,40080.0,40010.0,40060.0,9.1",
            "2023-12-13T00:02:00+00:00,40060.0,40090.0,40030.0,40075.0,7.8",
        ]
    )
    (root / "bars.csv").write_text(bars_csv + "\n", encoding="utf-8")

    (root / "statement.txt").write_text(
        "The Federal Open Market Committee decided to maintain the target range.",
        encoding="utf-8",
    )

    press_conf = {
        "paragraphs": [
            {
                "timestamp": "2023-12-13T19:30:00+00:00",
                "text": "We see growing evidence that the disinflationary process is on track.",
            },
        ],
    }
    (root / "press_conf.json").write_text(json.dumps(press_conf), encoding="utf-8")


def test_from_disk_loads_bars(tmp_path: Path) -> None:  # T-REP-1
    from qts.data.real_episode import RealEpisode

    root = tmp_path / "fomc_test"
    _write_curated(root)

    ep = RealEpisode.from_disk(root, symbol="BTCUSDT", source="fomc:test")
    assert len(ep.terrain.bars) == 3
    first = ep.terrain.bars[0]
    assert first.timestamp == datetime(2023, 12, 13, 0, 0, tzinfo=UTC)
    assert first.open == 40_000.0
    assert first.close == 40_050.0
    assert first.symbol == "BTCUSDT"


def test_from_disk_loads_text_events(tmp_path: Path) -> None:  # T-REP-2
    from qts.data.real_episode import RealEpisode

    root = tmp_path / "fomc_test"
    _write_curated(root)

    ep = RealEpisode.from_disk(root, symbol="BTCUSDT", source="fomc:test")
    # Two events: the statement (timestamped at the press release moment) and the press-conf paragraph.
    assert len(ep.text_events) == 2

    statement = next(e for e in ep.text_events if e.source == "fed_press_release")
    assert "Federal Open Market Committee" in statement.text

    presser = next(e for e in ep.text_events if e.source == "powell_press_conf")
    assert presser.timestamp == datetime(2023, 12, 13, 19, 30, tzinfo=UTC)


def test_from_disk_attaches_source_and_terrain(tmp_path: Path) -> None:  # T-REP-3
    from qts.data.real_episode import RealEpisode

    root = tmp_path / "fomc_test"
    _write_curated(root)

    ep = RealEpisode.from_disk(root, symbol="BTCUSDT", source="fomc:2023-12-13")
    assert ep.source == "fomc:2023-12-13"
    assert ep.terrain.symbol == "BTCUSDT"
    assert ep.terrain.name.startswith("real:fomc")
