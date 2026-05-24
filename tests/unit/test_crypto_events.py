"""T-CRYPTO-EVENT-*: contagion-event loader (timestamp tz-normalised, sorted, severity parsed)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from qts.propagation.crypto.events import ContagionEvent, load_contagion_events


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "events.yaml"
    p.write_text(
        "events:\n"
        "  - {source_token: CRV, timestamp: '2023-07-30T12:00:00', event_type: exploit, usd_severity: 70e6}\n"
        "  - {source_token: FTT, timestamp: '2022-11-08T00:00:00+00:00', event_type: insolvency, usd_severity: 8e9}\n"
    )
    return p


def test_loads_sorted_tz_aware(tmp_path: Path) -> None:  # T-CRYPTO-EVENT-1
    evs = load_contagion_events(_write(tmp_path))
    assert [e.source_token for e in evs] == ["FTT", "CRV"]  # sorted by timestamp ascending
    assert all(isinstance(e, ContagionEvent) for e in evs)
    assert evs[0].timestamp == datetime(2022, 11, 8, tzinfo=UTC)
    assert evs[1].timestamp.tzinfo is UTC  # naive input got UTC-stamped
    assert evs[1].event_type == "exploit"


def test_severity_parsed_as_float(tmp_path: Path) -> None:  # T-CRYPTO-EVENT-2
    evs = load_contagion_events(_write(tmp_path))
    ftt = next(e for e in evs if e.source_token == "FTT")
    assert ftt.usd_severity == 8e9 and isinstance(ftt.usd_severity, float)
