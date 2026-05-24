"""Curated contagion events: the do()-intervention sources. Timestamps are FIRST PUBLIC DISCLOSURE
(never post-mortem) to avoid lookahead (spec §4, §10)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ContagionEvent:
    source_token: str
    timestamp: datetime  # first public disclosure, tz-aware (UTC)
    event_type: str  # hack | exploit | depeg | insolvency | liquidation
    usd_severity: float


def load_contagion_events(yaml_path: Path) -> list[ContagionEvent]:
    raw = yaml.safe_load(Path(yaml_path).read_text())
    out: list[ContagionEvent] = []
    for item in raw["events"]:
        ts = datetime.fromisoformat(str(item["timestamp"]))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        out.append(
            ContagionEvent(
                source_token=str(item["source_token"]),
                timestamp=ts,
                event_type=str(item["event_type"]),
                usd_severity=float(item["usd_severity"]),
            )
        )
    return sorted(out, key=lambda e: e.timestamp)
