"""RealEpisode: real-market analogue of SimulatedEpisode.

Loads a curated dataset from disk (bars.csv + statement.txt + press_conf.json)
into a MarketTerrain plus a list of TextEvents. Same wrapping pattern as
SimulatedEpisode so the existing run_terrain_backtest pipeline composes.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from qts.models.base import (
    Bar,
    Catalyst,
    LiquidityLevel,
    SentimentLevel,
    Trend,
    VolLevel,
)
from qts.models.terrain import MacroRegime, MarketEvent, MarketTerrain
from qts.world.events import TextEvent

# The press release fires at 14:00 ET = 19:00 UTC for US FOMC days.
_FOMC_PRESS_RELEASE_UTC_HOUR = 19


@dataclass
class RealEpisode:
    """Real-data analogue of SimulatedEpisode."""

    terrain: MarketTerrain
    text_events: list[TextEvent]
    source: str  # e.g. "fomc:2023-12-13"

    @classmethod
    def from_disk(
        cls,
        root: Path,
        symbol: str,
        source: str,
    ) -> RealEpisode:
        """Load bars.csv + statement.txt + press_conf.json from `root`."""
        root = Path(root)
        bars = _load_bars(root / "bars.csv", symbol)
        statement_text = (root / "statement.txt").read_text(encoding="utf-8").strip()
        press_conf_paragraphs = _load_press_conf(root / "press_conf.json")

        if not bars:
            raise ValueError(f"no bars found in {root}/bars.csv")

        first_bar = bars[0]
        last_bar = bars[-1]
        statement_timestamp = first_bar.timestamp.replace(
            hour=_FOMC_PRESS_RELEASE_UTC_HOUR, minute=0, second=0, microsecond=0
        )

        # Build TextEvents
        statement_event = TextEvent(
            timestamp=statement_timestamp,
            source="fed_press_release",
            persona=None,
            text=statement_text,
            metadata={"event_kind": "FOMC", "real": True},
        )
        press_conf_events = [
            TextEvent(
                timestamp=p["timestamp"],
                source="powell_press_conf",
                persona="powell",
                text=p["text"],
                metadata={"event_kind": "FOMC", "real": True},
            )
            for p in press_conf_paragraphs
        ]
        text_events = sorted(
            [statement_event, *press_conf_events],
            key=lambda e: e.timestamp,
        )

        # Build a placeholder MacroRegime — real-data tests don't depend on regime tags.
        regime = MacroRegime(
            trend=Trend.SIDEWAYS,
            volatility=VolLevel.HIGH,
            liquidity=LiquidityLevel.ABUNDANT,
            sentiment=SentimentLevel.NEUTRAL,
            catalyst=Catalyst.MACRO_EVENT,
            expected_drift=0.0,
            expected_vol=0.01,
            correlation_regime=0.5,
            scenario_description=f"real:{source}",
        )

        # Mirror text events as MarketEvents on the calendar (parallel to Phase 8 v1 SimulatedEpisode).
        calendar = [
            MarketEvent(
                timestamp=e.timestamp,
                event_type=f"text:{e.source}",
                description=e.text,
                impact_magnitude=0.0,
            )
            for e in text_events
        ]

        terrain = MarketTerrain(
            name=f"real:{source}",
            symbol=symbol,
            start=first_bar.timestamp,
            end=last_bar.timestamp,
            regime=regime,
            bars=bars,
            event_calendar=calendar,
        )

        return cls(terrain=terrain, text_events=text_events, source=source)


def _load_bars(path: Path, symbol: str) -> list[Bar]:
    bars: list[Bar] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bars.append(
                Bar(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    symbol=symbol,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return bars


def _load_press_conf(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    paragraphs = raw.get("paragraphs", [])
    out: list[dict] = []
    for p in paragraphs:
        out.append(
            {
                "timestamp": datetime.fromisoformat(p["timestamp"]),
                "text": p["text"],
            }
        )
    return out
