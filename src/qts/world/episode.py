"""SimulatedEpisode — the output of a world-simulator run.

Wraps the existing MarketTerrain so all downstream consumers
(run_terrain_backtest, Optuna, perturbator) work unchanged. Adds the
rich per-agent metadata and order log that v1 surface for inspection.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from qts.models.terrain import MarketTerrain

if TYPE_CHECKING:
    from qts.world.events import TextEvent  # noqa: F401


@dataclass
class OrderLogEntry:
    """One submitted order's fate during the agent simulation."""

    timestamp: datetime
    agent_id: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float
    fate: str  # "FILLED", "REJECTED", "PARTIAL", etc.


@dataclass
class AgentTrace:
    """Per-agent state timeline. Populated during stage 1."""

    agent_id: str
    style: str  # free-form: "sentiment", "trend", "mean_revert", "mm", "persona", "scheduler"
    sentiment_readings: list[tuple[datetime, float]] = field(default_factory=list)
    inventory_readings: list[tuple[datetime, float]] = field(default_factory=list)
    orders: list[OrderLogEntry] = field(default_factory=list)


@dataclass
class SimulatedEpisode:
    """One full run of the world simulator. Wraps a MarketTerrain."""

    terrain: MarketTerrain
    scenario_name: str
    seed: int
    text_events: list[TextEvent] = field(default_factory=list)
    agent_traces: dict[str, AgentTrace] = field(default_factory=dict)
    llm_corpus_refs: list[str] = field(default_factory=list)
    order_log: list[OrderLogEntry] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialise the episode metadata to JSON.

        The wrapped MarketTerrain is summarised (name, symbol, start/end,
        bar count) rather than dumped in full — bars belong in Parquet,
        not in this trace.
        """
        payload: dict[str, object] = {
            "scenario_name": self.scenario_name,
            "seed": self.seed,
            "terrain": {
                "name": self.terrain.name,
                "symbol": self.terrain.symbol,
                "start": self.terrain.start.isoformat(),
                "end": self.terrain.end.isoformat(),
                "bar_count": len(self.terrain.bars),
            },
            "llm_corpus_refs": list(self.llm_corpus_refs),
            "order_log": [_order_to_dict(o) for o in self.order_log],
            "agent_traces": {
                aid: {
                    "style": tr.style,
                    "sentiment_readings": [[t.isoformat(), v] for t, v in tr.sentiment_readings],
                    "inventory_readings": [[t.isoformat(), v] for t, v in tr.inventory_readings],
                    "orders": [_order_to_dict(o) for o in tr.orders],
                }
                for aid, tr in self.agent_traces.items()
            },
        }
        return json.dumps(payload)


def _order_to_dict(o: OrderLogEntry) -> dict[str, object]:
    d = asdict(o)
    d["timestamp"] = o.timestamp.isoformat()
    return d
