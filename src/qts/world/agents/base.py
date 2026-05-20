"""WorldAgent protocol + AgentContext (shared per-tick state).

Agents are plain Python objects, not Nautilus actors. Stage 1 of the
simulator iterates the clock and dispatches each tick to every agent
via on_tick. Events are dispatched via on_event. Fills come from the
SimpleOrderBook via on_fill.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from qts.models.base import Bar
from qts.world.events import MacroEvent, TextEvent


@dataclass
class AgentContext:
    """Read-only context passed to every agent each tick."""

    now: datetime
    last_price: float
    rng: random.Random
    recent_bars: list[Bar] = field(default_factory=list)
    recent_text: list[TextEvent] = field(default_factory=list)


@dataclass
class AgentOrder:
    """Order intent emitted by an agent. Translated to OrderLogEntry by the order book."""

    agent_id: str
    side: str  # "BUY" or "SELL"
    quantity: float
    price: float | None = None  # None => market order


@dataclass
class AgentFill:
    """A fill reported back to the agent that placed the order."""

    timestamp: datetime
    side: str
    quantity: float
    price: float


@runtime_checkable
class WorldAgent(Protocol):
    """Every world agent (scheduler, persona, anon, MM) implements this."""

    agent_id: str

    def on_tick(self, ctx: AgentContext) -> list[AgentOrder | TextEvent | MacroEvent]:
        """Called every tick. May return any number of orders/events to publish."""
        ...

    def on_event(
        self, event: TextEvent | MacroEvent, ctx: AgentContext
    ) -> list[AgentOrder | TextEvent | MacroEvent]:
        """Called when another agent publishes an event. May return reactions."""
        ...

    def on_fill(self, fill: AgentFill, ctx: AgentContext) -> None:
        """Called when one of the agent's orders fills."""
        ...
