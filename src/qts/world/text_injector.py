"""TextEventInjector — Nautilus actor that fires pre-recorded TextEvents.

Plugged into the Nautilus BacktestEngine alongside the QTSStrategy actor.
Holds the (timestamp, event) list from stage 1; on each bar it checks for
events that should fire by the bar's timestamp and forwards them via the
QTSStrategy actor's on_text_event method.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from qts.world.events import TextEvent


@dataclass
class PendingTextEvent:
    when: datetime
    event: TextEvent


class TextEventInjector:
    """Pure-Python injector — not a Nautilus actor.

    The runner walks bars sequentially and calls dispatch_up_to(bar.timestamp)
    after each bar. This avoids a tight coupling to Nautilus's actor lifecycle
    and keeps the v1 implementation simple.
    """

    def __init__(self, events: list[TextEvent]) -> None:
        self._pending = sorted(events, key=lambda e: e.timestamp)
        self._cursor = 0

    def dispatch_up_to(self, now: datetime, sink) -> None:  # noqa: ANN001
        """Forward every queued event with timestamp <= now to sink(event)."""
        while self._cursor < len(self._pending) and self._pending[self._cursor].timestamp <= now:
            sink(self._pending[self._cursor])
            self._cursor += 1
