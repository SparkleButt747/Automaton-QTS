"""Event types emitted by the world simulator.

TextEvent carries any textual signal (persona tweets, press releases, retail
posts). MacroEvent represents a scheduled macro data release; it may carry
an accompanying TextEvent for the release statement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class TextEvent:
    """A text emission attributable to a source (persona, agency, anon)."""

    timestamp: datetime
    source: str
    persona: str | None
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MacroEvent:
    """A scheduled macro data release with a surprise component."""

    timestamp: datetime
    kind: str
    expected: float
    actual: float
    surprise: float
    text_event: TextEvent | None = None
