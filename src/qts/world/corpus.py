"""PersonaCorpus — pre-generated persona reactions, keyed by (persona, event, regime).

v1 ships in corpus mode only: load YAML, sample with seeded RNG.
v1.5+ live_cached mode will add LLM-backed cache misses (deferred).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PersonaCorpus:
    """In-memory store of pre-generated persona reactions.

    Schema (YAML):

        <persona>:
          <event_kind>:
            <bucket>:
              - "<statement 1>"
              - "<statement 2>"
              ...
    """

    entries: dict[str, dict[str, dict[str, list[str]]]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> PersonaCorpus:
        with path.open("r", encoding="utf-8") as fh:
            raw: Any = yaml.safe_load(fh) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"corpus YAML root must be a dict, got {type(raw).__name__}")
        return cls(entries=raw)

    def lookup(self, persona: str, event: str, bucket: str) -> list[str]:
        """Return the list of statements for (persona, event, bucket), or []."""
        return list(self.entries.get(persona, {}).get(event, {}).get(bucket, []))

    def sample(self, persona: str, event: str, bucket: str, rng: random.Random) -> str:
        """Deterministically pick one statement; "" if no entries exist."""
        choices = self.lookup(persona, event, bucket)
        if not choices:
            return ""
        return rng.choice(choices)

    @staticmethod
    def key(persona: str, event: str, bucket: str) -> str:
        """Stable string key for this triple (used in llm_corpus_refs)."""
        return f"{persona}:{event}:{bucket}"
