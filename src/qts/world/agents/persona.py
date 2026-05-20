"""PersonaAgent — emits in-character TextEvents from a corpus on macro events.

v1 implements only Powell; the agent is parameterised so v2 can add Trump,
Musk, congressional figures by varying persona_id + corpus YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qts.world.agents.base import AgentContext, AgentFill, AgentOrder
from qts.world.corpus import PersonaCorpus
from qts.world.events import MacroEvent, TextEvent


@dataclass
class PersonaAgent:
    """A named-persona agent that emits text reactions to macro events.

    The agent is silent on on_tick (no spontaneous tweets in v1) and reacts
    to MacroEvent.kind == its trigger_kind by sampling one statement from
    the corpus at (persona_id, trigger_kind, surprise_bucket).
    """

    agent_id: str
    persona_id: str
    corpus: PersonaCorpus
    surprise_bucket: str  # "hawkish", "dovish", "neutral"
    trigger_kind: str = "fomc"
    consumed_corpus_keys: list[str] = field(default_factory=list)

    def on_tick(self, ctx: AgentContext) -> list[AgentOrder | TextEvent | MacroEvent]:
        return []

    def on_event(
        self, event: TextEvent | MacroEvent, ctx: AgentContext
    ) -> list[AgentOrder | TextEvent | MacroEvent]:
        if not isinstance(event, MacroEvent):
            return []
        if event.kind != self.trigger_kind:
            return []

        text = self.corpus.sample(
            self.persona_id,
            self.trigger_kind,
            self.surprise_bucket,
            rng=ctx.rng,
        )
        if not text:
            return []

        key = PersonaCorpus.key(self.persona_id, self.trigger_kind, self.surprise_bucket)
        self.consumed_corpus_keys.append(key)

        return [
            TextEvent(
                timestamp=ctx.now,
                source=self.persona_id,
                persona=self.persona_id,
                text=text,
                metadata={
                    "surprise_bucket": self.surprise_bucket,
                    "trigger_kind": self.trigger_kind,
                    "corpus_key": key,
                },
            )
        ]

    def on_fill(self, fill: AgentFill, ctx: AgentContext) -> None:
        return None
