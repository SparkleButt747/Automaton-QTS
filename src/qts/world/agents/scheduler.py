"""SchedulerAgent — fires the scheduled MacroEvents at the right tick."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from qts.world.agents.base import AgentContext, AgentFill, AgentOrder
from qts.world.events import MacroEvent, TextEvent


@dataclass
class SchedulerAgent:
    """Clock-driven event publisher.

    v1 only fires one FOMC event. v2 adds CPI, NFP, etc. by extending the
    config + on_tick check.
    """

    agent_id: str
    fomc_at: datetime
    fomc_expected_rate: float
    fomc_actual_rate: float
    fomc_release_text: str = "The Committee announced its decision on the federal funds rate."
    _fired: bool = False

    def surprise(self) -> float:
        return self.fomc_actual_rate - self.fomc_expected_rate

    def surprise_bucket(self) -> str:
        s = self.surprise()
        if s > 0.05:
            return "hawkish"
        if s < -0.05:
            return "dovish"
        return "neutral"

    def on_tick(self, ctx: AgentContext) -> list[AgentOrder | TextEvent | MacroEvent]:
        if self._fired or ctx.now < self.fomc_at:
            return []
        self._fired = True
        text = TextEvent(
            timestamp=ctx.now,
            source="fed_press_release",
            persona=None,
            text=self.fomc_release_text,
            metadata={"surprise_bucket": self.surprise_bucket()},
        )
        return [
            MacroEvent(
                timestamp=ctx.now,
                kind="fomc",
                expected=self.fomc_expected_rate,
                actual=self.fomc_actual_rate,
                surprise=self.surprise(),
                text_event=text,
            ),
        ]

    def on_event(
        self, event: TextEvent | MacroEvent, ctx: AgentContext
    ) -> list[AgentOrder | TextEvent | MacroEvent]:
        return []

    def on_fill(self, fill: AgentFill, ctx: AgentContext) -> None:
        return None
