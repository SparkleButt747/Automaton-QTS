"""AnonRetailAgent - configurable retail order-flow generators.

Three v1 styles:
  - "sentiment":   reads TextEvents via VADER+regex, places market order in the matching direction
  - "trend":       buys when last N bars trended up by > threshold; mirrors for down
  - "mean_revert": fades a last-bar return > threshold

All anons use the crowd-grade SentimentScorer for text - never Qwen.
That asymmetry is the alpha hypothesis (see grill log).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qts.world.agents.base import AgentContext, AgentFill, AgentOrder
from qts.world.events import MacroEvent, TextEvent
from qts.world.sentiment import SentimentScorer

_VALID_STYLES = {"sentiment", "trend", "mean_revert"}


@dataclass
class AnonRetailAgent:
    """One anon retail trader. Style + parameters drive behaviour."""

    agent_id: str
    style: str
    aggressiveness: float = 1.0
    base_order_size: float = 0.05  # base units of the asset

    # trend params
    trend_lookback: int = 5
    trend_threshold: float = 0.001  # 10bps over the lookback triggers

    # mean_revert params
    revert_lookback: int = 5
    revert_threshold: float = 0.005  # 50bps last-bar move triggers

    # sentiment params
    sentiment_threshold: float = 0.1

    scorer: SentimentScorer = field(default_factory=SentimentScorer)
    sentiment_readings: list[tuple[object, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.style not in _VALID_STYLES:
            raise ValueError(f"unknown style {self.style!r}; expected one of {_VALID_STYLES}")

    def _order_size(self) -> float:
        return self.base_order_size * self.aggressiveness

    def on_tick(self, ctx: AgentContext) -> list[AgentOrder | TextEvent | MacroEvent]:
        if self.style == "trend":
            return self._on_tick_trend(ctx)
        if self.style == "mean_revert":
            return self._on_tick_mean_revert(ctx)
        # sentiment-style anons only act on on_event
        return []

    def _on_tick_trend(self, ctx: AgentContext) -> list[AgentOrder | TextEvent | MacroEvent]:
        if len(ctx.recent_bars) < self.trend_lookback:
            return []
        window = ctx.recent_bars[-self.trend_lookback :]
        ret = (window[-1].close - window[0].close) / window[0].close if window[0].close else 0.0
        if ret > self.trend_threshold:
            return [AgentOrder(self.agent_id, "BUY", self._order_size())]
        if ret < -self.trend_threshold:
            return [AgentOrder(self.agent_id, "SELL", self._order_size())]
        return []

    def _on_tick_mean_revert(self, ctx: AgentContext) -> list[AgentOrder | TextEvent | MacroEvent]:
        if len(ctx.recent_bars) < 2:
            return []
        prev = ctx.recent_bars[-2]
        last = ctx.recent_bars[-1]
        if prev.close <= 0:
            return []
        ret = (last.close - prev.close) / prev.close
        if ret > self.revert_threshold:
            return [AgentOrder(self.agent_id, "SELL", self._order_size())]
        if ret < -self.revert_threshold:
            return [AgentOrder(self.agent_id, "BUY", self._order_size())]
        return []

    def on_event(
        self, event: TextEvent | MacroEvent, ctx: AgentContext
    ) -> list[AgentOrder | TextEvent | MacroEvent]:
        if self.style != "sentiment" or not isinstance(event, TextEvent):
            return []
        score = self.scorer.score(event.text)
        self.sentiment_readings.append((ctx.now, score))
        if score > self.sentiment_threshold:
            return [AgentOrder(self.agent_id, "BUY", self._order_size())]
        if score < -self.sentiment_threshold:
            return [AgentOrder(self.agent_id, "SELL", self._order_size())]
        return []

    def on_fill(self, fill: AgentFill, ctx: AgentContext) -> None:
        return None
