"""InventoryAwareMM — v1 market maker.

Quotes a single bid/ask each tick. Spread = base + k * recent realised vol.
Inventory aware: when net long, shifts mid down to attract sellers (and
vice-versa). Inventory updates from its own fills.

Quote is published as a TextEvent with source="mm" and bid/ask in metadata
so the agent_sim loop can route it to the SimpleOrderBook uniformly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qts.world.agents.base import AgentContext, AgentFill, AgentOrder
from qts.world.events import MacroEvent, TextEvent


@dataclass
class InventoryAwareMM:
    """Single-MM market maker for v1."""

    agent_id: str
    base_spread_bps: float
    vol_widen_k: float
    inventory_lean_bps: float
    quote_size: float
    max_inventory: float = 5.0
    vol_lookback: int = 10

    inventory: float = 0.0
    sentiment_drift_bps: float = 0.0  # set externally by the sim loop on text events

    def _realised_vol(self, ctx: AgentContext) -> float:
        bars = ctx.recent_bars[-self.vol_lookback :]
        if len(bars) < 2:
            return 0.0
        rets: list[float] = []
        for prev, cur in zip(bars[:-1], bars[1:], strict=False):
            if prev.close <= 0:
                continue
            rets.append((cur.close - prev.close) / prev.close)
        if not rets:
            return 0.0
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
        return math.sqrt(var)

    def _build_quote(self, ctx: AgentContext) -> tuple[float, float]:
        mid = ctx.last_price * (1.0 + self.sentiment_drift_bps / 1e4)
        spread_bps = self.base_spread_bps + self.vol_widen_k * self._realised_vol(ctx) * 1e4
        half = mid * spread_bps / 2.0 / 1e4

        # Inventory lean: positive inventory => shift mid DOWN (encourage SELL flow into bid)
        lean_frac = max(-1.0, min(1.0, self.inventory / max(self.max_inventory, 1e-9)))
        lean = mid * lean_frac * (self.inventory_lean_bps / 1e4)
        mid_leaned = mid - lean

        return (mid_leaned - half, mid_leaned + half)

    def on_tick(self, ctx: AgentContext) -> list[AgentOrder | TextEvent | MacroEvent]:
        bid, ask = self._build_quote(ctx)
        quote_event = TextEvent(
            timestamp=ctx.now,
            source="mm",
            persona=None,
            text=f"quote bid={bid:.2f} ask={ask:.2f}",
            metadata={"bid": bid, "ask": ask, "size": self.quote_size},
        )
        return [quote_event]

    def on_event(
        self, event: TextEvent | MacroEvent, ctx: AgentContext
    ) -> list[AgentOrder | TextEvent | MacroEvent]:
        # v1: MM does not react to text directly. The sim loop is responsible
        # for setting sentiment_drift_bps externally on regime-changing events.
        return []

    def on_fill(self, fill: AgentFill, ctx: AgentContext) -> None:
        # MM is the counterparty: BUY fill means anon bought from MM => MM sold => inventory--
        # But here on_fill represents an MM-side fill: the MM "filled" via providing the quote.
        # Convention: the sim loop reports the COUNTERPARTY side from the MM's view.
        if fill.side == "BUY":
            self.inventory += fill.quantity
        else:
            self.inventory -= fill.quantity
