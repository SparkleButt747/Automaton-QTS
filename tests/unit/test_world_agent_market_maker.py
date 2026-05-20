"""Tests for qts.world.agents.market_maker."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from qts.models.base import Bar
from qts.world.agents.base import AgentContext, AgentFill


def _ctx(last_price: float, recent_closes: list[float]) -> AgentContext:
    start = datetime(2025, 3, 19, 14, 0, tzinfo=UTC)
    bars = [
        Bar(
            symbol="BTCUSDT",
            timestamp=start + timedelta(minutes=i),
            open=c,
            high=c,
            low=c,
            close=c,
            volume=1.0,
        )
        for i, c in enumerate(recent_closes)
    ]
    return AgentContext(
        now=start + timedelta(minutes=len(recent_closes)),
        last_price=last_price,
        rng=random.Random(0),
        recent_bars=bars,
        recent_text=[],
    )


def test_mm_quotes_symmetric_around_mid_with_no_inventory() -> None:  # T-WMM-1
    from qts.world.agents.market_maker import InventoryAwareMM

    from qts.world.events import TextEvent

    mm = InventoryAwareMM(
        agent_id="mm",
        base_spread_bps=10.0,
        vol_widen_k=2.0,
        inventory_lean_bps=5.0,
        quote_size=1.0,
    )
    out = mm.on_tick(_ctx(last_price=30000.0, recent_closes=[30000.0] * 5))
    # First output is the quote update; encoded as a TextEvent with source=mm
    quotes = [o for o in out if isinstance(o, TextEvent) and o.source == "mm"]
    assert len(quotes) == 1
    bid = quotes[0].metadata["bid"]
    ask = quotes[0].metadata["ask"]
    mid = (bid + ask) / 2
    assert abs(mid - 30000.0) < 0.01
    spread_bps = (ask - bid) / mid * 1e4
    assert 8.0 < spread_bps < 12.0


def test_mm_widens_spread_with_recent_vol() -> None:  # T-WMM-2
    from qts.world.agents.market_maker import InventoryAwareMM

    from qts.world.events import TextEvent

    mm = InventoryAwareMM(
        agent_id="mm",
        base_spread_bps=10.0,
        vol_widen_k=5.0,
        inventory_lean_bps=5.0,
        quote_size=1.0,
    )
    # Calm bars produce a tight spread
    calm_out = mm.on_tick(_ctx(last_price=30000.0, recent_closes=[30000.0] * 10))
    calm_quote = next(o for o in calm_out if isinstance(o, TextEvent) and o.source == "mm")
    calm_spread = calm_quote.metadata["ask"] - calm_quote.metadata["bid"]

    # Volatile bars produce a wider spread
    volatile = [30000.0, 30150.0, 29850.0, 30200.0, 29800.0, 30100.0, 29900.0]
    mm2 = InventoryAwareMM(
        agent_id="mm",
        base_spread_bps=10.0,
        vol_widen_k=5.0,
        inventory_lean_bps=5.0,
        quote_size=1.0,
    )
    vol_out = mm2.on_tick(_ctx(last_price=30000.0, recent_closes=volatile))
    vol_quote = next(o for o in vol_out if isinstance(o, TextEvent) and o.source == "mm")
    vol_spread = vol_quote.metadata["ask"] - vol_quote.metadata["bid"]

    assert vol_spread > calm_spread


def test_mm_leans_quotes_when_long() -> None:  # T-WMM-3
    from qts.world.agents.market_maker import InventoryAwareMM

    from qts.world.events import TextEvent

    mm = InventoryAwareMM(
        agent_id="mm",
        base_spread_bps=10.0,
        vol_widen_k=2.0,
        inventory_lean_bps=20.0,  # exaggerate
        quote_size=1.0,
        max_inventory=1.0,
    )
    # Bought 0.5 BTC (half max inventory) — long
    mm.on_fill(
        AgentFill(
            timestamp=datetime(2025, 3, 19, tzinfo=UTC), side="BUY", quantity=0.5, price=30000.0
        ),
        _ctx(30000.0, [30000.0] * 5),
    )
    out = mm.on_tick(_ctx(30000.0, [30000.0] * 5))
    quote = next(o for o in out if isinstance(o, TextEvent) and o.source == "mm")
    # When long, the MM wants to sell — its bid/ask both shift DOWN to attract sellers
    flat_mid = 30000.0
    leaned_mid = (quote.metadata["bid"] + quote.metadata["ask"]) / 2
    assert leaned_mid < flat_mid


def test_mm_inventory_updates_on_fill() -> None:  # T-WMM-4
    from qts.world.agents.market_maker import InventoryAwareMM

    mm = InventoryAwareMM(
        agent_id="mm",
        base_spread_bps=10.0,
        vol_widen_k=2.0,
        inventory_lean_bps=5.0,
        quote_size=1.0,
    )
    assert mm.inventory == 0.0
    mm.on_fill(
        AgentFill(
            timestamp=datetime(2025, 3, 19, tzinfo=UTC), side="BUY", quantity=0.3, price=30000.0
        ),
        _ctx(30000.0, []),
    )
    assert mm.inventory == 0.3
    mm.on_fill(
        AgentFill(
            timestamp=datetime(2025, 3, 19, tzinfo=UTC), side="SELL", quantity=0.1, price=30100.0
        ),
        _ctx(30100.0, []),
    )
    assert abs(mm.inventory - 0.2) < 1e-9
