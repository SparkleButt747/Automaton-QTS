"""Stage 1: the agent simulation loop.

Runs the WorldAgent roster on the SimulatedClock, routes their outputs
through the SimpleOrderBook, and aggregates trades into 1m Bars.

Returns AgentSimResult containing bars, text_events, macro_events, and
the per-agent traces. Stage 2 (the strategy backtest) consumes these.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime

from qts.models.base import Bar
from qts.world.agents.anon import AnonRetailAgent
from qts.world.agents.base import AgentContext, AgentFill, AgentOrder, WorldAgent
from qts.world.agents.market_maker import InventoryAwareMM
from qts.world.agents.persona import PersonaAgent
from qts.world.agents.scheduler import SchedulerAgent
from qts.world.bar_aggregator import BarAggregator
from qts.world.clock import SimulatedClock
from qts.world.corpus import PersonaCorpus
from qts.world.episode import AgentTrace, OrderLogEntry
from qts.world.events import MacroEvent, TextEvent
from qts.world.order_book import SimpleOrderBook
from qts.world.scenario import ScenarioConfig


@dataclass
class AgentSimResult:
    bars: list[Bar]
    text_events: list[TextEvent] = field(default_factory=list)
    macro_events: list[MacroEvent] = field(default_factory=list)
    order_log: list[OrderLogEntry] = field(default_factory=list)
    agent_traces: dict[str, AgentTrace] = field(default_factory=dict)
    consumed_corpus_keys: list[str] = field(default_factory=list)


def _build_agents(
    scenario: ScenarioConfig,
    corpus: PersonaCorpus,
    fomc_actual_rate: float,
) -> tuple[SchedulerAgent, PersonaAgent, list[AnonRetailAgent], InventoryAwareMM]:
    scheduler = SchedulerAgent(
        agent_id="scheduler",
        fomc_at=scenario.fomc_announcement_at,
        fomc_expected_rate=scenario.fomc_expected_rate,
        fomc_actual_rate=fomc_actual_rate,
    )
    persona = PersonaAgent(
        agent_id=scenario.powell_persona_id,
        persona_id=scenario.powell_persona_id,
        corpus=corpus,
        surprise_bucket=scheduler.surprise_bucket(),
    )
    anons = [
        AnonRetailAgent(
            agent_id=cfg.agent_id,
            style=cfg.style,
            aggressiveness=cfg.aggressiveness,
        )
        for cfg in scenario.anon_agents
    ]
    mm = InventoryAwareMM(
        agent_id="mm",
        base_spread_bps=scenario.mm_base_spread_bps,
        vol_widen_k=scenario.mm_vol_widen_k,
        inventory_lean_bps=5.0,
        quote_size=1.0,
    )
    return scheduler, persona, anons, mm


def run_agent_sim(
    scenario: ScenarioConfig,
    corpus: PersonaCorpus,
    seed: int,
    fomc_actual_rate: float,
) -> AgentSimResult:
    """Step the clock tick by tick, route agent outputs through the book."""
    master_rng = random.Random(seed)  # noqa: S311 - deterministic sim, not crypto
    scheduler, persona, anons, mm = _build_agents(scenario, corpus, fomc_actual_rate)

    # Each agent gets its own RNG derived from the master seed
    anon_rngs = {
        a.agent_id: random.Random(master_rng.randint(0, 2**31))  # noqa: S311
        for a in anons
    }
    persona_rng = random.Random(master_rng.randint(0, 2**31))  # noqa: S311
    mm_rng = random.Random(master_rng.randint(0, 2**31))  # noqa: S311

    clock = SimulatedClock(start=scenario.start, end=scenario.end, tick=scenario.tick)
    book = SimpleOrderBook()
    aggregator = BarAggregator(symbol=scenario.symbol, tick=scenario.tick)
    # Prime the aggregator so empty ticks produce carry-bars from scenario.start.
    aggregator.add_trade(scenario.start, price=scenario.starting_price, qty=0.0)

    last_price = scenario.starting_price
    recent_bars: list[Bar] = []
    recent_text: list[TextEvent] = []

    result = AgentSimResult(bars=[])
    result.agent_traces = {
        scheduler.agent_id: AgentTrace(agent_id=scheduler.agent_id, style="scheduler"),
        persona.agent_id: AgentTrace(agent_id=persona.agent_id, style="persona"),
        mm.agent_id: AgentTrace(agent_id=mm.agent_id, style="mm"),
        **{a.agent_id: AgentTrace(agent_id=a.agent_id, style=a.style) for a in anons},
    }

    def _ctx_for(rng: random.Random, now: datetime) -> AgentContext:
        return AgentContext(
            now=now,
            last_price=last_price,
            rng=rng,
            recent_bars=list(recent_bars[-30:]),
            recent_text=list(recent_text[-30:]),
        )

    def _handle_outputs(
        outputs: list[AgentOrder | TextEvent | MacroEvent],
        emitter: object,
        now: datetime,
    ) -> None:
        nonlocal last_price
        for item in outputs:
            if isinstance(item, AgentOrder):
                fill_trade = book.market_order(side=item.side, quantity=item.quantity)
                if fill_trade is not None:
                    last_price = fill_trade.price
                    aggregator.add_trade(now, price=fill_trade.price, qty=fill_trade.quantity)
                    log = OrderLogEntry(
                        timestamp=now,
                        agent_id=item.agent_id,
                        side=item.side,
                        quantity=fill_trade.quantity,
                        price=fill_trade.price,
                        fate="FILLED",
                    )
                    result.order_log.append(log)
                    if item.agent_id in result.agent_traces:
                        result.agent_traces[item.agent_id].orders.append(log)
                    # Report fill back to MM (counterparty side from MM's view)
                    mm_side = "SELL" if item.side == "BUY" else "BUY"
                    mm.on_fill(
                        AgentFill(
                            timestamp=now,
                            side=mm_side,
                            quantity=fill_trade.quantity,
                            price=fill_trade.price,
                        ),
                        _ctx_for(mm_rng, now),
                    )
                else:
                    result.order_log.append(
                        OrderLogEntry(
                            timestamp=now,
                            agent_id=item.agent_id,
                            side=item.side,
                            quantity=item.quantity,
                            price=0.0,
                            fate="REJECTED",
                        )
                    )
            elif isinstance(item, TextEvent):
                # MM quote events update the book; others fan out
                if item.source == "mm":
                    book.set_quote(
                        bid=float(item.metadata["bid"]),
                        ask=float(item.metadata["ask"]),
                        size=float(item.metadata.get("size", 1.0)),
                    )
                else:
                    recent_text.append(item)
                    result.text_events.append(item)
                    if hasattr(emitter, "consumed_corpus_keys"):
                        result.consumed_corpus_keys.extend(emitter.consumed_corpus_keys)
                    # Fan out to anons + persona via on_event
                    receivers: list[tuple[WorldAgent, random.Random]] = [
                        *((a, anon_rngs[a.agent_id]) for a in anons),
                        (persona, persona_rng),
                    ]
                    for receiver, rng_ in receivers:
                        nested = receiver.on_event(item, _ctx_for(rng_, now))
                        if nested:
                            _handle_outputs(nested, receiver, now)
            elif isinstance(item, MacroEvent):
                result.macro_events.append(item)
                # NEW: also publish the embedded press-release text so anons/persona see it
                if item.text_event is not None:
                    recent_text.append(item.text_event)
                    result.text_events.append(item.text_event)
                    text_receivers: list[tuple[WorldAgent, random.Random]] = [
                        *((a, anon_rngs[a.agent_id]) for a in anons),
                        (persona, persona_rng),
                    ]
                    for receiver, rng_ in text_receivers:
                        nested = receiver.on_event(item.text_event, _ctx_for(rng_, now))
                        if nested:
                            _handle_outputs(nested, receiver, now)
                # Existing MacroEvent fan-out to persona + anons (PersonaAgent reacts here)
                macro_receivers: list[tuple[WorldAgent, random.Random]] = [
                    (persona, persona_rng),
                    *((a, anon_rngs[a.agent_id]) for a in anons),
                ]
                for receiver, rng_ in macro_receivers:
                    nested = receiver.on_event(item, _ctx_for(rng_, now))
                    if nested:
                        _handle_outputs(nested, receiver, now)

    for now in clock.iter_ticks():
        # 1. MM publishes a quote first so anons have something to hit
        _handle_outputs(mm.on_tick(_ctx_for(mm_rng, now)), mm, now)
        # 2. Scheduler may emit FOMC + accompanying text
        _handle_outputs(scheduler.on_tick(_ctx_for(master_rng, now)), scheduler, now)
        # 3. Anon agents act
        for a in anons:
            _handle_outputs(a.on_tick(_ctx_for(anon_rngs[a.agent_id], now)), a, now)

    result.bars = aggregator.flush(end=scenario.end)
    return result
