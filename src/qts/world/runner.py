"""run_simulation — top-level orchestrator for Phase 8 v1.

Stage 1: run_agent_sim produces bars + text events + macro events.
Stage 2: build a MarketTerrain from those, run the strategy via the
standard run_terrain_backtest, with text events forwarded to the
strategy through QTSStrategy.on_text_event.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from qts.models.base import Catalyst, LiquidityLevel, SentimentLevel, Trend, VolLevel
from qts.models.terrain import MacroRegime, MarketEvent, MarketTerrain
from qts.nautilus.config import VenueConfig
from qts.world.agent_sim import run_agent_sim
from qts.world.corpus import PersonaCorpus
from qts.world.episode import SimulatedEpisode
from qts.world.scenario import ScenarioConfig

# Anchor to the repo root (src/qts/world/runner.py -> ../../../../data/...)
# so run_simulation works regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_CORPUS = _REPO_ROOT / "data" / "world" / "persona_corpus" / "powell_fomc.yaml"


def _macro_regime_for(scenario: ScenarioConfig, surprise_bucket: str) -> MacroRegime:
    if surprise_bucket == "hawkish":
        return MacroRegime(
            trend=Trend.BEAR,
            volatility=VolLevel.HIGH,
            liquidity=LiquidityLevel.TIGHT,
            sentiment=SentimentLevel.FEARFUL,
            catalyst=Catalyst.MACRO_EVENT,
            expected_drift=-0.0002,
            expected_vol=0.015,
            correlation_regime=0.6,
            scenario_description="FOMC hawkish surprise",
        )
    if surprise_bucket == "dovish":
        return MacroRegime(
            trend=Trend.BULL,
            volatility=VolLevel.HIGH,
            liquidity=LiquidityLevel.ABUNDANT,
            sentiment=SentimentLevel.EUPHORIC,
            catalyst=Catalyst.MACRO_EVENT,
            expected_drift=0.0003,
            expected_vol=0.012,
            correlation_regime=0.4,
            scenario_description="FOMC dovish surprise",
        )
    return MacroRegime(
        trend=Trend.SIDEWAYS,
        volatility=VolLevel.HIGH,
        liquidity=LiquidityLevel.ABUNDANT,
        sentiment=SentimentLevel.NEUTRAL,
        catalyst=Catalyst.MACRO_EVENT,
        expected_drift=0.0,
        expected_vol=0.01,
        correlation_regime=0.5,
        scenario_description="FOMC neutral",
    )


def _surprise_bucket(actual: float, expected: float) -> str:
    diff = actual - expected
    if diff > 0.05:
        return "hawkish"
    if diff < -0.05:
        return "dovish"
    return "neutral"


def run_simulation(
    scenario: ScenarioConfig,
    strategy: object,
    seed: int,
    fomc_actual_rate: float,
    *,
    llm_mode: Literal["corpus", "live_cached"] = "corpus",
    persona_corpus_path: Path | None = None,
    venue_config: VenueConfig | None = None,
) -> SimulatedEpisode:
    """Full Phase 8 v1 pipeline. Returns a SimulatedEpisode.

    Args:
        scenario: Declarative episode spec.
        strategy: A QTS Strategy (with or without on_text).
        seed: Master deterministic seed.
        fomc_actual_rate: The rate the FOMC announces (drives surprise bucket).
        llm_mode: Currently only "corpus" is implemented; "live_cached" raises.
        persona_corpus_path: Override default Powell FOMC corpus path.
        venue_config: Override venue config for the strategy backtest stage.
    """
    if llm_mode != "corpus":
        raise NotImplementedError(
            "llm_mode='live_cached' is reserved for v1.5; v1 ships corpus mode only"
        )

    corpus = PersonaCorpus.from_yaml(persona_corpus_path or _DEFAULT_CORPUS)

    # ----- Stage 1: agent simulation -----
    sim = run_agent_sim(
        scenario=scenario,
        corpus=corpus,
        seed=seed,
        fomc_actual_rate=fomc_actual_rate,
    )

    surprise_bucket = _surprise_bucket(fomc_actual_rate, scenario.fomc_expected_rate)
    regime = _macro_regime_for(scenario, surprise_bucket)

    market_events: list[MarketEvent] = [
        MarketEvent(
            timestamp=me.timestamp,
            event_type=me.kind,
            description=me.text_event.text if me.text_event else me.kind,
            impact_magnitude=max(-1.0, min(1.0, me.surprise)),
        )
        for me in sim.macro_events
    ] + [
        MarketEvent(
            timestamp=te.timestamp,
            event_type=f"text:{te.source}",
            description=te.text,
            impact_magnitude=0.0,
        )
        for te in sim.text_events
    ]

    terrain = MarketTerrain(
        name=scenario.name,
        symbol=scenario.symbol,
        start=scenario.start,
        end=scenario.end,
        regime=regime,
        bars=sim.bars,
        event_calendar=market_events,
    )

    # ----- Stage 2: forward text events to the strategy -----
    # We're not running the full Nautilus backtest here — the v1 acceptance
    # uses run_terrain_backtest separately (T-WRUN-3). For the strategy passed
    # to run_simulation, we just forward TextEvents in chronological order via
    # its on_text method if present (mirrors QTSStrategy.on_text_event semantics).
    on_text = getattr(strategy, "on_text", None)
    if callable(on_text):
        for evt in sim.text_events:
            on_text(evt)

    episode = SimulatedEpisode(
        terrain=terrain,
        scenario_name=scenario.name,
        seed=seed,
        agent_traces=sim.agent_traces,
        # sorted() because set ordering is non-deterministic across runs
        # (CPython hash randomisation); episode reproducibility depends on this.
        llm_corpus_refs=sorted(set(sim.consumed_corpus_keys)),
        order_log=list(sim.order_log),
    )
    return episode
