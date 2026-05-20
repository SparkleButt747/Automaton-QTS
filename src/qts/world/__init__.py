"""World simulator — multi-agent synthetic environment for the QTS strategies.

See docs/specs/2026-05-20-phase-8-world-simulator.md for the v1 design.
"""

from qts.world.corpus import PersonaCorpus
from qts.world.episode import AgentTrace, OrderLogEntry, SimulatedEpisode
from qts.world.events import MacroEvent, TextEvent
from qts.world.runner import run_simulation
from qts.world.scenario import AnonAgentConfig, ScenarioConfig, load_scenario_yaml

__all__ = [
    "AgentTrace",
    "AnonAgentConfig",
    "MacroEvent",
    "OrderLogEntry",
    "PersonaCorpus",
    "ScenarioConfig",
    "SimulatedEpisode",
    "TextEvent",
    "load_scenario_yaml",
    "run_simulation",
]
