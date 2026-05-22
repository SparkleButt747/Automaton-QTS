"""Adversarial sim: events as do() interventions with a correlation-misleads confound."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

N_ASSETS = 8
N_FACTORS = 2
N_EVENT_TYPES = 3
FEATURE_DIM = 4  # dims [0:2] = factor loadings, dims [2:4] = sector code


@dataclass(frozen=True)
class EventTriple:
    named: int
    substitute: int
    decoy: int


@dataclass(frozen=True)
class PropagationSimConfig:
    n_assets: int = N_ASSETS
    n_factors: int = N_FACTORS
    n_event_types: int = N_EVENT_TYPES
    feature_dim: int = FEATURE_DIM
    n_regimes: int = 2
    factor_vol: float = 1.0
    idiosyncratic_vol: float = 0.3
    merit_vol: float = 1.0
    propagation_gain: float = 1.5
    seed: int = 0


@dataclass(frozen=True)
class GroundTruthWorld:
    config: PropagationSimConfig
    features: np.ndarray  # (n_assets, feature_dim)
    loadings: np.ndarray  # (n_assets, n_factors)
    triples: tuple[EventTriple, ...]
    regime_signs: np.ndarray  # (n_regimes,)

    def substitute_indices(self, event_type: np.ndarray) -> np.ndarray:
        subs = np.array([t.substitute for t in self.triples])
        return cast(np.ndarray, subs[event_type])


def _unit(rng: np.random.Generator, d: int) -> np.ndarray:
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def _rot90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]])


def build_world(config: PropagationSimConfig) -> GroundTruthWorld:
    """Deterministic construction satisfying the confound bounds by design.

    Roles: named={0,1,2}, substitute={3,4,5} (sub of type k is k+3), decoys={6,7}.
    Factor loadings (dims 0:2): named/decoy share a direction (high corr); each substitute
    is the 90deg rotation of its named (~zero corr). Sector codes (dims 2:4): shared within a
    (named, substitute) pair, distinct across pairs; decoys get near-zero sector code.
    """
    rng = np.random.default_rng(config.seed)
    f = np.zeros((config.n_assets, config.feature_dim))

    u = _unit(rng, 2)  # named0, named2, decoy6 factor direction
    w = _unit(rng, 2)  # named1, decoy7 factor direction
    f[0, :2] = u
    f[1, :2] = w
    f[2, :2] = u + 0.05 * rng.standard_normal(2)
    f[3, :2] = _rot90(u)
    f[4, :2] = _rot90(w)
    f[5, :2] = _rot90(f[2, :2])
    f[6, :2] = u + 0.05 * rng.standard_normal(2)
    f[7, :2] = w + 0.05 * rng.standard_normal(2)

    s0, s1, s2 = _unit(rng, 2), _unit(rng, 2), _unit(rng, 2)
    f[0, 2:], f[3, 2:] = s0, s0
    f[1, 2:], f[4, 2:] = s1, s1
    f[2, 2:], f[5, 2:] = s2, s2
    f[6, 2:] = 0.05 * rng.standard_normal(2)
    f[7, 2:] = 0.05 * rng.standard_normal(2)

    triples = (
        EventTriple(named=0, substitute=3, decoy=6),
        EventTriple(named=1, substitute=4, decoy=7),
        EventTriple(named=2, substitute=5, decoy=6),
    )
    regime_signs = np.array([1.0, -1.0])
    return GroundTruthWorld(
        config=config,
        features=f,
        loadings=f[:, : config.n_factors].copy(),
        triples=triples,
        regime_signs=regime_signs,
    )


@dataclass(frozen=True)
class EventBatch:
    named_idx: np.ndarray  # (B,)
    merit: np.ndarray  # (B,)
    regime: np.ndarray  # (B,)
    reactions: np.ndarray  # (B, n_assets)
    event_type: np.ndarray  # (B,)

    def __len__(self) -> int:
        return int(self.named_idx.shape[0])


def generate_events(
    world: GroundTruthWorld,
    n: int,
    rng: np.random.Generator,
    allowed_types: tuple[int, ...] | None = None,
) -> EventBatch:
    cfg = world.config
    types = np.arange(cfg.n_event_types) if allowed_types is None else np.array(allowed_types)
    event_type = rng.choice(types, size=n)
    regime = rng.integers(0, cfg.n_regimes, size=n)
    g = rng.normal(0.0, cfg.factor_vol, (n, cfg.n_factors))
    merit = rng.normal(0.0, cfg.merit_vol, n)
    eps = rng.normal(0.0, cfg.idiosyncratic_vol, (n, cfg.n_assets))

    reactions = g @ world.loadings.T + eps
    named = np.array([world.triples[k].named for k in event_type])
    sub = np.array([world.triples[k].substitute for k in event_type])
    rows = np.arange(n)
    reactions[rows, named] += merit
    reactions[rows, sub] += world.regime_signs[regime] * cfg.propagation_gain * merit
    return EventBatch(
        named_idx=named, merit=merit, regime=regime, reactions=reactions, event_type=event_type
    )


def make_splits(
    world: GroundTruthWorld,
    rng: np.random.Generator,
    *,
    n_train: int = 4000,
    n_val: int = 1000,
    n_test: int = 1000,
    n_transfer: int = 1000,
) -> tuple[EventBatch, EventBatch, EventBatch, EventBatch]:
    train = generate_events(world, n_train, rng, allowed_types=(0, 1))
    val = generate_events(world, n_val, rng, allowed_types=(0, 1))
    test = generate_events(world, n_test, rng, allowed_types=(0, 1))
    transfer = generate_events(world, n_transfer, rng, allowed_types=(2,))
    return train, val, test, transfer
