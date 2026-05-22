"""Adversarial sim: events as do() interventions with a correlation-misleads confound."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

N_EVENT_TYPES = 6  # train on the first 5 pairs, hold out the last for the transfer gate
N_ASSETS = 3 * N_EVENT_TYPES  # named [0:6], substitute [6:12], decoy [12:18]
N_FACTORS = 2
FEATURE_DIM = 8  # dims [0:2] = factor loadings, dims [2:8] = 6-dim sector code


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

    Roles (``n = n_event_types`` disjoint triples over ``3 * n`` assets): named ``[0:n]``,
    substitute ``[n:2n]`` (substitute of type k is ``n + k``), decoy ``[2n:3n]`` (decoy of type k is
    ``2n + k``). Each named asset has a DISTINCT factor direction, so the model cannot memorise
    routing via factor betas and must learn the sector mechanism — which is the thing that transfers
    to an unseen pair. Per triple k: substitute = 90deg rotation of named (factor-orthogonal,
    ~zero corr) sharing named's sector code (cos=1); decoy = named's factor direction + noise
    (high corr) with a near-zero sector code (no substitution affinity).
    """
    rng = np.random.default_rng(config.seed)
    f = np.zeros((config.n_assets, config.feature_dim))
    n_types = config.n_event_types
    nf = config.n_factors
    sector_dim = config.feature_dim - nf
    for k in range(n_types):
        u = _unit(rng, nf)  # named k factor direction (distinct per k)
        s = _unit(rng, sector_dim)  # named/substitute k sector code
        f[k, :nf] = u
        f[k, nf:] = s
        f[n_types + k, :nf] = _rot90(u)  # substitute: factor-orthogonal to named
        f[n_types + k, nf:] = s  # substitute: sector-matched to named
        f[2 * n_types + k, :nf] = u + 0.05 * rng.standard_normal(nf)  # decoy: factor-aligned
        f[2 * n_types + k, nf:] = 0.05 * rng.standard_normal(sector_dim)  # decoy: ~no sector

    triples = tuple(
        EventTriple(named=k, substitute=n_types + k, decoy=2 * n_types + k) for k in range(n_types)
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
    n = world.config.n_event_types
    train_types = tuple(range(n - 1))  # all event types except the last
    transfer_types = (n - 1,)  # the held-out pair the model never saw coupled
    train = generate_events(world, n_train, rng, allowed_types=train_types)
    val = generate_events(world, n_val, rng, allowed_types=train_types)
    test = generate_events(world, n_test, rng, allowed_types=train_types)
    transfer = generate_events(world, n_transfer, rng, allowed_types=transfer_types)
    return train, val, test, transfer
