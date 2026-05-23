"""Adversarial sim: events as do() interventions with a correlation-misleads confound.

v0.1 (multi-hop): each event is a 2-hop chain named(A) -> substitute(B, via relation R1) ->
terminal(C, via relation R2). B and C are both factor-decorrelated from A (invisible to a
correlational baseline); the decoy is factor-correlated with A (bait). A->C has no direct feature
match — it exists only as the composition R1 ∘ R2, so reaching C requires two learned relations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

N_EVENT_TYPES = 6  # 5 chains trained, 1 held out for the transfer gate
N_ASSETS = 4 * N_EVENT_TYPES  # named [0:6], substitute [6:12], terminal [12:18], decoy [18:24]
N_FACTORS = 3  # 3 so the 2-hop terminal can be factor-orthogonal to BOTH named and substitute
FEATURE_DIM = 15  # [0:3] factor, [3:9] R1 ("substitution") code, [9:15] R2 ("supply") code


@dataclass(frozen=True)
class EventChain:
    named: int  # A
    substitute: int  # B — 1-hop; shares A's R1-code, factor-orthogonal to A
    terminal: int  # C — 2-hop; shares B's R2-code, factor-orthogonal to A and B
    decoy: int  # factor-correlated with A, no causal edge (baseline bait)


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
    propagation_gain: float = 1.5  # hop 1 (A -> B)
    propagation_gain2: float = 1.0  # hop 2 (B -> C)
    seed: int = 0


@dataclass(frozen=True)
class GroundTruthWorld:
    config: PropagationSimConfig
    features: np.ndarray  # (n_assets, feature_dim)
    loadings: np.ndarray  # (n_assets, n_factors)
    chains: tuple[EventChain, ...]
    regime_signs: np.ndarray  # (n_regimes,)

    def substitute_indices(self, event_type: np.ndarray) -> np.ndarray:
        idx = np.array([c.substitute for c in self.chains])
        return cast(np.ndarray, idx[event_type])

    def terminal_indices(self, event_type: np.ndarray) -> np.ndarray:
        idx = np.array([c.terminal for c in self.chains])
        return cast(np.ndarray, idx[event_type])


def _unit(rng: np.random.Generator, d: int) -> np.ndarray:
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)


def _orthonormal_basis(rng: np.random.Generator, dim: int, k: int) -> np.ndarray:
    """``k`` mutually orthonormal row-vectors in ``R^dim`` (via QR of a random matrix)."""
    q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    rows: np.ndarray = q[:k]
    return rows


def build_world(config: PropagationSimConfig) -> GroundTruthWorld:
    """Deterministic 2-hop-chain construction satisfying the confound bounds by design.

    Roles (``n = n_event_types`` disjoint chains over ``4 * n`` assets): named ``[0:n]``,
    substitute ``[n:2n]``, terminal ``[2n:3n]``, decoy ``[3n:4n]`` (asset of role r, chain k is
    ``r*n + k``). Per chain k: named/substitute/terminal get a random ORTHONORMAL factor triplet
    (mutually ~zero corr); substitute shares named's R1-code (A->B match); terminal shares
    substitute's R2-code (B->C match); decoy shares named's factor direction (high corr) with
    ~zero R1/R2 codes. A->C has no direct R1 or R2 match — only the composition R1 ∘ R2.
    """
    rng = np.random.default_rng(config.seed)
    f = np.zeros((config.n_assets, config.feature_dim))
    n = config.n_event_types
    nf = config.n_factors
    code_dim = config.feature_dim - nf
    r1_dim = code_dim // 2
    r1_lo, r1_hi = nf, nf + r1_dim
    r2_lo = nf + r1_dim
    r2_dim = config.feature_dim - r2_lo
    for k in range(n):
        basis = _orthonormal_basis(rng, nf, 3)  # a_A, a_B, a_C mutually orthogonal
        r1 = _unit(rng, r1_dim)
        r2 = _unit(rng, r2_dim)
        a, b, c, d = k, n + k, 2 * n + k, 3 * n + k
        f[a, :nf] = basis[0]  # named A
        f[a, r1_lo:r1_hi] = r1
        f[b, :nf] = basis[1]  # substitute B: factor ⊥ A
        f[b, r1_lo:r1_hi] = r1  # ... shares A's R1-code (A->B)
        f[b, r2_lo:] = r2  # ... is the R2-source for C
        f[c, :nf] = basis[2]  # terminal C: factor ⊥ A and B
        f[c, r2_lo:] = r2  # ... shares B's R2-code (B->C)
        f[d, :nf] = basis[0] + 0.05 * rng.standard_normal(nf)  # decoy: factor ~ A (corr high)
        f[d, r1_lo:r1_hi] = 0.05 * rng.standard_normal(r1_dim)
        f[d, r2_lo:] = 0.05 * rng.standard_normal(r2_dim)

    chains = tuple(
        EventChain(named=k, substitute=n + k, terminal=2 * n + k, decoy=3 * n + k) for k in range(n)
    )
    regime_signs = np.array([1.0, -1.0])
    return GroundTruthWorld(
        config=config,
        features=f,
        loadings=f[:, : config.n_factors].copy(),
        chains=chains,
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
    named = np.array([world.chains[k].named for k in event_type])
    sub = np.array([world.chains[k].substitute for k in event_type])
    term = np.array([world.chains[k].terminal for k in event_type])
    rows = np.arange(n)
    sign = world.regime_signs[regime]
    reactions[rows, named] += merit
    reactions[rows, sub] += sign * cfg.propagation_gain * merit  # hop 1: A -> B
    reactions[rows, term] += sign * cfg.propagation_gain * cfg.propagation_gain2 * merit  # hop 2
    return EventBatch(
        named_idx=named, merit=merit, regime=regime, reactions=reactions, event_type=event_type
    )


def generate_hop_events(
    world: GroundTruthWorld,
    n: int,
    rng: np.random.Generator,
    allowed_types: tuple[int, ...] | None = None,
) -> EventBatch:
    """1-hop training events: name a relation-bearing source (A or B), supervise its successor.

    Each event fires exactly one edge — A->B (gain1) or B->C (gain2). Naming B is what lets R2 be
    learned at all; without it the unroll's second hop fires an untrained edge (spec §2).
    """
    cfg = world.config
    nt = cfg.n_event_types
    types = np.arange(nt) if allowed_types is None else np.array(allowed_types)
    event_type = rng.choice(types, size=n)
    hop = rng.integers(0, 2, size=n)  # 0: A->B (gain1), 1: B->C (gain2)
    regime = rng.integers(0, cfg.n_regimes, size=n)
    g = rng.normal(0.0, cfg.factor_vol, (n, cfg.n_factors))
    merit = rng.normal(0.0, cfg.merit_vol, n)
    eps = rng.normal(0.0, cfg.idiosyncratic_vol, (n, cfg.n_assets))
    reactions = g @ world.loadings.T + eps

    named = np.where(hop == 0, event_type, nt + event_type)  # A_k or B_k
    succ = np.where(hop == 0, nt + event_type, 2 * nt + event_type)  # B_k or C_k
    gain = np.where(hop == 0, cfg.propagation_gain, cfg.propagation_gain2)
    rows = np.arange(n)
    sign = world.regime_signs[regime]
    reactions[rows, named] += merit
    reactions[rows, succ] += sign * gain * merit
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
    transfer_types = (n - 1,)  # the held-out chain the model never saw coupled
    train = generate_events(world, n_train, rng, allowed_types=train_types)
    val = generate_events(world, n_val, rng, allowed_types=train_types)
    test = generate_events(world, n_test, rng, allowed_types=train_types)
    transfer = generate_events(world, n_transfer, rng, allowed_types=transfer_types)
    return train, val, test, transfer
