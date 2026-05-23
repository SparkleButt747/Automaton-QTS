"""Meta-learned propagation: episodic relation-resampling cracks n-hop transfer (design §15).

The fixed-world ``GatedPropagationGraph`` memorises its world's relation-directions and fails to
transfer a learned 2-hop composition to an unseen chain (the ``T-PROP-GATE-3`` xfail). The fix is a
*training-objective* change, not an architecture change (design §14.6 guessed this): resample the
world's relations every training step from a POOL of distinct worlds, with ONE shared
feature-functional bilinear operator ``M``. Unable to memorise per-world directions, the operator is
forced into the generic composable rule, which transfers to entirely unseen worlds.

Key empirical facts (design §15): ~10 distinct worlds is the diversity floor; ``prop_steps=3`` +
gradient clipping is the stability sweet spot (``prop_steps=8`` diverges under diverse worlds);
zero-shot capture decays with hop depth (a structural ceiling — ~52%/31%/3% of 1-/2-/3-hop signal),
and :func:`few_shot_adapt` recovers most of the signal at every depth given enough examples."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from qts.propagation.baselines import CorrelationalBaseline
from qts.propagation.sim import (
    EventBatch,
    GroundTruthWorld,
    PropagationSimConfig,
    build_world,
    generate_events,
)

logger = logging.getLogger(__name__)

META_PROP_STEPS = 3  # reaches a 2-hop terminal; 8 diverges under diverse worlds (design §15)
META_N_REGIMES = 2
META_STEPS = 20000
META_BATCH = 256
META_LR = 3e-3
META_L1 = 1e-3
GRAD_CLIP = 1.0
ADAPT_STEPS = 50
ADAPT_LR = 5e-3
ADAPT_ANCHOR = 0.01  # L2 pull back to the meta-init; stops small-K support from overfitting

WorldFactory = Callable[[int], GroundTruthWorld]
EventSampler = Callable[[GroundTruthWorld, int, np.random.Generator], EventBatch]


class MetaPropagationGraph(nn.Module):
    """Shared feature-functional bilinear operator: ``W[i,j] = sum_r pi_r * (xi_i^T M_r xi_j)``.

    Unlike :class:`~qts.propagation.model.GatedPropagationGraph`, features are NOT baked in — they
    are passed at ``forward`` time, because ``M`` is shared across worlds with different assets.
    That is precisely what lets one trained operator generalise to an unseen world's relations.
    """

    def __init__(
        self,
        feature_dim: int,
        n_regimes: int = META_N_REGIMES,
        prop_steps: int = META_PROP_STEPS,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.n_regimes = n_regimes
        self.prop_steps = prop_steps
        self.M = nn.Parameter(0.01 * torch.randn(n_regimes, feature_dim, feature_dim))
        self.gate = nn.Linear(n_regimes, n_regimes)

    def forward(self, feats: Tensor, named_idx: Tensor, merit: Tensor, regime: Tensor) -> Tensor:
        a = torch.einsum("nf,rfg,mg->rnm", feats, self.M, feats)  # (R, N, N)
        onehot = torch.nn.functional.one_hot(regime, self.n_regimes).float()  # (B, R)
        pi = torch.softmax(self.gate(onehot), dim=-1)  # (B, R)
        w = torch.einsum("br,rnm->bnm", pi, a)  # (B, N, N)
        clamp = torch.nn.functional.one_hot(named_idx, feats.shape[0]).float()  # (B, N)
        source = clamp * merit[:, None]  # do()-clamp: merit at the named node, 0 elsewhere
        clamp_mask = clamp.bool()
        x = source.clone()
        for _ in range(self.prop_steps):
            x = torch.einsum("bnm,bm->bn", w, x)
            x = torch.where(clamp_mask, source, x)  # re-pin the do() source each step
        return x

    @torch.no_grad()
    def predict_np(
        self, feats: np.ndarray, named_idx: np.ndarray, merit: np.ndarray, regime: np.ndarray
    ) -> np.ndarray:
        self.eval()
        out = self(
            torch.as_tensor(feats, dtype=torch.float32),
            torch.as_tensor(named_idx, dtype=torch.long),
            torch.as_tensor(merit, dtype=torch.float32),
            torch.as_tensor(regime, dtype=torch.long),
        )
        return out.cpu().numpy()  # type: ignore[no-any-return]


def _default_world(seed: int) -> GroundTruthWorld:
    return build_world(PropagationSimConfig(seed=seed))


def train_meta(
    world_seeds: Sequence[int],
    *,
    build_fn: WorldFactory = _default_world,
    generate_fn: EventSampler = generate_events,
    steps: int = META_STEPS,
    batch_size: int = META_BATCH,
    lr: float = META_LR,
    l1_lambda: float = META_L1,
    grad_clip: float = GRAD_CLIP,
    prop_steps: int = META_PROP_STEPS,
    n_regimes: int = META_N_REGIMES,
    seed: int = 0,
) -> MetaPropagationGraph:
    """Episodic relation-resampling: each step samples a world from the pool + a fresh event batch.

    A shared operator trained over >= ~10 distinct worlds (the diversity floor, design §15) learns
    the generic composable rule and transfers to unseen worlds. Returns the trained operator.
    """
    if len(world_seeds) == 0:
        raise ValueError("world_seeds must be non-empty")
    rng = np.random.default_rng(seed)
    pool = [build_fn(int(s)) for s in world_seeds]
    feature_dim = pool[0].features.shape[1]
    feats_cache = {id(w): torch.as_tensor(w.features, dtype=torch.float32) for w in pool}
    torch.manual_seed(seed)
    model = MetaPropagationGraph(feature_dim, n_regimes=n_regimes, prop_steps=prop_steps)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        w = pool[int(rng.integers(len(pool)))]
        batch = generate_fn(w, batch_size, rng)
        pred = model(
            feats_cache[id(w)],
            torch.as_tensor(batch.named_idx, dtype=torch.long),
            torch.as_tensor(batch.merit, dtype=torch.float32),
            torch.as_tensor(batch.regime, dtype=torch.long),
        )
        y = torch.as_tensor(batch.reactions, dtype=torch.float32)
        loss = torch.nn.functional.mse_loss(pred, y) + l1_lambda * model.M.abs().sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
    return model


def few_shot_adapt(
    model: MetaPropagationGraph,
    feats: np.ndarray,
    support: EventBatch,
    *,
    steps: int = ADAPT_STEPS,
    lr: float = ADAPT_LR,
    anchor: float = ADAPT_ANCHOR,
    grad_clip: float = GRAD_CLIP,
) -> MetaPropagationGraph:
    """Test-time adaptation on K support events from a new world, L2-anchored to the meta-init.

    Returns a NEW adapted operator (the input ``model`` is not mutated). The anchor stops small-K
    runs from overfitting the support set's factor/idiosyncratic noise (design §15: K<~32 overfits
    without it). Capture climbs toward the oracle as K grows (~87%/91%/72% at K=512, 1-/2-/3-hop).
    """
    adapted = MetaPropagationGraph(
        model.feature_dim, n_regimes=model.n_regimes, prop_steps=model.prop_steps
    )
    adapted.load_state_dict(model.state_dict())
    base = [p.detach().clone() for p in adapted.parameters()]
    ft = torch.as_tensor(feats, dtype=torch.float32)
    ni = torch.as_tensor(support.named_idx, dtype=torch.long)
    me = torch.as_tensor(support.merit, dtype=torch.float32)
    rg = torch.as_tensor(support.regime, dtype=torch.long)
    y = torch.as_tensor(support.reactions, dtype=torch.float32)
    opt = torch.optim.Adam(adapted.parameters(), lr=lr)
    for _ in range(steps):
        anchor_pen = sum(
            ((p - b) ** 2).sum() for p, b in zip(adapted.parameters(), base, strict=True)
        )
        loss = torch.nn.functional.mse_loss(adapted(ft, ni, me, rg), y) + anchor * anchor_pen
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapted.parameters(), grad_clip)
        opt.step()
    return adapted


@dataclass(frozen=True)
class MetaTransferReport:
    sub_mse_graph: float
    sub_mse_corr: float
    terminal_mse_graph: float
    terminal_mse_corr: float
    sub_capture: (
        float  # fraction of the substitute (1-hop B) merit-signal recovered vs predict-zero
    )
    terminal_capture: float  # same for the terminal (2-hop C)
    sub_win: bool  # graph beats the correlational baseline on the substitute
    terminal_win: bool


def _role_mse(pred: np.ndarray, batch: EventBatch, role_idx: np.ndarray) -> float:
    rows = np.arange(len(batch))
    return float(np.mean((pred[rows, role_idx] - batch.reactions[rows, role_idx]) ** 2))


def evaluate_meta_transfer(
    model: MetaPropagationGraph,
    world: GroundTruthWorld,
    eval_batch: EventBatch,
    *,
    n_history: int = 10000,
    seed: int = 0,
) -> MetaTransferReport:
    """Capture and win-vs-correlational for the 1-hop substitute (B) and 2-hop terminal (C).

    ``capture = (corr_mse - graph_mse) / signal_var``, where ``signal_var`` is the role's
    merit-driven reaction variance (the correlational baseline ≈ predicts-zero for the
    factor-orthogonal roles, so its MSE ≈ ``signal_var + noise``). 1.0 == the per-world oracle.
    Evaluate on an *unseen* world to measure transfer.
    """
    pg = model.predict_np(world.features, eval_batch.named_idx, eval_batch.merit, eval_batch.regime)
    corr = CorrelationalBaseline.from_history(world, n_samples=n_history, seed=seed)
    pc = corr.predict(eval_batch)
    sub = world.substitute_indices(eval_batch.event_type)
    term = world.terminal_indices(eval_batch.event_type)
    sg, sc = _role_mse(pg, eval_batch, sub), _role_mse(pc, eval_batch, sub)
    tg, tc = _role_mse(pg, eval_batch, term), _role_mse(pc, eval_batch, term)
    cfg = world.config
    sub_sig = (cfg.propagation_gain * cfg.merit_vol) ** 2
    term_sig = (cfg.propagation_gain * cfg.propagation_gain2 * cfg.merit_vol) ** 2
    return MetaTransferReport(
        sub_mse_graph=sg,
        sub_mse_corr=sc,
        terminal_mse_graph=tg,
        terminal_mse_corr=tc,
        sub_capture=(sc - sg) / sub_sig,
        terminal_capture=(tc - tg) / term_sig,
        sub_win=sg < sc,
        terminal_win=tg < tc,
    )
