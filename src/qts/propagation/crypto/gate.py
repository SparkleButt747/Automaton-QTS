"""Fit the relation-typed operator on contagion events; score against the nulls (spec §7, §12).

Null A = event-study linked-vs-unlinked; Null B = operator beats per-pair correlation.
Gate/backtest take a PREDICTIONS array (not the model) for deterministic testing;
the CLI computes predictions once via ``model.predict_np``.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
import torch
from scipy import stats

from qts.propagation.crypto.links import CRYPTO_RELATIONS
from qts.propagation.crypto.samples import ContagionSample
from qts.propagation.equity.baseline import EquityCorrelationalBaseline
from qts.propagation.equity.model import RelationTypedPropagation


def fit_crypto_propagation(
    samples: list[ContagionSample],
    *,
    adj_type: np.ndarray,
    feature_dim: int,
    steps: int = 2000,
    lr: float = 5e-3,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    seed: int = 0,
) -> RelationTypedPropagation:
    """Meta-train the per-relation-type operator over events.

    Uses ``n_relations=len(CRYPTO_RELATIONS)`` (7) — the equity ``fit_typed_propagation``
    hardcodes 4 and would index out of range on crypto edges.
    """
    torch.manual_seed(seed)
    feats = torch.as_tensor(samples[0].features, dtype=torch.float32)
    adj = torch.as_tensor(adj_type, dtype=torch.long)
    named = torch.as_tensor([s.named_idx for s in samples], dtype=torch.long)
    merit = torch.as_tensor([s.merit for s in samples], dtype=torch.float32)
    y = torch.as_tensor(np.stack([s.reactions for s in samples]), dtype=torch.float32)
    model = RelationTypedPropagation(feature_dim=feature_dim, n_relations=len(CRYPTO_RELATIONS))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        loss = torch.nn.functional.mse_loss(model(feats, adj, named, merit), y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
    return model


@dataclass(frozen=True)
class CryptoGateReport:
    n_linked_obs: int
    graph_mse: float
    pairwise_mse: float
    graph_hit: float
    pairwise_hit: float
    beats_pairwise: bool


def evaluate_crypto_gate(
    predictions: np.ndarray, samples: list[ContagionSample], *, adj_type: np.ndarray
) -> CryptoGateReport:
    """Operator vs per-pair correlational baseline on LINKED-peer idiosyncratic CARs (Null B)."""
    react = np.stack([s.reactions for s in samples])
    named = np.array([s.named_idx for s in samples])
    rows = np.arange(len(samples))
    gerr, berr, ghit, bhit = [], [], [], []
    for b in range(len(samples)):
        src = named[b]
        for j in np.where(adj_type[src] >= 0)[0]:
            actual = react[b, j]
            gerr.append((predictions[b, j] - actual) ** 2)
            base = EquityCorrelationalBaseline.from_history(
                named_returns=react[rows, src], peer_returns=react[rows, j]
            )
            bp = base.predict(named_reaction=react[b, src])
            berr.append((bp - actual) ** 2)
            if actual != 0:
                ghit.append(float(np.sign(predictions[b, j]) == np.sign(actual)))
                bhit.append(float(np.sign(bp) == np.sign(actual)))
    gm, bm = float(np.mean(gerr)), float(np.mean(berr))
    return CryptoGateReport(
        n_linked_obs=len(gerr),
        graph_mse=gm,
        pairwise_mse=bm,
        graph_hit=float(np.mean(ghit)) if ghit else float("nan"),
        pairwise_hit=float(np.mean(bhit)) if bhit else float("nan"),
        beats_pairwise=gm < bm,
    )


@dataclass(frozen=True)
class EventStudyReport:
    n_linked: int
    n_unlinked: int
    mean_linked_car: float
    mean_unlinked_car: float
    mann_whitney_p: float
    significant: bool  # linked significantly more negative at p < 0.05


def event_study_linked_vs_unlinked(
    samples: list[ContagionSample], *, adj_type: np.ndarray
) -> EventStudyReport:
    """Null A: do graph-linked peers drop significantly more than unlinked tokens? One-sided
    Mann-Whitney on the pooled (event x peer) abnormal CARs. Excludes the source node itself."""
    linked, unlinked = [], []
    for s in samples:
        src = s.named_idx
        for j in range(s.n_nodes):
            if j == src:
                continue
            (linked if adj_type[src, j] >= 0 else unlinked).append(float(s.reactions[j]))
    la, ua = np.array(linked), np.array(unlinked)
    if len(la) < 3 or len(ua) < 3:
        p = float("nan")
    else:
        # alternative='less': linked CARs stochastically smaller (more negative) than unlinked
        p = float(stats.mannwhitneyu(la, ua, alternative="less").pvalue)
    return EventStudyReport(
        n_linked=len(la),
        n_unlinked=len(ua),
        mean_linked_car=float(np.mean(la)) if len(la) else float("nan"),
        mean_unlinked_car=float(np.mean(ua)) if len(ua) else float("nan"),
        mann_whitney_p=p,
        significant=bool(p < 0.05),
    )


@dataclass(frozen=True)
class BacktestResult:
    n_trades: int
    market_neutral_mean: float
    market_neutral_sharpe: float
    outright_mean: float
    outright_sharpe: float


def _sharpe(per_event: list[float], periods_per_year: float) -> float:
    arr = np.array(per_event)
    if len(arr) < 2 or arr.std(ddof=1) == 0:
        return 0.0
    return float(arr.mean() / arr.std(ddof=1) * np.sqrt(periods_per_year))


def contagion_backtest(
    predictions: np.ndarray,
    dataset: object,
    *,
    token_names: tuple[str, ...],
    top_k: int = 3,
    cost_bps: float = 7.5,
    horizon: int = 24,
    events_per_year: float = 12.0,
) -> BacktestResult:
    """Short the top-K predicted-to-drop LINKED peers per event.

    Reports market-neutral (idiosyncratic) and outright (raw) mean P&L + annualised Sharpe,
    net of round-trip ``cost_bps``.
    """
    grid, closes, adj = dataset.grid, dataset.closes, dataset.adj_type  # type: ignore[attr-defined]
    cost = cost_bps / 1e4
    mn_per_event, out_per_event, n_trades = [], [], 0
    for b, s in enumerate(dataset.samples):  # type: ignore[attr-defined]
        src = s.named_idx
        peers = [j for j in np.where(adj[src] >= 0)[0] if j != src]
        if not peers:
            continue
        ranked = sorted(peers, key=lambda j: predictions[b, j])[:top_k]
        e_idx = bisect.bisect_left(grid, s.event_ts)
        mn_legs, out_legs = [], []
        for j in ranked:
            mn_legs.append(-float(s.reactions[j]) - cost)
            tok = token_names[j]
            tc = closes.get(tok)
            if tc is not None and e_idx + horizon < len(tc):
                raw = tc[e_idx + horizon] / tc[e_idx] - 1.0
                out_legs.append(-float(raw) - cost)
        if mn_legs:
            mn_per_event.append(float(np.mean(mn_legs)))
            out_per_event.append(float(np.mean(out_legs)) if out_legs else 0.0)
            n_trades += len(mn_legs)
    return BacktestResult(
        n_trades=n_trades,
        market_neutral_mean=float(np.mean(mn_per_event)) if mn_per_event else 0.0,
        market_neutral_sharpe=_sharpe(mn_per_event, events_per_year),
        outright_mean=float(np.mean(out_per_event)) if out_per_event else 0.0,
        outright_sharpe=_sharpe(out_per_event, events_per_year),
    )
