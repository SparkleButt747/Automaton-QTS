"""Assemble the crypto contagion dataset: hybrid typed graph + per-event reactions/features ->
ContagionSamples (spec §9). Async: the LLM classifier and Binance adapter are async."""

from __future__ import annotations

import bisect
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qts.oversight.llm_client import LLMClientProtocol
from qts.propagation.crypto.events import ContagionEvent
from qts.propagation.crypto.features import crypto_node_feature_vector
from qts.propagation.crypto.graph import CryptoLinkGraph, build_crypto_typed_adjacency
from qts.propagation.crypto.links import CryptoLink, CryptoLinkClassifier
from qts.propagation.crypto.prices import fetch_price_panel
from qts.propagation.crypto.reactions import btc_adjusted_car
from qts.propagation.crypto.samples import ContagionSample
from qts.propagation.crypto.structural_links import load_structural_links
from qts.propagation.crypto.universe import CryptoUniverse

logger = logging.getLogger(__name__)
FEATURE_WINDOW = 250  # hourly bars before the event used to compute node features


@dataclass(frozen=True)
class ContagionDataset:
    samples: list[ContagionSample]
    adj_type: np.ndarray
    feature_dim: int
    graph: CryptoLinkGraph


async def _classify_pairs(
    clf: CryptoLinkClassifier, universe: CryptoUniverse
) -> list[CryptoLink]:  # pragma: no cover - exercised via stub in dataset test
    """Classify every ordered within-universe pair (v0 context = token identity + cluster)."""
    links: list[CryptoLink] = []
    for src in universe.tokens:
        for peer in universe.tokens:
            if src == peer:
                continue
            ctx = f"{src} ({universe.cluster_of(src)}) and {peer} ({universe.cluster_of(peer)})"
            links.append(await clf.classify(src, peer, context=ctx))
    return links


async def build_crypto_contagion_dataset(
    universe: CryptoUniverse,
    events: list[ContagionEvent],
    *,
    bar_adapter: object,
    structural_seed_path: Path,
    llm: LLMClientProtocol,
    cache_dir: Path,
    horizon: int,
    est_window: int,
    start: str,
    end: str,
    min_confidence: float = 0.5,
) -> ContagionDataset:
    # 1. hybrid typed graph: LLM soft links + curated structural seed
    clf = CryptoLinkClassifier(llm, cache_dir=cache_dir)
    llm_links = await _classify_pairs(clf, universe)
    structural = load_structural_links(structural_seed_path)
    graph = CryptoLinkGraph.from_links(structural, llm_links, min_confidence=min_confidence)
    adj = build_crypto_typed_adjacency(graph, universe)

    # 2. aligned hourly price panel (universe tokens that have data + BTC)
    grid, closes = await fetch_price_panel(
        list(universe.tokens), bar_adapter=bar_adapter, start=start, end=end, interval="1h"
    )
    btc = closes["BTC"]

    # 3. per-event reactions + features -> ContagionSample
    samples: list[ContagionSample] = []
    for ev in events:
        if ev.source_token not in universe.tokens or ev.source_token not in closes:
            continue
        e_idx = bisect.bisect_left(grid, ev.timestamp)
        if e_idx < FEATURE_WINDOW or e_idx < est_window or e_idx + horizon >= len(grid):
            continue  # event outside the usable panel window
        reactions = np.zeros(len(universe.tokens))
        rows: list[np.ndarray | None] = []
        for i, tok in enumerate(universe.tokens):
            tc = closes.get(tok)
            if tc is None:
                rows.append(None)
                continue
            reactions[i] = btc_adjusted_car(
                token_closes=tc,
                btc_closes=btc,
                event_close_idx=e_idx,
                horizon=horizon,
                est_window=est_window,
            )
            rows.append(
                crypto_node_feature_vector(
                    cluster=universe.cluster_of(tok),
                    clusters=universe.unique_clusters,
                    token_closes=tc[e_idx - FEATURE_WINDOW : e_idx],
                    btc_closes=btc[e_idx - FEATURE_WINDOW : e_idx],
                    is_stablecoin=universe.is_stablecoin[i],
                    is_exchange_token=universe.is_exchange_token[i],
                )
            )
        src_idx = universe.index_of(ev.source_token)
        if rows[src_idx] is None:
            continue
        dim = len(rows[src_idx])
        feats = np.zeros((len(universe.tokens), dim))
        for i, r in enumerate(rows):
            if r is not None:
                feats[i] = r
        samples.append(
            ContagionSample(
                named_idx=src_idx,
                merit=float(reactions[src_idx]),  # realized-move seed
                event_ts=ev.timestamp,
                features=feats,
                reactions=reactions,
            )
        )
    feature_dim = samples[0].features.shape[1] if samples else len(universe.unique_clusters) + 5
    return ContagionDataset(samples=samples, adj_type=adj, feature_dim=feature_dim, graph=graph)
