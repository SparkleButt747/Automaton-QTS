"""Hybrid contagion-link graph: merge structural + LLM links, dedupe (max confidence), filter,
and build the (N, N) relation-index adjacency the operator consumes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qts.propagation.crypto.links import CRYPTO_RELATIONS, CryptoLink
from qts.propagation.crypto.universe import CryptoUniverse

_REL_INDEX = {r: i for i, r in enumerate(CRYPTO_RELATIONS)}


@dataclass(frozen=True)
class CryptoLinkGraph:
    edges: tuple[CryptoLink, ...]

    @classmethod
    def from_links(
        cls, *link_lists: list[CryptoLink], min_confidence: float = 0.5
    ) -> CryptoLinkGraph:
        """Merge any number of link sources. Drop ``none``/below-threshold; on a duplicate directed
        (source, peer) keep the highest-confidence link."""
        best: dict[tuple[str, str], CryptoLink] = {}
        for links in link_lists:
            for e in links:
                if e.relation == "none" or e.confidence < min_confidence:
                    continue
                k = (e.source, e.peer)
                cur = best.get(k)
                if cur is None or e.confidence > cur.confidence:
                    best[k] = e
        return cls(edges=tuple(best.values()))

    def peers_of(self, source: str) -> list[CryptoLink]:
        return [e for e in self.edges if e.source == source]


def build_crypto_typed_adjacency(graph: CryptoLinkGraph, universe: CryptoUniverse) -> np.ndarray:
    """(N, N) int matrix: ``adj[i, j]`` = relation index of directed link i->j, or -1 if none."""
    n = len(universe.tokens)
    adj = np.full((n, n), -1, dtype=np.int64)
    for e in graph.edges:
        if e.relation not in _REL_INDEX:
            continue
        if e.source not in universe.tokens or e.peer not in universe.tokens:
            continue
        adj[universe.index_of(e.source), universe.index_of(e.peer)] = _REL_INDEX[e.relation]
    return adj
