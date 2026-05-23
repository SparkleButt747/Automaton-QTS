"""Filtered, queryable economic-link graph (Path A v2 Phase 1, spec §5)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from qts.propagation.equity.economic_links import EconomicLink


@dataclass(frozen=True)
class LinkGraph:
    """Confidence-filtered causal links + 1-hop / 2-hop queries."""

    edges: tuple[EconomicLink, ...]

    @classmethod
    def from_links(cls, links: list[EconomicLink], *, min_confidence: float = 0.5) -> LinkGraph:
        kept = tuple(e for e in links if e.relation != "none" and e.confidence >= min_confidence)
        return cls(edges=kept)

    def _adjacency(self) -> dict[str, list[EconomicLink]]:
        adj: dict[str, list[EconomicLink]] = defaultdict(list)
        for e in self.edges:
            adj[e.source].append(e)
        return adj

    def peers_of(self, source: str) -> list[EconomicLink]:
        return [e for e in self.edges if e.source == source]

    def two_hop_chains(self, source: str) -> list[tuple[str, str, str]]:
        """A->B->C where C is NOT directly linked to A (decorrelated 2-hop terminal, spec §5)."""
        adj = self._adjacency()
        direct = {e.peer for e in adj.get(source, [])}
        chains: list[tuple[str, str, str]] = []
        for ab in adj.get(source, []):
            b = ab.peer
            for bc in adj.get(b, []):
                c = bc.peer
                if c != source and c not in direct:
                    chains.append((source, b, c))
        return chains
