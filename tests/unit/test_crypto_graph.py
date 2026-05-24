"""T-CRYPTO-GRAPH-*: hybrid (structural + LLM) merge -> filtered graph -> typed adjacency."""

from __future__ import annotations

from pathlib import Path

from qts.propagation.crypto.graph import CryptoLinkGraph, build_crypto_typed_adjacency

from qts.propagation.crypto.links import CRYPTO_RELATIONS, CryptoLink
from qts.propagation.crypto.universe import load_crypto_universe


def _uni(tmp_path: Path):
    p = tmp_path / "uni.yaml"
    p.write_text(
        "tokens:\n  FTT: {cluster: ExchangeToken}\n  SOL: {cluster: L1}\n  SRM: {cluster: ExchangeEco}\n"
    )
    return load_crypto_universe(p)


def test_merge_dedupes_keeping_max_confidence_and_filters() -> None:  # T-CRYPTO-GRAPH-1
    structural = [CryptoLink("FTT", "SOL", "entity_exposure", "negative", 0.95)]
    llm = [
        CryptoLink("FTT", "SOL", "entity_exposure", "negative", 0.6),  # dup, lower conf -> dropped
        CryptoLink("FTT", "SRM", "competitor", "negative", 0.7),
        CryptoLink("SOL", "SRM", "none", "none", 0.9),  # none -> filtered
        CryptoLink("FTT", "SRM", "competitor", "negative", 0.2),  # below threshold copy -> n/a
    ]
    g = CryptoLinkGraph.from_links(structural, llm, min_confidence=0.5)
    pairs = {(e.source, e.peer): e.confidence for e in g.edges}
    assert pairs[("FTT", "SOL")] == 0.95  # max-confidence wins on dedupe
    assert ("FTT", "SRM") in pairs
    assert ("SOL", "SRM") not in pairs  # none filtered


def test_typed_adjacency_indices(tmp_path: Path) -> None:  # T-CRYPTO-GRAPH-2
    uni = _uni(tmp_path)
    g = CryptoLinkGraph.from_links([CryptoLink("FTT", "SOL", "entity_exposure", "negative", 0.95)])
    adj = build_crypto_typed_adjacency(g, uni)
    assert adj.shape == (3, 3)
    i, j = uni.index_of("FTT"), uni.index_of("SOL")
    assert adj[i, j] == CRYPTO_RELATIONS.index("entity_exposure")
    assert adj[j, i] == -1  # directed
    assert (adj == -1).sum() == 8  # only one directed edge set
