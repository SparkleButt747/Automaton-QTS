"""T-PATHA-GRAPH-*: filtered economic-link graph queries."""

from __future__ import annotations

from qts.propagation.equity.economic_links import EconomicLink
from qts.propagation.equity.graph import LinkGraph


def _link(s: str, p: str, rel: str = "competitor", conf: float = 0.9) -> EconomicLink:
    return EconomicLink(source=s, peer=p, relation=rel, direction="negative", confidence=conf)


def test_graph_filters_none_and_low_confidence() -> None:  # T-PATHA-GRAPH-1
    links = [
        _link("AMD", "NVDA", conf=0.9),
        _link("AAPL", "XOM", rel="none", conf=0.9),  # incidental -> dropped
        _link("F", "GM", conf=0.2),  # below threshold -> dropped
    ]
    g = LinkGraph.from_links(links, min_confidence=0.5)
    assert {(e.source, e.peer) for e in g.edges} == {("AMD", "NVDA")}


def test_peers_of_returns_one_hop() -> None:  # T-PATHA-GRAPH-2
    g = LinkGraph.from_links([_link("AMD", "NVDA"), _link("AMD", "INTC")], min_confidence=0.5)
    peers = {e.peer for e in g.peers_of("AMD")}
    assert peers == {"NVDA", "INTC"}


def test_two_hop_chains_exclude_direct_links() -> None:  # T-PATHA-GRAPH-3
    links = [
        _link("A", "B"),
        _link("B", "C"),
        _link("A", "C"),
        _link("B", "D"),
    ]
    g = LinkGraph.from_links(links, min_confidence=0.5)
    chains = set(g.two_hop_chains("A"))
    assert ("A", "B", "D") in chains
    assert ("A", "B", "C") not in chains  # C directly linked to A -> excluded


def test_public_api_exported() -> None:  # T-PATHA-GRAPH-4
    import qts.propagation.equity as eq

    for name in (
        "Article",
        "CoMentionEdge",
        "build_comention_edges",
        "EconomicLink",
        "EconomicLinkClassifier",
        "LinkGraph",
    ):
        assert hasattr(eq, name), name
