"""Crypto contagion propagation (spec docs/specs/2026-05-24-crypto-contagion-propagation-v0.md)."""

from qts.propagation.crypto.graph import CryptoLinkGraph, build_crypto_typed_adjacency
from qts.propagation.crypto.links import CRYPTO_RELATIONS, CryptoLink, CryptoLinkClassifier
from qts.propagation.crypto.structural_links import load_structural_links
from qts.propagation.crypto.universe import CryptoUniverse, load_crypto_universe

__all__ = [
    "CRYPTO_RELATIONS",
    "CryptoLink",
    "CryptoLinkClassifier",
    "CryptoLinkGraph",
    "CryptoUniverse",
    "build_crypto_typed_adjacency",
    "load_crypto_universe",
    "load_structural_links",
]
