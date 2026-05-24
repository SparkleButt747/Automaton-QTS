"""Crypto contagion propagation (spec docs/specs/2026-05-24-crypto-contagion-propagation-v0.md)."""

from qts.propagation.crypto.dataset import ContagionDataset, build_crypto_contagion_dataset
from qts.propagation.crypto.events import ContagionEvent, load_contagion_events
from qts.propagation.crypto.graph import CryptoLinkGraph, build_crypto_typed_adjacency
from qts.propagation.crypto.links import CRYPTO_RELATIONS, CryptoLink, CryptoLinkClassifier
from qts.propagation.crypto.reactions import btc_adjusted_car
from qts.propagation.crypto.samples import ContagionSample
from qts.propagation.crypto.structural_links import load_structural_links
from qts.propagation.crypto.universe import CryptoUniverse, load_crypto_universe

__all__ = [
    "CRYPTO_RELATIONS",
    "ContagionDataset",
    "ContagionEvent",
    "ContagionSample",
    "CryptoLink",
    "CryptoLinkClassifier",
    "CryptoLinkGraph",
    "CryptoUniverse",
    "btc_adjusted_car",
    "build_crypto_contagion_dataset",
    "build_crypto_typed_adjacency",
    "load_contagion_events",
    "load_crypto_universe",
    "load_structural_links",
]
