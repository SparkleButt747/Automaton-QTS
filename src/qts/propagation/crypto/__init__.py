"""Crypto contagion propagation (spec docs/specs/2026-05-24-crypto-contagion-propagation-v0.md)."""

from qts.propagation.crypto.dataset import ContagionDataset, build_crypto_contagion_dataset
from qts.propagation.crypto.detect import IdiosyncraticDropDetector, ShockDetector
from qts.propagation.crypto.events import ContagionEvent, load_contagion_events
from qts.propagation.crypto.gate import (
    BacktestResult,
    CryptoGateReport,
    EventStudyReport,
    contagion_backtest,
    evaluate_crypto_gate,
    event_study_linked_vs_unlinked,
    fit_crypto_propagation,
)
from qts.propagation.crypto.graph import CryptoLinkGraph, build_crypto_typed_adjacency
from qts.propagation.crypto.links import CRYPTO_RELATIONS, CryptoLink, CryptoLinkClassifier
from qts.propagation.crypto.reactions import btc_adjusted_car
from qts.propagation.crypto.samples import ContagionSample
from qts.propagation.crypto.structural_links import load_structural_links
from qts.propagation.crypto.universe import CryptoUniverse, load_crypto_universe

__all__ = [
    "CRYPTO_RELATIONS",
    "BacktestResult",
    "ContagionDataset",
    "ContagionEvent",
    "ContagionSample",
    "CryptoGateReport",
    "CryptoLink",
    "CryptoLinkClassifier",
    "CryptoLinkGraph",
    "CryptoUniverse",
    "EventStudyReport",
    "IdiosyncraticDropDetector",
    "ShockDetector",
    "btc_adjusted_car",
    "build_crypto_contagion_dataset",
    "build_crypto_typed_adjacency",
    "contagion_backtest",
    "evaluate_crypto_gate",
    "event_study_linked_vs_unlinked",
    "fit_crypto_propagation",
    "load_contagion_events",
    "load_crypto_universe",
    "load_structural_links",
]
