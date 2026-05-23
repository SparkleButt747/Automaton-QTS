"""Path A v2 — real-equity economic-link graph (Phase 1 + Phase 2 + Phase 3)."""

from qts.propagation.equity.baseline import EquityCorrelationalBaseline
from qts.propagation.equity.comention import (
    Article,
    CoMentionEdge,
    RawArticle,
    build_comention_edges,
    extract_tickers,
    load_fnspid_articles,
)
from qts.propagation.equity.dataset import assemble_event_samples, build_path_a_dataset
from qts.propagation.equity.earnings import (
    EarningsEvent,
    EarningsRow,
    compute_sue,
    normalize_yf_earnings,
)
from qts.propagation.equity.economic_links import EconomicLink, EconomicLinkClassifier
from qts.propagation.equity.features import (
    market_beta,
    momentum_12_1,
    node_feature_vector,
    realised_vol,
)
from qts.propagation.equity.gate import PathAReport, evaluate_path_a_gate, fit_typed_propagation
from qts.propagation.equity.graph import LinkGraph
from qts.propagation.equity.labels import market_model_abnormal_return
from qts.propagation.equity.model import RELATIONS, RelationTypedPropagation, build_typed_adjacency
from qts.propagation.equity.samples import EventSample
from qts.propagation.equity.universe import EquityUniverse, load_universe

__all__ = [
    "Article",
    "CoMentionEdge",
    "EarningsEvent",
    "EarningsRow",
    "EconomicLink",
    "EconomicLinkClassifier",
    "EquityCorrelationalBaseline",
    "EquityUniverse",
    "EventSample",
    "LinkGraph",
    "PathAReport",
    "RELATIONS",
    "RawArticle",
    "RelationTypedPropagation",
    "assemble_event_samples",
    "build_comention_edges",
    "build_path_a_dataset",
    "build_typed_adjacency",
    "compute_sue",
    "evaluate_path_a_gate",
    "extract_tickers",
    "fit_typed_propagation",
    "load_fnspid_articles",
    "load_universe",
    "market_beta",
    "market_model_abnormal_return",
    "momentum_12_1",
    "node_feature_vector",
    "normalize_yf_earnings",
    "realised_vol",
]
