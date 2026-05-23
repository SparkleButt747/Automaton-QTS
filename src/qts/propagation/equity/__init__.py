"""Path A v2 — real-equity economic-link graph (Phase 1 + Phase 2)."""

from qts.propagation.equity.comention import (
    Article,
    CoMentionEdge,
    RawArticle,
    build_comention_edges,
    extract_tickers,
    load_fnspid_articles,
)
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
from qts.propagation.equity.graph import LinkGraph
from qts.propagation.equity.labels import market_model_abnormal_return
from qts.propagation.equity.samples import EventSample
from qts.propagation.equity.universe import EquityUniverse, load_universe

__all__ = [
    "Article",
    "CoMentionEdge",
    "EarningsEvent",
    "EarningsRow",
    "EconomicLink",
    "EconomicLinkClassifier",
    "EquityUniverse",
    "EventSample",
    "LinkGraph",
    "RawArticle",
    "build_comention_edges",
    "compute_sue",
    "extract_tickers",
    "load_fnspid_articles",
    "load_universe",
    "market_beta",
    "market_model_abnormal_return",
    "momentum_12_1",
    "node_feature_vector",
    "normalize_yf_earnings",
    "realised_vol",
]
