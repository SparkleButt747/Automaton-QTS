"""T-PATHA-FEAT-*: point-in-time node features."""

from __future__ import annotations

import numpy as np

from qts.propagation.equity.features import (
    market_beta,
    momentum_12_1,
    node_feature_vector,
    realised_vol,
)


def test_market_beta_recovers_known_slope() -> None:  # T-PATHA-FEAT-1
    rng = np.random.default_rng(0)
    mkt = rng.normal(0, 0.01, 500)
    asset = 1.8 * mkt + rng.normal(0, 1e-4, 500)
    assert abs(market_beta(asset, mkt) - 1.8) < 0.05


def test_realised_vol_positive_and_scales() -> None:  # T-PATHA-FEAT-2
    rng = np.random.default_rng(1)
    quiet = rng.normal(0, 0.005, 300)
    wild = rng.normal(0, 0.05, 300)
    assert realised_vol(wild) > realised_vol(quiet) > 0.0


def test_momentum_12_1_excludes_recent_month() -> None:  # T-PATHA-FEAT-3
    closes = np.concatenate([np.full(231, 100.0), np.linspace(100.0, 150.0, 21)])
    assert abs(momentum_12_1(closes)) < 1e-6


def test_node_feature_vector_dim_and_sector_onehot() -> None:  # T-PATHA-FEAT-4
    sectors = ("Autos", "Retail", "Semiconductors")
    rng = np.random.default_rng(2)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
    mkt = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
    vec = node_feature_vector(
        sector="Retail", sectors=sectors, log_mktcap=25.0, closes=closes, market_closes=mkt
    )
    assert vec.shape == (len(sectors) + 4,)  # sector one-hot + [log_mktcap, beta, momentum, vol]
    assert vec[1] == 1.0 and vec[0] == 0.0 and vec[2] == 0.0
