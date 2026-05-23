"""T-PATHA-LABEL-*: market-model abnormal-return reaction labels."""

from __future__ import annotations

import numpy as np

from qts.propagation.equity.labels import market_model_abnormal_return


def test_abnormal_return_isolates_idiosyncratic_shock() -> None:  # T-PATHA-LABEL-1
    rng = np.random.default_rng(0)
    n = 300
    mkt = rng.normal(0, 0.01, n)
    asset = 1.5 * mkt.copy()
    est_window = 250
    event_idx = est_window
    asset[event_idx] += 0.03
    car = market_model_abnormal_return(
        asset_returns=asset, market_returns=mkt, event_idx=event_idx, k=1, est_window=est_window
    )
    assert abs(car - 0.03) < 1e-3


def test_abnormal_return_zero_when_pure_market() -> None:  # T-PATHA-LABEL-2
    rng = np.random.default_rng(1)
    n = 300
    mkt = rng.normal(0, 0.01, n)
    asset = 0.8 * mkt
    car = market_model_abnormal_return(
        asset_returns=asset, market_returns=mkt, event_idx=260, k=3, est_window=250
    )
    assert abs(car) < 1e-6


def test_window_k_accumulates() -> None:  # T-PATHA-LABEL-3
    n = 300
    mkt = np.zeros(n)
    asset = np.zeros(n)
    asset[255:258] = 0.01
    car = market_model_abnormal_return(
        asset_returns=asset, market_returns=mkt, event_idx=255, k=3, est_window=250
    )
    assert abs(car - 0.03) < 1e-9
