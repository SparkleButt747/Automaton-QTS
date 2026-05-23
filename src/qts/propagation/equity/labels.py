"""Market-model abnormal returns: the prediction targets (spec §6, §7)."""

from __future__ import annotations

import numpy as np


def market_model_abnormal_return(
    *,
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    event_idx: int,
    k: int = 3,
    est_window: int = 250,
) -> float:
    """Cumulative abnormal return over ``[event_idx, event_idx + k)``.

    Estimate (alpha, beta) by OLS on the pre-event window ``[event_idx - est_window, event_idx)``
    (point-in-time), then CAR = sum of (r_asset - (alpha + beta * r_market)) over the event window.
    """
    a = np.asarray(asset_returns, dtype=float)
    m = np.asarray(market_returns, dtype=float)
    lo = event_idx - est_window
    if lo < 0:
        raise ValueError("not enough pre-event history for the estimation window")
    am, mm = a[lo:event_idx], m[lo:event_idx]
    var = float(np.var(mm, ddof=1))
    beta = float(np.cov(am, mm)[0, 1] / var) if var > 0 else 0.0
    alpha = float(np.mean(am) - beta * np.mean(mm))
    ev_a, ev_m = a[event_idx : event_idx + k], m[event_idx : event_idx + k]
    abnormal = ev_a - (alpha + beta * ev_m)
    return float(np.sum(abnormal))
