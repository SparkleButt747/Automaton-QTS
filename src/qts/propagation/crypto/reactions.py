"""BTC-adjusted abnormal returns: the contagion prediction targets (spec §8). BTC is the market
factor (the 'everything dumps with BTC' component is removed). The source token's own entry in the
reactions vector IS the do() seed/merit (realized-move seeding — the equity 'Design B')."""

from __future__ import annotations

import numpy as np

from qts.propagation.equity.labels import market_model_abnormal_return


def _returns(closes: np.ndarray) -> np.ndarray:
    closes = np.asarray(closes, dtype=float)
    return np.diff(closes) / closes[:-1]


def btc_adjusted_car(
    *,
    token_closes: np.ndarray,
    btc_closes: np.ndarray,
    event_close_idx: int,
    horizon: int,
    est_window: int,
) -> float:
    """Cumulative abnormal return of ``token`` vs BTC over ``[event, event+horizon)`` hourly bars.

    ``event_close_idx`` indexes the CLOSES array at the event bar; (alpha, beta) vs BTC are
    estimated on the ``est_window`` returns immediately before it. Wraps
    ``market_model_abnormal_return`` (returns series, where return index i == close index i)."""
    return market_model_abnormal_return(
        asset_returns=_returns(token_closes),
        market_returns=_returns(btc_closes),
        event_idx=event_close_idx,
        k=horizon,
        est_window=est_window,
    )
