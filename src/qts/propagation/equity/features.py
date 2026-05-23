"""Point-in-time node features xi — the basis the bilinear operator matches relations from."""

from __future__ import annotations

import numpy as np

TRADING_DAYS_MONTH = 21
TRADING_DAYS_YEAR = 252


def _returns(closes: np.ndarray) -> np.ndarray:
    closes = np.asarray(closes, dtype=float)
    return np.diff(closes) / closes[:-1]


def realised_vol(returns_or_closes: np.ndarray, *, are_returns: bool = True) -> float:
    r = np.asarray(returns_or_closes, dtype=float)
    if not are_returns:
        r = _returns(r)
    return float(np.std(r))


def momentum_12_1(closes: np.ndarray) -> float:
    """12-month minus most-recent-month return: closes[-21]/closes[-252] - 1 (skips last month)."""
    closes = np.asarray(closes, dtype=float)
    if len(closes) < TRADING_DAYS_YEAR:
        return 0.0
    return float(closes[-TRADING_DAYS_MONTH] / closes[-TRADING_DAYS_YEAR] - 1.0)


def market_beta(asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
    a = np.asarray(asset_returns, dtype=float)
    m = np.asarray(market_returns, dtype=float)
    var = float(np.var(m))
    if var == 0.0:
        return 0.0
    return float(np.cov(a, m)[0, 1] / var)


def node_feature_vector(
    *,
    sector: str,
    sectors: tuple[str, ...],
    log_mktcap: float,
    closes: np.ndarray,
    market_closes: np.ndarray,
) -> np.ndarray:
    """[sector one-hot | log_mktcap, beta, momentum_12_1, realised_vol]. All from data < event t."""
    onehot = np.zeros(len(sectors), dtype=float)
    onehot[sectors.index(sector)] = 1.0
    ar, mr = _returns(closes), _returns(market_closes)
    n = min(len(ar), len(mr))
    extras = np.array(
        [
            float(log_mktcap),
            market_beta(ar[-n:], mr[-n:]),
            momentum_12_1(closes),
            realised_vol(ar),
        ],
        dtype=float,
    )
    return np.concatenate([onehot, extras])
