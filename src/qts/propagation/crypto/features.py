"""Point-in-time crypto node features xi (spec §6, v0 subset): the basis the bilinear operator
matches relations from. All computed from data strictly before the event. v0 = klines + config only
(log-mcap / TVL deferred — need external data)."""

from __future__ import annotations

import numpy as np

from qts.propagation.equity.features import market_beta, realised_vol


def _returns(closes: np.ndarray) -> np.ndarray:
    closes = np.asarray(closes, dtype=float)
    return np.diff(closes) / closes[:-1]


def crypto_node_feature_vector(
    *,
    cluster: str,
    clusters: tuple[str, ...],
    token_closes: np.ndarray,
    btc_closes: np.ndarray,
    is_stablecoin: bool,
    is_exchange_token: bool,
) -> np.ndarray:
    """Return [cluster one-hot | btc_beta, realised_vol, window_return, stable_flag, exch_flag]."""
    onehot = np.zeros(len(clusters), dtype=float)
    onehot[clusters.index(cluster)] = 1.0
    tr, br = _returns(token_closes), _returns(btc_closes)
    n = min(len(tr), len(br))
    closes = np.asarray(token_closes, dtype=float)
    window_return = float(closes[-1] / closes[0] - 1.0) if len(closes) > 1 else 0.0
    extras = np.array(
        [
            market_beta(tr[-n:], br[-n:]),
            realised_vol(tr),
            window_return,
            1.0 if is_stablecoin else 0.0,
            1.0 if is_exchange_token else 0.0,
        ],
        dtype=float,
    )
    return np.concatenate([onehot, extras])
