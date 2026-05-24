"""T-CRYPTO-FEAT-*: point-in-time crypto node features [cluster one-hot | beta, vol, ret, stable, exch]."""

from __future__ import annotations

import numpy as np
from qts.propagation.crypto.features import crypto_node_feature_vector

CLUSTERS = ("DeFiLending", "ExchangeToken", "L1")


def test_vector_layout_and_flags() -> None:  # T-CRYPTO-FEAT-1
    closes = 100.0 * np.cumprod(1 + np.full(300, 0.001))  # steady uptrend
    btc = 100.0 * np.cumprod(1 + np.full(300, 0.001))
    v = crypto_node_feature_vector(
        cluster="L1",
        clusters=CLUSTERS,
        token_closes=closes,
        btc_closes=btc,
        is_stablecoin=False,
        is_exchange_token=False,
    )
    assert v.shape == (len(CLUSTERS) + 5,)  # one-hot + [beta, vol, ret, stable, exch]
    assert v[CLUSTERS.index("L1")] == 1.0 and v[:3].sum() == 1.0  # one cluster hot
    assert v[-2] == 0.0 and v[-1] == 0.0  # not stablecoin, not exchange token


def test_flags_and_ret_sign() -> None:  # T-CRYPTO-FEAT-2
    up = 100.0 * np.cumprod(1 + np.full(300, 0.002))
    btc = 100.0 * np.cumprod(1 + np.full(300, 0.001))
    v = crypto_node_feature_vector(
        cluster="ExchangeToken",
        clusters=CLUSTERS,
        token_closes=up,
        btc_closes=btc,
        is_stablecoin=False,
        is_exchange_token=True,
    )
    assert v[-1] == 1.0  # exchange-token flag set
    assert v[len(CLUSTERS) + 2] > 0.0  # ret feature positive on an uptrend
