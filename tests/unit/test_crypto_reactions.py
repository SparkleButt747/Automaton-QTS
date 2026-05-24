"""T-CRYPTO-REACT-*: BTC-adjusted CAR on a synthetic hourly panel with known answers."""

from __future__ import annotations

import numpy as np
from qts.propagation.crypto.reactions import btc_adjusted_car


def test_token_drops_btc_flat_gives_negative_car() -> None:  # T-CRYPTO-REACT-1
    # 400 flat hourly closes, then BTC stays flat while the token drops over the 3h window.
    n = 400
    btc = np.full(n, 100.0)
    token = np.full(n, 50.0)
    token[300] = 50.0 * 0.99  # -1% each step across the event window [299,302)
    token[301] = 50.0 * 0.98
    token[302] = 50.0 * 0.97
    car = btc_adjusted_car(
        token_closes=token, btc_closes=btc, event_close_idx=299, horizon=3, est_window=250
    )
    assert car < -0.02  # clearly negative abnormal move, BTC contributed nothing


def test_token_tracks_btc_gives_near_zero_car() -> None:  # T-CRYPTO-REACT-2
    # token == 0.5 * BTC exactly: abnormal return ~ 0.
    rng = np.random.default_rng(0)
    steps = rng.normal(0, 0.01, 399)
    btc = 100.0 * np.cumprod(np.concatenate([[1.0], 1 + steps]))
    token = 0.5 * btc
    car = btc_adjusted_car(
        token_closes=token, btc_closes=btc, event_close_idx=300, horizon=3, est_window=250
    )
    assert abs(car) < 1e-6  # token is a pure multiple of BTC -> zero abnormal return


def test_raises_without_estimation_history() -> None:  # T-CRYPTO-REACT-3
    btc = np.full(300, 100.0)
    token = np.full(300, 10.0)
    import pytest

    with pytest.raises(ValueError, match="pre-event history"):
        btc_adjusted_car(
            token_closes=token, btc_closes=btc, event_close_idx=10, horizon=3, est_window=250
        )
