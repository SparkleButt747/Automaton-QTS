"""Combined alpha signal computation.

Combines normalised technical indicators and sentiment into a single
alpha score, scaled by the current volatility regime.
"""
from __future__ import annotations

import numpy as np

from qts.config import StrategyParams
from qts.models.base import SignalSnapshot, VolRegime
from qts.signals.indicators import normalise_rsi


def _normalise_macd_histogram(histogram: float) -> float:
    """Normalise MACD histogram to [-1, 1] using tanh squashing.

    Args:
        histogram: Raw MACD histogram value.

    Returns:
        Float in (-1, 1).
    """
    if np.isnan(histogram):
        return 0.0
    return float(np.tanh(histogram))


def _normalise_bb_position(bb_position: float) -> float:
    """Map BB position from [0, 1] to [-1, 1].

    A value of 0 (at lower band) maps to -1 (bearish pressure),
    a value of 1 (at upper band) maps to +1 (bullish pressure).

    Args:
        bb_position: BB position in [0, 1].

    Returns:
        Float in [-1, 1].
    """
    if np.isnan(bb_position):
        return 0.0
    return float(2.0 * bb_position - 1.0)


def _normalise_momentum(momentum: float) -> float:
    """Normalise momentum to [-1, 1] using tanh squashing.

    Args:
        momentum: Raw rate-of-change momentum value.

    Returns:
        Float in (-1, 1).
    """
    if np.isnan(momentum):
        return 0.0
    return float(np.tanh(momentum * 10.0))  # scale factor for typical values


def _regime_scalar(regime: VolRegime) -> float:
    """Return the position-size scalar for the given regime.

    HIGH volatility -> full signal (1.0).
    LOW volatility  -> dampened signal (0.5).

    Args:
        regime: Current VolRegime.

    Returns:
        1.0 for HIGH, 0.5 for LOW.
    """
    return 1.0 if regime == VolRegime.HIGH else 0.5


def combined_alpha(snapshot: SignalSnapshot, params: StrategyParams) -> float:
    """Compute the combined alpha signal from a SignalSnapshot.

    Formula:
        rsi_norm    = normalise_rsi(snapshot.rsi)
        macd_norm   = tanh(snapshot.macd_histogram)
        bb_norm     = 2 * snapshot.bb_position - 1
        mom_norm    = tanh(snapshot.momentum_5 * 10)
        raw_alpha   = (w_rsi * rsi_norm
                       + w_macd * macd_norm
                       + w_bb   * bb_norm
                       + w_mom  * mom_norm
                       + w_sentiment * snapshot.sentiment_score)
        regime_scl  = 1.0 if HIGH else 0.5
        alpha       = clip(raw_alpha * regime_scl, -1, 1)

    Output range: [-1, 1].

    Args:
        snapshot: Complete signal state at a point in time.
        params: Strategy parameters containing signal weights.

    Returns:
        Combined alpha float in [-1, 1].
    """
    w = params.weights

    rsi_val = normalise_rsi(snapshot.rsi) if not np.isnan(snapshot.rsi) else 0.0
    macd_val = _normalise_macd_histogram(snapshot.macd_histogram)
    bb_val = _normalise_bb_position(snapshot.bb_position)
    mom_val = _normalise_momentum(snapshot.momentum_5)
    sentiment_val = float(np.clip(snapshot.sentiment_score, -1.0, 1.0))

    raw_alpha = (
        w.w_rsi * rsi_val
        + w.w_macd * macd_val
        + w.w_bb * bb_val
        + w.w_mom * mom_val
        + w.w_sentiment * sentiment_val
    )

    scalar = _regime_scalar(snapshot.vol_regime)
    return float(np.clip(raw_alpha * scalar, -1.0, 1.0))
