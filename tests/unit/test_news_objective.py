"""T-NOBJ-1..3: CVaR-of-excess objective + degenerate-guard + valid inner params."""

from __future__ import annotations

import numpy as np

from qts.optimisation.news_objective import (
    _build_inner_params,
    _cvar_of_excess,
    _hold_return,
)


def test_build_inner_params_satisfies_exit_below_entry() -> None:  # T-NOBJ-1
    # entry at the range floor must still yield exit < entry (validator)
    p = _build_inner_params(entry_threshold=0.02)
    assert p.exit_threshold < p.entry_threshold
    assert p.entry_threshold == 0.02


def test_cvar_picks_lower_quantile() -> None:  # T-NOBJ-2
    excesses = [0.10, 0.05, -0.02, 0.08, -0.10]
    expected = float(np.quantile(excesses, 0.25))
    assert abs(_cvar_of_excess(excesses, 0.25) - expected) < 1e-12


def test_hold_return_from_bars() -> None:  # T-NOBJ-3
    class _Bar:
        def __init__(self, close: float) -> None:
            self.close = close

    class _Terrain:
        bars = [_Bar(100.0), _Bar(110.0)]

    assert abs(_hold_return(_Terrain()) - 0.10) < 1e-12
