"""Shock detectors: turn a price panel into ContagionEvents — the live trigger, run
offline here for the gate. v0 fires on an idiosyncratic (BTC-adjusted) drawdown, matching
how the operator is trained (BTC-adjusted CARs) and filtering market-wide beta sell-offs.
A news/keyword detector can later implement the same ShockDetector protocol."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np

from qts.propagation.crypto.events import ContagionEvent
from qts.propagation.crypto.reactions import btc_adjusted_car


@runtime_checkable
class ShockDetector(Protocol):
    def detect(
        self, closes: dict[str, np.ndarray], grid: list[datetime], btc_closes: np.ndarray
    ) -> list[ContagionEvent]: ...


class IdiosyncraticDropDetector:
    """Fire when a token's trailing BTC-adjusted abnormal return over ``window`` bars
    falls below ``-threshold``; at most one event per token per ``cooldown`` bars."""

    def __init__(
        self,
        *,
        threshold: float = 0.15,
        window: int = 24,
        est_window: int = 720,
        cooldown: int = 72,
    ) -> None:
        self.threshold = threshold
        self.window = window
        self.est_window = est_window
        self.cooldown = cooldown

    def detect(
        self, closes: dict[str, np.ndarray], grid: list[datetime], btc_closes: np.ndarray
    ) -> list[ContagionEvent]:
        events: list[ContagionEvent] = []
        n = len(grid)
        first = self.est_window + self.window
        for token, tc in closes.items():
            if token == "BTC":  # noqa: S105
                continue
            last_fire = -self.cooldown - 1
            # re-estimates OLS over est_window each bar: fine offline, profile before live reuse
            for t in range(first, n):
                if t - last_fire <= self.cooldown:
                    continue
                # event started ``window`` bars ago; CAR covers [t-window, t)
                ar = btc_adjusted_car(
                    token_closes=tc,
                    btc_closes=btc_closes,
                    event_close_idx=t - self.window,
                    horizon=self.window,
                    est_window=self.est_window,
                )
                if ar < -self.threshold:
                    events.append(
                        ContagionEvent(
                            source_token=token,
                            timestamp=grid[t],
                            event_type="drop",
                            # usd_severity = fractional CAR magnitude (no USD notional offline)
                            usd_severity=float(abs(ar)),
                        )
                    )
                    last_fire = t
        return sorted(events, key=lambda e: e.timestamp)
