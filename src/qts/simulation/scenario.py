"""Stress scenario injection for backtest robustness testing.

Provides methods to synthetically alter bar sequences and signal snapshots
to simulate adverse market conditions.
"""
from __future__ import annotations

import copy
import logging
from datetime import timedelta
from typing import Sequence

import numpy as np

from qts.models.base import Bar, SignalSnapshot

logger = logging.getLogger(__name__)


class StressScenarioRunner:
    """Inject stress scenarios into bar / snapshot sequences.

    All methods return *new* lists; the originals are not mutated.
    """

    # ------------------------------------------------------------------
    # Flash crash
    # ------------------------------------------------------------------

    def flash_crash(
        self,
        bars: Sequence[Bar],
        magnitude: float = -0.15,
        duration_bars: int = 5,
        start_index: int | None = None,
    ) -> list[Bar]:
        """Inject a flash crash starting at a given bar.

        Prices fall linearly by `magnitude` over `duration_bars` bars,
        then recover linearly back to the pre-crash level.

        Args:
            bars: Original sequence of bars.
            magnitude: Fractional price change at nadir (e.g. -0.15 = -15%).
                       Must be negative.
            duration_bars: Number of bars for the decline phase. Recovery
                           takes the same number of bars.
            start_index: Bar index where the crash begins. Defaults to
                         midpoint of the sequence.

        Returns:
            New list of Bar objects with the crash injected.
        """
        if magnitude >= 0:
            raise ValueError(f"magnitude must be negative, got {magnitude}")
        if duration_bars < 1:
            raise ValueError(f"duration_bars must be >= 1, got {duration_bars}")

        result = list(bars)
        n = len(result)
        if n == 0:
            return result

        if start_index is None:
            start_index = n // 3

        end_crash = min(start_index + duration_bars, n)
        end_recovery = min(end_crash + duration_bars, n)
        baseline_close = result[start_index].close

        # Decline phase
        for i in range(start_index, end_crash):
            frac = (i - start_index + 1) / duration_bars
            multiplier = 1.0 + magnitude * frac
            result[i] = _scale_bar(result[i], multiplier)

        # Recovery phase
        nadir_mult = 1.0 + magnitude
        for i in range(end_crash, end_recovery):
            frac = (i - end_crash + 1) / duration_bars
            multiplier = nadir_mult + (-magnitude) * frac  # rises back
            result[i] = _scale_bar(result[i], multiplier)

        logger.debug(
            "flash_crash: start=%d, magnitude=%.2f, duration=%d",
            start_index,
            magnitude,
            duration_bars,
        )
        return result

    # ------------------------------------------------------------------
    # Sentiment spike
    # ------------------------------------------------------------------

    def sentiment_spike(
        self,
        snapshots: Sequence[SignalSnapshot],
        magnitude: float = -0.9,
        start_index: int | None = None,
        duration_bars: int = 5,
    ) -> list[SignalSnapshot]:
        """Inject a sentiment shock into a snapshot sequence.

        Sets the sentiment_score on the affected snapshots to `magnitude`
        and resets combined_alpha to 0.0 (caller should recompute if needed).

        Args:
            snapshots: Original sequence of SignalSnapshot objects.
            magnitude: Sentiment value to inject, in [-1, 1].
            start_index: Index where the spike begins. Defaults to midpoint.
            duration_bars: Number of bars the shock persists.

        Returns:
            New list of SignalSnapshot objects with the shock injected.
        """
        if not -1.0 <= magnitude <= 1.0:
            raise ValueError(f"magnitude must be in [-1, 1], got {magnitude}")

        result = list(snapshots)
        n = len(result)
        if n == 0:
            return result

        if start_index is None:
            start_index = n // 3

        end = min(start_index + duration_bars, n)

        for i in range(start_index, end):
            s = result[i]
            # SignalSnapshot is frozen; replace with updated values
            result[i] = SignalSnapshot(
                timestamp=s.timestamp,
                symbol=s.symbol,
                rsi=s.rsi,
                macd_histogram=s.macd_histogram,
                macd_line=s.macd_line,
                macd_signal=s.macd_signal,
                bb_position=s.bb_position,
                bb_upper=s.bb_upper,
                bb_lower=s.bb_lower,
                atr=s.atr,
                momentum_5=s.momentum_5,
                vol_regime=s.vol_regime,
                vol_regime_confidence=s.vol_regime_confidence,
                sentiment_score=magnitude,
                combined_alpha=0.0,  # invalidated; recompute if needed
            )

        logger.debug(
            "sentiment_spike: start=%d, magnitude=%.2f, duration=%d",
            start_index,
            magnitude,
            duration_bars,
        )
        return result

    # ------------------------------------------------------------------
    # Data gap
    # ------------------------------------------------------------------

    def data_gap(
        self,
        bars: Sequence[Bar],
        gap_duration_bars: int = 30,
        start_index: int | None = None,
    ) -> list[Bar]:
        """Simulate a data gap by replacing bars with stale (flat) data.

        The bars within the gap are replaced with repeated copies of the
        bar immediately preceding the gap. This simulates a feed outage
        or exchange halt.

        Args:
            bars: Original sequence of bars.
            gap_duration_bars: Number of bars to replace.
            start_index: Index where the gap begins. Defaults to midpoint.

        Returns:
            New list of Bar objects with the gap injected.
        """
        if gap_duration_bars < 1:
            raise ValueError(f"gap_duration_bars must be >= 1, got {gap_duration_bars}")

        result = list(bars)
        n = len(result)
        if n == 0:
            return result

        if start_index is None:
            start_index = n // 3

        if start_index == 0:
            # No previous bar; use the first bar as the stale reference
            stale_bar = result[0]
        else:
            stale_bar = result[start_index - 1]

        end = min(start_index + gap_duration_bars, n)
        for i in range(start_index, end):
            # Preserve timestamps but freeze price data
            result[i] = Bar(
                timestamp=result[i].timestamp,
                symbol=result[i].symbol,
                open=stale_bar.close,
                high=stale_bar.close,
                low=stale_bar.close,
                close=stale_bar.close,
                volume=0.0,
                bar_count=0,
            )

        logger.debug(
            "data_gap: start=%d, duration=%d, stale_close=%.4f",
            start_index,
            gap_duration_bars,
            stale_bar.close,
        )
        return result


# ── Helpers ───────────────────────────────────────────────────────────────────


def _scale_bar(bar: Bar, multiplier: float) -> Bar:
    """Return a new Bar with OHLC prices scaled by multiplier.

    Args:
        bar: Original Bar.
        multiplier: Scaling factor applied to open/high/low/close.

    Returns:
        New Bar with scaled prices; volume unchanged.
    """
    return Bar(
        timestamp=bar.timestamp,
        symbol=bar.symbol,
        open=bar.open * multiplier,
        high=bar.high * multiplier,
        low=bar.low * multiplier,
        close=bar.close * multiplier,
        volume=bar.volume,
        bar_count=bar.bar_count,
    )
