"""Market terrain data models — the trading equivalent of Track + TrackAnnotations.

A MarketTerrain is a pre-computed environment that a strategy navigates,
analogous to a racing track with curvature, width, and corner annotations.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from qts.models.base import (
    Bar,
    Catalyst,
    LiquidityLevel,
    SentimentLevel,
    Trend,
    VolLevel,
)


@dataclass(frozen=True, slots=True)
class MacroRegime:
    """The 'track geometry' — what kind of market environment we are in.

    Produced by the Macro Engine (classifier mode for real data,
    generator mode for synthetic scenarios).
    """

    trend: Trend
    volatility: VolLevel
    liquidity: LiquidityLevel
    sentiment: SentimentLevel
    catalyst: Catalyst

    # Conditioning signals
    expected_drift: float  # annualised, like track gradient
    expected_vol: float  # like curvature intensity
    correlation_regime: float  # cross-asset correlation

    # Human-readable description
    scenario_description: str


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """A scheduled or detected market event — like a corner annotation on a track."""

    timestamp: datetime
    event_type: str  # e.g. "earnings", "fomc", "cpi_release", "flash_crash"
    description: str
    impact_magnitude: float  # estimated impact [-1, 1]


@dataclass(frozen=True, slots=True)
class Level:
    """Support/resistance price level — like track boundaries."""

    price: float
    strength: float  # 0-1, how many times tested
    level_type: str  # "support" or "resistance"
    first_seen: datetime


@dataclass(frozen=True, slots=True)
class RegimeSpan:
    """A time-bounded macro regime — allows regime changes within a terrain."""

    start: datetime
    end: datetime
    regime: MacroRegime


@dataclass(frozen=True)
class MarketTerrain:
    """The complete 'track' — pre-computed environment for a strategy to navigate.

    A MarketTerrain wraps historical or synthetic bar data with macro regime
    annotations, volatility surface, support/resistance levels, liquidity
    profile, and an event calendar. It is the trading equivalent of
    Track + TrackAnnotations in the racing lab.

    The terrain is frozen after construction. Strategies receive it as
    read-only context; the Nautilus BacktestNode consumes the bar data
    via a ParquetDataCatalog.
    """

    # Identity
    name: str
    symbol: str
    start: datetime
    end: datetime

    # The primary regime (may be overridden by regime_spans for sub-periods)
    regime: MacroRegime

    # Bar data (historical or synthetic)
    bars: Sequence[Bar] = field(default_factory=tuple)

    # Time-varying annotations
    regime_spans: Sequence[RegimeSpan] = field(default_factory=tuple)
    volatility_surface: Sequence[float] = field(default_factory=tuple)
    support_resistance: Sequence[Level] = field(default_factory=tuple)
    liquidity_profile: Sequence[float] = field(default_factory=tuple)
    event_calendar: Sequence[MarketEvent] = field(default_factory=tuple)

    # Parquet catalog path (for Nautilus ingestion)
    catalog_path: Path | None = None

    # ------------------------------------------------------------------
    # Query methods (like Track.get_curvature_at(s))
    # ------------------------------------------------------------------

    def regime_at(self, t: datetime) -> MacroRegime:
        """Return the macro regime active at time t.

        Falls back to the primary regime if no regime_spans cover t.
        """
        for span in self.regime_spans:
            if span.start <= t <= span.end:
                return span.regime
        return self.regime

    def liquidity_at(self, t: datetime) -> float:
        """Return interpolated liquidity at time t.

        Returns 1.0 (normal) if no liquidity profile is set.
        """
        if not self.liquidity_profile or not self.bars:
            return 1.0
        idx = self._bar_index_at(t)
        if idx < len(self.liquidity_profile):
            return self.liquidity_profile[idx]
        return self.liquidity_profile[-1] if self.liquidity_profile else 1.0

    def volatility_at(self, t: datetime) -> float:
        """Return the volatility surface value at time t.

        Returns the regime's expected_vol if no surface is set.
        """
        if not self.volatility_surface or not self.bars:
            return self.regime.expected_vol
        idx = self._bar_index_at(t)
        if idx < len(self.volatility_surface):
            return self.volatility_surface[idx]
        return self.volatility_surface[-1] if self.volatility_surface else self.regime.expected_vol

    def nearest_event(self, t: datetime) -> MarketEvent | None:
        """Return the nearest future event from t, or None."""
        if not self.event_calendar:
            return None
        future = [e for e in self.event_calendar if e.timestamp >= t]
        return min(future, key=lambda e: e.timestamp) if future else None

    def events_in_range(self, start: datetime, end: datetime) -> list[MarketEvent]:
        """Return all events within [start, end]."""
        return [e for e in self.event_calendar if start <= e.timestamp <= end]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bar_index_at(self, t: datetime) -> int:
        """Find the bar index closest to time t via binary search."""
        if not self.bars:
            return 0
        timestamps = [b.timestamp for b in self.bars]
        idx = bisect.bisect_right(timestamps, t)
        return max(0, idx - 1)
