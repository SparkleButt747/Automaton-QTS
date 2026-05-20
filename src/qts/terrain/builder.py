"""TerrainBuilder — constructs MarketTerrain from bars + regime metadata.

Analogous to TrackGenerator in the racing lab. Takes raw bar data and
regime annotations, computes derived features (volatility surface,
support/resistance, liquidity profile), and produces a frozen
MarketTerrain ready for backtesting.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import numpy as np

from qts.models.base import Bar, VolLevel
from qts.models.terrain import (
    Level,
    MacroRegime,
    MarketEvent,
    MarketTerrain,
    RegimeSpan,
)
from qts.nautilus.catalog import ensure_catalog
from qts.signals.indicators import compute_atr

logger = logging.getLogger(__name__)


class TerrainBuilder:
    """Builds MarketTerrain objects from raw components.

    Usage::

        builder = TerrainBuilder()
        terrain = (
            builder
            .set_identity("covid_crash", "SPY")
            .set_bars(bars)
            .set_regime(regime)
            .compute_features()
            .build()
        )
    """

    def __init__(self) -> None:
        self._name: str = ""
        self._symbol: str = ""
        self._bars: list[Bar] = []
        self._regime: MacroRegime | None = None
        self._regime_spans: list[RegimeSpan] = []
        self._events: list[MarketEvent] = []
        self._levels: list[Level] = []
        self._volatility_surface: list[float] = []
        self._liquidity_profile: list[float] = []
        self._catalog_root: Path | None = None

    def set_identity(self, name: str, symbol: str) -> TerrainBuilder:
        """Set the terrain name and symbol."""
        self._name = name
        self._symbol = symbol
        return self

    def set_bars(self, bars: Sequence[Bar]) -> TerrainBuilder:
        """Set the bar data (historical or synthetic)."""
        self._bars = list(bars)
        return self

    def set_regime(self, regime: MacroRegime) -> TerrainBuilder:
        """Set the primary macro regime for the terrain."""
        self._regime = regime
        return self

    def add_regime_span(self, span: RegimeSpan) -> TerrainBuilder:
        """Add a time-bounded sub-regime within the terrain."""
        self._regime_spans.append(span)
        return self

    def add_event(self, event: MarketEvent) -> TerrainBuilder:
        """Add a market event to the event calendar."""
        self._events.append(event)
        return self

    def add_level(self, level: Level) -> TerrainBuilder:
        """Add a support/resistance level."""
        self._levels.append(level)
        return self

    def set_catalog_root(self, path: Path) -> TerrainBuilder:
        """Set the Parquet catalog root for Nautilus data persistence."""
        self._catalog_root = path
        return self

    def compute_features(self) -> TerrainBuilder:
        """Auto-compute volatility surface and liquidity profile from bars.

        Volatility surface: rolling ATR normalised to [0, 1].
        Liquidity profile: volume-based proxy normalised to [0, 1].
        """
        if not self._bars:
            return self

        highs = np.array([b.high for b in self._bars], dtype=np.float64)
        lows = np.array([b.low for b in self._bars], dtype=np.float64)
        closes = np.array([b.close for b in self._bars], dtype=np.float64)
        volumes = np.array([b.volume for b in self._bars], dtype=np.float64)

        # Volatility surface from ATR
        atr = compute_atr(highs, lows, closes)
        valid_atr = atr[~np.isnan(atr)]
        if len(valid_atr) > 0:
            atr_min, atr_max = float(np.min(valid_atr)), float(np.max(valid_atr))
            atr_range = atr_max - atr_min if atr_max > atr_min else 1.0
            self._volatility_surface = [
                float((v - atr_min) / atr_range) if not np.isnan(v) else 0.5
                for v in atr
            ]
        else:
            self._volatility_surface = [0.5] * len(self._bars)

        # Liquidity profile from volume
        valid_vol = volumes[volumes > 0]
        if len(valid_vol) > 0:
            vol_median = float(np.median(valid_vol))
            self._liquidity_profile = [
                min(1.0, float(v / vol_median)) if vol_median > 0 else 1.0
                for v in volumes
            ]
        else:
            self._liquidity_profile = [1.0] * len(self._bars)

        return self

    def compute_support_resistance(self, lookback: int = 20) -> TerrainBuilder:
        """Auto-detect support/resistance levels from price pivots."""
        if len(self._bars) < lookback * 2:
            return self

        closes = [b.close for b in self._bars]
        levels: list[Level] = []

        for i in range(lookback, len(closes) - lookback):
            window = closes[i - lookback : i + lookback + 1]
            price = closes[i]

            # Local minimum → support
            if price == min(window):
                levels.append(
                    Level(
                        price=price,
                        strength=0.5,
                        level_type="support",
                        first_seen=self._bars[i].timestamp,
                    )
                )

            # Local maximum → resistance
            if price == max(window):
                levels.append(
                    Level(
                        price=price,
                        strength=0.5,
                        level_type="resistance",
                        first_seen=self._bars[i].timestamp,
                    )
                )

        self._levels = levels
        return self

    def build(self) -> MarketTerrain:
        """Build the frozen MarketTerrain.

        Also writes bars to the Parquet catalog if a catalog root is set.

        Raises:
            ValueError: If required fields are missing.
        """
        if not self._name:
            raise ValueError("Terrain name is required")
        if not self._symbol:
            raise ValueError("Terrain symbol is required")
        if self._regime is None:
            raise ValueError("Terrain regime is required")
        # Bars are optional at build time — YAML-loaded terrains may not have
        # bar data yet (it gets loaded lazily when the terrain is actually used).
        # The terrain is still valid as a metadata container.

        # Persist to Parquet catalog if configured
        catalog_path: Path | None = None
        if self._catalog_root is not None and self._bars:
            catalog_path = ensure_catalog(self._bars, self._catalog_root)

        # Derive start/end from bars if available, else use placeholder
        from datetime import datetime, UTC  # noqa: PLC0415

        start = self._bars[0].timestamp if self._bars else datetime(2000, 1, 1, tzinfo=UTC)
        end = self._bars[-1].timestamp if self._bars else datetime(2000, 1, 1, tzinfo=UTC)

        terrain = MarketTerrain(
            name=self._name,
            symbol=self._symbol,
            start=start,
            end=end,
            regime=self._regime,
            bars=tuple(self._bars),
            regime_spans=tuple(self._regime_spans),
            volatility_surface=tuple(self._volatility_surface),
            support_resistance=tuple(self._levels),
            liquidity_profile=tuple(self._liquidity_profile),
            event_calendar=tuple(sorted(self._events, key=lambda e: e.timestamp)),
            catalog_path=catalog_path,
        )

        logger.info(
            "Built terrain '%s': %d bars, %d events, %d levels, %d regime spans",
            terrain.name,
            len(terrain.bars),
            len(terrain.event_calendar),
            len(terrain.support_resistance),
            len(terrain.regime_spans),
        )
        return terrain
