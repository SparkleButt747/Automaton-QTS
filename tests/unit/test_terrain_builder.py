"""Unit tests for qts.terrain.builder.

Covers:
- T-TBLDR-1  to T-TBLDR-13
- Validation errors on missing required fields
- compute_features: volatility_surface length matches bars, zero-volume fallback
- compute_support_resistance: level detection on oscillating prices, short-bars no-op
- Empty bars: compute_features is a no-op; build succeeds with placeholder timestamps
- set_catalog_root: catalog_path is None when bars are absent
- Fluent API returns self at each step
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from qts.models.base import Bar, Catalyst, LiquidityLevel, SentimentLevel, Trend, VolLevel
from qts.models.terrain import MacroRegime
from qts.terrain.builder import TerrainBuilder

# ── Helpers ───────────────────────────────────────────────────────────────────

_T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _make_regime(
    trend: Trend = Trend.BULL,
    vol: VolLevel = VolLevel.LOW,
    liq: LiquidityLevel = LiquidityLevel.ABUNDANT,
) -> MacroRegime:
    return MacroRegime(
        trend=trend,
        volatility=vol,
        liquidity=liq,
        sentiment=SentimentLevel.NEUTRAL,
        catalyst=Catalyst.NONE,
        expected_drift=0.10,
        expected_vol=0.15,
        correlation_regime=0.5,
        scenario_description="test regime",
    )


def _make_flat_bars(n: int, price: float = 100.0, volume: float = 1000.0) -> list[Bar]:
    """Bars with constant price — produces zero ATR range."""
    return [
        Bar(
            timestamp=_T0 + timedelta(days=i),
            symbol="SPY",
            open=price,
            high=price + 1.0,
            low=price - 1.0,
            close=price,
            volume=volume,
            bar_count=1,
        )
        for i in range(n)
    ]


def _make_oscillating_bars(n: int, amplitude: float = 10.0, base: float = 100.0) -> list[Bar]:
    """Bars with a sinusoidal close — forces clear local min/max for S/R detection."""
    bars = []
    for i in range(n):
        p = base + amplitude * math.sin(i * 0.2)
        bars.append(
            Bar(
                timestamp=_T0 + timedelta(days=i),
                symbol="SPY",
                open=p,
                high=p + 1.0,
                low=p - 1.0,
                close=p,
                volume=1000.0,
                bar_count=1,
            )
        )
    return bars


# ── Validation ─────────────────────────────────────────────────────────────────


class TestTerrainBuilderValidation:
    def test_build_raises_without_name(self) -> None:  # T-TBLDR-1
        """Build must fail when name is empty string."""
        with pytest.raises(ValueError, match="name is required"):
            TerrainBuilder().set_identity("", "SPY").set_regime(_make_regime()).build()

    def test_build_raises_without_symbol(self) -> None:  # T-TBLDR-2
        """Build must fail when symbol is empty string."""
        with pytest.raises(ValueError, match="symbol is required"):
            TerrainBuilder().set_identity("t", "").set_regime(_make_regime()).build()

    def test_build_raises_without_regime(self) -> None:  # T-TBLDR-3
        """Build must fail when no regime has been set."""
        with pytest.raises(ValueError, match="regime is required"):
            TerrainBuilder().set_identity("t", "SPY").build()


# ── Fluent API ─────────────────────────────────────────────────────────────────


class TestTerrainBuilderFluentApi:
    def test_set_identity_returns_self(self) -> None:  # T-TBLDR-4
        """Each fluent setter must return the same builder instance."""
        b = TerrainBuilder()
        assert b.set_identity("n", "S") is b

    def test_set_bars_returns_self(self) -> None:  # T-TBLDR-5
        """set_bars must return self for chaining."""
        b = TerrainBuilder()
        assert b.set_bars([]) is b

    def test_compute_features_returns_self(self) -> None:  # T-TBLDR-6
        """compute_features must return self even when bars are empty."""
        b = TerrainBuilder()
        assert b.compute_features() is b


# ── Empty bars ─────────────────────────────────────────────────────────────────


class TestTerrainBuilderEmptyBars:
    def test_compute_features_noop_on_empty_bars(self) -> None:  # T-TBLDR-7
        """compute_features with no bars must leave volatility_surface empty."""
        b = TerrainBuilder()
        b.compute_features()
        assert b._volatility_surface == []

    def test_build_succeeds_with_no_bars(self) -> None:  # T-TBLDR-8
        """build() must succeed even with zero bars; start/end use placeholder epoch."""
        terrain = TerrainBuilder().set_identity("empty", "SPY").set_regime(_make_regime()).build()
        assert terrain.name == "empty"
        assert len(terrain.bars) == 0
        assert terrain.start == datetime(2000, 1, 1, tzinfo=UTC)

    def test_compute_support_resistance_noop_on_short_bars(self) -> None:  # T-TBLDR-9
        """compute_support_resistance with fewer than lookback*2 bars must produce no levels."""
        b = TerrainBuilder().set_identity("t", "SPY")
        b.set_bars(_make_flat_bars(5))
        b.compute_support_resistance(lookback=20)
        assert b._levels == []


# ── Feature computation ────────────────────────────────────────────────────────


class TestTerrainBuilderComputeFeatures:
    def test_volatility_surface_length_matches_bars(self) -> None:  # T-TBLDR-10
        """volatility_surface must have one entry per bar after compute_features."""
        n = 300
        bars = _make_flat_bars(n)
        terrain = (
            TerrainBuilder()
            .set_identity("t", "SPY")
            .set_bars(bars)
            .set_regime(_make_regime())
            .compute_features()
            .build()
        )
        assert len(terrain.volatility_surface) == n

    def test_liquidity_profile_length_matches_bars(self) -> None:  # T-TBLDR-11
        """liquidity_profile must have one entry per bar after compute_features."""
        n = 200
        bars = _make_flat_bars(n)
        terrain = (
            TerrainBuilder()
            .set_identity("t", "SPY")
            .set_bars(bars)
            .set_regime(_make_regime())
            .compute_features()
            .build()
        )
        assert len(terrain.liquidity_profile) == n

    def test_volatility_surface_values_in_unit_interval(self) -> None:  # T-TBLDR-12
        """Every normalised ATR value must sit in [0, 1]."""
        bars = _make_oscillating_bars(200)
        terrain = (
            TerrainBuilder()
            .set_identity("t", "SPY")
            .set_bars(bars)
            .set_regime(_make_regime())
            .compute_features()
            .build()
        )
        for v in terrain.volatility_surface:
            assert 0.0 <= v <= 1.0, f"Out-of-range value: {v}"

    def test_zero_volume_bars_produce_uniform_liquidity_profile(self) -> None:  # T-TBLDR-13
        """All-zero volume bars must produce a liquidity profile of all 1.0 (fallback)."""
        bars = _make_flat_bars(100, volume=0.0)
        b = TerrainBuilder().set_identity("t", "SPY")
        b.set_bars(bars)
        b.compute_features()
        assert all(v == 1.0 for v in b._liquidity_profile)


# ── Support / resistance ───────────────────────────────────────────────────────


class TestTerrainBuilderSupportResistance:
    def test_levels_detected_on_oscillating_prices(self) -> None:  # T-TBLDR-14
        """Oscillating prices must yield at least one support and one resistance level."""
        bars = _make_oscillating_bars(300)
        terrain = (
            TerrainBuilder()
            .set_identity("t", "SPY")
            .set_bars(bars)
            .set_regime(_make_regime())
            .compute_support_resistance()
            .build()
        )
        types = {lvl.level_type for lvl in terrain.support_resistance}
        assert "support" in types
        assert "resistance" in types

    def test_level_prices_within_bar_range(self) -> None:  # T-TBLDR-15
        """Every detected level price must appear somewhere in the bar close series."""
        bars = _make_oscillating_bars(300)
        terrain = (
            TerrainBuilder()
            .set_identity("t", "SPY")
            .set_bars(bars)
            .set_regime(_make_regime())
            .compute_support_resistance()
            .build()
        )
        closes = {b.close for b in bars}
        for lvl in terrain.support_resistance:
            assert lvl.price in closes, f"Level price {lvl.price} not in close series"


# ── Catalog root ───────────────────────────────────────────────────────────────


class TestTerrainBuilderCatalogRoot:
    def test_catalog_path_none_without_root(self) -> None:  # T-TBLDR-16
        """catalog_path must be None when no catalog_root is set."""
        terrain = TerrainBuilder().set_identity("t", "SPY").set_regime(_make_regime()).build()
        assert terrain.catalog_path is None

    def test_catalog_path_none_when_no_bars(self) -> None:  # T-TBLDR-17
        """catalog_path must remain None when catalog_root is set but bars are empty."""
        from pathlib import Path

        terrain = (
            TerrainBuilder()
            .set_identity("t", "SPY")
            .set_regime(_make_regime())
            .set_catalog_root(Path("/tmp"))
            .build()
        )
        assert terrain.catalog_path is None
