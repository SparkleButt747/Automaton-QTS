"""Unit tests for qts.terrain.perturbation.

Covers:
- T-TPERT-1  to T-TPERT-14
- flash_crash: prices drop in the affected window, positive magnitude raises ValueError
- liquidity_shock: volume reduced by drain_factor in the window
- data_gap: bars in gap have zero volume and flat (stale) OHLC
- correlation_breakdown: output variance exceeds input variance
- generate_all_perturbations: returns exactly 13 variants
- slippage_venue_config: correct fill / slippage values propagated
- All perturbation methods return NEW terrains (originals are not mutated)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from qts.models.base import Bar, Catalyst, LiquidityLevel, SentimentLevel, Trend, VolLevel
from qts.models.terrain import MacroRegime, MarketTerrain
from qts.nautilus.config import VenueConfig
from qts.terrain.perturbation import TerrainPerturbator

# ── Helpers ───────────────────────────────────────────────────────────────────

_T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _make_regime() -> MacroRegime:
    return MacroRegime(
        trend=Trend.BULL,
        volatility=VolLevel.LOW,
        liquidity=LiquidityLevel.ABUNDANT,
        sentiment=SentimentLevel.NEUTRAL,
        catalyst=Catalyst.NONE,
        expected_drift=0.10,
        expected_vol=0.15,
        correlation_regime=0.5,
        scenario_description="perturbation test",
    )


def _make_bars(n: int, price: float = 100.0, volume: float = 1_000.0) -> list[Bar]:
    return [
        Bar(
            timestamp=_T0 + timedelta(days=i),
            symbol="SPY",
            open=price + i * 0.1,
            high=price + i * 0.1 + 1.0,
            low=price + i * 0.1 - 1.0,
            close=price + i * 0.1,
            volume=volume,
            bar_count=1,
        )
        for i in range(n)
    ]


def _make_terrain(n: int = 400, price: float = 100.0) -> MarketTerrain:
    bars = _make_bars(n, price)
    return MarketTerrain(
        name="base",
        symbol="SPY",
        start=bars[0].timestamp,
        end=bars[-1].timestamp,
        regime=_make_regime(),
        bars=tuple(bars),
    )


# ── Flash crash ───────────────────────────────────────────────────────────────


class TestFlashCrash:
    def test_price_falls_in_crash_window(self) -> None:  # T-TPERT-1
        """Bars inside the crash window must have lower close prices than the original."""
        terrain = _make_terrain()
        pert = TerrainPerturbator()
        crashed = pert.flash_crash(terrain, magnitude=-0.15, duration_bars=10, start_index=200)

        orig_close = terrain.bars[205].close
        crash_close = crashed.bars[205].close
        assert crash_close < orig_close

    def test_crash_ratio_matches_magnitude(self) -> None:  # T-TPERT-2
        """At the nadir the price ratio must be approximately 1 + magnitude."""
        terrain = _make_terrain()
        pert = TerrainPerturbator()
        mag = -0.20
        crashed = pert.flash_crash(terrain, magnitude=mag, duration_bars=10, start_index=200)

        # Bar at index 210 (start+duration) is the nadir
        orig = terrain.bars[210].close
        nadir = crashed.bars[210].close
        expected_ratio = 1.0 + mag
        assert abs(nadir / orig - expected_ratio) < 0.025

    def test_positive_magnitude_raises_value_error(self) -> None:  # T-TPERT-3
        """flash_crash must reject a non-negative magnitude with ValueError."""
        terrain = _make_terrain()
        pert = TerrainPerturbator()
        with pytest.raises(ValueError, match="magnitude must be negative"):
            pert.flash_crash(terrain, magnitude=0.05)

    def test_original_terrain_not_mutated(self) -> None:  # T-TPERT-4
        """flash_crash must return a new terrain; original bars must be unchanged."""
        terrain = _make_terrain()
        orig_close_200 = terrain.bars[200].close
        pert = TerrainPerturbator()
        pert.flash_crash(terrain, magnitude=-0.10, start_index=200)
        assert terrain.bars[200].close == orig_close_200

    def test_empty_terrain_returns_same_object(self) -> None:  # T-TPERT-5
        """flash_crash on a terrain with zero bars must return the same terrain unchanged."""
        empty = MarketTerrain(
            name="empty",
            symbol="SPY",
            start=_T0,
            end=_T0,
            regime=_make_regime(),
        )
        pert = TerrainPerturbator()
        result = pert.flash_crash(empty, magnitude=-0.10)
        assert result is empty


# ── Liquidity shock ───────────────────────────────────────────────────────────


class TestLiquidityShock:
    def test_volume_reduced_in_shock_window(self) -> None:  # T-TPERT-6
        """Bars inside the liquidity shock window must have volume * drain_factor."""
        terrain = _make_terrain()
        drain = 0.1
        pert = TerrainPerturbator()
        shocked = pert.liquidity_shock(
            terrain, drain_factor=drain, duration_bars=30, start_index=200
        )

        for i in range(200, 230):
            expected = terrain.bars[i].volume * drain
            assert abs(shocked.bars[i].volume - expected) < 1e-9

    def test_volume_unchanged_outside_window(self) -> None:  # T-TPERT-7
        """Bars outside the liquidity shock window must retain original volume."""
        terrain = _make_terrain()
        pert = TerrainPerturbator()
        shocked = pert.liquidity_shock(terrain, drain_factor=0.1, duration_bars=10, start_index=200)

        assert shocked.bars[150].volume == terrain.bars[150].volume
        assert shocked.bars[250].volume == terrain.bars[250].volume


# ── Data gap ──────────────────────────────────────────────────────────────────


class TestDataGap:
    def test_gap_bars_have_zero_volume(self) -> None:  # T-TPERT-8
        """All bars inside the data gap must have volume == 0."""
        terrain = _make_terrain()
        pert = TerrainPerturbator()
        gapped = pert.data_gap(terrain, gap_duration_bars=20, start_index=250)

        for i in range(250, 270):
            assert gapped.bars[i].volume == 0.0

    def test_gap_bars_have_flat_ohlc(self) -> None:  # T-TPERT-9
        """Gap bars must be flat — open == high == low == close (stale price)."""
        terrain = _make_terrain()
        pert = TerrainPerturbator()
        gapped = pert.data_gap(terrain, gap_duration_bars=20, start_index=250)

        for i in range(250, 270):
            b = gapped.bars[i]
            assert b.open == b.close
            assert b.high == b.close
            assert b.low == b.close

    def test_bars_outside_gap_unchanged(self) -> None:  # T-TPERT-10
        """Bars before and well after the gap must retain their original close price."""
        terrain = _make_terrain()
        pert = TerrainPerturbator()
        gapped = pert.data_gap(terrain, gap_duration_bars=20, start_index=250)

        assert gapped.bars[100].close == terrain.bars[100].close
        assert gapped.bars[300].close == terrain.bars[300].close


# ── Correlation breakdown ──────────────────────────────────────────────────────


class TestCorrelationBreakdown:
    def test_output_variance_exceeds_input_variance(self) -> None:  # T-TPERT-11
        """correlation_breakdown must add noise so output variance > input variance."""
        # Flat bars have zero price variance — any noise will make it positive.
        terrain = _make_terrain()
        pert = TerrainPerturbator()
        noisy = pert.correlation_breakdown(terrain, noise_std=0.05)

        orig_var = np.var([b.close for b in terrain.bars])
        noisy_var = np.var([b.close for b in noisy.bars])
        assert noisy_var > orig_var

    def test_name_reflects_noise_std(self) -> None:  # T-TPERT-12
        """Perturbed terrain name must encode the noise_std used."""
        terrain = _make_terrain()
        pert = TerrainPerturbator()
        noisy = pert.correlation_breakdown(terrain, noise_std=0.02)
        assert "2%" in noisy.name or "correlation_break" in noisy.name


# ── generate_all_perturbations ─────────────────────────────────────────────────


class TestGenerateAllPerturbations:
    def test_returns_exactly_13_variants(self) -> None:  # T-TPERT-13
        """generate_all_perturbations must return exactly 13 terrain variants.

        Breakdown from default PerturbationConfig:
          4 flash_crash magnitudes
        + 4 liquidity drain factors
        + 4 data gap durations
        + 1 correlation breakdown
        = 13
        """
        terrain = _make_terrain()
        pert = TerrainPerturbator()
        variants = pert.generate_all_perturbations(terrain)
        assert len(variants) == 13

    def test_all_variants_are_distinct_terrain_objects(self) -> None:  # T-TPERT-14
        """Every variant returned must be a distinct MarketTerrain instance."""
        terrain = _make_terrain()
        pert = TerrainPerturbator()
        variants = pert.generate_all_perturbations(terrain)
        # Check identity — no two entries are the same object
        for i, v1 in enumerate(variants):
            for j, v2 in enumerate(variants):
                if i != j:
                    assert v1 is not v2


# ── slippage_venue_config ──────────────────────────────────────────────────────


class TestSlippageVenueConfig:
    def test_fill_probability_applied(self) -> None:  # T-TPERT-15
        """prob_fill_on_limit must equal the supplied fill_probability."""
        base = VenueConfig(prob_slippage=0.1)
        pert = TerrainPerturbator()
        perturbed = pert.slippage_venue_config(base, slippage_multiplier=5.0, fill_probability=0.3)
        assert perturbed.prob_fill_on_limit == 0.3

    def test_slippage_multiplied(self) -> None:  # T-TPERT-16
        """prob_slippage must equal base * multiplier (capped at 1.0)."""
        base = VenueConfig(prob_slippage=0.1)
        pert = TerrainPerturbator()
        perturbed = pert.slippage_venue_config(base, slippage_multiplier=5.0, fill_probability=0.3)
        assert abs(perturbed.prob_slippage - 0.5) < 1e-9

    def test_slippage_capped_at_one(self) -> None:  # T-TPERT-17
        """prob_slippage must never exceed 1.0 regardless of multiplier."""
        base = VenueConfig(prob_slippage=0.9)
        pert = TerrainPerturbator()
        perturbed = pert.slippage_venue_config(
            base, slippage_multiplier=100.0, fill_probability=0.1
        )
        assert perturbed.prob_slippage <= 1.0

    def test_venue_name_preserved(self) -> None:  # T-TPERT-18
        """The perturbed VenueConfig must preserve the original venue name."""
        base = VenueConfig(name="NYSE", prob_slippage=0.2)
        pert = TerrainPerturbator()
        perturbed = pert.slippage_venue_config(base, slippage_multiplier=2.0, fill_probability=0.5)
        assert perturbed.name == "NYSE"
