"""Perturbation testing — inject stress scenarios into MarketTerrain objects.

Analogous to racing lab's P4a-P4e perturbation phases:
- P4a: State noise → price noise injection
- P4b: Actuator delay → execution delay (Nautilus fill model)
- P4c: Model mismatch → regime mismatch (train on one regime, test on another)
- P4d: Unseen tracks → unseen market periods
- P4e: Drivetrain transfer → asset class transfer

Additional market-specific perturbations:
- Flash crash injection
- Liquidity shock
- Sentiment spike
- Correlation breakdown
- Data gap (exchange halt)
- Slippage spike
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from qts.models.base import Bar
from qts.models.terrain import MarketTerrain
from qts.nautilus.config import VenueConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerturbationConfig:
    """Configuration for perturbation testing.

    Analogous to racing lab's PerturbationConfig in FairnessConfig.
    """

    # Flash crash
    flash_crash_magnitudes: tuple[float, ...] = (-0.05, -0.10, -0.15, -0.20)
    flash_crash_duration_bars: int = 5

    # Liquidity shock
    liquidity_drain_factors: tuple[float, ...] = (0.5, 0.3, 0.1, 0.0)
    liquidity_drain_duration_bars: int = 10

    # Sentiment spike
    sentiment_magnitudes: tuple[float, ...] = (-0.9, -0.5, 0.5, 0.9)
    sentiment_duration_bars: int = 5

    # Slippage spike (via Nautilus fill model)
    slippage_multipliers: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)
    fill_probability_overrides: tuple[float, ...] = (0.8, 0.5, 0.3, 0.1)

    # Data gap
    data_gap_durations: tuple[int, ...] = (5, 15, 30, 60)


class TerrainPerturbator:
    """Generates perturbed variants of MarketTerrain objects.

    All methods return NEW terrains; originals are not mutated.
    """

    def __init__(self, config: PerturbationConfig | None = None) -> None:
        self._config = config or PerturbationConfig()

    # ------------------------------------------------------------------
    # Flash crash (P4a analogue — state noise)
    # ------------------------------------------------------------------

    def flash_crash(
        self,
        terrain: MarketTerrain,
        magnitude: float = -0.15,
        duration_bars: int = 5,
        start_index: int | None = None,
    ) -> MarketTerrain:
        """Inject a flash crash into a terrain.

        Prices fall linearly by magnitude, then recover.
        """
        if magnitude >= 0:
            raise ValueError(f"magnitude must be negative, got {magnitude}")

        bars = list(terrain.bars)
        n = len(bars)
        if n == 0:
            return terrain

        start = start_index if start_index is not None else n // 3
        end_crash = min(start + duration_bars, n)
        end_recovery = min(end_crash + duration_bars, n)

        # Decline phase
        for i in range(start, end_crash):
            frac = (i - start + 1) / duration_bars
            mult = 1.0 + magnitude * frac
            bars[i] = _scale_bar(bars[i], mult)

        # Recovery phase
        nadir = 1.0 + magnitude
        for i in range(end_crash, end_recovery):
            frac = (i - end_crash + 1) / duration_bars
            mult = nadir + (-magnitude) * frac
            bars[i] = _scale_bar(bars[i], mult)

        return _rebuild_terrain(terrain, bars, f"{terrain.name}_flash_crash_{abs(magnitude):.0%}")

    # ------------------------------------------------------------------
    # Liquidity shock
    # ------------------------------------------------------------------

    def liquidity_shock(
        self,
        terrain: MarketTerrain,
        drain_factor: float = 0.1,
        duration_bars: int = 10,
        start_index: int | None = None,
    ) -> MarketTerrain:
        """Drain volume from a terrain to simulate a liquidity crisis."""
        bars = list(terrain.bars)
        n = len(bars)
        if n == 0:
            return terrain

        start = start_index if start_index is not None else n // 3
        end = min(start + duration_bars, n)

        for i in range(start, end):
            b = bars[i]
            bars[i] = Bar(
                timestamp=b.timestamp,
                symbol=b.symbol,
                open=b.open,
                high=b.high,
                low=b.low,
                close=b.close,
                volume=b.volume * drain_factor,
                bar_count=b.bar_count,
            )

        return _rebuild_terrain(terrain, bars, f"{terrain.name}_liquidity_shock_{drain_factor:.0%}")

    # ------------------------------------------------------------------
    # Data gap (exchange halt)
    # ------------------------------------------------------------------

    def data_gap(
        self,
        terrain: MarketTerrain,
        gap_duration_bars: int = 30,
        start_index: int | None = None,
    ) -> MarketTerrain:
        """Simulate a data gap by replacing bars with stale flat data."""
        bars = list(terrain.bars)
        n = len(bars)
        if n == 0:
            return terrain

        start = start_index if start_index is not None else n // 3
        stale = bars[max(0, start - 1)]
        end = min(start + gap_duration_bars, n)

        for i in range(start, end):
            bars[i] = Bar(
                timestamp=bars[i].timestamp,
                symbol=bars[i].symbol,
                open=stale.close,
                high=stale.close,
                low=stale.close,
                close=stale.close,
                volume=0.0,
                bar_count=0,
            )

        return _rebuild_terrain(terrain, bars, f"{terrain.name}_data_gap_{gap_duration_bars}bars")

    # ------------------------------------------------------------------
    # Slippage spike (P4b analogue — execution perturbation via fill model)
    # ------------------------------------------------------------------

    def slippage_venue_config(
        self,
        base_config: VenueConfig,
        slippage_multiplier: float = 5.0,
        fill_probability: float = 0.3,
    ) -> VenueConfig:
        """Create a perturbed venue config with increased slippage.

        This perturbs the 'physics engine' itself — the Nautilus fill model
        simulates worse execution conditions.
        """
        return VenueConfig(
            name=base_config.name,
            oms_type=base_config.oms_type,
            account_type=base_config.account_type,
            base_currency=base_config.base_currency,
            starting_balance=base_config.starting_balance,
            prob_fill_on_limit=fill_probability,
            prob_fill_on_stop=min(1.0, fill_probability + 0.1),
            prob_slippage=min(1.0, base_config.prob_slippage * slippage_multiplier),
            fill_model_seed=base_config.fill_model_seed,
        )

    # ------------------------------------------------------------------
    # Correlation breakdown
    # ------------------------------------------------------------------

    def correlation_breakdown(
        self,
        terrain: MarketTerrain,
        noise_std: float = 0.02,
    ) -> MarketTerrain:
        """Add uncorrelated noise to prices, simulating correlation breakdown."""
        rng = np.random.default_rng(42)
        bars = list(terrain.bars)

        for i in range(len(bars)):
            b = bars[i]
            noise = 1.0 + rng.normal(0, noise_std)
            bars[i] = _scale_bar(b, noise)

        return _rebuild_terrain(terrain, bars, f"{terrain.name}_correlation_break_{noise_std:.0%}")

    # ------------------------------------------------------------------
    # Batch perturbation generation
    # ------------------------------------------------------------------

    def generate_all_perturbations(self, terrain: MarketTerrain) -> list[MarketTerrain]:
        """Generate all configured perturbation variants of a terrain."""
        results: list[MarketTerrain] = []

        for mag in self._config.flash_crash_magnitudes:
            results.append(
                self.flash_crash(
                    terrain,
                    magnitude=mag,
                    duration_bars=self._config.flash_crash_duration_bars,
                )
            )

        for factor in self._config.liquidity_drain_factors:
            results.append(
                self.liquidity_shock(
                    terrain,
                    drain_factor=factor,
                    duration_bars=self._config.liquidity_drain_duration_bars,
                )
            )

        for duration in self._config.data_gap_durations:
            results.append(self.data_gap(terrain, gap_duration_bars=duration))

        results.append(self.correlation_breakdown(terrain))

        logger.info(
            "Generated %d perturbation variants of terrain '%s'",
            len(results),
            terrain.name,
        )
        return results


# ── Helpers ──────────────────────────────────────────────────────────────────


def _scale_bar(bar: Bar, multiplier: float) -> Bar:
    """Return a new Bar with OHLC prices scaled."""
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


def _rebuild_terrain(terrain: MarketTerrain, bars: list[Bar], name: str) -> MarketTerrain:
    """Rebuild a terrain with modified bars, preserving all other metadata."""
    return MarketTerrain(
        name=name,
        symbol=terrain.symbol,
        start=bars[0].timestamp if bars else terrain.start,
        end=bars[-1].timestamp if bars else terrain.end,
        regime=terrain.regime,
        bars=tuple(bars),
        regime_spans=terrain.regime_spans,
        volatility_surface=terrain.volatility_surface,
        support_resistance=terrain.support_resistance,
        liquidity_profile=terrain.liquidity_profile,
        event_calendar=terrain.event_calendar,
        catalog_path=None,  # perturbed terrains don't inherit catalog
    )
