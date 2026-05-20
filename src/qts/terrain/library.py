"""TerrainLibrary — loads, stores, and manages a catalog of annotated terrains.

Analogous to the racing lab's train/test track split. Provides:
- ``get_train_terrains()``: terrains for optimisation (diverse regimes)
- ``get_test_terrains()``: held-out terrains for final evaluation
- ``get_perturbation_terrains()``: stress-tested variants
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from qts.models.base import (
    Bar,
    Catalyst,
    LiquidityLevel,
    SentimentLevel,
    Trend,
    VolLevel,
)
from qts.models.terrain import MacroRegime, MarketEvent, MarketTerrain
from qts.terrain.builder import TerrainBuilder

logger = logging.getLogger(__name__)

# Default paths
_CONFIG_TERRAINS_DIR = Path("config/terrains")


class TerrainLibrary:
    """Manages a catalog of MarketTerrain objects for training and evaluation.

    Terrains are defined in YAML files in config/terrains/. Each file
    specifies regime metadata and a reference to bar data (CSV or Parquet).

    Usage::

        library = TerrainLibrary.from_config_dir()
        train = library.get_train_terrains()
        test = library.get_test_terrains()
    """

    def __init__(self, terrains: list[MarketTerrain] | None = None) -> None:
        self._terrains: list[MarketTerrain] = terrains or []
        self._train_names: set[str] = set()
        self._test_names: set[str] = set()

    @property
    def all_terrains(self) -> list[MarketTerrain]:
        """All loaded terrains."""
        return list(self._terrains)

    def add(self, terrain: MarketTerrain, split: str = "train") -> None:
        """Add a terrain to the library.

        Args:
            terrain: The terrain to add.
            split: One of "train" or "test".
        """
        self._terrains.append(terrain)
        if split == "train":
            self._train_names.add(terrain.name)
        else:
            self._test_names.add(terrain.name)

    def get_train_terrains(self) -> list[MarketTerrain]:
        """Return terrains designated for training/optimisation."""
        if self._train_names:
            return [t for t in self._terrains if t.name in self._train_names]
        # Default: first 70% are train
        split_idx = max(1, int(len(self._terrains) * 0.7))
        return self._terrains[:split_idx]

    def get_test_terrains(self) -> list[MarketTerrain]:
        """Return terrains designated for out-of-sample testing."""
        if self._test_names:
            return [t for t in self._terrains if t.name in self._test_names]
        split_idx = max(1, int(len(self._terrains) * 0.7))
        return self._terrains[split_idx:]

    def get_by_name(self, name: str) -> MarketTerrain | None:
        """Look up a terrain by name."""
        for t in self._terrains:
            if t.name == name:
                return t
        return None

    def get_by_regime(
        self,
        trend: Trend | None = None,
        volatility: VolLevel | None = None,
        liquidity: LiquidityLevel | None = None,
    ) -> list[MarketTerrain]:
        """Filter terrains by regime attributes."""
        results: list[MarketTerrain] = []
        for t in self._terrains:
            if trend is not None and t.regime.trend != trend:
                continue
            if volatility is not None and t.regime.volatility != volatility:
                continue
            if liquidity is not None and t.regime.liquidity != liquidity:
                continue
            results.append(t)
        return results

    # ------------------------------------------------------------------
    # Loading from config directory
    # ------------------------------------------------------------------

    @classmethod
    def from_config_dir(
        cls,
        config_dir: Path | None = None,
        bar_loader: Any | None = None,
        catalog_root: Path | None = None,
    ) -> TerrainLibrary:
        """Load all terrain definitions from a config directory.

        Each .yaml file in the directory defines one terrain. Bar data
        is loaded via the bar_loader callable or left empty for lazy loading.

        Args:
            config_dir: Directory containing YAML terrain definitions.
            bar_loader: Optional callable(symbol, start, end) -> list[Bar].
            catalog_root: Parquet catalog root for Nautilus.
        """
        directory = config_dir or _CONFIG_TERRAINS_DIR
        if not directory.exists():
            logger.warning("Terrain config directory not found: %s", directory)
            return cls()

        library = cls()
        yaml_files = sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))

        for yaml_path in yaml_files:
            try:
                terrain = _load_terrain_from_yaml(yaml_path, bar_loader, catalog_root)
                # Convention: files starting with "test_" go to test split,
                # or use explicit split field from YAML
                split = "test" if yaml_path.stem.startswith("test_") else "train"
                library.add(terrain, split=split)
                logger.info("Loaded terrain '%s' (%s split)", terrain.name, split)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to load terrain from %s", yaml_path)

        logger.info(
            "TerrainLibrary: loaded %d terrains (%d train, %d test)",
            len(library._terrains),
            len(library.get_train_terrains()),
            len(library.get_test_terrains()),
        )
        return library


def _load_terrain_from_yaml(
    path: Path,
    bar_loader: Any | None = None,
    catalog_root: Path | None = None,
) -> MarketTerrain:
    """Parse a YAML terrain definition and construct a MarketTerrain."""
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    name = raw["name"]
    symbol = raw["symbol"]
    start = raw["start"]
    end = raw["end"]
    regime_data = raw["regime"]

    # Build MacroRegime from YAML
    regime = MacroRegime(
        trend=Trend(regime_data["trend"]),
        volatility=VolLevel(regime_data["volatility"]),
        liquidity=LiquidityLevel(regime_data["liquidity"]),
        sentiment=SentimentLevel(regime_data["sentiment"]),
        catalyst=Catalyst(regime_data.get("catalyst", "NONE")),
        expected_drift=float(regime_data.get("expected_drift", 0.0)),
        expected_vol=float(regime_data.get("expected_vol", 0.2)),
        correlation_regime=float(regime_data.get("correlation_regime", 0.5)),
        scenario_description=regime_data.get("scenario_description", ""),
    )

    # Load events if present
    events: list[MarketEvent] = []
    for evt in raw.get("events", []):
        from datetime import datetime  # noqa: PLC0415

        events.append(
            MarketEvent(
                timestamp=datetime.fromisoformat(evt["timestamp"]),
                event_type=evt["event_type"],
                description=evt["description"],
                impact_magnitude=float(evt.get("impact_magnitude", 0.0)),
            )
        )

    # Load bars via bar_loader if available
    bars: list[Bar] = []
    if bar_loader is not None:
        bars = bar_loader(symbol, start, end)

    # Build terrain
    builder = TerrainBuilder()
    builder.set_identity(name, symbol).set_bars(bars).set_regime(regime)

    for event in events:
        builder.add_event(event)

    if catalog_root:
        builder.set_catalog_root(catalog_root)

    if bars:
        builder.compute_features()
        builder.compute_support_resistance()

    return builder.build()
