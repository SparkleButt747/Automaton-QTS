"""Unit tests for qts.terrain.library.

Covers:
- T-TLIB-1  to T-TLIB-14
- from_config_dir: loads 25 terrains from real config/terrains/
- Train/test split: 20 train, 5 test (files prefixed test_)
- get_by_name: returns correct terrain or None for missing
- get_by_regime: filters by trend, volatility, liquidity, or combinations
- add() / manual split management
- Empty library (non-existent directory) returns zero terrains
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from qts.models.base import Catalyst, LiquidityLevel, SentimentLevel, Trend, VolLevel
from qts.models.terrain import MacroRegime, MarketTerrain
from qts.terrain.library import TerrainLibrary

# ── Project root (two levels up from this file: tests/unit -> tests -> project) ──
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_TERRAIN_CONFIG_DIR = _PROJECT_ROOT / "config" / "terrains"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_terrain(
    name: str,
    trend: Trend = Trend.BULL,
    vol: VolLevel = VolLevel.LOW,
    liq: LiquidityLevel = LiquidityLevel.ABUNDANT,
) -> MarketTerrain:
    regime = MacroRegime(
        trend=trend,
        volatility=vol,
        liquidity=liq,
        sentiment=SentimentLevel.NEUTRAL,
        catalyst=Catalyst.NONE,
        expected_drift=0.10,
        expected_vol=0.15,
        correlation_regime=0.5,
        scenario_description="unit test terrain",
    )
    return MarketTerrain(
        name=name,
        symbol="SPY",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 6, 1, tzinfo=UTC),
        regime=regime,
    )


# ── from_config_dir (real files) ───────────────────────────────────────────────


class TestTerrainLibraryFromConfigDir:
    def test_loads_25_terrains(self) -> None:  # T-TLIB-1
        """from_config_dir must load exactly 25 terrain definitions from config/terrains/."""
        lib = TerrainLibrary.from_config_dir(_TERRAIN_CONFIG_DIR)
        assert len(lib.all_terrains) == 25

    def test_train_split_is_20(self) -> None:  # T-TLIB-2
        """Non-test-prefixed YAML files must form the 20-terrain training set."""
        lib = TerrainLibrary.from_config_dir(_TERRAIN_CONFIG_DIR)
        assert len(lib.get_train_terrains()) == 20

    def test_test_split_is_5(self) -> None:  # T-TLIB-3
        """YAML files prefixed 'test_' must form the 5-terrain held-out test set."""
        lib = TerrainLibrary.from_config_dir(_TERRAIN_CONFIG_DIR)
        assert len(lib.get_test_terrains()) == 5

    def test_test_terrains_all_have_test_prefix(self) -> None:  # T-TLIB-4
        """Every terrain in the test split must have a name beginning with 'test_'."""
        lib = TerrainLibrary.from_config_dir(_TERRAIN_CONFIG_DIR)
        for t in lib.get_test_terrains():
            assert t.name.startswith("test_"), f"Unexpected name in test split: {t.name}"

    def test_known_terrain_loaded_correctly(self) -> None:  # T-TLIB-5
        """The covid_crash terrain must be present with regime BEAR/HIGH/CRISIS."""
        lib = TerrainLibrary.from_config_dir(_TERRAIN_CONFIG_DIR)
        t = lib.get_by_name("covid_crash")
        assert t is not None
        assert t.symbol == "SPY"
        assert t.regime.trend == Trend.BEAR
        assert t.regime.volatility == VolLevel.HIGH
        assert t.regime.liquidity == LiquidityLevel.CRISIS

    def test_events_loaded_for_terrain_with_events(self) -> None:  # T-TLIB-6
        """Terrains defined with events in YAML must have a non-empty event_calendar."""
        lib = TerrainLibrary.from_config_dir(_TERRAIN_CONFIG_DIR)
        t = lib.get_by_name("covid_crash")
        assert t is not None
        assert len(t.event_calendar) > 0

    def test_missing_directory_returns_empty_library(self) -> None:  # T-TLIB-7
        """from_config_dir with a non-existent directory must return zero terrains."""
        lib = TerrainLibrary.from_config_dir(Path("/tmp/no_such_terrain_dir_xyz"))
        assert len(lib.all_terrains) == 0


# ── get_by_name ───────────────────────────────────────────────────────────────


class TestTerrainLibraryGetByName:
    def test_get_by_name_returns_correct_terrain(self) -> None:  # T-TLIB-8
        """get_by_name must return the terrain with the matching name."""
        lib = TerrainLibrary([_make_terrain("alpha"), _make_terrain("beta")])
        t = lib.get_by_name("alpha")
        assert t is not None
        assert t.name == "alpha"

    def test_get_by_name_returns_none_for_missing(self) -> None:  # T-TLIB-9
        """get_by_name must return None when no terrain matches the requested name."""
        lib = TerrainLibrary([_make_terrain("alpha")])
        assert lib.get_by_name("does_not_exist") is None


# ── get_by_regime ─────────────────────────────────────────────────────────────


class TestTerrainLibraryGetByRegime:
    def _populated_lib(self) -> TerrainLibrary:
        lib = TerrainLibrary()
        lib.add(_make_terrain("a", trend=Trend.BULL, vol=VolLevel.LOW), split="train")
        lib.add(_make_terrain("b", trend=Trend.BEAR, vol=VolLevel.HIGH), split="train")
        lib.add(
            _make_terrain("c", trend=Trend.BEAR, vol=VolLevel.HIGH, liq=LiquidityLevel.CRISIS),
            split="test",
        )
        lib.add(_make_terrain("d", trend=Trend.SIDEWAYS, vol=VolLevel.TRANSITIONING), split="train")
        return lib

    def test_filter_by_trend_only(self) -> None:  # T-TLIB-10
        """get_by_regime(trend=BEAR) must return only BEAR terrains."""
        lib = self._populated_lib()
        bears = lib.get_by_regime(trend=Trend.BEAR)
        assert {t.name for t in bears} == {"b", "c"}

    def test_filter_by_volatility_only(self) -> None:  # T-TLIB-11
        """get_by_regime(volatility=HIGH) must return only HIGH volatility terrains."""
        lib = self._populated_lib()
        high_vol = lib.get_by_regime(volatility=VolLevel.HIGH)
        assert {t.name for t in high_vol} == {"b", "c"}

    def test_filter_by_multiple_dimensions(self) -> None:  # T-TLIB-12
        """Combining trend and volatility filters must return only the intersection."""
        lib = self._populated_lib()
        result = lib.get_by_regime(
            trend=Trend.BEAR, volatility=VolLevel.HIGH, liquidity=LiquidityLevel.CRISIS
        )
        assert len(result) == 1
        assert result[0].name == "c"

    def test_filter_with_no_match_returns_empty(self) -> None:  # T-TLIB-13
        """get_by_regime with no matching terrains must return an empty list."""
        lib = self._populated_lib()
        result = lib.get_by_regime(trend=Trend.BULL, volatility=VolLevel.HIGH)
        assert result == []

    def test_no_filter_returns_all_terrains(self) -> None:  # T-TLIB-14
        """get_by_regime with no arguments must return every terrain."""
        lib = self._populated_lib()
        assert len(lib.get_by_regime()) == 4


# ── Train / test split via add() ──────────────────────────────────────────────


class TestTerrainLibraryAddSplit:
    def test_add_train_terrain(self) -> None:  # T-TLIB-15
        """Terrains added with split='train' must appear in get_train_terrains()."""
        lib = TerrainLibrary()
        lib.add(_make_terrain("train_a"), split="train")
        lib.add(_make_terrain("train_b"), split="train")
        lib.add(_make_terrain("test_c"), split="test")
        assert len(lib.get_train_terrains()) == 2
        assert len(lib.get_test_terrains()) == 1

    def test_default_split_is_70_percent(self) -> None:  # T-TLIB-16
        """A library with no explicit splits uses the 70% rule for get_train_terrains."""
        lib = TerrainLibrary([_make_terrain(str(i)) for i in range(10)])
        # 70% of 10 = 7
        assert len(lib.get_train_terrains()) == 7
        assert len(lib.get_test_terrains()) == 3
