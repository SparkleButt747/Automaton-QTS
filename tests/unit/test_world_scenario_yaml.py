"""Tests for loading scenario YAML."""

from __future__ import annotations

from pathlib import Path


def test_shipped_fomc_scenario_loads() -> None:  # T-WSY-1
    from qts.world.scenario import load_scenario_yaml

    cfg = load_scenario_yaml(Path("config/scenarios/fomc_btcusdt_v1.yaml"))
    assert cfg.name == "fomc_btcusdt_v1"
    assert cfg.symbol == "BTCUSDT"
    assert len(cfg.anon_agents) == 3
    styles = {a.style for a in cfg.anon_agents}
    assert styles == {"sentiment", "trend", "mean_revert"}


def test_yaml_loader_rejects_missing_fields(tmp_path: Path) -> None:  # T-WSY-2
    import pytest

    from qts.world.scenario import load_scenario_yaml

    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\n")
    with pytest.raises(KeyError):
        load_scenario_yaml(bad)
