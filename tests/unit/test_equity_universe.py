"""T-PATHA-UNIV-*: equity universe + alias map loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from qts.propagation.equity.universe import EquityUniverse, load_universe

_YAML = """\
tickers:
  NVDA:
    sector: Semiconductors
    aliases: [nvidia]
  AMD:
    sector: Semiconductors
    aliases: [advanced micro devices]
  TSM:
    sector: Semiconductors
    aliases: [tsmc, taiwan semiconductor]
"""


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "u.yaml"
    p.write_text(_YAML)
    return p


def test_load_universe_tickers_and_sectors(tmp_path: Path) -> None:  # T-PATHA-UNIV-1
    u = load_universe(_write(tmp_path))
    assert isinstance(u, EquityUniverse)
    assert u.tickers == ("AMD", "NVDA", "TSM")
    assert u.sector_of("NVDA") == "Semiconductors"


def test_alias_map_lowercased_includes_ticker_and_aliases(tmp_path: Path) -> None:  # T-PATHA-UNIV-2
    u = load_universe(_write(tmp_path))
    am = u.alias_map()
    assert am["nvidia"] == "NVDA"
    assert am["nvda"] == "NVDA"
    assert am["taiwan semiconductor"] == "TSM"
    assert all(k == k.lower() for k in am)


def test_index_of_is_stable(tmp_path: Path) -> None:  # T-PATHA-UNIV-3
    u = load_universe(_write(tmp_path))
    assert u.index_of("AMD") == 0 and u.index_of("TSM") == 2


def test_unknown_ticker_raises(tmp_path: Path) -> None:  # T-PATHA-UNIV-4
    u = load_universe(_write(tmp_path))
    with pytest.raises(KeyError):
        u.sector_of("ZZZZ")
