"""T-UNI-*: union builders for the live contagion universe + structural seed."""

from __future__ import annotations

from pathlib import Path

import yaml

from qts.propagation.crypto.structural_links import (
    build_live_structural_links,
    load_structural_links,
)
from qts.propagation.crypto.universe import build_live_universe, load_crypto_universe


def test_universe_union_merges_all_event_configs(tmp_path: Path) -> None:  # T-UNI-1
    (tmp_path / "crypto_contagion_ftx.yaml").write_text(
        yaml.safe_dump({"tokens": {"FTT": {"cluster": "exchange"}, "BTC": {"cluster": "major"}}})
    )
    (tmp_path / "crypto_contagion_terra.yaml").write_text(
        yaml.safe_dump({"tokens": {"LUNA": {"cluster": "l1"}, "BTC": {"cluster": "major"}}})
    )
    out = tmp_path / "crypto_contagion_live.yaml"
    merged = build_live_universe(tmp_path, out)
    assert set(merged) == {"FTT", "BTC", "LUNA"}
    uni = load_crypto_universe(out)
    assert set(uni.tokens) == {"FTT", "BTC", "LUNA"}


def test_structural_union_dedupes_keeping_higher_confidence(tmp_path: Path) -> None:  # T-UNI-2
    (tmp_path / "crypto_structural_ftx.yaml").write_text(
        yaml.safe_dump(
            {
                "links": [
                    {
                        "source": "FTT",
                        "peer": "SOL",
                        "relation": "entity_exposure",
                        "confidence": 0.8,
                    }
                ]
            }
        )
    )
    (tmp_path / "crypto_structural_terra.yaml").write_text(
        yaml.safe_dump(
            {
                "links": [
                    {
                        "source": "FTT",
                        "peer": "SOL",
                        "relation": "entity_exposure",
                        "confidence": 0.95,
                    },
                    {
                        "source": "LUNA",
                        "peer": "UST",
                        "relation": "collateral_of",
                        "confidence": 0.99,
                    },
                ]
            }
        )
    )
    out = tmp_path / "crypto_structural_live.yaml"
    links = build_live_structural_links(tmp_path, out)
    by_pair = {(lnk["source"], lnk["peer"]): lnk for lnk in links}
    assert by_pair[("FTT", "SOL")]["confidence"] == 0.95  # higher kept
    assert ("LUNA", "UST") in by_pair
    assert len(load_structural_links(out)) == 2
