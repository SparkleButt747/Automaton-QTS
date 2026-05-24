"""T-CRYPTO-LINK-5/6: curated structural-seed loader -> typed CryptoLinks."""

from __future__ import annotations

from pathlib import Path

from qts.propagation.crypto.structural_links import load_structural_links

from qts.propagation.crypto.links import CryptoLink


def _write_seed(tmp_path: Path) -> Path:
    p = tmp_path / "seed.yaml"
    p.write_text(
        "links:\n"
        "  - {source: FTT, peer: SOL, relation: entity_exposure, direction: negative, confidence: 0.95}\n"
        "  - {source: UST, peer: LUNA, relation: collateral_of, direction: negative}\n"
    )
    return p


def test_loads_typed_links_with_defaults(tmp_path: Path) -> None:  # T-CRYPTO-LINK-5
    links = load_structural_links(_write_seed(tmp_path))
    assert all(isinstance(x, CryptoLink) for x in links)
    ftt = next(x for x in links if x.source == "FTT")
    assert ftt.peer == "SOL" and ftt.relation == "entity_exposure" and ftt.confidence == 0.95
    ust = next(x for x in links if x.source == "UST")
    assert ust.direction == "negative" and ust.confidence == 0.9  # default confidence


def test_rejects_unknown_relation(tmp_path: Path) -> None:  # T-CRYPTO-LINK-6
    p = tmp_path / "bad.yaml"
    p.write_text("links:\n  - {source: A, peer: B, relation: frenemy}\n")
    import pytest

    with pytest.raises(ValueError, match="frenemy"):
        load_structural_links(p)
