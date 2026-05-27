"""Curated structural-link seed (verifiable on-chain/DeFiLlama relationships) -> typed CryptoLinks.

The gold-standard backbone the equity study lacked. Built from a CONTEMPORANEOUS integration
snapshot, NOT from event post-mortems (avoid hindsight bias — spec §10).
"""

from __future__ import annotations

from pathlib import Path

import yaml

from qts.propagation.crypto.links import CRYPTO_RELATIONS, CryptoLink


def load_structural_links(yaml_path: Path) -> list[CryptoLink]:
    raw = yaml.safe_load(Path(yaml_path).read_text())
    out: list[CryptoLink] = []
    for item in raw["links"]:
        relation = str(item["relation"])
        if relation not in CRYPTO_RELATIONS:
            raise ValueError(f"unknown structural relation: {relation}")
        out.append(
            CryptoLink(
                source=str(item["source"]),
                peer=str(item["peer"]),
                relation=relation,  # type: ignore[arg-type]
                direction=str(item.get("direction", "negative")),  # type: ignore[arg-type]
                confidence=float(item.get("confidence", 0.9)),
            )
        )
    return out


def build_live_structural_links(src_dir: Path, out_path: Path) -> list[dict]:
    """Merge every ``crypto_structural_*.yaml`` link list in ``src_dir`` into one
    seed written to ``out_path``. De-dupe on (source, peer, relation), keep the
    highest confidence."""
    merged: dict[tuple[str, str, str], dict] = {}
    for path in sorted(src_dir.glob("crypto_structural_*.yaml")):
        if path.name == out_path.name:
            continue
        raw = yaml.safe_load(path.read_text()) or {}
        for link in raw.get("links", []):
            key = (str(link["source"]), str(link["peer"]), str(link["relation"]))
            prev = merged.get(key)
            if prev is None or float(link.get("confidence", 0.9)) > float(
                prev.get("confidence", 0.9)
            ):
                merged[key] = link
    links = list(merged.values())
    out_path.write_text(yaml.safe_dump({"links": links}, sort_keys=True))
    return links
