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
