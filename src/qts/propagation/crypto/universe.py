"""Crypto universe: tokens, clusters, and per-token metadata for the contagion graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class CryptoUniverse:
    tokens: tuple[str, ...]  # sorted; position = node index
    clusters: tuple[str, ...]  # cluster per token, aligned to ``tokens``
    is_stablecoin: tuple[bool, ...]  # aligned to ``tokens``
    is_exchange_token: tuple[bool, ...]  # aligned to ``tokens``
    _aliases: tuple[tuple[str, ...], ...]  # alias list per token, aligned

    def index_of(self, token: str) -> int:
        if token not in self.tokens:
            raise KeyError(token)
        return self.tokens.index(token)

    def cluster_of(self, token: str) -> str:
        return self.clusters[self.index_of(token)]

    def alias_map(self) -> dict[str, str]:
        """Lowercased name aliases -> token (bare symbols excluded; the extractor matches those
        case-sensitively, mirroring the equity fix for homograph collisions)."""
        out: dict[str, str] = {}
        for tok, aliases in zip(self.tokens, self._aliases, strict=True):
            for a in aliases:
                out[a.lower()] = tok
        return out

    @property
    def unique_clusters(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.clusters)))


def load_crypto_universe(yaml_path: Path) -> CryptoUniverse:
    raw = yaml.safe_load(Path(yaml_path).read_text())
    items = raw["tokens"]
    tokens = tuple(sorted(items))
    clusters = tuple(items[t]["cluster"] for t in tokens)
    stable = tuple(bool(items[t].get("is_stablecoin", False)) for t in tokens)
    exch = tuple(bool(items[t].get("is_exchange_token", False)) for t in tokens)
    aliases = tuple(tuple(items[t].get("aliases", [])) for t in tokens)
    return CryptoUniverse(
        tokens=tokens,
        clusters=clusters,
        is_stablecoin=stable,
        is_exchange_token=exch,
        _aliases=aliases,
    )


def build_live_universe(src_dir: Path, out_path: Path) -> dict[str, dict]:
    """Merge every ``crypto_contagion_*.yaml`` token map in ``src_dir`` into one
    universe written to ``out_path``. First definition of a token wins."""
    merged: dict[str, dict] = {}
    for path in sorted(src_dir.glob("crypto_contagion_*.yaml")):
        if path.name == out_path.name:
            continue
        raw = yaml.safe_load(path.read_text()) or {}
        for token, meta in raw.get("tokens", {}).items():
            merged.setdefault(token, meta)
    out_path.write_text(yaml.safe_dump({"tokens": merged}, sort_keys=True))
    return merged
