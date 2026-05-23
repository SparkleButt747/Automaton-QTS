"""Equity universe: tickers, sectors, and the alias map that feeds ticker extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EquityUniverse:
    tickers: tuple[str, ...]  # sorted; position = node index
    sectors: tuple[str, ...]  # sector per ticker, aligned to ``tickers``
    _aliases: tuple[tuple[str, ...], ...]  # alias list per ticker, aligned

    def index_of(self, ticker: str) -> int:
        if ticker not in self.tickers:
            raise KeyError(ticker)
        return self.tickers.index(ticker)

    def sector_of(self, ticker: str) -> str:
        if ticker not in self.tickers:
            raise KeyError(ticker)
        return self.sectors[self.tickers.index(ticker)]

    def alias_map(self) -> dict[str, str]:
        """Lowercased alias/ticker -> ticker, for qts.propagation.equity.extract_tickers."""
        out: dict[str, str] = {}
        for tkr, aliases in zip(self.tickers, self._aliases, strict=True):
            out[tkr.lower()] = tkr
            for a in aliases:
                out[a.lower()] = tkr
        return out

    @property
    def unique_sectors(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.sectors)))


def load_universe(yaml_path: Path) -> EquityUniverse:
    raw = yaml.safe_load(Path(yaml_path).read_text())
    items = raw["tickers"]
    tickers = tuple(sorted(items))
    sectors = tuple(items[t]["sector"] for t in tickers)
    aliases = tuple(tuple(items[t].get("aliases", [])) for t in tickers)
    return EquityUniverse(tickers=tickers, sectors=sectors, _aliases=aliases)
