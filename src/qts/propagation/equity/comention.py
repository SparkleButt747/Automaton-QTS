"""FNSPID -> point-in-time co-mention edges (Path A v2 Phase 1, spec §5)."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# FNSPID CSV column names — verify against the downloaded header (see plan FNSPID schema note).
FNSPID_DATE_COL = "Date"
FNSPID_SYMBOL_COL = "Stock_symbol"
FNSPID_TEXT_COL = "Article"


@dataclass(frozen=True)
class RawArticle:
    """One FNSPID row, normalised: publication date + primary symbol + body text."""

    date: date
    primary_symbol: str
    text: str


@dataclass(frozen=True)
class Article:
    """An article tagged with the set of in-universe tickers it mentions."""

    date: date
    tickers: tuple[str, ...]
    text: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tickers", tuple(sorted(set(self.tickers))))


def extract_tickers(text: str, alias_map: dict[str, str]) -> set[str]:
    """Return the set of universe tickers whose symbol/alias appears in ``text`` (word-boundary)."""
    lowered = text.lower()
    found: set[str] = set()
    for alias, ticker in alias_map.items():
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            found.add(ticker)
    return found


def load_fnspid_articles(csv_path: Path) -> list[RawArticle]:
    """Read the FNSPID CSV into normalised RawArticle rows (drops rows missing date/symbol/text)."""
    df = pd.read_csv(csv_path, usecols=[FNSPID_DATE_COL, FNSPID_SYMBOL_COL, FNSPID_TEXT_COL])
    out: list[RawArticle] = []
    for _, row in df.iterrows():
        try:
            d = datetime.fromisoformat(str(row[FNSPID_DATE_COL])[:10]).date()
        except ValueError:
            continue
        sym, text = row[FNSPID_SYMBOL_COL], row[FNSPID_TEXT_COL]
        if pd.isna(sym) or pd.isna(text):
            continue
        out.append(RawArticle(date=d, primary_symbol=str(sym).upper(), text=str(text)))
    return out


@dataclass(frozen=True)
class CoMentionEdge:
    a: str
    b: str  # invariant: a < b (unordered pair, canonicalised)
    weight: float
    count: int
    last_seen: date


def _to_articles(
    raws: list[RawArticle], universe: set[str], alias_map: dict[str, str]
) -> list[Article]:
    arts: list[Article] = []
    for r in raws:
        tickers = extract_tickers(r.text, alias_map)
        if r.primary_symbol in universe:
            tickers.add(r.primary_symbol)
        tickers &= universe
        if len(tickers) >= 2:
            arts.append(Article(date=r.date, tickers=tuple(tickers), text=r.text))
    return arts


def build_comention_edges(
    articles: list[Article],
    *,
    as_of: date,
    half_life_days: float = 365.0,
) -> list[CoMentionEdge]:
    """Aggregate recency-weighted co-mention edges from articles strictly BEFORE ``as_of``.

    Weight per article = 0.5 ** (age_days / half_life_days); summed per unordered ticker pair. Edges
    are point-in-time: only articles with ``date < as_of`` contribute (no look-ahead, spec §7).
    """
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    last: dict[tuple[str, str], date] = {}
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for art in articles:
        if art.date >= as_of:
            continue
        age = (as_of - art.date).days
        w = 0.5 ** (age / half_life_days)
        toks = art.tickers
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                key = (toks[i], toks[j])  # already sorted in Article
                acc[key].append(w)
                counts[key] += 1
                last[key] = max(last.get(key, art.date), art.date)
    return [
        CoMentionEdge(a=k[0], b=k[1], weight=float(sum(v)), count=counts[k], last_seen=last[k])
        for k, v in acc.items()
    ]
