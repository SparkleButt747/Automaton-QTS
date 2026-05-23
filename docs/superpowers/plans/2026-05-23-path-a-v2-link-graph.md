# Path A v2 — Phase 1: Economic-Link Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a point-in-time **economic-link graph** for US equities from the FNSPID news corpus: extract ticker co-mentions, then LLM-filter them into typed causal links (competitor/supplier/customer/partner). This is the relation graph the meta-propagation operator trains over.

**Architecture:** Three modules under a new `src/qts/propagation/equity/` package. (1) `comention.py` normalises FNSPID rows into `Article(date, tickers, text)` and aggregates point-in-time **co-mention edges**. (2) `economic_links.py` classifies each candidate edge with the existing local LLM (`LLMClientProtocol.query_json`) into a typed `EconomicLink`, content-addressed cache like `NewsClassifier`. (3) `graph.py` assembles confidence-filtered links into a queryable `LinkGraph` exposing `peers_of` (1-hop) and `two_hop_chains` (A→B→C with C not directly linked to A — the decorrelated terminal).

**Tech Stack:** Python 3.11, numpy, pandas (FNSPID CSV), the repo's `LLMClientProtocol` (llama.cpp/Qwen), pytest `--no-cov`. No network in tests — the LLM is stubbed with a canned-dict fake (mirrors the existing `_FakeLLM` pattern in `tests/unit/test_*classifier*`).

**Spec:** `docs/specs/2026-05-23-path-a-v2-nhop-meta-feasibility.md` §3–§5. **Design:** event-propagation design doc §15.

---

## Full pipeline arc (this plan = Phase 1 only)

| Phase | Subsystem | Plan |
|-------|-----------|------|
| **1** | **Economic-link graph** (FNSPID co-mention → LLM-filtered typed links) | **this plan** |
| 2 | Event/label layer (universe + node features + SUE earnings via yfinance + abnormal-return labels) | follow-up plan |
| 3 | Model + gate (real-data correlational baseline + real-data meta-train loop + few-shot + feasibility gate) | follow-up plan |

Each phase is independently testable. Phase 1 outputs a `LinkGraph` consumed by Phase 3's training loop.

## File Structure (Phase 1)

- Create: `src/qts/propagation/equity/__init__.py` — package exports
- Create: `src/qts/propagation/equity/comention.py` — `Article`, `RawArticle`, `load_fnspid_articles`, `extract_tickers`, `CoMentionEdge`, `build_comention_edges`
- Create: `src/qts/propagation/equity/economic_links.py` — `EconomicLink`, `EconomicLinkClassifier`
- Create: `src/qts/propagation/equity/graph.py` — `LinkGraph`
- Test: `tests/unit/test_equity_comention.py` (T-PATHA-COMENTION-*)
- Test: `tests/unit/test_equity_economic_links.py` (T-PATHA-LINK-*)
- Test: `tests/unit/test_equity_link_graph.py` (T-PATHA-GRAPH-*)

> **FNSPID schema note (do in Task 1, step 0):** download a sample of FNSPID (github.com/Zdong104/FNSPID_Financial_News_Dataset) and confirm the CSV header. The dataset exposes a publication date, a primary stock symbol, and article text. The loader below assumes columns `Date`, `Stock_symbol`, `Article`; if the real headers differ, adjust ONLY the column constants at the top of `comention.py` — the normalised `RawArticle` interface downstream stays fixed.

---

### Task 1: FNSPID loader + ticker extraction

**Files:**
- Create: `src/qts/propagation/equity/comention.py`
- Test: `tests/unit/test_equity_comention.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_equity_comention.py
"""T-PATHA-COMENTION-*: FNSPID co-mention edge construction."""

from __future__ import annotations

from datetime import date

from qts.propagation.equity.comention import (
    Article,
    extract_tickers,
)


def test_extract_tickers_matches_symbols_and_aliases() -> None:  # T-PATHA-COMENTION-1
    alias_map = {"aapl": "AAPL", "apple": "AAPL", "nvda": "NVDA", "nvidia": "NVDA"}
    text = "Apple's new chip pressures NVDA; analysts cite AAPL supply gains."
    found = extract_tickers(text, alias_map)
    assert found == {"AAPL", "NVDA"}


def test_extract_tickers_word_boundary_no_false_substring() -> None:  # T-PATHA-COMENTION-2
    # "CAT" (Caterpillar) must not match inside "category"
    alias_map = {"cat": "CAT"}
    assert extract_tickers("the category grew", alias_map) == set()
    assert extract_tickers("CAT raised guidance", alias_map) == {"CAT"}


def test_article_tickers_dedup_and_sorted() -> None:  # T-PATHA-COMENTION-3
    art = Article(date=date(2021, 5, 1), tickers=("NVDA", "AAPL", "AAPL"), text="x")
    # constructor normalises: unique + sorted for deterministic pairing
    assert art.tickers == ("AAPL", "NVDA")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_comention.py --no-cov -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'qts.propagation.equity'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/qts/propagation/equity/comention.py
"""FNSPID -> point-in-time co-mention edges (Path A v2 Phase 1, spec §5)."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_comention.py --no-cov -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qts/propagation/equity/comention.py tests/unit/test_equity_comention.py
git commit -m "feat(equity): FNSPID loader + ticker extraction (Path A v2 Phase 1)"
```

---

### Task 2: Co-mention edge builder (point-in-time, weighted)

**Files:**
- Modify: `tests/unit/test_equity_comention.py` (add edge-builder tests)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/unit/test_equity_comention.py
from qts.propagation.equity.comention import (
    CoMentionEdge,
    Article,
    build_comention_edges,
)


def test_build_edges_canonical_pairs_and_pit_cutoff() -> None:  # T-PATHA-COMENTION-4
    arts = [
        Article(date=date(2020, 1, 1), tickers=("AAPL", "NVDA"), text="x"),
        Article(date=date(2020, 6, 1), tickers=("AAPL", "NVDA"), text="y"),
        Article(date=date(2025, 1, 1), tickers=("AAPL", "NVDA"), text="future"),  # after as_of
    ]
    edges = build_comention_edges(arts, as_of=date(2021, 1, 1), half_life_days=365.0)
    assert len(edges) == 1
    e = edges[0]
    assert (e.a, e.b) == ("AAPL", "NVDA")  # canonical a < b
    assert e.count == 2  # the 2025 article is excluded (look-ahead)
    assert e.last_seen == date(2020, 6, 1)
    assert e.weight > 0.0


def test_build_edges_recency_weight_decays() -> None:  # T-PATHA-COMENTION-5
    recent = [Article(date=date(2020, 12, 31), tickers=("AAPL", "NVDA"), text="r")]
    old = [Article(date=date(2018, 1, 1), tickers=("AAPL", "NVDA"), text="o")]
    as_of = date(2021, 1, 1)
    w_recent = build_comention_edges(recent, as_of=as_of)[0].weight
    w_old = build_comention_edges(old, as_of=as_of)[0].weight
    assert w_recent > w_old  # newer co-mentions weigh more
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_comention.py --no-cov -q`
Expected: FAIL — the import of `build_comention_edges` succeeds (written in Task 1) but if Task 1's body was minimal, confirm both new tests pass. If you implemented `build_comention_edges` fully in Task 1, these tests pass immediately — that is fine; this task locks the behaviour with tests.

- [ ] **Step 3: Write minimal implementation**

Already implemented in Task 1's `comention.py` (`build_comention_edges`). No new code needed — this task's purpose is the behavioural test lock for the point-in-time cutoff and recency decay.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_comention.py --no-cov -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_equity_comention.py
git commit -m "test(equity): lock co-mention point-in-time cutoff + recency decay"
```

---

### Task 3: Economic-link LLM classifier (filter co-mention → typed causal link)

**Files:**
- Create: `src/qts/propagation/equity/economic_links.py`
- Test: `tests/unit/test_equity_economic_links.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_equity_economic_links.py
"""T-PATHA-LINK-*: LLM economic-link classification (stubbed LLM, no network)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from qts.propagation.equity.economic_links import EconomicLink, EconomicLinkClassifier


class _FakeLLM:
    """Canned-dict LLM stub mirroring the repo's _FakeLLM convention."""

    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    async def query(self, system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
        raise AssertionError("classifier must use query_json")

    async def query_json(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 4096
    ) -> dict:
        self.calls.append((system_prompt, user_prompt))
        return self._responses.pop(0)


def test_classify_returns_typed_link(tmp_path: Path) -> None:  # T-PATHA-LINK-1
    llm = _FakeLLM([{"relation": "competitor", "direction": "negative", "confidence": 0.8}])
    clf = EconomicLinkClassifier(llm, cache_dir=tmp_path)
    link = asyncio.run(clf.classify("AMD", "NVDA", context="AMD's launch pressures NVDA share"))
    assert isinstance(link, EconomicLink)
    assert link.source == "AMD" and link.peer == "NVDA"
    assert link.relation == "competitor"
    assert link.confidence == 0.8


def test_classify_none_relation_is_incidental(tmp_path: Path) -> None:  # T-PATHA-LINK-2
    llm = _FakeLLM([{"relation": "none", "direction": "none", "confidence": 0.1}])
    clf = EconomicLinkClassifier(llm, cache_dir=tmp_path)
    link = asyncio.run(clf.classify("AAPL", "XOM", context="both mentioned in a market wrap"))
    assert link.relation == "none"


def test_classify_is_cached(tmp_path: Path) -> None:  # T-PATHA-LINK-3
    llm = _FakeLLM([{"relation": "supplier", "direction": "positive", "confidence": 0.9}])
    clf = EconomicLinkClassifier(llm, cache_dir=tmp_path)
    a = asyncio.run(clf.classify("TSM", "AAPL", context="TSMC supplies Apple"))
    b = asyncio.run(clf.classify("TSM", "AAPL", context="TSMC supplies Apple"))
    assert a == b
    assert len(llm.calls) == 1  # second call served from cache, LLM not hit again


def test_invalid_relation_coerced_to_none(tmp_path: Path) -> None:  # T-PATHA-LINK-4
    llm = _FakeLLM([{"relation": "frenemy", "direction": "?", "confidence": 2.0}])
    clf = EconomicLinkClassifier(llm, cache_dir=tmp_path)
    link = asyncio.run(clf.classify("A", "B", context="garbage"))
    assert link.relation == "none"  # unknown relation -> none
    assert 0.0 <= link.confidence <= 1.0  # confidence clamped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_economic_links.py --no-cov -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'qts.propagation.equity.economic_links'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/qts/propagation/equity/economic_links.py
"""LLM filter: co-mention -> typed causal economic link (Path A v2 Phase 1, spec §5.2)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from qts.oversight.llm_client import LLMClientProtocol

Relation = Literal["competitor", "supplier", "customer", "partner", "none"]
_VALID: frozenset[str] = frozenset(get_args(Relation))

CACHE_VERSION = "v1"
SYSTEM_PROMPT = (
    "You are a financial-relations analyst. Given two US-listed companies co-mentioned in news, "
    "decide whether there is a DIRECT economic link from the first (source) to the second (peer) "
    "such that material news about the source would causally move the peer. Respond ONLY with JSON: "
    '{"relation": one of ["competitor","supplier","customer","partner","none"], '
    '"direction": one of ["positive","negative","none"], "confidence": float 0..1}. '
    "Use 'none' when the co-mention is incidental (e.g. both in a market wrap) with no causal link."
)


@dataclass(frozen=True)
class EconomicLink:
    source: str
    peer: str
    relation: Relation
    direction: Literal["positive", "negative", "none"]
    confidence: float


class EconomicLinkClassifier:
    """Classifies a (source, peer, context) co-mention into a typed link; content-addressed cache."""

    def __init__(self, llm: LLMClientProtocol, cache_dir: Path) -> None:
        self._llm = llm
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, source: str, peer: str, context: str) -> str:
        raw = f"{CACHE_VERSION}|{source}|{peer}|{context}".encode()
        return hashlib.sha256(raw).hexdigest()

    async def classify(self, source: str, peer: str, *, context: str) -> EconomicLink:
        key = self._key(source, peer, context)
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            return self._from_dict(source, peer, json.loads(path.read_text()))
        user = f"Source: {source}\nPeer: {peer}\nNews context: {context}"
        raw = await self._llm.query_json(SYSTEM_PROMPT, user)
        path.write_text(json.dumps(raw))
        return self._from_dict(source, peer, raw)

    @staticmethod
    def _from_dict(source: str, peer: str, raw: dict) -> EconomicLink:
        relation = str(raw.get("relation", "none")).lower()
        if relation not in _VALID:
            relation = "none"
        direction = str(raw.get("direction", "none")).lower()
        if direction not in ("positive", "negative", "none"):
            direction = "none"
        try:
            conf = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        return EconomicLink(
            source=source,
            peer=peer,
            relation=relation,  # type: ignore[arg-type]
            direction=direction,  # type: ignore[arg-type]
            confidence=conf,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_economic_links.py --no-cov -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qts/propagation/equity/economic_links.py tests/unit/test_equity_economic_links.py
git commit -m "feat(equity): LLM economic-link classifier with content-addressed cache"
```

---

### Task 4: LinkGraph assembler + package exports

**Files:**
- Create: `src/qts/propagation/equity/graph.py`
- Create: `src/qts/propagation/equity/__init__.py`
- Test: `tests/unit/test_equity_link_graph.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_equity_link_graph.py
"""T-PATHA-GRAPH-*: filtered economic-link graph queries."""

from __future__ import annotations

from qts.propagation.equity.economic_links import EconomicLink
from qts.propagation.equity.graph import LinkGraph


def _link(s: str, p: str, rel: str = "competitor", conf: float = 0.9) -> EconomicLink:
    return EconomicLink(source=s, peer=p, relation=rel, direction="negative", confidence=conf)


def test_graph_filters_none_and_low_confidence() -> None:  # T-PATHA-GRAPH-1
    links = [
        _link("AMD", "NVDA", conf=0.9),
        _link("AAPL", "XOM", rel="none", conf=0.9),  # incidental -> dropped
        _link("F", "GM", conf=0.2),  # below threshold -> dropped
    ]
    g = LinkGraph.from_links(links, min_confidence=0.5)
    assert {(e.source, e.peer) for e in g.edges} == {("AMD", "NVDA")}


def test_peers_of_returns_one_hop() -> None:  # T-PATHA-GRAPH-2
    g = LinkGraph.from_links([_link("AMD", "NVDA"), _link("AMD", "INTC")], min_confidence=0.5)
    peers = {e.peer for e in g.peers_of("AMD")}
    assert peers == {"NVDA", "INTC"}


def test_two_hop_chains_exclude_direct_links() -> None:  # T-PATHA-GRAPH-3
    # A->B, B->C exist; A->C also exists, so A->B->C is NOT a valid (decorrelated) terminal chain.
    # A->B, B->D exist; A->D does NOT exist => A->B->D is a valid 2-hop chain.
    links = [
        _link("A", "B"),
        _link("B", "C"),
        _link("A", "C"),
        _link("B", "D"),
    ]
    g = LinkGraph.from_links(links, min_confidence=0.5)
    chains = set(g.two_hop_chains("A"))
    assert ("A", "B", "D") in chains
    assert ("A", "B", "C") not in chains  # C directly linked to A -> excluded


def test_public_api_exported() -> None:  # T-PATHA-GRAPH-4
    import qts.propagation.equity as eq

    for name in (
        "Article",
        "CoMentionEdge",
        "build_comention_edges",
        "EconomicLink",
        "EconomicLinkClassifier",
        "LinkGraph",
    ):
        assert hasattr(eq, name), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_link_graph.py --no-cov -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'qts.propagation.equity.graph'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/qts/propagation/equity/graph.py
"""Filtered, queryable economic-link graph (Path A v2 Phase 1, spec §5)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from qts.propagation.equity.economic_links import EconomicLink


@dataclass(frozen=True)
class LinkGraph:
    """Confidence-filtered causal links + 1-hop / 2-hop queries."""

    edges: tuple[EconomicLink, ...]

    @classmethod
    def from_links(
        cls, links: list[EconomicLink], *, min_confidence: float = 0.5
    ) -> LinkGraph:
        kept = tuple(
            e for e in links if e.relation != "none" and e.confidence >= min_confidence
        )
        return cls(edges=kept)

    def _adjacency(self) -> dict[str, list[EconomicLink]]:
        adj: dict[str, list[EconomicLink]] = defaultdict(list)
        for e in self.edges:
            adj[e.source].append(e)
        return adj

    def peers_of(self, source: str) -> list[EconomicLink]:
        return [e for e in self.edges if e.source == source]

    def two_hop_chains(self, source: str) -> list[tuple[str, str, str]]:
        """A->B->C where C is NOT directly linked to A (the decorrelated 2-hop terminal, spec §5)."""
        adj = self._adjacency()
        direct = {e.peer for e in adj.get(source, [])}
        chains: list[tuple[str, str, str]] = []
        for ab in adj.get(source, []):
            b = ab.peer
            for bc in adj.get(b, []):
                c = bc.peer
                if c != source and c not in direct:
                    chains.append((source, b, c))
        return chains
```

```python
# src/qts/propagation/equity/__init__.py
"""Path A v2 — real-equity economic-link graph (Phase 1)."""

from qts.propagation.equity.comention import (
    Article,
    CoMentionEdge,
    RawArticle,
    build_comention_edges,
    extract_tickers,
    load_fnspid_articles,
)
from qts.propagation.equity.economic_links import EconomicLink, EconomicLinkClassifier
from qts.propagation.equity.graph import LinkGraph

__all__ = [
    "Article",
    "CoMentionEdge",
    "EconomicLink",
    "EconomicLinkClassifier",
    "LinkGraph",
    "RawArticle",
    "build_comention_edges",
    "extract_tickers",
    "load_fnspid_articles",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_link_graph.py --no-cov -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Run the full Phase-1 suite + ruff**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_*.py --no-cov -q && .venv/bin/ruff check src/qts/propagation/equity/ tests/unit/test_equity_*.py && .venv/bin/ruff format src/qts/propagation/equity/ tests/unit/test_equity_*.py`
Expected: all tests pass; ruff clean (line length 99).

- [ ] **Step 6: Commit**

```bash
git add src/qts/propagation/equity/graph.py src/qts/propagation/equity/__init__.py tests/unit/test_equity_link_graph.py
git commit -m "feat(equity): filtered LinkGraph with 1-hop + decorrelated 2-hop chain queries"
```

---

## Phases 2 & 3 (outline — separate follow-up plans)

**Phase 2 — Event/label layer:**
- `config/universe/path_a_v2.yaml` + loader (yaml.safe_load + pydantic, à la `world/scenario.py::load_scenario_yaml`): the S&P-500 universe + per-ticker alias list (feeds `extract_tickers`) + GICS sector.
- `src/qts/data/market/yfinance_adapter.py` — add `yfinance` dep; fetch earnings dates + SUE (point-in-time). Tests mock via `responses`/fixtures, no live calls.
- Node features `ξ` builder (sector embedding + log mktcap + market beta / 12-1 momentum / realised vol from Alpaca bars).
- Reaction-label builder: abnormal returns over `[t, t+K]` (K≈1–3d), market/sector-residualised.

**Phase 3 — Model + gate:**
- Real-data `CorrelationalBaseline` analog (β-projection on the named firm's realised move).
- Real-data meta-training loop: train `MetaPropagationGraph` (`qts.propagation.meta`) over **training-link** earnings events on the single universe graph; hold out links/firms; `few_shot_adapt` per link. Reuse `evaluate_meta_transfer` capture/win logic.
- Feasibility gate (1-hop) + 2-hop extension + few-shot value-of-data curve; CLI `scripts/run_path_a_v2.py`.

---

## Self-Review

**Spec coverage (Phase 1 scope):** §3 FNSPID source → Task 1 loader. §5.1 co-mention edges → Tasks 1–2. §5.2 LLM economic-link filter → Task 3. §5.4 2-hop decorrelated chains → Task 4 `two_hop_chains`. §7 point-in-time (date < event) → `build_comention_edges` `as_of` cutoff (T-PATHA-COMENTION-4). Phases 2–3 explicitly deferred to follow-up plans (spec §4 items 1,3,4,5,6,7,8).

**Placeholder scan:** none — every step has runnable code or an explicit "already implemented in Task N" note. FNSPID column names are real constants with a documented verify-the-header step.

**Type consistency:** `EconomicLink` fields (`source, peer, relation, direction, confidence`) are identical across `economic_links.py`, `graph.py`, and all tests. `Article.tickers` is a sorted-unique tuple everywhere. `LinkGraph.from_links(min_confidence=)` matches its test calls. `Relation` literal set is the single source of truth (`get_args`).

---

## Execution Handoff

Phase 1 plan complete and saved. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, two-stage review (spec compliance, then code quality) between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session via executing-plans, batch with checkpoints.
