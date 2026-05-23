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
