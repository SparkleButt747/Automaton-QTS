"""T-CRYPTO-LINK-*: crypto relation vocab + LLM soft-link classifier (stubbed LLM, no network)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from qts.propagation.crypto.links import (
    CLASSIFY_ATTEMPTS,
    CRYPTO_RELATIONS,
    CryptoLink,
    CryptoLinkClassifier,
)


class _FakeLLM:
    def __init__(self, responses: list[dict]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def query(self, *a: object, **k: object) -> str:
        raise AssertionError("classifier must use query_json")

    async def query_json(self, *a: object, **k: object) -> dict:
        self.calls += 1
        return self._responses.pop(0)


class _FlakyLLM:
    def __init__(self, fail_times: int, ok: dict | None = None) -> None:
        self._fail_times = fail_times
        self._ok = ok
        self.calls = 0

    async def query(self, *a: object, **k: object) -> str:
        raise AssertionError("classifier must use query_json")

    async def query_json(self, *a: object, **k: object) -> dict:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise ValueError("llama.cpp returned invalid JSON")
        assert self._ok is not None
        return self._ok


def test_vocab_excludes_none(tmp_path: Path) -> None:  # T-CRYPTO-LINK-1
    assert "none" not in CRYPTO_RELATIONS
    assert "entity_exposure" in CRYPTO_RELATIONS and "shares_oracle" in CRYPTO_RELATIONS


def test_classify_returns_typed_link(tmp_path: Path) -> None:  # T-CRYPTO-LINK-2
    llm = _FakeLLM([{"relation": "entity_exposure", "direction": "negative", "confidence": 0.9}])
    clf = CryptoLinkClassifier(llm, cache_dir=tmp_path)
    link = asyncio.run(clf.classify("FTT", "SOL", context="Alameda held a large SOL position"))
    assert isinstance(link, CryptoLink)
    assert link.relation == "entity_exposure" and link.confidence == 0.9


def test_unknown_relation_coerced_to_none(tmp_path: Path) -> None:  # T-CRYPTO-LINK-3
    llm = _FakeLLM([{"relation": "frenemy", "direction": "?", "confidence": 5.0}])
    clf = CryptoLinkClassifier(llm, cache_dir=tmp_path)
    link = asyncio.run(clf.classify("A", "B", context="garbage"))
    assert link.relation == "none" and 0.0 <= link.confidence <= 1.0


def test_persistent_invalid_json_falls_back_to_none(tmp_path: Path) -> None:  # T-CRYPTO-LINK-4
    llm = _FlakyLLM(fail_times=CLASSIFY_ATTEMPTS + 5)
    clf = CryptoLinkClassifier(llm, cache_dir=tmp_path)
    link = asyncio.run(clf.classify("FTT", "SOL", context="ctx"))
    assert link.relation == "none" and llm.calls == CLASSIFY_ATTEMPTS
