"""Tests for the Qwen-backed multi-axis news classifier."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest


def _event() -> TextEvent:  # noqa: F821
    from qts.world.events import TextEvent

    return TextEvent(
        timestamp=datetime(2023, 12, 13, 19, 0, tzinfo=UTC),
        source="fed_press_release",
        persona=None,
        text=(
            "The Committee decided to maintain the target range for the federal funds "
            "rate at 5-1/4 to 5-1/2 percent. The Committee's longer-run goals are to "
            "promote maximum employment and price stability."
        ),
        metadata={"event_kind": "FOMC"},
    )


class _FakeLLM:
    """In-memory stub of LLMClientProtocol.

    Configured with a queue of canned responses; raises if depleted.
    """

    def __init__(self, responses: list[dict] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[tuple[str, str]] = []

    async def query(self, system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:  # noqa: ARG002
        import json

        return json.dumps(self._responses.pop(0))

    async def query_json(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 4096
    ) -> dict:  # noqa: ARG002
        self.calls.append((system_prompt, user_prompt))
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_classifier_returns_news_signal(tmp_path) -> None:  # T-CLF-1
    from qts.macro.news_classifier import NewsClassifier
    from qts.macro.news_signal import NewsSignal

    llm = _FakeLLM(
        responses=[
            {
                "direction": "bull",
                "confidence": 0.7,
                "relevance": 0.9,
                "magnitude": 0.6,
            }
        ]
    )
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    sig = await clf.classify_async(_event())
    assert isinstance(sig, NewsSignal)
    assert sig.direction == "bull"
    assert sig.confidence == 0.7


@pytest.mark.asyncio
async def test_classifier_falls_back_to_neutral_on_malformed(tmp_path) -> None:  # T-CLF-2
    from qts.macro.news_classifier import NewsClassifier

    # Missing required keys.
    llm = _FakeLLM(responses=[{"foo": "bar"}])
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    sig = await clf.classify_async(_event())
    # Fall-back is direction=neutral with all axes zero.
    assert sig.direction == "neutral"
    assert sig.confidence == 0.0
    assert sig.relevance == 0.0
    assert sig.magnitude == 0.0


@pytest.mark.asyncio
async def test_classifier_calls_llm_with_event_text(tmp_path) -> None:  # T-CLF-3
    from qts.macro.news_classifier import NewsClassifier

    llm = _FakeLLM(
        responses=[{"direction": "bear", "confidence": 0.5, "relevance": 0.5, "magnitude": 0.5}]
    )
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    await clf.classify_async(_event())
    assert len(llm.calls) == 1
    system_prompt, user_prompt = llm.calls[0]
    # The event text must appear in the user prompt.
    assert "target range for the federal funds rate" in user_prompt
    # The system prompt should mention BTC (the asset under analysis).
    assert "BTC" in system_prompt or "Bitcoin" in system_prompt


@pytest.mark.asyncio
async def test_classifier_caches_response(tmp_path) -> None:  # T-CLF-4
    from qts.macro.news_classifier import NewsClassifier

    llm = _FakeLLM(
        responses=[{"direction": "bull", "confidence": 0.7, "relevance": 0.9, "magnitude": 0.6}]
    )
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    sig1 = await clf.classify_async(_event())
    # Second call must NOT hit the LLM — fully cached.
    sig2 = await clf.classify_async(_event())

    assert sig1 == sig2
    assert len(llm.calls) == 1  # LLM only called once


def test_classifier_sync_classify_requires_cache_hit(tmp_path) -> None:  # T-CLF-5
    from qts.macro.news_classifier import NewsClassifier

    llm = _FakeLLM(responses=[])
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    with pytest.raises(KeyError, match="not in cache"):
        clf.classify(_event())


@pytest.mark.asyncio
async def test_classifier_sync_classify_works_after_warm(tmp_path) -> None:  # T-CLF-6
    from qts.macro.news_classifier import NewsClassifier

    llm = _FakeLLM(
        responses=[{"direction": "bear", "confidence": 0.5, "relevance": 0.7, "magnitude": 0.4}]
    )
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    await clf.warm_cache_for([_event()])
    # Synchronous lookup — must not call the LLM again.
    sig = clf.classify(_event())

    assert sig.direction == "bear"
    assert sig.confidence == 0.5


@pytest.mark.asyncio
async def test_cache_key_content_addressed(tmp_path) -> None:  # T-CLF-7
    from datetime import timedelta

    from qts.macro.news_classifier import NewsClassifier
    from qts.world.events import TextEvent

    llm = _FakeLLM(
        responses=[
            {"direction": "bull", "confidence": 0.7, "relevance": 0.9, "magnitude": 0.6},
            {"direction": "bear", "confidence": 0.7, "relevance": 0.9, "magnitude": 0.6},
        ]
    )
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    e1 = _event()
    # Same text, different timestamp → SAME cache key (content-addressed).
    e2 = TextEvent(
        timestamp=e1.timestamp + timedelta(hours=1),
        source=e1.source,
        persona=e1.persona,
        text=e1.text,
        metadata={"event_kind": "FOMC"},
    )

    sig1 = await clf.classify_async(e1)
    sig2 = await clf.classify_async(e2)

    # Same text → same cached signal (LLM only called once).
    assert sig1 == sig2
    assert len(llm.calls) == 1
