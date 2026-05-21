"""Qwen-backed multi-axis news classifier.

Strategy-grade decode of arbitrary text into a NewsSignal. Pre-warming
the cache before a backtest is the recommended pattern — see warm_cache_for
(added in Task 5).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from qts.macro.news_signal import NewsSignal

if TYPE_CHECKING:
    from qts.oversight.llm_client import LLMClientProtocol
    from qts.world.events import TextEvent

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert macroeconomic and crypto-market analyst. Given a piece of text
(typically a Fed/FOMC release, central-bank speech, or financial news headline),
classify its likely impact on Bitcoin (BTC) spot price.

Respond ONLY with a valid JSON object matching this schema, no markdown fences,
no commentary, no preamble:

  {
    "direction":  "bull" | "bear" | "neutral",
    "confidence": <float 0.0 to 1.0>,
    "relevance":  <float 0.0 to 1.0>,
    "magnitude":  <float 0.0 to 1.0>
  }

- direction: which way is BTC most likely to move on this text?
- confidence: how sure are you about the direction call?
- relevance: how directly does this text affect BTC price?
- magnitude: how large a price move does this imply?

Be calibrated. Hedged Fed language often warrants confidence < 0.5. Off-topic
text (sports, weather) warrants relevance < 0.2.
"""


class NewsClassifier:
    """Multi-axis classifier with content-addressed disk cache.

    Recommended workflow:
        clf = NewsClassifier(llm_client, cache_dir=Path("data/news_cache"))
        await clf.warm_cache_for(events)        # pre-warm: async LLM calls
        # ... in the strategy's on_text path:
        signal = clf.classify(event)            # sync, cache-only lookup
    """

    _CACHE_KEY_VERSION = "v1"  # bump to invalidate all cached entries

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        cache_dir: Path,
    ) -> None:
        self._llm = llm_client
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, NewsSignal] = {}

    # ---- Public API ------------------------------------------------------

    async def classify_async(self, event: TextEvent) -> NewsSignal:
        """Classify an event, hitting cache first; otherwise call the LLM."""
        key = self._cache_key(event)
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        user_prompt = self._render_user_prompt(event)
        try:
            raw = await self._llm.query_json(_SYSTEM_PROMPT, user_prompt)
        except Exception:  # noqa: BLE001
            logger.exception("News classifier LLM call failed; returning neutral signal")
            return NewsSignal(direction="neutral", confidence=0.0, relevance=0.0, magnitude=0.0)

        signal = self._parse_response(raw)
        self._write_cache(key, signal)
        return signal

    def classify(self, event: TextEvent) -> NewsSignal:
        """Synchronous lookup. Requires the event to be in cache (call warm_cache_for first)."""
        key = self._cache_key(event)
        cached = self._read_cache(key)
        if cached is None:
            raise KeyError(
                f"event with cache key {key!r} not in cache — call warm_cache_for(events) first"
            )
        return cached

    async def warm_cache_for(self, events: list[TextEvent]) -> None:
        """Pre-classify a batch of events so .classify() can serve them synchronously."""
        for event in events:
            await self.classify_async(event)

    # ---- Internal helpers -----------------------------------------------

    def _cache_key(self, event: TextEvent) -> str:
        """Content-addressed key: sha256(source || text || version)."""
        payload = f"{event.source}\n{event.text}\n{self._CACHE_KEY_VERSION}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> NewsSignal | None:
        if key in self._memory_cache:
            return self._memory_cache[key]
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            signal = NewsSignal(**raw)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            logger.warning("Failed to load cache entry %s: %s", path, exc)
            return None
        self._memory_cache[key] = signal
        return signal

    def _write_cache(self, key: str, signal: NewsSignal) -> None:
        self._memory_cache[key] = signal
        path = self._cache_dir / f"{key}.json"
        path.write_text(
            json.dumps(
                {
                    "direction": signal.direction,
                    "confidence": signal.confidence,
                    "relevance": signal.relevance,
                    "magnitude": signal.magnitude,
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _render_user_prompt(event: TextEvent) -> str:
        return (
            f"Source: {event.source}\n"
            f"Timestamp: {event.timestamp.isoformat()}\n"
            f'Text:\n"""\n{event.text}\n"""\n'
        )

    @staticmethod
    def _parse_response(raw: dict) -> NewsSignal:
        """Parse LLM JSON into a NewsSignal; fall back to neutral on malformed."""
        try:
            return NewsSignal(
                direction=raw["direction"],
                confidence=float(raw["confidence"]),
                relevance=float(raw["relevance"]),
                magnitude=float(raw["magnitude"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("News classifier produced malformed response %r: %s", raw, exc)
            return NewsSignal(direction="neutral", confidence=0.0, relevance=0.0, magnitude=0.0)
