"""Qwen-backed multi-axis news classifier.

Strategy-grade decode of arbitrary text into a NewsSignal. Pre-warming
the cache before a backtest is the recommended pattern — see warm_cache_for
(added in Task 5).
"""

from __future__ import annotations

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
    """Multi-axis classifier with optional disk cache (Task 5 adds caching)."""

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        cache_dir: Path,
    ) -> None:
        self._llm = llm_client
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def classify_async(self, event: TextEvent) -> NewsSignal:
        """Classify a TextEvent into a NewsSignal via Qwen."""
        user_prompt = self._render_user_prompt(event)
        try:
            raw = await self._llm.query_json(_SYSTEM_PROMPT, user_prompt)
        except Exception:  # noqa: BLE001
            logger.exception("News classifier LLM call failed; returning neutral signal")
            return NewsSignal(direction="neutral", confidence=0.0, relevance=0.0, magnitude=0.0)

        return self._parse_response(raw)

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
