"""Async Claude API client for LLM oversight jobs."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Protocol, runtime_checkable

import anthropic

logger = logging.getLogger(__name__)

# Retry delays for exponential backoff (seconds)
_RETRY_BASE_DELAY: float = 1.0
_RETRY_MAX_DELAY: float = 30.0


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol for dependency injection of LLM clients."""

    async def query(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        """Send a query to the LLM and return the raw text response."""
        ...

    async def query_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> dict:  # type: ignore[type-arg]
        """Send a query to the LLM and return the parsed JSON response."""
        ...


class LLMClient:
    """Async wrapper around the Anthropic Claude API.

    Handles retry logic with exponential backoff for timeouts and rate limits.

    Args:
        api_key: Anthropic API key.
        model: Claude model ID to use.
        max_retries: Maximum number of retry attempts on transient errors.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        max_retries: int = 3,
    ) -> None:
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_retries = max_retries

    async def query(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        """Send a query to Claude and return the raw text response.

        Retries with exponential backoff on APITimeoutError and RateLimitError.

        Args:
            system_prompt: The system prompt to set context for the model.
            user_prompt: The user message / query content.
            max_tokens: Maximum number of tokens in the response.

        Returns:
            Raw text content of the model's response.

        Raises:
            anthropic.APIError: If the request fails after all retries.
        """
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                # Extract text from the first content block
                text_blocks = [
                    block.text
                    for block in response.content
                    if hasattr(block, "text")
                ]
                return "\n".join(text_blocks)

            except (anthropic.APITimeoutError, anthropic.RateLimitError) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    delay = min(
                        _RETRY_BASE_DELAY * (2**attempt),
                        _RETRY_MAX_DELAY,
                    )
                    logger.warning(
                        "LLM request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1,
                        self._max_retries + 1,
                        type(exc).__name__,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "LLM request failed after %d attempts: %s",
                        self._max_retries + 1,
                        exc,
                    )
                    raise

        # Should not be reached, but satisfies type checker
        raise last_error or RuntimeError("LLM query failed with no error recorded")

    async def query_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> dict:  # type: ignore[type-arg]
        """Send a query to Claude and parse the response as JSON.

        Args:
            system_prompt: The system prompt to set context for the model.
            user_prompt: The user message / query content.
            max_tokens: Maximum number of tokens in the response.

        Returns:
            Parsed JSON dict from the model's response.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON.
            anthropic.APIError: If the request fails after all retries.
        """
        raw = await self.query(system_prompt, user_prompt, max_tokens)

        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            # Remove opening fence (with optional language specifier)
            lines = text.split("\n")
            # First line is the fence, last line is closing fence
            inner_lines = lines[1:]
            if inner_lines and inner_lines[-1].strip().startswith("```"):
                inner_lines = inner_lines[:-1]
            text = "\n".join(inner_lines).strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned invalid JSON: {exc}\nRaw response: {raw[:500]}"
            ) from exc

        if not isinstance(result, dict):
            raise ValueError(
                f"LLM returned valid JSON but not a dict (got {type(result).__name__}): {raw[:200]}"
            )

        return result  # type: ignore[return-value]
