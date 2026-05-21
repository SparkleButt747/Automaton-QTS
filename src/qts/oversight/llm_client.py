"""Async LLM client supporting Anthropic Claude and local Ollama backends."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Protocol, runtime_checkable

import anthropic
import httpx

logger = logging.getLogger(__name__)

# Retry delays for exponential backoff (seconds)
_RETRY_BASE_DELAY: float = 1.0
_RETRY_MAX_DELAY: float = 30.0

# Default Ollama configuration
_OLLAMA_DEFAULT_BASE_URL: str = "http://localhost:11434"
_OLLAMA_DEFAULT_MODEL: str = "glm-5:cloud"

# Default llama.cpp (llama-server) configuration. llama-server serves whatever
# GGUF was loaded at startup; the model field is sent for OpenAI compatibility
# but is typically ignored (or matched against --alias).
_LLAMACPP_DEFAULT_BASE_URL: str = "http://localhost:8080"
_LLAMACPP_DEFAULT_MODEL: str = "local"


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


def _extract_json_dict(raw: str, source_label: str) -> dict:  # type: ignore[type-arg]
    """Parse a JSON dict out of a raw LLM response.

    Strips markdown code fences and ``<think>...</think>`` reasoning preambles,
    then extracts the first ``{...}`` block if the text isn't already clean JSON.

    Args:
        raw: The raw text returned by the model.
        source_label: Human-readable backend name for error messages.

    Returns:
        The parsed JSON object as a dict.

    Raises:
        ValueError: If the response cannot be parsed as a JSON dict.
    """
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        inner_lines = lines[1:]
        if inner_lines and inner_lines[-1].strip().startswith("```"):
            inner_lines = inner_lines[:-1]
        text = "\n".join(inner_lines).strip()

    # Some reasoning models emit <think>...</think> before the JSON answer.
    if "<think>" in text and "</think>" in text:
        think_end = text.rfind("</think>")
        text = text[think_end + len("</think>") :].strip()

    # As a last resort, extract the first {...} block from prose.
    if not text.startswith("{"):
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            text = text[brace_start : brace_end + 1]

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{source_label} returned invalid JSON: {exc}\nRaw response: {raw[:500]}"
        ) from exc

    if not isinstance(result, dict):
        raise ValueError(
            f"{source_label} returned valid JSON but not a dict "
            f"(got {type(result).__name__}): {raw[:200]}"
        )
    return result


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
                text_blocks = [block.text for block in response.content if hasattr(block, "text")]
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

        return result


class OllamaClient:
    """Async LLM client for a locally-running Ollama instance.

    Uses the Ollama chat API (POST /api/chat) with httpx.  Implements the
    same :class:`LLMClientProtocol` interface as :class:`LLMClient` so it
    can serve as a drop-in replacement for local development without an
    Anthropic API key.

    Args:
        model: Ollama model tag to use (e.g. ``"glm5-128k:latest"``).
        base_url: Base URL of the Ollama API server.
        max_retries: Maximum number of retry attempts on connection errors.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model: str = _OLLAMA_DEFAULT_MODEL,
        base_url: str = _OLLAMA_DEFAULT_BASE_URL,
        max_retries: int = 3,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._timeout = timeout

    async def query(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        """Send a query to the local Ollama instance and return the raw text.

        Retries with exponential backoff on :class:`httpx.ConnectError` and
        :class:`httpx.TimeoutException`.

        Args:
            system_prompt: The system prompt to set context for the model.
            user_prompt: The user message / query content.
            max_tokens: Passed as ``num_predict`` in Ollama options.

        Returns:
            Raw text content of the model's response.

        Raises:
            httpx.HTTPError: If the request fails after all retries.
        """
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
            },
        }

        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as http:
                    response = await http.post(
                        f"{self._base_url}/api/chat",
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    # Ollama response: {"message": {"content": "..."}}
                    content: str = data["message"]["content"]
                    return content

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    delay = min(
                        _RETRY_BASE_DELAY * (2**attempt),
                        _RETRY_MAX_DELAY,
                    )
                    logger.warning(
                        "Ollama request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1,
                        self._max_retries + 1,
                        type(exc).__name__,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Ollama request failed after %d attempts: %s",
                        self._max_retries + 1,
                        exc,
                    )
                    raise

        # Should not be reached, but satisfies type checker
        raise last_error or RuntimeError("Ollama query failed with no error recorded")

    async def query_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> dict:  # type: ignore[type-arg]
        """Send a query to Ollama and parse the response as JSON.

        Strips markdown code fences if the model wraps the JSON output in them.

        Args:
            system_prompt: The system prompt to set context for the model.
            user_prompt: The user message / query content.
            max_tokens: Passed as ``num_predict`` in Ollama options.

        Returns:
            Parsed JSON dict from the model's response.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON.
            httpx.HTTPError: If the request fails after all retries.
        """
        raw = await self.query(system_prompt, user_prompt, max_tokens)

        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            inner_lines = lines[1:]
            if inner_lines and inner_lines[-1].strip().startswith("```"):
                inner_lines = inner_lines[:-1]
            text = "\n".join(inner_lines).strip()

        # Some reasoning models (e.g. deepseek-r1) emit <think>...</think> tags
        # before the actual JSON answer.  Strip everything up to and including
        # the closing tag so json.loads sees clean JSON.
        if "<think>" in text and "</think>" in text:
            think_end = text.rfind("</think>")
            text = text[think_end + len("</think>") :].strip()

        # As a last resort, extract the first {...} block from the text
        if not text.startswith("{"):
            brace_start = text.find("{")
            brace_end = text.rfind("}")
            if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
                text = text[brace_start : brace_end + 1]

        try:
            result = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Ollama returned invalid JSON: {exc}\nRaw response: {raw[:500]}"
            ) from exc

        if not isinstance(result, dict):
            raise ValueError(
                f"Ollama returned valid JSON but not a dict "
                f"(got {type(result).__name__}): {raw[:200]}"
            )

        return result


class LlamaCppClient:
    """Async LLM client for a locally-running llama.cpp server (``llama-server``).

    Talks to llama-server's OpenAI-compatible chat-completions endpoint
    (``POST /v1/chat/completions``). Implements :class:`LLMClientProtocol` so it
    is a drop-in replacement for :class:`OllamaClient` / :class:`LLMClient`.

    Args:
        model: Model name/alias sent for OpenAI compatibility. llama-server
            serves whatever GGUF was loaded at startup, so this is usually
            ignored (or matched against the server's ``--alias``).
        base_url: Base URL of the llama-server instance.
        max_retries: Maximum retry attempts on connection errors.
        timeout: HTTP request timeout in seconds.
        temperature: Sampling temperature. Defaults to 0.0 for deterministic
            classification output.
    """

    def __init__(
        self,
        model: str = _LLAMACPP_DEFAULT_MODEL,
        base_url: str = _LLAMACPP_DEFAULT_BASE_URL,
        max_retries: int = 3,
        timeout: float = 120.0,
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._timeout = timeout
        self._temperature = temperature

    async def query(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat-completion request to llama-server and return the text.

        Retries with exponential backoff on :class:`httpx.ConnectError` and
        :class:`httpx.TimeoutException`.

        Args:
            system_prompt: The system prompt to set context for the model.
            user_prompt: The user message / query content.
            max_tokens: Maximum number of tokens in the response.

        Returns:
            Raw text content of the model's response.

        Raises:
            httpx.HTTPError: If the request fails after all retries.
        """
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": self._temperature,
            "stream": False,
        }

        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as http:
                    response = await http.post(
                        f"{self._base_url}/v1/chat/completions",
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    # OpenAI-compatible: {"choices": [{"message": {"content": "..."}}]}
                    content: str = data["choices"][0]["message"]["content"]
                    return content

            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < self._max_retries:
                    delay = min(
                        _RETRY_BASE_DELAY * (2**attempt),
                        _RETRY_MAX_DELAY,
                    )
                    logger.warning(
                        "llama.cpp request failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1,
                        self._max_retries + 1,
                        type(exc).__name__,
                        delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "llama.cpp request failed after %d attempts: %s",
                        self._max_retries + 1,
                        exc,
                    )
                    raise

        # Should not be reached, but satisfies type checker
        raise last_error or RuntimeError("llama.cpp query failed with no error recorded")

    async def query_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> dict:  # type: ignore[type-arg]
        """Send a query to llama-server and parse the response as a JSON dict.

        Args:
            system_prompt: The system prompt to set context for the model.
            user_prompt: The user message / query content.
            max_tokens: Maximum number of tokens in the response.

        Returns:
            Parsed JSON dict from the model's response.

        Raises:
            ValueError: If the response cannot be parsed as valid JSON.
            httpx.HTTPError: If the request fails after all retries.
        """
        raw = await self.query(system_prompt, user_prompt, max_tokens)
        return _extract_json_dict(raw, "llama.cpp")


def create_llm_client(backend: str = "ollama", **kwargs: Any) -> LLMClientProtocol:
    """Create an LLM client based on the chosen backend.

    Args:
        backend: ``"llamacpp"`` for a locally-running llama.cpp server,
            ``"ollama"`` for a locally-running Ollama instance, or
            ``"anthropic"`` for the Anthropic Claude API.
        **kwargs: Backend-specific keyword arguments forwarded verbatim to
            the underlying client constructor.

            For ``"llamacpp"``:
                - ``model`` (str): Model name/alias.  Defaults to ``"local"``.
                - ``base_url`` (str): llama-server URL.  Defaults to
                  ``"http://localhost:8080"``.
                - ``max_retries`` (int): Retry attempts on connection errors.
                - ``timeout`` (float): HTTP timeout in seconds.
                - ``temperature`` (float): Sampling temperature.  Defaults to 0.0.

            For ``"ollama"``:
                - ``model`` (str): Ollama model tag.  Defaults to
                  ``"glm-5:cloud"``.
                - ``base_url`` (str): Ollama server URL.  Defaults to
                  ``"http://localhost:11434"``.
                - ``max_retries`` (int): Retry attempts on connection errors.
                - ``timeout`` (float): HTTP timeout in seconds.

            For ``"anthropic"``:
                - ``api_key`` (str, required): Anthropic API key.
                - ``model`` (str): Claude model ID.
                - ``max_retries`` (int): Retry attempts on transient errors.

    Returns:
        An :class:`LLMClientProtocol`-compatible client instance.

    Raises:
        ValueError: If *backend* is not ``"llamacpp"``, ``"ollama"``, or
            ``"anthropic"``.
        TypeError: If required keyword arguments for the chosen backend are
            missing (e.g. ``api_key`` for ``"anthropic"``).
    """
    if backend == "llamacpp":
        return LlamaCppClient(**kwargs)
    if backend == "ollama":
        return OllamaClient(**kwargs)
    if backend == "anthropic":
        return LLMClient(**kwargs)
    raise ValueError(
        f"Unknown LLM backend {backend!r}. Supported backends: 'llamacpp', 'ollama', 'anthropic'."
    )
