"""Unit tests for qts.oversight.llm_client.

Tests:
- Handles API timeout gracefully (mock raises Timeout)
- Retries on rate limit error
- Parses valid JSON response
- Raises ValueError on invalid JSON
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
import pytest

from qts.oversight.llm_client import LLMClient, LLMClientProtocol


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_mock_response(text: str) -> MagicMock:
    """Create a mock Anthropic API response with the given text content."""
    content_block = MagicMock()
    content_block.text = text
    response = MagicMock()
    response.content = [content_block]
    return response


# ── Protocol conformance ───────────────────────────────────────────────────────


class TestLLMClientProtocol:
    def test_llm_client_implements_protocol(self) -> None:
        """LLMClient must implement LLMClientProtocol."""
        client = LLMClient(api_key="test-key")
        assert isinstance(client, LLMClientProtocol)

    def test_protocol_is_runtime_checkable(self) -> None:
        """LLMClientProtocol must be runtime_checkable."""
        # An object that matches the protocol shape
        class FakeClient:
            async def query(self, system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
                return ""

            async def query_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> dict:
                return {}

        assert isinstance(FakeClient(), LLMClientProtocol)


# ── query() method ─────────────────────────────────────────────────────────────


class TestLLMClientQuery:
    @pytest.mark.asyncio
    async def test_successful_query_returns_text(self) -> None:
        """A successful API call should return the text content."""
        client = LLMClient(api_key="test-key", max_retries=0)
        mock_response = _make_mock_response("Hello from Claude")

        with patch.object(
            client._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.query("system", "user")

        assert result == "Hello from Claude"

    @pytest.mark.asyncio
    async def test_handles_api_timeout_raises_after_retries(self) -> None:
        """Should raise APITimeoutError after exhausting retries."""
        client = LLMClient(api_key="test-key", max_retries=2)

        with patch.object(
            client._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=anthropic.APITimeoutError(request=MagicMock()),
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(anthropic.APITimeoutError):
                    await client.query("system", "user")

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self) -> None:
        """Should retry on RateLimitError and eventually succeed."""
        client = LLMClient(api_key="test-key", max_retries=2)
        mock_response = _make_mock_response("Success after retry")

        # First two calls raise RateLimitError, third succeeds
        rate_limit_error = anthropic.RateLimitError(
            message="rate limited",
            response=MagicMock(headers={}),
            body={},
        )

        call_results = [
            rate_limit_error,
            rate_limit_error,
            mock_response,
        ]

        async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
            result = call_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        with patch.object(client._client.messages, "create", side_effect=mock_create):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await client.query("system", "user")

        assert result == "Success after retry"

    @pytest.mark.asyncio
    async def test_retries_on_timeout_eventually_succeeds(self) -> None:
        """Should retry on APITimeoutError and succeed on the last attempt."""
        client = LLMClient(api_key="test-key", max_retries=3)
        mock_response = _make_mock_response("Final success")

        timeout_error = anthropic.APITimeoutError(request=MagicMock())

        call_results: list = [timeout_error, timeout_error, mock_response]

        async def mock_create(**kwargs):  # type: ignore[no-untyped-def]
            result = call_results.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        with patch.object(client._client.messages, "create", side_effect=mock_create):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await client.query("system", "user")

        assert result == "Final success"

    @pytest.mark.asyncio
    async def test_exhausts_all_retries_on_persistent_timeout(self) -> None:
        """Should raise after max_retries+1 attempts all fail."""
        client = LLMClient(api_key="test-key", max_retries=2)

        with patch.object(
            client._client.messages,
            "create",
            new_callable=AsyncMock,
            side_effect=anthropic.APITimeoutError(request=MagicMock()),
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                with pytest.raises(anthropic.APITimeoutError):
                    await client.query("system", "user")

        # Should have slept for retries (max_retries=2 => 2 sleeps between attempts)
        assert mock_sleep.call_count == 2


# ── query_json() method ────────────────────────────────────────────────────────


class TestLLMClientQueryJson:
    @pytest.mark.asyncio
    async def test_parses_valid_json_response(self) -> None:
        """Should parse a valid JSON string response into a dict."""
        client = LLMClient(api_key="test-key", max_retries=0)
        payload = {"key": "value", "number": 42}
        mock_response = _make_mock_response(json.dumps(payload))

        with patch.object(
            client._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.query_json("system", "user")

        assert result == payload

    @pytest.mark.asyncio
    async def test_parses_json_with_markdown_fences(self) -> None:
        """Should strip markdown code fences before parsing."""
        client = LLMClient(api_key="test-key", max_retries=0)
        payload = {"analysis": "good session"}
        fenced = "```json\n" + json.dumps(payload) + "\n```"
        mock_response = _make_mock_response(fenced)

        with patch.object(
            client._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await client.query_json("system", "user")

        assert result == payload

    @pytest.mark.asyncio
    async def test_raises_value_error_on_invalid_json(self) -> None:
        """Should raise ValueError when the response is not valid JSON."""
        client = LLMClient(api_key="test-key", max_retries=0)
        mock_response = _make_mock_response("This is not JSON at all!")

        with patch.object(
            client._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(ValueError, match="invalid JSON"):
                await client.query_json("system", "user")

    @pytest.mark.asyncio
    async def test_raises_value_error_on_json_array(self) -> None:
        """Should raise ValueError when JSON is valid but not a dict."""
        client = LLMClient(api_key="test-key", max_retries=0)
        mock_response = _make_mock_response("[1, 2, 3]")

        with patch.object(
            client._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(ValueError):
                await client.query_json("system", "user")

    @pytest.mark.asyncio
    async def test_raises_value_error_on_empty_response(self) -> None:
        """Should raise ValueError when the response is empty."""
        client = LLMClient(api_key="test-key", max_retries=0)
        mock_response = _make_mock_response("")

        with patch.object(
            client._client.messages,
            "create",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            with pytest.raises(ValueError):
                await client.query_json("system", "user")
