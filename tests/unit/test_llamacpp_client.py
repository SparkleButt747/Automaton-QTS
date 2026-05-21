"""Unit tests for qts.oversight.llm_client.LlamaCppClient.

Tests:
- Protocol conformance (isinstance check against LLMClientProtocol)
- create_llm_client('llamacpp') factory returns LlamaCppClient
- Successful query() returns the assistant message text (OpenAI shape)
- query() sends the OpenAI-compatible chat payload
- query_json() parses valid JSON, strips fences, strips <think> tags,
  extracts a JSON block from prose, and raises on invalid/non-dict JSON
- Connection errors trigger retry-then-raise behaviour
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from qts.oversight.llm_client import (
    LlamaCppClient,
    LLMClientProtocol,
    create_llm_client,
)

# ── Helpers ────────────────────────────────────────────────────────────────────


def _mock_httpx_response(content: str, status_code: int = 200) -> MagicMock:
    """Return a mock httpx.Response whose .json() gives the OpenAI chat structure."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": content}}]
    }
    mock_resp.raise_for_status = MagicMock()  # no-op for 200
    return mock_resp


# ── Protocol conformance ───────────────────────────────────────────────────────


class TestLlamaCppClientProtocol:
    def test_llamacpp_client_implements_protocol(self) -> None:
        client = LlamaCppClient()
        assert isinstance(client, LLMClientProtocol)

    def test_create_llm_client_llamacpp_returns_llamacpp_client(self) -> None:
        client = create_llm_client(backend="llamacpp")
        assert isinstance(client, LlamaCppClient)

    def test_create_llm_client_llamacpp_implements_protocol(self) -> None:
        client = create_llm_client(backend="llamacpp")
        assert isinstance(client, LLMClientProtocol)


# ── query() ────────────────────────────────────────────────────────────────────


class TestLlamaCppClientQuery:
    @pytest.mark.asyncio
    async def test_successful_query_returns_text(self) -> None:
        client = LlamaCppClient(max_retries=0)
        mock_resp = _mock_httpx_response("Hello from llama.cpp")

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await client.query("system", "user")

        assert result == "Hello from llama.cpp"

    @pytest.mark.asyncio
    async def test_query_sends_openai_payload(self) -> None:
        client = LlamaCppClient(model="qwen-test", max_retries=0, temperature=0.0)
        mock_resp = _mock_httpx_response("ok")
        captured: dict = {}

        async def mock_post(url, **kwargs):  # type: ignore[no-untyped-def]
            captured["url"] = url
            captured.update(kwargs)
            return mock_resp

        with patch("httpx.AsyncClient.post", side_effect=mock_post):
            await client.query("sys", "usr", max_tokens=512)

        assert captured["url"].endswith("/v1/chat/completions")
        payload = captured["json"]
        assert payload["model"] == "qwen-test"
        assert payload["stream"] is False
        assert payload["max_tokens"] == 512
        assert payload["temperature"] == 0.0
        assert payload["messages"][0] == {"role": "system", "content": "sys"}
        assert payload["messages"][1] == {"role": "user", "content": "usr"}

    @pytest.mark.asyncio
    async def test_connect_error_retries_then_raises(self) -> None:
        client = LlamaCppClient(max_retries=2)

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connection refused"),
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                with pytest.raises(httpx.ConnectError):
                    await client.query("sys", "usr")

        # max_retries=2 -> 3 attempts -> 2 sleeps
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_connect_error_retries_then_succeeds(self) -> None:
        client = LlamaCppClient(max_retries=2)
        mock_resp = _mock_httpx_response("Success after retry")
        call_count = 0

        async def mock_post(url, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("connection refused")
            return mock_resp

        with patch("httpx.AsyncClient.post", side_effect=mock_post):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await client.query("sys", "usr")

        assert result == "Success after retry"


# ── query_json() ───────────────────────────────────────────────────────────────


class TestLlamaCppClientQueryJson:
    @pytest.mark.asyncio
    async def test_parses_valid_json_response(self) -> None:
        client = LlamaCppClient(max_retries=0)
        payload = {"direction": "bull", "confidence": 0.8}
        mock_resp = _mock_httpx_response(json.dumps(payload))

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.query_json("sys", "usr")

        assert result == payload

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self) -> None:
        client = LlamaCppClient(max_retries=0)
        payload = {"key": "value"}
        fence = "`" * 3
        fenced = fence + "json\n" + json.dumps(payload) + "\n" + fence
        mock_resp = _mock_httpx_response(fenced)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.query_json("sys", "usr")

        assert result == payload

    @pytest.mark.asyncio
    async def test_strips_think_tags(self) -> None:
        client = LlamaCppClient(max_retries=0)
        payload = {"answer": 42}
        raw = "<think>reasoning here...</think>\n" + json.dumps(payload)
        mock_resp = _mock_httpx_response(raw)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.query_json("sys", "usr")

        assert result == payload

    @pytest.mark.asyncio
    async def test_extracts_json_block_from_prose(self) -> None:
        client = LlamaCppClient(max_retries=0)
        payload = {"score": 0.9}
        raw = "Here is the result: " + json.dumps(payload) + " Done."
        mock_resp = _mock_httpx_response(raw)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.query_json("sys", "usr")

        assert result == payload

    @pytest.mark.asyncio
    async def test_raises_value_error_on_invalid_json(self) -> None:
        client = LlamaCppClient(max_retries=0)
        mock_resp = _mock_httpx_response("This is definitely not JSON")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(ValueError, match="invalid JSON"):
                await client.query_json("sys", "usr")

    @pytest.mark.asyncio
    async def test_raises_value_error_on_json_array(self) -> None:
        client = LlamaCppClient(max_retries=0)
        mock_resp = _mock_httpx_response("[1, 2, 3]")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(ValueError):
                await client.query_json("sys", "usr")
