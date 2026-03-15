"""Unit tests for qts.oversight.llm_client.OllamaClient.

Tests:
- Protocol conformance (isinstance check against LLMClientProtocol)
- Successful query() returns the assistant message text
- Successful query_json() parses a valid JSON response
- query_json() strips markdown code fences
- query_json() strips <think>...</think> reasoning preambles
- query_json() raises ValueError on invalid JSON
- query_json() raises ValueError when JSON is not a dict
- Connection errors trigger retry-then-raise behaviour
- create_llm_client() factory returns the correct type
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from qts.oversight.llm_client import (
    LLMClientProtocol,
    OllamaClient,
    create_llm_client,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _mock_httpx_response(content: str, status_code: int = 200) -> MagicMock:
    """Return a mock httpx.Response whose .json() gives the Ollama chat structure."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {
        "message": {"role": "assistant", "content": content}
    }
    mock_resp.raise_for_status = MagicMock()  # no-op for 200
    return mock_resp


# ── Protocol conformance ───────────────────────────────────────────────────────


class TestOllamaClientProtocol:
    def test_ollama_client_implements_protocol(self) -> None:
        """OllamaClient must satisfy LLMClientProtocol at runtime."""
        client = OllamaClient()
        assert isinstance(client, LLMClientProtocol)

    def test_create_llm_client_ollama_returns_ollama_client(self) -> None:
        """create_llm_client('ollama') must return an OllamaClient."""
        client = create_llm_client(backend="ollama")
        assert isinstance(client, OllamaClient)

    def test_create_llm_client_ollama_implements_protocol(self) -> None:
        """OllamaClient from factory must implement LLMClientProtocol."""
        client = create_llm_client(backend="ollama")
        assert isinstance(client, LLMClientProtocol)

    def test_create_llm_client_unknown_backend_raises(self) -> None:
        """create_llm_client with an unknown backend must raise ValueError."""
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            create_llm_client(backend="unknown-backend")


# ── query() ────────────────────────────────────────────────────────────────────


class TestOllamaClientQuery:
    @pytest.mark.asyncio
    async def test_successful_query_returns_text(self) -> None:
        """A successful POST /api/chat call should return the assistant content."""
        client = OllamaClient(max_retries=0)
        mock_resp = _mock_httpx_response("Hello from Ollama")

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ):
            result = await client.query("system", "user")

        assert result == "Hello from Ollama"

    @pytest.mark.asyncio
    async def test_query_sends_correct_payload(self) -> None:
        """The POST body must include model, messages, and stream=False."""
        client = OllamaClient(model="test-model", max_retries=0)
        mock_resp = _mock_httpx_response("ok")
        captured_kwargs: dict = {}

        async def mock_post(url, **kwargs):  # type: ignore[no-untyped-def]
            captured_kwargs.update(kwargs)
            return mock_resp

        with patch("httpx.AsyncClient.post", side_effect=mock_post):
            await client.query("sys", "usr", max_tokens=512)

        payload = captured_kwargs["json"]
        assert payload["model"] == "test-model"
        assert payload["stream"] is False
        assert payload["messages"][0] == {"role": "system", "content": "sys"}
        assert payload["messages"][1] == {"role": "user", "content": "usr"}
        assert payload["options"]["num_predict"] == 512

    @pytest.mark.asyncio
    async def test_connect_error_retries_then_raises(self) -> None:
        """ConnectError should trigger retry logic and eventually raise."""
        client = OllamaClient(max_retries=2)

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connection refused"),
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                with pytest.raises(httpx.ConnectError):
                    await client.query("sys", "usr")

        # max_retries=2 means 3 total attempts => 2 sleeps between them
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_connect_error_retries_then_succeeds(self) -> None:
        """Should retry on ConnectError and succeed when a later attempt works."""
        client = OllamaClient(max_retries=2)
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

    @pytest.mark.asyncio
    async def test_timeout_retries_then_raises(self) -> None:
        """TimeoutException should trigger retry logic and eventually raise."""
        client = OllamaClient(max_retries=1)

        with patch(
            "httpx.AsyncClient.post",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("timed out"),
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(httpx.TimeoutException):
                    await client.query("sys", "usr")


# ── query_json() ───────────────────────────────────────────────────────────────


class TestOllamaClientQueryJson:
    @pytest.mark.asyncio
    async def test_parses_valid_json_response(self) -> None:
        """Should parse a valid JSON string returned by Ollama."""
        client = OllamaClient(max_retries=0)
        payload = {"regime": "normal", "confidence": 0.8}
        mock_resp = _mock_httpx_response(json.dumps(payload))

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.query_json("sys", "usr")

        assert result == payload

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self) -> None:
        """Should unwrap code-fenced JSON before parsing."""
        client = OllamaClient(max_retries=0)
        payload = {"key": "value"}
        fence = "`" * 3
        fenced = fence + "json\n" + json.dumps(payload) + "\n" + fence
        mock_resp = _mock_httpx_response(fenced)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.query_json("sys", "usr")

        assert result == payload

    @pytest.mark.asyncio
    async def test_strips_think_tags(self) -> None:
        """Should strip <think>...</think> reasoning preamble before parsing."""
        client = OllamaClient(max_retries=0)
        payload = {"answer": 42}
        raw = "<think>Let me reason through this step by step...</think>\n" + json.dumps(payload)
        mock_resp = _mock_httpx_response(raw)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.query_json("sys", "usr")

        assert result == payload

    @pytest.mark.asyncio
    async def test_extracts_json_block_from_prose(self) -> None:
        """Should extract the first {...} block from prose-wrapped responses."""
        client = OllamaClient(max_retries=0)
        payload = {"score": 0.9}
        raw = "Here is my analysis: " + json.dumps(payload) + " That is my final answer."
        mock_resp = _mock_httpx_response(raw)

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            result = await client.query_json("sys", "usr")

        assert result == payload

    @pytest.mark.asyncio
    async def test_raises_value_error_on_invalid_json(self) -> None:
        """Should raise ValueError when the response is not valid JSON."""
        client = OllamaClient(max_retries=0)
        mock_resp = _mock_httpx_response("This is definitely not JSON")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(ValueError, match="invalid JSON"):
                await client.query_json("sys", "usr")

    @pytest.mark.asyncio
    async def test_raises_value_error_on_json_array(self) -> None:
        """Should raise ValueError when JSON is valid but not a dict."""
        client = OllamaClient(max_retries=0)
        mock_resp = _mock_httpx_response("[1, 2, 3]")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(ValueError):
                await client.query_json("sys", "usr")

    @pytest.mark.asyncio
    async def test_raises_value_error_on_empty_response(self) -> None:
        """Should raise ValueError when the response text is empty."""
        client = OllamaClient(max_retries=0)
        mock_resp = _mock_httpx_response("")

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(ValueError):
                await client.query_json("sys", "usr")
