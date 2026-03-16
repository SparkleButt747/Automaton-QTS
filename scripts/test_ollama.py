#!/usr/bin/env python3
"""Functional test script for the OllamaClient.

Demonstrates that the local Ollama backend works end-to-end:
  1. A plain-text query using a quant-analyst system prompt.
  2. A JSON-structured query asking for a mock parameter-change proposal.

Usage::

    python scripts/test_ollama.py

Requirements:
    - Ollama must be running locally (``ollama serve``).
    - At least one model must be pulled (``ollama pull <model>``).
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys

# Ensure the src package is importable when the script is run directly
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

from qts.oversight.llm_client import OllamaClient, create_llm_client  # noqa: E402

# ---- Configuration -----------------------------------------------------------

MODEL = "glm-5:cloud"
BASE_URL = "http://localhost:11434"

# ---- Prompts -----------------------------------------------------------------

PLAIN_SYSTEM = "You are a quant analyst. Be concise and precise."
PLAIN_USER = (
    "Summarize in 2 sentences: " "BTC dropped 5% today on high volume with negative news sentiment."
)

JSON_SYSTEM = (
    "You are a quantitative trading system oversight AI. "
    "Respond ONLY with a JSON object, no extra text, no markdown fences.\n"
    "The JSON must have this exact structure:\n"
    "{\n"
    '  "parameter": "<string>",\n'
    '  "current_value": <float>,\n'
    '  "proposed_value": <float>,\n'
    '  "reason": "<string>",\n'
    '  "confidence": <float between 0.0 and 1.0>\n'
    "}"
)
JSON_USER = (
    "Given that BTC dropped 5% on high volume with negative sentiment today, "
    "propose a parameter change for w_sentiment (currently 0.20). "
    "Return the JSON object only."
)

# ---- Main --------------------------------------------------------------------


async def run_tests() -> None:
    sep = "=" * 60
    print(sep)
    print("OllamaClient Functional Test")
    print(f"Model : {MODEL}")
    print(f"Server: {BASE_URL}")
    print(sep)

    # Test 1: plain-text query via OllamaClient constructor
    print("\n[1/2] Plain-text query via OllamaClient(...)")
    print(f"  User: {PLAIN_USER}")
    client = OllamaClient(model=MODEL, base_url=BASE_URL, max_retries=1)
    text_response = await client.query(PLAIN_SYSTEM, PLAIN_USER)
    print(f"  Response:\n    {text_response.strip()}")

    # Test 2: JSON query via create_llm_client() factory
    print("\n[2/2] JSON query via create_llm_client(backend='ollama')")
    print(f"  User: {JSON_USER}")
    factory_client = create_llm_client(
        backend="ollama",
        model=MODEL,
        base_url=BASE_URL,
        max_retries=1,
    )
    json_response = await factory_client.query_json(JSON_SYSTEM, JSON_USER)
    print(f"  Parsed JSON:\n    {json.dumps(json_response, indent=4)}")

    print("\n" + sep)
    print("All tests passed.")
    print(sep)


def main() -> None:
    try:
        asyncio.run(run_tests())
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
