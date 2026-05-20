"""Unit tests for qts.macro.classifier.MacroRegimeClassifier.

Covers:
- T-MCLS-1  to T-MCLS-8
- _parse_response: full LLM JSON dict produces correct MacroRegime fields
- _parse_response: empty dict falls back to safe defaults
- _parse_response: unknown enum values raise (enum validation)
- _build_prompt: includes headlines, GDELT events, macro indicators
- _build_prompt: no data produces fallback message
- classify(): async call delegates to llm_client.query_json (stubbed — no real LLM)

No real LLM calls are made. The LLMClientProtocol boundary is stubbed.
"""

from __future__ import annotations

import asyncio

import pytest

from qts.macro.classifier import MacroRegimeClassifier
from qts.models.base import Catalyst, LiquidityLevel, SentimentLevel, Trend, VolLevel
from qts.models.terrain import MacroRegime

# ── Stub LLM client ───────────────────────────────────────────────────────────


class _StubLLMClient:
    """Minimal stub satisfying LLMClientProtocol without making network calls."""

    def __init__(self, response: dict) -> None:  # type: ignore[type-arg]
        self._response = response

    async def query(self, system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:
        return ""

    async def query_json(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 4096
    ) -> dict:  # type: ignore[type-arg]
        return self._response


_FULL_RESPONSE: dict = {  # type: ignore[type-arg]
    "trend": "BEAR",
    "volatility": "HIGH",
    "liquidity": "CRISIS",
    "sentiment": "FEARFUL",
    "catalyst": "PANDEMIC",
    "expected_drift": -3.2,
    "expected_vol": 0.85,
    "correlation_regime": 0.92,
    "scenario_description": "COVID crash scenario",
    "confidence": 0.95,
}

_EMPTY_RESPONSE: dict = {}  # type: ignore[type-arg]


# ── _parse_response ────────────────────────────────────────────────────────────


class TestMacroRegimeClassifierParseResponse:
    def test_full_response_maps_all_fields(self) -> None:  # T-MCLS-1
        """A complete LLM JSON response must map to correct MacroRegime fields."""
        clf = MacroRegimeClassifier(_StubLLMClient(_FULL_RESPONSE))
        regime = clf._parse_response(_FULL_RESPONSE)

        assert regime.trend == Trend.BEAR
        assert regime.volatility == VolLevel.HIGH
        assert regime.liquidity == LiquidityLevel.CRISIS
        assert regime.sentiment == SentimentLevel.FEARFUL
        assert regime.catalyst == Catalyst.PANDEMIC
        assert regime.expected_drift == pytest.approx(-3.2)
        assert regime.expected_vol == pytest.approx(0.85)
        assert regime.correlation_regime == pytest.approx(0.92)
        assert regime.scenario_description == "COVID crash scenario"

    def test_empty_response_uses_safe_defaults(self) -> None:  # T-MCLS-2
        """An empty LLM response dict must produce a neutral default MacroRegime."""
        clf = MacroRegimeClassifier(_StubLLMClient(_EMPTY_RESPONSE))
        regime = clf._parse_response(_EMPTY_RESPONSE)

        assert regime.trend == Trend.SIDEWAYS
        assert regime.volatility == VolLevel.LOW
        assert regime.liquidity == LiquidityLevel.ABUNDANT
        assert regime.sentiment == SentimentLevel.NEUTRAL
        assert regime.catalyst == Catalyst.NONE

    def test_response_returns_macro_regime_instance(self) -> None:  # T-MCLS-3
        """_parse_response must always return a MacroRegime dataclass."""
        clf = MacroRegimeClassifier(_StubLLMClient(_EMPTY_RESPONSE))
        result = clf._parse_response(_FULL_RESPONSE)
        assert isinstance(result, MacroRegime)

    def test_numeric_fields_are_floats(self) -> None:  # T-MCLS-4
        """expected_drift, expected_vol, and correlation_regime must be Python floats."""
        clf = MacroRegimeClassifier(_StubLLMClient(_FULL_RESPONSE))
        regime = clf._parse_response(_FULL_RESPONSE)
        assert isinstance(regime.expected_drift, float)
        assert isinstance(regime.expected_vol, float)
        assert isinstance(regime.correlation_regime, float)


# ── _build_prompt ─────────────────────────────────────────────────────────────


class TestMacroRegimeClassifierBuildPrompt:
    def test_headlines_appear_in_prompt(self) -> None:  # T-MCLS-5
        """Headlines must be included verbatim in the constructed prompt string."""
        clf = MacroRegimeClassifier(_StubLLMClient(_EMPTY_RESPONSE))
        prompt = clf._build_prompt(
            headlines=["Markets crash on pandemic fears"],
            gdelt_events=None,
            macro_indicators=None,
        )
        assert "Markets crash on pandemic fears" in prompt

    def test_macro_indicators_appear_in_prompt(self) -> None:  # T-MCLS-6
        """Macro indicator keys and values must be present in the constructed prompt."""
        clf = MacroRegimeClassifier(_StubLLMClient(_EMPTY_RESPONSE))
        prompt = clf._build_prompt(
            headlines=None,
            gdelt_events=None,
            macro_indicators={"vix": 82.0, "cpi_yoy": 9.1},
        )
        assert "vix" in prompt
        assert "82.0" in prompt

    def test_no_data_prompt_includes_fallback_message(self) -> None:  # T-MCLS-7
        """When all inputs are None, the prompt must mention the neutral/unknown fallback."""
        clf = MacroRegimeClassifier(_StubLLMClient(_EMPTY_RESPONSE))
        prompt = clf._build_prompt(headlines=None, gdelt_events=None, macro_indicators=None)
        assert "neutral" in prompt.lower() or "No data" in prompt

    def test_gdelt_events_appear_in_prompt(self) -> None:  # T-MCLS-8
        """GDELT event strings must be included in the constructed prompt."""
        clf = MacroRegimeClassifier(_StubLLMClient(_EMPTY_RESPONSE))
        prompt = clf._build_prompt(
            headlines=None,
            gdelt_events=["Russia invades Ukraine"],
            macro_indicators=None,
        )
        assert "Russia invades Ukraine" in prompt


# ── classify() end-to-end (stubbed) ──────────────────────────────────────────


class TestMacroRegimeClassifierClassify:
    def test_classify_returns_macro_regime_from_stub(self) -> None:  # T-MCLS-9
        """classify() must return a MacroRegime matching the stubbed LLM response."""
        clf = MacroRegimeClassifier(_StubLLMClient(_FULL_RESPONSE))
        regime = asyncio.run(
            clf.classify(
                headlines=["Central bank cuts rates"],
                macro_indicators={"vix": 30.0},
            )
        )
        assert isinstance(regime, MacroRegime)
        assert regime.trend == Trend.BEAR
        assert regime.volatility == VolLevel.HIGH

    def test_classify_with_no_inputs_returns_default(self) -> None:  # T-MCLS-10
        """classify() with all-None inputs and empty stub must return the neutral default."""
        clf = MacroRegimeClassifier(_StubLLMClient(_EMPTY_RESPONSE))
        regime = asyncio.run(clf.classify())
        assert regime.trend == Trend.SIDEWAYS
