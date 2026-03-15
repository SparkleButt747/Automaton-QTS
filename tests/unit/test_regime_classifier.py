"""Unit tests for qts.oversight.regime_classifier.

Tests:
- Output always contains valid scalar in [0.0, 1.0]
- regime_override.json written correctly
- Stale regime override returns None
- Fresh regime override loaded correctly
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from qts.oversight.regime_classifier import LLMRegimeAssessment, RegimeClassifierLLM


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_valid_regime_response() -> dict:  # type: ignore[type-arg]
    return {
        "regime": "geopolitical_risk_off",
        "confidence": 0.85,
        "recommended_position_scalar": 0.4,
        "reasoning": "High geopolitical tension from multiple conflict headlines.",
    }


def _make_llm_client_mock(response_dict: dict) -> MagicMock:  # type: ignore[type-arg]
    """Create a mock LLM client that returns a pre-configured JSON dict."""
    mock = MagicMock()
    mock.query_json = AsyncMock(return_value=response_dict)
    return mock


def _make_classifier(tmp_path: Path, response_dict: dict) -> RegimeClassifierLLM:  # type: ignore[type-arg]
    output_path = tmp_path / "regime_override.json"
    llm = _make_llm_client_mock(response_dict)
    return RegimeClassifierLLM(llm_client=llm, output_path=output_path)


# ── classify() tests ───────────────────────────────────────────────────────────


class TestRegimeClassifierClassify:
    @pytest.mark.asyncio
    async def test_returns_llm_regime_assessment(self, tmp_path: Path) -> None:
        """classify() should return an LLMRegimeAssessment."""
        classifier = _make_classifier(tmp_path, _make_valid_regime_response())
        result = await classifier.classify(
            headlines=["War breaks out in region X"],
            gdelt_events=["Military conflict event Y"],
        )
        assert isinstance(result, LLMRegimeAssessment)

    @pytest.mark.asyncio
    async def test_regime_field_matches_llm_output(self, tmp_path: Path) -> None:
        """regime field should match the LLM response."""
        classifier = _make_classifier(tmp_path, _make_valid_regime_response())
        result = await classifier.classify(
            headlines=["Conflict"],
            gdelt_events=[],
        )
        assert result.regime == "geopolitical_risk_off"

    @pytest.mark.asyncio
    async def test_scalar_always_in_zero_to_one_range(self, tmp_path: Path) -> None:
        """recommended_position_scalar must always be clamped to [0.0, 1.0]."""
        # Test with an extreme value from LLM
        extreme_response = {
            "regime": "normal",
            "confidence": 0.5,
            "recommended_position_scalar": 1.5,  # Out of bounds
            "reasoning": "Test",
        }
        classifier = _make_classifier(tmp_path, extreme_response)
        result = await classifier.classify(headlines=[], gdelt_events=[])
        assert 0.0 <= result.recommended_position_scalar <= 1.0

    @pytest.mark.asyncio
    async def test_scalar_negative_clamped_to_zero(self, tmp_path: Path) -> None:
        """A negative scalar from LLM should be clamped to 0.0."""
        negative_response = {
            "regime": "risk_off",
            "confidence": 0.9,
            "recommended_position_scalar": -0.5,  # Negative
            "reasoning": "Very bad market",
        }
        classifier = _make_classifier(tmp_path, negative_response)
        result = await classifier.classify(headlines=[], gdelt_events=[])
        assert result.recommended_position_scalar == 0.0

    @pytest.mark.asyncio
    async def test_confidence_clamped_to_range(self, tmp_path: Path) -> None:
        """Confidence must be clamped to [0.0, 1.0]."""
        extreme_response = {
            "regime": "normal",
            "confidence": 2.0,  # Exceeds 1.0
            "recommended_position_scalar": 0.5,
            "reasoning": "Overconfident LLM",
        }
        classifier = _make_classifier(tmp_path, extreme_response)
        result = await classifier.classify(headlines=[], gdelt_events=[])
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_reasoning_populated(self, tmp_path: Path) -> None:
        """The reasoning field should be a non-empty string."""
        classifier = _make_classifier(tmp_path, _make_valid_regime_response())
        result = await classifier.classify(headlines=["test"], gdelt_events=[])
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    @pytest.mark.asyncio
    async def test_timestamp_is_utc(self, tmp_path: Path) -> None:
        """Timestamp should be timezone-aware (UTC)."""
        classifier = _make_classifier(tmp_path, _make_valid_regime_response())
        result = await classifier.classify(headlines=[], gdelt_events=[])
        assert result.timestamp.tzinfo is not None


# ── regime_override.json persistence tests ────────────────────────────────────


class TestRegimeClassifierPersistence:
    @pytest.mark.asyncio
    async def test_writes_regime_override_json(self, tmp_path: Path) -> None:
        """classify() should write regime_override.json."""
        classifier = _make_classifier(tmp_path, _make_valid_regime_response())
        await classifier.classify(headlines=[], gdelt_events=[])

        override_path = tmp_path / "regime_override.json"
        assert override_path.exists()

    @pytest.mark.asyncio
    async def test_override_json_contains_required_fields(self, tmp_path: Path) -> None:
        """regime_override.json should contain regime, confidence, scalar, reasoning, timestamp."""
        classifier = _make_classifier(tmp_path, _make_valid_regime_response())
        await classifier.classify(headlines=[], gdelt_events=[])

        with (tmp_path / "regime_override.json").open("r") as fh:
            data = json.load(fh)

        assert "regime" in data
        assert "confidence" in data
        assert "recommended_position_scalar" in data
        assert "reasoning" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_override_json_values_match_response(self, tmp_path: Path) -> None:
        """Override file values should match the LLM response."""
        classifier = _make_classifier(tmp_path, _make_valid_regime_response())
        result = await classifier.classify(headlines=[], gdelt_events=[])

        with (tmp_path / "regime_override.json").open("r") as fh:
            data = json.load(fh)

        assert data["regime"] == result.regime
        assert data["recommended_position_scalar"] == result.recommended_position_scalar


# ── load_regime_override() tests ──────────────────────────────────────────────


class TestLoadRegimeOverride:
    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        """Should return None if regime_override.json does not exist."""
        output_path = tmp_path / "regime_override.json"
        classifier = RegimeClassifierLLM(
            llm_client=MagicMock(),
            output_path=output_path,
        )
        result = classifier.load_regime_override()
        assert result is None

    def test_returns_none_when_file_stale(self, tmp_path: Path) -> None:
        """Should return None if the override file is older than max_age_minutes."""
        output_path = tmp_path / "regime_override.json"
        # Write a stale timestamp (2 hours ago)
        stale_time = datetime.now(tz=timezone.utc) - timedelta(hours=2)
        data = {
            "regime": "normal",
            "confidence": 0.7,
            "recommended_position_scalar": 0.8,
            "reasoning": "Old data",
            "timestamp": stale_time.isoformat(),
        }
        with output_path.open("w") as fh:
            json.dump(data, fh)

        classifier = RegimeClassifierLLM(llm_client=MagicMock(), output_path=output_path)
        result = classifier.load_regime_override(max_age_minutes=30)
        assert result is None

    def test_returns_assessment_when_fresh(self, tmp_path: Path) -> None:
        """Should return the assessment if the override is fresh (< max_age_minutes)."""
        output_path = tmp_path / "regime_override.json"
        # Write a fresh timestamp (5 minutes ago)
        fresh_time = datetime.now(tz=timezone.utc) - timedelta(minutes=5)
        data = {
            "regime": "geopolitical_risk_off",
            "confidence": 0.85,
            "recommended_position_scalar": 0.4,
            "reasoning": "Recent conflict event",
            "timestamp": fresh_time.isoformat(),
        }
        with output_path.open("w") as fh:
            json.dump(data, fh)

        classifier = RegimeClassifierLLM(llm_client=MagicMock(), output_path=output_path)
        result = classifier.load_regime_override(max_age_minutes=30)

        assert result is not None
        assert isinstance(result, LLMRegimeAssessment)

    def test_fresh_override_regime_matches_file(self, tmp_path: Path) -> None:
        """Loaded assessment should have the regime from the file."""
        output_path = tmp_path / "regime_override.json"
        fresh_time = datetime.now(tz=timezone.utc) - timedelta(minutes=5)
        data = {
            "regime": "risk_on",
            "confidence": 0.6,
            "recommended_position_scalar": 0.9,
            "reasoning": "Positive sentiment",
            "timestamp": fresh_time.isoformat(),
        }
        with output_path.open("w") as fh:
            json.dump(data, fh)

        classifier = RegimeClassifierLLM(llm_client=MagicMock(), output_path=output_path)
        result = classifier.load_regime_override(max_age_minutes=30)

        assert result is not None
        assert result.regime == "risk_on"
        assert result.confidence == 0.6
        assert result.recommended_position_scalar == 0.9

    def test_fresh_override_scalar_in_valid_range(self, tmp_path: Path) -> None:
        """Loaded scalar must be in [0.0, 1.0]."""
        output_path = tmp_path / "regime_override.json"
        fresh_time = datetime.now(tz=timezone.utc) - timedelta(minutes=5)
        data = {
            "regime": "normal",
            "confidence": 0.5,
            "recommended_position_scalar": 0.75,
            "reasoning": "Normal conditions",
            "timestamp": fresh_time.isoformat(),
        }
        with output_path.open("w") as fh:
            json.dump(data, fh)

        classifier = RegimeClassifierLLM(llm_client=MagicMock(), output_path=output_path)
        result = classifier.load_regime_override(max_age_minutes=30)

        assert result is not None
        assert 0.0 <= result.recommended_position_scalar <= 1.0

    def test_returns_none_on_exactly_max_age(self, tmp_path: Path) -> None:
        """Override at exactly max_age_minutes old should be considered stale."""
        output_path = tmp_path / "regime_override.json"
        exactly_at_limit = datetime.now(tz=timezone.utc) - timedelta(minutes=30, seconds=1)
        data = {
            "regime": "normal",
            "confidence": 0.5,
            "recommended_position_scalar": 0.8,
            "reasoning": "At limit",
            "timestamp": exactly_at_limit.isoformat(),
        }
        with output_path.open("w") as fh:
            json.dump(data, fh)

        classifier = RegimeClassifierLLM(llm_client=MagicMock(), output_path=output_path)
        result = classifier.load_regime_override(max_age_minutes=30)
        assert result is None

    def test_returns_none_on_malformed_json(self, tmp_path: Path) -> None:
        """Should return None if the file contains malformed JSON."""
        output_path = tmp_path / "regime_override.json"
        output_path.write_text("not valid json", encoding="utf-8")

        classifier = RegimeClassifierLLM(llm_client=MagicMock(), output_path=output_path)
        result = classifier.load_regime_override()
        assert result is None
