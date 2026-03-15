"""Unit tests for qts.oversight.debrief.

Tests:
- Builds correct prompt structure from SessionSummary
- Parses valid LLM response into DebriefReport
- Handles malformed LLM response gracefully
- Proposals saved to file
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from qts.models.base import (
    ExitReason,
    FailureMode,
    TradeDirection,
    TradeOutcome,
    TradeRecord,
    VolRegime,
)
from qts.oversight.debrief import (
    DebriefEngine,
    DebriefReport,
    Proposal,
    SessionSummary,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_trade_record() -> TradeRecord:
    return TradeRecord(
        trade_id=str(uuid.uuid4()),
        symbol="BTCUSDT",
        entry_time=datetime(2024, 6, 1, 9, 0, tzinfo=timezone.utc),
        exit_time=datetime(2024, 6, 1, 10, 0, tzinfo=timezone.utc),
        direction=TradeDirection.LONG,
        entry_price=65000.0,
        exit_price=64000.0,
        size=0.1,
        pnl_usd=-100.0,
        pnl_pct=-0.015,
        slippage_usd=5.0,
        fees_usd=2.5,
        rsi_at_entry=55.0,
        macd_at_entry=0.05,
        sentiment_at_entry=0.3,
        social_velocity_at_entry=0.1,
        gdelt_intensity_at_entry=0.2,
        vol_regime_at_entry=VolRegime.HIGH,
        combined_alpha_at_entry=0.6,
        params_version="1.0.0",
        outcome=TradeOutcome.LOSS,
        exit_reason=ExitReason.STOP_LOSS,
        failure_mode=FailureMode.SENTIMENT_FADE,
    )


def _make_session_summary() -> SessionSummary:
    return SessionSummary(
        total_trades=20,
        win_rate=0.55,
        total_pnl=250.0,
        sharpe_today=1.2,
        failure_mode_summary={"SENTIMENT_FADE": {"count": 3, "avg_pnl": -80.0}},
        sentiment_hit_rate=0.62,
        sentiment_pnl_correlation=0.35,
        top_loss_trades=[_make_trade_record()],
        current_params={
            "weights": {
                "w_rsi": 0.20,
                "w_macd": 0.20,
                "w_bb": 0.15,
                "w_mom": 0.15,
                "w_sentiment": 0.30,
            },
            "entry_threshold": 0.25,
            "exit_threshold": -0.10,
        },
        vol_regime="HIGH",
        gdelt_top_events=["Tensions in region X", "Trade deal Y announced"],
    )


def _make_valid_llm_response() -> str:
    return json.dumps(
        {
            "analysis": "The session showed moderate performance with SENTIMENT_FADE as the primary failure mode.",
            "proposals": [
                {
                    "parameter": "w_sentiment",
                    "current_value": 0.30,
                    "proposed_value": 0.25,
                    "reason": "Sentiment fades account for 3 trades; reduce weight.",
                    "confidence": 0.72,
                }
            ],
            "regime_alert": None,
        }
    )


def _make_llm_client_mock(response_text: str) -> MagicMock:
    mock = MagicMock()
    mock.query = AsyncMock(return_value=response_text)
    return mock


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestDebriefEnginePromptBuilding:
    def test_build_prompt_returns_tuple(self, tmp_path: Path) -> None:
        """_build_prompt should return a (system, user) tuple."""
        engine = DebriefEngine(
            llm_client=MagicMock(),
            config_dir=tmp_path,
        )
        summary = _make_session_summary()
        result = engine._build_prompt(summary)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_system_prompt_contains_json_instruction(self, tmp_path: Path) -> None:
        """System prompt should instruct the LLM to respond in JSON."""
        engine = DebriefEngine(llm_client=MagicMock(), config_dir=tmp_path)
        system, _ = engine._build_prompt(_make_session_summary())
        assert "JSON" in system

    def test_user_prompt_contains_session_metrics(self, tmp_path: Path) -> None:
        """User prompt should include the session metrics."""
        engine = DebriefEngine(llm_client=MagicMock(), config_dir=tmp_path)
        _, user = engine._build_prompt(_make_session_summary())
        assert "total_trades" in user
        assert "win_rate" in user
        assert "sharpe_today" in user

    def test_user_prompt_contains_failure_modes(self, tmp_path: Path) -> None:
        """User prompt should include failure mode summary."""
        engine = DebriefEngine(llm_client=MagicMock(), config_dir=tmp_path)
        summary = _make_session_summary()
        _, user = engine._build_prompt(summary)
        assert "SENTIMENT_FADE" in user

    def test_user_prompt_contains_current_params(self, tmp_path: Path) -> None:
        """User prompt should include current strategy parameters."""
        engine = DebriefEngine(llm_client=MagicMock(), config_dir=tmp_path)
        _, user = engine._build_prompt(_make_session_summary())
        assert "w_sentiment" in user

    def test_user_prompt_contains_gdelt_events(self, tmp_path: Path) -> None:
        """User prompt should include GDELT top events."""
        engine = DebriefEngine(llm_client=MagicMock(), config_dir=tmp_path)
        _, user = engine._build_prompt(_make_session_summary())
        assert "Tensions in region X" in user

    def test_user_prompt_contains_top_loss_trades(self, tmp_path: Path) -> None:
        """User prompt should include top loss trade details."""
        engine = DebriefEngine(llm_client=MagicMock(), config_dir=tmp_path)
        _, user = engine._build_prompt(_make_session_summary())
        assert "top_loss_trades" in user


class TestDebriefEngineParseResponse:
    def test_parses_valid_response_into_debrief_report(self, tmp_path: Path) -> None:
        """A valid JSON response should produce a DebriefReport."""
        engine = DebriefEngine(llm_client=MagicMock(), config_dir=tmp_path)
        report = engine._parse_response(_make_valid_llm_response())
        assert isinstance(report, DebriefReport)

    def test_analysis_field_populated(self, tmp_path: Path) -> None:
        """The analysis field should be a non-empty string."""
        engine = DebriefEngine(llm_client=MagicMock(), config_dir=tmp_path)
        report = engine._parse_response(_make_valid_llm_response())
        assert isinstance(report.analysis, str)
        assert len(report.analysis) > 0

    def test_proposals_parsed(self, tmp_path: Path) -> None:
        """Proposals should be parsed from the JSON response."""
        engine = DebriefEngine(llm_client=MagicMock(), config_dir=tmp_path)
        report = engine._parse_response(_make_valid_llm_response())
        assert len(report.proposals) == 1
        p = report.proposals[0]
        assert p.parameter == "w_sentiment"
        assert p.current_value == 0.30
        assert p.proposed_value == 0.25
        assert isinstance(p.confidence, float)

    def test_regime_alert_none_when_null(self, tmp_path: Path) -> None:
        """regime_alert should be None when the LLM returns null."""
        engine = DebriefEngine(llm_client=MagicMock(), config_dir=tmp_path)
        report = engine._parse_response(_make_valid_llm_response())
        assert report.regime_alert is None

    def test_regime_alert_populated_when_present(self, tmp_path: Path) -> None:
        """regime_alert should be a string when the LLM returns one."""
        engine = DebriefEngine(llm_client=MagicMock(), config_dir=tmp_path)
        response = json.dumps(
            {
                "analysis": "Risk-off regime detected.",
                "proposals": [],
                "regime_alert": "GEOPOLITICAL_RISK_OFF: elevated tension in X",
            }
        )
        report = engine._parse_response(response)
        assert report.regime_alert == "GEOPOLITICAL_RISK_OFF: elevated tension in X"

    def test_raw_response_stored(self, tmp_path: Path) -> None:
        """The raw_response field should store the original LLM text."""
        engine = DebriefEngine(llm_client=MagicMock(), config_dir=tmp_path)
        raw = _make_valid_llm_response()
        report = engine._parse_response(raw)
        assert report.raw_response == raw

    def test_handles_malformed_json_gracefully(self, tmp_path: Path) -> None:
        """A non-JSON response should return a fallback DebriefReport, not raise."""
        engine = DebriefEngine(llm_client=MagicMock(), config_dir=tmp_path)
        report = engine._parse_response("This is not JSON at all!")
        assert isinstance(report, DebriefReport)
        assert report.proposals == []
        assert "malformed" in report.analysis.lower() or "manual review" in report.analysis.lower()

    def test_handles_partial_json_gracefully(self, tmp_path: Path) -> None:
        """A response with missing required keys should use defaults."""
        engine = DebriefEngine(llm_client=MagicMock(), config_dir=tmp_path)
        partial = json.dumps({"analysis": "Partial response"})
        report = engine._parse_response(partial)
        assert report.analysis == "Partial response"
        assert report.proposals == []
        assert report.regime_alert is None

    def test_handles_malformed_proposals_gracefully(self, tmp_path: Path) -> None:
        """Malformed proposal entries should be skipped without crashing."""
        engine = DebriefEngine(llm_client=MagicMock(), config_dir=tmp_path)
        response = json.dumps(
            {
                "analysis": "Some analysis",
                "proposals": [
                    {"bad": "data"},  # Missing required fields
                    {
                        "parameter": "w_rsi",
                        "current_value": 0.2,
                        "proposed_value": 0.15,
                        "reason": "Valid proposal",
                        "confidence": 0.8,
                    },
                ],
                "regime_alert": None,
            }
        )
        report = engine._parse_response(response)
        # Should parse the valid one and skip the bad one
        assert len(report.proposals) == 1
        assert report.proposals[0].parameter == "w_rsi"

    def test_parses_markdown_fenced_json(self, tmp_path: Path) -> None:
        """Should handle markdown-fenced JSON responses."""
        engine = DebriefEngine(llm_client=MagicMock(), config_dir=tmp_path)
        payload = {
            "analysis": "Good session overall.",
            "proposals": [],
            "regime_alert": None,
        }
        fenced = "```json\n" + json.dumps(payload) + "\n```"
        report = engine._parse_response(fenced)
        assert report.analysis == "Good session overall."


class TestDebriefEngineRunDebrief:
    @pytest.mark.asyncio
    async def test_run_debrief_returns_report(self, tmp_path: Path) -> None:
        """run_debrief should return a DebriefReport."""
        llm = _make_llm_client_mock(_make_valid_llm_response())
        engine = DebriefEngine(llm_client=llm, config_dir=tmp_path)
        summary = _make_session_summary()
        report = await engine.run_debrief(summary)
        assert isinstance(report, DebriefReport)

    @pytest.mark.asyncio
    async def test_proposals_written_to_file(self, tmp_path: Path) -> None:
        """run_debrief should write proposals.json to config_dir."""
        llm = _make_llm_client_mock(_make_valid_llm_response())
        engine = DebriefEngine(llm_client=llm, config_dir=tmp_path)
        summary = _make_session_summary()
        await engine.run_debrief(summary)

        proposals_path = tmp_path / "proposals.json"
        assert proposals_path.exists()

        with proposals_path.open("r") as fh:
            data = json.load(fh)

        assert "proposals" in data
        assert len(data["proposals"]) == 1
        assert data["proposals"][0]["parameter"] == "w_sentiment"

    @pytest.mark.asyncio
    async def test_proposals_file_status_is_pending(self, tmp_path: Path) -> None:
        """Written proposals should have status='pending'."""
        llm = _make_llm_client_mock(_make_valid_llm_response())
        engine = DebriefEngine(llm_client=llm, config_dir=tmp_path)
        await engine.run_debrief(_make_session_summary())

        with (tmp_path / "proposals.json").open("r") as fh:
            data = json.load(fh)

        for p in data["proposals"]:
            assert p["status"] == "pending"

    @pytest.mark.asyncio
    async def test_run_debrief_with_empty_proposals(self, tmp_path: Path) -> None:
        """Should handle a response with no proposals gracefully."""
        response = json.dumps({
            "analysis": "Good session, no changes needed.",
            "proposals": [],
            "regime_alert": None,
        })
        llm = _make_llm_client_mock(response)
        engine = DebriefEngine(llm_client=llm, config_dir=tmp_path)
        report = await engine.run_debrief(_make_session_summary())
        assert report.proposals == []
