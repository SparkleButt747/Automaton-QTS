"""Daily trading session debrief via LLM analysis."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qts.models.base import TradeRecord
from qts.oversight.llm_client import LLMClientProtocol

logger = logging.getLogger(__name__)

# ── Domain types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Proposal:
    """A single parameter change proposal from the LLM.

    Attributes:
        proposal_id: Unique identifier (UUID string).
        parameter: The parameter name to change (e.g., "w_sentiment").
        current_value: The current parameter value.
        proposed_value: The proposed new parameter value.
        reason: Human-readable justification for the proposal.
        confidence: LLM confidence in the proposal (0-1).
    """

    proposal_id: str
    parameter: str
    current_value: float
    proposed_value: float
    reason: str
    confidence: float

    @classmethod
    def new(
        cls,
        parameter: str,
        current_value: float,
        proposed_value: float,
        reason: str,
        confidence: float,
    ) -> Proposal:
        """Create a new Proposal with an auto-generated UUID."""
        return cls(
            proposal_id=str(uuid.uuid4()),
            parameter=parameter,
            current_value=current_value,
            proposed_value=proposed_value,
            reason=reason,
            confidence=confidence,
        )


@dataclass(frozen=True)
class DebriefReport:
    """Result of a daily debrief analysis.

    Attributes:
        session_date: ISO date string for the session.
        analysis: Free-form LLM analysis narrative.
        proposals: List of parameter change proposals.
        regime_alert: Optional alert string if a regime change is detected.
        raw_response: Raw LLM response text.
    """

    session_date: str
    analysis: str
    proposals: list[Proposal]
    regime_alert: str | None
    raw_response: str


@dataclass
class SessionSummary:
    """Summary data for a trading session, passed to the debrief engine.

    Attributes:
        total_trades: Number of trades executed.
        win_rate: Fraction of trades that were wins (0-1).
        total_pnl: Total P&L for the session in USD.
        sharpe_today: Intraday Sharpe ratio.
        failure_mode_summary: Counts / stats per failure mode.
        sentiment_hit_rate: Fraction of trades where sentiment matched direction.
        sentiment_pnl_correlation: Pearson correlation between sentiment and P&L.
        top_loss_trades: Up to 5 worst-loss TradeRecord objects.
        current_params: Current strategy parameter dict.
        vol_regime: Current volatility regime string (e.g., "HIGH", "LOW").
        gdelt_top_events: Top GDELT geopolitical event descriptions.
    """

    total_trades: int
    win_rate: float
    total_pnl: float
    sharpe_today: float
    failure_mode_summary: dict[str, Any]
    sentiment_hit_rate: float
    sentiment_pnl_correlation: float
    top_loss_trades: list[TradeRecord]
    current_params: dict[str, Any]
    vol_regime: str
    gdelt_top_events: list[str]


# ── Debrief engine ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a quantitative trading system oversight AI.
Your role is to analyze daily trading session performance and suggest \
evidence-based parameter adjustments.

You MUST respond with a JSON object with the following structure:
{
  "analysis": "<string: narrative analysis of the session>",
  "proposals": [
    {
      "parameter": "<string: parameter name>",
      "current_value": <float>,
      "proposed_value": <float>,
      "reason": "<string: justification>",
      "confidence": <float between 0.0 and 1.0>
    }
  ],
  "regime_alert": "<string or null: alert message if regime change detected>"
}

Rules:
- Only propose changes to STRATEGY parameters (weights, thresholds), NEVER to risk_limits fields.
- All proposed parameter values must be within reasonable bounds.
- Confidence values must be between 0.0 and 1.0.
- The analysis field must be a non-empty string.
- The proposals list may be empty if no changes are warranted.
- Set regime_alert to null if no regime change is detected.
"""


class DebriefEngine:
    """Runs the daily debrief analysis using an LLM.

    Args:
        llm_client: An LLMClientProtocol-compatible client.
        config_dir: Path to the directory where proposals.json will be written.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        config_dir: Path,
    ) -> None:
        self._llm = llm_client
        self._config_dir = config_dir

    async def run_debrief(self, session_summary: SessionSummary) -> DebriefReport:
        """Run the daily debrief analysis.

        Constructs a structured prompt, sends it to the LLM, parses the
        response, and saves proposals to disk.

        Args:
            session_summary: Session performance data.

        Returns:
            A DebriefReport with analysis, proposals, and regime alert.
        """
        system_prompt, user_prompt = self._build_prompt(session_summary)
        raw_response = await self._llm.query(system_prompt, user_prompt)
        report = self._parse_response(raw_response)
        self._save_proposals(report)
        return report

    def _build_prompt(self, summary: SessionSummary) -> tuple[str, str]:
        """Construct the system and user prompts from the session summary.

        Args:
            summary: Session performance data.

        Returns:
            A (system_prompt, user_prompt) tuple.
        """
        today = datetime.now(tz=UTC).date().isoformat()

        top_loss_info = []
        for t in summary.top_loss_trades:
            top_loss_info.append(
                {
                    "trade_id": t.trade_id,
                    "symbol": t.symbol,
                    "pnl_usd": t.pnl_usd,
                    "failure_mode": t.failure_mode.value if t.failure_mode else None,
                    "exit_reason": t.exit_reason.value,
                    "sentiment_at_entry": t.sentiment_at_entry,
                    "vol_regime": t.vol_regime_at_entry.value,
                }
            )

        user_data = {
            "session_date": today,
            "performance": {
                "total_trades": summary.total_trades,
                "win_rate": round(summary.win_rate, 4),
                "total_pnl_usd": round(summary.total_pnl, 2),
                "sharpe_today": round(summary.sharpe_today, 4),
            },
            "failure_mode_summary": summary.failure_mode_summary,
            "sentiment_analysis": {
                "hit_rate": round(summary.sentiment_hit_rate, 4),
                "pnl_correlation": round(summary.sentiment_pnl_correlation, 4),
            },
            "top_loss_trades": top_loss_info,
            "current_params": summary.current_params,
            "vol_regime": summary.vol_regime,
            "gdelt_top_events": summary.gdelt_top_events,
        }

        json_str = json.dumps(user_data, indent=2)
        user_prompt = (
            "Please analyze this trading session and provide your assessment:\n\n"
            "```json\n" + json_str + "\n```"
        )

        return _SYSTEM_PROMPT, user_prompt

    def _parse_response(self, response: str) -> DebriefReport:
        """Parse the raw LLM response into a DebriefReport.

        Falls back gracefully if the response is malformed.

        Args:
            response: Raw text from the LLM.

        Returns:
            A DebriefReport instance.
        """
        today = datetime.now(tz=UTC).date().isoformat()

        # Attempt to strip markdown fences
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            inner = lines[1:]
            if inner and inner[-1].strip().startswith("```"):
                inner = inner[:-1]
            text = "\n".join(inner).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Debrief: LLM returned non-JSON response; returning fallback report")
            return DebriefReport(
                session_date=today,
                analysis="LLM returned a malformed response. Manual review required.",
                proposals=[],
                regime_alert=None,
                raw_response=response,
            )

        # Parse proposals
        proposals: list[Proposal] = []
        for raw_prop in data.get("proposals", []):
            try:
                proposals.append(
                    Proposal(
                        proposal_id=raw_prop.get("proposal_id", str(uuid.uuid4())),
                        parameter=str(raw_prop["parameter"]),
                        current_value=float(raw_prop["current_value"]),
                        proposed_value=float(raw_prop["proposed_value"]),
                        reason=str(raw_prop["reason"]),
                        confidence=float(raw_prop["confidence"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Debrief: skipping malformed proposal %s: %s", raw_prop, exc)

        analysis = str(data.get("analysis", "No analysis provided."))
        regime_alert_raw = data.get("regime_alert")
        regime_alert: str | None = str(regime_alert_raw) if regime_alert_raw else None

        return DebriefReport(
            session_date=today,
            analysis=analysis,
            proposals=proposals,
            regime_alert=regime_alert,
            raw_response=response,
        )

    def _save_proposals(self, report: DebriefReport) -> None:
        """Persist the debrief proposals to proposals.json.

        Args:
            report: The debrief report containing proposals to save.
        """
        self._config_dir.mkdir(parents=True, exist_ok=True)
        proposals_path = self._config_dir / "proposals.json"

        proposals_data = {
            "session_date": report.session_date,
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "proposals": [
                {
                    "proposal_id": p.proposal_id,
                    "parameter": p.parameter,
                    "current_value": p.current_value,
                    "proposed_value": p.proposed_value,
                    "reason": p.reason,
                    "confidence": p.confidence,
                    "status": "pending",
                }
                for p in report.proposals
            ],
        }

        with proposals_path.open("w", encoding="utf-8") as fh:
            json.dump(proposals_data, fh, indent=2)

        logger.info(
            "Debrief: wrote %d proposals to %s",
            len(report.proposals),
            proposals_path,
        )
