"""LLM-powered geopolitical regime classification."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from qts.oversight.llm_client import LLMClientProtocol

logger = logging.getLogger(__name__)

# ── Domain types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LLMRegimeAssessment:
    """Result of an LLM-based geopolitical regime classification.

    Attributes:
        regime: Regime label (e.g., "geopolitical_risk_off", "normal", "risk_on").
        confidence: Confidence in the classification (0-1).
        recommended_position_scalar: Recommended position size scalar (0-1).
        reasoning: Human-readable explanation of the classification.
        timestamp: UTC timestamp when the assessment was produced.
    """

    regime: str
    confidence: float
    recommended_position_scalar: float
    reasoning: str
    timestamp: datetime


# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a geopolitical risk analyst for a quantitative trading system.
Analyze the provided news headlines and GDELT events to classify the current market regime.

You MUST respond with a JSON object with the following structure:
{
  "regime": "<one of: geopolitical_risk_off | risk_off | normal | risk_on | geopolitical_risk_on>",
  "confidence": <float between 0.0 and 1.0>,
  "recommended_position_scalar": <float between 0.0 and 1.0>,
  "reasoning": "<string: explanation of the classification>"
}

Guidelines:
- regime: Use "geopolitical_risk_off" for high geopolitical tension,
  "normal" for typical conditions, "risk_on" for high growth/optimism.
- confidence: How certain you are about the regime (0=not confident, 1=very confident).
- recommended_position_scalar: 0.0 means do not trade, 1.0 means full position sizing.
  Use lower values during high uncertainty or risk-off conditions.
- reasoning: Brief explanation citing specific headlines or events.
"""


# ── Classifier ─────────────────────────────────────────────────────────────────


class RegimeClassifierLLM:
    """Classifies the current geopolitical/market regime using an LLM.

    Args:
        llm_client: An LLMClientProtocol-compatible client.
        output_path: Path where regime_override.json will be written.
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        output_path: Path,
    ) -> None:
        self._llm = llm_client
        self._output_path = output_path

    async def classify(
        self,
        headlines: list[str],
        gdelt_events: list[str],
    ) -> LLMRegimeAssessment:
        """Classify the current regime from headlines and GDELT events.

        Sends data to Claude, parses the structured response, and writes
        the result to regime_override.json.

        Args:
            headlines: Recent news headlines.
            gdelt_events: Recent GDELT geopolitical event descriptions.

        Returns:
            An LLMRegimeAssessment with regime, confidence, scalar, and reasoning.
        """
        user_prompt = self._build_prompt(headlines, gdelt_events)
        response = await self._llm.query_json(_SYSTEM_PROMPT, user_prompt)
        assessment = self._parse_response(response)
        self._write_override(assessment)
        return assessment

    def _build_prompt(self, headlines: list[str], gdelt_events: list[str]) -> str:
        """Build the user prompt from headlines and events.

        Args:
            headlines: Recent news headlines.
            gdelt_events: GDELT event descriptions.

        Returns:
            Formatted user prompt string.
        """
        headlines_text = "\n".join(f"- {h}" for h in headlines) if headlines else "(none)"
        gdelt_text = "\n".join(f"- {e}" for e in gdelt_events) if gdelt_events else "(none)"

        return (
            "Please classify the current market regime based on the following information:\n\n"
            f"## Recent News Headlines\n{headlines_text}\n\n"
            f"## GDELT Geopolitical Events\n{gdelt_text}"
        )

    def _parse_response(self, response: dict) -> LLMRegimeAssessment:  # type: ignore[type-arg]
        """Parse and validate the LLM response dict into an LLMRegimeAssessment.

        Args:
            response: Parsed JSON dict from the LLM.

        Returns:
            An LLMRegimeAssessment with validated values.
        """
        regime = str(response.get("regime", "normal"))
        confidence = float(response.get("confidence", 0.5))
        scalar = float(response.get("recommended_position_scalar", 0.5))
        reasoning = str(response.get("reasoning", "No reasoning provided."))

        # Clamp confidence and scalar to [0, 1]
        confidence = max(0.0, min(1.0, confidence))
        scalar = max(0.0, min(1.0, scalar))

        return LLMRegimeAssessment(
            regime=regime,
            confidence=confidence,
            recommended_position_scalar=scalar,
            reasoning=reasoning,
            timestamp=datetime.now(tz=UTC),
        )

    def _write_override(self, assessment: LLMRegimeAssessment) -> None:
        """Write the regime assessment to regime_override.json.

        Args:
            assessment: The assessment to persist.
        """
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "regime": assessment.regime,
            "confidence": assessment.confidence,
            "recommended_position_scalar": assessment.recommended_position_scalar,
            "reasoning": assessment.reasoning,
            "timestamp": assessment.timestamp.isoformat(),
        }
        with self._output_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info(
            "Regime: wrote override to %s (regime=%s)", self._output_path, assessment.regime
        )

    def load_regime_override(self, max_age_minutes: int = 30) -> LLMRegimeAssessment | None:
        """Load a cached regime override if it exists and is fresh.

        Args:
            max_age_minutes: Maximum age in minutes for the cached override.

        Returns:
            The cached LLMRegimeAssessment if fresh, or None if stale/missing.
        """
        if not self._output_path.exists():
            return None

        try:
            with self._output_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Regime: failed to load override from %s: %s", self._output_path, exc)
            return None

        try:
            timestamp = datetime.fromisoformat(data["timestamp"])
        except (KeyError, ValueError) as exc:
            logger.warning("Regime: invalid timestamp in override file: %s", exc)
            return None

        # Ensure timestamp is timezone-aware
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        now = datetime.now(tz=UTC)
        age = now - timestamp
        if age > timedelta(minutes=max_age_minutes):
            logger.debug(
                "Regime: override is stale (age=%.1f min > max=%d min)",
                age.total_seconds() / 60,
                max_age_minutes,
            )
            return None

        try:
            return LLMRegimeAssessment(
                regime=str(data["regime"]),
                confidence=float(data["confidence"]),
                recommended_position_scalar=float(data["recommended_position_scalar"]),
                reasoning=str(data["reasoning"]),
                timestamp=timestamp,
            )
        except (KeyError, ValueError) as exc:
            logger.warning("Regime: malformed override data: %s", exc)
            return None
