"""LLM-powered regime classification — upgraded from oversight/regime_classifier.py.

Produces MacroRegime objects (not the old LLMRegimeAssessment) using
the richer regime taxonomy (Trend, VolLevel, LiquidityLevel, etc.).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from qts.models.base import (
    Catalyst,
    LiquidityLevel,
    SentimentLevel,
    Trend,
    VolLevel,
)
from qts.models.terrain import MacroRegime
from qts.oversight.llm_client import LLMClientProtocol

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a macro regime analyst for a quantitative trading system.
Analyze the provided data and classify the current market regime.

Respond with a JSON object:
{
  "trend": "<BULL | BEAR | SIDEWAYS>",
  "volatility": "<HIGH | LOW | TRANSITIONING>",
  "liquidity": "<ABUNDANT | TIGHT | CRISIS>",
  "sentiment": "<EUPHORIC | FEARFUL | NEUTRAL>",
  "catalyst": "<EARNINGS | MACRO_EVENT | GEOPOLITICAL | PANDEMIC | REGULATORY | NONE>",
  "expected_drift": <float, annualised>,
  "expected_vol": <float, annualised>,
  "correlation_regime": <float, 0-1>,
  "scenario_description": "<string: brief scenario narrative>",
  "confidence": <float, 0-1>
}

Guidelines:
- trend: Overall market direction over the analysis period
- volatility: HIGH when VIX>25 or ATR elevated; LOW when VIX<15; TRANSITIONING otherwise
- liquidity: CRISIS during stress events; TIGHT during tightening; ABUNDANT normally
- sentiment: EUPHORIC during manias; FEARFUL during panics; NEUTRAL otherwise
- catalyst: The primary driver of current conditions
- expected_drift/vol: Annualised estimates based on the data
- scenario_description: 1-2 sentence narrative of the regime
"""


class MacroRegimeClassifier:
    """Classifies macro regime using an LLM.

    Replaces the old RegimeClassifierLLM from oversight with a richer
    output format matching the MacroRegime dataclass.
    """

    def __init__(self, llm_client: LLMClientProtocol) -> None:
        self._llm = llm_client

    async def classify(
        self,
        headlines: list[str] | None = None,
        gdelt_events: list[str] | None = None,
        macro_indicators: dict[str, float] | None = None,
    ) -> MacroRegime:
        """Classify the current macro regime using LLM analysis.

        Args:
            headlines: Recent news headlines.
            gdelt_events: Recent geopolitical event descriptions.
            macro_indicators: Dict of macro indicators (vix, cpi, etc.).

        Returns:
            MacroRegime classification.
        """
        prompt = self._build_prompt(headlines, gdelt_events, macro_indicators)
        response = await self._llm.query_json(_SYSTEM_PROMPT, prompt)
        return self._parse_response(response)

    def _build_prompt(
        self,
        headlines: list[str] | None,
        gdelt_events: list[str] | None,
        macro_indicators: dict[str, float] | None,
    ) -> str:
        """Build the user prompt from available data."""
        parts = ["Classify the current market regime based on:\n"]

        if headlines:
            parts.append("## Headlines")
            parts.extend(f"- {h}" for h in headlines[:20])
            parts.append("")

        if gdelt_events:
            parts.append("## Geopolitical Events")
            parts.extend(f"- {e}" for e in gdelt_events[:10])
            parts.append("")

        if macro_indicators:
            parts.append("## Macro Indicators")
            for key, val in macro_indicators.items():
                parts.append(f"- {key}: {val}")
            parts.append("")

        if not headlines and not gdelt_events and not macro_indicators:
            parts.append("(No data provided — classify as neutral/unknown)")

        return "\n".join(parts)

    def _parse_response(self, response: dict) -> MacroRegime:  # type: ignore[type-arg]
        """Parse LLM JSON response into a MacroRegime."""
        return MacroRegime(
            trend=Trend(response.get("trend", "SIDEWAYS")),
            volatility=VolLevel(response.get("volatility", "LOW")),
            liquidity=LiquidityLevel(response.get("liquidity", "ABUNDANT")),
            sentiment=SentimentLevel(response.get("sentiment", "NEUTRAL")),
            catalyst=Catalyst(response.get("catalyst", "NONE")),
            expected_drift=float(response.get("expected_drift", 0.0)),
            expected_vol=float(response.get("expected_vol", 0.2)),
            correlation_regime=float(response.get("correlation_regime", 0.5)),
            scenario_description=str(
                response.get("scenario_description", "LLM-classified regime")
            ),
        )
