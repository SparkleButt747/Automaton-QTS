"""Oversight sub-package: human-in-the-loop controls, approval workflows, and audit trails.

Modules:
- llm_client: Async Claude API wrapper with retry logic.
- debrief: Daily trading session debrief via LLM analysis.
- regime_classifier: LLM-powered geopolitical regime classification.
- proposals: LLM-generated strategy proposal management.
- approval_cli: Rich terminal UI for human proposal review.
"""

from qts.oversight.debrief import (
    DebriefEngine,
    DebriefReport,
    Proposal,
    SessionSummary,
)
from qts.oversight.llm_client import LLMClient, LLMClientProtocol
from qts.oversight.proposals import ProposalManager
from qts.oversight.regime_classifier import LLMRegimeAssessment, RegimeClassifierLLM

__all__ = [
    "LLMClient",
    "LLMClientProtocol",
    "DebriefEngine",
    "DebriefReport",
    "Proposal",
    "SessionSummary",
    "RegimeClassifierLLM",
    "LLMRegimeAssessment",
    "ProposalManager",
]
