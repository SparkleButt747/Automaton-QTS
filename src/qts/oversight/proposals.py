"""Parse and manage LLM-generated strategy proposals."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from qts.oversight.debrief import Proposal

logger = logging.getLogger(__name__)

# ── Proposal manager ──────────────────────────────────────────────────────────

# Reasonable bounds for parameter values
_WEIGHT_MIN: float = 0.0
_WEIGHT_MAX: float = 1.0
_THRESHOLD_MIN: float = -1.0
_THRESHOLD_MAX: float = 1.0
_GENERIC_MIN: float = -1e6
_GENERIC_MAX: float = 1e6


class ProposalManager:
    """Manages the lifecycle of LLM-generated parameter change proposals.

    Proposals are stored in ``config_dir/proposals.json``.
    Historical proposals (approved/rejected) are stored in
    ``config_dir/proposals_history.json``.
    Approved proposals are patched into ``config_dir/params.json``.

    Args:
        config_dir: Directory containing params.json and proposals.json.
        risk_limits_fields: Set of field names from risk_limits.json that
            must never be touched by LLM proposals.
    """

    def __init__(
        self,
        config_dir: Path,
        risk_limits_fields: set[str],
    ) -> None:
        self._config_dir = config_dir
        self._risk_limits_fields = risk_limits_fields
        self._proposals_path = config_dir / "proposals.json"
        self._history_path = config_dir / "proposals_history.json"
        self._params_path = config_dir / "params.json"

    # ── Persistence helpers ───────────────────────────────────────────────────

    def _load_proposals_file(self) -> dict:  # type: ignore[type-arg]
        """Load the raw proposals JSON file, or return empty structure."""
        if not self._proposals_path.exists():
            return {"proposals": []}
        with self._proposals_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)  # type: ignore[no-any-return]

    def _write_proposals_file(self, data: dict) -> None:  # type: ignore[type-arg]
        """Write the raw proposals JSON file."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        with self._proposals_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def _load_history(self) -> list[dict]:  # type: ignore[type-arg]
        """Load proposals history, returning empty list if missing."""
        if not self._history_path.exists():
            return []
        with self._history_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []

    def _write_history(self, history: list[dict]) -> None:  # type: ignore[type-arg]
        """Write the proposals history file."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        with self._history_path.open("w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)

    def _load_params(self) -> dict:  # type: ignore[type-arg]
        """Load params.json."""
        if not self._params_path.exists():
            return {}
        with self._params_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)  # type: ignore[no-any-return]

    def _write_params(self, params: dict) -> None:  # type: ignore[type-arg]
        """Write params.json."""
        with self._params_path.open("w", encoding="utf-8") as fh:
            json.dump(params, fh, indent=2)

    # ── Public API ─────────────────────────────────────────────────────────────

    def save_proposals(self, proposals: list[Proposal]) -> Path:
        """Persist a list of proposals to proposals.json with a timestamp.

        Args:
            proposals: List of Proposal objects to save.

        Returns:
            Path to the written proposals.json file.
        """
        self._config_dir.mkdir(parents=True, exist_ok=True)
        data = {
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
                for p in proposals
            ],
        }
        self._write_proposals_file(data)
        logger.info(
            "ProposalManager: saved %d proposals to %s", len(proposals), self._proposals_path
        )
        return self._proposals_path

    def load_pending(self) -> list[Proposal]:
        """Load all pending proposals from proposals.json.

        Returns:
            List of pending Proposal objects.
        """
        data = self._load_proposals_file()
        pending: list[Proposal] = []
        for raw in data.get("proposals", []):
            if raw.get("status", "pending") == "pending":
                try:
                    pending.append(
                        Proposal(
                            proposal_id=str(raw["proposal_id"]),
                            parameter=str(raw["parameter"]),
                            current_value=float(raw["current_value"]),
                            proposed_value=float(raw["proposed_value"]),
                            reason=str(raw["reason"]),
                            confidence=float(raw["confidence"]),
                        )
                    )
                except (KeyError, ValueError) as exc:
                    logger.warning("ProposalManager: skipping malformed proposal: %s", exc)
        return pending

    def validate_proposal(self, proposal: Proposal) -> bool:
        """Validate a proposal, rejecting any that touch risk_limits.json fields.

        Args:
            proposal: The proposal to validate.

        Returns:
            True if the proposal is valid and safe to apply.
        """
        # Hard reject: never allow modifying risk limits fields
        if proposal.parameter in self._risk_limits_fields:
            logger.warning(
                "ProposalManager: REJECTED proposal for protected field '%s'",
                proposal.parameter,
            )
            return False

        # Validate proposed_value is within reasonable bounds
        param = proposal.parameter
        v = proposal.proposed_value

        # Weight parameters must be in [0, 1]
        if param.startswith("w_") or "weight" in param.lower():
            if not (_WEIGHT_MIN <= v <= _WEIGHT_MAX):
                logger.warning(
                    "ProposalManager: REJECTED proposal '%s'=%.4f out of weight bounds [0,1]",
                    param,
                    v,
                )
                return False

        # Threshold parameters must be in [-1, 1]
        if "threshold" in param.lower():
            if not (_THRESHOLD_MIN <= v <= _THRESHOLD_MAX):
                logger.warning(
                    "ProposalManager: REJECTED proposal '%s'=%.4f out of threshold bounds [-1,1]",
                    param,
                    v,
                )
                return False

        # Generic sanity check
        if not (_GENERIC_MIN <= v <= _GENERIC_MAX):
            logger.warning(
                "ProposalManager: REJECTED proposal '%s'=%.4f out of generic bounds",
                param,
                v,
            )
            return False

        return True

    def approve(self, proposal_id: str, approver: str) -> None:
        """Approve a proposal: patch params.json and move it to history.

        Args:
            proposal_id: ID of the proposal to approve.
            approver: Identifier of the approver (e.g., username).

        Raises:
            ValueError: If the proposal is not found or fails validation.
        """
        data = self._load_proposals_file()
        proposals_list = data.get("proposals", [])

        target_idx: int | None = None
        target_raw: dict | None = None  # type: ignore[type-arg]
        for idx, raw in enumerate(proposals_list):
            if raw.get("proposal_id") == proposal_id:
                target_idx = idx
                target_raw = raw
                break

        if target_raw is None or target_idx is None:
            raise ValueError(f"Proposal '{proposal_id}' not found")

        proposal = Proposal(
            proposal_id=str(target_raw["proposal_id"]),
            parameter=str(target_raw["parameter"]),
            current_value=float(target_raw["current_value"]),
            proposed_value=float(target_raw["proposed_value"]),
            reason=str(target_raw["reason"]),
            confidence=float(target_raw["confidence"]),
        )

        if not self.validate_proposal(proposal):
            raise ValueError(
                f"Proposal '{proposal_id}' failed validation (parameter='{proposal.parameter}')"
            )

        # Patch params.json with the new value
        params = self._load_params()
        self._apply_param_patch(params, proposal.parameter, proposal.proposed_value)
        self._write_params(params)
        logger.info(
            "ProposalManager: approved %s, patched %s=%.4f in params.json",
            proposal_id,
            proposal.parameter,
            proposal.proposed_value,
        )

        # Move to history
        history_entry = dict(target_raw)
        history_entry.update(
            {
                "status": "approved",
                "approver": approver,
                "actioned_at": datetime.now(tz=UTC).isoformat(),
            }
        )
        history = self._load_history()
        history.append(history_entry)
        self._write_history(history)

        # Remove from pending
        proposals_list.pop(target_idx)
        data["proposals"] = proposals_list
        self._write_proposals_file(data)

    def reject(self, proposal_id: str, rejector: str, reason: str) -> None:
        """Reject a proposal and record the reason in history.

        Args:
            proposal_id: ID of the proposal to reject.
            rejector: Identifier of the rejector.
            reason: Human-readable rejection reason.

        Raises:
            ValueError: If the proposal is not found.
        """
        data = self._load_proposals_file()
        proposals_list = data.get("proposals", [])

        target_idx: int | None = None
        target_raw: dict | None = None  # type: ignore[type-arg]
        for idx, raw in enumerate(proposals_list):
            if raw.get("proposal_id") == proposal_id:
                target_idx = idx
                target_raw = raw
                break

        if target_raw is None or target_idx is None:
            raise ValueError(f"Proposal '{proposal_id}' not found")

        history_entry = dict(target_raw)
        history_entry.update(
            {
                "status": "rejected",
                "rejector": rejector,
                "rejection_reason": reason,
                "actioned_at": datetime.now(tz=UTC).isoformat(),
            }
        )
        history = self._load_history()
        history.append(history_entry)
        self._write_history(history)

        # Remove from pending
        proposals_list.pop(target_idx)
        data["proposals"] = proposals_list
        self._write_proposals_file(data)
        logger.info("ProposalManager: rejected %s (reason: %s)", proposal_id, reason)

    def get_history(self) -> list[dict]:  # type: ignore[type-arg]
        """Return the full proposals history list.

        Returns:
            List of proposal history dicts.
        """
        return self._load_history()

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _apply_param_patch(
        params: dict,  # type: ignore[type-arg]
        parameter: str,
        value: float,
    ) -> None:
        """Apply a parameter patch to the params dict in-place.

        Handles nested parameters like "weights.w_sentiment" and flat params.

        Args:
            params: The params dict to modify in-place.
            parameter: The parameter key (may use dot notation for nesting).
            value: The new value to set.
        """
        parts = parameter.split(".")
        if len(parts) == 1:
            # Check if it's a weight parameter nested under "weights"
            if parameter.startswith("w_"):
                weights = params.setdefault("weights", {})
                weights[parameter] = value
            else:
                params[parameter] = value
        else:
            # Navigate nested structure
            current = params
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = value
