"""Unit tests for qts.oversight.proposals.

Tests:
- Rejects any proposal touching risk_limits.json fields
- Approve patches params.json correctly
- Reject records rejection reason
- History is retained
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from qts.oversight.debrief import Proposal
from qts.oversight.proposals import ProposalManager

# ── Fixtures / Helpers ─────────────────────────────────────────────────────────

_RISK_LIMITS_FIELDS = {
    "max_daily_drawdown_pct",
    "max_position_size_pct",
    "max_open_positions",
    "circuit_breaker_cooldown_seconds",
    "sentiment_signal_max_scalar",
}

_DEFAULT_PARAMS = {
    "version": "1.0.0",
    "weights": {
        "w_rsi": 0.20,
        "w_macd": 0.20,
        "w_bb": 0.15,
        "w_mom": 0.15,
        "w_sentiment": 0.30,
    },
    "entry_threshold": 0.25,
    "exit_threshold": -0.10,
    "max_hold_bars": 48,
}


def _make_manager(tmp_path: Path) -> ProposalManager:
    """Create a ProposalManager with a prepopulated params.json."""
    params_path = tmp_path / "params.json"
    with params_path.open("w") as fh:
        json.dump(_DEFAULT_PARAMS, fh, indent=2)
    return ProposalManager(
        config_dir=tmp_path,
        risk_limits_fields=_RISK_LIMITS_FIELDS,
    )


def _make_proposal(
    parameter: str = "w_sentiment",
    current_value: float = 0.30,
    proposed_value: float = 0.25,
    confidence: float = 0.75,
) -> Proposal:
    return Proposal(
        proposal_id=str(uuid.uuid4()),
        parameter=parameter,
        current_value=current_value,
        proposed_value=proposed_value,
        reason="Test reason",
        confidence=confidence,
    )


# ── validate_proposal() tests ──────────────────────────────────────────────────


class TestValidateProposal:
    def test_rejects_risk_limits_fields(self, tmp_path: Path) -> None:
        """Any proposal touching risk_limits.json fields must be rejected."""
        manager = _make_manager(tmp_path)
        for field in _RISK_LIMITS_FIELDS:
            proposal = _make_proposal(parameter=field, current_value=0.02, proposed_value=0.03)
            assert manager.validate_proposal(proposal) is False, (
                f"Expected rejection for risk limits field '{field}'"
            )

    def test_rejects_max_daily_drawdown_pct(self, tmp_path: Path) -> None:
        """Specifically verify max_daily_drawdown_pct is rejected."""
        manager = _make_manager(tmp_path)
        proposal = _make_proposal(
            parameter="max_daily_drawdown_pct",
            current_value=0.02,
            proposed_value=0.05,
        )
        assert manager.validate_proposal(proposal) is False

    def test_accepts_valid_weight_param(self, tmp_path: Path) -> None:
        """A valid strategy weight proposal should be accepted."""
        manager = _make_manager(tmp_path)
        proposal = _make_proposal(parameter="w_sentiment", current_value=0.30, proposed_value=0.25)
        assert manager.validate_proposal(proposal) is True

    def test_rejects_weight_param_above_one(self, tmp_path: Path) -> None:
        """Weight parameters > 1.0 should be rejected."""
        manager = _make_manager(tmp_path)
        proposal = _make_proposal(parameter="w_rsi", current_value=0.20, proposed_value=1.5)
        assert manager.validate_proposal(proposal) is False

    def test_rejects_weight_param_below_zero(self, tmp_path: Path) -> None:
        """Weight parameters < 0.0 should be rejected."""
        manager = _make_manager(tmp_path)
        proposal = _make_proposal(parameter="w_macd", current_value=0.20, proposed_value=-0.1)
        assert manager.validate_proposal(proposal) is False

    def test_accepts_valid_entry_threshold(self, tmp_path: Path) -> None:
        """A valid entry_threshold proposal should be accepted."""
        manager = _make_manager(tmp_path)
        proposal = _make_proposal(
            parameter="entry_threshold",
            current_value=0.25,
            proposed_value=0.30,
        )
        assert manager.validate_proposal(proposal) is True

    def test_rejects_threshold_out_of_range(self, tmp_path: Path) -> None:
        """A threshold proposal outside [-1, 1] should be rejected."""
        manager = _make_manager(tmp_path)
        proposal = _make_proposal(
            parameter="entry_threshold",
            current_value=0.25,
            proposed_value=1.5,  # Out of [-1, 1]
        )
        assert manager.validate_proposal(proposal) is False


# ── save_proposals() / load_pending() tests ────────────────────────────────────


class TestSaveAndLoadProposals:
    def test_save_proposals_creates_file(self, tmp_path: Path) -> None:
        """save_proposals should create proposals.json."""
        manager = _make_manager(tmp_path)
        proposals = [_make_proposal()]
        path = manager.save_proposals(proposals)
        assert path.exists()

    def test_load_pending_returns_saved_proposals(self, tmp_path: Path) -> None:
        """load_pending should return proposals saved by save_proposals."""
        manager = _make_manager(tmp_path)
        proposals = [
            _make_proposal(parameter="w_sentiment"),
            _make_proposal(parameter="w_rsi", current_value=0.20, proposed_value=0.15),
        ]
        manager.save_proposals(proposals)
        pending = manager.load_pending()
        assert len(pending) == 2

    def test_load_pending_returns_empty_when_no_file(self, tmp_path: Path) -> None:
        """load_pending should return empty list when proposals.json doesn't exist."""
        manager = _make_manager(tmp_path)
        pending = manager.load_pending()
        assert pending == []

    def test_saved_proposals_have_pending_status(self, tmp_path: Path) -> None:
        """Saved proposals should have 'pending' status."""
        manager = _make_manager(tmp_path)
        manager.save_proposals([_make_proposal()])

        with (tmp_path / "proposals.json").open("r") as fh:
            data = json.load(fh)
        assert all(p["status"] == "pending" for p in data["proposals"])

    def test_save_returns_correct_path(self, tmp_path: Path) -> None:
        """save_proposals should return the path to proposals.json."""
        manager = _make_manager(tmp_path)
        path = manager.save_proposals([])
        assert path == tmp_path / "proposals.json"


# ── approve() tests ────────────────────────────────────────────────────────────


class TestApproveProposal:
    def test_approve_patches_params_json(self, tmp_path: Path) -> None:
        """approve should update the parameter value in params.json."""
        manager = _make_manager(tmp_path)
        proposal = _make_proposal(parameter="w_sentiment", proposed_value=0.25)
        manager.save_proposals([proposal])

        manager.approve(proposal.proposal_id, approver="alice")

        with (tmp_path / "params.json").open("r") as fh:
            params = json.load(fh)

        assert params["weights"]["w_sentiment"] == 0.25

    def test_approve_moves_to_history(self, tmp_path: Path) -> None:
        """After approval, the proposal should appear in history."""
        manager = _make_manager(tmp_path)
        proposal = _make_proposal()
        manager.save_proposals([proposal])
        manager.approve(proposal.proposal_id, approver="alice")

        history = manager.get_history()
        assert len(history) == 1
        assert history[0]["status"] == "approved"
        assert history[0]["approver"] == "alice"

    def test_approve_removes_from_pending(self, tmp_path: Path) -> None:
        """After approval, the proposal should no longer appear in pending."""
        manager = _make_manager(tmp_path)
        proposal = _make_proposal()
        manager.save_proposals([proposal])
        manager.approve(proposal.proposal_id, approver="bob")

        pending = manager.load_pending()
        assert len(pending) == 0

    def test_approve_raises_on_missing_proposal(self, tmp_path: Path) -> None:
        """Approving a non-existent proposal ID should raise ValueError."""
        manager = _make_manager(tmp_path)
        manager.save_proposals([])
        with pytest.raises(ValueError, match="not found"):
            manager.approve("nonexistent-id", approver="alice")

    def test_approve_raises_on_risk_limits_field(self, tmp_path: Path) -> None:
        """Approving a proposal for a risk_limits field should raise ValueError."""
        manager = _make_manager(tmp_path)
        # Manually write a proposal that touches a protected field to proposals.json
        bad_proposal_id = str(uuid.uuid4())
        data = {
            "generated_at": "2024-01-01T00:00:00+00:00",
            "proposals": [
                {
                    "proposal_id": bad_proposal_id,
                    "parameter": "max_daily_drawdown_pct",
                    "current_value": 0.02,
                    "proposed_value": 0.05,
                    "reason": "Bad proposal",
                    "confidence": 0.9,
                    "status": "pending",
                }
            ],
        }
        with (tmp_path / "proposals.json").open("w") as fh:
            json.dump(data, fh)

        with pytest.raises(ValueError, match="validation"):
            manager.approve(bad_proposal_id, approver="alice")

    def test_approve_multiple_proposals_patches_correct_one(self, tmp_path: Path) -> None:
        """Approving one proposal should only affect its parameter in params.json."""
        manager = _make_manager(tmp_path)
        p1 = _make_proposal(parameter="w_rsi", current_value=0.20, proposed_value=0.18)
        p2 = _make_proposal(parameter="w_macd", current_value=0.20, proposed_value=0.22)
        manager.save_proposals([p1, p2])

        manager.approve(p1.proposal_id, approver="alice")

        with (tmp_path / "params.json").open("r") as fh:
            params = json.load(fh)

        assert params["weights"]["w_rsi"] == 0.18
        assert params["weights"]["w_macd"] == 0.20  # Unchanged


# ── reject() tests ─────────────────────────────────────────────────────────────


class TestRejectProposal:
    def test_reject_moves_to_history(self, tmp_path: Path) -> None:
        """Rejecting a proposal should move it to history."""
        manager = _make_manager(tmp_path)
        proposal = _make_proposal()
        manager.save_proposals([proposal])
        manager.reject(proposal.proposal_id, rejector="alice", reason="Not convinced")

        history = manager.get_history()
        assert len(history) == 1
        assert history[0]["status"] == "rejected"

    def test_reject_records_rejection_reason(self, tmp_path: Path) -> None:
        """The rejection reason should appear in the history entry."""
        manager = _make_manager(tmp_path)
        proposal = _make_proposal()
        manager.save_proposals([proposal])
        manager.reject(
            proposal.proposal_id,
            rejector="bob",
            reason="Insufficient evidence for parameter change",
        )

        history = manager.get_history()
        assert history[0]["rejection_reason"] == "Insufficient evidence for parameter change"
        assert history[0]["rejector"] == "bob"

    def test_reject_removes_from_pending(self, tmp_path: Path) -> None:
        """After rejection, the proposal should no longer appear in pending."""
        manager = _make_manager(tmp_path)
        proposal = _make_proposal()
        manager.save_proposals([proposal])
        manager.reject(proposal.proposal_id, rejector="alice", reason="No")

        pending = manager.load_pending()
        assert len(pending) == 0

    def test_reject_does_not_modify_params_json(self, tmp_path: Path) -> None:
        """Rejecting a proposal must NOT modify params.json."""
        manager = _make_manager(tmp_path)
        proposal = _make_proposal(parameter="w_sentiment", proposed_value=0.10)
        manager.save_proposals([proposal])

        with (tmp_path / "params.json").open("r") as fh:
            original_params = json.load(fh)

        manager.reject(proposal.proposal_id, rejector="alice", reason="No")

        with (tmp_path / "params.json").open("r") as fh:
            params_after = json.load(fh)

        assert params_after == original_params

    def test_reject_raises_on_missing_proposal(self, tmp_path: Path) -> None:
        """Rejecting a non-existent proposal ID should raise ValueError."""
        manager = _make_manager(tmp_path)
        manager.save_proposals([])
        with pytest.raises(ValueError, match="not found"):
            manager.reject("nonexistent-id", rejector="alice", reason="N/A")


# ── get_history() tests ────────────────────────────────────────────────────────


class TestGetHistory:
    def test_history_retained_across_multiple_actions(self, tmp_path: Path) -> None:
        """History should accumulate entries from multiple approve/reject calls."""
        manager = _make_manager(tmp_path)
        p1 = _make_proposal(parameter="w_rsi", current_value=0.20, proposed_value=0.18)
        p2 = _make_proposal(parameter="w_bb", current_value=0.15, proposed_value=0.12)
        p3 = _make_proposal(parameter="w_mom", current_value=0.15, proposed_value=0.20)

        manager.save_proposals([p1, p2, p3])
        manager.approve(p1.proposal_id, approver="alice")
        manager.reject(p2.proposal_id, rejector="bob", reason="Disagree")
        # p3 is left pending

        history = manager.get_history()
        assert len(history) == 2

        statuses = {h["proposal_id"]: h["status"] for h in history}
        assert statuses[p1.proposal_id] == "approved"
        assert statuses[p2.proposal_id] == "rejected"

    def test_history_empty_initially(self, tmp_path: Path) -> None:
        """History should be empty when no actions have been taken."""
        manager = _make_manager(tmp_path)
        assert manager.get_history() == []

    def test_history_entries_have_actioned_at_timestamp(self, tmp_path: Path) -> None:
        """Each history entry should have an actioned_at timestamp."""
        manager = _make_manager(tmp_path)
        proposal = _make_proposal()
        manager.save_proposals([proposal])
        manager.approve(proposal.proposal_id, approver="alice")

        history = manager.get_history()
        assert "actioned_at" in history[0]
        assert history[0]["actioned_at"]  # Non-empty

    def test_history_entries_include_proposal_metadata(self, tmp_path: Path) -> None:
        """History entries should preserve original proposal metadata."""
        manager = _make_manager(tmp_path)
        proposal = _make_proposal(
            parameter="w_sentiment",
            current_value=0.30,
            proposed_value=0.25,
            confidence=0.82,
        )
        manager.save_proposals([proposal])
        manager.approve(proposal.proposal_id, approver="charlie")

        history = manager.get_history()
        entry = history[0]
        assert entry["parameter"] == "w_sentiment"
        assert entry["current_value"] == 0.30
        assert entry["proposed_value"] == 0.25
        assert entry["confidence"] == 0.82
