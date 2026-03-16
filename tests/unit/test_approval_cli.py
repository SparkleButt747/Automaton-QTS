"""Unit tests for qts.oversight.approval_cli.

Covers:
- T-ACLI-1 to T-ACLI-14
- run_approval_cli: no proposals (early return)
- run_approval_cli: approve, reject, skip flows
- _trigger_git_commit: success and failure paths
- main() entry point doesn't crash when no proposals exist
"""
from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from qts.oversight.approval_cli import _trigger_git_commit, run_approval_cli
from qts.oversight.debrief import Proposal
from qts.oversight.proposals import ProposalManager


# ── Helpers ───────────────────────────────────────────────────────────────────

_RISK_LIMITS_FIELDS: set[str] = {
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
    "exit_threshold": 0.05,
    "max_hold_bars": 48,
    "sentiment_fusion_weights": {"news": 0.5, "social": 0.3, "geopolitical": 0.2},
}


def _make_proposal(
    parameter: str = "w_sentiment",
    current_value: float = 0.30,
    proposed_value: float = 0.25,
) -> Proposal:
    return Proposal(
        proposal_id=str(uuid.uuid4()),
        parameter=parameter,
        current_value=current_value,
        proposed_value=proposed_value,
        reason="Test reason",
        confidence=0.75,
    )


def _setup_config_dir(tmp_path: Path) -> tuple[Path, ProposalManager]:
    params_path = tmp_path / "params.json"
    with params_path.open("w") as fh:
        json.dump(_DEFAULT_PARAMS, fh)
    manager = ProposalManager(config_dir=tmp_path, risk_limits_fields=_RISK_LIMITS_FIELDS)
    return tmp_path, manager


# ── run_approval_cli: no proposals ────────────────────────────────────────────


class TestRunApprovalCliNoProposals:
    def test_T_ACLI_1_returns_early_when_no_pending(self, tmp_path: Path) -> None:
        """run_approval_cli should exit immediately when there are no pending proposals."""
        config_dir, _ = _setup_config_dir(tmp_path)
        # No proposals.json saved, so pending is empty
        # Patching Prompt.ask so it never gets called
        with patch("rich.prompt.Prompt.ask") as mock_ask:
            run_approval_cli(config_dir, _RISK_LIMITS_FIELDS)
            mock_ask.assert_not_called()

    def test_T_ACLI_2_no_exception_with_no_pending(self, tmp_path: Path) -> None:
        config_dir, _ = _setup_config_dir(tmp_path)
        run_approval_cli(config_dir, _RISK_LIMITS_FIELDS)  # should not raise


# ── run_approval_cli: skip ─────────────────────────────────────────────────────


class TestRunApprovalCliSkip:
    def test_T_ACLI_3_skip_does_not_modify_params(self, tmp_path: Path) -> None:
        config_dir, manager = _setup_config_dir(tmp_path)
        proposal = _make_proposal(parameter="w_sentiment", proposed_value=0.10)
        manager.save_proposals([proposal])

        with patch("rich.prompt.Prompt.ask", return_value="s"):
            run_approval_cli(config_dir, _RISK_LIMITS_FIELDS)

        with (tmp_path / "params.json").open() as fh:
            params = json.load(fh)
        assert params["weights"]["w_sentiment"] == pytest.approx(0.30)

    def test_T_ACLI_4_skip_leaves_proposal_in_pending(self, tmp_path: Path) -> None:
        config_dir, manager = _setup_config_dir(tmp_path)
        proposal = _make_proposal()
        manager.save_proposals([proposal])

        with patch("rich.prompt.Prompt.ask", return_value="s"):
            run_approval_cli(config_dir, _RISK_LIMITS_FIELDS)

        # Proposal should still be pending (skip doesn't change status)
        pending = manager.load_pending()
        assert len(pending) == 1


# ── run_approval_cli: approve ──────────────────────────────────────────────────


class TestRunApprovalCliApprove:
    def test_T_ACLI_5_approve_updates_params_json(self, tmp_path: Path) -> None:
        config_dir, manager = _setup_config_dir(tmp_path)
        proposal = _make_proposal(parameter="w_sentiment", proposed_value=0.20)
        manager.save_proposals([proposal])

        # Mock Prompt.ask to return "a" (approve), and suppress git commit
        with (
            patch("rich.prompt.Prompt.ask", return_value="a"),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            run_approval_cli(config_dir, _RISK_LIMITS_FIELDS, approver="alice")

        with (tmp_path / "params.json").open() as fh:
            params = json.load(fh)
        assert params["weights"]["w_sentiment"] == pytest.approx(0.20)

    def test_T_ACLI_6_approve_removes_from_pending(self, tmp_path: Path) -> None:
        config_dir, manager = _setup_config_dir(tmp_path)
        proposal = _make_proposal()
        manager.save_proposals([proposal])

        with (
            patch("rich.prompt.Prompt.ask", return_value="a"),
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
        ):
            run_approval_cli(config_dir, _RISK_LIMITS_FIELDS)

        pending = manager.load_pending()
        assert len(pending) == 0


# ── run_approval_cli: reject ───────────────────────────────────────────────────


class TestRunApprovalCliReject:
    def test_T_ACLI_7_reject_moves_to_history(self, tmp_path: Path) -> None:
        config_dir, manager = _setup_config_dir(tmp_path)
        proposal = _make_proposal()
        manager.save_proposals([proposal])

        # First ask returns "r", second ask (rejection reason) returns "Not convinced"
        with patch("rich.prompt.Prompt.ask", side_effect=["r", "Not convinced"]):
            run_approval_cli(config_dir, _RISK_LIMITS_FIELDS, approver="bob")

        history = manager.get_history()
        assert len(history) == 1
        assert history[0]["status"] == "rejected"

    def test_T_ACLI_8_reject_does_not_modify_params(self, tmp_path: Path) -> None:
        config_dir, manager = _setup_config_dir(tmp_path)
        proposal = _make_proposal(parameter="w_sentiment", proposed_value=0.05)
        manager.save_proposals([proposal])

        with patch("rich.prompt.Prompt.ask", side_effect=["r", "Disagree"]):
            run_approval_cli(config_dir, _RISK_LIMITS_FIELDS)

        with (tmp_path / "params.json").open() as fh:
            params = json.load(fh)
        assert params["weights"]["w_sentiment"] == pytest.approx(0.30)  # unchanged


# ── run_approval_cli: multiple proposals ──────────────────────────────────────


class TestRunApprovalCliMultiple:
    def test_T_ACLI_9_processes_all_proposals(self, tmp_path: Path) -> None:
        config_dir, manager = _setup_config_dir(tmp_path)
        p1 = _make_proposal(parameter="w_rsi", proposed_value=0.18)
        p2 = _make_proposal(parameter="w_macd", proposed_value=0.22)
        manager.save_proposals([p1, p2])

        # Skip both
        with patch("rich.prompt.Prompt.ask", return_value="s"):
            run_approval_cli(config_dir, _RISK_LIMITS_FIELDS)

        # Both still pending (skipped)
        pending = manager.load_pending()
        assert len(pending) == 2


# ── _trigger_git_commit ───────────────────────────────────────────────────────


class TestTriggerGitCommit:
    def test_T_ACLI_10_calls_git_add_and_commit(self, tmp_path: Path) -> None:
        proposal = _make_proposal()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            _trigger_git_commit(tmp_path, proposal)
        assert mock_run.call_count == 2
        first_call_args = mock_run.call_args_list[0][0][0]
        assert "git" in first_call_args
        assert "add" in first_call_args

    def test_T_ACLI_11_git_failure_does_not_raise(self, tmp_path: Path) -> None:
        proposal = _make_proposal()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git")
            _trigger_git_commit(tmp_path, proposal)  # should not raise

    def test_T_ACLI_12_commit_message_contains_proposal_id_prefix(
        self, tmp_path: Path
    ) -> None:
        proposal = _make_proposal()
        short_id = proposal.proposal_id[:8]
        captured_messages = []

        def capture_run(args, **kwargs):
            if "commit" in args:
                # -m is followed by the message
                idx = args.index("-m")
                captured_messages.append(args[idx + 1])
            return MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=capture_run):
            _trigger_git_commit(tmp_path, proposal)

        assert any(short_id in msg for msg in captured_messages)


# ── Uppercase input normalised ─────────────────────────────────────────────────


class TestUppercaseChoices:
    def test_T_ACLI_13_uppercase_S_treated_as_skip(self, tmp_path: Path) -> None:
        """Prompt returning 'S' (uppercase) should be treated as skip."""
        config_dir, manager = _setup_config_dir(tmp_path)
        proposal = _make_proposal()
        manager.save_proposals([proposal])

        with patch("rich.prompt.Prompt.ask", return_value="S"):
            run_approval_cli(config_dir, _RISK_LIMITS_FIELDS)

        # Proposal should be unchanged (skipped)
        pending = manager.load_pending()
        assert len(pending) == 1

    def test_T_ACLI_14_uppercase_A_treated_as_approve(self, tmp_path: Path) -> None:
        config_dir, manager = _setup_config_dir(tmp_path)
        proposal = _make_proposal(parameter="w_sentiment", proposed_value=0.20)
        manager.save_proposals([proposal])

        with (
            patch("rich.prompt.Prompt.ask", return_value="A"),
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
        ):
            run_approval_cli(config_dir, _RISK_LIMITS_FIELDS, approver="alice")

        with (tmp_path / "params.json").open() as fh:
            params = json.load(fh)
        assert params["weights"]["w_sentiment"] == pytest.approx(0.20)
