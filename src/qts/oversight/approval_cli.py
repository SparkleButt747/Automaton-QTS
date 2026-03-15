"""Rich CLI for reviewing and approving LLM strategy proposals."""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from qts.oversight.debrief import Proposal
from qts.oversight.proposals import ProposalManager

logger = logging.getLogger(__name__)

console = Console()


def _trigger_git_commit(config_dir: Path, proposal: Proposal) -> None:
    """Attempt to create a git commit recording a parameter change.

    Args:
        config_dir: Directory containing params.json.
        proposal: The approved proposal.
    """
    params_path = config_dir / "params.json"
    try:
        subprocess.run(
            ["git", "add", str(params_path)],
            check=True,
            capture_output=True,
        )
        commit_msg = (
            f"chore(params): apply LLM proposal {proposal.proposal_id[:8]}\n\n"
            f"Parameter: {proposal.parameter}\n"
            f"Change: {proposal.current_value} -> {proposal.proposed_value}\n"
            f"Reason: {proposal.reason}\n"
            f"Confidence: {proposal.confidence:.2f}"
        )
        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            check=True,
            capture_output=True,
        )
        console.print(f"[green]  Git commit created for proposal {proposal.proposal_id[:8]}[/green]")
    except subprocess.CalledProcessError as exc:
        logger.warning("Failed to create git commit for proposal %s: %s", proposal.proposal_id, exc)
        console.print(
            f"[yellow]  Warning: git commit failed (non-fatal): {exc}[/yellow]"
        )


def run_approval_cli(
    config_dir: Path,
    risk_limits_fields: set[str],
    approver: str = "human",
) -> None:
    """Run the interactive approval CLI for pending proposals.

    Displays proposals in a Rich table and prompts the user to approve,
    reject, or skip each one.

    Args:
        config_dir: Directory containing proposals.json and params.json.
        risk_limits_fields: Set of protected risk_limits field names.
        approver: Identifier string for the human approver.
    """
    manager = ProposalManager(config_dir=config_dir, risk_limits_fields=risk_limits_fields)
    pending = manager.load_pending()

    if not pending:
        console.print("[bold yellow]No pending proposals found.[/bold yellow]")
        return

    console.print("\n[bold cyan]QTS LLM Strategy Proposal Review[/bold cyan]")
    console.print(f"Found [bold]{len(pending)}[/bold] pending proposal(s).\n")

    # Build display table
    table = Table(
        title="Pending Proposals",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("ID", style="dim", width=10)
    table.add_column("Parameter", style="cyan")
    table.add_column("Current", justify="right")
    table.add_column("Proposed", justify="right")
    table.add_column("Reason")
    table.add_column("Confidence", justify="right")

    for i, proposal in enumerate(pending, start=1):
        table.add_row(
            str(i),
            proposal.proposal_id[:8],
            proposal.parameter,
            f"{proposal.current_value:.4f}",
            f"{proposal.proposed_value:.4f}",
            proposal.reason,
            f"{proposal.confidence:.2f}",
        )

    console.print(table)
    console.print()

    approved_count = 0
    rejected_count = 0
    skipped_count = 0

    for i, proposal in enumerate(pending, start=1):
        console.print(
            f"\n[bold]Proposal {i}/{len(pending)}[/bold]: "
            f"[cyan]{proposal.parameter}[/cyan] "
            f"{proposal.current_value:.4f} \u2192 [green]{proposal.proposed_value:.4f}[/green]"
        )
        console.print(f"  Reason: {proposal.reason}")
        console.print(f"  Confidence: {proposal.confidence:.2f}")

        while True:
            choice = Prompt.ask(
                "  [A]pprove / [R]eject / [S]kip",
                choices=["a", "r", "s", "A", "R", "S"],
                default="s",
            ).lower()

            if choice == "a":
                try:
                    manager.approve(proposal.proposal_id, approver)
                    _trigger_git_commit(config_dir, proposal)
                    console.print(f"  [green]APPROVED[/green] proposal {proposal.proposal_id[:8]}")
                    approved_count += 1
                except ValueError as exc:
                    console.print(f"  [red]ERROR: {exc}[/red]")
                break

            elif choice == "r":
                rejection_reason = Prompt.ask("  Rejection reason")
                manager.reject(proposal.proposal_id, approver, rejection_reason)
                console.print(f"  [red]REJECTED[/red] proposal {proposal.proposal_id[:8]}")
                rejected_count += 1
                break

            elif choice == "s":
                console.print(f"  [yellow]SKIPPED[/yellow] proposal {proposal.proposal_id[:8]}")
                skipped_count += 1
                break

    # Summary
    console.print(
        f"\n[bold]Summary:[/bold] "
        f"[green]{approved_count} approved[/green], "
        f"[red]{rejected_count} rejected[/red], "
        f"[yellow]{skipped_count} skipped[/yellow]"
    )


def main() -> None:
    """Entry point for the approval CLI when invoked as a script."""
    import argparse

    from qts.config import get_settings

    parser = argparse.ArgumentParser(description="QTS LLM Proposal Approval CLI")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="Path to config directory (default: auto-detected from project)",
    )
    parser.add_argument(
        "--approver",
        type=str,
        default="human",
        help="Identifier for the approver (default: 'human')",
    )
    args = parser.parse_args()

    settings = get_settings()
    config_dir = args.config_dir or (Path(__file__).parent.parent.parent.parent / "config")

    # Load risk limits fields from risk_limits.json
    risk_limits_path = config_dir / "risk_limits.json"
    if risk_limits_path.exists():
        import json

        with risk_limits_path.open("r") as fh:
            raw = json.load(fh)
        raw.pop("_comment", None)
        risk_limits_fields = set(raw.keys())
    else:
        risk_limits_fields = set()

    run_approval_cli(
        config_dir=config_dir,
        risk_limits_fields=risk_limits_fields,
        approver=args.approver,
    )


if __name__ == "__main__":
    main()
