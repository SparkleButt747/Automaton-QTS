"""QTS CLI entry point.

Provides the main Click command group and sub-commands for running
backtests, live trading, debriefs, and managing proposals.
"""

from __future__ import annotations

import asyncio
import logging
import sys

import click
from rich.console import Console

console = Console()
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(package_name="qts")
@click.option("--dry-run/--live", default=True, help="Run in dry-run (paper) mode (default: true)")
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Log level",
)
@click.option(
    "--skip-db-init",
    is_flag=True,
    default=False,
    help="Skip automatic database table creation on startup",
)
@click.pass_context
def main(ctx: click.Context, dry_run: bool, log_level: str, skip_db_init: bool) -> None:
    """QTS - Quant Trading System CLI.

    A multi-signal crypto/equity strategy engine with sentiment fusion.
    """
    ctx.ensure_object(dict)
    ctx.obj["dry_run"] = dry_run
    ctx.obj["log_level"] = log_level

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    if not skip_db_init:
        from qts.db.engine import ensure_tables

        try:
            engine = asyncio.run(ensure_tables())
            ctx.obj["engine"] = engine
        except Exception:
            logger.warning("Database init failed; continuing without DB", exc_info=True)


@main.command()
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--start", required=True, help="Backtest start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="Backtest end date (YYYY-MM-DD)")
@click.option("--output", "-o", default="results/", help="Output directory for results")
@click.pass_context
def backtest(ctx: click.Context, symbol: str, start: str, end: str, output: str) -> None:
    """Run a historical backtest for the configured strategy."""
    console.print(f"[bold green]Running backtest:[/bold green] {symbol} | {start} -> {end}")
    console.print(f"[dim]Output: {output}[/dim]")
    console.print("[yellow]Backtest engine not yet implemented.[/yellow]")
    sys.exit(0)


@main.command()
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.pass_context
def live(ctx: click.Context, symbol: str) -> None:
    """Start the live trading engine."""
    dry_run: bool = ctx.obj.get("dry_run", True)
    mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[bold red]LIVE TRADING[/bold red]"
    console.print(f"[bold]Starting {mode} for {symbol}[/bold]")
    if not dry_run:
        click.confirm("Are you sure you want to start live trading?", abort=True)
    console.print("[yellow]Live trading engine not yet implemented.[/yellow]")


@main.command()
@click.pass_context
def debrief(ctx: click.Context) -> None:
    """Run post-session debrief and performance analysis."""
    console.print("[bold]Running post-session debrief...[/bold]")
    console.print("[yellow]Debrief engine not yet implemented.[/yellow]")


@main.command()
@click.pass_context
def approve(ctx: click.Context) -> None:
    """Review and approve/reject pending trade proposals."""
    console.print("[bold]Trade Proposal Approval Interface[/bold]")
    console.print("[yellow]Approval workflow not yet implemented.[/yellow]")


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show system status and configuration health."""
    from qts.config import get_settings

    settings = get_settings()
    issues = settings.validate_production_readiness()

    console.print("[bold]QTS System Status[/bold]")
    console.print(f"  Environment : [cyan]{settings.qts_env}[/cyan]")
    console.print(f"  Dry Run     : [cyan]{settings.qts_dry_run}[/cyan]")
    console.print(f"  Log Level   : [cyan]{settings.qts_log_level}[/cyan]")
    console.print(f"  Database    : [cyan]{settings.database.database_url}[/cyan]")
    console.print(f"  Redis       : [cyan]{settings.redis_url}[/cyan]")

    if issues:
        console.print("\n[bold yellow]Configuration Warnings:[/bold yellow]")
        for issue in issues:
            console.print(f"  [yellow]! {issue}[/yellow]")
    else:
        console.print("\n[bold green]Configuration OK[/bold green]")
