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
    """Run a historical backtest via NautilusTrader against a MarketTerrain."""
    console.print(f"[bold green]Running backtest:[/bold green] {symbol} | {start} -> {end}")
    console.print(f"[dim]Output: {output}[/dim]")
    console.print("[yellow]Use `qts terrain-backtest` for terrain-based backtesting.[/yellow]")
    sys.exit(0)


@main.command(name="terrain-backtest")
@click.option("--terrain", "-t", required=True, help="Terrain name from config/terrains/")
@click.option("--venue", default="BINANCE", help="Venue name for simulation")
@click.pass_context
def terrain_backtest(ctx: click.Context, terrain: str, venue: str) -> None:
    """Run a strategy against a named MarketTerrain via NautilusTrader."""
    from qts.terrain.library import TerrainLibrary

    console.print(f"[bold green]Terrain backtest:[/bold green] {terrain} @ {venue}")

    library = TerrainLibrary.from_config_dir()
    t = library.get_by_name(terrain)
    if t is None:
        console.print(f"[red]Terrain '{terrain}' not found in catalog.[/red]")
        available = [t.name for t in library.all_terrains]
        console.print(f"[dim]Available: {', '.join(available)}[/dim]")
        sys.exit(1)

    console.print(f"  Regime: {t.regime.trend}/{t.regime.volatility}/{t.regime.liquidity}")
    console.print(f"  Bars: {len(t.bars)}")
    console.print(f"  Period: {t.start} → {t.end}")

    if not t.bars:
        console.print("[yellow]Terrain has no bar data loaded. Load data first.[/yellow]")
        sys.exit(1)

    from qts.config import get_settings
    from qts.nautilus.config import VenueConfig
    from qts.nautilus.runner import run_terrain_backtest
    from qts.strategies.momentum import MomentumStrategy

    settings = get_settings()
    strategy = MomentumStrategy(
        params=settings.strategy,
        risk_limits=settings.risk,
    )
    result = run_terrain_backtest(
        terrain=t,
        strategy=strategy,
        venue_config=VenueConfig(name=venue),
    )

    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Sharpe:     {result.sharpe_ratio:.4f}")
    console.print(f"  Return:     {result.total_return:.2%}")
    console.print(f"  Max DD:     {result.max_drawdown:.2%}")
    console.print(f"  Win Rate:   {result.win_rate:.2%}")
    console.print(f"  Trades:     {result.total_trades}")


@main.command()
@click.option("--symbol", "-s", default="BTCUSDT", help="Trading symbol")
@click.option("--venue", default="BINANCE", help="Venue name")
@click.option("--testnet/--production", default=True, help="Use testnet (default: true)")
@click.pass_context
def live(ctx: click.Context, symbol: str, venue: str, testnet: bool) -> None:
    """Start live trading via NautilusTrader TradingNode."""
    dry_run: bool = ctx.obj.get("dry_run", True)
    mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[bold red]LIVE TRADING[/bold red]"
    console.print(f"[bold]Starting {mode} for {symbol} @ {venue}[/bold]")

    if not testnet and not dry_run:
        click.confirm(
            "You are about to start LIVE PRODUCTION trading. Continue?",
            abort=True,
        )

    from qts.config import get_settings
    from qts.nautilus.live import LiveVenueConfig, run_live_strategy
    from qts.strategies.momentum import MomentumStrategy

    settings = get_settings()
    strategy = MomentumStrategy(
        params=settings.strategy,
        risk_limits=settings.risk,
    )

    venue_config = LiveVenueConfig(
        venue_name=venue,
        is_testnet=testnet,
        api_key=settings.exchange.binance_api_key,
        api_secret=settings.exchange.binance_api_secret,
    )

    run_live_strategy(
        strategy=strategy,
        venue_config=venue_config,
        symbols=[symbol],
    )


@main.command()
@click.option("--n-trials", "-n", default=50, help="Number of Optuna trials")
@click.option("--workers", "-w", default=1, help="Parallel workers")
@click.option("--study-name", default="qts_momentum", help="Optuna study name")
@click.pass_context
def optimise(ctx: click.Context, n_trials: int, workers: int, study_name: str) -> None:
    """Run Optuna parameter optimisation across diverse terrains."""
    from qts.config import get_settings
    from qts.optimisation.tuner import TunerConfig, run_strategy_study
    from qts.terrain.library import TerrainLibrary

    settings = get_settings()
    library = TerrainLibrary.from_config_dir()
    train = library.get_train_terrains()

    if not train:
        console.print("[red]No training terrains found in config/terrains/[/red]")
        sys.exit(1)

    console.print(f"[bold green]Optimisation:[/bold green] {n_trials} trials, {workers} workers")
    console.print(f"  Training terrains: {len(train)}")
    for t in train:
        console.print(f"    - {t.name} ({t.regime.trend}/{t.regime.volatility})")

    config = TunerConfig(
        n_trials=n_trials,
        n_workers=workers,
        study_name=study_name,
    )

    result = run_strategy_study(
        train_terrains=train,
        risk_limits=settings.risk,
        config=config,
    )

    console.print("\n[bold]Results:[/bold]")
    console.print(f"  Completed: {result.n_completed}")
    console.print(f"  Pruned:    {result.n_pruned}")
    console.print(f"  Best score: {result.best_score:.4f}")
    console.print(f"  Best params: {result.best_params}")


@main.command(name="terrain-list")
def terrain_list() -> None:
    """List all available terrains from the catalog."""
    from qts.terrain.library import TerrainLibrary

    library = TerrainLibrary.from_config_dir()

    console.print("[bold]Available Terrains:[/bold]\n")
    for t in library.all_terrains:
        regime = t.regime
        console.print(
            f"  {t.name:<35s} "
            f"{regime.trend:<10s} {regime.volatility:<15s} "
            f"{regime.liquidity:<10s} {regime.sentiment}"
        )


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
