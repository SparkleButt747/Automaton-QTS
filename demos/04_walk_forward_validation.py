#!/usr/bin/env python3
"""QTS Demo 4: Walk-Forward Validation with Monte Carlo Significance Testing.

Generates synthetic bars, runs walk-forward backtesting with SMA crossover,
and tests statistical significance via Monte Carlo permutation.

Usage:
    PYTHONPATH=src .venv/bin/python demos/04_walk_forward_validation.py
"""

from __future__ import annotations

import asyncio

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from qts.data.market.mock_adapter import MockBarAdapter
from qts.simulation.backtest import BacktestSettings
from qts.simulation.walk_forward import (
    WalkForwardConfig,
    WalkForwardEngine,
    WalkForwardResult,
    monte_carlo_permutation_test,
)
from qts.strategies.sma_crossover import SMACrossoverStrategy

console = Console()

N_BARS = 2000
N_WINDOWS = 5
SYMBOL = "BTC-USD"
INITIAL_CAPITAL = 100_000.0
MC_PERMUTATIONS = 1_000
SIGNIFICANCE_THRESHOLD = 0.05


def fmt_pct(v: float) -> Text:
    """Format a fraction as a signed percentage with colour."""
    pct = v * 100
    sign = "+" if pct >= 0 else ""
    style = "green" if pct >= 0 else "red"
    return Text(f"{sign}{pct:.2f}%", style=style)


def fmt_sharpe(v: float) -> Text:
    """Format a Sharpe ratio with colour."""
    style = "green" if v > 0 else ("red" if v < 0 else "dim")
    return Text(f"{v:+.3f}", style=style)


async def generate_bars():
    """Generate synthetic bars using MockBarAdapter."""
    adapter = MockBarAdapter(seed=42, n_bars=N_BARS, initial_price=50_000.0, volatility=0.03)
    return await adapter.get_historical_bars(SYMBOL, start="", end="")


def build_window_table(result: WalkForwardResult) -> Table:
    """Build a rich Table showing per-window walk-forward results."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Window", justify="center", min_width=8)
    table.add_column("Train SR", justify="right", min_width=10)
    table.add_column("Test SR", justify="right", min_width=10)
    table.add_column("Test Return", justify="right", min_width=12)
    table.add_column("# Trades", justify="right", min_width=10)

    for w in result.windows:
        train_sr = w.train_result.sharpe_ratio if w.train_result else float("nan")
        test_sr = w.test_result.sharpe_ratio if w.test_result else float("nan")
        test_ret = w.test_result.total_return if w.test_result else float("nan")
        n_trades = len(w.test_result.trades) if w.test_result else 0

        table.add_row(
            str(w.window_index + 1),
            fmt_sharpe(train_sr),
            fmt_sharpe(test_sr),
            fmt_pct(test_ret),
            str(n_trades),
        )

    return table


def collect_test_returns(result: WalkForwardResult) -> list[float]:
    """Collect all per-bar equity returns from test windows for Monte Carlo."""
    all_returns: list[float] = []
    for w in result.windows:
        if w.test_result and len(w.test_result.equity_curve) > 1:
            eq = np.array(w.test_result.equity_curve, dtype=np.float64)
            returns = np.diff(eq) / eq[:-1]
            all_returns.extend(returns[np.isfinite(returns)].tolist())
    return all_returns


def main() -> None:
    console.print(
        Panel(
            "[bold cyan]QTS Demo 4: Walk-Forward Validation[/bold cyan]",
            style="bold",
            expand=False,
        )
    )

    # -- Generate bars --
    bars = asyncio.run(generate_bars())
    console.print(
        f"\n[bold]Running {N_WINDOWS}-window walk-forward on {len(bars)} bars...[/bold]"
    )

    # -- Configure and run walk-forward --
    config = WalkForwardConfig(
        n_windows=N_WINDOWS,
        train_ratio=0.70,
        validate_ratio=0.15,
        test_ratio=0.15,
        min_bars_per_window=50,
    )
    settings = BacktestSettings(initial_capital=INITIAL_CAPITAL)

    def strategy_factory():
        return SMACrossoverStrategy(fast_period=10, slow_period=30, quantity=1.0)

    engine = WalkForwardEngine(strategy_factory, settings, config)
    result = engine.run(bars)

    # -- Per-window results --
    console.print()
    console.print("[bold]Per-Window Results[/bold]")
    console.print(build_window_table(result))

    # -- Monte Carlo significance test --
    test_returns = collect_test_returns(result)
    if test_returns:
        mc_p_value = monte_carlo_permutation_test(
            test_returns, n_permutations=MC_PERMUTATIONS, seed=42
        )
    else:
        mc_p_value = 1.0

    is_mc_significant = mc_p_value < SIGNIFICANCE_THRESHOLD

    # -- Aggregate results panel --
    console.print()

    agg_lines = [
        f"  Aggregate Test Sharpe:   {result.aggregate_test_sharpe:+.3f}",
        f"  Aggregate Test Return:   {result.aggregate_test_return * 100:+.2f}%",
        f"  Max Drawdown:            {result.aggregate_test_max_drawdown * 100:.2f}%",
        f"  Positive-Sharpe Windows: {sum(1 for s in result.per_window_test_sharpes if s > 0)}/{len(result.per_window_test_sharpes)}",
        f"  Majority Significant:    {'YES' if result.is_significant else 'NO'}",
    ]
    console.print(
        Panel(
            "\n".join(agg_lines),
            title="[bold]Aggregate Results[/bold]",
            style="cyan",
            expand=False,
        )
    )

    # -- Monte Carlo panel --
    console.print()
    if is_mc_significant:
        sig_text = f"p-value: {mc_p_value:.3f} (< {SIGNIFICANCE_THRESHOLD} -> SIGNIFICANT)"
        sig_style = "bold green"
    else:
        sig_text = f"p-value: {mc_p_value:.3f} (>= {SIGNIFICANCE_THRESHOLD} -> NOT SIGNIFICANT)"
        sig_style = "bold red"

    mc_lines = [
        f"  Permutations:  {MC_PERMUTATIONS:,}",
        f"  Test returns:  {len(test_returns)} data points",
        f"  {sig_text}",
    ]
    console.print(
        Panel(
            "\n".join(mc_lines),
            title="[bold]Monte Carlo Significance[/bold]",
            style="green" if is_mc_significant else "red",
            expand=False,
        )
    )

    console.print(f"\n[{sig_style}]{sig_text}[/{sig_style}]")
    console.print("\n[bold green]Walk-forward validation complete[/bold green]")


if __name__ == "__main__":
    main()
