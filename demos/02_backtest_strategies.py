#!/usr/bin/env python3
"""QTS Demo 2: Strategy Backtest Comparison.

Runs backtests with all 3 strategies on synthetic bars and compares results.

Usage:
    PYTHONPATH=src .venv/bin/python demos/02_backtest_strategies.py
"""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from qts.config import get_settings
from qts.data.market.mock_adapter import MockBarAdapter
from qts.simulation.backtest import BacktestEngine, BacktestResult, BacktestSettings
from qts.strategies.mean_reversion import MeanReversionStrategy
from qts.strategies.momentum import MomentumStrategy
from qts.strategies.sma_crossover import SMACrossoverStrategy

console = Console()

N_BARS = 500
SYMBOL = "BTC-USD"
INITIAL_CAPITAL = 100_000.0

SPARKLINE_CHARS = "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"


def make_sparkline(values: list[float], width: int = 40) -> Text:
    """Map a numeric series to a sparkline string using block characters."""
    if not values:
        return Text("")

    # Downsample to `width` points
    step = max(1, len(values) // width)
    sampled = values[::step][:width]

    lo = min(sampled)
    hi = max(sampled)
    span = hi - lo if hi != lo else 1.0

    chars = []
    for v in sampled:
        idx = int((v - lo) / span * (len(SPARKLINE_CHARS) - 1))
        idx = max(0, min(idx, len(SPARKLINE_CHARS) - 1))
        chars.append(SPARKLINE_CHARS[idx])

    # Colour: green if ended higher, red if lower, yellow if flat
    final_return = sampled[-1] - sampled[0]
    if final_return > 0:
        colour = "green"
    elif final_return < 0:
        colour = "red"
    else:
        colour = "yellow"

    return Text("".join(chars), style=colour)


def fmt_pct(v: float) -> Text:
    """Format a fraction as a signed percentage with colour."""
    pct = v * 100
    sign = "+" if pct >= 0 else ""
    style = "green" if pct >= 0 else "red"
    return Text(f"{sign}{pct:.2f}%", style=style)


def fmt_ratio(v: float) -> Text:
    """Format a ratio with colour (green if positive)."""
    style = "green" if v > 0 else ("red" if v < 0 else "dim")
    return Text(f"{v:.2f}", style=style)


def fmt_drawdown(v: float) -> Text:
    """Format max drawdown (always negative display)."""
    pct = v * 100
    return Text(f"-{pct:.2f}%", style="red")


async def generate_bars():
    """Generate synthetic bars using MockBarAdapter."""
    adapter = MockBarAdapter(seed=42, n_bars=N_BARS, initial_price=50_000.0)
    bars = await adapter.get_historical_bars(SYMBOL, start="", end="")
    return bars


def run_backtest(strategy, bars) -> BacktestResult:
    """Run a backtest with the given strategy."""
    settings = BacktestSettings(initial_capital=INITIAL_CAPITAL)
    engine = BacktestEngine(strategy, bars, settings)
    return engine.run()


def build_comparison_table(
    names: list[str], results: list[BacktestResult]
) -> Table:
    """Build a rich Table comparing strategy metrics."""
    table = Table(title="Strategy Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold", min_width=18)
    for name in names:
        table.add_column(name, justify="right", min_width=12)

    # Total Return
    table.add_row("Total Return", *[fmt_pct(r.total_return) for r in results])
    table.add_row("Sharpe Ratio", *[fmt_ratio(r.sharpe_ratio) for r in results])
    table.add_row("Max Drawdown", *[fmt_drawdown(r.max_drawdown) for r in results])
    table.add_row(
        "Win Rate",
        *[Text(f"{r.win_rate * 100:.1f}%", style="bold") for r in results],
    )
    table.add_row("Profit Factor", *[fmt_ratio(r.profit_factor) for r in results])
    table.add_row("Sortino Ratio", *[fmt_ratio(r.sortino_ratio) for r in results])
    table.add_row("Calmar Ratio", *[fmt_ratio(r.calmar_ratio) for r in results])
    table.add_row(
        "# Trades",
        *[Text(str(len(r.trades)), style="bold") for r in results],
    )

    return table


def build_trade_table(name: str, result: BacktestResult) -> Table:
    """Build a rich Table showing individual trades for a strategy."""
    table = Table(title=f"Trade Log: {name}", show_header=True, header_style="bold magenta")
    table.add_column("Entry Time", style="dim", min_width=20)
    table.add_column("Exit Time", style="dim", min_width=20)
    table.add_column("Direction", min_width=8)
    table.add_column("Entry Price", justify="right")
    table.add_column("Exit Price", justify="right")
    table.add_column("PnL ($)", justify="right")
    table.add_column("Outcome", justify="center")

    for trade in result.trades:
        pnl_style = "green" if trade.pnl_usd >= 0 else "red"
        outcome_style = "bold green" if trade.outcome.value == "win" else "bold red"

        table.add_row(
            trade.entry_time.strftime("%Y-%m-%d %H:%M"),
            trade.exit_time.strftime("%Y-%m-%d %H:%M"),
            trade.direction.value.upper(),
            f"${trade.entry_price:,.2f}",
            f"${trade.exit_price:,.2f}",
            Text(f"{trade.pnl_usd:+,.2f}", style=pnl_style),
            Text(trade.outcome.value.upper(), style=outcome_style),
        )

    return table


def main() -> None:
    console.print(
        Panel(
            "[bold cyan]QTS Demo 2: Strategy Backtest Comparison[/bold cyan]",
            style="bold",
            expand=False,
        )
    )

    # -- Generate bars --
    console.print(f"\n[bold]Generating {N_BARS} synthetic bars...[/bold]")
    bars = asyncio.run(generate_bars())
    console.print(f"  [dim]Symbol: {SYMBOL} | Bars: {len(bars)} | Seed: 42[/dim]\n")

    # -- Build strategies --
    settings = get_settings()
    params = settings.strategy
    risk_limits = settings.risk

    strategies: dict[str, object] = {
        "SMA Crossover": SMACrossoverStrategy(fast_period=10, slow_period=30, quantity=1.0),
        "Momentum": MomentumStrategy(params, risk_limits, portfolio_value=INITIAL_CAPITAL),
        "Mean Reversion": MeanReversionStrategy(portfolio_value=INITIAL_CAPITAL),
    }

    # -- Run backtests --
    names: list[str] = []
    results: list[BacktestResult] = []

    for name, strategy in strategies.items():
        console.print(f"  Running [bold]{name}[/bold]...", end=" ")
        result = run_backtest(strategy, bars)
        names.append(name)
        results.append(result)
        n_trades = len(result.trades)
        console.print(f"[green]done[/green] ({n_trades} trades)")

    # -- Comparison table --
    console.print()
    console.print(build_comparison_table(names, results))

    # -- Equity curves --
    console.print()
    console.print("[bold]Equity Curves (sparkline)[/bold]")
    max_name_len = max(len(n) for n in names)
    for name, result in zip(names, results):
        label = f"  {name:<{max_name_len}}: "
        sparkline = make_sparkline(result.equity_curve)
        line = Text(label)
        line.append(sparkline)
        console.print(line)

    # -- Trade logs --
    for name, result in zip(names, results):
        console.print()
        if result.trades:
            console.print(build_trade_table(name, result))
        else:
            console.print(f"[dim]  {name}: no trades executed[/dim]")

    # -- Winner summary --
    console.print()
    best_idx = max(range(len(results)), key=lambda i: results[i].sharpe_ratio)
    best_name = names[best_idx]
    best_sharpe = results[best_idx].sharpe_ratio

    console.print(
        Panel(
            f"[bold yellow]Best Strategy: {best_name} (Sharpe: {best_sharpe:.2f})[/bold yellow]",
            title="Winner",
            style="bold yellow",
            expand=False,
        )
    )

    console.print("\n[bold green]All backtests completed successfully[/bold green]")


if __name__ == "__main__":
    main()
