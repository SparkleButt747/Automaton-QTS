#!/usr/bin/env python3
"""QTS Demo 5: Live Binance Market Data.

Fetches real BTCUSDT 1-minute bars from the Binance public API (no key needed),
runs indicators on them, and performs a quick backtest on live data.

Usage:
    PYTHONPATH=src .venv/bin/python demos/05_binance_live_data.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from qts.data.market.binance_adapter import BinanceBarAdapter
from qts.models.base import Bar
from qts.signals.indicators import (
    compute_bollinger_bands,
    compute_bb_position,
    compute_macd,
    compute_rsi,
)
from qts.simulation.backtest import BacktestEngine, BacktestSettings
from qts.strategies.sma_crossover import SMACrossoverStrategy

console = Console()

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
FETCH_HOURS = 2
INITIAL_CAPITAL = 100_000.0
DISPLAY_BARS = 10


def fmt_price(v: float) -> str:
    """Format a price value."""
    return f"${v:,.2f}"


def fmt_volume(v: float) -> str:
    """Format volume with appropriate scaling."""
    if v >= 1_000:
        return f"{v:,.0f}"
    return f"{v:.4f}"


def fmt_pct(v: float) -> Text:
    """Format a fraction as a signed percentage with colour."""
    pct = v * 100
    sign = "+" if pct >= 0 else ""
    style = "green" if pct >= 0 else "red"
    return Text(f"{sign}{pct:.2f}%", style=style)


def fmt_ratio(v: float) -> Text:
    """Format a ratio with colour."""
    style = "green" if v > 0 else ("red" if v < 0 else "dim")
    return Text(f"{v:.2f}", style=style)


def build_price_table(bars: list[Bar]) -> Table:
    """Build a rich Table showing recent bar data."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Timestamp", style="dim", min_width=20)
    table.add_column("Open", justify="right", min_width=12)
    table.add_column("High", justify="right", min_width=12)
    table.add_column("Low", justify="right", min_width=12)
    table.add_column("Close", justify="right", min_width=12)
    table.add_column("Volume", justify="right", min_width=10)

    for bar in bars[-DISPLAY_BARS:]:
        change = bar.close - bar.open
        close_style = "green" if change >= 0 else "red"
        table.add_row(
            bar.timestamp.strftime("%Y-%m-%d %H:%M"),
            fmt_price(bar.open),
            fmt_price(bar.high),
            fmt_price(bar.low),
            Text(fmt_price(bar.close), style=close_style),
            fmt_volume(bar.volume),
        )

    return table


async def fetch_bars() -> list[Bar]:
    """Fetch recent bars from Binance."""
    adapter = BinanceBarAdapter()
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=FETCH_HOURS)
        bars = await adapter.get_historical_bars(
            SYMBOL, start.isoformat(), end.isoformat(), interval=INTERVAL
        )
        return bars
    finally:
        await adapter.close()


def compute_indicators(bars: list[Bar]) -> dict[str, float]:
    """Compute key indicators on bar data."""
    closes = np.array([b.close for b in bars], dtype=np.float64)

    rsi_arr = compute_rsi(closes)
    macd_line, signal_line, histogram = compute_macd(closes)
    bb_upper, _bb_mid, bb_lower = compute_bollinger_bands(closes)

    def _last_valid(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        return float(valid[-1]) if len(valid) > 0 else float("nan")

    bb_upper_val = _last_valid(bb_upper)
    bb_lower_val = _last_valid(bb_lower)
    bb_pos = compute_bb_position(closes[-1], bb_upper_val, bb_lower_val)

    return {
        "rsi": _last_valid(rsi_arr),
        "macd_histogram": _last_valid(histogram),
        "macd_line": _last_valid(macd_line),
        "macd_signal": _last_valid(signal_line),
        "bb_position": bb_pos,
    }


def run_backtest(bars: list[Bar]):
    """Run a quick SMA crossover backtest on the provided bars."""
    strategy = SMACrossoverStrategy(fast_period=10, slow_period=30, quantity=0.01)
    settings = BacktestSettings(initial_capital=INITIAL_CAPITAL)
    engine = BacktestEngine(strategy, bars, settings)
    return engine.run()


def main() -> None:
    console.print(
        Panel(
            "[bold cyan]QTS Demo 5: Live Binance Market Data[/bold cyan]",
            style="bold",
            expand=False,
        )
    )

    # -- Fetch live data --
    console.print(f"\n[bold]Fetching {SYMBOL} {INTERVAL} bars from Binance...[/bold]")

    try:
        bars = asyncio.run(fetch_bars())
    except Exception as exc:
        console.print(f"\n[bold red]Failed to fetch data from Binance:[/bold red] {exc}")
        console.print("[dim]Check your internet connection and try again.[/dim]")
        sys.exit(1)

    if not bars:
        console.print("[bold red]No bars returned from Binance.[/bold red]")
        sys.exit(1)

    console.print(f"  [dim]Fetched {len(bars)} bars | {bars[0].timestamp.strftime('%H:%M')} - {bars[-1].timestamp.strftime('%H:%M')} UTC[/dim]\n")

    # -- Price table --
    console.print(f"[bold]Recent Market Data (last {DISPLAY_BARS} bars)[/bold]")
    console.print(build_price_table(bars))

    # -- Indicators --
    console.print()
    console.print("[bold]Indicators on Live Data[/bold]")

    indicators = compute_indicators(bars)

    rsi_val = indicators["rsi"]
    rsi_style = "red" if rsi_val > 70 else ("green" if rsi_val < 30 else "yellow")

    ind_lines = [
        f"  RSI(14):        [{rsi_style}]{rsi_val:.1f}[/{rsi_style}]",
        f"  MACD:           histogram={indicators['macd_histogram']:.2f}, line={indicators['macd_line']:.2f}, signal={indicators['macd_signal']:.2f}",
        f"  BB Position:    {indicators['bb_position']:.2f}",
    ]
    console.print(
        Panel(
            "\n".join(ind_lines),
            style="cyan",
            expand=False,
        )
    )

    # -- Quick backtest --
    console.print()
    console.print("[bold]Quick Backtest on Live Data (SMA Crossover)[/bold]")

    result = run_backtest(bars)

    bt_table = Table(show_header=True, header_style="bold magenta")
    bt_table.add_column("Metric", style="bold", min_width=18)
    bt_table.add_column("Value", justify="right", min_width=14)

    bt_table.add_row("Total Return", fmt_pct(result.total_return))
    bt_table.add_row("Sharpe Ratio", fmt_ratio(result.sharpe_ratio))
    bt_table.add_row(
        "Max Drawdown",
        Text(f"-{result.max_drawdown * 100:.2f}%", style="red"),
    )
    bt_table.add_row("# Trades", Text(str(len(result.trades)), style="bold"))
    bt_table.add_row(
        "Win Rate",
        Text(f"{result.win_rate * 100:.1f}%", style="bold"),
    )

    console.print(bt_table)

    # -- Summary --
    console.print("\n[bold green]Live data pipeline verified[/bold green]")


if __name__ == "__main__":
    main()
