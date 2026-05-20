"""QTS Demo 1: Signals & Technical Indicators.

Generates synthetic OHLCV data via MockBarAdapter, computes all individual
technical indicators, runs the full SignalPipeline, and displays results
using rich tables, panels, and colour-coded output.

Run:
    PYTHONPATH=src .venv/bin/python demos/01_signals_and_indicators.py
"""

from __future__ import annotations

import asyncio

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from qts.config import SentimentFusionWeights, SignalWeights, StrategyParams
from qts.data.market.mock_adapter import MockBarAdapter
from qts.models.base import Bar, VolLevel
from qts.signals.alpha import combined_alpha
from qts.signals.indicators import (
    compute_atr,
    compute_bb_position,
    compute_bollinger_bands,
    compute_macd,
    compute_momentum,
    compute_rsi,
    normalise_rsi,
)
from qts.signals.pipeline import SignalPipeline
from qts.signals.regime import RegimeDetector

console = Console()

SYMBOL = "BTCUSDT"
N_BARS = 200
SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────────────


def _colour_rsi(val: float) -> str:
    """Return colour tag based on RSI level."""
    if np.isnan(val):
        return "dim"
    if val >= 70:
        return "red"
    if val <= 30:
        return "green"
    return "yellow"


def _colour_value(val: float, *, bullish_positive: bool = True) -> str:
    """Green for bullish, red for bearish, yellow for near-zero."""
    if np.isnan(val):
        return "dim"
    if abs(val) < 1e-6:
        return "yellow"
    if bullish_positive:
        return "green" if val > 0 else "red"
    return "red" if val > 0 else "green"


def _colour_regime(regime: VolLevel) -> str:
    return "red bold" if regime == VolLevel.HIGH else "green bold"


def _fmt(val: float, decimals: int = 4) -> str:
    if np.isnan(val):
        return "NaN"
    return f"{val:.{decimals}f}"


# ── Data fetching ────────────────────────────────────────────────────────────


async def _fetch_bars() -> list[Bar]:
    adapter = MockBarAdapter(seed=SEED, n_bars=N_BARS)
    bars = await adapter.get_historical_bars(SYMBOL, "2024-01-01", "2024-12-31")
    await adapter.close()
    return bars


# ── Display sections ─────────────────────────────────────────────────────────


def show_header(n_bars: int) -> None:
    console.print()
    console.print(
        Panel(
            Text("QTS Demo 1: Signals & Technical Indicators", justify="center", style="bold cyan"),
            border_style="bright_cyan",
            padding=(1, 4),
        )
    )
    console.print(
        f"  [bold]Generated [cyan]{n_bars}[/cyan] synthetic bars for [yellow]{SYMBOL}[/yellow][/bold]"
    )
    console.print()


def show_price_table(bars: list[Bar], last_n: int = 10) -> None:
    console.print(Rule("[bold]Last 10 Bars[/bold]", style="bright_blue"))
    console.print()

    table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("Timestamp", style="dim")
    table.add_column("Open", justify="right")
    table.add_column("High", justify="right", style="green")
    table.add_column("Low", justify="right", style="red")
    table.add_column("Close", justify="right", style="bold")
    table.add_column("Volume", justify="right", style="cyan")

    for bar in bars[-last_n:]:
        ts = bar.timestamp.strftime("%Y-%m-%d %H:%M")
        close_style = "green" if bar.close >= bar.open else "red"
        table.add_row(
            ts,
            f"{bar.open:,.2f}",
            f"{bar.high:,.2f}",
            f"{bar.low:,.2f}",
            f"[{close_style}]{bar.close:,.2f}[/{close_style}]",
            f"{bar.volume:,.1f}",
        )

    console.print(table)
    console.print()


def show_indicators(bars: list[Bar], last_n: int = 20) -> None:
    console.print(Rule("[bold]Technical Indicators (last 20 values)[/bold]", style="bright_blue"))
    console.print()

    closes = np.array([b.close for b in bars], dtype=np.float64)
    highs = np.array([b.high for b in bars], dtype=np.float64)
    lows = np.array([b.low for b in bars], dtype=np.float64)

    rsi = compute_rsi(closes)
    macd_line, macd_signal, macd_hist = compute_macd(closes)
    bb_upper, bb_middle, bb_lower = compute_bollinger_bands(closes)
    atr = compute_atr(highs, lows, closes)
    momentum = compute_momentum(closes)

    table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("#", style="dim", justify="right")
    table.add_column("RSI", justify="right")
    table.add_column("MACD Line", justify="right")
    table.add_column("MACD Signal", justify="right")
    table.add_column("MACD Hist", justify="right")
    table.add_column("BB Upper", justify="right")
    table.add_column("BB Middle", justify="right")
    table.add_column("BB Lower", justify="right")
    table.add_column("ATR", justify="right")
    table.add_column("Momentum", justify="right")

    start = len(bars) - last_n
    for i in range(start, len(bars)):
        rsi_c = _colour_rsi(rsi[i])
        macd_c = _colour_value(macd_hist[i])
        mom_c = _colour_value(momentum[i])

        table.add_row(
            str(i),
            f"[{rsi_c}]{_fmt(rsi[i], 1)}[/{rsi_c}]",
            f"{_fmt(macd_line[i], 2)}",
            f"{_fmt(macd_signal[i], 2)}",
            f"[{macd_c}]{_fmt(macd_hist[i], 4)}[/{macd_c}]",
            f"{_fmt(bb_upper[i], 2)}",
            f"{_fmt(bb_middle[i], 2)}",
            f"{_fmt(bb_lower[i], 2)}",
            f"{_fmt(atr[i], 2)}",
            f"[{mom_c}]{_fmt(momentum[i], 4)}[/{mom_c}]",
        )

    console.print(table)
    console.print()


def show_pipeline(bars: list[Bar]) -> None:
    console.print(Rule("[bold]Signal Pipeline Output[/bold]", style="bright_blue"))
    console.print()

    pipeline = SignalPipeline(symbol=SYMBOL)
    snapshot = pipeline.compute(bars)

    if snapshot is None:
        console.print("[red bold]Pipeline returned None — insufficient bars.[/red bold]")
        return

    # Build a descriptive panel
    def _signal_line(label: str, value: float, fmt: str, interpretation: str, colour: str) -> str:
        return f"  [bold]{label}:[/bold]  [{colour}]{value:{fmt}}[/{colour}]  [dim]({interpretation})[/dim]"

    # RSI interpretation
    rsi_interp = "overbought" if snapshot.rsi >= 70 else ("oversold" if snapshot.rsi <= 30 else "neutral")
    rsi_colour = _colour_rsi(snapshot.rsi)

    # MACD histogram interpretation
    macd_interp = "bullish" if snapshot.macd_histogram > 0 else ("bearish" if snapshot.macd_histogram < 0 else "flat")
    macd_colour = _colour_value(snapshot.macd_histogram)

    # BB position interpretation
    if snapshot.bb_position > 0.8:
        bb_interp = "upper range"
    elif snapshot.bb_position < 0.2:
        bb_interp = "lower range"
    else:
        bb_interp = "mid range"
    bb_colour = "red" if snapshot.bb_position > 0.8 else ("green" if snapshot.bb_position < 0.2 else "yellow")

    # Momentum interpretation
    mom_interp = "positive" if snapshot.momentum_5 > 0 else ("negative" if snapshot.momentum_5 < 0 else "flat")
    mom_colour = _colour_value(snapshot.momentum_5)

    # Regime
    regime_colour = _colour_regime(snapshot.vol_regime)

    lines = [
        _signal_line("RSI", snapshot.rsi, ".1f", rsi_interp, rsi_colour),
        _signal_line("MACD Histogram", snapshot.macd_histogram, ".4f", macd_interp, macd_colour),
        _signal_line("MACD Line", snapshot.macd_line, ".4f", "", "dim"),
        _signal_line("MACD Signal", snapshot.macd_signal, ".4f", "", "dim"),
        _signal_line("BB Position", snapshot.bb_position, ".4f", bb_interp, bb_colour),
        f"  [bold]BB Bounds:[/bold]  [dim]{snapshot.bb_lower:,.2f} — {snapshot.bb_upper:,.2f}[/dim]",
        _signal_line("ATR", snapshot.atr, ".2f", "", "cyan"),
        _signal_line("Momentum (5)", snapshot.momentum_5, ".4f", mom_interp, mom_colour),
        "",
        f"  [bold]Vol Regime:[/bold]  [{regime_colour}]{snapshot.vol_regime.value}[/{regime_colour}]  [dim](confidence: {snapshot.vol_regime_confidence:.2f})[/dim]",
    ]

    # Compute combined alpha with default params
    params = StrategyParams(
        version="demo-v1",
        weights=SignalWeights(w_rsi=0.25, w_macd=0.25, w_bb=0.15, w_mom=0.20, w_sentiment=0.15),
        entry_threshold=0.3,
        exit_threshold=0.1,
        max_hold_bars=50,
        sentiment_fusion_weights=SentimentFusionWeights(news=0.5, social=0.3, geopolitical=0.2),
    )
    alpha = combined_alpha(snapshot, params)
    alpha_colour = _colour_value(alpha)
    alpha_interp = "bullish" if alpha > 0.1 else ("bearish" if alpha < -0.1 else "neutral")
    lines.append(
        f"  [bold]Combined Alpha:[/bold]  [{alpha_colour}]{alpha:.4f}[/{alpha_colour}]  [dim]({alpha_interp})[/dim]"
    )

    content = "\n".join(lines)
    console.print(
        Panel(content, title="[bold cyan]SignalSnapshot[/bold cyan]", border_style="bright_cyan", padding=(1, 2))
    )
    console.print()


def show_regime(bars: list[Bar], last_n: int = 20) -> None:
    console.print(Rule("[bold]Regime Detection (HMM)[/bold]", style="bright_blue"))
    console.print()

    closes = np.array([b.close for b in bars], dtype=np.float64)
    highs = np.array([b.high for b in bars], dtype=np.float64)
    lows = np.array([b.low for b in bars], dtype=np.float64)

    atr_arr = compute_atr(highs, lows, closes)

    detector = RegimeDetector(random_state=SEED)
    detector.fit(atr_arr)

    # Get regime for sliding windows of bars
    table = Table(show_header=True, header_style="bold magenta", border_style="dim")
    table.add_column("Bar #", style="dim", justify="right")
    table.add_column("Timestamp", style="dim")
    table.add_column("Close", justify="right")
    table.add_column("ATR", justify="right", style="cyan")
    table.add_column("Regime", justify="center")
    table.add_column("Confidence", justify="right")

    start = len(bars) - last_n
    for i in range(start, len(bars)):
        # Predict using ATR history up to bar i
        atr_slice = atr_arr[: i + 1]
        regime, confidence = detector.predict(atr_slice)

        regime_colour = _colour_regime(regime)
        conf_colour = "green" if confidence > 0.8 else ("yellow" if confidence > 0.6 else "red")

        table.add_row(
            str(i),
            bars[i].timestamp.strftime("%Y-%m-%d %H:%M"),
            f"{bars[i].close:,.2f}",
            f"{_fmt(atr_arr[i], 2)}",
            f"[{regime_colour}]{regime.value}[/{regime_colour}]",
            f"[{conf_colour}]{confidence:.2%}[/{conf_colour}]",
        )

    console.print(table)
    console.print()


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    bars = asyncio.run(_fetch_bars())

    show_header(len(bars))
    show_price_table(bars)
    show_indicators(bars)
    show_pipeline(bars)
    show_regime(bars)

    console.print("[bold green]All indicators computed successfully.[/bold green]")
    console.print()


if __name__ == "__main__":
    main()
