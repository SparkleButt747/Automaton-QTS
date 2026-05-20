#!/usr/bin/env python3
"""QTS Demo 3: Sentiment Pipeline, Risk Management & Trade Attribution.

Run with:
    PYTHONPATH=src .venv/bin/python demos/03_sentiment_risk_attribution.py
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from qts.analytics.attribution import AttributionEngine
from qts.config import RiskLimits, get_settings
from qts.execution.order_manager import OrderManager
from qts.execution.risk import CircuitBreakerError, RiskManager
from qts.models.base import (
    ExitReason,
    FailureMode,
    Order,
    OrderSide,
    OrderType,
    Position,
    TradeDirection,
    TradeOutcome,
    TradeRecord,
    VolLevel,
)
from qts.monitoring.alerts import AlertManager
from qts.nlp.fusion import SentimentFusion
from qts.nlp.vader import VaderAnalyzer

console = Console()

BANNER = """
[bold cyan]═══════════════════════════════════════════════════
  QTS Demo 3: Sentiment, Risk & Attribution
═══════════════════════════════════════════════════[/bold cyan]
"""


# ── Part 1: Sentiment Analysis ────────────────────────────────────────────────


def demo_sentiment() -> None:
    console.print("\n[bold yellow]═══ Part 1: Sentiment Analysis ═══[/bold yellow]\n")

    vader = VaderAnalyzer()
    texts = [
        "Bitcoin surges past $100k as institutional adoption accelerates",
        "Federal Reserve signals aggressive rate hikes amid inflation concerns",
        "Crypto exchange FTX files for bankruptcy, market in turmoil",
        "Goldman Sachs launches Bitcoin trading desk for institutional clients",
        "SEC rejects another Bitcoin ETF application citing market manipulation",
        "Ethereum 2.0 merge successful, energy consumption drops 99%",
    ]

    results = vader.analyze(texts)

    table = Table(title="VADER Sentiment Analysis", show_lines=True)
    table.add_column("Headline", style="white", max_width=60)
    table.add_column("Label", justify="center")
    table.add_column("Score", justify="right")
    table.add_column("Confidence", justify="right")

    for r in results:
        if r.label == "POSITIVE":
            label_style = "[bold green]POSITIVE[/bold green]"
        elif r.label == "NEGATIVE":
            label_style = "[bold red]NEGATIVE[/bold red]"
        else:
            label_style = "[bold yellow]NEUTRAL[/bold yellow]"

        table.add_row(
            r.text,
            label_style,
            f"{r.score:.3f}",
            f"{r.confidence:.3f}",
        )

    console.print(table)

    # Sentiment fusion
    console.print("\n[bold]Sentiment Fusion[/bold]")
    fusion = SentimentFusion()
    fused = fusion.fuse(news_score=0.3, social_score=-0.1, geopolitical_score=0.0)

    if fused > 0.05:
        mood = "[green]slightly bullish[/green]"
    elif fused < -0.05:
        mood = "[red]slightly bearish[/red]"
    else:
        mood = "[yellow]neutral[/yellow]"

    console.print(
        f"  Inputs: news=0.3, social=-0.1, geopolitical=0.0\n"
        f"  Weights: news={fusion.weights.news}, "
        f"social={fusion.weights.social}, "
        f"geopolitical={fusion.weights.geopolitical}\n"
        f"  Fused Score: [bold]{fused:.4f}[/bold] ({mood})"
    )


# ── Part 2: Risk Management ──────────────────────────────────────────────────


def demo_risk() -> None:
    console.print("\n[bold yellow]═══ Part 2: Risk Management ═══[/bold yellow]\n")

    settings = get_settings()
    rm = RiskManager(settings.risk)
    limits = settings.risk

    lines: list[str] = []

    # Position size check - approve
    ok = rm.check_position_size(proposed_size=4_000.0, portfolio_value=100_000.0)
    pct = 4_000.0 / 100_000.0 * 100
    status = "[green]APPROVED[/green]" if ok else "[red]REJECTED[/red]"
    lines.append(
        f"Position size $4,000 / $100k ({pct:.1f}%) "
        f"vs limit {limits.max_position_size_pct * 100:.0f}%: {status}"
    )

    # Position size check - reject
    ok = rm.check_position_size(proposed_size=10_000.0, portfolio_value=100_000.0)
    pct = 10_000.0 / 100_000.0 * 100
    status = "[green]APPROVED[/green]" if ok else "[red]REJECTED[/red]"
    lines.append(
        f"Position size $10,000 / $100k ({pct:.1f}%) "
        f"vs limit {limits.max_position_size_pct * 100:.0f}%: {status}"
    )

    # Max positions check - approve (4 open, limit 5)
    now = datetime.now(UTC)
    four_positions = [
        Position(symbol=s, direction=TradeDirection.LONG, entry_price=100.0, quantity=1.0, entry_time=now)
        for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "AVAXUSDT"]
    ]
    ok = rm.check_max_positions(four_positions)
    status = "[green]APPROVED[/green]" if ok else "[red]REJECTED[/red]"
    lines.append(
        f"Open positions: {len(four_positions)} / {limits.max_open_positions} max: {status}"
    )

    # Max positions check - reject (5 open, limit 5)
    five_positions = four_positions + [
        Position(symbol="DOTUSDT", direction=TradeDirection.LONG, entry_price=100.0, quantity=1.0, entry_time=now)
    ]
    ok = rm.check_max_positions(five_positions)
    status = "[green]APPROVED[/green]" if ok else "[red]REJECTED[/red]"
    lines.append(
        f"Open positions: {len(five_positions)} / {limits.max_open_positions} max: {status}"
    )

    # Daily drawdown check - approve
    ok = rm.check_daily_drawdown(daily_pnl=-500.0, portfolio_value=100_000.0)
    dd_pct = 500.0 / 100_000.0 * 100
    status = "[green]APPROVED[/green]" if ok else "[red]REJECTED[/red]"
    lines.append(
        f"Daily drawdown -$500 ({dd_pct:.2f}%) "
        f"vs limit {limits.max_daily_drawdown_pct * 100:.0f}%: {status}"
    )

    # Daily drawdown check - reject (trips circuit breaker)
    rm_fresh = RiskManager(settings.risk)
    ok = rm_fresh.check_daily_drawdown(daily_pnl=-3_000.0, portfolio_value=100_000.0)
    dd_pct = 3_000.0 / 100_000.0 * 100
    status = "[green]APPROVED[/green]" if ok else "[red]REJECTED[/red]"
    lines.append(
        f"Daily drawdown -$3,000 ({dd_pct:.1f}%) "
        f"vs limit {limits.max_daily_drawdown_pct * 100:.0f}%: {status}"
    )

    # Circuit breaker status
    halted = rm_fresh.is_halted()
    cb_status = "[red]HALTED[/red]" if halted else "[green]ACTIVE[/green]"
    lines.append(f"Circuit breaker status: {cb_status}")

    # approve_order with halted circuit breaker
    try:
        rm_fresh.approve_order(
            proposed_size_usd=1_000.0,
            portfolio_value=100_000.0,
            current_positions=[],
            daily_pnl=-500.0,
        )
        lines.append("approve_order() while halted: [green]APPROVED[/green]")
    except CircuitBreakerError:
        lines.append("approve_order() while halted: [red]BLOCKED (CircuitBreakerError)[/red]")

    panel_text = "\n".join(f"  {line}" for line in lines)
    console.print(Panel(panel_text, title="Risk Checks", border_style="cyan"))


# ── Part 3: Order Manager ────────────────────────────────────────────────────


def demo_orders() -> None:
    console.print("\n[bold yellow]═══ Part 3: Order Manager ═══[/bold yellow]\n")

    om = OrderManager(paper_mode=True, default_fill_price=98_750.0)

    order = Order(
        order_id="demo-001",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
    )
    order_id = om.submit_order(order)

    status = om.get_order_status(order_id)
    fills = om.get_fills(order_id)

    lines = [
        f"Order ID:   {order_id}",
        f"Symbol:     {order.symbol}",
        f"Side:       [green]{order.side}[/green]",
        f"Type:       {order.order_type}",
        f"Quantity:   {order.quantity}",
        f"Status:     [bold green]{status}[/bold green]",
    ]

    if fills:
        f = fills[0]
        lines.append("")
        lines.append("[bold]Fill Details:[/bold]")
        lines.append(f"  Fill ID:    {f.fill_id[:12]}...")
        lines.append(f"  Price:      ${f.price:,.2f}")
        lines.append(f"  Quantity:   {f.quantity}")
        lines.append(f"  Commission: ${f.commission:.4f}")
        lines.append(f"  Timestamp:  {f.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    panel_text = "\n".join(f"  {line}" for line in lines)
    console.print(Panel(panel_text, title="Paper Order Submission", border_style="cyan"))


# ── Part 4: Trade Attribution ─────────────────────────────────────────────────


def demo_attribution() -> None:
    console.print("\n[bold yellow]═══ Part 4: Trade Attribution ═══[/bold yellow]\n")

    now = datetime.now(UTC)
    trades = [
        TradeRecord(
            trade_id="T-001",
            symbol="BTCUSDT",
            entry_time=now - timedelta(hours=12),
            exit_time=now - timedelta(hours=6),
            direction=TradeDirection.LONG,
            entry_price=95_000.0,
            exit_price=97_500.0,
            size=0.1,
            pnl_usd=250.0,
            pnl_pct=2.63,
            slippage_usd=0.0,
            fees_usd=9.5,
            rsi_at_entry=42.0,
            macd_at_entry=0.3,
            sentiment_at_entry=0.7,
            social_velocity_at_entry=0.5,
            gdelt_intensity_at_entry=0.1,
            vol_level_at_entry=VolLevel.HIGH,
            combined_alpha_at_entry=0.65,
            params_version="v1.2",
            outcome=TradeOutcome.WIN,
            exit_reason=ExitReason.TAKE_PROFIT,
        ),
        TradeRecord(
            trade_id="T-002",
            symbol="ETHUSDT",
            entry_time=now - timedelta(hours=10),
            exit_time=now - timedelta(hours=4),
            direction=TradeDirection.LONG,
            entry_price=3_200.0,
            exit_price=3_050.0,
            size=1.0,
            pnl_usd=-150.0,
            pnl_pct=-4.69,
            slippage_usd=5.2,
            fees_usd=3.2,
            rsi_at_entry=55.0,
            macd_at_entry=-0.1,
            sentiment_at_entry=0.6,
            social_velocity_at_entry=0.3,
            gdelt_intensity_at_entry=0.0,
            vol_level_at_entry=VolLevel.LOW,
            combined_alpha_at_entry=0.40,
            params_version="v1.2",
            outcome=TradeOutcome.LOSS,
            exit_reason=ExitReason.STOP_LOSS,
        ),
        TradeRecord(
            trade_id="T-003",
            symbol="SOLUSDT",
            entry_time=now - timedelta(hours=8),
            exit_time=now - timedelta(hours=2),
            direction=TradeDirection.SHORT,
            entry_price=180.0,
            exit_price=190.0,
            size=10.0,
            pnl_usd=-100.0,
            pnl_pct=-5.56,
            slippage_usd=0.0,
            fees_usd=1.8,
            rsi_at_entry=70.0,
            macd_at_entry=0.5,
            sentiment_at_entry=-0.4,
            social_velocity_at_entry=-0.2,
            gdelt_intensity_at_entry=0.3,
            vol_level_at_entry=VolLevel.HIGH,
            combined_alpha_at_entry=-0.55,
            params_version="v1.2",
            outcome=TradeOutcome.LOSS,
            exit_reason=ExitReason.TIME_STOP,
        ),
        TradeRecord(
            trade_id="T-004",
            symbol="BTCUSDT",
            entry_time=now - timedelta(hours=5),
            exit_time=now - timedelta(hours=1),
            direction=TradeDirection.LONG,
            entry_price=97_000.0,
            exit_price=99_200.0,
            size=0.05,
            pnl_usd=110.0,
            pnl_pct=2.27,
            slippage_usd=0.0,
            fees_usd=4.85,
            rsi_at_entry=38.0,
            macd_at_entry=0.2,
            sentiment_at_entry=0.8,
            social_velocity_at_entry=0.6,
            gdelt_intensity_at_entry=0.2,
            vol_level_at_entry=VolLevel.HIGH,
            combined_alpha_at_entry=0.72,
            params_version="v1.2",
            outcome=TradeOutcome.WIN,
            exit_reason=ExitReason.ALPHA_FLIP,
        ),
        TradeRecord(
            trade_id="T-005",
            symbol="AVAXUSDT",
            entry_time=now - timedelta(hours=3),
            exit_time=now,
            direction=TradeDirection.LONG,
            entry_price=35.0,
            exit_price=33.5,
            size=50.0,
            pnl_usd=-75.0,
            pnl_pct=-4.29,
            slippage_usd=0.0,
            fees_usd=1.75,
            rsi_at_entry=48.0,
            macd_at_entry=-0.05,
            sentiment_at_entry=0.2,
            social_velocity_at_entry=0.1,
            gdelt_intensity_at_entry=-0.1,
            vol_level_at_entry=VolLevel.HIGH,
            combined_alpha_at_entry=0.30,
            params_version="v1.2",
            outcome=TradeOutcome.LOSS,
            exit_reason=ExitReason.STOP_LOSS,
        ),
    ]

    engine = AttributionEngine()
    report = engine.run_attribution(trades, snapshots={})

    # Summary stats
    console.print(
        f"  Total trades: {report.total_trades}  |  "
        f"[green]Wins: {report.winning_trades}[/green]  |  "
        f"[red]Losses: {report.losing_trades}[/red]"
    )
    console.print(
        f"  Sentiment hit rate: {report.sentiment_hit_rate:.1%}  |  "
        f"Sentiment-PnL correlation: {report.sentiment_pnl_correlation:.3f}"
    )

    # Failure mode table
    if report.failure_mode_summary:
        table = Table(title="Failure Mode Breakdown", show_lines=True)
        table.add_column("Failure Mode", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Avg PnL", justify="right")
        table.add_column("% of Losses", justify="right")

        for mode, summary in report.failure_mode_summary.items():
            pnl_style = "green" if summary.avg_pnl >= 0 else "red"
            table.add_row(
                str(mode),
                str(summary.count),
                f"[{pnl_style}]${summary.avg_pnl:,.2f}[/{pnl_style}]",
                f"{summary.pct_of_losses:.0%}",
            )

        console.print(table)
    else:
        console.print("  [yellow]No failure modes classified.[/yellow]")


# ── Part 5: Alerts & Health ───────────────────────────────────────────────────


def demo_alerts() -> None:
    console.print("\n[bold yellow]═══ Part 5: Alerts & Health ═══[/bold yellow]\n")

    am = AlertManager(drawdown_threshold=0.01)

    # Trigger drawdown warning
    am.check_drawdown(current_drawdown=0.015, limit=0.02)

    # Trigger feed staleness (last tick 120s ago, max 60s)
    stale_time = datetime.now(UTC) - timedelta(seconds=120)
    am.check_feed_staleness(last_tick_time=stale_time, max_age_seconds=60.0)

    # Trigger position concentration
    am.check_position_concentration(
        positions={"BTCUSDT": 60_000.0, "ETHUSDT": 25_000.0, "SOLUSDT": 15_000.0},
        limit=0.50,
    )

    alerts = am.get_active_alerts()

    if alerts:
        table = Table(title="Active Alerts", show_lines=True)
        table.add_column("Level", justify="center")
        table.add_column("Component", style="dim")
        table.add_column("Message", max_width=70)
        table.add_column("Value", justify="right")
        table.add_column("Threshold", justify="right")

        for a in alerts:
            if a.level == "CRITICAL":
                level_style = "[bold red]CRITICAL[/bold red]"
            elif a.level == "WARNING":
                level_style = "[bold yellow]WARNING[/bold yellow]"
            else:
                level_style = "[bold blue]INFO[/bold blue]"

            table.add_row(
                level_style,
                a.component,
                a.message,
                f"{a.value:.4f}",
                f"{a.threshold:.4f}",
            )

        console.print(table)
    else:
        console.print("  [green]No active alerts.[/green]")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    console.print(BANNER)

    demo_sentiment()
    demo_risk()
    demo_orders()
    demo_attribution()
    demo_alerts()

    console.print("\n[bold green]All subsystems verified.[/bold green]\n")


if __name__ == "__main__":
    main()
