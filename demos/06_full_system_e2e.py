#!/usr/bin/env python3
"""QTS Demo 6: Full System End-to-End Demonstration.

The "crown jewel" demo — exercises every major subsystem in sequence:
data -> signals -> sentiment -> strategy -> backtest -> walk-forward ->
risk -> orders -> attribution -> monitoring.

Run:
    PYTHONPATH=src .venv/bin/python demos/06_full_system_e2e.py
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import UTC, datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

SYMBOL = "BTCUSDT"
N_BARS = 500
SEED = 42
INITIAL_CAPITAL = 100_000.0
N_STEPS = 11

# ── Tracking ────────────────────────────────────────────────────────────────

subsystem_results: list[tuple[str, bool, float]] = []  # (name, passed, elapsed_s)


def step_header(step: int, title: str) -> None:
    dots = "." * (52 - len(title) - len(f"[Step {step}/{N_STEPS}] "))
    console.print(f"\n[bold cyan][Step {step}/{N_STEPS}][/bold cyan] {title} {dots}", end=" ")


def step_pass(elapsed: float) -> None:
    console.print(f"[bold green]PASS[/bold green] [dim]({elapsed:.2f}s)[/dim]")


def step_fail(elapsed: float, reason: str) -> None:
    console.print(f"[bold red]FAIL[/bold red] [dim]({elapsed:.2f}s)[/dim]")
    console.print(f"    [red]{reason}[/red]")


def run_step(step: int, title: str, func):
    """Run a step function, catching exceptions and tracking results."""
    step_header(step, title)
    t0 = time.perf_counter()
    try:
        result = func()
        elapsed = time.perf_counter() - t0
        step_pass(elapsed)
        subsystem_results.append((title, True, elapsed))
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        step_fail(elapsed, str(exc))
        subsystem_results.append((title, False, elapsed))
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Configuration
# ═══════════════════════════════════════════════════════════════════════════

def step_config():
    from qts.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    risk = settings.risk
    strat = settings.strategy

    console.print(f"    Risk: max_drawdown={risk.max_daily_drawdown_pct:.0%}, "
                  f"max_position={risk.max_position_size_pct:.0%}, "
                  f"max_positions={risk.max_open_positions}")
    console.print(f"    Strategy: entry={strat.entry_threshold}, "
                  f"exit={strat.exit_threshold}, "
                  f"max_hold={strat.max_hold_bars} bars")
    console.print(f"    Env: {settings.qts_env} | Dry-run: {settings.qts_dry_run}")
    return settings


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Market Data
# ═══════════════════════════════════════════════════════════════════════════

def step_data():
    from qts.data.market.mock_adapter import MockBarAdapter

    adapter = MockBarAdapter(seed=SEED, n_bars=N_BARS, initial_price=50_000.0)
    bars = asyncio.run(adapter.get_historical_bars(SYMBOL, start="", end=""))

    first_ts = bars[0].timestamp.strftime("%Y-%m-%d")
    last_ts = bars[-1].timestamp.strftime("%Y-%m-%d")
    console.print(f"    {len(bars)} bars loaded ({SYMBOL}, {first_ts} to {last_ts})")
    console.print(f"    Price range: ${min(b.low for b in bars):,.2f} - "
                  f"${max(b.high for b in bars):,.2f}")
    return bars


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Signal Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def step_signals(bars):
    from qts.signals.pipeline import SignalPipeline

    pipeline = SignalPipeline(symbol=SYMBOL)
    snapshots = []
    for i in range(30, len(bars)):
        snapshot = pipeline.compute(bars[: i + 1])
        if snapshot:
            snapshots.append(snapshot)

    latest = snapshots[-1]
    console.print(f"    {len(snapshots)} snapshots computed "
                  f"({len(bars) - len(snapshots)} warm-up bars skipped)")
    console.print(f"    Latest: RSI={latest.rsi:.1f}, MACD={latest.macd_histogram:.4f}, "
                  f"BB_pos={latest.bb_position:.2f}, "
                  f"Regime={latest.vol_regime.value}")
    return snapshots


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Sentiment Analysis
# ═══════════════════════════════════════════════════════════════════════════

def step_sentiment():
    from qts.nlp.fusion import SentimentFusion
    from qts.nlp.vader import VaderAnalyzer

    headlines = [
        "Bitcoin surges past $50K as institutional adoption grows",
        "Crypto market faces regulatory headwinds in EU",
        "BTC mining difficulty hits all-time high",
        "Major bank announces crypto custody service",
        "Whale wallet moves 10,000 BTC to exchange",
        "Layer 2 scaling solution sees record TVL growth",
    ]

    analyzer = VaderAnalyzer()
    results = analyzer.analyze(headlines)

    avg_compound = sum(
        r.score * (1 if r.label == "POSITIVE" else -1 if r.label == "NEGATIVE" else 0)
        for r in results
    ) / len(results)

    pos_count = sum(1 for r in results if r.label == "POSITIVE")
    neg_count = sum(1 for r in results if r.label == "NEGATIVE")
    neu_count = sum(1 for r in results if r.label == "NEUTRAL")

    bias = "bullish" if avg_compound > 0.01 else ("bearish" if avg_compound < -0.01 else "neutral")
    console.print(f"    VADER on {len(headlines)} headlines -> "
                  f"{pos_count} positive, {neg_count} negative, {neu_count} neutral")
    console.print(f"    Avg directional score: {avg_compound:+.3f} (slightly {bias})")

    # Fuse scores (use avg as news, 0.0 as social/geo placeholders)
    fusion = SentimentFusion()
    fused = fusion.fuse(news_score=avg_compound, social_score=0.05, geopolitical_score=0.0)
    console.print(f"    Fused sentiment: {fused:+.3f}")
    return fused


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Strategy Backtests
# ═══════════════════════════════════════════════════════════════════════════

def step_backtests(bars, settings):
    from qts.simulation.backtest import BacktestEngine, BacktestSettings
    from qts.strategies.mean_reversion import MeanReversionStrategy
    from qts.strategies.momentum import MomentumStrategy
    from qts.strategies.sma_crossover import SMACrossoverStrategy

    bt_settings = BacktestSettings(initial_capital=INITIAL_CAPITAL)
    params = settings.strategy
    risk_limits = settings.risk

    strategies = {
        "SMA Cross": SMACrossoverStrategy(fast_period=10, slow_period=30, quantity=1.0),
        "Momentum": MomentumStrategy(params, risk_limits, portfolio_value=INITIAL_CAPITAL),
        "Mean Rev": MeanReversionStrategy(portfolio_value=INITIAL_CAPITAL),
    }

    names = []
    results = []
    for name, strategy in strategies.items():
        engine = BacktestEngine(strategy, bars, bt_settings)
        result = engine.run()
        names.append(name)
        results.append(result)

    # Build comparison table
    table = Table(show_header=True, header_style="bold cyan", border_style="dim", padding=(0, 1))
    table.add_column("", style="bold", min_width=16)
    for name in names:
        table.add_column(name, justify="right", min_width=10)

    def _pct(v: float) -> str:
        style = "green" if v >= 0 else "red"
        return f"[{style}]{v * 100:+.2f}%[/{style}]"

    def _ratio(v: float) -> str:
        style = "green" if v > 0 else ("red" if v < 0 else "dim")
        return f"[{style}]{v:.2f}[/{style}]"

    table.add_row("Sharpe Ratio", *[_ratio(r.sharpe_ratio) for r in results])
    table.add_row("Total Return", *[_pct(r.total_return) for r in results])
    table.add_row("Max Drawdown", *[f"[red]-{r.max_drawdown * 100:.2f}%[/red]" for r in results])
    table.add_row("Win Rate", *[f"{r.win_rate * 100:.1f}%" for r in results])
    table.add_row("Profit Factor", *[_ratio(r.profit_factor) for r in results])
    table.add_row("# Trades", *[str(len(r.trades)) for r in results])

    console.print()
    console.print(table)

    best_idx = max(range(len(results)), key=lambda i: results[i].sharpe_ratio)
    best_name = names[best_idx]
    best_sharpe = results[best_idx].sharpe_ratio
    console.print(f"    [bold yellow]Best: {best_name} "
                  f"(Sharpe {best_sharpe:.2f})[/bold yellow]")

    return names, results, best_idx


# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Walk-Forward Validation
# ═══════════════════════════════════════════════════════════════════════════

def step_walk_forward(bars, best_idx, settings):
    from qts.simulation.backtest import BacktestSettings
    from qts.simulation.walk_forward import (
        WalkForwardConfig,
        WalkForwardEngine,
        monte_carlo_permutation_test,
    )
    from qts.strategies.mean_reversion import MeanReversionStrategy
    from qts.strategies.momentum import MomentumStrategy
    from qts.strategies.sma_crossover import SMACrossoverStrategy

    params = settings.strategy
    risk_limits = settings.risk

    factories = [
        lambda: SMACrossoverStrategy(fast_period=10, slow_period=30, quantity=1.0),
        lambda: MomentumStrategy(params, risk_limits, portfolio_value=INITIAL_CAPITAL),
        lambda: MeanReversionStrategy(portfolio_value=INITIAL_CAPITAL),
    ]

    factory = factories[best_idx]
    bt_settings = BacktestSettings(initial_capital=INITIAL_CAPITAL)
    wf_config = WalkForwardConfig(n_windows=5, min_bars_per_window=50)

    engine = WalkForwardEngine(factory, bt_settings, wf_config)
    wf_result = engine.run(bars)

    console.print(f"    {wf_config.n_windows} windows, "
                  f"aggregate Sharpe: {wf_result.aggregate_test_sharpe:.2f}")
    console.print(f"    Per-window Sharpes: "
                  f"{', '.join(f'{s:.2f}' for s in wf_result.per_window_test_sharpes)}")

    # Monte Carlo on the best backtest's trade returns
    # Grab returns from the test windows
    all_returns = []
    for w in wf_result.windows:
        if w.test_result and w.test_result.trades:
            all_returns.extend(t.pnl_pct for t in w.test_result.trades)

    if all_returns:
        p_value = monte_carlo_permutation_test(all_returns, n_permutations=500, seed=SEED)
        sig = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
        style = "green" if p_value < 0.05 else "red"
        console.print(f"    Monte Carlo p-value: {p_value:.3f} -> [{style}]{sig}[/{style}]")
    else:
        console.print("    [dim]No trades for Monte Carlo test[/dim]")

    return wf_result


# ═══════════════════════════════════════════════════════════════════════════
# Step 7: Risk Management
# ═══════════════════════════════════════════════════════════════════════════

def step_risk(settings):
    from qts.execution.risk import RiskManager
    from qts.models.base import Position, TradeDirection

    risk_mgr = RiskManager(settings.risk)
    portfolio_value = INITIAL_CAPITAL

    # Position size check — 1% of portfolio
    proposed = portfolio_value * 0.01
    pos_ok = risk_mgr.check_position_size(proposed, portfolio_value)
    limit_pct = settings.risk.max_position_size_pct * 100
    console.print(f"    Position size check: "
                  f"{'[green]PASS[/green]' if pos_ok else '[red]FAIL[/red]'} "
                  f"(1% < {limit_pct:.0f}% limit)")

    # Drawdown check — 0.5% daily loss
    dd_ok = risk_mgr.check_daily_drawdown(-500.0, portfolio_value)
    dd_limit = settings.risk.max_daily_drawdown_pct * 100
    console.print(f"    Drawdown check:      "
                  f"{'[green]PASS[/green]' if dd_ok else '[red]FAIL[/red]'} "
                  f"(0.5% < {dd_limit:.0f}% limit)")

    # Max positions check — 1 open position
    fake_pos = [Position(
        symbol=SYMBOL,
        direction=TradeDirection.LONG,
        entry_price=50_000.0,
        quantity=0.1,
        entry_time=datetime.now(UTC),
    )]
    max_ok = risk_mgr.check_max_positions(fake_pos)
    console.print(f"    Max positions check: "
                  f"{'[green]PASS[/green]' if max_ok else '[red]FAIL[/red]'} "
                  f"(1 < {settings.risk.max_open_positions} limit)")

    # Circuit breaker status
    halted = risk_mgr.is_halted()
    console.print(f"    Circuit breaker:     "
                  f"{'[red]TRIPPED[/red]' if halted else '[green]CLEAR[/green]'}")

    return risk_mgr


# ═══════════════════════════════════════════════════════════════════════════
# Step 8: Order Flow
# ═══════════════════════════════════════════════════════════════════════════

def step_orders():
    from qts.execution.order_manager import OrderManager
    from qts.models.base import Order, OrderSide, OrderType

    om = OrderManager(paper_mode=True, default_fill_price=50_123.45)

    order = Order(
        order_id=str(uuid.uuid4()),
        symbol=SYMBOL,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=0.1,
        timestamp=datetime.now(UTC),
    )

    oid = om.submit_order(order)
    status = om.get_order_status(oid)
    fills = om.get_fills(oid)

    console.print(f"    Order submitted: {order.side} {order.quantity} "
                  f"{order.symbol} @ {order.order_type}")
    if fills:
        fill = fills[0]
        console.print(f"    Fill received: ${fill.price:,.2f}, "
                      f"commission: ${fill.commission:.2f}")
    console.print(f"    Status: [green]{status}[/green]")

    return om


# ═══════════════════════════════════════════════════════════════════════════
# Step 9: Trade Attribution
# ═══════════════════════════════════════════════════════════════════════════

def step_attribution(backtest_results, best_idx):
    from qts.analytics.attribution import AttributionEngine

    best_result = backtest_results[best_idx]
    trades = best_result.trades

    if not trades:
        console.print("    [dim]No trades to attribute (strategy produced 0 trades)[/dim]")
        return None

    engine = AttributionEngine()
    # Run attribution with empty snapshot dict (no exit snapshots available)
    report = engine.run_attribution(trades, snapshots={})

    wins = report.winning_trades
    losses = report.losing_trades
    total = report.total_trades
    win_rate = wins / total * 100 if total > 0 else 0.0

    console.print(f"    {total} trades analysed")
    console.print(f"    Win rate: {win_rate:.1f}% ({wins}W / {losses}L)")

    if report.failure_mode_summary:
        # Find top failure mode
        top_mode = max(report.failure_mode_summary.items(), key=lambda x: x[1].count)
        mode_name = top_mode[0].value
        mode_count = top_mode[1].count
        mode_avg_pnl = top_mode[1].avg_pnl
        console.print(f"    Top failure mode: [yellow]{mode_name}[/yellow] "
                      f"({mode_count} trades, avg PnL ${mode_avg_pnl:+,.2f})")
    else:
        console.print("    [dim]No failure modes classified[/dim]")

    # Profit factor from backtest
    pf = best_result.profit_factor
    console.print(f"    Profit factor: {pf:.2f}")

    return report


# ═══════════════════════════════════════════════════════════════════════════
# Step 10: Health & Monitoring
# ═══════════════════════════════════════════════════════════════════════════

def step_health():
    from qts.monitoring.alerts import AlertManager
    from qts.monitoring.health import HealthChecker, HealthStatus

    # Health checks (no external deps configured — expect degraded/unhealthy
    # for DB/Redis, but the check itself should not crash)
    checker = HealthChecker()
    health_results = checker.check_all()

    for h in health_results:
        style = {
            HealthStatus.HEALTHY: "green",
            HealthStatus.DEGRADED: "yellow",
            HealthStatus.UNHEALTHY: "red",
        }.get(h.status, "dim")
        console.print(f"    {h.component:<15} [{style}]{h.status}[/{style}] "
                      f"[dim]{h.message}[/dim]")

    # Alert manager — run checks with safe values (should produce no alerts)
    alert_mgr = AlertManager()
    alert_mgr.check_drawdown(current_drawdown=0.001, limit=0.02)
    alert_mgr.check_position_concentration(
        positions={"BTCUSDT": 5_000.0, "ETHUSDT": 3_000.0}, limit=0.10
    )
    active = alert_mgr.get_active_alerts()
    console.print(f"    Active alerts: [bold]{'[green]0[/green]' if not active else f'[red]{len(active)}[/red]'}")

    return health_results


# ═══════════════════════════════════════════════════════════════════════════
# Step 11: LLM / Ollama Connectivity (optional)
# ═══════════════════════════════════════════════════════════════════════════

def step_llm_check(settings):
    llm = settings.llm
    console.print(f"    Backend: {llm.backend}")
    if llm.backend == "ollama":
        console.print(f"    Model: {llm.ollama_model}")
        console.print(f"    Base URL: {llm.ollama_base_url}")
        # Quick connectivity check — don't block if Ollama isn't running
        try:
            import urllib.request

            req = urllib.request.Request(f"{llm.ollama_base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                console.print(f"    Ollama status: [green]REACHABLE[/green] ({resp.status})")
        except Exception:
            console.print("    Ollama status: [yellow]UNREACHABLE[/yellow] (not required)")
    elif llm.backend == "anthropic":
        configured = "yes" if llm.configured else "no"
        console.print(f"    Configured: {configured}")
    else:
        console.print(f"    [dim]Unknown backend: {llm.backend}[/dim]")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    total_start = time.perf_counter()

    # Banner
    console.print()
    banner = (
        "[bold bright_cyan]"
        "================================================================\n"
        "   QTS Full System End-to-End Demonstration\n"
        "   Quantitative Trading System v0.1.0\n"
        "================================================================"
        "[/bold bright_cyan]"
    )
    console.print(Panel(banner, border_style="bright_cyan", padding=(0, 2)))

    # Step 1
    settings = run_step(1, "Configuration", step_config)

    # Step 2
    bars = run_step(2, "Market Data", step_data)

    # Step 3
    snapshots = None
    if bars:
        snapshots = run_step(3, "Signal Pipeline", lambda: step_signals(bars))

    # Step 4
    fused_sentiment = run_step(4, "Sentiment Analysis", step_sentiment)

    # Step 5
    bt_data = None
    if bars and settings:
        bt_data = run_step(5, "Strategy Backtests", lambda: step_backtests(bars, settings))

    # Step 6
    if bars and bt_data and settings:
        names, results, best_idx = bt_data
        run_step(
            6,
            "Walk-Forward Validation",
            lambda: step_walk_forward(bars, best_idx, settings),
        )

    # Step 7
    if settings:
        run_step(7, "Risk Management", lambda: step_risk(settings))

    # Step 8
    run_step(8, "Order Flow", step_orders)

    # Step 9
    if bt_data:
        names, results, best_idx = bt_data
        run_step(9, "Trade Attribution", lambda: step_attribution(results, best_idx))

    # Step 10
    run_step(10, "Health & Monitoring", step_health)

    # Step 11
    if settings:
        run_step(11, "LLM Backend", lambda: step_llm_check(settings))

    # ── Final Summary ──────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_start
    console.print()
    console.print(
        Panel(
            "[bold bright_white]SYSTEM VERIFICATION SUMMARY[/bold bright_white]",
            border_style="bright_cyan",
            padding=(0, 2),
        )
    )

    summary_table = Table(
        show_header=True,
        header_style="bold cyan",
        border_style="bright_cyan",
        padding=(0, 2),
    )
    summary_table.add_column("Subsystem", style="bold", min_width=24)
    summary_table.add_column("Status", justify="center", min_width=8)
    summary_table.add_column("Time", justify="right", min_width=8)

    pass_count = 0
    fail_count = 0
    for name, passed, elapsed in subsystem_results:
        status = "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"
        if passed:
            pass_count += 1
        else:
            fail_count += 1
        summary_table.add_row(name, status, f"{elapsed:.2f}s")

    console.print(summary_table)

    console.print()
    total_sub = len(subsystem_results)
    if fail_count == 0:
        console.print(
            f"[bold green]All {total_sub} subsystems verified. "
            f"QTS is operational.[/bold green] "
            f"[dim](total: {total_elapsed:.2f}s)[/dim]"
        )
    else:
        console.print(
            f"[bold yellow]{pass_count}/{total_sub} subsystems passed, "
            f"{fail_count} failed.[/bold yellow] "
            f"[dim](total: {total_elapsed:.2f}s)[/dim]"
        )
    console.print()


if __name__ == "__main__":
    main()
