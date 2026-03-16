"""Real-time trading monitoring dashboard using Rich."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from qts.models.base import Position, TradeRecord, VolRegime
from qts.monitoring.alerts import Alert, AlertLevel

logger = logging.getLogger(__name__)

# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class SignalValues:
    """Signal values for a single symbol displayed in the dashboard.

    Attributes:
        symbol: Ticker symbol.
        rsi: RSI(14) value.
        macd: MACD histogram value.
        sentiment: Sentiment score in [-1, 1].
        combined_alpha: Combined alpha in [-1, 1].
    """

    symbol: str
    rsi: float = float("nan")
    macd: float = float("nan")
    sentiment: float = float("nan")
    combined_alpha: float = float("nan")


@dataclass
class DashboardState:
    """All data required to render one frame of the dashboard.

    Attributes:
        portfolio_value: Current total portfolio value in USD.
        unrealized_pnl: Sum of unrealised PnL across all open positions.
        daily_pnl: Realised + unrealised PnL for today.
        daily_drawdown: Positive fraction representing today's drawdown.
        drawdown_limit: Maximum allowed drawdown fraction (from RiskLimits).
        positions: List of currently open positions.
        recent_trades: Up to the last 10 completed trade records.
        signals: Per-symbol signal values.
        vol_regime: Current volatility regime.
        llm_regime_assessment: Free-text regime assessment from LLM.
        data_timestamps: Mapping of data source name -> last update timestamp.
        db_connected: Whether the database is reachable.
        pending_proposals: Count of pending oversight proposals.
        active_alerts: Active alerts from the AlertManager.
        last_updated: When this state object was created.
    """

    portfolio_value: float = 0.0
    unrealized_pnl: float = 0.0
    daily_pnl: float = 0.0
    daily_drawdown: float = 0.0
    drawdown_limit: float = 0.02
    positions: list[Position] = field(default_factory=list)
    recent_trades: list[TradeRecord] = field(default_factory=list)
    signals: list[SignalValues] = field(default_factory=list)
    vol_regime: VolRegime = VolRegime.LOW
    llm_regime_assessment: str = "N/A"
    data_timestamps: dict[str, datetime] = field(default_factory=dict)
    db_connected: bool = True
    pending_proposals: int = 0
    active_alerts: list[Alert] = field(default_factory=list)
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


# ── Dashboard renderer ────────────────────────────────────────────────────────


class TradingDashboard:
    """Rich-powered CLI dashboard for live trading monitoring.

    Renders a multi-panel layout using :class:`rich.live.Live` and
    :class:`rich.layout.Layout`.  Panels are refreshed in-place via
    :meth:`update`.

    Args:
        refresh_per_second: How many times per second to re-render.
        console: Optional :class:`rich.console.Console` instance; a new
                 one is created if not provided.
    """

    def __init__(
        self,
        refresh_per_second: float = 1.0,
        console: Console | None = None,
    ) -> None:
        self.refresh_per_second = refresh_per_second
        self._console = console or Console()
        self._state: DashboardState = DashboardState()
        self._layout = self._build_layout()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> Layout:
        """Build the Rich Layout structure."""
        layout = Layout(name="root")

        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right"),
        )

        layout["left"].split_column(
            Layout(name="portfolio"),
            Layout(name="signals"),
            Layout(name="regime"),
        )

        layout["right"].split_column(
            Layout(name="recent_trades"),
            Layout(name="system_health"),
            Layout(name="alerts"),
        )

        return layout

    # ------------------------------------------------------------------
    # Panel builders
    # ------------------------------------------------------------------

    def _render_header(self, state: DashboardState) -> Panel:
        ts = state.last_updated.strftime("%Y-%m-%d %H:%M:%S UTC")
        header_text = Text(f"QTS Trading Dashboard  |  {ts}", style="bold cyan", justify="center")
        return Panel(header_text, style="cyan")

    def _render_footer(self, state: DashboardState) -> Panel:
        pending = state.pending_proposals
        color = "yellow" if pending > 0 else "green"
        footer_text = Text(
            f"Pending proposals: {pending}  |  Press Ctrl+C to exit",
            style=color,
            justify="center",
        )
        return Panel(footer_text)

    def _render_portfolio(self, state: DashboardState) -> Panel:
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Metric", style="dim", width=24)
        table.add_column("Value", justify="right")

        pnl_color = "green" if state.unrealized_pnl >= 0 else "red"
        daily_color = "green" if state.daily_pnl >= 0 else "red"

        drawdown_pct = state.daily_drawdown * 100
        limit_pct = state.drawdown_limit * 100
        dd_ratio = state.daily_drawdown / state.drawdown_limit if state.drawdown_limit > 0 else 0
        dd_color = "red" if dd_ratio >= 0.8 else ("yellow" if dd_ratio >= 0.5 else "green")

        table.add_row("Portfolio Value", f"${state.portfolio_value:,.2f}")
        table.add_row(
            "Unrealised PnL",
            Text(f"${state.unrealized_pnl:+,.2f}", style=pnl_color),
        )
        table.add_row(
            "Daily PnL",
            Text(f"${state.daily_pnl:+,.2f}", style=daily_color),
        )
        table.add_row(
            "Daily Drawdown",
            Text(f"{drawdown_pct:.2f}% / {limit_pct:.2f}%", style=dd_color),
        )
        table.add_row("Open Positions", str(len(state.positions)))

        for pos in state.positions:
            upnl_color = "green" if pos.unrealized_pnl >= 0 else "red"
            table.add_row(
                f"  {pos.symbol} ({pos.direction})",
                Text(f"${pos.unrealized_pnl:+,.2f}", style=upnl_color),
            )

        return Panel(table, title="[bold]Portfolio[/bold]", border_style="blue")

    def _render_recent_trades(self, state: DashboardState) -> Panel:
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Symbol", width=10)
        table.add_column("Dir", width=5)
        table.add_column("PnL", justify="right", width=12)
        table.add_column("Outcome", width=10)

        trades = state.recent_trades[-10:]
        for trade in reversed(trades):
            pnl_color = "green" if trade.pnl_usd >= 0 else "red"
            outcome_color = (
                "green"
                if trade.outcome == "WIN"
                else ("red" if trade.outcome == "LOSS" else "yellow")
            )
            table.add_row(
                trade.symbol,
                trade.direction,
                Text(f"${trade.pnl_usd:+,.2f}", style=pnl_color),
                Text(trade.outcome, style=outcome_color),
            )

        return Panel(table, title="[bold]Recent Trades (last 10)[/bold]", border_style="blue")

    def _render_signals(self, state: DashboardState) -> Panel:
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Symbol", width=10)
        table.add_column("RSI", justify="right", width=8)
        table.add_column("MACD", justify="right", width=10)
        table.add_column("Sentiment", justify="right", width=12)
        table.add_column("Alpha", justify="right", width=10)

        for sig in state.signals:
            rsi_color = "red" if sig.rsi > 70 else ("green" if sig.rsi < 30 else "white")
            alpha_color = (
                "green"
                if sig.combined_alpha > 0
                else ("red" if sig.combined_alpha < 0 else "white")
            )
            sent_color = "green" if sig.sentiment > 0 else ("red" if sig.sentiment < 0 else "white")
            table.add_row(
                sig.symbol,
                Text(f"{sig.rsi:.1f}", style=rsi_color),
                f"{sig.macd:.4f}",
                Text(f"{sig.sentiment:.3f}", style=sent_color),
                Text(f"{sig.combined_alpha:.3f}", style=alpha_color),
            )

        return Panel(table, title="[bold]Signals[/bold]", border_style="blue")

    def _render_regime(self, state: DashboardState) -> Panel:
        regime_color = "red" if state.vol_regime == VolRegime.HIGH else "green"
        content = Text()
        content.append("Vol Regime: ", style="dim")
        content.append(str(state.vol_regime), style=f"bold {regime_color}")
        content.append("\n\nLLM Assessment:\n", style="dim")
        content.append(state.llm_regime_assessment, style="italic")
        return Panel(content, title="[bold]Regime[/bold]", border_style="blue")

    def _render_system_health(self, state: DashboardState) -> Panel:
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Source", width=14)
        table.add_column("Last Update", width=22)
        table.add_column("Status", width=10)

        now = datetime.now(UTC)
        for source, ts in state.data_timestamps.items():
            age_s = (now - ts).total_seconds()
            if age_s < 60:
                status_text = Text("FRESH", style="green")
            elif age_s < 300:
                status_text = Text("STALE", style="yellow")
            else:
                status_text = Text("DEAD", style="red")
            table.add_row(source, ts.strftime("%H:%M:%S UTC"), status_text)

        db_color = "green" if state.db_connected else "red"
        db_status = Text("OK" if state.db_connected else "FAIL", style=db_color)
        table.add_row("database", "—", db_status)

        return Panel(table, title="[bold]System Health[/bold]", border_style="blue")

    def _render_alerts(self, state: DashboardState) -> Panel:
        if not state.active_alerts:
            content = Text("No active alerts.", style="dim green")
            return Panel(content, title="[bold]Alerts[/bold]", border_style="green")

        table = Table(show_header=False, box=None)
        table.add_column("Level", width=10)
        table.add_column("Message")

        level_styles = {
            AlertLevel.INFO: "dim",
            AlertLevel.WARNING: "yellow",
            AlertLevel.CRITICAL: "bold red",
        }
        for alert in state.active_alerts[-8:]:
            style = level_styles.get(alert.level, "white")
            table.add_row(
                Text(str(alert.level), style=style),
                Text(alert.message, style=style),
            )

        return Panel(table, title="[bold]Alerts[/bold]", border_style="red")

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update(self, state: DashboardState) -> None:
        """Refresh all dashboard panels with new state data.

        Args:
            state: New :class:`DashboardState` to render.
        """
        self._state = state
        self._layout["header"].update(self._render_header(state))
        self._layout["portfolio"].update(self._render_portfolio(state))
        self._layout["recent_trades"].update(self._render_recent_trades(state))
        self._layout["signals"].update(self._render_signals(state))
        self._layout["regime"].update(self._render_regime(state))
        self._layout["system_health"].update(self._render_system_health(state))
        self._layout["alerts"].update(self._render_alerts(state))
        self._layout["footer"].update(self._render_footer(state))

    # ------------------------------------------------------------------
    # Live mode
    # ------------------------------------------------------------------

    def run(
        self,
        state_callback: object | None = None,
        refresh_interval_seconds: float = 1.0,
    ) -> None:
        """Start the live dashboard loop.

        Renders the dashboard and updates it at ``refresh_interval_seconds``
        intervals.  If a *state_callback* is provided it will be called each
        tick to obtain a fresh :class:`DashboardState`; otherwise the last
        state set via :meth:`update` is re-rendered.

        Args:
            state_callback: Callable[[], DashboardState] that returns fresh
                            state data each tick.  Optional.
            refresh_interval_seconds: Seconds between re-renders.
        """
        # Render initial frame so the layout is non-empty
        self.update(self._state)

        with Live(
            self._layout,
            console=self._console,
            refresh_per_second=max(1, int(1 / refresh_interval_seconds)),
            screen=True,
        ) as _live:
            try:
                while True:
                    if state_callback is not None and callable(state_callback):
                        new_state = state_callback()
                        if isinstance(new_state, DashboardState):
                            self.update(new_state)
                    time.sleep(refresh_interval_seconds)
            except KeyboardInterrupt:
                logger.info("TradingDashboard: live loop stopped by user")
