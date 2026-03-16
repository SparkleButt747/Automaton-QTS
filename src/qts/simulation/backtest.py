"""Backtest engine: deterministic event-driven simulation.

Given a Strategy, a sequence of Bars, and BacktestSettings, iterates
through each bar, computes signals via SignalPipeline, calls
strategy.on_bar(), simulates fills, and tracks portfolio state.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from qts.models.base import (
    Bar,
    ExitReason,
    Fill,
    Order,
    OrderSide,
    Position,
    SignalSnapshot,
    TradeDirection,
    TradeOutcome,
    TradeRecord,
    VolRegime,
)
from qts.signals.pipeline import SignalPipeline
from qts.strategies.base import Strategy

logger = logging.getLogger(__name__)


# ── Settings ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BacktestSettings:
    """Configuration for a backtest run.

    Attributes:
        initial_capital: Starting portfolio cash.
        slippage_pct: Fraction of price added/subtracted as slippage.
        commission_pct: Fraction of trade notional charged as commission.
        signal_window: Number of bars fed to SignalPipeline per step.
    """

    initial_capital: float = 100_000.0
    slippage_pct: float = 0.0005  # 0.05%
    commission_pct: float = 0.001  # 0.10%
    signal_window: int = 50


# ── Results ───────────────────────────────────────────────────────────────────


@dataclass
class BacktestResult:
    """Results of a backtest run.

    Attributes:
        equity_curve: Portfolio value at each bar (list of floats).
        trades: List of completed TradeRecord objects.
        sharpe_ratio: Annualised Sharpe ratio of daily returns.
        max_drawdown: Maximum peak-to-trough drawdown fraction.
        win_rate: Fraction of trades that were winners.
        profit_factor: Gross profit / gross loss.
        sortino_ratio: Annualised Sortino ratio (downside deviation).
        calmar_ratio: Annualised return / max drawdown.
        total_return: Total portfolio return as a fraction.
    """

    equity_curve: list[float] = field(default_factory=list)
    trades: list[TradeRecord] = field(default_factory=list)
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    total_return: float = 0.0


# ── Statistics (standalone) ───────────────────────────────────────────────────


def compute_backtest_statistics(result: BacktestResult, initial_capital: float) -> None:
    """Compute and populate performance statistics on a BacktestResult in-place.

    This is the shared statistics computation used by both the custom
    BacktestEngine and the NautilusBacktestAdapter.

    Args:
        result: BacktestResult to populate in-place.
        initial_capital: Starting capital for return calculation.
    """
    equity = np.array(result.equity_curve, dtype=np.float64)
    if len(equity) < 2:
        return

    # Total return
    result.total_return = float((equity[-1] - initial_capital) / initial_capital)

    # Daily returns (assumes each bar = 1 step)
    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]

    if len(returns) == 0:
        return

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0

    # Sharpe (annualised assuming 252 trading days)
    bars_per_year = 252
    if std_ret > 0:
        result.sharpe_ratio = float(mean_ret / std_ret * np.sqrt(bars_per_year))

    # Sortino (downside deviation)
    downside = returns[returns < 0]
    if len(downside) > 0:
        downside_std = float(np.std(downside, ddof=1))
        if downside_std > 0:
            result.sortino_ratio = float(mean_ret / downside_std * np.sqrt(bars_per_year))

    # Maximum drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    result.max_drawdown = float(np.max(drawdown))

    # Calmar
    ann_return = mean_ret * bars_per_year
    if result.max_drawdown > 0:
        result.calmar_ratio = float(ann_return / result.max_drawdown)

    # Win rate and profit factor
    trades = result.trades
    if trades:
        wins = [t for t in trades if t.outcome == TradeOutcome.WIN]
        losses = [t for t in trades if t.outcome == TradeOutcome.LOSS]
        result.win_rate = float(len(wins) / len(trades))

        gross_profit = sum(t.pnl_usd for t in wins)
        gross_loss = abs(sum(t.pnl_usd for t in losses))
        if gross_loss > 0:
            result.profit_factor = float(gross_profit / gross_loss)
        elif gross_profit > 0:
            result.profit_factor = float("inf")
        else:
            result.profit_factor = 0.0


# ── Engine ────────────────────────────────────────────────────────────────────


class BacktestEngine:
    """Deterministic event-driven backtest engine.

    Processes bars sequentially. For each bar:
    1. Feeds the rolling bar window to SignalPipeline to get a SignalSnapshot.
    2. Calls strategy.on_bar() to get orders.
    3. Simulates fills at bar close with slippage and commission.
    4. Updates portfolio state (cash, positions, equity).

    The engine is fully deterministic: given the same bars, strategy,
    and settings, it always produces identical results.

    Usage::

        engine = BacktestEngine(strategy, bars, settings)
        result = engine.run()
    """

    def __init__(
        self,
        strategy: Strategy,
        bars: Sequence[Bar],
        settings: BacktestSettings | None = None,
        signal_pipeline: SignalPipeline | None = None,
    ) -> None:
        """Initialise the BacktestEngine.

        Args:
            strategy: Strategy instance implementing the Strategy protocol.
            bars: Ordered sequence of Bar objects (oldest first).
            settings: Backtest configuration; defaults are used if None.
            signal_pipeline: Optional pre-configured SignalPipeline.
                             If None, one is created from the first bar's symbol.
        """
        self._strategy = strategy
        self._bars = list(bars)
        self._settings = settings or BacktestSettings()
        symbol = self._bars[0].symbol if self._bars else "UNKNOWN"
        self._pipeline: SignalPipeline = signal_pipeline or SignalPipeline(symbol=symbol)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the backtest and return aggregated results.

        Returns:
            BacktestResult with equity curve, trade list, and statistics.
        """
        cash = self._settings.initial_capital
        positions: list[Position] = []
        equity_curve: list[float] = []
        completed_trades: list[TradeRecord] = []

        # Track open trade metadata for attribution
        # Maps symbol -> entry bar index + signal snapshot at entry
        open_trade_meta: dict[str, dict[str, object]] = {}

        bars = self._bars
        window = self._settings.signal_window

        for idx, bar in enumerate(bars):
            # Build rolling window for signal computation
            start = max(0, idx - window + 1)
            bar_window = bars[start : idx + 1]

            snapshot = self._pipeline.compute(bar_window)
            if snapshot is None:
                # Use a neutral placeholder snapshot
                snapshot = self._make_neutral_snapshot(bar)

            # Update bars_held for open positions
            for pos in positions:
                pos.bars_held += 1

            # Ask strategy for orders
            orders = self._strategy.on_bar(bar, snapshot, list(positions))

            # Simulate fills
            for order in orders:
                fill, new_positions, closed_trades = self._simulate_fill(
                    order,
                    bar,
                    positions,
                    snapshot,
                    open_trade_meta,
                    cash,
                )
                if fill is not None:
                    # Update cash
                    if order.side == OrderSide.BUY:
                        cash -= fill.price * fill.quantity + fill.commission
                    else:
                        cash += fill.price * fill.quantity - fill.commission

                    positions = new_positions
                    completed_trades.extend(closed_trades)
                    self._strategy.on_fill(fill)

            # Mark-to-market equity
            mark_value = sum(p.quantity * bar.close for p in positions)
            equity = cash + mark_value
            equity_curve.append(equity)

        # Close any remaining open positions at last bar price
        if self._bars and positions:
            last_bar = self._bars[-1]
            for pos in list(positions):
                trade = self._close_position(
                    pos,
                    last_bar.close,
                    last_bar,
                    snapshot,  # type: ignore[arg-type]
                    ExitReason.MANUAL,
                )
                completed_trades.append(trade)

        result = BacktestResult(equity_curve=equity_curve, trades=completed_trades)
        self._compute_statistics(result, self._settings.initial_capital)
        return result

    # ------------------------------------------------------------------
    # Fill simulation
    # ------------------------------------------------------------------

    def _simulate_fill(
        self,
        order: Order,
        bar: Bar,
        positions: list[Position],
        snapshot: SignalSnapshot,
        open_trade_meta: dict[str, dict[str, object]],
        cash: float,
    ) -> tuple[Fill | None, list[Position], list[TradeRecord]]:
        """Simulate a fill for an order at bar close with slippage.

        Args:
            order: The order to fill.
            bar: Current bar (price reference).
            positions: Current open positions (mutated in place via copy).
            snapshot: Signal snapshot at fill time.
            open_trade_meta: Metadata about open positions for attribution.
            cash: Current cash (used for sizing checks).

        Returns:
            Tuple of (fill, updated_positions, completed_trades).
        """
        slippage_factor = (
            1.0 + self._settings.slippage_pct
            if order.side == OrderSide.BUY
            else 1.0 - self._settings.slippage_pct
        )
        fill_price = bar.close * slippage_factor
        notional = fill_price * order.quantity
        commission = notional * self._settings.commission_pct

        fill = Fill(
            order_id=order.order_id,
            fill_id=str(uuid.uuid4()),
            symbol=order.symbol,
            side=order.side,
            price=fill_price,
            quantity=order.quantity,
            commission=commission,
            timestamp=bar.timestamp,
            slippage=abs(fill_price - bar.close) * order.quantity,
        )

        new_positions = list(positions)
        completed_trades: list[TradeRecord] = []

        if order.side == OrderSide.BUY:
            # Check for open SHORT to close
            short_idx = next(
                (i for i, p in enumerate(new_positions) if p.direction == TradeDirection.SHORT),
                None,
            )
            if short_idx is not None:
                closed = new_positions.pop(short_idx)
                trade = self._close_position(
                    closed, fill_price, bar, snapshot, ExitReason.ALPHA_FLIP
                )
                completed_trades.append(trade)
                open_trade_meta.pop(order.symbol, None)
            else:
                # Open LONG
                pos = Position(
                    symbol=order.symbol,
                    direction=TradeDirection.LONG,
                    entry_price=fill_price,
                    quantity=order.quantity,
                    entry_time=bar.timestamp,
                )
                new_positions.append(pos)
                open_trade_meta[order.symbol] = {
                    "snapshot": snapshot,
                    "entry_bar": bar,
                }
        else:  # SELL
            # Check for open LONG to close
            long_idx = next(
                (i for i, p in enumerate(new_positions) if p.direction == TradeDirection.LONG),
                None,
            )
            if long_idx is not None:
                closed = new_positions.pop(long_idx)
                trade = self._close_position(
                    closed, fill_price, bar, snapshot, ExitReason.ALPHA_FLIP
                )
                completed_trades.append(trade)
                open_trade_meta.pop(order.symbol, None)
            else:
                # Open SHORT
                pos = Position(
                    symbol=order.symbol,
                    direction=TradeDirection.SHORT,
                    entry_price=fill_price,
                    quantity=order.quantity,
                    entry_time=bar.timestamp,
                )
                new_positions.append(pos)
                open_trade_meta[order.symbol] = {
                    "snapshot": snapshot,
                    "entry_bar": bar,
                }

        return fill, new_positions, completed_trades

    def _close_position(
        self,
        position: Position,
        exit_price: float,
        bar: Bar,
        snapshot: SignalSnapshot,
        exit_reason: ExitReason,
    ) -> TradeRecord:
        """Build a TradeRecord for a closed position.

        Args:
            position: The position being closed.
            exit_price: Fill price at exit.
            bar: The bar at which the position was closed.
            snapshot: Signal snapshot at close time.
            exit_reason: Reason for exit.

        Returns:
            A completed TradeRecord.
        """
        if position.direction == TradeDirection.LONG:
            pnl_usd = (exit_price - position.entry_price) * position.quantity
        else:
            pnl_usd = (position.entry_price - exit_price) * position.quantity

        pnl_pct = pnl_usd / (position.entry_price * position.quantity)
        outcome = (
            TradeOutcome.WIN
            if pnl_usd > 0
            else (TradeOutcome.LOSS if pnl_usd < 0 else TradeOutcome.BREAKEVEN)
        )

        # Estimate slippage (one-way)
        slippage_usd = abs(exit_price - bar.close) * position.quantity
        commission_usd = exit_price * position.quantity * self._settings.commission_pct

        return TradeRecord(
            trade_id=TradeRecord.generate_id(),
            symbol=position.symbol,
            entry_time=position.entry_time,
            exit_time=bar.timestamp,
            direction=position.direction,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.quantity,
            pnl_usd=pnl_usd,
            pnl_pct=pnl_pct,
            slippage_usd=slippage_usd,
            fees_usd=commission_usd,
            rsi_at_entry=snapshot.rsi if not np.isnan(snapshot.rsi) else 50.0,
            macd_at_entry=snapshot.macd_histogram,
            sentiment_at_entry=snapshot.sentiment_score,
            social_velocity_at_entry=0.0,
            gdelt_intensity_at_entry=0.0,
            vol_regime_at_entry=snapshot.vol_regime,
            combined_alpha_at_entry=snapshot.combined_alpha,
            params_version="backtest",
            outcome=outcome,
            exit_reason=exit_reason,
            failure_mode=None,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _compute_statistics(self, result: BacktestResult, initial_capital: float) -> None:
        """Compute and populate performance statistics on result.

        Args:
            result: BacktestResult to populate in-place.
            initial_capital: Starting capital for return calculation.
        """
        compute_backtest_statistics(result, initial_capital)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_neutral_snapshot(bar: Bar) -> SignalSnapshot:
        """Create a neutral (all-zero) SignalSnapshot for warm-up bars.

        Args:
            bar: Current bar (provides timestamp and symbol).

        Returns:
            A SignalSnapshot with neutral indicator values.
        """
        return SignalSnapshot(
            timestamp=bar.timestamp,
            symbol=bar.symbol,
            rsi=50.0,
            macd_histogram=0.0,
            macd_line=0.0,
            macd_signal=0.0,
            bb_position=0.5,
            bb_upper=bar.close * 1.02,
            bb_lower=bar.close * 0.98,
            atr=bar.high - bar.low,
            momentum_5=0.0,
            vol_regime=VolRegime.LOW,
            vol_regime_confidence=0.5,
            sentiment_score=0.0,
            combined_alpha=0.0,
        )
