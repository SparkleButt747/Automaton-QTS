"""Trading execution engine."""
from __future__ import annotations

import logging
import time
from typing import Any, Optional, Protocol

from qts.models.base import Bar, Fill, Position
from qts.trade_logging.signal_logger import SignalLogger
from qts.trade_logging.trade_logger import TradeLogger

logger = logging.getLogger(__name__)


class ExecutionProtocol(Protocol):
    """Protocol for dependency-injection of ExecutionEngine."""

    def process_bar(
        self,
        bar: Bar,
        strategy: Any,
        signal_pipeline: Any,
    ) -> list[Fill]:
        """Process a single bar and return any fills generated."""
        ...

    def run_live_loop(
        self,
        provider: Any,
        strategy: Any,
        signal_pipeline: Any,
    ) -> None:
        """Run the main live/paper trading event loop."""
        ...


class ExecutionEngine:
    """Orchestrates bar-by-bar signal computation, strategy evaluation,
    risk-gating, and order submission.

    The engine is intentionally thin: it wires together the signal pipeline,
    strategy, risk manager, and loggers so that each component can be tested
    and replaced independently.

    Args:
        risk_manager: :class:`~qts.execution.risk.RiskManager` instance that
                      enforces position-size and drawdown limits.
        trade_logger: :class:`~qts.trade_logging.trade_logger.TradeLogger` for
                      persisting completed trade records.
        signal_logger: :class:`~qts.trade_logging.signal_logger.SignalLogger` for
                       persisting signal snapshots.
        portfolio_value: Starting portfolio value in USD.
        paper_mode: When True, simulate fills without hitting a live broker.
    """

    def __init__(
        self,
        risk_manager: Any,
        trade_logger: Optional[TradeLogger] = None,
        signal_logger: Optional[SignalLogger] = None,
        portfolio_value: float = 100_000.0,
        paper_mode: bool = True,
    ) -> None:
        self.risk_manager = risk_manager
        self.trade_logger = trade_logger or TradeLogger()
        self.signal_logger = signal_logger or SignalLogger()
        self.portfolio_value = portfolio_value
        self.paper_mode = paper_mode
        self._positions: list[Position] = []
        self._daily_pnl: float = 0.0
        self._bar_buffer: dict[str, list[Bar]] = {}

    # ------------------------------------------------------------------
    # Bar processing
    # ------------------------------------------------------------------

    def process_bar(
        self,
        bar: Bar,
        strategy: Any,
        signal_pipeline: Any,
    ) -> list[Fill]:
        """Process a single OHLCV bar through the full signal-to-fill pipeline.

        Steps:
        1. Append bar to the rolling buffer for its symbol.
        2. Compute signals via *signal_pipeline*.
        3. Evaluate strategy to generate an order proposal.
        4. Validate the proposal against risk limits.
        5. Simulate or submit the order, returning fills.

        Args:
            bar: Incoming OHLCV bar.
            strategy: Object with a ``generate_order(snapshot, positions)``
                      method returning an :class:`~qts.models.base.Order`
                      or None.
            signal_pipeline: Object with a ``compute(bars)`` method returning
                             a :class:`~qts.models.base.SignalSnapshot` or None.

        Returns:
            List of :class:`~qts.models.base.Fill` objects (may be empty).
        """
        # Buffer bars per symbol
        buf = self._bar_buffer.setdefault(bar.symbol, [])
        buf.append(bar)

        # Compute signals
        snapshot = signal_pipeline.compute(buf)
        if snapshot is None:
            return []

        # Log signal snapshot
        self.signal_logger.log_snapshot(snapshot)

        # Ask strategy for an order
        order = None
        if hasattr(strategy, "generate_order"):
            try:
                order = strategy.generate_order(snapshot, self._positions)
            except Exception:  # noqa: BLE001
                logger.exception("Strategy raised an exception; skipping bar %s", bar.timestamp)
                return []

        if order is None:
            return []

        # Risk gate
        try:
            approved = self.risk_manager.approve_order(
                proposed_size_usd=order.quantity * bar.close,
                portfolio_value=self.portfolio_value,
                current_positions=self._positions,
                daily_pnl=self._daily_pnl,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Risk check blocked order: %s", exc)
            return []

        if not approved:
            logger.info("Order rejected by risk manager: %s", order)
            return []

        # Simulate or submit fill
        fills = self._execute_order(order, bar)
        return fills

    def _execute_order(self, order: Any, bar: Bar) -> list[Fill]:
        """Generate a simulated fill for paper trading.

        Args:
            order: :class:`~qts.models.base.Order` to fill.
            bar: Current bar providing the fill price.

        Returns:
            List containing a single simulated :class:`~qts.models.base.Fill`.
        """
        import uuid  # noqa: PLC0415
        from datetime import timezone  # noqa: PLC0415

        from qts.models.base import Fill  # noqa: PLC0415

        fill_price = bar.close  # market order at close
        commission = fill_price * order.quantity * 0.001  # 0.1% taker fee

        fill = Fill(
            order_id=order.order_id,
            fill_id=str(uuid.uuid4()),
            symbol=order.symbol,
            side=order.side,
            price=fill_price,
            quantity=order.quantity,
            commission=commission,
            timestamp=bar.timestamp,
            slippage=0.0,
        )

        logger.info(
            "ExecutionEngine: fill %s | %s %s @ %.4f | qty=%.4f",
            fill.fill_id,
            fill.side,
            fill.symbol,
            fill.price,
            fill.quantity,
        )
        return [fill]

    # ------------------------------------------------------------------
    # Live loop
    # ------------------------------------------------------------------

    def run_live_loop(
        self,
        provider: Any,
        strategy: Any,
        signal_pipeline: Any,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """Main event loop for live or paper trading.

        Polls *provider* for new bars and processes each one.  The loop
        runs until a KeyboardInterrupt or an unrecoverable provider error.

        Args:
            provider: Object with a ``get_latest_bar(symbol)`` method or
                      an iterable of :class:`~qts.models.base.Bar` objects.
            strategy: Strategy object passed through to :meth:`process_bar`.
            signal_pipeline: Signal pipeline passed through to :meth:`process_bar`.
            poll_interval_seconds: Seconds to sleep between polling cycles.
        """
        logger.info(
            "ExecutionEngine: starting live loop (paper_mode=%s, poll=%.1fs)",
            self.paper_mode,
            poll_interval_seconds,
        )

        try:
            # If provider is iterable (e.g. in tests / backtests)
            if hasattr(provider, "__iter__"):
                for bar in provider:
                    fills = self.process_bar(bar, strategy, signal_pipeline)
                    if fills:
                        logger.debug("Live loop: %d fill(s) on %s", len(fills), bar.symbol)
                return

            # Otherwise poll get_latest_bar
            while True:
                try:
                    bar = provider.get_latest_bar()
                    if bar is not None:
                        self.process_bar(bar, strategy, signal_pipeline)
                except Exception:  # noqa: BLE001
                    logger.exception("ExecutionEngine: error in live loop tick; continuing")

                time.sleep(poll_interval_seconds)

        except KeyboardInterrupt:
            logger.info("ExecutionEngine: live loop stopped by user")

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def reset_daily_state(self) -> None:
        """Reset intra-day PnL counters and lift any expired circuit breakers."""
        self._daily_pnl = 0.0
        if hasattr(self.risk_manager, "reset_daily_state"):
            self.risk_manager.reset_daily_state()
        logger.debug("ExecutionEngine: daily state reset")
