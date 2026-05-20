"""QTSStrategy actor — bridges QTS signal/decision logic into NautilusTrader.

The actor receives bar events from Nautilus's data engine, computes signals
via our pipeline, runs the strategy's decision logic, and submits orders
through Nautilus's order management. The strategy protocol implementation
is completely unaware of NautilusTrader.

This is the equivalent of the racing lab's Controller adapter — the controller
doesn't know about the physics engine, and the strategy doesn't know about
the execution engine.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar as NtBar
from nautilus_trader.model.data import BarSpecification, BarType
from nautilus_trader.model.enums import (
    AggregationSource,
    BarAggregation,
    PriceType,
    TimeInForce,
)
from nautilus_trader.model.enums import (
    OrderSide as NtOrderSide,
)
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy as NtStrategy

from qts.models.base import Bar, OrderType, Position, TradeDirection
from qts.nautilus.converters import nautilus_bar_to_qts, nautilus_fill_to_qts
from qts.signals.pipeline import SignalPipeline
from qts.strategies.base import Strategy

if TYPE_CHECKING:
    from nautilus_trader.model.events.order import OrderFilled

logger = logging.getLogger(__name__)


class QTSStrategyConfig(StrategyConfig, frozen=True):
    """Nautilus-side config for the QTSStrategy actor."""

    instrument_id: str = ""
    bar_window: int = 50
    sentiment_score: float = 0.0


class QTSStrategy(NtStrategy):
    """NautilusTrader actor wrapping QTS signal + decision logic.

    This actor:
    1. Subscribes to bar data from Nautilus's data engine
    2. Converts Nautilus bars to QTS bars
    3. Computes signals via our SignalPipeline
    4. Calls the inner strategy's on_bar() for decision logic
    5. Submits resulting orders through Nautilus's order management

    The inner QTS Strategy implementation is plugged in via set_qts_strategy().
    """

    def __init__(self, config: QTSStrategyConfig) -> None:
        super().__init__(config)
        self._config = config
        self._bar_buffer: list[Bar] = []
        self._signal_pipeline: SignalPipeline | None = None
        self._inner_strategy: Strategy | None = None
        self._instrument_id: InstrumentId | None = None

    def set_qts_strategy(self, strategy: Strategy) -> None:
        """Plug in the QTS strategy implementation.

        Called before the actor starts. The strategy is the pure decision
        logic — it has no knowledge of NautilusTrader.
        """
        self._inner_strategy = strategy

    def on_start(self) -> None:
        """Called when the strategy actor starts (Nautilus lifecycle)."""
        if not self._config.instrument_id:
            logger.error("QTSStrategy: no instrument_id configured")
            return

        self._instrument_id = InstrumentId.from_str(self._config.instrument_id)
        symbol = str(self._instrument_id).split(".")[0]
        self._signal_pipeline = SignalPipeline(
            symbol=symbol,
            sentiment_score=self._config.sentiment_score,
        )

        # Subscribe to 1-minute bars
        bar_spec = BarSpecification(
            step=1,
            aggregation=BarAggregation.MINUTE,
            price_type=PriceType.LAST,
        )
        bar_type = BarType(
            instrument_id=self._instrument_id,
            bar_spec=bar_spec,
            aggregation_source=AggregationSource.EXTERNAL,
        )
        self.subscribe_bars(bar_type)
        logger.info("QTSStrategy: started, subscribed to %s", bar_type)

    def on_bar(self, bar: NtBar) -> None:
        """Handle a new bar from Nautilus's data engine."""
        if self._inner_strategy is None or self._signal_pipeline is None:
            return

        # Convert Nautilus bar → QTS bar
        qts_bar = nautilus_bar_to_qts(bar)
        self._bar_buffer.append(qts_bar)

        # Keep rolling window
        window = self._config.bar_window
        if len(self._bar_buffer) > window * 2:
            self._bar_buffer = self._bar_buffer[-window:]

        # Compute signals — forward the inner strategy's params so the
        # pipeline can produce a non-zero combined_alpha for entry decisions.
        bar_window = self._bar_buffer[-window:]
        params = getattr(self._inner_strategy, "params", None)
        snapshot = self._signal_pipeline.compute(bar_window, params=params)
        if snapshot is None:
            return

        # Get current positions from Nautilus portfolio, valued at this bar's close
        positions = self._get_qts_positions(reference_price=bar.close)

        # Run strategy decision logic
        orders = self._inner_strategy.on_bar(qts_bar, snapshot, positions)

        # Submit orders through Nautilus
        for order in orders:
            self._submit_qts_order(order, bar)

    def on_order_filled(self, event: OrderFilled) -> None:
        """Handle fill event from Nautilus — forward to QTS strategy."""
        if self._inner_strategy is None:
            return
        fill = nautilus_fill_to_qts(event)
        self._inner_strategy.on_fill(fill)

    def on_text_event(self, event: object) -> None:
        """Forward a TextEvent (or any object) to the inner strategy.

        The Strategy protocol does not require on_text; news-aware strategies
        define it, the rest don't. We duck-type-check at the call site so a
        missing method is a no-op — never an error.
        """
        if self._inner_strategy is None:
            return
        handler = getattr(self._inner_strategy, "on_text", None)
        if callable(handler):
            handler(event)

    def on_stop(self) -> None:
        """Called when the strategy actor stops."""
        logger.info("QTSStrategy: stopped")

    def on_reset(self) -> None:
        """Called on strategy reset."""
        self._bar_buffer.clear()
        if self._signal_pipeline is not None:
            symbol = str(self._instrument_id).split(".")[0] if self._instrument_id else "UNKNOWN"
            self._signal_pipeline = SignalPipeline(symbol=symbol)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_qts_positions(self, reference_price: object | None = None) -> list[Position]:
        """Convert Nautilus portfolio positions to QTS positions.

        Args:
            reference_price: Nautilus Price (or None) used to mark open
                positions to market. When provided, unrealized_pnl is
                computed via Nautilus's Position.unrealized_pnl(price);
                otherwise unrealized_pnl falls back to 0.0.
        """
        if self._instrument_id is None:
            return []

        from datetime import UTC, datetime  # noqa: PLC0415

        from nautilus_trader.model.enums import PositionSide  # noqa: PLC0415

        positions: list[Position] = []
        for nt_pos in self.cache.positions(instrument_id=self._instrument_id):
            if nt_pos.is_closed:
                continue

            direction = (
                TradeDirection.LONG if nt_pos.side == PositionSide.LONG else TradeDirection.SHORT
            )
            entry_time = datetime.fromtimestamp(nt_pos.ts_opened / 1e9, tz=UTC)

            unrealized = 0.0
            if reference_price is not None:
                try:
                    unrealized = float(nt_pos.unrealized_pnl(reference_price))
                except (TypeError, ValueError):
                    unrealized = 0.0

            positions.append(
                Position(
                    symbol=str(nt_pos.instrument_id).split(".")[0],
                    direction=direction,
                    entry_price=float(nt_pos.avg_px_open),
                    quantity=float(nt_pos.quantity),
                    entry_time=entry_time,
                    unrealized_pnl=unrealized,
                )
            )
        return positions

    def _is_cash_account(self) -> bool:
        """True if the venue runs a spot CASH account (no shorting allowed)."""
        if self._instrument_id is None:
            return False
        from nautilus_trader.model.enums import AccountType  # noqa: PLC0415

        account = self.cache.account_for_venue(self._instrument_id.venue)
        return account is not None and account.type == AccountType.CASH

    def _current_long_quantity(self) -> float:
        """Total open LONG quantity on this instrument, summed across positions."""
        if self._instrument_id is None:
            return 0.0
        from nautilus_trader.model.enums import PositionSide  # noqa: PLC0415

        total = 0.0
        for nt_pos in self.cache.positions(instrument_id=self._instrument_id):
            if not nt_pos.is_closed and nt_pos.side == PositionSide.LONG:
                total += float(nt_pos.quantity)
        return total

    def _submit_qts_order(self, order: Any, bar: NtBar) -> None:
        """Convert a QTS Order and submit it through Nautilus."""
        if self._instrument_id is None:
            return

        from nautilus_trader.model.identifiers import ClientOrderId  # noqa: PLC0415

        # Match the instrument's actual price/size precision — Nautilus
        # rejects orders whose precision exceeds the instrument's.
        instrument = self.cache.instrument(self._instrument_id)
        if instrument is None:
            logger.warning("QTSStrategy: instrument %s not in cache", self._instrument_id)
            return

        nt_side = NtOrderSide.BUY if order.side.value == "BUY" else NtOrderSide.SELL
        size_prec = instrument.size_precision
        price_prec = instrument.price_precision

        rounded_qty = round(order.quantity, size_prec)

        # Spot CASH accounts cannot short. Two layered defences:
        # 1. Clamp SELL quantity to the currently-held long, so we don't
        #    accidentally over-sell within a single bar's order batch.
        # 2. Mark SELL orders reduce_only=True so Nautilus itself refuses
        #    to flip a long position into a net short via a multi-order
        #    sequence in the same bar.
        cash_account = nt_side == NtOrderSide.SELL and self._is_cash_account()
        if cash_account:
            long_qty = self._current_long_quantity()
            rounded_qty = min(rounded_qty, round(long_qty, size_prec))

        if rounded_qty <= 0:
            return
        qty = Quantity.from_str(f"{rounded_qty:.{size_prec}f}")
        client_order_id = ClientOrderId(order.order_id)

        if order.order_type == OrderType.MARKET:
            nt_order = self.order_factory.market(
                instrument_id=self._instrument_id,
                order_side=nt_side,
                quantity=qty,
                time_in_force=TimeInForce.IOC,
                reduce_only=cash_account,
            )
        elif order.order_type == OrderType.LIMIT and order.price is not None:
            from nautilus_trader.model.objects import Price  # noqa: PLC0415

            nt_order = self.order_factory.limit(
                instrument_id=self._instrument_id,
                order_side=nt_side,
                quantity=qty,
                price=Price.from_str(f"{round(order.price, price_prec):.{price_prec}f}"),
                time_in_force=TimeInForce.GTC,
                reduce_only=cash_account,
            )
        elif order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and order.stop_price:
            from nautilus_trader.model.objects import Price  # noqa: PLC0415

            nt_order = self.order_factory.stop_market(
                instrument_id=self._instrument_id,
                order_side=nt_side,
                quantity=qty,
                trigger_price=Price.from_str(
                    f"{round(order.stop_price, price_prec):.{price_prec}f}"
                ),
                time_in_force=TimeInForce.GTC,
                reduce_only=cash_account,
            )
        else:
            nt_order = self.order_factory.market(
                instrument_id=self._instrument_id,
                order_side=nt_side,
                quantity=qty,
                time_in_force=TimeInForce.IOC,
                reduce_only=cash_account,
            )

        self.submit_order(nt_order)
        logger.debug(
            "QTSStrategy: submitted %s %s qty=%.4f",
            nt_side.name,
            self._instrument_id,
            order.quantity,
        )
