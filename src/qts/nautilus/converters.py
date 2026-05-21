"""Data type converters between QTS and NautilusTrader domains.

All conversion functions are stateless and side-effect free.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from qts.models.base import (
    Bar,
    Fill,
    Order,
    OrderSide,
    OrderType,
    Position,
    TradeDirection,
)

if TYPE_CHECKING:
    from nautilus_trader.model.data import Bar as NtBar
    from nautilus_trader.model.events.order import OrderFilled
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.orders import Order as NtOrder
    from nautilus_trader.model.position import Position as NtPosition

    from qts.nautilus.news_data import NewsDataPoint
    from qts.world.events import TextEvent


# ------------------------------------------------------------------
# QTS → Nautilus
# ------------------------------------------------------------------


def text_event_to_news_data(event: TextEvent) -> NewsDataPoint:
    """Convert a QTS TextEvent to a Nautilus NewsDataPoint custom data object."""
    from qts.nautilus.news_data import NewsDataPoint  # noqa: PLC0415

    ts = event.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    ts_ns = int(ts.timestamp() * 1e9)
    return NewsDataPoint(
        source=event.source,
        persona=event.persona or "",
        text=event.text,
        ts_event=ts_ns,
        ts_init=ts_ns,
    )


def qts_bar_to_nautilus(
    bar: Bar,
    instrument_id: InstrumentId,
    price_precision: int = 2,
    size_precision: int = 6,
) -> NtBar:
    """Convert a QTS Bar to a NautilusTrader Bar."""
    from nautilus_trader.model.data import Bar as NtBar  # noqa: PLC0415
    from nautilus_trader.model.data import BarSpecification, BarType  # noqa: PLC0415
    from nautilus_trader.model.enums import (  # noqa: PLC0415
        AggregationSource,
        BarAggregation,
        PriceType,
    )
    from nautilus_trader.model.objects import Price, Quantity  # noqa: PLC0415

    bar_spec = BarSpecification(
        step=1,
        aggregation=BarAggregation.MINUTE,
        price_type=PriceType.LAST,
    )
    bar_type = BarType(
        instrument_id=instrument_id,
        bar_spec=bar_spec,
        aggregation_source=AggregationSource.EXTERNAL,
    )

    ts = bar.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    ts_ns = int(ts.timestamp() * 1e9)

    # Nautilus validates OHLC constraints: low <= open/close <= high
    o, h, lo, c = bar.open, bar.high, bar.low, bar.close
    h = max(h, o, c)
    lo = min(lo, o, c)

    pfmt = f".{price_precision}f"
    sfmt = f".{size_precision}f"

    return NtBar(
        bar_type=bar_type,
        open=Price.from_str(f"{o:{pfmt}}"),
        high=Price.from_str(f"{h:{pfmt}}"),
        low=Price.from_str(f"{lo:{pfmt}}"),
        close=Price.from_str(f"{c:{pfmt}}"),
        volume=Quantity.from_str(f"{max(bar.volume, 0.0):{sfmt}}"),
        ts_event=ts_ns,
        ts_init=ts_ns,
    )


def qts_order_to_nautilus(
    order: Order,
    instrument_id: InstrumentId,
    strategy_id: Any,
) -> NtOrder:
    """Convert a QTS Order to a NautilusTrader Order.

    Returns a MarketOrder, LimitOrder, or StopMarketOrder depending on
    the QTS order type.
    """
    from nautilus_trader.model.enums import OrderSide as NtOrderSide  # noqa: PLC0415
    from nautilus_trader.model.enums import TimeInForce  # noqa: PLC0415
    from nautilus_trader.model.identifiers import (  # noqa: PLC0415
        ClientOrderId,
        StrategyId,
    )
    from nautilus_trader.model.objects import Price, Quantity  # noqa: PLC0415
    from nautilus_trader.model.orders import (  # noqa: PLC0415
        LimitOrder,
        MarketOrder,
        StopMarketOrder,
    )

    nt_side = NtOrderSide.BUY if order.side == OrderSide.BUY else NtOrderSide.SELL
    client_order_id = ClientOrderId(order.order_id)
    qty = Quantity.from_str(f"{order.quantity:.8f}")

    if order.order_type == OrderType.MARKET:
        return MarketOrder(
            trader_id=strategy_id.trader_id,
            strategy_id=StrategyId(str(strategy_id)),
            instrument_id=instrument_id,
            client_order_id=client_order_id,
            order_side=nt_side,
            quantity=qty,
            time_in_force=TimeInForce.IOC,
            init_id=client_order_id,
            ts_init=0,
        )
    elif order.order_type == OrderType.LIMIT and order.price is not None:
        return LimitOrder(
            trader_id=strategy_id.trader_id,
            strategy_id=StrategyId(str(strategy_id)),
            instrument_id=instrument_id,
            client_order_id=client_order_id,
            order_side=nt_side,
            quantity=qty,
            price=Price.from_str(f"{order.price:.8f}"),
            time_in_force=TimeInForce.GTC,
            init_id=client_order_id,
            ts_init=0,
        )
    elif order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and order.stop_price:
        return StopMarketOrder(
            trader_id=strategy_id.trader_id,
            strategy_id=StrategyId(str(strategy_id)),
            instrument_id=instrument_id,
            client_order_id=client_order_id,
            order_side=nt_side,
            quantity=qty,
            trigger_price=Price.from_str(f"{order.stop_price:.8f}"),
            time_in_force=TimeInForce.GTC,
            init_id=client_order_id,
            ts_init=0,
        )
    else:
        # Fallback to market order
        return MarketOrder(
            trader_id=strategy_id.trader_id,
            strategy_id=StrategyId(str(strategy_id)),
            instrument_id=instrument_id,
            client_order_id=client_order_id,
            order_side=nt_side,
            quantity=qty,
            time_in_force=TimeInForce.IOC,
            init_id=client_order_id,
            ts_init=0,
        )


# ------------------------------------------------------------------
# Nautilus → QTS
# ------------------------------------------------------------------


def nautilus_fill_to_qts(event: OrderFilled) -> Fill:
    """Convert a NautilusTrader OrderFilled event to a QTS Fill."""
    from nautilus_trader.model.enums import OrderSide as NtOrderSide  # noqa: PLC0415

    side = OrderSide.BUY if event.order_side == NtOrderSide.BUY else OrderSide.SELL
    ts_seconds = event.ts_event / 1e9
    timestamp = datetime.fromtimestamp(ts_seconds, tz=UTC)

    return Fill(
        order_id=str(event.client_order_id),
        fill_id=str(event.trade_id),
        symbol=str(event.instrument_id),
        side=side,
        price=float(event.last_px),
        quantity=float(event.last_qty),
        commission=float(event.commission.as_double()) if event.commission else 0.0,
        timestamp=timestamp,
        slippage=0.0,
    )


def nautilus_position_to_qts(position: NtPosition) -> Position:
    """Convert a NautilusTrader Position to a QTS Position."""
    from nautilus_trader.model.enums import PositionSide  # noqa: PLC0415

    direction = TradeDirection.LONG if position.side == PositionSide.LONG else TradeDirection.SHORT
    ts_seconds = position.ts_opened / 1e9
    entry_time = datetime.fromtimestamp(ts_seconds, tz=UTC)

    return Position(
        symbol=str(position.instrument_id),
        direction=direction,
        entry_price=float(position.avg_px_open),
        quantity=float(position.quantity),
        entry_time=entry_time,
        unrealized_pnl=float(position.unrealized_pnl) if position.unrealized_pnl else 0.0,
    )


def nautilus_bar_to_qts(bar: NtBar) -> Bar:
    """Convert a NautilusTrader Bar to a QTS Bar."""
    ts_seconds = bar.ts_event / 1e9
    timestamp = datetime.fromtimestamp(ts_seconds, tz=UTC)
    symbol = str(bar.bar_type.instrument_id).split(".")[0]

    return Bar(
        timestamp=timestamp,
        symbol=symbol,
        open=float(bar.open),
        high=float(bar.high),
        low=float(bar.low),
        close=float(bar.close),
        volume=float(bar.volume),
    )


def news_data_to_text_event(data: NewsDataPoint) -> TextEvent:
    """Convert a Nautilus NewsDataPoint back to a QTS TextEvent.

    persona/metadata are not round-tripped, but the classifier cache key is
    sha256(source + text + version) — timestamp/persona-free — so classify()
    still hits the warmed cache after the roundtrip.
    """
    from qts.world.events import TextEvent  # noqa: PLC0415

    ts = datetime.fromtimestamp(data.ts_event / 1e9, tz=UTC)
    return TextEvent(
        timestamp=ts,
        source=data.source,
        persona=data.persona or None,
        text=data.text,
        metadata={},
    )
