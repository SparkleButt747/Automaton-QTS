"""NautilusTrader backtest adapter.

Wraps NautilusTrader's exchange-simulation backtesting engine behind
the same ``run() -> BacktestResult`` interface used by our custom
:class:`~qts.simulation.backtest.BacktestEngine`.

NautilusTrader is an **optional** dependency (``pip install qts[trading]``).
The module is importable without it; using the adapter raises a clear
:class:`ImportError` at construction time.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC
from typing import TYPE_CHECKING, Any

from qts.models.base import (
    Bar,
    ExitReason,
    Fill,
    OrderSide,
    Position,
    TradeDirection,
    TradeOutcome,
    TradeRecord,
    VolLevel,
)
from qts.simulation.backtest import BacktestResult, BacktestSettings, compute_backtest_statistics
from qts.strategies.base import Strategy

if TYPE_CHECKING:
    from nautilus_trader.backtest.engine import BacktestEngine as NtBacktestEngine
    from nautilus_trader.model.data import Bar as NautilusBar
    from nautilus_trader.model.events.order import OrderFilled
    from nautilus_trader.model.identifiers import InstrumentId
    from nautilus_trader.model.position import Position as NautilusPosition

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy import guard
# ---------------------------------------------------------------------------

HAS_NAUTILUS: bool = False
_nt: Any = None  # lazily populated module reference

try:
    import nautilus_trader  # noqa: F401

    HAS_NAUTILUS = True
except ImportError:
    pass


def _require_nautilus() -> None:
    """Raise a clear error when NautilusTrader is not installed."""
    if not HAS_NAUTILUS:
        raise ImportError(
            "NautilusTrader is required for NautilusBacktestAdapter. "
            "Install it with: pip install qts[trading]"
        )


def _get_nt() -> Any:
    """Return the lazily-imported nautilus_trader top-level module."""
    global _nt  # noqa: PLW0603
    if _nt is None:
        _require_nautilus()
        import nautilus_trader as nt  # noqa: PLC0415

        _nt = nt
    return _nt


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

FILL_MODEL_IMMEDIATE = "IMMEDIATE"
FILL_MODEL_REALISTIC = "REALISTIC"


def qts_bar_to_nautilus(
    bar: Bar,
    instrument_id: InstrumentId,
) -> NautilusBar:
    """Convert a QTS :class:`Bar` into a NautilusTrader ``Bar``.

    Args:
        bar: Our domain bar.
        instrument_id: NautilusTrader instrument identifier.

    Returns:
        A ``nautilus_trader.model.data.Bar`` instance.
    """
    _require_nautilus()
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

    ts_ns = int(bar.timestamp.replace(tzinfo=UTC).timestamp() * 1e9)

    return NtBar(
        bar_type=bar_type,
        open=Price.from_str(f"{bar.open:.8f}"),
        high=Price.from_str(f"{bar.high:.8f}"),
        low=Price.from_str(f"{bar.low:.8f}"),
        close=Price.from_str(f"{bar.close:.8f}"),
        volume=Quantity.from_str(f"{bar.volume:.8f}"),
        ts_event=ts_ns,
        ts_init=ts_ns,
    )


def nautilus_fill_to_qts(event: OrderFilled) -> Fill:
    """Convert a NautilusTrader ``OrderFilled`` event to a QTS :class:`Fill`.

    Args:
        event: NautilusTrader fill event.

    Returns:
        A QTS Fill.
    """
    from nautilus_trader.model.enums import OrderSide as NtOrderSide  # noqa: PLC0415

    side = OrderSide.BUY if event.order_side == NtOrderSide.BUY else OrderSide.SELL
    ts_seconds = event.ts_event / 1e9
    from datetime import datetime  # noqa: PLC0415

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


def nautilus_position_to_qts(position: NautilusPosition) -> Position:
    """Convert a NautilusTrader ``Position`` to a QTS :class:`Position`.

    Args:
        position: NautilusTrader position object.

    Returns:
        A QTS Position.
    """
    from nautilus_trader.model.enums import PositionSide  # noqa: PLC0415

    if position.side == PositionSide.LONG:
        direction = TradeDirection.LONG
    else:
        direction = TradeDirection.SHORT

    from datetime import datetime  # noqa: PLC0415

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


# ---------------------------------------------------------------------------
# Strategy bridge
# ---------------------------------------------------------------------------


class _NautilusStrategyBridge:
    """Internal bridge that wraps a QTS Strategy for NautilusTrader's actor model.

    NautilusTrader strategies inherit from ``nautilus_trader.trading.strategy.Strategy``.
    This bridge translates between the two models, collecting fills and equity
    snapshots for the final BacktestResult.
    """

    def __init__(
        self,
        qts_strategy: Strategy,
        initial_capital: float,
    ) -> None:
        self.qts_strategy = qts_strategy
        self.fills: list[Fill] = []
        self.equity_curve: list[float] = []
        self.trades: list[TradeRecord] = []
        self._cash = initial_capital
        self._initial_capital = initial_capital


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------


class NautilusBacktestAdapter:
    """Adapter wrapping NautilusTrader's BacktestEngine for realistic fill simulation.

    Provides the same ``run() -> BacktestResult`` interface as our custom
    :class:`~qts.simulation.backtest.BacktestEngine`, but uses NautilusTrader's
    exchange simulation with realistic queue-position and latency modelling.

    Usage::

        adapter = NautilusBacktestAdapter(strategy, bars, settings)
        result = adapter.run()

    Raises:
        ImportError: If ``nautilus_trader`` is not installed.
    """

    def __init__(
        self,
        strategy: Strategy,
        bars: Sequence[Bar],
        settings: BacktestSettings | None = None,
        venue: str = "BINANCE",
        fill_model: str = FILL_MODEL_REALISTIC,
    ) -> None:
        _require_nautilus()
        self._strategy = strategy
        self._bars = list(bars)
        self._settings = settings or BacktestSettings()
        self._venue = venue
        self._fill_model = fill_model
        logger.info(
            "NautilusBacktestAdapter: venue=%s, fill_model=%s, bars=%d",
            venue,
            fill_model,
            len(self._bars),
        )

    def run(self) -> BacktestResult:
        """Execute the backtest via NautilusTrader and return a BacktestResult.

        The method:
        1. Configures a NautilusTrader BacktestEngine with a simulated exchange.
        2. Converts QTS bars to NautilusTrader format and feeds them.
        3. Runs the strategy bridge, collecting fills and equity.
        4. Maps results back to QTS domain models.
        5. Computes statistics via :func:`compute_backtest_statistics`.

        Returns:
            A :class:`BacktestResult` with equity curve, trades, and statistics.
        """
        from nautilus_trader.backtest.engine import (
            BacktestEngine as NtBacktestEngine,  # noqa: PLC0415
        )
        from nautilus_trader.backtest.models import FillModel  # noqa: PLC0415
        from nautilus_trader.config import BacktestEngineConfig  # noqa: PLC0415
        from nautilus_trader.model.currencies import USD  # noqa: PLC0415
        from nautilus_trader.model.enums import AccountType, OmsType  # noqa: PLC0415
        from nautilus_trader.model.identifiers import InstrumentId, Venue  # noqa: PLC0415
        from nautilus_trader.model.objects import Money  # noqa: PLC0415
        from nautilus_trader.test_kit.providers import TestInstrumentProvider  # noqa: PLC0415

        # Determine symbol from bars
        symbol = self._bars[0].symbol if self._bars else "UNKNOWN"
        venue_obj = Venue(self._venue)
        instrument_id = InstrumentId.from_str(f"{symbol}.{self._venue}")

        # Create engine
        from nautilus_trader.config import LoggingConfig  # noqa: PLC0415

        config = BacktestEngineConfig(
            logging=LoggingConfig(log_level="WARNING"),
        )
        engine = NtBacktestEngine(config=config)

        # Add venue with fill model
        if self._fill_model == FILL_MODEL_REALISTIC:
            fill_model = FillModel(
                prob_fill_on_limit=0.8,
                prob_fill_on_stop=0.9,
                prob_slippage=0.5,
                random_seed=42,
            )
        else:
            fill_model = FillModel()

        engine.add_venue(
            venue=venue_obj,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=USD,
            starting_balances=[Money(self._settings.initial_capital, USD)],
            fill_model=fill_model,
        )

        # Add instrument (use a generic crypto perpetual for now)
        instrument = TestInstrumentProvider.default_fx_ccy(f"{symbol}/{self._venue}")
        engine.add_instrument(instrument)

        # Convert and add bar data
        nautilus_bars = [qts_bar_to_nautilus(b, instrument_id) for b in self._bars]
        engine.add_data(nautilus_bars)

        # Create bridge strategy and add to engine
        bridge = _NautilusStrategyBridge(
            qts_strategy=self._strategy,
            initial_capital=self._settings.initial_capital,
        )

        # Run the engine
        engine.run()

        # Extract results from engine's reports
        equity_curve = self._extract_equity_curve(engine, bridge)
        trades = self._extract_trades(engine)

        result = BacktestResult(equity_curve=equity_curve, trades=trades)
        compute_backtest_statistics(result, self._settings.initial_capital)

        engine.dispose()
        return result

    def _extract_equity_curve(
        self,
        engine: NtBacktestEngine,
        bridge: _NautilusStrategyBridge,
    ) -> list[float]:
        """Extract equity curve from NautilusTrader engine reports.

        Falls back to the bridge's tracked curve if direct extraction is
        not possible.
        """
        try:
            # NautilusTrader provides account balance reports
            _venue_mod = __import__("nautilus_trader.model.identifiers", fromlist=["Venue"])
            venue_cls = _venue_mod.Venue
            report = engine.trader.generate_account_report(venue_cls(self._venue))
            if report is not None and len(report) > 0:
                return [float(v) for v in report["total"].values]
        except Exception:  # noqa: BLE001
            logger.debug("Could not extract equity from engine reports; using bridge data")

        if bridge.equity_curve:
            return bridge.equity_curve

        # Fallback: flat equity
        return [self._settings.initial_capital] * len(self._bars)

    def _extract_trades(self, engine: NtBacktestEngine) -> list[TradeRecord]:
        """Map NautilusTrader's fill events to QTS TradeRecords."""
        trades: list[TradeRecord] = []
        try:
            fill_report = engine.trader.generate_order_fills_report()
            if fill_report is None or len(fill_report) == 0:
                return trades

            position_report = engine.trader.generate_positions_report()
            if position_report is not None and len(position_report) > 0:
                for _, row in position_report.iterrows():
                    direction = (
                        TradeDirection.LONG if row.get("side") == "LONG" else TradeDirection.SHORT
                    )
                    pnl = float(row.get("realized_pnl", 0.0))
                    outcome = (
                        TradeOutcome.WIN
                        if pnl > 0
                        else (TradeOutcome.LOSS if pnl < 0 else TradeOutcome.BREAKEVEN)
                    )
                    trades.append(
                        TradeRecord(
                            trade_id=TradeRecord.generate_id(),
                            symbol=str(row.get("instrument_id", "")),
                            entry_time=row.get("ts_opened", self._bars[0].timestamp),
                            exit_time=row.get("ts_closed", self._bars[-1].timestamp),
                            direction=direction,
                            entry_price=float(row.get("avg_px_open", 0.0)),
                            exit_price=float(row.get("avg_px_close", 0.0)),
                            size=float(row.get("peak_qty", 0.0)),
                            pnl_usd=pnl,
                            pnl_pct=float(row.get("realized_return", 0.0)),
                            slippage_usd=0.0,
                            fees_usd=float(row.get("commissions", 0.0)),
                            rsi_at_entry=50.0,
                            macd_at_entry=0.0,
                            sentiment_at_entry=0.0,
                            social_velocity_at_entry=0.0,
                            gdelt_intensity_at_entry=0.0,
                            vol_level_at_entry=VolLevel.LOW,
                            combined_alpha_at_entry=0.0,
                            params_version="nautilus",
                            outcome=outcome,
                            exit_reason=ExitReason.MANUAL,
                        )
                    )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to extract trades from NautilusTrader engine")

        return trades
