"""Unit tests for qts.simulation.nautilus_adapter.

Covers:
- T-NAUT-1: Module importable without nautilus_trader installed
- T-NAUT-2: HAS_NAUTILUS flag detection
- T-NAUT-3: NautilusBacktestAdapter raises ImportError without nautilus_trader
- T-NAUT-4: qts_bar_to_nautilus conversion (requires nautilus_trader)
- T-NAUT-5: nautilus_fill_to_qts conversion (requires nautilus_trader)
- T-NAUT-6: nautilus_position_to_qts conversion (requires nautilus_trader)
- T-NAUT-7: compute_backtest_statistics with known equity curve
- T-NAUT-8: compute_backtest_statistics with flat equity
- T-NAUT-9: compute_backtest_statistics with too-short equity
- T-NAUT-10: compute_backtest_statistics win_rate and profit_factor
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from qts.models.base import (
    Bar,
    ExitReason,
    OrderSide,
    TradeDirection,
    TradeOutcome,
    TradeRecord,
    VolLevel,
)
from qts.simulation.backtest import BacktestResult, compute_backtest_statistics
from qts.simulation.nautilus_adapter import (
    FILL_MODEL_IMMEDIATE,
    FILL_MODEL_REALISTIC,
    HAS_NAUTILUS,
    NautilusBacktestAdapter,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_bar(close: float = 100.0, idx: int = 0) -> Bar:
    return Bar(
        timestamp=datetime(2024, 1, 1, 0, idx, 0),
        symbol="BTCUSDT",
        open=close * 0.999,
        high=close * 1.002,
        low=close * 0.998,
        close=close,
        volume=1000.0,
        bar_count=1,
    )


def _make_bars(n: int = 100, base: float = 100.0) -> list[Bar]:
    """Generate a rising series of bars."""
    return [_make_bar(close=base + i * 0.5, idx=i) for i in range(n)]


def _make_trade(pnl: float) -> TradeRecord:
    outcome = (
        TradeOutcome.WIN if pnl > 0 else (TradeOutcome.LOSS if pnl < 0 else TradeOutcome.BREAKEVEN)
    )
    return TradeRecord(
        trade_id=TradeRecord.generate_id(),
        symbol="BTCUSDT",
        entry_time=datetime(2024, 1, 1),
        exit_time=datetime(2024, 1, 2),
        direction=TradeDirection.LONG,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        size=1.0,
        pnl_usd=pnl,
        pnl_pct=pnl / 100.0,
        slippage_usd=0.0,
        fees_usd=0.0,
        rsi_at_entry=50.0,
        macd_at_entry=0.0,
        sentiment_at_entry=0.0,
        social_velocity_at_entry=0.0,
        gdelt_intensity_at_entry=0.0,
        vol_level_at_entry=VolLevel.LOW,
        combined_alpha_at_entry=0.0,
        params_version="test",
        outcome=outcome,
        exit_reason=ExitReason.MANUAL,
    )


class _DummyStrategy:
    """Minimal strategy for adapter construction tests."""

    @property
    def name(self) -> str:
        return "dummy"

    def on_bar(self, bar, snapshot, positions):
        return []

    def on_fill(self, fill):
        pass


# ── T-NAUT-1: Module is importable regardless ────────────────────────────────


def test_module_importable():
    """T-NAUT-1: nautilus_adapter module is importable without nautilus_trader."""
    import qts.simulation.nautilus_adapter as mod

    assert hasattr(mod, "HAS_NAUTILUS")
    assert hasattr(mod, "NautilusBacktestAdapter")
    assert hasattr(mod, "qts_bar_to_nautilus")
    assert hasattr(mod, "nautilus_fill_to_qts")
    assert hasattr(mod, "nautilus_position_to_qts")


# ── T-NAUT-2: HAS_NAUTILUS flag ──────────────────────────────────────────────


def test_has_nautilus_is_bool():
    """T-NAUT-2: HAS_NAUTILUS is a boolean."""
    assert isinstance(HAS_NAUTILUS, bool)


# ── T-NAUT-3: ImportError when nautilus not installed ─────────────────────────


@pytest.mark.skip(reason="nautilus_trader is now a required dep — optional import guard removed")
def test_adapter_raises_without_nautilus():
    """T-NAUT-3: NautilusBacktestAdapter raises ImportError without nautilus_trader."""
    bars = _make_bars(10)
    strategy = _DummyStrategy()

    with patch("qts.simulation.nautilus_adapter.HAS_NAUTILUS", False):
        with pytest.raises(ImportError, match="NautilusTrader is required"):
            NautilusBacktestAdapter(strategy=strategy, bars=bars)


@pytest.mark.skip(reason="nautilus_trader is now a required dep — optional import guard removed")
def test_qts_bar_to_nautilus_raises_without_nautilus():
    """qts_bar_to_nautilus raises ImportError when nautilus_trader is absent."""
    bar = _make_bar()
    with patch("qts.simulation.nautilus_adapter.HAS_NAUTILUS", False):
        from qts.simulation.nautilus_adapter import qts_bar_to_nautilus

        with pytest.raises(ImportError, match="NautilusTrader is required"):
            qts_bar_to_nautilus(bar, MagicMock())


# ── T-NAUT-4 to T-NAUT-6: Conversion with nautilus_trader installed ──────────

_skip_no_nautilus = pytest.mark.skipif(
    not HAS_NAUTILUS,
    reason="nautilus_trader not installed",
)


@_skip_no_nautilus
def test_qts_bar_to_nautilus_conversion():
    """T-NAUT-4: Bar conversion produces correct OHLCV values."""
    from nautilus_trader.model.identifiers import InstrumentId

    from qts.simulation.nautilus_adapter import qts_bar_to_nautilus

    bar = _make_bar(close=50000.0)
    instrument_id = InstrumentId.from_str("BTCUSDT.BINANCE")
    nt_bar = qts_bar_to_nautilus(bar, instrument_id)

    assert float(nt_bar.close) == pytest.approx(50000.0, rel=1e-6)
    assert float(nt_bar.open) == pytest.approx(bar.open, rel=1e-4)
    assert float(nt_bar.high) == pytest.approx(bar.high, rel=1e-4)
    assert float(nt_bar.low) == pytest.approx(bar.low, rel=1e-4)


@_skip_no_nautilus
def test_nautilus_fill_to_qts_conversion():
    """T-NAUT-5: OrderFilled event maps to QTS Fill correctly."""
    from nautilus_trader.model.enums import OrderSide as NtOrderSide

    from qts.simulation.nautilus_adapter import nautilus_fill_to_qts

    event = MagicMock()
    event.order_side = NtOrderSide.BUY
    event.client_order_id = "ord-123"
    event.trade_id = "trd-456"
    event.instrument_id = "BTCUSDT.BINANCE"
    event.last_px = 50000.0
    event.last_qty = 1.5
    event.commission.as_double.return_value = 5.0
    event.ts_event = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1e9)

    fill = nautilus_fill_to_qts(event)
    assert fill.side == OrderSide.BUY
    assert fill.price == 50000.0
    assert fill.quantity == 1.5
    assert fill.commission == 5.0
    assert fill.symbol == "BTCUSDT.BINANCE"


@_skip_no_nautilus
def test_nautilus_position_to_qts_conversion():
    """T-NAUT-6: NautilusTrader Position maps to QTS Position correctly."""
    from nautilus_trader.model.enums import PositionSide

    from qts.simulation.nautilus_adapter import nautilus_position_to_qts

    pos = MagicMock()
    pos.side = PositionSide.LONG
    pos.instrument_id = "BTCUSDT.BINANCE"
    pos.avg_px_open = 50000.0
    pos.quantity = 2.0
    pos.ts_opened = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1e9)
    pos.unrealized_pnl = 100.0

    qts_pos = nautilus_position_to_qts(pos)
    assert qts_pos.direction == TradeDirection.LONG
    assert qts_pos.entry_price == 50000.0
    assert qts_pos.quantity == 2.0


# ── T-NAUT-7 to T-NAUT-10: compute_backtest_statistics ───────────────────────


def test_compute_stats_rising_equity():
    """T-NAUT-7: Statistics on a rising equity curve produce positive Sharpe."""
    # Steadily rising equity: 100k -> 110k over 100 steps
    equity = [100_000 + i * 100 for i in range(100)]
    result = BacktestResult(equity_curve=equity)
    compute_backtest_statistics(result, 100_000.0)

    assert result.total_return == pytest.approx(0.099, rel=0.01)
    assert result.sharpe_ratio > 0
    assert result.max_drawdown == 0.0  # monotonically rising
    # Sortino is 0 for monotonically rising equity (no downside returns)
    assert result.sortino_ratio == 0.0


def test_compute_stats_flat_equity():
    """T-NAUT-8: Flat equity curve yields zero returns and zero Sharpe."""
    equity = [100_000.0] * 50
    result = BacktestResult(equity_curve=equity)
    compute_backtest_statistics(result, 100_000.0)

    assert result.total_return == 0.0
    assert result.sharpe_ratio == 0.0
    assert result.max_drawdown == 0.0


def test_compute_stats_short_equity():
    """T-NAUT-9: Equity with < 2 points leaves stats at defaults."""
    result = BacktestResult(equity_curve=[100_000.0])
    compute_backtest_statistics(result, 100_000.0)

    assert result.sharpe_ratio == 0.0
    assert result.total_return == 0.0


def test_compute_stats_win_rate_and_profit_factor():
    """T-NAUT-10: Win rate and profit factor computed from trades."""
    trades = [
        _make_trade(pnl=500.0),
        _make_trade(pnl=300.0),
        _make_trade(pnl=-200.0),
    ]
    equity = [100_000 + i * 50 for i in range(50)]
    result = BacktestResult(equity_curve=equity, trades=trades)
    compute_backtest_statistics(result, 100_000.0)

    assert result.win_rate == pytest.approx(2 / 3)
    assert result.profit_factor == pytest.approx(800.0 / 200.0)


def test_compute_stats_all_winners():
    """Profit factor should be inf when there are no losses."""
    trades = [_make_trade(pnl=100.0), _make_trade(pnl=200.0)]
    equity = [100_000 + i * 50 for i in range(20)]
    result = BacktestResult(equity_curve=equity, trades=trades)
    compute_backtest_statistics(result, 100_000.0)

    assert result.win_rate == 1.0
    assert result.profit_factor == float("inf")


def test_compute_stats_drawdown():
    """Max drawdown should be detected in a V-shaped equity curve."""
    # 100k -> 90k -> 95k  =>  drawdown = 10%
    equity = [100_000.0, 95_000.0, 90_000.0, 92_000.0, 95_000.0]
    result = BacktestResult(equity_curve=equity)
    compute_backtest_statistics(result, 100_000.0)

    assert result.max_drawdown == pytest.approx(0.10, rel=0.01)


def test_fill_model_constants():
    """Fill model constant values are as expected."""
    assert FILL_MODEL_IMMEDIATE == "IMMEDIATE"
    assert FILL_MODEL_REALISTIC == "REALISTIC"
