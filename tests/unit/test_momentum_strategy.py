"""Unit tests for qts.strategies.momentum.

Covers:
- T-MOM-1 to T-MOM-25
- MomentumStrategy name property
- on_bar: LONG entry, SHORT entry, exit on alpha, exit on time-stop
- on_bar: no double-entry, no order when alpha is neutral
- _kelly_quantity: sizing, zero-price guard
- on_fill: accepts fill notifications
"""

from __future__ import annotations

from datetime import datetime

import pytest

from qts.config import RiskLimits, StrategyParams
from qts.models.base import (
    Bar,
    Fill,
    OrderSide,
    OrderType,
    Position,
    SignalSnapshot,
    TradeDirection,
    VolLevel,
)
from qts.strategies.momentum import MomentumStrategy

# ── Helpers ───────────────────────────────────────────────────────────────────

_RISK_LIMITS = RiskLimits(
    max_daily_drawdown_pct=0.05,
    max_position_size_pct=0.20,
    max_open_positions=5,
    circuit_breaker_cooldown_seconds=300,
    sentiment_signal_max_scalar=3.0,
)

_STRATEGY_PARAMS_DATA = {
    "version": "1.0.0",
    "weights": {
        "w_rsi": 0.20,
        "w_macd": 0.20,
        "w_bb": 0.15,
        "w_mom": 0.15,
        "w_sentiment": 0.30,
    },
    "entry_threshold": 0.25,
    "exit_threshold": 0.05,
    "max_hold_bars": 24,
    "sentiment_fusion_weights": {"news": 0.5, "social": 0.3, "geopolitical": 0.2},
}

_PARAMS = StrategyParams(**_STRATEGY_PARAMS_DATA)


def _make_strategy(portfolio_value: float = 100_000.0) -> MomentumStrategy:
    return MomentumStrategy(
        params=_PARAMS,
        risk_limits=_RISK_LIMITS,
        portfolio_value=portfolio_value,
    )


def _make_bar(close: float = 100.0, symbol: str = "SYNTH") -> Bar:
    return Bar(
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        symbol=symbol,
        open=close * 0.999,
        high=close * 1.002,
        low=close * 0.998,
        close=close,
        volume=10_000.0,
        bar_count=1,
    )


def _make_snapshot(combined_alpha: float = 0.0, symbol: str = "SYNTH") -> SignalSnapshot:
    return SignalSnapshot(
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        symbol=symbol,
        rsi=50.0,
        macd_histogram=0.0,
        macd_line=0.0,
        macd_signal=0.0,
        bb_position=0.5,
        bb_upper=110.0,
        bb_lower=90.0,
        atr=1.0,
        momentum_5=0.0,
        vol_level=VolLevel.HIGH,
        vol_level_confidence=0.8,
        sentiment_score=0.0,
        combined_alpha=combined_alpha,
    )


def _make_position(
    direction: TradeDirection,
    bars_held: int = 0,
    quantity: float = 10.0,
    symbol: str = "SYNTH",
) -> Position:
    return Position(
        symbol=symbol,
        direction=direction,
        entry_price=100.0,
        quantity=quantity,
        entry_time=datetime(2024, 1, 1),
        bars_held=bars_held,
    )


def _make_fill(order_id: str = "test-order") -> Fill:
    return Fill(
        order_id=order_id,
        fill_id="test-fill",
        symbol="SYNTH",
        side=OrderSide.BUY,
        price=100.0,
        quantity=10.0,
        commission=0.1,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        slippage=0.0,
    )


# ── Strategy name ─────────────────────────────────────────────────────────────


class TestStrategyName:
    def test_T_MOM_1_name_is_momentum_sentiment(self) -> None:
        strategy = _make_strategy()
        assert strategy.name == "MomentumSentiment"

    def test_T_MOM_2_name_is_string(self) -> None:
        strategy = _make_strategy()
        assert isinstance(strategy.name, str)


# ── No-op when alpha is neutral ───────────────────────────────────────────────


class TestNeutralAlpha:
    def test_T_MOM_3_no_orders_when_alpha_is_zero(self) -> None:
        strategy = _make_strategy()
        bar = _make_bar()
        snap = _make_snapshot(combined_alpha=0.0)
        orders = strategy.on_bar(bar, snap, [])
        assert orders == []

    def test_T_MOM_4_no_orders_below_entry_threshold(self) -> None:
        """alpha just below entry_threshold (0.25) should produce no entry."""
        strategy = _make_strategy()
        bar = _make_bar()
        snap = _make_snapshot(combined_alpha=0.24)
        orders = strategy.on_bar(bar, snap, [])
        assert orders == []

    def test_T_MOM_5_on_bar_returns_list(self) -> None:
        strategy = _make_strategy()
        bar = _make_bar()
        snap = _make_snapshot(combined_alpha=0.0)
        result = strategy.on_bar(bar, snap, [])
        assert isinstance(result, list)


# ── LONG entry ────────────────────────────────────────────────────────────────


class TestLongEntry:
    def test_T_MOM_6_enters_long_above_entry_threshold(self) -> None:
        strategy = _make_strategy()
        bar = _make_bar(close=100.0)
        snap = _make_snapshot(combined_alpha=0.50)
        orders = strategy.on_bar(bar, snap, [])
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].order_type == OrderType.MARKET

    def test_T_MOM_7_long_order_has_positive_quantity(self) -> None:
        strategy = _make_strategy()
        bar = _make_bar(close=100.0)
        snap = _make_snapshot(combined_alpha=0.50)
        orders = strategy.on_bar(bar, snap, [])
        assert orders[0].quantity > 0.0

    def test_T_MOM_8_no_double_long_when_long_exists(self) -> None:
        strategy = _make_strategy()
        bar = _make_bar()
        snap = _make_snapshot(combined_alpha=0.80)
        existing = _make_position(TradeDirection.LONG, bars_held=2)
        orders = strategy.on_bar(bar, snap, [existing])
        buy_orders = [o for o in orders if o.side == OrderSide.BUY]
        assert len(buy_orders) == 0

    def test_T_MOM_9_long_order_symbol_matches_bar(self) -> None:
        strategy = _make_strategy()
        bar = _make_bar(symbol="AAPL")
        snap = _make_snapshot(combined_alpha=0.5, symbol="AAPL")
        orders = strategy.on_bar(bar, snap, [])
        assert orders[0].symbol == "AAPL"


# ── SHORT entry ───────────────────────────────────────────────────────────────


class TestShortEntry:
    def test_T_MOM_10_enters_short_below_negative_threshold(self) -> None:
        strategy = _make_strategy()
        bar = _make_bar(close=100.0)
        snap = _make_snapshot(combined_alpha=-0.50)
        orders = strategy.on_bar(bar, snap, [])
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].order_type == OrderType.MARKET

    def test_T_MOM_11_no_double_short_when_short_exists(self) -> None:
        strategy = _make_strategy()
        bar = _make_bar()
        snap = _make_snapshot(combined_alpha=-0.80)
        existing = _make_position(TradeDirection.SHORT, bars_held=2)
        orders = strategy.on_bar(bar, snap, [existing])
        sell_orders = [o for o in orders if o.side == OrderSide.SELL]
        assert len(sell_orders) == 0

    def test_T_MOM_12_short_quantity_positive(self) -> None:
        strategy = _make_strategy()
        bar = _make_bar(close=100.0)
        snap = _make_snapshot(combined_alpha=-0.50)
        orders = strategy.on_bar(bar, snap, [])
        assert orders[0].quantity > 0.0


# ── Exit on alpha ─────────────────────────────────────────────────────────────


class TestAlphaExit:
    def test_T_MOM_13_exits_long_when_alpha_below_exit_threshold(self) -> None:
        """alpha below exit_threshold (0.05) should close LONG."""
        strategy = _make_strategy()
        bar = _make_bar()
        snap = _make_snapshot(combined_alpha=0.04)  # below exit_threshold=0.05
        pos = _make_position(TradeDirection.LONG, bars_held=3, quantity=5.0)
        orders = strategy.on_bar(bar, snap, [pos])
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].quantity == pytest.approx(5.0)

    def test_T_MOM_14_exits_short_when_alpha_above_neg_exit_threshold(self) -> None:
        """alpha above -exit_threshold should close SHORT."""
        strategy = _make_strategy()
        bar = _make_bar()
        snap = _make_snapshot(combined_alpha=-0.04)  # above -0.05
        pos = _make_position(TradeDirection.SHORT, bars_held=3, quantity=7.0)
        orders = strategy.on_bar(bar, snap, [pos])
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].quantity == pytest.approx(7.0)

    def test_T_MOM_15_long_not_exited_when_alpha_above_threshold(self) -> None:
        """LONG with alpha >= exit_threshold and within max_hold_bars should stay open."""
        strategy = _make_strategy()
        bar = _make_bar()
        snap = _make_snapshot(combined_alpha=0.10)  # above exit_threshold
        pos = _make_position(TradeDirection.LONG, bars_held=2)
        orders = strategy.on_bar(bar, snap, [pos])
        sell_orders = [o for o in orders if o.side == OrderSide.SELL]
        assert len(sell_orders) == 0


# ── Time-stop exit ────────────────────────────────────────────────────────────


class TestTimeStop:
    def test_T_MOM_16_exits_long_after_max_hold_bars(self) -> None:
        """bars_held >= max_hold_bars (24) triggers LONG exit."""
        strategy = _make_strategy()
        bar = _make_bar()
        snap = _make_snapshot(combined_alpha=0.80)  # high enough not to trigger alpha exit
        pos = _make_position(TradeDirection.LONG, bars_held=24)
        orders = strategy.on_bar(bar, snap, [pos])
        sell_orders = [o for o in orders if o.side == OrderSide.SELL]
        assert len(sell_orders) == 1

    def test_T_MOM_17_exits_short_after_max_hold_bars(self) -> None:
        strategy = _make_strategy()
        bar = _make_bar()
        snap = _make_snapshot(combined_alpha=-0.80)
        pos = _make_position(TradeDirection.SHORT, bars_held=24)
        orders = strategy.on_bar(bar, snap, [pos])
        buy_orders = [o for o in orders if o.side == OrderSide.BUY]
        assert len(buy_orders) == 1

    def test_T_MOM_18_no_exit_before_max_hold_bars(self) -> None:
        strategy = _make_strategy()
        bar = _make_bar()
        snap = _make_snapshot(combined_alpha=0.80)
        pos = _make_position(TradeDirection.LONG, bars_held=23)
        orders = strategy.on_bar(bar, snap, [pos])
        sell_orders = [o for o in orders if o.side == OrderSide.SELL]
        assert len(sell_orders) == 0


# ── _kelly_quantity ────────────────────────────────────────────────────────────


class TestKellyQuantity:
    def test_T_MOM_19_zero_price_returns_zero(self) -> None:
        strategy = _make_strategy(portfolio_value=100_000.0)
        qty = strategy._kelly_quantity(alpha=0.5, price=0.0)
        assert qty == pytest.approx(0.0)

    def test_T_MOM_20_positive_alpha_positive_quantity(self) -> None:
        strategy = _make_strategy(portfolio_value=100_000.0)
        qty = strategy._kelly_quantity(alpha=0.5, price=100.0)
        assert qty > 0.0

    def test_T_MOM_21_capped_by_max_position_size_pct(self) -> None:
        """Kelly fraction cannot exceed max_position_size_pct * portfolio_value / price."""
        strategy = _make_strategy(portfolio_value=100_000.0)
        max_usd = 100_000.0 * _RISK_LIMITS.max_position_size_pct  # 20_000
        qty = strategy._kelly_quantity(alpha=0.99, price=100.0)
        assert qty <= max_usd / 100.0 + 1e-9

    def test_T_MOM_22_larger_alpha_produces_larger_quantity(self) -> None:
        strategy = _make_strategy(portfolio_value=100_000.0)
        qty_small = strategy._kelly_quantity(alpha=0.30, price=100.0)
        qty_large = strategy._kelly_quantity(alpha=0.70, price=100.0)
        assert qty_large >= qty_small

    def test_T_MOM_23_negative_alpha_treated_as_absolute_value(self) -> None:
        """_kelly_quantity uses abs(alpha) so negative input gives same result as positive."""
        strategy = _make_strategy()
        qty_pos = strategy._kelly_quantity(alpha=0.5, price=100.0)
        qty_neg = strategy._kelly_quantity(alpha=-0.5, price=100.0)
        assert qty_pos == pytest.approx(qty_neg)


# ── on_fill ───────────────────────────────────────────────────────────────────


class TestOnFill:
    def test_T_MOM_24_on_fill_accepts_fill_without_error(self) -> None:
        strategy = _make_strategy()
        fill = _make_fill()
        strategy.on_fill(fill)  # should not raise

    def test_T_MOM_25_on_fill_returns_none(self) -> None:
        strategy = _make_strategy()
        fill = _make_fill()
        result = strategy.on_fill(fill)
        assert result is None
