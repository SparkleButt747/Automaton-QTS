"""Unit tests for qts.strategies.mean_reversion.

Verifies:
- Strategy name is "mean_reversion"
- Enters long when BB position below oversold threshold with non-bearish sentiment
- Enters short when BB position above overbought threshold with non-bullish sentiment
- Exits when BB position crosses target
- Time-stop exits after max_hold_bars
- Respects sentiment filter
- Doesn't double-enter same symbol
- Protocol conformance (has on_bar, on_fill, name)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from qts.models.base import (
    Bar,
    Fill,
    OrderSide,
    OrderType,
    Position,
    SignalSnapshot,
    TradeDirection,
    VolRegime,
)
from qts.strategies.mean_reversion import MeanReversionStrategy

if TYPE_CHECKING:
    pass


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_bar(
    symbol: str = "SYNTH",
    close: float = 100.0,
    timestamp: datetime | None = None,
) -> Bar:
    """Build a minimal Bar for testing."""
    ts = timestamp or datetime(2024, 1, 1, 12, 0, 0)
    return Bar(
        timestamp=ts,
        symbol=symbol,
        open=close * 0.999,
        high=close * 1.002,
        low=close * 0.998,
        close=close,
        volume=10_000.0,
    )


def _make_snapshot(
    bb_position: float = 0.5,
    sentiment_score: float = 0.0,
    symbol: str = "SYNTH",
    timestamp: datetime | None = None,
) -> SignalSnapshot:
    """Build a SignalSnapshot with configurable bb_position and sentiment_score."""
    ts = timestamp or datetime(2024, 1, 1, 12, 0, 0)
    return SignalSnapshot(
        timestamp=ts,
        symbol=symbol,
        rsi=50.0,
        macd_histogram=0.0,
        macd_line=0.0,
        macd_signal=0.0,
        bb_position=bb_position,
        bb_upper=110.0,
        bb_lower=90.0,
        atr=1.0,
        momentum_5=0.0,
        vol_regime=VolRegime.HIGH,
        vol_regime_confidence=0.8,
        sentiment_score=sentiment_score,
        combined_alpha=0.0,
    )


def _make_position(
    direction: TradeDirection,
    bars_held: int = 0,
    symbol: str = "SYNTH",
    entry_price: float = 100.0,
    quantity: float = 10.0,
) -> Position:
    """Build a Position for testing."""
    return Position(
        symbol=symbol,
        direction=direction,
        entry_price=entry_price,
        quantity=quantity,
        entry_time=datetime(2024, 1, 1),
        bars_held=bars_held,
    )


def _make_fill(
    order_id: str = "test-order",
    symbol: str = "SYNTH",
    side: OrderSide = OrderSide.BUY,
    price: float = 100.0,
    quantity: float = 10.0,
) -> Fill:
    """Build a Fill for testing."""
    return Fill(
        order_id=order_id,
        fill_id="test-fill",
        symbol=symbol,
        side=side,
        price=price,
        quantity=quantity,
        commission=0.1,
        timestamp=datetime(2024, 1, 1, 12, 0, 0),
        slippage=0.0,
    )


# ── Strategy name ──────────────────────────────────────────────────────────────


class TestStrategyName:
    def test_name_is_mean_reversion(self) -> None:
        strategy = MeanReversionStrategy()
        assert strategy.name == "mean_reversion"

    def test_name_is_string(self) -> None:
        strategy = MeanReversionStrategy()
        assert isinstance(strategy.name, str)


# ── Protocol conformance ──────────────────────────────────────────────────────


class TestProtocolConformance:
    def test_has_on_bar(self) -> None:
        strategy = MeanReversionStrategy()
        assert hasattr(strategy, "on_bar")
        assert callable(strategy.on_bar)

    def test_has_on_fill(self) -> None:
        strategy = MeanReversionStrategy()
        assert hasattr(strategy, "on_fill")
        assert callable(strategy.on_fill)

    def test_has_name_property(self) -> None:
        strategy = MeanReversionStrategy()
        assert hasattr(strategy, "name")

    def test_on_bar_returns_list(self) -> None:
        strategy = MeanReversionStrategy()
        bar = _make_bar()
        snapshot = _make_snapshot(bb_position=0.5)
        result = strategy.on_bar(bar, snapshot, [])
        assert isinstance(result, list)

    def test_on_fill_returns_none(self) -> None:
        strategy = MeanReversionStrategy()
        fill = _make_fill()
        result = strategy.on_fill(fill)
        assert result is None


# ── Long entry logic ──────────────────────────────────────────────────────────


class TestLongEntry:
    def test_enters_long_when_bb_below_oversold(self) -> None:
        """bb_position < bb_oversold (0.2) and neutral sentiment -> LONG BUY order."""
        strategy = MeanReversionStrategy(bb_oversold=0.2, bb_overbought=0.8)
        bar = _make_bar(close=100.0)
        snapshot = _make_snapshot(bb_position=0.1, sentiment_score=0.0)

        orders = strategy.on_bar(bar, snapshot, [])

        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY
        assert orders[0].order_type == OrderType.MARKET
        assert orders[0].symbol == "SYNTH"

    def test_long_entry_at_exact_oversold_boundary_no_entry(self) -> None:
        """bb_position == bb_oversold should NOT trigger entry (strict less-than)."""
        strategy = MeanReversionStrategy(bb_oversold=0.2)
        bar = _make_bar()
        snapshot = _make_snapshot(bb_position=0.2)

        orders = strategy.on_bar(bar, snapshot, [])

        assert len(orders) == 0

    def test_long_entry_quantity_positive(self) -> None:
        strategy = MeanReversionStrategy(portfolio_value=100_000.0, position_size=0.03)
        bar = _make_bar(close=100.0)
        snapshot = _make_snapshot(bb_position=0.05)

        orders = strategy.on_bar(bar, snapshot, [])

        assert len(orders) == 1
        assert orders[0].quantity > 0.0

    def test_long_entry_quantity_correct(self) -> None:
        """qty = portfolio_value * position_size / price = 100000 * 0.03 / 100 = 30."""
        strategy = MeanReversionStrategy(portfolio_value=100_000.0, position_size=0.03)
        bar = _make_bar(close=100.0)
        snapshot = _make_snapshot(bb_position=0.05)

        orders = strategy.on_bar(bar, snapshot, [])

        assert orders[0].quantity == pytest.approx(30.0)

    def test_no_entry_in_neutral_zone(self) -> None:
        """bb_position in [0.2, 0.8] should not trigger any entry."""
        strategy = MeanReversionStrategy()
        bar = _make_bar()
        snapshot = _make_snapshot(bb_position=0.5)

        orders = strategy.on_bar(bar, snapshot, [])

        assert orders == []


# ── Short entry logic ─────────────────────────────────────────────────────────


class TestShortEntry:
    def test_enters_short_when_bb_above_overbought(self) -> None:
        """bb_position > bb_overbought (0.8) and neutral sentiment -> SHORT SELL order."""
        strategy = MeanReversionStrategy(bb_oversold=0.2, bb_overbought=0.8)
        bar = _make_bar(close=100.0)
        snapshot = _make_snapshot(bb_position=0.9, sentiment_score=0.0)

        orders = strategy.on_bar(bar, snapshot, [])

        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL
        assert orders[0].order_type == OrderType.MARKET

    def test_short_entry_at_exact_overbought_boundary_no_entry(self) -> None:
        """bb_position == bb_overbought should NOT trigger entry (strict greater-than)."""
        strategy = MeanReversionStrategy(bb_overbought=0.8)
        bar = _make_bar()
        snapshot = _make_snapshot(bb_position=0.8)

        orders = strategy.on_bar(bar, snapshot, [])

        assert len(orders) == 0


# ── Exit logic ────────────────────────────────────────────────────────────────


class TestLongExit:
    def test_exits_long_when_bb_crosses_target(self) -> None:
        """bb_position > bb_target (0.5) should trigger LONG exit (SELL)."""
        strategy = MeanReversionStrategy(bb_target=0.5)
        bar = _make_bar()
        snapshot = _make_snapshot(bb_position=0.6)  # above target
        long_pos = _make_position(TradeDirection.LONG, bars_held=5)

        orders = strategy.on_bar(bar, snapshot, [long_pos])

        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL

    def test_long_not_exited_below_target(self) -> None:
        """bb_position <= bb_target should NOT exit a LONG (without time-stop)."""
        strategy = MeanReversionStrategy(bb_target=0.5, max_hold_bars=100)
        bar = _make_bar()
        snapshot = _make_snapshot(bb_position=0.4)  # below target
        long_pos = _make_position(TradeDirection.LONG, bars_held=2)

        orders = strategy.on_bar(bar, snapshot, [long_pos])

        assert len(orders) == 0

    def test_long_exit_quantity_matches_position(self) -> None:
        strategy = MeanReversionStrategy(bb_target=0.5)
        bar = _make_bar()
        snapshot = _make_snapshot(bb_position=0.7)
        long_pos = _make_position(TradeDirection.LONG, quantity=25.0)

        orders = strategy.on_bar(bar, snapshot, [long_pos])

        assert len(orders) == 1
        assert orders[0].quantity == pytest.approx(25.0)


class TestShortExit:
    def test_exits_short_when_bb_crosses_below_target(self) -> None:
        """bb_position < (1 - bb_target) = 0.5 should trigger SHORT exit (BUY)."""
        strategy = MeanReversionStrategy(bb_target=0.5)
        bar = _make_bar()
        snapshot = _make_snapshot(bb_position=0.4)  # below (1-0.5)=0.5
        short_pos = _make_position(TradeDirection.SHORT, bars_held=3)

        orders = strategy.on_bar(bar, snapshot, [short_pos])

        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY

    def test_short_not_exited_above_target(self) -> None:
        """bb_position >= (1 - bb_target) should NOT exit a SHORT (without time-stop)."""
        strategy = MeanReversionStrategy(bb_target=0.5, max_hold_bars=100)
        bar = _make_bar()
        snapshot = _make_snapshot(bb_position=0.6)  # above (1-0.5)=0.5
        short_pos = _make_position(TradeDirection.SHORT, bars_held=2)

        orders = strategy.on_bar(bar, snapshot, [short_pos])

        assert len(orders) == 0


# ── Time-stop exit ────────────────────────────────────────────────────────────


class TestTimeStop:
    def test_time_stop_exits_long_after_max_hold_bars(self) -> None:
        """LONG position held longer than max_hold_bars should be closed."""
        strategy = MeanReversionStrategy(max_hold_bars=10, bb_target=0.5)
        bar = _make_bar()
        # bb_position below target, so won't exit on target — only time-stop
        snapshot = _make_snapshot(bb_position=0.3)
        long_pos = _make_position(TradeDirection.LONG, bars_held=11)

        orders = strategy.on_bar(bar, snapshot, [long_pos])

        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL

    def test_time_stop_exits_short_after_max_hold_bars(self) -> None:
        """SHORT position held longer than max_hold_bars should be closed."""
        strategy = MeanReversionStrategy(max_hold_bars=10, bb_target=0.5)
        bar = _make_bar()
        # bb_position above (1-bb_target)=0.5, so won't exit on target — only time-stop
        snapshot = _make_snapshot(bb_position=0.7)
        short_pos = _make_position(TradeDirection.SHORT, bars_held=11)

        orders = strategy.on_bar(bar, snapshot, [short_pos])

        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY

    def test_no_time_stop_before_max_hold_bars(self) -> None:
        """LONG position within max_hold_bars should NOT be time-stopped."""
        strategy = MeanReversionStrategy(max_hold_bars=24, bb_target=0.5)
        bar = _make_bar()
        snapshot = _make_snapshot(bb_position=0.3)  # below target, no exit on target
        long_pos = _make_position(TradeDirection.LONG, bars_held=20)

        orders = strategy.on_bar(bar, snapshot, [long_pos])

        assert len(orders) == 0

    def test_time_stop_exactly_at_max_hold_bars_no_exit(self) -> None:
        """bars_held == max_hold_bars should NOT trigger time-stop (strict greater-than)."""
        strategy = MeanReversionStrategy(max_hold_bars=10, bb_target=0.5)
        bar = _make_bar()
        snapshot = _make_snapshot(bb_position=0.3)
        long_pos = _make_position(TradeDirection.LONG, bars_held=10)

        orders = strategy.on_bar(bar, snapshot, [long_pos])

        assert len(orders) == 0

    def test_time_stop_one_bar_over_triggers_exit(self) -> None:
        """bars_held == max_hold_bars + 1 should trigger time-stop."""
        strategy = MeanReversionStrategy(max_hold_bars=10, bb_target=0.5)
        bar = _make_bar()
        snapshot = _make_snapshot(bb_position=0.3)
        long_pos = _make_position(TradeDirection.LONG, bars_held=11)

        orders = strategy.on_bar(bar, snapshot, [long_pos])

        assert len(orders) == 1


# ── Sentiment filter ──────────────────────────────────────────────────────────


class TestSentimentFilter:
    def test_blocks_long_on_extreme_bearish_sentiment(self) -> None:
        """LONG entry blocked when sentiment < -sentiment_confirm_threshold."""
        strategy = MeanReversionStrategy(sentiment_confirm_threshold=0.1)
        bar = _make_bar()
        # bb_position is oversold, but sentiment is strongly negative
        snapshot = _make_snapshot(bb_position=0.1, sentiment_score=-0.5)

        orders = strategy.on_bar(bar, snapshot, [])

        assert len(orders) == 0

    def test_allows_long_with_mildly_negative_sentiment(self) -> None:
        """LONG allowed when sentiment is mildly negative (> -threshold)."""
        strategy = MeanReversionStrategy(sentiment_confirm_threshold=0.5)
        bar = _make_bar()
        snapshot = _make_snapshot(bb_position=0.1, sentiment_score=-0.3)  # > -0.5

        orders = strategy.on_bar(bar, snapshot, [])

        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY

    def test_blocks_short_on_extreme_bullish_sentiment(self) -> None:
        """SHORT entry blocked when sentiment > sentiment_confirm_threshold."""
        strategy = MeanReversionStrategy(sentiment_confirm_threshold=0.1)
        bar = _make_bar()
        # bb_position is overbought, but sentiment is strongly positive
        snapshot = _make_snapshot(bb_position=0.9, sentiment_score=0.5)

        orders = strategy.on_bar(bar, snapshot, [])

        assert len(orders) == 0

    def test_allows_short_with_mildly_positive_sentiment(self) -> None:
        """SHORT allowed when sentiment is mildly positive (< threshold)."""
        strategy = MeanReversionStrategy(sentiment_confirm_threshold=0.5)
        bar = _make_bar()
        snapshot = _make_snapshot(bb_position=0.9, sentiment_score=0.3)  # < 0.5

        orders = strategy.on_bar(bar, snapshot, [])

        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL

    def test_allows_long_with_zero_sentiment(self) -> None:
        """LONG allowed when sentiment is exactly zero."""
        strategy = MeanReversionStrategy(sentiment_confirm_threshold=0.1)
        bar = _make_bar()
        snapshot = _make_snapshot(bb_position=0.1, sentiment_score=0.0)

        orders = strategy.on_bar(bar, snapshot, [])

        assert len(orders) == 1

    def test_long_blocked_at_exact_threshold(self) -> None:
        """LONG blocked when sentiment == -threshold (equal to the boundary)."""
        strategy = MeanReversionStrategy(sentiment_confirm_threshold=0.3)
        bar = _make_bar()
        # sentiment == -threshold → NOT > -threshold → blocked
        snapshot = _make_snapshot(bb_position=0.1, sentiment_score=-0.3)

        orders = strategy.on_bar(bar, snapshot, [])

        assert len(orders) == 0

    def test_short_blocked_at_exact_threshold(self) -> None:
        """SHORT blocked when sentiment == +threshold."""
        strategy = MeanReversionStrategy(sentiment_confirm_threshold=0.3)
        bar = _make_bar()
        snapshot = _make_snapshot(bb_position=0.9, sentiment_score=0.3)

        orders = strategy.on_bar(bar, snapshot, [])

        assert len(orders) == 0


# ── No double-entry ───────────────────────────────────────────────────────────


class TestNoDoubleEntry:
    def test_no_new_long_when_long_already_open(self) -> None:
        """Should not open a second LONG position if one is already open."""
        strategy = MeanReversionStrategy(bb_target=0.9, max_hold_bars=100)
        bar = _make_bar()
        # bb_position is below target (won't exit) and below oversold (would enter)
        snapshot = _make_snapshot(bb_position=0.05)
        existing_long = _make_position(TradeDirection.LONG, bars_held=2)

        orders = strategy.on_bar(bar, snapshot, [existing_long])

        # No new entry orders — the existing LONG should keep the position slot occupied
        buy_orders = [o for o in orders if o.side == OrderSide.BUY]
        assert len(buy_orders) == 0

    def test_no_new_short_when_short_already_open(self) -> None:
        """Should not open a second SHORT position if one is already open."""
        strategy = MeanReversionStrategy(bb_target=0.1, max_hold_bars=100)
        bar = _make_bar()
        # bb_position is above (1-target)=0.9 (won't exit) and above overbought (would enter)
        snapshot = _make_snapshot(bb_position=0.95)
        existing_short = _make_position(TradeDirection.SHORT, bars_held=2)

        orders = strategy.on_bar(bar, snapshot, [existing_short])

        sell_orders = [o for o in orders if o.side == OrderSide.SELL]
        assert len(sell_orders) == 0

    def test_exits_then_can_reenter_in_same_bar(self) -> None:
        """After a LONG exit (via target), no immediate re-entry in same call."""
        strategy = MeanReversionStrategy(bb_oversold=0.2, bb_overbought=0.8, bb_target=0.5)
        bar = _make_bar()
        # bb_position triggers LONG exit (above target) — NOT in the oversold zone
        snapshot = _make_snapshot(bb_position=0.7)
        long_pos = _make_position(TradeDirection.LONG, bars_held=3)

        orders = strategy.on_bar(bar, snapshot, [long_pos])

        # Should exit LONG only; bb_position=0.7 is not in [0, 0.2) so no new long
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL


# ── Constructor validation ────────────────────────────────────────────────────


class TestConstructorValidation:
    def test_invalid_oversold_overbought_raises(self) -> None:
        with pytest.raises(ValueError):
            MeanReversionStrategy(bb_oversold=0.8, bb_overbought=0.2)

    def test_oversold_equal_overbought_raises(self) -> None:
        with pytest.raises(ValueError):
            MeanReversionStrategy(bb_oversold=0.5, bb_overbought=0.5)

    def test_invalid_bb_target_raises(self) -> None:
        with pytest.raises(ValueError):
            MeanReversionStrategy(bb_target=0.0)

    def test_invalid_max_hold_bars_raises(self) -> None:
        with pytest.raises(ValueError):
            MeanReversionStrategy(max_hold_bars=0)

    def test_invalid_position_size_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            MeanReversionStrategy(position_size=0.0)

    def test_valid_defaults_no_raise(self) -> None:
        strategy = MeanReversionStrategy()
        assert strategy is not None


# ── Position sizing ───────────────────────────────────────────────────────────


class TestPositionSizing:
    def test_zero_price_returns_no_order(self) -> None:
        """Price of 0 should produce no order (qty=0 means no trade)."""
        strategy = MeanReversionStrategy()
        bar = _make_bar(close=0.0)
        snapshot = _make_snapshot(bb_position=0.05)

        orders = strategy.on_bar(bar, snapshot, [])

        assert len(orders) == 0

    def test_position_size_scales_with_portfolio_value(self) -> None:
        """Larger portfolio value => larger position quantity."""
        s_small = MeanReversionStrategy(portfolio_value=10_000.0, position_size=0.05)
        s_large = MeanReversionStrategy(portfolio_value=1_000_000.0, position_size=0.05)
        bar = _make_bar(close=100.0)
        snapshot = _make_snapshot(bb_position=0.05)

        orders_small = s_small.on_bar(bar, snapshot, [])
        orders_large = s_large.on_bar(bar, snapshot, [])

        assert len(orders_small) == 1
        assert len(orders_large) == 1
        assert orders_large[0].quantity > orders_small[0].quantity

    def test_max_position_size_pct_caps_quantity(self) -> None:
        """max_position_size_pct < position_size should cap the effective size."""
        strategy = MeanReversionStrategy(
            portfolio_value=100_000.0,
            position_size=0.10,
            max_position_size_pct=0.02,
        )
        bar = _make_bar(close=100.0)
        snapshot = _make_snapshot(bb_position=0.05)

        orders = strategy.on_bar(bar, snapshot, [])

        assert len(orders) == 1
        # Expected: 100_000 * 0.02 / 100 = 20.0
        assert orders[0].quantity == pytest.approx(20.0)


# ── on_fill ───────────────────────────────────────────────────────────────────


class TestOnFill:
    def test_on_fill_accepts_buy(self) -> None:
        strategy = MeanReversionStrategy()
        fill = _make_fill(side=OrderSide.BUY)
        strategy.on_fill(fill)  # should not raise

    def test_on_fill_accepts_sell(self) -> None:
        strategy = MeanReversionStrategy()
        fill = _make_fill(side=OrderSide.SELL)
        strategy.on_fill(fill)  # should not raise
