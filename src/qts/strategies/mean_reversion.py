"""Mean reversion strategy with Bollinger Band + sentiment overlay.

Entry rules:
    LONG:  bb_position < bb_oversold  AND  sentiment_score > -sentiment_confirm_threshold
    SHORT: bb_position > bb_overbought AND  sentiment_score <  sentiment_confirm_threshold

Exit rules:
    LONG:  bb_position > bb_target                OR bars_held > max_hold_bars
    SHORT: bb_position < (1 - bb_target)          OR bars_held > max_hold_bars

Position sizing:
    Fixed fraction of portfolio value, capped by risk limits.
"""

from __future__ import annotations

import logging
import uuid

from qts.models.base import (
    Bar,
    Fill,
    Order,
    OrderSide,
    OrderType,
    Position,
    SignalSnapshot,
    TradeDirection,
)

logger = logging.getLogger(__name__)


class MeanReversionStrategy:
    """Bollinger Band mean-reversion strategy with optional sentiment overlay.

    The strategy waits for price to reach an extreme Bollinger Band position
    (oversold or overbought) and then enters a mean-reversion trade toward
    the centre of the band.  An optional sentiment filter prevents entering
    trades that contradict an extreme directional sentiment signal.

    Entry rules:
        - **LONG**:  ``bb_position < bb_oversold``  AND
          ``sentiment_score > -sentiment_confirm_threshold``
          (i.e. not extremely bearish)
        - **SHORT**: ``bb_position > bb_overbought`` AND
          ``sentiment_score < sentiment_confirm_threshold``
          (i.e. not extremely bullish)

    Exit rules:
        - **LONG exit**:  ``bb_position > bb_target`` OR
          ``bars_held > max_hold_bars``
        - **SHORT exit**: ``bb_position < (1 - bb_target)`` OR
          ``bars_held > max_hold_bars``

    Position sizing:
        Fixed fraction (``position_size``) of ``portfolio_value``, capped by
        ``max_position_size_pct`` from risk limits when provided.

    Attributes:
        bb_oversold: Bollinger Band position threshold for long entries (default 0.2).
        bb_overbought: Bollinger Band position threshold for short entries (default 0.8).
        bb_target: Band position at which to exit (default 0.5 = mid-band).
        sentiment_confirm_threshold: Minimum sentiment magnitude to block entry (default 0.1).
        max_hold_bars: Maximum bars to hold before forced time-stop exit (default 24).
        position_size: Fixed fraction of portfolio value per position (default 0.03).
        portfolio_value: Notional portfolio size for sizing calculations.
        max_position_size_pct: Risk-limit cap on position size as fraction (default 1.0).
    """

    def __init__(
        self,
        bb_oversold: float = 0.2,
        bb_overbought: float = 0.8,
        bb_target: float = 0.5,
        sentiment_confirm_threshold: float = 0.1,
        max_hold_bars: int = 24,
        position_size: float = 0.03,
        portfolio_value: float = 100_000.0,
        max_position_size_pct: float = 1.0,
    ) -> None:
        """Initialise the MeanReversionStrategy.

        Args:
            bb_oversold: Bollinger Band position below which to consider LONG entry.
            bb_overbought: Bollinger Band position above which to consider SHORT entry.
            bb_target: Band position at which LONG/SHORT exits are triggered.
            sentiment_confirm_threshold: Magnitude threshold for the sentiment filter.
                A LONG entry is blocked when ``sentiment_score < -threshold``;
                a SHORT entry is blocked when ``sentiment_score > threshold``.
            max_hold_bars: Time-stop: maximum bars to hold a position.
            position_size: Fixed fraction of ``portfolio_value`` per trade.
            portfolio_value: Notional portfolio value used for sizing.
            max_position_size_pct: Hard cap on position size as fraction of portfolio.
        """
        if not 0.0 <= bb_oversold < bb_overbought <= 1.0:
            raise ValueError(
                f"Must satisfy 0 <= bb_oversold ({bb_oversold}) "
                f"< bb_overbought ({bb_overbought}) <= 1"
            )
        if not 0.0 < bb_target <= 1.0:
            raise ValueError(f"bb_target must be in (0, 1], got {bb_target}")
        if max_hold_bars < 1:
            raise ValueError(f"max_hold_bars must be >= 1, got {max_hold_bars}")
        if not 0.0 < position_size <= 1.0:
            raise ValueError(f"position_size must be in (0, 1], got {position_size}")

        self.bb_oversold = bb_oversold
        self.bb_overbought = bb_overbought
        self.bb_target = bb_target
        self.sentiment_confirm_threshold = sentiment_confirm_threshold
        self.max_hold_bars = max_hold_bars
        self.position_size = position_size
        self.portfolio_value = portfolio_value
        self.max_position_size_pct = max_position_size_pct

    # ------------------------------------------------------------------
    # Strategy protocol
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return strategy name."""
        return "mean_reversion"

    def on_bar(
        self,
        bar: Bar,
        snapshot: SignalSnapshot,
        positions: list[Position],
    ) -> list[Order]:
        """Process a new bar and return any orders to execute.

        Evaluates exit conditions first (to free up the position slot before
        considering new entries), then evaluates entry conditions.

        Args:
            bar: The newly closed OHLCV bar.
            snapshot: Latest signal snapshot containing ``bb_position`` and
                ``sentiment_score``.
            positions: List of currently open positions (any symbol).

        Returns:
            List of :class:`~qts.models.base.Order` objects to submit.
        """
        bb_pos = snapshot.bb_position
        sentiment = snapshot.sentiment_score
        orders: list[Order] = []

        existing_long: Position | None = next(
            (p for p in positions if p.direction == TradeDirection.LONG), None
        )
        existing_short: Position | None = next(
            (p for p in positions if p.direction == TradeDirection.SHORT), None
        )

        # ── Exit logic ───────────────────────────────────────────────────────
        if existing_long is not None:
            time_stop = existing_long.bars_held > self.max_hold_bars
            target_reached = bb_pos > self.bb_target
            if time_stop or target_reached:
                reason = "time_stop" if time_stop else "bb_target"
                logger.debug(
                    "MeanReversion: closing LONG reason=%s bb_pos=%.3f bars_held=%d",
                    reason,
                    bb_pos,
                    existing_long.bars_held,
                )
                orders.append(
                    Order(
                        order_id=str(uuid.uuid4()),
                        symbol=bar.symbol,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=existing_long.quantity,
                        timestamp=bar.timestamp,
                    )
                )
                existing_long = None

        if existing_short is not None:
            time_stop = existing_short.bars_held > self.max_hold_bars
            target_reached = bb_pos < (1.0 - self.bb_target)
            if time_stop or target_reached:
                reason = "time_stop" if time_stop else "bb_target"
                logger.debug(
                    "MeanReversion: closing SHORT reason=%s bb_pos=%.3f bars_held=%d",
                    reason,
                    bb_pos,
                    existing_short.bars_held,
                )
                orders.append(
                    Order(
                        order_id=str(uuid.uuid4()),
                        symbol=bar.symbol,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=existing_short.quantity,
                        timestamp=bar.timestamp,
                    )
                )
                existing_short = None

        # ── Entry logic ──────────────────────────────────────────────────────
        if existing_long is None and existing_short is None:
            qty = self._position_quantity(bar.close)

            if qty > 0.0:
                if bb_pos < self.bb_oversold:
                    # Sentiment filter: block LONG entry on extreme bearish signal
                    if sentiment > -self.sentiment_confirm_threshold:
                        logger.debug(
                            "MeanReversion: opening LONG bb_pos=%.3f sentiment=%.3f qty=%.4f",
                            bb_pos,
                            sentiment,
                            qty,
                        )
                        orders.append(
                            Order(
                                order_id=str(uuid.uuid4()),
                                symbol=bar.symbol,
                                side=OrderSide.BUY,
                                order_type=OrderType.MARKET,
                                quantity=qty,
                                timestamp=bar.timestamp,
                            )
                        )
                    else:
                        logger.debug(
                            "MeanReversion: LONG entry blocked by bearish sentiment=%.3f",
                            sentiment,
                        )

                elif bb_pos > self.bb_overbought:
                    # Sentiment filter: block SHORT entry on extreme bullish signal
                    if sentiment < self.sentiment_confirm_threshold:
                        logger.debug(
                            "MeanReversion: opening SHORT bb_pos=%.3f sentiment=%.3f qty=%.4f",
                            bb_pos,
                            sentiment,
                            qty,
                        )
                        orders.append(
                            Order(
                                order_id=str(uuid.uuid4()),
                                symbol=bar.symbol,
                                side=OrderSide.SELL,
                                order_type=OrderType.MARKET,
                                quantity=qty,
                                timestamp=bar.timestamp,
                            )
                        )
                    else:
                        logger.debug(
                            "MeanReversion: SHORT entry blocked by bullish sentiment=%.3f",
                            sentiment,
                        )

        return orders

    def on_fill(self, fill: Fill) -> None:
        """Handle fill notification (informational logging only).

        Args:
            fill: Execution fill report.
        """
        logger.debug(
            "MeanReversion: fill received order_id=%s side=%s qty=%.4f price=%.4f",
            fill.order_id,
            fill.side,
            fill.quantity,
            fill.price,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _position_quantity(self, price: float) -> float:
        """Compute the order quantity based on fixed-fraction position sizing.

        Formula::

            notional = portfolio_value * min(position_size, max_position_size_pct)
            quantity  = notional / price

        Args:
            price: Current asset price.

        Returns:
            Order quantity as a positive float, or 0.0 if price is non-positive.
        """
        if price <= 0.0:
            return 0.0
        effective_size = min(self.position_size, self.max_position_size_pct)
        notional = self.portfolio_value * effective_size
        return notional / price
