"""Core domain models for the trading system."""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass
from datetime import datetime


class OrderSide(enum.StrEnum):
    """Trade direction."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(enum.StrEnum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(enum.StrEnum):
    """Order lifecycle status."""

    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class TradeDirection(enum.StrEnum):
    """Position direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class TradeOutcome(enum.StrEnum):
    """Trade result classification."""

    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"


class ExitReason(enum.StrEnum):
    """Why a trade was closed."""

    ALPHA_FLIP = "ALPHA_FLIP"
    TIME_STOP = "TIME_STOP"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    MANUAL = "MANUAL"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class VolLevel(enum.StrEnum):
    """Volatility level within a macro regime."""

    HIGH = "HIGH"
    LOW = "LOW"
    TRANSITIONING = "TRANSITIONING"


class Trend(enum.StrEnum):
    """Market trend direction."""

    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"


class LiquidityLevel(enum.StrEnum):
    """Market liquidity conditions."""

    ABUNDANT = "ABUNDANT"
    TIGHT = "TIGHT"
    CRISIS = "CRISIS"


class SentimentLevel(enum.StrEnum):
    """Market sentiment state."""

    EUPHORIC = "EUPHORIC"
    FEARFUL = "FEARFUL"
    NEUTRAL = "NEUTRAL"


class Catalyst(enum.StrEnum):
    """Event catalyst driving the current regime."""

    EARNINGS = "EARNINGS"
    MACRO_EVENT = "MACRO_EVENT"
    GEOPOLITICAL = "GEOPOLITICAL"
    PANDEMIC = "PANDEMIC"
    REGULATORY = "REGULATORY"
    NONE = "NONE"


class FailureMode(enum.StrEnum):
    """Failure mode taxonomy for loss attribution."""

    SENTIMENT_FADE = "SENTIMENT_FADE"
    NEWS_FRONT_RUN = "NEWS_FRONT_RUN"
    REGIME_MISMATCH = "REGIME_MISMATCH"
    EXECUTION_SLIP = "EXECUTION_SLIP"
    STOP_HUNT = "STOP_HUNT"
    MACRO_OVERRIDE = "MACRO_OVERRIDE"
    FALSE_ALPHA = "FALSE_ALPHA"


@dataclass(frozen=True, slots=True)
class Tick:
    """Single trade tick from exchange."""

    timestamp: datetime
    symbol: str
    price: float
    quantity: float
    side: OrderSide


@dataclass(frozen=True, slots=True)
class Bar:
    """OHLCV bar."""

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bar_count: int = 0


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    """Single price level in order book."""

    price: float
    quantity: float


@dataclass(frozen=True, slots=True)
class OrderBookSnapshot:
    """L2 order book snapshot."""

    timestamp: datetime
    symbol: str
    bids: tuple[OrderBookLevel, ...]
    asks: tuple[OrderBookLevel, ...]


@dataclass(frozen=True, slots=True)
class Order:
    """Order to be submitted."""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    stop_price: float | None = None
    timestamp: datetime | None = None


@dataclass(frozen=True, slots=True)
class Fill:
    """Execution fill report."""

    order_id: str
    fill_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    commission: float
    timestamp: datetime
    slippage: float = 0.0


@dataclass(slots=True)
class Position:
    """Open position state."""

    symbol: str
    direction: TradeDirection
    entry_price: float
    quantity: float
    entry_time: datetime
    unrealized_pnl: float = 0.0
    bars_held: int = 0


@dataclass(frozen=True, slots=True)
class SignalSnapshot:
    """Complete signal state at a point in time."""

    timestamp: datetime
    symbol: str
    rsi: float
    macd_histogram: float
    macd_line: float
    macd_signal: float
    bb_position: float
    bb_upper: float
    bb_lower: float
    atr: float
    momentum_5: float
    vol_level: VolLevel
    vol_level_confidence: float
    sentiment_score: float = 0.0
    combined_alpha: float = 0.0


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """Complete trade record with signal context for attribution."""

    trade_id: str
    symbol: str
    entry_time: datetime
    exit_time: datetime
    direction: TradeDirection
    entry_price: float
    exit_price: float
    size: float
    pnl_usd: float
    pnl_pct: float
    slippage_usd: float
    fees_usd: float
    rsi_at_entry: float
    macd_at_entry: float
    sentiment_at_entry: float
    social_velocity_at_entry: float
    gdelt_intensity_at_entry: float
    vol_level_at_entry: VolLevel
    combined_alpha_at_entry: float
    params_version: str
    outcome: TradeOutcome
    exit_reason: ExitReason
    failure_mode: FailureMode | None = None

    @staticmethod
    def generate_id() -> str:
        """Generate a unique trade ID.

        Returns:
            A UUID4 string suitable for use as a trade identifier.
        """
        return str(uuid.uuid4())
