"""SQLAlchemy ORM table definitions for the trading system.

Defines mapped tables for:
- ticks: Individual trade ticks from exchanges
- bars: OHLCV bar data
- signal_snapshots: Technical and sentiment signal state at a point in time
- trade_records: Complete trade records with signal context for attribution
- news_articles: Ingested news articles for sentiment analysis
- geopolitical_events: Geopolitical events from GDELT and similar sources
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from qts.db.engine import metadata


class Base(DeclarativeBase):
    """Base class for all ORM-mapped tables.

    Uses the shared metadata instance with naming conventions for constraints.
    """

    metadata = metadata


class TickRow(Base):
    """ORM-mapped table for raw trade ticks.

    Each row represents a single trade tick received from an exchange.
    """

    __tablename__ = "ticks"
    __table_args__ = (
        Index("ix_ticks_timestamp_symbol", "timestamp", "symbol"),
        Index("ix_ticks_symbol", "symbol"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)  # BUY | SELL


class BarRow(Base):
    """ORM-mapped table for OHLCV bar data.

    Each row represents a single OHLCV candle for a given symbol and interval.
    """

    __tablename__ = "bars"
    __table_args__ = (
        Index("ix_bars_timestamp_symbol", "timestamp", "symbol"),
        Index("ix_bars_symbol", "symbol"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    interval: Mapped[str] = mapped_column(String(8), nullable=False, default="1m")
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False)
    bar_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class SignalSnapshotRow(Base):
    """ORM-mapped table for complete signal state snapshots.

    Each row captures the full signal state at a specific timestamp,
    used for attribution and post-trade analysis.
    """

    __tablename__ = "signal_snapshots"
    __table_args__ = (
        Index("ix_signal_snapshots_timestamp_symbol", "timestamp", "symbol"),
        Index("ix_signal_snapshots_symbol", "symbol"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    rsi: Mapped[float] = mapped_column(Float, nullable=False)
    macd_histogram: Mapped[float] = mapped_column(Float, nullable=False)
    macd_line: Mapped[float] = mapped_column(Float, nullable=False)
    macd_signal: Mapped[float] = mapped_column(Float, nullable=False)
    bb_position: Mapped[float] = mapped_column(Float, nullable=False)
    bb_upper: Mapped[float] = mapped_column(Float, nullable=False)
    bb_lower: Mapped[float] = mapped_column(Float, nullable=False)
    atr: Mapped[float] = mapped_column(Float, nullable=False)
    momentum_5: Mapped[float] = mapped_column(Float, nullable=False)
    vol_regime: Mapped[str] = mapped_column(String(8), nullable=False)  # HIGH | LOW
    vol_regime_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    combined_alpha: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)


class TradeRecordRow(Base):
    """ORM-mapped table for complete trade records.

    Stores full trade lifecycle data including entry/exit context,
    signal state at entry, and post-trade attribution metadata.
    """

    __tablename__ = "trade_records"
    __table_args__ = (
        Index("ix_trade_records_entry_time", "entry_time"),
        Index("ix_trade_records_symbol", "symbol"),
        Index("ix_trade_records_exit_time", "exit_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    exit_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    direction: Mapped[str] = mapped_column(String(8), nullable=False)  # LONG | SHORT
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_price: Mapped[float] = mapped_column(Float, nullable=False)
    size: Mapped[float] = mapped_column(Float, nullable=False)
    pnl_usd: Mapped[float] = mapped_column(Float, nullable=False)
    pnl_pct: Mapped[float] = mapped_column(Float, nullable=False)
    slippage_usd: Mapped[float] = mapped_column(Float, nullable=False)
    fees_usd: Mapped[float] = mapped_column(Float, nullable=False)
    rsi_at_entry: Mapped[float] = mapped_column(Float, nullable=False)
    macd_at_entry: Mapped[float] = mapped_column(Float, nullable=False)
    sentiment_at_entry: Mapped[float] = mapped_column(Float, nullable=False)
    social_velocity_at_entry: Mapped[float] = mapped_column(Float, nullable=False)
    gdelt_intensity_at_entry: Mapped[float] = mapped_column(Float, nullable=False)
    vol_regime_at_entry: Mapped[str] = mapped_column(String(8), nullable=False)
    combined_alpha_at_entry: Mapped[float] = mapped_column(Float, nullable=False)
    params_version: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)  # WIN | LOSS | BREAKEVEN
    exit_reason: Mapped[str] = mapped_column(String(32), nullable=False)
    failure_mode: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)


class NewsArticleRow(Base):
    """ORM-mapped table for ingested news articles.

    Stores raw article data along with derived sentiment scores
    used for news-based signal generation.
    """

    __tablename__ = "news_articles"
    __table_args__ = (
        Index("ix_news_articles_published_at", "published_at"),
        Index("ix_news_articles_symbol", "symbol"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    symbol: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    headline: Mapped[str] = mapped_column(String(512), nullable=False)
    body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    relevance_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class GeopoliticalEventRow(Base):
    """ORM-mapped table for geopolitical events.

    Stores events sourced from GDELT and similar feeds,
    with intensity and sentiment metadata for geopolitical signal generation.
    """

    __tablename__ = "geopolitical_events"
    __table_args__ = (
        Index("ix_geopolitical_events_event_date", "event_date"),
        Index("ix_geopolitical_events_country_code", "country_code"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    event_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str] = mapped_column(String(128), nullable=False, default="GDELT")
    country_code: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    event_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    intensity: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    sentiment_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    goldstein_scale: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    num_mentions: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    num_sources: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
