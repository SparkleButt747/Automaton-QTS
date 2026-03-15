"""Database sub-package: SQLAlchemy engine, session management, and Alembic migrations.

Exports the key objects needed to set up and interact with the database:
- create_engine: Factory to create an async SQLAlchemy engine
- create_session_factory: Factory to create an async session maker
- get_session: Async context manager yielding a database session
- metadata: Shared SQLAlchemy MetaData with naming conventions
- Base: Declarative base class for all ORM-mapped tables
- MarketDataRepository: Repository for tick and OHLCV bar data
- SignalRepository: Repository for signal snapshot data
- TradeRepository: Repository for trade records
"""
from __future__ import annotations

from qts.db.engine import create_engine, create_session_factory, get_session, metadata
from qts.db.repository import MarketDataRepository, SignalRepository, TradeRepository
from qts.db.tables import (
    BarRow,
    Base,
    GeopoliticalEventRow,
    NewsArticleRow,
    SignalSnapshotRow,
    TickRow,
    TradeRecordRow,
)

__all__ = [
    "Base",
    "BarRow",
    "GeopoliticalEventRow",
    "MarketDataRepository",
    "NewsArticleRow",
    "SignalRepository",
    "SignalSnapshotRow",
    "TickRow",
    "TradeRecordRow",
    "TradeRepository",
    "create_engine",
    "create_session_factory",
    "get_session",
    "metadata",
]
