"""Database engine factory and session management."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

logger = logging.getLogger(__name__)

# Naming convention for constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)


def create_engine(database_url: str, echo: bool = False) -> AsyncEngine:
    """Create async database engine.

    Args:
        database_url: Database connection URL. For SQLite, use 'sqlite+aiosqlite:///path'.
        echo: If True, log all SQL statements.

    Returns:
        Configured async engine.
    """
    connect_args: dict[str, object] = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    return create_async_engine(
        database_url,
        echo=echo,
        connect_args=connect_args,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create session factory bound to engine.

    Args:
        engine: Async database engine.

    Returns:
        Session factory producing AsyncSession instances.
    """
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def get_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session, ensuring cleanup.

    Args:
        session_factory: Factory to create sessions.

    Yields:
        Database session.
    """
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def ensure_tables() -> AsyncEngine:
    """Create the engine from settings and ensure all tables exist.

    Converts bare ``sqlite:///`` URLs to ``sqlite+aiosqlite:///`` so that the
    async driver is used automatically.

    Returns:
        The configured :class:`AsyncEngine` with tables guaranteed to exist.
    """
    from qts.config import get_settings

    url = get_settings().database.database_url

    # Ensure async driver for SQLite URLs
    if url.startswith("sqlite:///") and "+aiosqlite" not in url:
        url = url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)

    engine = create_engine(url)
    await init_db(engine)
    logger.info("Database tables ensured at %s", url.split("?")[0])
    return engine


async def init_db(engine: AsyncEngine) -> None:
    """Create all database tables. For dev/testing use.

    Runs :meth:`sqlalchemy.schema.MetaData.create_all` inside an async
    connection context so that the full ORM table schema is materialised in
    the target database.  Safe to call multiple times (``checkfirst=True`` is
    the SQLAlchemy default for ``create_all``).

    Args:
        engine: Async database engine to use for table creation.
    """
    from qts.db.tables import Base  # local import to avoid circular dependency

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
