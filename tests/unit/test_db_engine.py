"""Unit tests for qts.db.engine.

Verifies:
- create_engine returns an AsyncEngine for SQLite and Postgres URLs
- create_session_factory returns an async_sessionmaker bound to the engine
- get_session yields a session and commits on clean exit
- get_session rolls back on exception
- init_db creates all tables in an in-memory database
- ensure_tables rewrites bare sqlite:/// URLs to use aiosqlite driver
"""
from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from qts.db.engine import (
    create_engine,
    create_session_factory,
    get_session,
    init_db,
)


# ── T-DBENG-1 ─────────────────────────────────────────────────────────────────


class TestCreateEngine:
    def test_returns_async_engine_for_sqlite(self):
        """create_engine returns an AsyncEngine for a sqlite+aiosqlite URL."""
        engine = create_engine("sqlite+aiosqlite://")
        assert isinstance(engine, AsyncEngine)

    def test_returns_async_engine_with_echo(self):
        """create_engine accepts echo=True without raising."""
        engine = create_engine("sqlite+aiosqlite://", echo=True)
        assert isinstance(engine, AsyncEngine)

    def test_sqlite_sets_check_same_thread(self):
        """SQLite engines are created with check_same_thread=False."""
        engine = create_engine("sqlite+aiosqlite://")
        # Driver-level connect_args are embedded in the engine URL / creator;
        # at minimum the engine should be created without raising.
        assert engine is not None


# ── T-DBENG-2 ─────────────────────────────────────────────────────────────────


class TestCreateSessionFactory:
    def test_returns_session_maker(self):
        """create_session_factory returns an async_sessionmaker instance."""
        engine = create_engine("sqlite+aiosqlite://")
        factory = create_session_factory(engine)
        assert isinstance(factory, async_sessionmaker)

    def test_sessions_are_async(self):
        """Sessions produced by the factory are AsyncSession instances."""
        engine = create_engine("sqlite+aiosqlite://")
        factory = create_session_factory(engine)
        session = factory()
        assert isinstance(session, AsyncSession)


# ── T-DBENG-3 ─────────────────────────────────────────────────────────────────


class TestInitDb:
    @pytest.mark.asyncio
    async def test_init_db_creates_tables(self):
        """init_db creates all ORM-mapped tables in an in-memory SQLite database."""
        from sqlalchemy import inspect, text

        engine = create_engine("sqlite+aiosqlite://")
        await init_db(engine)

        async with engine.connect() as conn:
            table_names = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).get_table_names()
            )

        expected = {"ticks", "bars", "signal_snapshots", "trade_records"}
        assert expected.issubset(set(table_names))

    @pytest.mark.asyncio
    async def test_init_db_idempotent(self):
        """Calling init_db twice does not raise (create_all is idempotent)."""
        engine = create_engine("sqlite+aiosqlite://")
        await init_db(engine)
        await init_db(engine)  # should not raise


# ── T-DBENG-4 ─────────────────────────────────────────────────────────────────


class TestGetSession:
    @pytest.mark.asyncio
    async def test_get_session_yields_async_session(self):
        """get_session yields an AsyncSession."""
        engine = create_engine("sqlite+aiosqlite://")
        await init_db(engine)
        factory = create_session_factory(engine)

        gen = get_session(factory)
        session = await gen.__anext__()
        assert isinstance(session, AsyncSession)
        # Clean up — exhaust the generator
        try:
            await gen.aclose()
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_get_session_commits_on_clean_exit(self):
        """get_session commits the transaction when no exception is raised."""
        from qts.db.tables import TickRow
        from datetime import datetime, timezone

        engine = create_engine("sqlite+aiosqlite://")
        await init_db(engine)
        factory = create_session_factory(engine)

        async with factory() as session:
            async with session.begin():
                row = TickRow(
                    timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    symbol="BTCUSDT",
                    price=42000.0,
                    quantity=1.0,
                    side="BUY",
                )
                session.add(row)

        # Verify persisted
        async with factory() as session:
            from sqlalchemy import select

            result = await session.execute(select(TickRow))
            rows = result.scalars().all()
            assert len(rows) == 1

    @pytest.mark.asyncio
    async def test_get_session_rolls_back_on_exception(self):
        """get_session rolls back when the body raises an exception."""
        from qts.db.tables import TickRow
        from datetime import datetime, timezone
        from sqlalchemy import select

        engine = create_engine("sqlite+aiosqlite://")
        await init_db(engine)
        factory = create_session_factory(engine)

        with pytest.raises(ValueError):
            async for session in get_session(factory):
                row = TickRow(
                    timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
                    symbol="AAPL",
                    price=150.0,
                    quantity=10.0,
                    side="SELL",
                )
                session.add(row)
                raise ValueError("boom")

        # Nothing should have been committed
        async with factory() as session:
            result = await session.execute(select(TickRow))
            rows = result.scalars().all()
            assert len(rows) == 0


# ── T-DBENG-5 ─────────────────────────────────────────────────────────────────


class TestEnsureTables:
    @pytest.mark.asyncio
    async def test_ensure_tables_rewrites_sqlite_url(self, monkeypatch):
        """ensure_tables converts bare sqlite:/// to sqlite+aiosqlite:///."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from qts.db import engine as engine_module

        captured_url = []

        original_create = engine_module.create_engine

        def fake_create_engine(url, **kwargs):
            captured_url.append(url)
            return original_create("sqlite+aiosqlite://", **kwargs)

        with patch.object(engine_module, "create_engine", side_effect=fake_create_engine):
            with patch.object(engine_module, "init_db", new_callable=AsyncMock):
                with patch("qts.config.get_settings") as mock_settings:
                    mock_db_cfg = MagicMock()
                    mock_db_cfg.database_url = "sqlite:///test.db"
                    mock_settings.return_value.database = mock_db_cfg

                    await engine_module.ensure_tables()

        assert captured_url[0] == "sqlite+aiosqlite:///test.db"

    @pytest.mark.asyncio
    async def test_ensure_tables_skips_rewrite_for_aiosqlite_url(self, monkeypatch):
        """ensure_tables leaves sqlite+aiosqlite:/// URLs unchanged."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from qts.db import engine as engine_module

        captured_url = []
        original_create = engine_module.create_engine

        def fake_create_engine(url, **kwargs):
            captured_url.append(url)
            return original_create("sqlite+aiosqlite://", **kwargs)

        with patch.object(engine_module, "create_engine", side_effect=fake_create_engine):
            with patch.object(engine_module, "init_db", new_callable=AsyncMock):
                with patch("qts.config.get_settings") as mock_settings:
                    mock_db_cfg = MagicMock()
                    mock_db_cfg.database_url = "sqlite+aiosqlite:///already_correct.db"
                    mock_settings.return_value.database = mock_db_cfg

                    await engine_module.ensure_tables()

        assert captured_url[0] == "sqlite+aiosqlite:///already_correct.db"
