"""Repository pattern for database access.

Provides typed async repositories for each major domain entity:
- MarketDataRepository: Ticks and OHLCV bars
- SignalRepository: Signal snapshots
- TradeRepository: Trade records with attribution context
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from qts.db.tables import BarRow, SignalSnapshotRow, TickRow, TradeRecordRow
from qts.models.base import (
    Bar,
    ExitReason,
    FailureMode,
    OrderSide,
    SignalSnapshot,
    Tick,
    TradeDirection,
    TradeOutcome,
    TradeRecord,
    VolRegime,
)

logger = logging.getLogger(__name__)


class MarketDataRepository:
    """Repository for market data: ticks and OHLCV bars.

    Provides async CRUD methods for storing and retrieving tick and bar data.
    All methods operate within a provided AsyncSession.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialise the repository with a database session.

        Args:
            session: An async SQLAlchemy session.
        """
        self._session = session

    async def insert_tick(self, tick: Tick) -> TickRow:
        """Persist a single tick to the database.

        Args:
            tick: The Tick domain object to persist.

        Returns:
            The persisted TickRow ORM instance.
        """
        row = TickRow(
            timestamp=tick.timestamp,
            symbol=tick.symbol,
            price=tick.price,
            quantity=tick.quantity,
            side=tick.side.value,
        )
        self._session.add(row)
        await self._session.flush()
        logger.debug("Inserted tick: symbol=%s, price=%s", tick.symbol, tick.price)
        return row

    async def insert_ticks(self, ticks: list[Tick]) -> list[TickRow]:
        """Persist a batch of ticks to the database.

        Args:
            ticks: List of Tick domain objects to persist.

        Returns:
            List of persisted TickRow ORM instances.
        """
        rows = [
            TickRow(
                timestamp=t.timestamp,
                symbol=t.symbol,
                price=t.price,
                quantity=t.quantity,
                side=t.side.value,
            )
            for t in ticks
        ]
        self._session.add_all(rows)
        await self._session.flush()
        logger.debug("Inserted %d ticks", len(rows))
        return rows

    async def get_ticks(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> list[Tick]:
        """Retrieve ticks for a symbol within a time range.

        Args:
            symbol: The ticker symbol to query.
            start: Start of the time range (inclusive).
            end: End of the time range (inclusive).
            limit: Maximum number of ticks to return.

        Returns:
            List of Tick domain objects ordered by timestamp ascending.
        """
        stmt = (
            select(TickRow)
            .where(
                TickRow.symbol == symbol,
                TickRow.timestamp >= start,
                TickRow.timestamp <= end,
            )
            .order_by(TickRow.timestamp.asc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        return [
            Tick(
                timestamp=r.timestamp,
                symbol=r.symbol,
                price=r.price,
                quantity=r.quantity,
                side=OrderSide(r.side),
            )
            for r in rows
        ]

    async def insert_bar(self, bar: Bar, interval: str = "1m") -> BarRow:
        """Persist a single OHLCV bar to the database.

        Args:
            bar: The Bar domain object to persist.
            interval: Bar interval string (e.g. '1m', '5m', '1h').

        Returns:
            The persisted BarRow ORM instance.
        """
        row = BarRow(
            timestamp=bar.timestamp,
            symbol=bar.symbol,
            interval=interval,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            bar_count=bar.bar_count,
        )
        self._session.add(row)
        await self._session.flush()
        logger.debug("Inserted bar: symbol=%s, close=%s", bar.symbol, bar.close)
        return row

    async def insert_bars(self, bars: list[Bar], interval: str = "1m") -> list[BarRow]:
        """Persist a batch of OHLCV bars to the database.

        Args:
            bars: List of Bar domain objects to persist.
            interval: Bar interval string (e.g. '1m', '5m', '1h').

        Returns:
            List of persisted BarRow ORM instances.
        """
        rows = [
            BarRow(
                timestamp=b.timestamp,
                symbol=b.symbol,
                interval=interval,
                open=b.open,
                high=b.high,
                low=b.low,
                close=b.close,
                volume=b.volume,
                bar_count=b.bar_count,
            )
            for b in bars
        ]
        self._session.add_all(rows)
        await self._session.flush()
        logger.debug("Inserted %d bars", len(rows))
        return rows

    async def get_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1m",
        limit: int = 5000,
    ) -> list[Bar]:
        """Retrieve OHLCV bars for a symbol within a time range.

        Args:
            symbol: The ticker symbol to query.
            start: Start of the time range (inclusive).
            end: End of the time range (inclusive).
            interval: Bar interval filter (e.g. '1m', '5m', '1h').
            limit: Maximum number of bars to return.

        Returns:
            List of Bar domain objects ordered by timestamp ascending.
        """
        stmt = (
            select(BarRow)
            .where(
                BarRow.symbol == symbol,
                BarRow.interval == interval,
                BarRow.timestamp >= start,
                BarRow.timestamp <= end,
            )
            .order_by(BarRow.timestamp.asc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        return [
            Bar(
                timestamp=r.timestamp,
                symbol=r.symbol,
                open=r.open,
                high=r.high,
                low=r.low,
                close=r.close,
                volume=r.volume,
                bar_count=r.bar_count,
            )
            for r in rows
        ]

    async def get_latest_bar(self, symbol: str, interval: str = "1m") -> Optional[Bar]:
        """Retrieve the most recent bar for a symbol.

        Args:
            symbol: The ticker symbol to query.
            interval: Bar interval filter.

        Returns:
            The most recent Bar domain object, or None if no bars found.
        """
        stmt = (
            select(BarRow)
            .where(BarRow.symbol == symbol, BarRow.interval == interval)
            .order_by(BarRow.timestamp.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        row = result.scalars().first()
        if row is None:
            return None
        return Bar(
            timestamp=row.timestamp,
            symbol=row.symbol,
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=row.volume,
            bar_count=row.bar_count,
        )


class SignalRepository:
    """Repository for signal snapshot data.

    Provides async CRUD methods for storing and retrieving signal snapshots
    used for attribution and post-trade analysis.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialise the repository with a database session.

        Args:
            session: An async SQLAlchemy session.
        """
        self._session = session

    async def insert_snapshot(self, snapshot: SignalSnapshot) -> SignalSnapshotRow:
        """Persist a single signal snapshot to the database.

        Args:
            snapshot: The SignalSnapshot domain object to persist.

        Returns:
            The persisted SignalSnapshotRow ORM instance.
        """
        row = SignalSnapshotRow(
            timestamp=snapshot.timestamp,
            symbol=snapshot.symbol,
            rsi=snapshot.rsi,
            macd_histogram=snapshot.macd_histogram,
            macd_line=snapshot.macd_line,
            macd_signal=snapshot.macd_signal,
            bb_position=snapshot.bb_position,
            bb_upper=snapshot.bb_upper,
            bb_lower=snapshot.bb_lower,
            atr=snapshot.atr,
            momentum_5=snapshot.momentum_5,
            vol_regime=snapshot.vol_regime.value,
            vol_regime_confidence=snapshot.vol_regime_confidence,
            sentiment_score=snapshot.sentiment_score,
            combined_alpha=snapshot.combined_alpha,
        )
        self._session.add(row)
        await self._session.flush()
        logger.debug(
            "Inserted signal snapshot: symbol=%s, alpha=%s",
            snapshot.symbol,
            snapshot.combined_alpha,
        )
        return row

    async def insert_snapshots(self, snapshots: list[SignalSnapshot]) -> list[SignalSnapshotRow]:
        """Persist a batch of signal snapshots to the database.

        Args:
            snapshots: List of SignalSnapshot domain objects to persist.

        Returns:
            List of persisted SignalSnapshotRow ORM instances.
        """
        rows = [
            SignalSnapshotRow(
                timestamp=s.timestamp,
                symbol=s.symbol,
                rsi=s.rsi,
                macd_histogram=s.macd_histogram,
                macd_line=s.macd_line,
                macd_signal=s.macd_signal,
                bb_position=s.bb_position,
                bb_upper=s.bb_upper,
                bb_lower=s.bb_lower,
                atr=s.atr,
                momentum_5=s.momentum_5,
                vol_regime=s.vol_regime.value,
                vol_regime_confidence=s.vol_regime_confidence,
                sentiment_score=s.sentiment_score,
                combined_alpha=s.combined_alpha,
            )
            for s in snapshots
        ]
        self._session.add_all(rows)
        await self._session.flush()
        logger.debug("Inserted %d signal snapshots", len(rows))
        return rows

    async def get_snapshots(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> list[SignalSnapshot]:
        """Retrieve signal snapshots for a symbol within a time range.

        Args:
            symbol: The ticker symbol to query.
            start: Start of the time range (inclusive).
            end: End of the time range (inclusive).
            limit: Maximum number of snapshots to return.

        Returns:
            List of SignalSnapshot domain objects ordered by timestamp ascending.
        """
        stmt = (
            select(SignalSnapshotRow)
            .where(
                SignalSnapshotRow.symbol == symbol,
                SignalSnapshotRow.timestamp >= start,
                SignalSnapshotRow.timestamp <= end,
            )
            .order_by(SignalSnapshotRow.timestamp.asc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        return [
            SignalSnapshot(
                timestamp=r.timestamp,
                symbol=r.symbol,
                rsi=r.rsi,
                macd_histogram=r.macd_histogram,
                macd_line=r.macd_line,
                macd_signal=r.macd_signal,
                bb_position=r.bb_position,
                bb_upper=r.bb_upper,
                bb_lower=r.bb_lower,
                atr=r.atr,
                momentum_5=r.momentum_5,
                vol_regime=VolRegime(r.vol_regime),
                vol_regime_confidence=r.vol_regime_confidence,
                sentiment_score=r.sentiment_score,
                combined_alpha=r.combined_alpha,
            )
            for r in rows
        ]

    async def get_latest_snapshot(self, symbol: str) -> Optional[SignalSnapshot]:
        """Retrieve the most recent signal snapshot for a symbol.

        Args:
            symbol: The ticker symbol to query.

        Returns:
            The most recent SignalSnapshot, or None if none found.
        """
        stmt = (
            select(SignalSnapshotRow)
            .where(SignalSnapshotRow.symbol == symbol)
            .order_by(SignalSnapshotRow.timestamp.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        row = result.scalars().first()
        if row is None:
            return None
        return SignalSnapshot(
            timestamp=row.timestamp,
            symbol=row.symbol,
            rsi=row.rsi,
            macd_histogram=row.macd_histogram,
            macd_line=row.macd_line,
            macd_signal=row.macd_signal,
            bb_position=row.bb_position,
            bb_upper=row.bb_upper,
            bb_lower=row.bb_lower,
            atr=row.atr,
            momentum_5=row.momentum_5,
            vol_regime=VolRegime(row.vol_regime),
            vol_regime_confidence=row.vol_regime_confidence,
            sentiment_score=row.sentiment_score,
            combined_alpha=row.combined_alpha,
        )


class TradeRepository:
    """Repository for trade records.

    Provides async CRUD methods for storing and retrieving trade records
    with full signal context and attribution metadata.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Initialise the repository with a database session.

        Args:
            session: An async SQLAlchemy session.
        """
        self._session = session

    async def insert_trade(self, trade: TradeRecord) -> TradeRecordRow:
        """Persist a single trade record to the database.

        Args:
            trade: The TradeRecord domain object to persist.

        Returns:
            The persisted TradeRecordRow ORM instance.

        Raises:
            sqlalchemy.exc.IntegrityError: If a trade with the same trade_id already exists.
        """
        row = TradeRecordRow(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            entry_time=trade.entry_time,
            exit_time=trade.exit_time,
            direction=trade.direction.value,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            size=trade.size,
            pnl_usd=trade.pnl_usd,
            pnl_pct=trade.pnl_pct,
            slippage_usd=trade.slippage_usd,
            fees_usd=trade.fees_usd,
            rsi_at_entry=trade.rsi_at_entry,
            macd_at_entry=trade.macd_at_entry,
            sentiment_at_entry=trade.sentiment_at_entry,
            social_velocity_at_entry=trade.social_velocity_at_entry,
            gdelt_intensity_at_entry=trade.gdelt_intensity_at_entry,
            vol_regime_at_entry=trade.vol_regime_at_entry.value,
            combined_alpha_at_entry=trade.combined_alpha_at_entry,
            params_version=trade.params_version,
            outcome=trade.outcome.value,
            exit_reason=trade.exit_reason.value,
            failure_mode=trade.failure_mode.value if trade.failure_mode else None,
        )
        self._session.add(row)
        await self._session.flush()
        logger.info(
            "Inserted trade: id=%s, symbol=%s, pnl=%.2f",
            trade.trade_id,
            trade.symbol,
            trade.pnl_usd,
        )
        return row

    async def insert_trades(self, trades: list[TradeRecord]) -> list[TradeRecordRow]:
        """Persist a batch of trade records to the database.

        Args:
            trades: List of TradeRecord domain objects to persist.

        Returns:
            List of persisted TradeRecordRow ORM instances.

        Raises:
            sqlalchemy.exc.IntegrityError: If any trade_id already exists.
        """
        rows = [
            TradeRecordRow(
                trade_id=t.trade_id,
                symbol=t.symbol,
                entry_time=t.entry_time,
                exit_time=t.exit_time,
                direction=t.direction.value,
                entry_price=t.entry_price,
                exit_price=t.exit_price,
                size=t.size,
                pnl_usd=t.pnl_usd,
                pnl_pct=t.pnl_pct,
                slippage_usd=t.slippage_usd,
                fees_usd=t.fees_usd,
                rsi_at_entry=t.rsi_at_entry,
                macd_at_entry=t.macd_at_entry,
                sentiment_at_entry=t.sentiment_at_entry,
                social_velocity_at_entry=t.social_velocity_at_entry,
                gdelt_intensity_at_entry=t.gdelt_intensity_at_entry,
                vol_regime_at_entry=t.vol_regime_at_entry.value,
                combined_alpha_at_entry=t.combined_alpha_at_entry,
                params_version=t.params_version,
                outcome=t.outcome.value,
                exit_reason=t.exit_reason.value,
                failure_mode=t.failure_mode.value if t.failure_mode else None,
            )
            for t in trades
        ]
        self._session.add_all(rows)
        await self._session.flush()
        logger.debug("Inserted %d trade records", len(rows))
        return rows

    async def get_trade_by_id(self, trade_id: str) -> Optional[TradeRecord]:
        """Retrieve a single trade record by its unique ID.

        Args:
            trade_id: The UUID trade identifier.

        Returns:
            The TradeRecord domain object if found, else None.
        """
        stmt = select(TradeRecordRow).where(TradeRecordRow.trade_id == trade_id).limit(1)
        result = await self._session.execute(stmt)
        row = result.scalars().first()
        if row is None:
            return None
        return self._row_to_domain(row)

    async def get_trades(
        self,
        symbol: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 500,
    ) -> list[TradeRecord]:
        """Retrieve trade records with optional filtering.

        Args:
            symbol: Filter by ticker symbol (optional).
            start: Start entry time filter (inclusive, optional).
            end: End entry time filter (inclusive, optional).
            limit: Maximum number of trades to return.

        Returns:
            List of TradeRecord domain objects ordered by entry_time ascending.
        """
        stmt = select(TradeRecordRow)
        if symbol is not None:
            stmt = stmt.where(TradeRecordRow.symbol == symbol)
        if start is not None:
            stmt = stmt.where(TradeRecordRow.entry_time >= start)
        if end is not None:
            stmt = stmt.where(TradeRecordRow.entry_time <= end)
        stmt = stmt.order_by(TradeRecordRow.entry_time.asc()).limit(limit)

        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        return [self._row_to_domain(r) for r in rows]

    async def get_trades_by_outcome(
        self,
        outcome: TradeOutcome,
        symbol: Optional[str] = None,
        limit: int = 500,
    ) -> list[TradeRecord]:
        """Retrieve trade records filtered by outcome.

        Args:
            outcome: The trade outcome to filter on (WIN, LOSS, or BREAKEVEN).
            symbol: Optional additional filter by ticker symbol.
            limit: Maximum number of trades to return.

        Returns:
            List of TradeRecord domain objects matching the outcome filter.
        """
        stmt = select(TradeRecordRow).where(TradeRecordRow.outcome == outcome.value)
        if symbol is not None:
            stmt = stmt.where(TradeRecordRow.symbol == symbol)
        stmt = stmt.order_by(TradeRecordRow.entry_time.asc()).limit(limit)

        result = await self._session.execute(stmt)
        rows = result.scalars().all()
        return [self._row_to_domain(r) for r in rows]

    @staticmethod
    def _row_to_domain(row: TradeRecordRow) -> TradeRecord:
        """Convert a TradeRecordRow ORM instance to a TradeRecord domain object.

        Args:
            row: The ORM row to convert.

        Returns:
            A TradeRecord domain object with all fields populated.
        """
        return TradeRecord(
            trade_id=row.trade_id,
            symbol=row.symbol,
            entry_time=row.entry_time,
            exit_time=row.exit_time,
            direction=TradeDirection(row.direction),
            entry_price=row.entry_price,
            exit_price=row.exit_price,
            size=row.size,
            pnl_usd=row.pnl_usd,
            pnl_pct=row.pnl_pct,
            slippage_usd=row.slippage_usd,
            fees_usd=row.fees_usd,
            rsi_at_entry=row.rsi_at_entry,
            macd_at_entry=row.macd_at_entry,
            sentiment_at_entry=row.sentiment_at_entry,
            social_velocity_at_entry=row.social_velocity_at_entry,
            gdelt_intensity_at_entry=row.gdelt_intensity_at_entry,
            vol_regime_at_entry=VolRegime(row.vol_regime_at_entry),
            combined_alpha_at_entry=row.combined_alpha_at_entry,
            params_version=row.params_version,
            outcome=TradeOutcome(row.outcome),
            exit_reason=ExitReason(row.exit_reason),
            failure_mode=FailureMode(row.failure_mode) if row.failure_mode else None,
        )
