"""Unit tests for qts.db.repository.

Verifies:
- MarketDataRepository: insert/get ticks and bars, batch inserts, range queries,
  get_latest_bar returns None when empty
- SignalRepository: insert/get snapshots, batch inserts, get_latest_snapshot
- TradeRepository: insert/get trades, filter by outcome, get_trade_by_id,
  None result when unknown ID, failure_mode nullable roundtrip
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from qts.db.engine import create_engine, init_db, create_session_factory
from qts.db.repository import MarketDataRepository, SignalRepository, TradeRepository
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

# ── Fixtures ──────────────────────────────────────────────────────────────────

_T0 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
_T1 = datetime(2024, 1, 1, 0, 1, 0, tzinfo=timezone.utc)
_T2 = datetime(2024, 1, 1, 0, 2, 0, tzinfo=timezone.utc)


@pytest.fixture()
async def session_factory():
    """In-memory SQLite session factory with all tables created."""
    engine = create_engine("sqlite+aiosqlite://")
    await init_db(engine)
    factory = create_session_factory(engine)
    return factory


def _make_tick(ts=_T0, symbol="BTCUSDT", price=42000.0, qty=1.0, side=OrderSide.BUY) -> Tick:
    return Tick(timestamp=ts, symbol=symbol, price=price, quantity=qty, side=side)


def _make_bar(ts=_T0, symbol="BTCUSDT", close=42000.0) -> Bar:
    return Bar(
        timestamp=ts,
        symbol=symbol,
        open=close - 50,
        high=close + 100,
        low=close - 100,
        close=close,
        volume=150.0,
        bar_count=10,
    )


def _make_snapshot(ts=_T0, symbol="BTCUSDT", alpha=0.7) -> SignalSnapshot:
    return SignalSnapshot(
        timestamp=ts,
        symbol=symbol,
        rsi=55.0,
        macd_histogram=0.1,
        macd_line=0.2,
        macd_signal=0.1,
        bb_position=0.6,
        bb_upper=43000.0,
        bb_lower=41000.0,
        atr=250.0,
        momentum_5=0.02,
        vol_regime=VolRegime.LOW,
        vol_regime_confidence=0.8,
        sentiment_score=0.4,
        combined_alpha=alpha,
    )


def _make_trade(
    symbol="BTCUSDT",
    outcome=TradeOutcome.WIN,
    failure_mode=None,
) -> TradeRecord:
    return TradeRecord(
        trade_id=str(uuid.uuid4()),
        symbol=symbol,
        entry_time=_T0,
        exit_time=_T1,
        direction=TradeDirection.LONG,
        entry_price=42000.0,
        exit_price=42500.0,
        size=0.5,
        pnl_usd=250.0,
        pnl_pct=0.6,
        slippage_usd=2.0,
        fees_usd=5.0,
        rsi_at_entry=55.0,
        macd_at_entry=0.1,
        sentiment_at_entry=0.4,
        social_velocity_at_entry=0.1,
        gdelt_intensity_at_entry=0.0,
        vol_regime_at_entry=VolRegime.LOW,
        combined_alpha_at_entry=0.7,
        params_version="v1",
        outcome=outcome,
        exit_reason=ExitReason.TAKE_PROFIT,
        failure_mode=failure_mode,
    )


# ── MarketDataRepository: Ticks ───────────────────────────────────────────────


class TestMarketDataRepositoryTicks:
    @pytest.mark.asyncio
    async def test_insert_tick_returns_row(self, session_factory):
        """insert_tick persists a tick and returns the ORM row with an ID."""
        async with session_factory() as session:
            repo = MarketDataRepository(session)
            tick = _make_tick()
            row = await repo.insert_tick(tick)
            await session.flush()
            assert row.id is not None
            assert row.symbol == "BTCUSDT"
            assert row.price == 42000.0
            assert row.side == "BUY"

    @pytest.mark.asyncio
    async def test_insert_ticks_batch(self, session_factory):
        """insert_ticks persists all ticks and returns the correct count."""
        async with session_factory() as session:
            repo = MarketDataRepository(session)
            ticks = [_make_tick(ts=_T0), _make_tick(ts=_T1), _make_tick(ts=_T2)]
            rows = await repo.insert_ticks(ticks)
            assert len(rows) == 3

    @pytest.mark.asyncio
    async def test_get_ticks_round_trips(self, session_factory):
        """Ticks inserted and retrieved within a range match the originals."""
        async with session_factory() as session:
            repo = MarketDataRepository(session)
            tick = _make_tick(ts=_T1, price=43000.0)
            await repo.insert_tick(tick)
            await session.flush()

            result = await repo.get_ticks("BTCUSDT", _T0, _T2)

        assert len(result) == 1
        assert result[0].price == 43000.0
        assert result[0].side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_get_ticks_filters_by_symbol(self, session_factory):
        """get_ticks only returns ticks for the requested symbol."""
        async with session_factory() as session:
            repo = MarketDataRepository(session)
            await repo.insert_tick(_make_tick(symbol="BTCUSDT"))
            await repo.insert_tick(_make_tick(symbol="ETHUSDT"))
            await session.flush()

            btc = await repo.get_ticks("BTCUSDT", _T0, _T2)
            eth = await repo.get_ticks("ETHUSDT", _T0, _T2)

        assert all(t.symbol == "BTCUSDT" for t in btc)
        assert all(t.symbol == "ETHUSDT" for t in eth)
        assert len(btc) == 1
        assert len(eth) == 1

    @pytest.mark.asyncio
    async def test_get_ticks_empty_range_returns_empty(self, session_factory):
        """get_ticks returns [] when no ticks fall in the given range."""
        async with session_factory() as session:
            repo = MarketDataRepository(session)
            result = await repo.get_ticks("BTCUSDT", _T0, _T1)

        assert result == []

    @pytest.mark.asyncio
    async def test_get_ticks_ordered_by_timestamp(self, session_factory):
        """get_ticks returns ticks sorted ascending by timestamp."""
        async with session_factory() as session:
            repo = MarketDataRepository(session)
            await repo.insert_ticks([_make_tick(ts=_T2), _make_tick(ts=_T0), _make_tick(ts=_T1)])
            await session.flush()

            result = await repo.get_ticks("BTCUSDT", _T0, _T2)

        timestamps = [r.timestamp for r in result]
        assert timestamps == sorted(timestamps)


# ── MarketDataRepository: Bars ────────────────────────────────────────────────


class TestMarketDataRepositoryBars:
    @pytest.mark.asyncio
    async def test_insert_bar_returns_row(self, session_factory):
        """insert_bar persists a bar and returns the ORM row."""
        async with session_factory() as session:
            repo = MarketDataRepository(session)
            bar = _make_bar()
            row = await repo.insert_bar(bar)
            await session.flush()
            assert row.id is not None
            assert row.symbol == "BTCUSDT"
            assert row.interval == "1m"

    @pytest.mark.asyncio
    async def test_insert_bar_custom_interval(self, session_factory):
        """insert_bar stores the provided interval string."""
        async with session_factory() as session:
            repo = MarketDataRepository(session)
            row = await repo.insert_bar(_make_bar(), interval="5m")
            await session.flush()
            assert row.interval == "5m"

    @pytest.mark.asyncio
    async def test_insert_bars_batch(self, session_factory):
        """insert_bars persists all bars and returns the correct count."""
        async with session_factory() as session:
            repo = MarketDataRepository(session)
            bars = [_make_bar(ts=_T0), _make_bar(ts=_T1)]
            rows = await repo.insert_bars(bars)
            assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_get_bars_round_trips(self, session_factory):
        """Bars inserted and retrieved match the originals."""
        async with session_factory() as session:
            repo = MarketDataRepository(session)
            await repo.insert_bar(_make_bar(ts=_T1, close=43000.0))
            await session.flush()

            result = await repo.get_bars("BTCUSDT", _T0, _T2)

        assert len(result) == 1
        assert result[0].close == 43000.0

    @pytest.mark.asyncio
    async def test_get_bars_filters_by_interval(self, session_factory):
        """get_bars only returns bars for the requested interval."""
        async with session_factory() as session:
            repo = MarketDataRepository(session)
            await repo.insert_bar(_make_bar(ts=_T0), interval="1m")
            await repo.insert_bar(_make_bar(ts=_T1), interval="5m")
            await session.flush()

            bars_1m = await repo.get_bars("BTCUSDT", _T0, _T2, interval="1m")
            bars_5m = await repo.get_bars("BTCUSDT", _T0, _T2, interval="5m")

        assert len(bars_1m) == 1
        assert len(bars_5m) == 1

    @pytest.mark.asyncio
    async def test_get_latest_bar_returns_most_recent(self, session_factory):
        """get_latest_bar returns the bar with the most recent timestamp."""
        async with session_factory() as session:
            repo = MarketDataRepository(session)
            await repo.insert_bars([_make_bar(ts=_T0, close=100.0), _make_bar(ts=_T2, close=300.0)])
            await session.flush()

            bar = await repo.get_latest_bar("BTCUSDT")

        assert bar is not None
        assert bar.close == 300.0

    @pytest.mark.asyncio
    async def test_get_latest_bar_none_when_empty(self, session_factory):
        """get_latest_bar returns None when there are no bars for the symbol."""
        async with session_factory() as session:
            repo = MarketDataRepository(session)
            result = await repo.get_latest_bar("NONEXISTENT")

        assert result is None


# ── SignalRepository ──────────────────────────────────────────────────────────


class TestSignalRepository:
    @pytest.mark.asyncio
    async def test_insert_snapshot_returns_row(self, session_factory):
        """insert_snapshot persists a signal snapshot and returns the ORM row."""
        async with session_factory() as session:
            repo = SignalRepository(session)
            snap = _make_snapshot()
            row = await repo.insert_snapshot(snap)
            await session.flush()
            assert row.id is not None
            assert row.symbol == "BTCUSDT"
            assert row.vol_regime == "LOW"

    @pytest.mark.asyncio
    async def test_insert_snapshots_batch(self, session_factory):
        """insert_snapshots persists all snapshots."""
        async with session_factory() as session:
            repo = SignalRepository(session)
            snaps = [_make_snapshot(ts=_T0), _make_snapshot(ts=_T1)]
            rows = await repo.insert_snapshots(snaps)
            assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_get_snapshots_round_trips(self, session_factory):
        """Snapshots inserted and retrieved match the originals."""
        async with session_factory() as session:
            repo = SignalRepository(session)
            await repo.insert_snapshot(_make_snapshot(ts=_T1, alpha=0.9))
            await session.flush()

            result = await repo.get_snapshots("BTCUSDT", _T0, _T2)

        assert len(result) == 1
        assert result[0].combined_alpha == pytest.approx(0.9)
        assert result[0].vol_regime == VolRegime.LOW

    @pytest.mark.asyncio
    async def test_get_snapshots_empty_range(self, session_factory):
        """get_snapshots returns [] when no snapshots fall in the range."""
        async with session_factory() as session:
            repo = SignalRepository(session)
            result = await repo.get_snapshots("BTCUSDT", _T0, _T1)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_latest_snapshot_returns_most_recent(self, session_factory):
        """get_latest_snapshot returns the snapshot with the most recent timestamp."""
        async with session_factory() as session:
            repo = SignalRepository(session)
            await repo.insert_snapshots([
                _make_snapshot(ts=_T0, alpha=0.1),
                _make_snapshot(ts=_T2, alpha=0.99),
            ])
            await session.flush()

            snap = await repo.get_latest_snapshot("BTCUSDT")

        assert snap is not None
        assert snap.combined_alpha == pytest.approx(0.99)

    @pytest.mark.asyncio
    async def test_get_latest_snapshot_none_when_empty(self, session_factory):
        """get_latest_snapshot returns None when there are no snapshots for the symbol."""
        async with session_factory() as session:
            repo = SignalRepository(session)
            result = await repo.get_latest_snapshot("NONEXISTENT")
        assert result is None


# ── TradeRepository ───────────────────────────────────────────────────────────


class TestTradeRepository:
    @pytest.mark.asyncio
    async def test_insert_trade_returns_row(self, session_factory):
        """insert_trade persists a trade record and returns the ORM row."""
        async with session_factory() as session:
            repo = TradeRepository(session)
            trade = _make_trade()
            row = await repo.insert_trade(trade)
            await session.flush()
            assert row.id is not None
            assert row.trade_id == trade.trade_id
            assert row.outcome == "WIN"
            assert row.failure_mode is None

    @pytest.mark.asyncio
    async def test_insert_trade_with_failure_mode(self, session_factory):
        """insert_trade persists failure_mode when provided."""
        async with session_factory() as session:
            repo = TradeRepository(session)
            trade = _make_trade(outcome=TradeOutcome.LOSS, failure_mode=FailureMode.SENTIMENT_FADE)
            row = await repo.insert_trade(trade)
            await session.flush()
            assert row.failure_mode == "SENTIMENT_FADE"

    @pytest.mark.asyncio
    async def test_insert_trades_batch(self, session_factory):
        """insert_trades persists all trade records."""
        async with session_factory() as session:
            repo = TradeRepository(session)
            trades = [_make_trade(), _make_trade()]
            rows = await repo.insert_trades(trades)
            assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_get_trade_by_id_found(self, session_factory):
        """get_trade_by_id returns the correct trade when the ID exists."""
        async with session_factory() as session:
            repo = TradeRepository(session)
            trade = _make_trade()
            await repo.insert_trade(trade)
            await session.flush()

            found = await repo.get_trade_by_id(trade.trade_id)

        assert found is not None
        assert found.trade_id == trade.trade_id
        assert found.direction == TradeDirection.LONG

    @pytest.mark.asyncio
    async def test_get_trade_by_id_none_when_missing(self, session_factory):
        """get_trade_by_id returns None when the trade ID does not exist."""
        async with session_factory() as session:
            repo = TradeRepository(session)
            result = await repo.get_trade_by_id("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_trades_no_filters(self, session_factory):
        """get_trades with no filters returns all records."""
        async with session_factory() as session:
            repo = TradeRepository(session)
            await repo.insert_trades([_make_trade(), _make_trade()])
            await session.flush()

            result = await repo.get_trades()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_trades_filters_by_symbol(self, session_factory):
        """get_trades filters by symbol when provided."""
        async with session_factory() as session:
            repo = TradeRepository(session)
            await repo.insert_trade(_make_trade(symbol="BTCUSDT"))
            await repo.insert_trade(_make_trade(symbol="ETHUSDT"))
            await session.flush()

            btc = await repo.get_trades(symbol="BTCUSDT")
            eth = await repo.get_trades(symbol="ETHUSDT")

        assert all(t.symbol == "BTCUSDT" for t in btc)
        assert all(t.symbol == "ETHUSDT" for t in eth)

    @pytest.mark.asyncio
    async def test_get_trades_by_outcome_wins(self, session_factory):
        """get_trades_by_outcome returns only winning trades."""
        async with session_factory() as session:
            repo = TradeRepository(session)
            await repo.insert_trade(_make_trade(outcome=TradeOutcome.WIN))
            await repo.insert_trade(_make_trade(outcome=TradeOutcome.LOSS))
            await repo.insert_trade(_make_trade(outcome=TradeOutcome.WIN))
            await session.flush()

            wins = await repo.get_trades_by_outcome(TradeOutcome.WIN)
            losses = await repo.get_trades_by_outcome(TradeOutcome.LOSS)

        assert len(wins) == 2
        assert len(losses) == 1
        assert all(t.outcome == TradeOutcome.WIN for t in wins)

    @pytest.mark.asyncio
    async def test_get_trades_by_outcome_with_symbol_filter(self, session_factory):
        """get_trades_by_outcome also filters by symbol when provided."""
        async with session_factory() as session:
            repo = TradeRepository(session)
            await repo.insert_trade(_make_trade(symbol="BTCUSDT", outcome=TradeOutcome.WIN))
            await repo.insert_trade(_make_trade(symbol="ETHUSDT", outcome=TradeOutcome.WIN))
            await session.flush()

            result = await repo.get_trades_by_outcome(TradeOutcome.WIN, symbol="BTCUSDT")

        assert len(result) == 1
        assert result[0].symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_failure_mode_none_roundtrip(self, session_factory):
        """failure_mode=None is preserved through insert → get round-trip."""
        async with session_factory() as session:
            repo = TradeRepository(session)
            trade = _make_trade(failure_mode=None)
            await repo.insert_trade(trade)
            await session.flush()

            found = await repo.get_trade_by_id(trade.trade_id)

        assert found is not None
        assert found.failure_mode is None
