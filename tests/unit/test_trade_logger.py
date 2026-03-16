"""Unit tests for qts.trade_logging.trade_logger.

Covers:
- T-TLOG-1 to T-TLOG-18
- TradeLogger in-memory mode (no repository, no session_factory)
- TradeLogger with repository: delegates and falls back on exception
- TradeLogger with session_factory: delegates and falls back on exception
- Public accessors: trades, snapshots
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock

from qts.models.base import (
    ExitReason,
    SignalSnapshot,
    TradeDirection,
    TradeOutcome,
    TradeRecord,
    VolRegime,
)
from qts.trade_logging.trade_logger import TradeLogger

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_trade(trade_id: str = "trade-001") -> TradeRecord:
    return TradeRecord(
        trade_id=trade_id,
        symbol="SYNTH",
        entry_time=datetime(2024, 1, 1, 9, 0),
        exit_time=datetime(2024, 1, 1, 10, 0),
        direction=TradeDirection.LONG,
        entry_price=100.0,
        exit_price=105.0,
        size=10.0,
        pnl_usd=50.0,
        pnl_pct=0.05,
        slippage_usd=0.5,
        fees_usd=1.0,
        rsi_at_entry=55.0,
        macd_at_entry=0.1,
        sentiment_at_entry=0.3,
        social_velocity_at_entry=0.1,
        gdelt_intensity_at_entry=0.05,
        vol_regime_at_entry=VolRegime.HIGH,
        combined_alpha_at_entry=0.45,
        params_version="1.0.0",
        outcome=TradeOutcome.WIN,
        exit_reason=ExitReason.ALPHA_FLIP,
        failure_mode=None,
    )


def _make_snapshot(symbol: str = "SYNTH") -> SignalSnapshot:
    return SignalSnapshot(
        timestamp=datetime(2024, 1, 1, 9, 0),
        symbol=symbol,
        rsi=55.0,
        macd_histogram=0.05,
        macd_line=0.1,
        macd_signal=0.05,
        bb_position=0.6,
        bb_upper=110.0,
        bb_lower=90.0,
        atr=1.5,
        momentum_5=0.01,
        vol_regime=VolRegime.HIGH,
        vol_regime_confidence=0.85,
        sentiment_score=0.3,
        combined_alpha=0.45,
    )


# ── In-memory mode ────────────────────────────────────────────────────────────


class TestInMemoryMode:
    def test_T_TLOG_1_initialises_with_empty_stores(self) -> None:
        logger = TradeLogger()
        assert logger.trades == []
        assert logger.snapshots == []

    def test_T_TLOG_2_log_trade_stores_in_memory(self) -> None:
        logger = TradeLogger()
        trade = _make_trade()
        logger.log_trade(trade)
        assert len(logger.trades) == 1
        assert logger.trades[0].trade_id == "trade-001"

    def test_T_TLOG_3_log_signal_stores_in_memory(self) -> None:
        logger = TradeLogger()
        snap = _make_snapshot()
        logger.log_signal(snap)
        assert len(logger.snapshots) == 1
        assert logger.snapshots[0].symbol == "SYNTH"

    def test_T_TLOG_4_multiple_trades_accumulated(self) -> None:
        logger = TradeLogger()
        for i in range(5):
            logger.log_trade(_make_trade(trade_id=f"trade-{i:03d}"))
        assert len(logger.trades) == 5

    def test_T_TLOG_5_trades_property_returns_copy(self) -> None:
        """Modifying the returned list should not affect internal state."""
        logger = TradeLogger()
        logger.log_trade(_make_trade())
        trades = logger.trades
        trades.clear()
        assert len(logger.trades) == 1

    def test_T_TLOG_6_snapshots_property_returns_copy(self) -> None:
        logger = TradeLogger()
        logger.log_signal(_make_snapshot())
        snaps = logger.snapshots
        snaps.clear()
        assert len(logger.snapshots) == 1


# ── Repository mode ───────────────────────────────────────────────────────────


class TestRepositoryMode:
    def test_T_TLOG_7_delegates_save_trade_to_repository(self) -> None:
        repo = MagicMock()
        logger = TradeLogger(trade_repository=repo)
        trade = _make_trade()
        logger.log_trade(trade)
        repo.save_trade.assert_called_once_with(trade)

    def test_T_TLOG_8_delegates_save_snapshot_to_repository(self) -> None:
        repo = MagicMock()
        logger = TradeLogger(trade_repository=repo)
        snap = _make_snapshot()
        logger.log_signal(snap)
        repo.save_snapshot.assert_called_once_with(snap)

    def test_T_TLOG_9_falls_back_to_memory_when_repo_save_trade_raises(self) -> None:
        repo = MagicMock()
        repo.save_trade.side_effect = RuntimeError("DB down")
        logger = TradeLogger(trade_repository=repo)
        trade = _make_trade()
        logger.log_trade(trade)  # should not raise
        # Falls back to in-memory
        assert len(logger.trades) == 1

    def test_T_TLOG_10_falls_back_to_memory_when_repo_save_snapshot_raises(self) -> None:
        repo = MagicMock()
        repo.save_snapshot.side_effect = RuntimeError("DB down")
        logger = TradeLogger(trade_repository=repo)
        snap = _make_snapshot()
        logger.log_signal(snap)  # should not raise
        assert len(logger.snapshots) == 1

    def test_T_TLOG_11_no_in_memory_storage_on_successful_repo_save(self) -> None:
        """When repo.save_trade succeeds, nothing should go into in-memory store."""
        repo = MagicMock()
        logger = TradeLogger(trade_repository=repo)
        logger.log_trade(_make_trade())
        assert len(logger.trades) == 0


# ── Session-factory mode ──────────────────────────────────────────────────────


class TestSessionFactoryMode:
    def _make_session_factory(self, session: MagicMock):
        @contextmanager
        def factory():
            yield session

        return factory

    def test_T_TLOG_12_delegates_to_session_factory(self) -> None:
        session = MagicMock()
        logger = TradeLogger(session_factory=self._make_session_factory(session))
        trade = _make_trade()
        logger.log_trade(trade)
        session.add.assert_called_once_with(trade)
        session.commit.assert_called_once()

    def test_T_TLOG_13_session_factory_snapshot_delegates(self) -> None:
        session = MagicMock()
        logger = TradeLogger(session_factory=self._make_session_factory(session))
        snap = _make_snapshot()
        logger.log_signal(snap)
        session.add.assert_called_once_with(snap)
        session.commit.assert_called_once()

    def test_T_TLOG_14_falls_back_to_memory_when_session_commit_raises(self) -> None:
        session = MagicMock()
        session.commit.side_effect = RuntimeError("commit failed")

        logger = TradeLogger(session_factory=self._make_session_factory(session))
        trade = _make_trade()
        logger.log_trade(trade)  # should not raise
        assert len(logger.trades) == 1

    def test_T_TLOG_15_repository_takes_precedence_over_session_factory(self) -> None:
        """When both repo and session_factory are set, repo is tried first."""
        repo = MagicMock()
        session = MagicMock()
        logger = TradeLogger(
            session_factory=self._make_session_factory(session),
            trade_repository=repo,
        )
        logger.log_trade(_make_trade())
        repo.save_trade.assert_called_once()
        session.add.assert_not_called()
