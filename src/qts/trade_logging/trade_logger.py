"""Trade and signal logger for persisting execution records."""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from qts.models.base import SignalSnapshot, TradeRecord

logger = logging.getLogger(__name__)

# ── Repository protocol ───────────────────────────────────────────────────────

# We keep the repository interface minimal and defined locally to avoid a
# circular-import chain with the DB layer.  Concrete repositories can live
# in qts.db and be injected at runtime.


class TradeLogger:
    """Persists trade records and signal snapshots to a repository.

    ``TradeLogger`` is designed for use in the hot path of the execution engine
    where synchronous code interacts with an (optionally) async persistence
    layer.  A synchronous ``session_factory`` callback is the primary
    injection point; async repositories can be wrapped with ``asyncio.run``
    or an equivalent executor wrapper before being passed in.

    Args:
        session_factory: Zero-argument callable that returns a context manager
            for a database session (e.g. ``sessionmaker``).  May be *None* for
            in-memory / testing mode.
        trade_repository: Optional repository object with ``save_trade`` and
            ``save_snapshot`` methods.  When provided, *session_factory* is
            ignored for persistence calls.
    """

    def __init__(
        self,
        session_factory: Callable[[], Any] | None = None,
        trade_repository: Any | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._trade_repository = trade_repository
        self._in_memory_trades: list[TradeRecord] = []
        self._in_memory_snapshots: list[SignalSnapshot] = []

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _persist_trade(self, trade: TradeRecord) -> None:
        """Write *trade* to the repository or session.

        Args:
            trade: :class:`~qts.models.base.TradeRecord` to persist.
        """
        if self._trade_repository is not None:
            try:
                self._trade_repository.save_trade(trade)
                return
            except Exception:  # noqa: BLE001
                logger.exception("TradeLogger: repository save_trade failed for %s", trade.trade_id)

        if self._session_factory is not None:
            try:
                with self._session_factory() as session:
                    session.add(trade)
                    session.commit()
                return
            except Exception:  # noqa: BLE001
                logger.exception(
                    "TradeLogger: session save failed for trade %s", trade.trade_id
                )

        # Fall back to in-memory store
        self._in_memory_trades.append(trade)

    def _persist_snapshot(self, snapshot: SignalSnapshot) -> None:
        """Write *snapshot* to the repository or session.

        Args:
            snapshot: :class:`~qts.models.base.SignalSnapshot` to persist.
        """
        if self._trade_repository is not None:
            try:
                self._trade_repository.save_snapshot(snapshot)
                return
            except Exception:  # noqa: BLE001
                logger.exception(
                    "TradeLogger: repository save_snapshot failed for %s / %s",
                    snapshot.symbol,
                    snapshot.timestamp,
                )

        if self._session_factory is not None:
            try:
                with self._session_factory() as session:
                    session.add(snapshot)
                    session.commit()
                return
            except Exception:  # noqa: BLE001
                logger.exception(
                    "TradeLogger: session save failed for snapshot %s / %s",
                    snapshot.symbol,
                    snapshot.timestamp,
                )

        # Fall back to in-memory store
        self._in_memory_snapshots.append(snapshot)

    # ── Public API ────────────────────────────────────────────────────────────

    def log_trade(self, trade: TradeRecord) -> None:
        """Persist a completed trade record.

        Logs at INFO level and delegates to the configured repository or
        session factory.  On failure, falls back to an in-memory buffer
        so that the hot path is never interrupted.

        Args:
            trade: Completed :class:`~qts.models.base.TradeRecord`.
        """
        logger.info(
            "TradeLogger: logging trade %s | %s | pnl=%.4f | outcome=%s",
            trade.trade_id,
            trade.symbol,
            trade.pnl_usd,
            trade.outcome,
        )
        self._persist_trade(trade)

    def log_signal(self, snapshot: SignalSnapshot) -> None:
        """Persist a signal snapshot captured at the time of a trade.

        Args:
            snapshot: :class:`~qts.models.base.SignalSnapshot` at trade time.
        """
        logger.debug(
            "TradeLogger: logging signal snapshot %s @ %s",
            snapshot.symbol,
            snapshot.timestamp,
        )
        self._persist_snapshot(snapshot)

    # ── Convenience accessors (for in-memory / test mode) ─────────────────────

    @property
    def trades(self) -> list[TradeRecord]:
        """Return all in-memory trade records (populated when no repository is set)."""
        return list(self._in_memory_trades)

    @property
    def snapshots(self) -> list[SignalSnapshot]:
        """Return all in-memory signal snapshots."""
        return list(self._in_memory_snapshots)
