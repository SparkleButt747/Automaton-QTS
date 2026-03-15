"""Signal snapshot logger for capturing indicator state at trade time."""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

from qts.models.base import SignalSnapshot

logger = logging.getLogger(__name__)


class SignalLogger:
    """Logs and retrieves :class:`~qts.models.base.SignalSnapshot` records.

    Signal snapshots capture the full indicator state (RSI, MACD, Bollinger
    Bands, ATR, momentum, volatility regime, sentiment) at the moment a trade
    decision is made.  They are stored alongside :class:`TradeRecord` objects
    to enable post-trade attribution analysis.

    This implementation stores snapshots in memory indexed by symbol for
    fast retrieval in unit tests and backtests.  Production deployments
    should inject a repository that persists to the database.

    Args:
        repository: Optional repository object with ``save_snapshot`` and
            ``get_snapshots`` methods.  When *None*, snapshots are held
            in-process only.
    """

    def __init__(self, repository: Any | None = None) -> None:
        self._repository = repository
        # In-memory index: symbol -> list of snapshots (insertion-ordered)
        self._snapshots: dict[str, list[SignalSnapshot]] = defaultdict(list)

    # ── Public API ────────────────────────────────────────────────────────────

    def log_snapshot(self, snapshot: SignalSnapshot) -> None:
        """Persist a single signal snapshot.

        If a repository is configured, the snapshot is forwarded to
        ``repository.save_snapshot(snapshot)``.  Regardless, a copy is kept
        in the in-memory index for :meth:`get_snapshots`.

        Args:
            snapshot: :class:`~qts.models.base.SignalSnapshot` to persist.
        """
        logger.debug(
            "SignalLogger: snapshot %s @ %s | rsi=%.2f | sentiment=%.4f",
            snapshot.symbol,
            snapshot.timestamp,
            snapshot.rsi,
            snapshot.sentiment_score,
        )

        self._snapshots[snapshot.symbol].append(snapshot)

        if self._repository is not None:
            try:
                self._repository.save_snapshot(snapshot)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "SignalLogger: repository save_snapshot failed for %s @ %s",
                    snapshot.symbol,
                    snapshot.timestamp,
                )

    def get_snapshots(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[SignalSnapshot]:
        """Retrieve signal snapshots for *symbol* within a time range.

        If a repository is configured, it is queried first.  Otherwise, the
        in-memory index is used as a fallback.

        Args:
            symbol: Ticker symbol to filter on.
            start: Inclusive start of the time window (UTC).
            end: Inclusive end of the time window (UTC).

        Returns:
            List of :class:`~qts.models.base.SignalSnapshot` objects ordered
            by timestamp ascending.
        """
        if self._repository is not None:
            try:
                return self._repository.get_snapshots(symbol, start, end)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "SignalLogger: repository get_snapshots failed, falling back to in-memory"
                )

        all_snaps = self._snapshots.get(symbol, [])
        filtered = [s for s in all_snaps if start <= s.timestamp <= end]
        filtered.sort(key=lambda s: s.timestamp)
        return filtered

    def clear(self) -> None:
        """Clear the in-memory snapshot store (useful between test runs)."""
        self._snapshots.clear()
