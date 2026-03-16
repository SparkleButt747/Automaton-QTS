"""Alert management for threshold breaches and system events."""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

logger = logging.getLogger(__name__)


class AlertLevel(enum.StrEnum):
    """Severity level for a triggered alert."""

    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True, slots=True)
class Alert:
    """A single alert triggered by a threshold breach or system event.

    Attributes:
        level: Severity of the alert.
        message: Human-readable description of the alert.
        timestamp: UTC timestamp when the alert was created.
        component: Name of the component that generated the alert.
        value: The observed value that triggered the alert.
        threshold: The threshold that was breached.
    """

    level: AlertLevel
    message: str
    timestamp: datetime
    component: str
    value: float
    threshold: float


class AlertManagerProtocol(Protocol):
    """Protocol for dependency-injection of AlertManager."""

    def check_drawdown(self, current_drawdown: float, limit: float) -> Alert | None:
        """Check if drawdown exceeds the configured limit."""
        ...

    def check_feed_staleness(
        self, last_tick_time: datetime, max_age_seconds: float
    ) -> Alert | None:
        """Check if a market feed is stale."""
        ...

    def check_position_concentration(
        self, positions: dict[str, float], limit: float
    ) -> Alert | None:
        """Check if any single position exceeds the concentration limit."""
        ...

    def get_active_alerts(self) -> list[Alert]:
        """Return all currently active alerts."""
        ...


class AlertManager:
    """Monitors thresholds and raises alerts on breaches.

    Maintains a list of active alerts that accumulate until cleared.

    Args:
        drawdown_threshold: Fraction of portfolio drawdown that triggers WARNING (e.g. 0.01).
        latency_threshold_ms: Feed latency in milliseconds that triggers WARNING.
        concentration_threshold: Single-position fraction that triggers WARNING.
    """

    def __init__(
        self,
        drawdown_threshold: float = 0.01,
        latency_threshold_ms: float = 5000.0,
        concentration_threshold: float = 0.10,
    ) -> None:
        self.drawdown_threshold = drawdown_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.concentration_threshold = concentration_threshold
        self._active_alerts: list[Alert] = []

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_drawdown(self, current_drawdown: float, limit: float) -> Alert | None:
        """Check if daily drawdown has breached the configured limit.

        Args:
            current_drawdown: Observed drawdown as a positive fraction (e.g. 0.02 = 2%).
            limit: Hard limit fraction (e.g. 0.02 for 2% max drawdown).

        Returns:
            An Alert if the drawdown is at or above the limit, else None.
        """
        if current_drawdown >= limit:
            alert = Alert(
                level=AlertLevel.CRITICAL,
                message=(
                    f"Daily drawdown {current_drawdown:.2%} has reached "
                    f"the {limit:.2%} circuit-breaker limit."
                ),
                timestamp=datetime.now(UTC),
                component="risk_manager",
                value=current_drawdown,
                threshold=limit,
            )
            self._active_alerts.append(alert)
            logger.critical(alert.message)
            return alert

        if current_drawdown >= self.drawdown_threshold:
            alert = Alert(
                level=AlertLevel.WARNING,
                message=(
                    f"Daily drawdown {current_drawdown:.2%} is approaching "
                    f"the {limit:.2%} limit."
                ),
                timestamp=datetime.now(UTC),
                component="risk_manager",
                value=current_drawdown,
                threshold=self.drawdown_threshold,
            )
            self._active_alerts.append(alert)
            logger.warning(alert.message)
            return alert

        return None

    def check_feed_staleness(
        self, last_tick_time: datetime, max_age_seconds: float
    ) -> Alert | None:
        """Check whether a market data feed has gone stale.

        Args:
            last_tick_time: Timestamp of the most recent tick received.
            max_age_seconds: Maximum allowed age of the last tick in seconds.

        Returns:
            An Alert if the feed is stale, else None.
        """
        now = datetime.now(UTC)
        age_seconds = (now - last_tick_time).total_seconds()

        if age_seconds >= max_age_seconds:
            alert = Alert(
                level=AlertLevel.CRITICAL,
                message=(
                    f"Market data feed stale: last tick was {age_seconds:.1f}s ago "
                    f"(max {max_age_seconds:.1f}s)."
                ),
                timestamp=now,
                component="market_feed",
                value=age_seconds,
                threshold=max_age_seconds,
            )
            self._active_alerts.append(alert)
            logger.critical(alert.message)
            return alert

        warning_threshold = max_age_seconds * 0.75
        if age_seconds >= warning_threshold:
            alert = Alert(
                level=AlertLevel.WARNING,
                message=(
                    f"Market data feed approaching staleness: last tick was {age_seconds:.1f}s ago."
                ),
                timestamp=now,
                component="market_feed",
                value=age_seconds,
                threshold=warning_threshold,
            )
            self._active_alerts.append(alert)
            logger.warning(alert.message)
            return alert

        return None

    def check_position_concentration(
        self, positions: dict[str, float], limit: float
    ) -> Alert | None:
        """Check whether any single position exceeds the concentration limit.

        Args:
            positions: Mapping of symbol -> notional USD value.
            limit: Maximum allowed fraction of total portfolio in one position.

        Returns:
            The worst Alert if any position exceeds the limit, else None.
        """
        total = sum(positions.values())
        if total <= 0.0:
            return None

        worst_alert: Alert | None = None
        for symbol, value in positions.items():
            fraction = value / total
            if fraction >= limit:
                alert = Alert(
                    level=AlertLevel.WARNING,
                    message=(
                        f"Position {symbol} represents {fraction:.2%} of portfolio "
                        f"(limit {limit:.2%})."
                    ),
                    timestamp=datetime.now(UTC),
                    component="position_manager",
                    value=fraction,
                    threshold=limit,
                )
                self._active_alerts.append(alert)
                logger.warning(alert.message)
                worst_alert = alert

        return worst_alert

    # ------------------------------------------------------------------
    # Active alert access
    # ------------------------------------------------------------------

    def get_active_alerts(self) -> list[Alert]:
        """Return a snapshot of all currently active alerts.

        Returns:
            List of Alert objects in chronological order.
        """
        return list(self._active_alerts)

    def clear_alerts(self) -> None:
        """Remove all active alerts (e.g., after acknowledgement)."""
        self._active_alerts.clear()
        logger.debug("AlertManager: all alerts cleared")
