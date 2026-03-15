"""System health monitoring."""
from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class HealthStatus(enum.StrEnum):
    """Overall health status of a component."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


@dataclass(slots=True)
class ComponentHealth:
    """Health status report for a single system component.

    Attributes:
        component: Name of the component being checked.
        status: Current health status.
        last_check: UTC timestamp when the check was performed.
        message: Human-readable status message.
        latency_ms: How long the health check took, in milliseconds.
    """

    component: str
    status: HealthStatus
    last_check: datetime
    message: str
    latency_ms: float


class HealthChecker:
    """Performs health checks across all system components.

    Each ``check_*`` method executes an active probe and returns a
    :class:`ComponentHealth` describing the result.  The :meth:`check_all`
    method runs every registered check and aggregates the results.

    Args:
        db_engine: Optional SQLAlchemy engine for the database check.
        market_feed_provider: Optional market feed provider object.
        redis_url: Optional Redis connection URL for the Redis check.
    """

    def __init__(
        self,
        db_engine: Any | None = None,
        market_feed_provider: Any | None = None,
        redis_url: str | None = None,
    ) -> None:
        self._db_engine = db_engine
        self._market_feed_provider = market_feed_provider
        self._redis_url = redis_url

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_database(self, engine: Optional[Any] = None) -> ComponentHealth:
        """Check database connectivity by executing a trivial query.

        Args:
            engine: SQLAlchemy engine to probe.  Falls back to the engine
                    provided at construction time if *None*.

        Returns:
            :class:`ComponentHealth` reflecting the result.
        """
        resolved_engine = engine or self._db_engine
        start = time.monotonic()
        now = datetime.now(timezone.utc)

        if resolved_engine is None:
            return ComponentHealth(
                component="database",
                status=HealthStatus.UNHEALTHY,
                last_check=now,
                message="No database engine configured.",
                latency_ms=0.0,
            )

        try:
            from sqlalchemy import text  # noqa: PLC0415

            with resolved_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            latency_ms = (time.monotonic() - start) * 1000.0
            return ComponentHealth(
                component="database",
                status=HealthStatus.HEALTHY,
                last_check=now,
                message="Database connection OK.",
                latency_ms=latency_ms,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - start) * 1000.0
            logger.warning("Database health check failed: %s", exc)
            return ComponentHealth(
                component="database",
                status=HealthStatus.UNHEALTHY,
                last_check=now,
                message=f"Database connection failed: {exc}",
                latency_ms=latency_ms,
            )

    def check_market_feed(self, provider: Optional[Any] = None) -> ComponentHealth:
        """Check that the market data feed is alive and responsive.

        The provider is expected to expose an ``is_connected()`` method or
        a boolean ``connected`` attribute.

        Args:
            provider: Market feed provider.  Falls back to the provider
                      provided at construction time if *None*.

        Returns:
            :class:`ComponentHealth` reflecting the result.
        """
        resolved_provider = provider or self._market_feed_provider
        start = time.monotonic()
        now = datetime.now(timezone.utc)

        if resolved_provider is None:
            return ComponentHealth(
                component="market_feed",
                status=HealthStatus.DEGRADED,
                last_check=now,
                message="No market feed provider configured.",
                latency_ms=0.0,
            )

        try:
            # Support both callable is_connected() and attribute .connected
            if callable(getattr(resolved_provider, "is_connected", None)):
                connected = resolved_provider.is_connected()
            else:
                connected = bool(getattr(resolved_provider, "connected", False))

            latency_ms = (time.monotonic() - start) * 1000.0
            status = HealthStatus.HEALTHY if connected else HealthStatus.UNHEALTHY
            message = "Market feed connected." if connected else "Market feed disconnected."
            return ComponentHealth(
                component="market_feed",
                status=status,
                last_check=now,
                message=message,
                latency_ms=latency_ms,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - start) * 1000.0
            logger.warning("Market feed health check failed: %s", exc)
            return ComponentHealth(
                component="market_feed",
                status=HealthStatus.UNHEALTHY,
                last_check=now,
                message=f"Market feed check failed: {exc}",
                latency_ms=latency_ms,
            )

    def check_redis(self, url: Optional[str] = None) -> ComponentHealth:
        """Ping a Redis instance to verify connectivity.

        Args:
            url: Redis connection URL.  Falls back to the URL provided at
                 construction time if *None*.

        Returns:
            :class:`ComponentHealth` reflecting the result.
        """
        resolved_url = url or self._redis_url
        start = time.monotonic()
        now = datetime.now(timezone.utc)

        if resolved_url is None:
            return ComponentHealth(
                component="redis",
                status=HealthStatus.DEGRADED,
                last_check=now,
                message="No Redis URL configured.",
                latency_ms=0.0,
            )

        try:
            import redis as redis_lib  # noqa: PLC0415

            client = redis_lib.Redis.from_url(resolved_url, socket_connect_timeout=2)
            client.ping()
            latency_ms = (time.monotonic() - start) * 1000.0
            return ComponentHealth(
                component="redis",
                status=HealthStatus.HEALTHY,
                last_check=now,
                message="Redis ping OK.",
                latency_ms=latency_ms,
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - start) * 1000.0
            logger.warning("Redis health check failed: %s", exc)
            return ComponentHealth(
                component="redis",
                status=HealthStatus.UNHEALTHY,
                last_check=now,
                message=f"Redis ping failed: {exc}",
                latency_ms=latency_ms,
            )

    # ------------------------------------------------------------------
    # Aggregate check
    # ------------------------------------------------------------------

    def check_all(self) -> list[ComponentHealth]:
        """Run all configured health checks and return their results.

        Returns:
            List of :class:`ComponentHealth` objects, one per component.
        """
        results: list[ComponentHealth] = []

        results.append(self.check_database())
        results.append(self.check_market_feed())
        results.append(self.check_redis())

        for result in results:
            logger.debug(
                "Health check [%s]: %s — %s (%.1f ms)",
                result.component,
                result.status,
                result.message,
                result.latency_ms,
            )

        return results

    def is_healthy(self) -> bool:
        """Return True only if every component reports HEALTHY.

        Returns:
            True if all components are HEALTHY; False otherwise.
        """
        return all(h.status == HealthStatus.HEALTHY for h in self.check_all())
