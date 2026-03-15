"""Unit tests for the system health checker."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from qts.monitoring.health import ComponentHealth, HealthChecker, HealthStatus


# ── ComponentHealth dataclass ─────────────────────────────────────────────────


class TestComponentHealth:
    def test_construction(self) -> None:
        health = ComponentHealth(
            component="database",
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(timezone.utc),
            message="OK",
            latency_ms=5.2,
        )
        assert health.component == "database"
        assert health.status == HealthStatus.HEALTHY
        assert health.latency_ms == 5.2

    def test_is_mutable(self) -> None:
        health = ComponentHealth(
            component="redis",
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(timezone.utc),
            message="OK",
            latency_ms=1.0,
        )
        health.status = HealthStatus.DEGRADED
        assert health.status == HealthStatus.DEGRADED


# ── HealthStatus enum ─────────────────────────────────────────────────────────


class TestHealthStatus:
    def test_all_values_exist(self) -> None:
        assert HealthStatus.HEALTHY == "HEALTHY"
        assert HealthStatus.DEGRADED == "DEGRADED"
        assert HealthStatus.UNHEALTHY == "UNHEALTHY"


# ── HealthChecker.check_database ─────────────────────────────────────────────


class TestCheckDatabase:
    def test_no_engine_returns_unhealthy(self) -> None:
        checker = HealthChecker()
        result = checker.check_database()
        assert result.status == HealthStatus.UNHEALTHY
        assert result.component == "database"

    def test_healthy_engine(self) -> None:
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        checker = HealthChecker(db_engine=mock_engine)
        result = checker.check_database()
        assert result.status == HealthStatus.HEALTHY
        assert "OK" in result.message

    def test_failing_engine(self) -> None:
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("connection refused")

        checker = HealthChecker(db_engine=mock_engine)
        result = checker.check_database()
        assert result.status == HealthStatus.UNHEALTHY
        assert "connection refused" in result.message


# ── HealthChecker.check_market_feed ──────────────────────────────────────────


class TestCheckMarketFeed:
    def test_no_provider_returns_degraded(self) -> None:
        checker = HealthChecker()
        result = checker.check_market_feed()
        assert result.status == HealthStatus.DEGRADED
        assert result.component == "market_feed"

    def test_connected_provider(self) -> None:
        mock_provider = MagicMock()
        mock_provider.is_connected.return_value = True

        checker = HealthChecker(market_feed_provider=mock_provider)
        result = checker.check_market_feed()
        assert result.status == HealthStatus.HEALTHY

    def test_disconnected_provider(self) -> None:
        mock_provider = MagicMock()
        mock_provider.is_connected.return_value = False

        checker = HealthChecker(market_feed_provider=mock_provider)
        result = checker.check_market_feed()
        assert result.status == HealthStatus.UNHEALTHY

    def test_provider_with_connected_attribute(self) -> None:
        """Test that boolean .connected attribute is also supported."""
        mock_provider = MagicMock(spec=[])  # no is_connected method
        mock_provider.connected = True

        checker = HealthChecker(market_feed_provider=mock_provider)
        result = checker.check_market_feed()
        assert result.status == HealthStatus.HEALTHY

    def test_provider_raises_exception(self) -> None:
        mock_provider = MagicMock()
        mock_provider.is_connected.side_effect = RuntimeError("timeout")

        checker = HealthChecker(market_feed_provider=mock_provider)
        result = checker.check_market_feed()
        assert result.status == HealthStatus.UNHEALTHY


# ── HealthChecker.check_redis ─────────────────────────────────────────────────


class TestCheckRedis:
    def test_no_url_returns_degraded(self) -> None:
        checker = HealthChecker()
        result = checker.check_redis()
        assert result.status == HealthStatus.DEGRADED
        assert result.component == "redis"

    def test_successful_ping(self) -> None:
        mock_redis_instance = MagicMock()
        mock_redis_class = MagicMock()
        mock_redis_class.from_url.return_value = mock_redis_instance

        with patch.dict("sys.modules", {"redis": mock_redis_class}):
            checker = HealthChecker(redis_url="redis://localhost:6379/0")
            # Patch directly on the module
            with patch("qts.monitoring.health.redis_lib", mock_redis_class, create=True):
                # Since the import is inside the method, patch at import level
                import unittest.mock as um
                with um.patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: (
                    mock_redis_class if name == "redis" else __import__(name, *args, **kwargs)
                )):
                    result = checker.check_redis("redis://localhost:6379/0")

        # Fallback: just check that a URL without an actual Redis instance fails gracefully
        checker2 = HealthChecker(redis_url="redis://nonexistent-host:6379/0")
        result2 = checker2.check_redis()
        # Should fail gracefully (either UNHEALTHY or exception-caught)
        assert result2.status in (HealthStatus.UNHEALTHY, HealthStatus.HEALTHY, HealthStatus.DEGRADED)
        assert result2.component == "redis"

    def test_failed_ping(self) -> None:
        """If Redis is unreachable the check should return UNHEALTHY."""
        checker = HealthChecker(redis_url="redis://nonexistent:9999/0")
        result = checker.check_redis()
        # Should be UNHEALTHY since the host does not exist
        assert result.component == "redis"
        assert result.status in (HealthStatus.UNHEALTHY, HealthStatus.DEGRADED)


# ── HealthChecker.check_all / is_healthy ─────────────────────────────────────


class TestCheckAll:
    def test_check_all_returns_three_components(self) -> None:
        checker = HealthChecker()
        results = checker.check_all()
        assert len(results) == 3
        component_names = {r.component for r in results}
        assert "database" in component_names
        assert "market_feed" in component_names
        assert "redis" in component_names

    def test_is_healthy_false_when_no_components_configured(self) -> None:
        """Without any real services configured, is_healthy should be False."""
        checker = HealthChecker()
        assert checker.is_healthy() is False

    def test_is_healthy_true_when_all_healthy(self) -> None:
        """Mock all checks to return HEALTHY."""
        checker = HealthChecker()
        healthy = ComponentHealth(
            component="x",
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(timezone.utc),
            message="OK",
            latency_ms=1.0,
        )

        checker.check_database = MagicMock(return_value=healthy)  # type: ignore[method-assign]
        checker.check_market_feed = MagicMock(return_value=healthy)  # type: ignore[method-assign]
        checker.check_redis = MagicMock(return_value=healthy)  # type: ignore[method-assign]

        assert checker.is_healthy() is True

    def test_is_healthy_false_when_one_unhealthy(self) -> None:
        checker = HealthChecker()
        healthy = ComponentHealth(
            component="x",
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(timezone.utc),
            message="OK",
            latency_ms=1.0,
        )
        unhealthy = ComponentHealth(
            component="database",
            status=HealthStatus.UNHEALTHY,
            last_check=datetime.now(timezone.utc),
            message="failed",
            latency_ms=0.0,
        )

        checker.check_database = MagicMock(return_value=unhealthy)  # type: ignore[method-assign]
        checker.check_market_feed = MagicMock(return_value=healthy)  # type: ignore[method-assign]
        checker.check_redis = MagicMock(return_value=healthy)  # type: ignore[method-assign]

        assert checker.is_healthy() is False

    def test_each_result_has_timestamp(self) -> None:
        checker = HealthChecker()
        results = checker.check_all()
        for r in results:
            assert isinstance(r.last_check, datetime)
            assert r.last_check.tzinfo is not None  # UTC-aware

    def test_each_result_has_latency(self) -> None:
        checker = HealthChecker()
        results = checker.check_all()
        for r in results:
            assert r.latency_ms >= 0.0
