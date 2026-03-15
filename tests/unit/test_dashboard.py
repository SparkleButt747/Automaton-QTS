"""Unit tests for the monitoring dashboard and alert system."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from qts.models.base import VolRegime
from qts.monitoring.alerts import Alert, AlertLevel, AlertManager
from qts.monitoring.dashboard import DashboardState, SignalValues


# ── DashboardState ────────────────────────────────────────────────────────────


class TestDashboardState:
    def test_default_construction(self) -> None:
        state = DashboardState()
        assert state.portfolio_value == 0.0
        assert state.positions == []
        assert state.recent_trades == []
        assert state.signals == []
        assert state.vol_regime == VolRegime.LOW
        assert state.pending_proposals == 0
        assert state.db_connected is True
        assert state.active_alerts == []

    def test_custom_values(self) -> None:
        state = DashboardState(
            portfolio_value=50_000.0,
            unrealized_pnl=-200.0,
            daily_pnl=-150.0,
            daily_drawdown=0.003,
            drawdown_limit=0.02,
            vol_regime=VolRegime.HIGH,
            llm_regime_assessment="Elevated volatility expected.",
            pending_proposals=2,
        )
        assert state.portfolio_value == 50_000.0
        assert state.vol_regime == VolRegime.HIGH
        assert state.pending_proposals == 2

    def test_signals_list(self) -> None:
        signals = [
            SignalValues(symbol="BTCUSDT", rsi=72.5, macd=0.0012, sentiment=0.4, combined_alpha=0.3),
            SignalValues(symbol="ETHUSDT", rsi=45.0, macd=-0.0005, sentiment=-0.1, combined_alpha=-0.05),
        ]
        state = DashboardState(signals=signals)
        assert len(state.signals) == 2
        assert state.signals[0].symbol == "BTCUSDT"

    def test_data_timestamps(self) -> None:
        now = datetime.now(timezone.utc)
        state = DashboardState(
            data_timestamps={"binance": now, "news": now}
        )
        assert "binance" in state.data_timestamps
        assert "news" in state.data_timestamps

    def test_last_updated_is_set_automatically(self) -> None:
        state = DashboardState()
        assert state.last_updated is not None
        assert isinstance(state.last_updated, datetime)


# ── SignalValues ──────────────────────────────────────────────────────────────


class TestSignalValues:
    def test_default_nan_values(self) -> None:
        import math
        sig = SignalValues(symbol="AAPL")
        assert math.isnan(sig.rsi)
        assert math.isnan(sig.macd)
        assert math.isnan(sig.sentiment)
        assert math.isnan(sig.combined_alpha)

    def test_with_values(self) -> None:
        sig = SignalValues(symbol="BTCUSDT", rsi=55.0, macd=0.002, sentiment=0.3, combined_alpha=0.2)
        assert sig.symbol == "BTCUSDT"
        assert sig.rsi == 55.0


# ── AlertManager ─────────────────────────────────────────────────────────────


class TestAlertGeneration:
    def test_no_alert_when_within_limits(self) -> None:
        mgr = AlertManager(drawdown_threshold=0.01)
        alert = mgr.check_drawdown(current_drawdown=0.005, limit=0.02)
        assert alert is None

    def test_warning_alert_when_approaching_limit(self) -> None:
        mgr = AlertManager(drawdown_threshold=0.01)
        alert = mgr.check_drawdown(current_drawdown=0.015, limit=0.02)
        assert alert is not None
        assert alert.level == AlertLevel.WARNING

    def test_critical_alert_at_limit(self) -> None:
        mgr = AlertManager(drawdown_threshold=0.01)
        alert = mgr.check_drawdown(current_drawdown=0.02, limit=0.02)
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL

    def test_critical_alert_beyond_limit(self) -> None:
        mgr = AlertManager(drawdown_threshold=0.01)
        alert = mgr.check_drawdown(current_drawdown=0.025, limit=0.02)
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL

    def test_get_active_alerts_accumulates(self) -> None:
        mgr = AlertManager(drawdown_threshold=0.01)
        mgr.check_drawdown(0.015, 0.02)  # WARNING
        mgr.check_drawdown(0.02, 0.02)   # CRITICAL
        alerts = mgr.get_active_alerts()
        assert len(alerts) == 2

    def test_clear_alerts(self) -> None:
        mgr = AlertManager()
        mgr.check_drawdown(0.02, 0.02)
        mgr.clear_alerts()
        assert mgr.get_active_alerts() == []

    def test_feed_staleness_no_alert_fresh(self) -> None:
        from datetime import timedelta
        mgr = AlertManager()
        recent = datetime.now(timezone.utc)
        alert = mgr.check_feed_staleness(last_tick_time=recent, max_age_seconds=60.0)
        assert alert is None

    def test_feed_staleness_critical_when_old(self) -> None:
        from datetime import timedelta
        mgr = AlertManager()
        old_ts = datetime(2000, 1, 1, tzinfo=timezone.utc)
        alert = mgr.check_feed_staleness(last_tick_time=old_ts, max_age_seconds=60.0)
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL

    def test_position_concentration_no_alert_within_limit(self) -> None:
        mgr = AlertManager(concentration_threshold=0.5)
        positions = {"BTCUSDT": 4000.0, "ETHUSDT": 4000.0, "AAPL": 2000.0}
        alert = mgr.check_position_concentration(positions, limit=0.50)
        assert alert is None

    def test_position_concentration_alert_when_exceeded(self) -> None:
        mgr = AlertManager()
        positions = {"BTCUSDT": 9000.0, "ETHUSDT": 1000.0}
        alert = mgr.check_position_concentration(positions, limit=0.50)
        assert alert is not None
        assert alert.level == AlertLevel.WARNING
        assert alert.component == "position_manager"

    def test_position_concentration_empty_portfolio(self) -> None:
        mgr = AlertManager()
        alert = mgr.check_position_concentration({}, limit=0.50)
        assert alert is None

    def test_alert_dataclass_is_frozen(self) -> None:
        alert = Alert(
            level=AlertLevel.INFO,
            message="test",
            timestamp=datetime.now(timezone.utc),
            component="test",
            value=0.0,
            threshold=0.0,
        )
        with pytest.raises((AttributeError, TypeError)):
            alert.message = "modified"  # type: ignore[misc]
