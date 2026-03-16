"""Unit tests for qts.tasks Celery configuration and task definitions.

Tests:
- Celery app is configured correctly
- All tasks are registered
- Beat schedule has expected entries
- Tasks handle missing dependencies gracefully (mocked)
- Tasks do not actually start Celery or Redis
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestCeleryAppConfiguration:
    """Test that the Celery app is correctly configured."""

    def test_celery_app_name(self) -> None:
        from qts.tasks import app

        assert app.main == "qts"

    def test_task_serializer_is_json(self) -> None:
        from qts.tasks import app

        assert app.conf.task_serializer == "json"

    def test_result_serializer_is_json(self) -> None:
        from qts.tasks import app

        assert app.conf.result_serializer == "json"

    def test_accept_content_includes_json(self) -> None:
        from qts.tasks import app

        assert "json" in app.conf.accept_content

    def test_timezone_is_utc(self) -> None:
        from qts.tasks import app

        assert app.conf.timezone == "UTC"

    def test_enable_utc(self) -> None:
        from qts.tasks import app

        assert app.conf.enable_utc is True

    def test_result_expires_is_3600(self) -> None:
        from qts.tasks import app

        assert app.conf.result_expires == 3600

    def test_broker_uses_redis_url(self) -> None:
        from qts.tasks import app, redis_url

        assert app.conf.broker_url == redis_url or "redis" in (app.conf.broker_url or "")

    def test_redis_url_is_string(self) -> None:
        from qts.tasks import redis_url

        assert isinstance(redis_url, str)
        assert redis_url.startswith("redis://")


class TestTaskRegistration:
    """Test that all expected tasks are registered on the Celery app."""

    def test_refresh_news_sentiment_registered(self) -> None:
        from qts.tasks import app

        assert "qts.tasks.refresh_news_sentiment" in app.tasks

    def test_refresh_social_sentiment_registered(self) -> None:
        from qts.tasks import app

        assert "qts.tasks.refresh_social_sentiment" in app.tasks

    def test_poll_gdelt_events_registered(self) -> None:
        from qts.tasks import app

        assert "qts.tasks.poll_gdelt_events" in app.tasks

    def test_run_nightly_attribution_registered(self) -> None:
        from qts.tasks import app

        assert "qts.tasks.run_nightly_attribution" in app.tasks

    def test_run_debrief_registered(self) -> None:
        from qts.tasks import app

        assert "qts.tasks.run_debrief" in app.tasks

    def test_health_check_registered(self) -> None:
        from qts.tasks import app

        assert "qts.tasks.health_check" in app.tasks

    def test_all_six_tasks_exist(self) -> None:
        from qts.tasks import app

        expected = {
            "qts.tasks.refresh_news_sentiment",
            "qts.tasks.refresh_social_sentiment",
            "qts.tasks.poll_gdelt_events",
            "qts.tasks.run_nightly_attribution",
            "qts.tasks.run_debrief",
            "qts.tasks.health_check",
        }
        registered = set(app.tasks.keys())
        missing = expected - registered
        assert not missing, f"Missing tasks: {missing}"


class TestBeatSchedule:
    """Test that the beat schedule contains the expected periodic entries."""

    def test_beat_schedule_is_dict(self) -> None:
        from qts.tasks import app

        assert isinstance(app.conf.beat_schedule, dict)

    def test_refresh_news_sentiment_schedule_present(self) -> None:
        from qts.tasks import app

        tasks = {v["task"] for v in app.conf.beat_schedule.values()}
        assert "qts.tasks.refresh_news_sentiment" in tasks

    def test_refresh_social_sentiment_schedule_present(self) -> None:
        from qts.tasks import app

        tasks = {v["task"] for v in app.conf.beat_schedule.values()}
        assert "qts.tasks.refresh_social_sentiment" in tasks

    def test_poll_gdelt_events_schedule_present(self) -> None:
        from qts.tasks import app

        tasks = {v["task"] for v in app.conf.beat_schedule.values()}
        assert "qts.tasks.poll_gdelt_events" in tasks

    def test_run_nightly_attribution_schedule_present(self) -> None:
        from qts.tasks import app

        tasks = {v["task"] for v in app.conf.beat_schedule.values()}
        assert "qts.tasks.run_nightly_attribution" in tasks

    def test_run_debrief_schedule_present(self) -> None:
        from qts.tasks import app

        tasks = {v["task"] for v in app.conf.beat_schedule.values()}
        assert "qts.tasks.run_debrief" in tasks

    def test_health_check_schedule_present(self) -> None:
        from qts.tasks import app

        tasks = {v["task"] for v in app.conf.beat_schedule.values()}
        assert "qts.tasks.health_check" in tasks

    def test_refresh_news_sentiment_interval_5min(self) -> None:
        """refresh_news_sentiment should run every 5 minutes (300 seconds)."""
        from qts.tasks import app

        for entry in app.conf.beat_schedule.values():
            if entry["task"] == "qts.tasks.refresh_news_sentiment":
                assert entry["schedule"] == 300
                return
        raise AssertionError("refresh_news_sentiment not found in beat schedule")

    def test_refresh_social_sentiment_interval_15min(self) -> None:
        """refresh_social_sentiment should run every 15 minutes (900 seconds)."""
        from qts.tasks import app

        for entry in app.conf.beat_schedule.values():
            if entry["task"] == "qts.tasks.refresh_social_sentiment":
                assert entry["schedule"] == 900
                return
        raise AssertionError("refresh_social_sentiment not found in beat schedule")

    def test_poll_gdelt_events_interval_30min(self) -> None:
        """poll_gdelt_events should run every 30 minutes (1800 seconds)."""
        from qts.tasks import app

        for entry in app.conf.beat_schedule.values():
            if entry["task"] == "qts.tasks.poll_gdelt_events":
                assert entry["schedule"] == 1800
                return
        raise AssertionError("poll_gdelt_events not found in beat schedule")

    def test_run_nightly_attribution_uses_crontab(self) -> None:
        """run_nightly_attribution should use a crontab schedule."""
        from celery.schedules import crontab

        from qts.tasks import app

        for entry in app.conf.beat_schedule.values():
            if entry["task"] == "qts.tasks.run_nightly_attribution":
                assert isinstance(entry["schedule"], crontab)
                return
        raise AssertionError("run_nightly_attribution not found in beat schedule")

    def test_run_debrief_uses_crontab(self) -> None:
        """run_debrief should use a crontab schedule."""
        from celery.schedules import crontab

        from qts.tasks import app

        for entry in app.conf.beat_schedule.values():
            if entry["task"] == "qts.tasks.run_debrief":
                assert isinstance(entry["schedule"], crontab)
                return
        raise AssertionError("run_debrief not found in beat schedule")

    def test_health_check_interval_1min(self) -> None:
        """health_check should run every 1 minute (60 seconds)."""
        from qts.tasks import app

        for entry in app.conf.beat_schedule.values():
            if entry["task"] == "qts.tasks.health_check":
                assert entry["schedule"] == 60
                return
        raise AssertionError("health_check not found in beat schedule")


class TestTaskExecutionWithMocks:
    """Test task execution with mocked heavy dependencies.

    These tests verify that tasks handle exceptions and missing dependencies
    gracefully without actually connecting to Redis or Celery.
    """

    def test_refresh_news_sentiment_returns_dict(self) -> None:
        from qts.tasks import refresh_news_sentiment

        # Call the underlying function directly (bypassing Celery broker)
        with patch("qts.tasks._fetch_news_headlines", return_value=[]):
            result = refresh_news_sentiment.run("AAPL")
        assert isinstance(result, dict)
        assert "symbol" in result
        assert result["symbol"] == "AAPL"

    def test_refresh_news_sentiment_with_headlines(self) -> None:
        from qts.tasks import refresh_news_sentiment

        mock_result = MagicMock()
        mock_result.label = "POSITIVE"
        mock_result.score = 0.8

        with (
            patch(
                "qts.tasks._fetch_news_headlines", return_value=["Apple rises on strong earnings"]
            ),
            patch("qts.nlp.finbert.FinBERTAnalyzer.analyze", return_value=[mock_result]),
        ):
            result = refresh_news_sentiment.run("AAPL")

        assert isinstance(result, dict)
        assert result["symbol"] == "AAPL"
        assert "sentiment_score" in result

    def test_refresh_news_sentiment_handles_exception(self) -> None:
        """Task should return dict with 'error' key on failure."""
        from qts.tasks import refresh_news_sentiment

        with patch(
            "qts.tasks._fetch_news_headlines", side_effect=RuntimeError("connection refused")
        ):
            result = refresh_news_sentiment.run("AAPL")

        assert isinstance(result, dict)
        assert "error" in result

    def test_refresh_social_sentiment_returns_dict(self) -> None:
        from qts.tasks import refresh_social_sentiment

        with patch("qts.tasks._fetch_social_posts", return_value=[]):
            result = refresh_social_sentiment.run("TSLA")

        assert isinstance(result, dict)
        assert result["symbol"] == "TSLA"

    def test_refresh_social_sentiment_handles_exception(self) -> None:
        from qts.tasks import refresh_social_sentiment

        with patch("qts.tasks._fetch_social_posts", side_effect=RuntimeError("API timeout")):
            result = refresh_social_sentiment.run("TSLA")

        assert isinstance(result, dict)
        assert "error" in result

    def test_poll_gdelt_events_returns_dict(self) -> None:
        from qts.tasks import poll_gdelt_events

        with patch("qts.tasks._fetch_gdelt_events", return_value=[]):
            result = poll_gdelt_events.run()

        assert isinstance(result, dict)
        assert "event_count" in result
        assert result["event_count"] == 0

    def test_poll_gdelt_events_handles_exception(self) -> None:
        from qts.tasks import poll_gdelt_events

        with patch(
            "qts.tasks._fetch_gdelt_events", side_effect=ConnectionError("GDELT unavailable")
        ):
            result = poll_gdelt_events.run()

        assert isinstance(result, dict)
        assert "error" in result

    def test_run_nightly_attribution_returns_dict(self) -> None:
        from qts.tasks import run_nightly_attribution

        with patch("qts.tasks._load_todays_trades", return_value=[]):
            result = run_nightly_attribution.run()

        assert isinstance(result, dict)
        assert "total_trades" in result
        assert result["total_trades"] == 0

    def test_run_nightly_attribution_handles_exception(self) -> None:
        from qts.tasks import run_nightly_attribution

        with patch("qts.tasks._load_todays_trades", side_effect=Exception("DB error")):
            result = run_nightly_attribution.run()

        assert isinstance(result, dict)
        assert "error" in result

    def test_run_debrief_auto_date(self) -> None:
        """run_debrief with 'auto' session_date should use today's date."""
        from qts.tasks import run_debrief

        mock_report = MagicMock()
        mock_report.session_date = "2024-01-15"
        mock_report.proposals = []
        mock_report.regime_alert = None
        mock_report.analysis = "Test analysis"

        mock_engine = MagicMock()
        mock_engine.run_debrief.return_value = mock_report

        with (
            patch("qts.tasks._build_session_summary", return_value=MagicMock()),
            patch("qts.oversight.debrief.DebriefEngine", return_value=mock_engine),
            patch("qts.tasks.create_llm_client", return_value=MagicMock()),
            patch("asyncio.run", return_value=mock_report),
        ):
            result = run_debrief.run("auto")

        assert isinstance(result, dict)

    def test_run_debrief_handles_exception(self) -> None:
        from qts.tasks import run_debrief

        with patch("qts.tasks._build_session_summary", side_effect=Exception("LLM failure")):
            result = run_debrief.run("2024-01-15")

        assert isinstance(result, dict)
        assert "error" in result

    def test_health_check_returns_dict(self) -> None:
        from qts.tasks import health_check

        mock_component = MagicMock()
        mock_component.component = "redis"
        mock_component.status.value = "HEALTHY"
        mock_component.latency_ms = 1.5
        mock_component.message = "Redis ping OK."

        mock_checker = MagicMock()
        mock_checker.check_all.return_value = [mock_component]

        with patch("qts.monitoring.health.HealthChecker", return_value=mock_checker):
            result = health_check.run()

        assert isinstance(result, dict)
        assert "statuses" in result

    def test_health_check_handles_exception(self) -> None:
        from qts.tasks import health_check

        with patch("qts.monitoring.health.HealthChecker", side_effect=Exception("Redis down")):
            result = health_check.run()

        assert isinstance(result, dict)
        assert "error" in result
