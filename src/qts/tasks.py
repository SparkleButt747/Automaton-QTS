"""Celery task definitions for async background jobs."""
from __future__ import annotations

import logging
from typing import Any

from celery import Celery
from celery.schedules import crontab

from qts.config import get_settings
from qts.oversight.llm_client import create_llm_client

logger = logging.getLogger(__name__)

# ── Celery Application ────────────────────────────────────────────────────────

_settings = get_settings()
redis_url: str = _settings.redis_url

app = Celery("qts", broker=redis_url, backend=redis_url)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
)

# ── Beat Schedule (periodic tasks) ───────────────────────────────────────────

app.conf.beat_schedule = {
    "refresh-news-sentiment-every-5min": {
        "task": "qts.tasks.refresh_news_sentiment",
        "schedule": 300,  # every 5 minutes
        "args": ("default",),
    },
    "refresh-social-sentiment-every-15min": {
        "task": "qts.tasks.refresh_social_sentiment",
        "schedule": 900,  # every 15 minutes
        "args": ("default",),
    },
    "poll-gdelt-events-every-30min": {
        "task": "qts.tasks.poll_gdelt_events",
        "schedule": 1800,  # every 30 minutes
    },
    "run-nightly-attribution-at-midnight": {
        "task": "qts.tasks.run_nightly_attribution",
        "schedule": crontab(hour=0, minute=0),
    },
    "run-debrief-at-00-30-utc": {
        "task": "qts.tasks.run_debrief",
        "schedule": crontab(hour=0, minute=30),
        "args": ("auto",),
    },
    "health-check-every-minute": {
        "task": "qts.tasks.health_check",
        "schedule": 60,  # every 1 minute
    },
}


# ── Tasks ─────────────────────────────────────────────────────────────────────


@app.task(bind=True, name="qts.tasks.refresh_news_sentiment")  # type: ignore[untyped-decorator]
def refresh_news_sentiment(self: Any, symbol: str) -> dict[str, Any]:
    """Fetch recent news headlines and update sentiment for a symbol using FinBERT.

    Args:
        symbol: Asset symbol to refresh news sentiment for.

    Returns:
        Dict with symbol, sentiment score, and article count.
    """
    logger.info("refresh_news_sentiment: starting for symbol=%s", symbol)
    try:
        from qts.nlp.finbert import FinBERTAnalyzer  # noqa: PLC0415

        analyzer = FinBERTAnalyzer()

        # In a real implementation this would fetch from a news API.
        # Here we demonstrate the structure with a placeholder fetch.
        headlines: list[str] = _fetch_news_headlines(symbol)

        results = analyzer.analyze(headlines) if headlines else []

        scores = []
        for r in results:
            if r.label == "POSITIVE":
                scores.append(r.score)
            elif r.label == "NEGATIVE":
                scores.append(-r.score)
            else:
                scores.append(0.0)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        logger.info(
            "refresh_news_sentiment: done for symbol=%s, score=%.4f, articles=%d",
            symbol,
            avg_score,
            len(headlines),
        )
        return {"symbol": symbol, "sentiment_score": avg_score, "article_count": len(headlines)}

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "refresh_news_sentiment: error for symbol=%s: %s",
            symbol,
            exc,
            exc_info=True,
        )
        return {"symbol": symbol, "error": str(exc)}


@app.task(bind=True, name="qts.tasks.refresh_social_sentiment")  # type: ignore[untyped-decorator]
def refresh_social_sentiment(self: Any, symbol: str) -> dict[str, Any]:
    """Fetch recent Reddit/StockTwits posts and update sentiment using VADER.

    Args:
        symbol: Asset symbol to refresh social sentiment for.

    Returns:
        Dict with symbol, sentiment score, and post count.
    """
    logger.info("refresh_social_sentiment: starting for symbol=%s", symbol)
    try:
        from qts.nlp.vader import VaderAnalyzer  # noqa: PLC0415

        analyzer = VaderAnalyzer()

        # In a real implementation this would fetch from Reddit/StockTwits APIs.
        posts: list[str] = _fetch_social_posts(symbol)

        results = analyzer.analyze(posts) if posts else []

        scores = []
        for r in results:
            if r.label == "POSITIVE":
                scores.append(r.score)
            elif r.label == "NEGATIVE":
                scores.append(-r.score)
            else:
                scores.append(0.0)

        avg_score = sum(scores) / len(scores) if scores else 0.0

        logger.info(
            "refresh_social_sentiment: done for symbol=%s, score=%.4f, posts=%d",
            symbol,
            avg_score,
            len(posts),
        )
        return {"symbol": symbol, "sentiment_score": avg_score, "post_count": len(posts)}

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "refresh_social_sentiment: error for symbol=%s: %s",
            symbol,
            exc,
            exc_info=True,
        )
        return {"symbol": symbol, "error": str(exc)}


@app.task(bind=True, name="qts.tasks.poll_gdelt_events")  # type: ignore[untyped-decorator]
def poll_gdelt_events(self: Any) -> dict[str, Any]:
    """Fetch recent GDELT events and process them with GDELTProcessor.

    Returns:
        Dict with event count and processing status.
    """
    logger.info("poll_gdelt_events: starting")
    try:
        from qts.nlp.gdelt import GDELTProcessor  # noqa: PLC0415

        processor = GDELTProcessor()

        # In a real implementation this would fetch from the GDELT API.
        events = _fetch_gdelt_events()

        # Compute a sample intensity score for a set of default symbols
        sample_symbols = ["SPY", "energy", "tech"]
        results: dict[str, float] = {}
        for sym in sample_symbols:
            intensity = processor.compute_conflict_intensity(events, sym)
            results[sym] = intensity

        logger.info(
            "poll_gdelt_events: done, events=%d, symbols=%s",
            len(events),
            list(results.keys()),
        )
        return {"event_count": len(events), "intensities": results}

    except Exception as exc:  # noqa: BLE001
        logger.error("poll_gdelt_events: error: %s", exc, exc_info=True)
        return {"error": str(exc)}


@app.task(bind=True, name="qts.tasks.run_nightly_attribution")  # type: ignore[untyped-decorator]
def run_nightly_attribution(self: Any) -> dict[str, Any]:
    """Load today's completed trades and run the AttributionEngine.

    Returns:
        Dict with trade count, win rate, and failure mode summary.
    """
    logger.info("run_nightly_attribution: starting")
    try:
        from qts.analytics.attribution import AttributionEngine  # noqa: PLC0415

        engine = AttributionEngine()

        # In a real implementation this would load trades from the database.
        trades = _load_todays_trades()
        snapshots: dict[str, object] = {}

        report = engine.run_attribution(trades, snapshots)  # type: ignore[arg-type]

        logger.info(
            "run_nightly_attribution: done, total_trades=%d, win_rate=%.2f, losing=%d",
            report.total_trades,
            report.sentiment_hit_rate,
            report.losing_trades,
        )
        return {
            "total_trades": report.total_trades,
            "winning_trades": report.winning_trades,
            "losing_trades": report.losing_trades,
            "sentiment_hit_rate": report.sentiment_hit_rate,
            "sentiment_pnl_correlation": report.sentiment_pnl_correlation,
        }

    except Exception as exc:  # noqa: BLE001
        logger.error("run_nightly_attribution: error: %s", exc, exc_info=True)
        return {"error": str(exc)}


@app.task(bind=True, name="qts.tasks.run_debrief")  # type: ignore[untyped-decorator]
def run_debrief(self: Any, session_date: str) -> dict[str, Any]:
    """Construct a SessionSummary and run the DebriefEngine for LLM analysis.

    Args:
        session_date: ISO date string for the session (e.g. "2024-01-15").
            Pass "auto" to use today's date.

    Returns:
        Dict with session_date, proposal count, and analysis excerpt.
    """
    import asyncio  # noqa: PLC0415
    from datetime import datetime, timezone  # noqa: PLC0415

    if session_date == "auto":
        session_date = datetime.now(tz=timezone.utc).date().isoformat()

    logger.info("run_debrief: starting for session_date=%s", session_date)
    try:
        from pathlib import Path  # noqa: PLC0415

        from qts.oversight.debrief import DebriefEngine, SessionSummary  # noqa: PLC0415

        settings = get_settings()
        llm_settings = settings.llm
        llm_kwargs: dict[str, Any] = {"backend": llm_settings.backend}
        if llm_settings.backend == "anthropic":
            llm_kwargs["api_key"] = llm_settings.anthropic_api_key
            llm_kwargs["model"] = llm_settings.anthropic_model
        else:
            llm_kwargs["model"] = llm_settings.ollama_model
            llm_kwargs["base_url"] = llm_settings.ollama_base_url
        llm_client = create_llm_client(**llm_kwargs)

        config_dir = Path("config")
        engine = DebriefEngine(llm_client=llm_client, config_dir=config_dir)

        # In a real implementation this would aggregate real session data.
        summary = _build_session_summary(session_date)

        report = asyncio.run(engine.run_debrief(summary))

        logger.info(
            "run_debrief: done for session_date=%s, proposals=%d",
            session_date,
            len(report.proposals),
        )
        return {
            "session_date": report.session_date,
            "proposal_count": len(report.proposals),
            "regime_alert": report.regime_alert,
            "analysis_excerpt": report.analysis[:200] if report.analysis else "",
        }

    except Exception as exc:  # noqa: BLE001
        logger.error("run_debrief: error for session_date=%s: %s", session_date, exc, exc_info=True)
        return {"session_date": session_date, "error": str(exc)}


@app.task(bind=True, name="qts.tasks.health_check")  # type: ignore[untyped-decorator]
def health_check(self: Any) -> dict[str, Any]:
    """Run the HealthChecker and log results for all components.

    Returns:
        Dict mapping component names to their health status.
    """
    logger.info("health_check: starting")
    try:
        from qts.monitoring.health import HealthChecker  # noqa: PLC0415

        settings = get_settings()
        checker = HealthChecker(redis_url=settings.redis_url)
        results = checker.check_all()

        status_map: dict[str, str] = {}
        for component_health in results:
            status_map[component_health.component] = component_health.status.value
            logger.info(
                "health_check: component=%s status=%s latency=%.1fms message=%s",
                component_health.component,
                component_health.status,
                component_health.latency_ms,
                component_health.message,
            )

        logger.info("health_check: done, statuses=%s", status_map)
        return {"statuses": status_map}

    except Exception as exc:  # noqa: BLE001
        logger.error("health_check: error: %s", exc, exc_info=True)
        return {"error": str(exc)}


# ── Internal helpers (stubs for real data sources) ────────────────────────────


def _fetch_news_headlines(symbol: str) -> list[str]:
    """Stub: fetch recent news headlines for a symbol.

    In production this would query a news API (e.g. Alpha Vantage, NewsAPI).

    Args:
        symbol: Asset symbol.

    Returns:
        List of headline strings.
    """
    return []


def _fetch_social_posts(symbol: str) -> list[str]:
    """Stub: fetch recent Reddit/StockTwits posts for a symbol.

    In production this would query the Reddit PRAW API or StockTwits REST API.

    Args:
        symbol: Asset symbol.

    Returns:
        List of post text strings.
    """
    return []


def _fetch_gdelt_events() -> list[Any]:
    """Stub: fetch recent events from the GDELT API.

    Returns:
        List of GeopoliticalEvent objects.
    """
    return []


def _load_todays_trades() -> list[Any]:
    """Stub: load completed trades for today from the database.

    Returns:
        List of TradeRecord objects.
    """
    return []


def _build_session_summary(session_date: str) -> Any:
    """Stub: build a SessionSummary from aggregated session data.

    Args:
        session_date: ISO date string for the session.

    Returns:
        SessionSummary instance.
    """
    from qts.oversight.debrief import SessionSummary  # noqa: PLC0415

    return SessionSummary(
        total_trades=0,
        win_rate=0.0,
        total_pnl=0.0,
        sharpe_today=0.0,
        failure_mode_summary={},
        sentiment_hit_rate=0.0,
        sentiment_pnl_correlation=0.0,
        top_loss_trades=[],
        current_params={},
        vol_regime="LOW",
        gdelt_top_events=[],
    )
