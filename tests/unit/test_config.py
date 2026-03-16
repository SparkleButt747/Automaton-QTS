"""Unit tests for qts.config module.

Tests the Pydantic models for correctness: weight validation,
risk limits, settings loading, and production readiness checks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from qts.config import (
    AppSettings,
    DatabaseSettings,
    RiskLimits,
    SentimentFusionWeights,
    SignalWeights,
    StrategyParams,
)

# ── SignalWeights ────────────────────────────────────────────────────────────


class TestSignalWeights:
    def test_valid_weights_sum_to_one(self) -> None:
        w = SignalWeights(w_rsi=0.20, w_macd=0.20, w_bb=0.15, w_mom=0.15, w_sentiment=0.30)
        assert abs((w.w_rsi + w.w_macd + w.w_bb + w.w_mom + w.w_sentiment) - 1.0) < 1e-9

    def test_weights_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            SignalWeights(w_rsi=0.21, w_macd=0.20, w_bb=0.20, w_mom=0.20, w_sentiment=0.20)

    def test_negative_weight_raises(self) -> None:
        with pytest.raises(ValidationError):
            SignalWeights(w_rsi=-0.10, w_macd=0.30, w_bb=0.30, w_mom=0.30, w_sentiment=0.20)

    def test_frozen_model(self) -> None:
        w = SignalWeights(w_rsi=0.20, w_macd=0.20, w_bb=0.15, w_mom=0.15, w_sentiment=0.30)
        with pytest.raises(ValidationError):
            w.w_rsi = 0.50  # type: ignore[misc]


# ── SentimentFusionWeights ───────────────────────────────────────────────────


class TestSentimentFusionWeights:
    def test_valid_weights(self) -> None:
        sfw = SentimentFusionWeights(news=0.40, social=0.30, geopolitical=0.30)
        assert abs((sfw.news + sfw.social + sfw.geopolitical) - 1.0) < 1e-9

    def test_invalid_sum_raises(self) -> None:
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            SentimentFusionWeights(news=0.50, social=0.30, geopolitical=0.30)

    def test_frozen_model(self) -> None:
        sfw = SentimentFusionWeights(news=0.40, social=0.30, geopolitical=0.30)
        with pytest.raises(ValidationError):
            sfw.news = 0.60  # type: ignore[misc]


# ── RiskLimits ───────────────────────────────────────────────────────────────


class TestRiskLimits:
    def test_valid_risk_limits(self) -> None:
        rl = RiskLimits(
            max_daily_drawdown_pct=0.02,
            max_position_size_pct=0.05,
            max_open_positions=5,
            circuit_breaker_cooldown_seconds=3600,
            sentiment_signal_max_scalar=2.0,
        )
        assert rl.max_daily_drawdown_pct == 0.02

    def test_drawdown_too_high_raises(self) -> None:
        with pytest.raises(ValidationError):
            RiskLimits(
                max_daily_drawdown_pct=0.99,  # > 0.5 limit
                max_position_size_pct=0.05,
                max_open_positions=5,
                circuit_breaker_cooldown_seconds=3600,
                sentiment_signal_max_scalar=2.0,
            )

    def test_frozen_model(self) -> None:
        rl = RiskLimits(
            max_daily_drawdown_pct=0.02,
            max_position_size_pct=0.05,
            max_open_positions=5,
            circuit_breaker_cooldown_seconds=3600,
            sentiment_signal_max_scalar=2.0,
        )
        with pytest.raises(ValidationError):
            rl.max_daily_drawdown_pct = 0.99  # type: ignore[misc]

    def test_from_file(self, tmp_path: Path) -> None:
        data = {
            "max_daily_drawdown_pct": 0.02,
            "max_position_size_pct": 0.05,
            "max_open_positions": 5,
            "circuit_breaker_cooldown_seconds": 3600,
            "sentiment_signal_max_scalar": 2.0,
        }
        path = tmp_path / "risk_limits.json"
        path.write_text(json.dumps(data))
        rl = RiskLimits.from_file(path)
        assert rl.max_open_positions == 5

    def test_from_file_strips_comment_key(self, tmp_path: Path) -> None:
        data = {
            "_comment": "This is an immutable file",
            "max_daily_drawdown_pct": 0.02,
            "max_position_size_pct": 0.05,
            "max_open_positions": 5,
            "circuit_breaker_cooldown_seconds": 3600,
            "sentiment_signal_max_scalar": 2.0,
        }
        path = tmp_path / "risk_limits.json"
        path.write_text(json.dumps(data))
        rl = RiskLimits.from_file(path)
        assert rl.max_daily_drawdown_pct == 0.02


# ── StrategyParams ───────────────────────────────────────────────────────────


class TestStrategyParams:
    def test_valid_params(self) -> None:
        params = StrategyParams(
            version="1.0.0",
            weights=SignalWeights(w_rsi=0.20, w_macd=0.20, w_bb=0.15, w_mom=0.15, w_sentiment=0.30),
            entry_threshold=0.25,
            exit_threshold=-0.10,
            max_hold_bars=48,
            sentiment_fusion_weights=SentimentFusionWeights(
                news=0.40, social=0.30, geopolitical=0.30
            ),
        )
        assert params.version == "1.0.0"

    def test_exit_must_be_below_entry(self) -> None:
        with pytest.raises(ValidationError, match="exit_threshold"):
            StrategyParams(
                version="1.0.0",
                weights=SignalWeights(
                    w_rsi=0.20, w_macd=0.20, w_bb=0.15, w_mom=0.15, w_sentiment=0.30
                ),
                entry_threshold=0.25,
                exit_threshold=0.30,  # exit > entry — invalid
                max_hold_bars=48,
                sentiment_fusion_weights=SentimentFusionWeights(
                    news=0.40, social=0.30, geopolitical=0.30
                ),
            )

    def test_from_file(self, tmp_path: Path) -> None:
        data = {
            "version": "1.0.0",
            "weights": {
                "w_rsi": 0.20,
                "w_macd": 0.20,
                "w_bb": 0.15,
                "w_mom": 0.15,
                "w_sentiment": 0.30,
            },
            "entry_threshold": 0.25,
            "exit_threshold": -0.10,
            "max_hold_bars": 48,
            "sentiment_fusion_weights": {"news": 0.40, "social": 0.30, "geopolitical": 0.30},
        }
        path = tmp_path / "params.json"
        path.write_text(json.dumps(data))
        params = StrategyParams.from_file(path)
        assert params.max_hold_bars == 48
        assert params.weights.w_sentiment == 0.30


# ── DatabaseSettings ─────────────────────────────────────────────────────────


class TestDatabaseSettings:
    def test_default_sqlite(self) -> None:
        db = DatabaseSettings()
        assert db.is_sqlite
        assert not db.is_postgres

    def test_postgres_detection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost/db")
        db = DatabaseSettings()
        assert db.is_postgres
        assert not db.is_sqlite


# ── AppSettings ───────────────────────────────────────────────────────────────


class TestAppSettings:
    def test_defaults(self) -> None:
        settings = AppSettings()
        assert settings.qts_env == "development"
        assert settings.qts_dry_run is True
        assert settings.is_development

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QTS_ENV", "production")
        monkeypatch.setenv("QTS_DRY_RUN", "false")
        settings = AppSettings()
        assert settings.is_production
        assert settings.qts_dry_run is False

    def test_production_readiness_dry_run_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QTS_ENV", "production")
        monkeypatch.setenv("QTS_DRY_RUN", "true")
        settings = AppSettings()
        issues = settings.validate_production_readiness()
        assert any("dry_run" in i.lower() or "DRY_RUN" in i for i in issues)
