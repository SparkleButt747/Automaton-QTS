"""QTS Configuration module.

Provides Pydantic Settings models for all system configuration:
- RiskLimits: Immutable risk controls loaded from config/risk_limits.json
- StrategyParams: Strategy parameters loaded from config/params.json
- SentimentFusionWeights: Weights for multi-source sentiment fusion
- ExchangeSettings: Exchange API credentials from environment
- DatabaseSettings: Database connection settings from environment
- AppSettings: Master settings class composing all of the above
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# ── Path helpers ──────────────────────────────────────────────────────────────

# Project root is two levels up from this file: src/qts/config.py -> src/qts -> src -> project_root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


def _load_json(path: Path) -> dict[str, Any]:
    """Load and parse a JSON file, raising a clear error on failure."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. "
            f"Expected project layout with config/ at: {_CONFIG_DIR}"
        )
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[no-any-return]


# ── Risk Limits (IMMUTABLE) ───────────────────────────────────────────────────


class RiskLimits(BaseModel):
    """Immutable risk control limits.

    These values are loaded from config/risk_limits.json and are frozen
    at runtime. Any modification requires an explicit file change and
    will be rejected by the pre-commit hook unless --force is used.

    Attributes:
        max_daily_drawdown_pct: Maximum allowed daily portfolio drawdown (0-1).
        max_position_size_pct: Maximum single-position size as fraction of portfolio (0-1).
        max_open_positions: Maximum number of simultaneously open positions.
        circuit_breaker_cooldown_seconds: Cooldown after circuit breaker triggers (seconds).
        sentiment_signal_max_scalar: Maximum multiplier the sentiment signal can apply.
    """

    model_config = {"frozen": True}

    max_daily_drawdown_pct: Annotated[
        float, Field(gt=0, le=0.5, description="Max daily drawdown as fraction (e.g. 0.02 = 2%)")
    ]
    max_position_size_pct: Annotated[
        float, Field(gt=0, le=0.5, description="Max position size as fraction of portfolio")
    ]
    max_open_positions: Annotated[int, Field(ge=1, le=50, description="Max simultaneous positions")]
    circuit_breaker_cooldown_seconds: Annotated[
        int, Field(ge=0, description="Cooldown after circuit breaker trip (seconds)")
    ]
    sentiment_signal_max_scalar: Annotated[
        float, Field(ge=1.0, le=10.0, description="Max sentiment signal multiplier")
    ]

    @classmethod
    def from_file(cls, path: Path | None = None) -> "RiskLimits":
        """Load risk limits from the JSON config file."""
        resolved = path or (_CONFIG_DIR / "risk_limits.json")
        raw = _load_json(resolved)
        # Strip _comment key if present
        raw.pop("_comment", None)
        return cls(**raw)


# ── Sentiment Fusion Weights ──────────────────────────────────────────────────


class SentimentFusionWeights(BaseModel):
    """Weights for combining sentiment signals from multiple sources.

    Weights must sum to exactly 1.0.

    Attributes:
        news: Weight for news-derived sentiment signal.
        social: Weight for social media sentiment signal.
        geopolitical: Weight for geopolitical event sentiment signal.
    """

    model_config = {"frozen": True}

    news: Annotated[float, Field(ge=0.0, le=1.0)]
    social: Annotated[float, Field(ge=0.0, le=1.0)]
    geopolitical: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def weights_must_sum_to_one(self) -> "SentimentFusionWeights":
        """Validate that all fusion weights sum to 1.0 (within floating point tolerance)."""
        total = self.news + self.social + self.geopolitical
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"SentimentFusionWeights must sum to 1.0, got {total:.6f}. "
                f"Values: news={self.news}, social={self.social}, geopolitical={self.geopolitical}"
            )
        return self


# ── Strategy Signal Weights ───────────────────────────────────────────────────


class SignalWeights(BaseModel):
    """Weights for individual technical and sentiment signals.

    All weights must sum to exactly 1.0.

    Attributes:
        w_rsi: Weight for RSI signal component.
        w_macd: Weight for MACD signal component.
        w_bb: Weight for Bollinger Bands signal component.
        w_mom: Weight for momentum signal component.
        w_sentiment: Weight for the aggregated sentiment signal.
    """

    model_config = {"frozen": True}

    w_rsi: Annotated[float, Field(ge=0.0, le=1.0, description="RSI signal weight")]
    w_macd: Annotated[float, Field(ge=0.0, le=1.0, description="MACD signal weight")]
    w_bb: Annotated[float, Field(ge=0.0, le=1.0, description="Bollinger Bands signal weight")]
    w_mom: Annotated[float, Field(ge=0.0, le=1.0, description="Momentum signal weight")]
    w_sentiment: Annotated[float, Field(ge=0.0, le=1.0, description="Sentiment signal weight")]

    @model_validator(mode="after")
    def weights_must_sum_to_one(self) -> "SignalWeights":
        """Validate that all signal weights sum to 1.0 (within floating point tolerance)."""
        total = self.w_rsi + self.w_macd + self.w_bb + self.w_mom + self.w_sentiment
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"SignalWeights must sum to 1.0, got {total:.6f}. "
                f"Values: w_rsi={self.w_rsi}, w_macd={self.w_macd}, w_bb={self.w_bb}, "
                f"w_mom={self.w_mom}, w_sentiment={self.w_sentiment}"
            )
        return self


# ── Strategy Parameters ───────────────────────────────────────────────────────


class StrategyParams(BaseModel):
    """Strategy parameters loaded from config/params.json.

    Attributes:
        version: Config schema version string.
        weights: Signal weighting configuration.
        entry_threshold: Composite score threshold required to enter a position.
        exit_threshold: Composite score below which to exit a position.
        max_hold_bars: Maximum number of bars to hold a position before forced exit.
        sentiment_fusion_weights: How to combine multi-source sentiment scores.
    """

    model_config = {"frozen": True}

    version: str = Field(description="Config schema version")
    weights: SignalWeights
    entry_threshold: Annotated[
        float, Field(description="Minimum composite score to enter a position")
    ]
    exit_threshold: Annotated[
        float, Field(description="Composite score below which position is exited")
    ]
    max_hold_bars: Annotated[int, Field(ge=1, description="Max bars to hold a position")]
    sentiment_fusion_weights: SentimentFusionWeights

    @field_validator("entry_threshold")
    @classmethod
    def entry_must_be_positive(cls, v: float) -> float:
        """Entry threshold should be a positive score to avoid trivial triggers."""
        if v <= 0:
            raise ValueError(f"entry_threshold must be > 0, got {v}")
        return v

    @model_validator(mode="after")
    def exit_below_entry(self) -> "StrategyParams":
        """Exit threshold must be strictly less than entry threshold."""
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError(
                f"exit_threshold ({self.exit_threshold}) must be less than "
                f"entry_threshold ({self.entry_threshold})"
            )
        return self

    @classmethod
    def from_file(cls, path: Path | None = None) -> "StrategyParams":
        """Load strategy parameters from the JSON config file."""
        resolved = path or (_CONFIG_DIR / "params.json")
        raw = _load_json(resolved)
        # Nest the flat weights dict into the SignalWeights model
        weights_raw = raw.pop("weights", {})
        fusion_raw = raw.pop("sentiment_fusion_weights", {})
        return cls(
            weights=SignalWeights(**weights_raw),
            sentiment_fusion_weights=SentimentFusionWeights(**fusion_raw),
            **raw,
        )


# ── Exchange Settings ─────────────────────────────────────────────────────────


class ExchangeSettings(BaseSettings):
    """Exchange API credentials, loaded from environment variables.

    All values are read from environment variables with the prefix BINANCE_
    and ALPHA_VANTAGE_. Sensitive values are stored as SecretStr to prevent
    accidental logging.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    binance_api_key: str = Field(
        default="",
        alias="BINANCE_API_KEY",
        description="Binance API key",
    )
    binance_api_secret: str = Field(
        default="",
        alias="BINANCE_API_SECRET",
        description="Binance API secret",
    )
    alpha_vantage_api_key: str = Field(
        default="",
        alias="ALPHA_VANTAGE_API_KEY",
        description="Alpha Vantage API key for equity data",
    )

    @property
    def binance_configured(self) -> bool:
        """Return True if Binance credentials are set."""
        return bool(self.binance_api_key and self.binance_api_secret)

    @property
    def alpha_vantage_configured(self) -> bool:
        """Return True if Alpha Vantage API key is set."""
        return bool(self.alpha_vantage_api_key)


# ── Database Settings ─────────────────────────────────────────────────────────


class DatabaseSettings(BaseSettings):
    """Database connection settings, loaded from environment variables.

    Defaults to SQLite for local development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    database_url: str = Field(
        default="sqlite:///data/trading.db",
        alias="DATABASE_URL",
        description="SQLAlchemy database URL",
    )
    db_pool_size: int = Field(
        default=5,
        alias="DB_POOL_SIZE",
        description="SQLAlchemy connection pool size",
    )
    db_max_overflow: int = Field(
        default=10,
        alias="DB_MAX_OVERFLOW",
        description="SQLAlchemy max overflow connections",
    )
    db_pool_timeout: int = Field(
        default=30,
        alias="DB_POOL_TIMEOUT",
        description="SQLAlchemy pool connection timeout (seconds)",
    )

    @property
    def is_sqlite(self) -> bool:
        """Return True if using SQLite (e.g., local development)."""
        return self.database_url.startswith("sqlite")

    @property
    def is_postgres(self) -> bool:
        """Return True if using PostgreSQL / TimescaleDB."""
        return "postgresql" in self.database_url or "postgres" in self.database_url


# ── Reddit Settings ───────────────────────────────────────────────────────────


class RedditSettings(BaseSettings):
    """Reddit API credentials for social sentiment data."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    reddit_client_id: str = Field(
        default="",
        alias="REDDIT_CLIENT_ID",
        description="Reddit OAuth2 client ID",
    )
    reddit_client_secret: str = Field(
        default="",
        alias="REDDIT_CLIENT_SECRET",
        description="Reddit OAuth2 client secret",
    )
    reddit_user_agent: str = Field(
        default="QTS/1.0",
        alias="REDDIT_USER_AGENT",
        description="Reddit API user agent string",
    )

    @property
    def configured(self) -> bool:
        """Return True if Reddit credentials are set."""
        return bool(self.reddit_client_id and self.reddit_client_secret)


# ── LLM Settings ──────────────────────────────────────────────────────────────


class AlpacaSettings(BaseSettings):
    """Alpaca Markets API credentials for US equities data."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    alpaca_api_key: str = Field(
        default="",
        alias="ALPACA_API_KEY",
        description="Alpaca API key ID",
    )
    alpaca_api_secret: str = Field(
        default="",
        alias="ALPACA_API_SECRET",
        description="Alpaca API secret key",
    )
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        alias="ALPACA_BASE_URL",
        description="Alpaca trading API base URL",
    )
    alpaca_data_url: str = Field(
        default="https://data.alpaca.markets",
        alias="ALPACA_DATA_URL",
        description="Alpaca market data API base URL",
    )

    @property
    def configured(self) -> bool:
        """Return True if Alpaca API credentials are set."""
        return bool(self.alpaca_api_key and self.alpaca_api_secret)


class LLMSettings(BaseSettings):
    """LLM provider settings for AI-assisted analysis."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ── Backend selection ──────────────────────────────────────────────────────

    backend: str = Field(
        default="ollama",
        alias="LLM_BACKEND",
        description="LLM backend to use: 'ollama' (local) or 'anthropic' (Claude API)",
    )

    # ── Anthropic / Claude settings ────────────────────────────────────────────

    anthropic_api_key: str = Field(
        default="",
        alias="ANTHROPIC_API_KEY",
        description="Anthropic Claude API key",
    )
    anthropic_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        alias="ANTHROPIC_MODEL",
        description="Anthropic model ID to use",
    )

    # ── Ollama settings ────────────────────────────────────────────────────────

    ollama_model: str = Field(
        default="glm-5:cloud",
        alias="OLLAMA_MODEL",
        description="Ollama model tag to use (e.g. 'glm-5:cloud')",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        alias="OLLAMA_BASE_URL",
        description="Base URL for the local Ollama API server",
    )

    # ── Shared settings ────────────────────────────────────────────────────────

    llm_max_tokens: int = Field(
        default=4096,
        alias="LLM_MAX_TOKENS",
        description="Maximum tokens in LLM response",
    )
    llm_temperature: float = Field(
        default=0.0,
        alias="LLM_TEMPERATURE",
        ge=0.0,
        le=2.0,
        description="LLM sampling temperature (0=deterministic)",
    )

    @property
    def configured(self) -> bool:
        """Return True if the active backend has the required credentials/config.

        - ``"anthropic"``: requires ``ANTHROPIC_API_KEY`` to be set.
        - ``"ollama"``: always considered configured (no API key needed).
        """
        if self.backend == "anthropic":
            return bool(self.anthropic_api_key)
        # Ollama requires no credentials
        return True


# ── App Settings (Master) ─────────────────────────────────────────────────────


class AppSettings(BaseSettings):
    """Master application settings composing all sub-settings.

    Runtime environment variables take precedence over .env file values.
    JSON config files are loaded at instantiation time.

    Usage::

        from qts.config import get_settings
        settings = get_settings()
        print(settings.risk.max_daily_drawdown_pct)
        print(settings.strategy.weights.w_sentiment)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Application metadata
    qts_env: str = Field(
        default="development",
        alias="QTS_ENV",
        description="Runtime environment: development | staging | production",
    )
    qts_log_level: str = Field(
        default="INFO",
        alias="QTS_LOG_LEVEL",
        description="Root log level: DEBUG | INFO | WARNING | ERROR | CRITICAL",
    )
    qts_dry_run: bool = Field(
        default=True,
        alias="QTS_DRY_RUN",
        description="If True, no real orders are submitted (paper trading mode)",
    )

    # Redis / task queue
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        alias="REDIS_URL",
        description="Redis connection URL for Celery broker and result backend",
    )

    # Composed sub-settings (loaded lazily via properties)
    _risk: RiskLimits | None = None
    _strategy: StrategyParams | None = None
    _exchange: ExchangeSettings | None = None
    _database: DatabaseSettings | None = None
    _reddit: RedditSettings | None = None
    _llm: LLMSettings | None = None
    _alpaca: AlpacaSettings | None = None

    @property
    def risk(self) -> RiskLimits:
        """Immutable risk limits (loaded from config/risk_limits.json)."""
        if self._risk is None:
            self._risk = RiskLimits.from_file()
        return self._risk

    @property
    def strategy(self) -> StrategyParams:
        """Strategy parameters (loaded from config/params.json)."""
        if self._strategy is None:
            self._strategy = StrategyParams.from_file()
        return self._strategy

    @property
    def exchange(self) -> ExchangeSettings:
        """Exchange API credentials (from environment)."""
        if self._exchange is None:
            self._exchange = ExchangeSettings()
        return self._exchange

    @property
    def database(self) -> DatabaseSettings:
        """Database connection settings (from environment)."""
        if self._database is None:
            self._database = DatabaseSettings()
        return self._database

    @property
    def reddit(self) -> RedditSettings:
        """Reddit API credentials (from environment)."""
        if self._reddit is None:
            self._reddit = RedditSettings()
        return self._reddit

    @property
    def llm(self) -> LLMSettings:
        """LLM provider settings (from environment)."""
        if self._llm is None:
            self._llm = LLMSettings()
        return self._llm

    @property
    def alpaca(self) -> AlpacaSettings:
        """Alpaca Markets API credentials (from environment)."""
        if self._alpaca is None:
            self._alpaca = AlpacaSettings()
        return self._alpaca

    @property
    def is_production(self) -> bool:
        """Return True if running in a production environment."""
        return self.qts_env.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Return True if running in a development environment."""
        return self.qts_env.lower() == "development"

    def validate_production_readiness(self) -> list[str]:
        """Return a list of configuration issues that should block production deployment."""
        issues: list[str] = []

        if self.qts_dry_run and self.is_production:
            issues.append("QTS_DRY_RUN=true in production environment")

        if not self.exchange.binance_configured:
            issues.append("Binance API credentials not configured (BINANCE_API_KEY/SECRET)")

        if not self.llm.configured:
            issues.append(
                "LLM backend not fully configured "
                "(set ANTHROPIC_API_KEY for 'anthropic' backend, or use LLM_BACKEND=ollama)"
            )

        if not self.alpaca.configured:
            issues.append(
                "Alpaca API credentials not configured (ALPACA_API_KEY/ALPACA_API_SECRET)"
            )

        if self.database.is_sqlite and self.is_production:
            issues.append(
                "SQLite database in production; use PostgreSQL/TimescaleDB (DATABASE_URL)"
            )

        return issues


# ── Singleton accessor ────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the singleton AppSettings instance.

    Uses lru_cache to ensure settings are loaded once per process.
    In tests, call get_settings.cache_clear() to reset between test cases.
    """
    settings = AppSettings()
    logger.debug(
        "QTS settings loaded: env=%s, dry_run=%s, db=%s",
        settings.qts_env,
        settings.qts_dry_run,
        settings.database.database_url,
    )
    return settings
