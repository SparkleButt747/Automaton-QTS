"""NautilusTrader configuration models for QTS integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VenueConfig:
    """Venue configuration for backtest or live trading."""

    name: str = "BINANCE"
    oms_type: str = "NETTING"
    account_type: str = "MARGIN"
    base_currency: str = "USD"
    starting_balance: float = 100_000.0

    # Fill model configuration
    prob_fill_on_limit: float = 0.8
    prob_fill_on_stop: float = 0.9
    prob_slippage: float = 0.5
    fill_model_seed: int = 42


@dataclass(frozen=True)
class BacktestRunConfig:
    """Complete configuration for a terrain backtest run.

    Analogous to FairnessConfig in the racing lab — defines the
    experiment contract for reproducible evaluation.
    """

    strategy_config: QTSStrategyConfig
    venue_config: VenueConfig = field(default_factory=VenueConfig)
    log_level: str = "WARNING"

    # Evaluation settings
    bars_per_year: int = 252  # for annualised metrics


@dataclass
class BacktestResult:
    """Results from a NautilusTrader backtest run.

    Mirrors the old BacktestResult but populated from Nautilus reports.
    """

    equity_curve: list[float] = field(default_factory=list)
    trades: list[Any] = field(default_factory=list)
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    total_return: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0
