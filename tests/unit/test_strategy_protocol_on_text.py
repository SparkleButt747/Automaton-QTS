"""Strategy protocol now includes on_text(event) with a no-op stub on existing strategies."""

from __future__ import annotations

from datetime import UTC, datetime

from qts.config import RiskLimits, StrategyParams
from qts.strategies.base import Strategy
from qts.world.events import TextEvent

# ── Helpers ───────────────────────────────────────────────────────────────────

_RISK_LIMITS = RiskLimits(
    max_daily_drawdown_pct=0.05,
    max_position_size_pct=0.20,
    max_open_positions=5,
    circuit_breaker_cooldown_seconds=300,
    sentiment_signal_max_scalar=3.0,
)

_PARAMS = StrategyParams(
    version="1.0.0",
    weights={
        "w_rsi": 0.20,
        "w_macd": 0.20,
        "w_bb": 0.15,
        "w_mom": 0.15,
        "w_sentiment": 0.30,
    },
    entry_threshold=0.25,
    exit_threshold=0.05,
    max_hold_bars=24,
    sentiment_fusion_weights={"news": 0.5, "social": 0.3, "geopolitical": 0.2},
)


def _event() -> TextEvent:
    return TextEvent(
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        source="test",
        persona=None,
        text="dummy",
        metadata={},
    )


def test_protocol_declares_on_text() -> None:  # T-PROT-1
    # `on_text` is declared on the Strategy protocol surface.
    assert hasattr(Strategy, "on_text")


def test_momentum_on_text_is_noop() -> None:  # T-PROT-2
    from qts.strategies.momentum import MomentumStrategy

    strat = MomentumStrategy(params=_PARAMS, risk_limits=_RISK_LIMITS)
    # Must not raise; returns None.
    assert strat.on_text(_event()) is None


def test_mean_reversion_on_text_is_noop() -> None:  # T-PROT-3
    from qts.strategies.mean_reversion import MeanReversionStrategy

    strat = MeanReversionStrategy()
    assert strat.on_text(_event()) is None


def test_sma_crossover_on_text_is_noop() -> None:  # T-PROT-4
    from qts.strategies.sma_crossover import SMACrossoverStrategy

    strat = SMACrossoverStrategy(fast_period=10, slow_period=30)
    assert strat.on_text(_event()) is None
