"""Integration smoke tests for qts.nautilus.runner.

Covers the end-to-end terrain backtest path after the trade-bridge fix:
- T-NRUN-1: empty terrain returns an empty BacktestResult without crashing
- T-NRUN-2: a 500-bar synthetic BTCUSDT terrain produces a non-empty equity curve
  and the engine doesn't choke on Money-formatted PnL strings during extraction

These exercise the real NautilusTrader BacktestEngine. The runs are short
(~few seconds) and deterministic by seed.
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta

import pytest

from qts.config import get_settings
from qts.models.base import Bar, Catalyst, LiquidityLevel, SentimentLevel, Trend, VolLevel
from qts.models.terrain import MacroRegime
from qts.nautilus.config import BacktestResult, VenueConfig
from qts.nautilus.runner import run_terrain_backtest
from qts.strategies.momentum import MomentumStrategy
from qts.terrain.builder import TerrainBuilder


def _make_synthetic_terrain(seed: int = 42, n_bars: int = 500) -> object:
    rnd = random.Random(seed)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    bars: list[Bar] = []
    price = 30000.0
    for i in range(n_bars):
        ret = 0.0008 + rnd.gauss(0, 0.008)
        new_price = price * math.exp(ret)
        bars.append(
            Bar(
                symbol="BTCUSDT",
                timestamp=start + timedelta(hours=i),
                open=price,
                high=max(price, new_price) * 1.002,
                low=min(price, new_price) * 0.998,
                close=new_price,
                volume=rnd.uniform(800, 1500),
            )
        )
        price = new_price

    regime = MacroRegime(
        trend=Trend.BULL,
        volatility=VolLevel.LOW,
        liquidity=LiquidityLevel.ABUNDANT,
        sentiment=SentimentLevel.EUPHORIC,
        catalyst=Catalyst.NONE,
        expected_drift=0.0008,
        expected_vol=0.008,
        correlation_regime=0.4,
        scenario_description="synthetic bull",
    )
    return (
        TerrainBuilder()
        .set_identity("synthetic_bull", "BTCUSDT")
        .set_regime(regime)
        .set_bars(bars)
        .compute_features()
        .compute_support_resistance()
        .build()
    )


class TestRunTerrainBacktest:
    def test_empty_terrain_returns_empty_result(self) -> None:  # T-NRUN-1
        regime = MacroRegime(
            trend=Trend.BULL,
            volatility=VolLevel.LOW,
            liquidity=LiquidityLevel.ABUNDANT,
            sentiment=SentimentLevel.NEUTRAL,
            catalyst=Catalyst.NONE,
            expected_drift=0.0,
            expected_vol=0.01,
            correlation_regime=0.4,
            scenario_description="empty",
        )
        terrain = (
            TerrainBuilder()
            .set_identity("empty", "BTCUSDT")
            .set_regime(regime)
            .set_bars([])
            .build()
        )

        settings = get_settings()
        strategy = MomentumStrategy(params=settings.strategy, risk_limits=settings.risk)
        result = run_terrain_backtest(terrain, strategy, log_level="ERROR")

        assert isinstance(result, BacktestResult)
        assert result.total_trades == 0
        assert result.equity_curve == []

    @pytest.mark.slow
    def test_synthetic_bull_produces_equity_curve(self) -> None:  # T-NRUN-2
        """End-to-end run on a synthetic BTCUSDT terrain. The post-fix bridge
        should populate an equity curve and not raise during extraction."""
        terrain = _make_synthetic_terrain(seed=42, n_bars=500)

        settings = get_settings()
        strategy = MomentumStrategy(params=settings.strategy, risk_limits=settings.risk)
        result = run_terrain_backtest(
            terrain,
            strategy,
            venue_config=VenueConfig(),
            log_level="ERROR",
        )

        assert isinstance(result, BacktestResult)
        # The equity curve must be populated — earlier this was empty because
        # PnL parsing failed on Money-formatted strings.
        assert len(result.equity_curve) > 0
        # Some trades must have been executed (the bridge previously denied
        # all orders due to precision and account-currency bugs).
        assert result.total_trades > 0
        # Equity curve must be tracked in a single currency (USDT here),
        # not interleaved BTC + USDT rows from the multi-currency account.
        assert all(v > 1.0 for v in result.equity_curve), (
            "equity curve contains sub-1 values, suggesting BTC rows leaked "
            f"into the USDT series: {result.equity_curve[:10]}"
        )
        # Total return must be sane (well within +/-100%) because the
        # engine should not halt on a clean bull-regime run any more.
        assert -1.0 < result.total_return < 1.0


class TestGetInstrument:
    def test_btcusdt_returns_currency_pair(self) -> None:  # T-NRUN-3
        from qts.nautilus.runner import _get_instrument

        inst = _get_instrument("BTCUSDT", "BINANCE")
        assert type(inst).__name__ == "CurrencyPair"
        assert str(inst.quote_currency) == "USDT"

    def test_unknown_symbol_raises(self) -> None:  # T-NRUN-4
        from qts.nautilus.runner import UnsupportedSymbolError, _get_instrument

        with pytest.raises(UnsupportedSymbolError, match="No built-in instrument"):
            _get_instrument("AAPL", "NASDAQ")

    def test_fx_pair_supported(self) -> None:  # T-NRUN-5
        from qts.nautilus.runner import _get_instrument

        inst = _get_instrument("EURUSD", "SIM")
        # default_fx_ccy returns a CurrencyPair-like object
        assert inst is not None
