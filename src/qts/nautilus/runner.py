"""Terrain backtest runner — the equivalent of simulate_episode() in the racing lab.

The terrain provides the data; NautilusTrader provides the physics.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from qts.models.terrain import MarketTerrain
from qts.nautilus.config import BacktestResult, VenueConfig

if TYPE_CHECKING:
    from qts.strategies.base import Strategy

logger = logging.getLogger(__name__)


def run_terrain_backtest(
    terrain: MarketTerrain,
    strategy: Strategy,
    venue_config: VenueConfig | None = None,
    log_level: str = "WARNING",
) -> BacktestResult:
    """Run a strategy against a MarketTerrain via NautilusTrader BacktestNode.

    This is the equivalent of simulate_episode() in the racing lab:
    the terrain provides the data, Nautilus provides the physics engine.

    Args:
        terrain: MarketTerrain containing bars and regime annotations.
        strategy: QTS Strategy protocol implementation.
        venue_config: Venue configuration (defaults to standard BINANCE sim).
        log_level: Nautilus logging level.

    Returns:
        BacktestResult with equity curve, trades, and performance metrics.
    """
    from nautilus_trader.backtest.engine import (  # noqa: PLC0415
        BacktestEngine,
        BacktestEngineConfig,
    )
    from nautilus_trader.backtest.models import FillModel  # noqa: PLC0415
    from nautilus_trader.config import LoggingConfig  # noqa: PLC0415
    from nautilus_trader.model.enums import AccountType, OmsType  # noqa: PLC0415
    from nautilus_trader.model.identifiers import InstrumentId, Venue  # noqa: PLC0415
    from nautilus_trader.model.instruments import CurrencyPair  # noqa: PLC0415
    from nautilus_trader.model.objects import Money  # noqa: PLC0415

    from qts.nautilus.actor import QTSStrategy, QTSStrategyConfig  # noqa: PLC0415
    from qts.nautilus.converters import qts_bar_to_nautilus  # noqa: PLC0415

    vc = venue_config or VenueConfig()

    if not terrain.bars:
        logger.warning("run_terrain_backtest: terrain '%s' has no bars", terrain.name)
        return BacktestResult()

    # 1. Configure engine
    engine_config = BacktestEngineConfig(
        logging=LoggingConfig(log_level=log_level),
    )
    engine = BacktestEngine(config=engine_config)

    # 2. Resolve instrument first — its asset class drives the venue setup
    instrument_id_str = f"{terrain.symbol}.{vc.name}"
    instrument_id = InstrumentId.from_str(instrument_id_str)
    instrument = _get_instrument(terrain.symbol, vc.name)

    # 3. Add venue with fill model. Spot CurrencyPair instruments need a
    # multi-currency CASH account (base_currency=None) seeded in the
    # instrument's quote currency, e.g. USDT for BTCUSDT.
    venue_obj = Venue(vc.name)
    fill_model = FillModel(
        prob_fill_on_limit=vc.prob_fill_on_limit,
        prob_fill_on_stop=vc.prob_fill_on_stop,
        prob_slippage=vc.prob_slippage,
        random_seed=vc.fill_model_seed,
    )

    if isinstance(instrument, CurrencyPair):
        engine.add_venue(
            venue=venue_obj,
            oms_type=OmsType[vc.oms_type],
            account_type=AccountType.CASH,
            base_currency=None,  # multi-currency
            starting_balances=[Money(vc.starting_balance, instrument.quote_currency)],
            fill_model=fill_model,
        )
    else:
        from nautilus_trader.model.currencies import USD  # noqa: PLC0415

        engine.add_venue(
            venue=venue_obj,
            oms_type=OmsType[vc.oms_type],
            account_type=AccountType[vc.account_type],
            base_currency=USD,
            starting_balances=[Money(vc.starting_balance, USD)],
            fill_model=fill_model,
        )

    # 4. Add the instrument now that its venue is configured
    engine.add_instrument(instrument)

    # 4. Convert and add bar data (match instrument precision)
    price_prec = instrument.price_precision
    size_prec = instrument.size_precision
    nautilus_bars = [
        qts_bar_to_nautilus(b, instrument_id, price_prec, size_prec) for b in terrain.bars
    ]
    engine.add_data(nautilus_bars)

    # 5. Create and configure the QTSStrategy actor
    actor_config = QTSStrategyConfig(
        instrument_id=instrument_id_str,
        bar_window=50,
    )
    actor = QTSStrategy(config=actor_config)
    actor.set_qts_strategy(strategy)
    engine.add_strategy(actor)

    # 6. Run the backtest
    logger.info(
        "run_terrain_backtest: terrain='%s' bars=%d venue=%s",
        terrain.name,
        len(terrain.bars),
        vc.name,
    )
    engine.run()

    # 7. Extract results
    result = _extract_results(engine, vc)

    engine.dispose()
    return result


def _get_instrument(symbol: str, venue_name: str) -> object:
    """Get an appropriate Nautilus instrument for the given symbol."""
    from nautilus_trader.test_kit.providers import (  # noqa: PLC0415
        TestInstrumentProvider,
    )

    # Map common symbols to pre-built instruments
    sym_upper = symbol.upper()
    if sym_upper == "BTCUSDT":
        return TestInstrumentProvider.btcusdt_binance()
    if sym_upper == "ETHUSDT":
        return TestInstrumentProvider.ethusdt_binance()
    if sym_upper == "ADAUSDT":
        return TestInstrumentProvider.adausdt_binance()

    # For equities and other symbols, create a generic FX pair instrument
    # Pad or trim to 6-7 chars as required by default_fx_ccy
    pair = f"{sym_upper[:3]}USD"
    if len(pair) < 6:
        pair = f"{sym_upper}USD"
    if len(pair) > 7:
        pair = pair[:7]
    try:
        return TestInstrumentProvider.default_fx_ccy(pair)
    except (ValueError, Exception):
        # Fallback to BTCUSDT as a generic instrument
        logger.warning("Could not create instrument for '%s', using BTCUSDT fallback", symbol)
        return TestInstrumentProvider.btcusdt_binance()


def _extract_results(engine: object, venue_config: VenueConfig) -> BacktestResult:
    """Extract BacktestResult from NautilusTrader engine reports."""
    result = BacktestResult()

    try:
        from nautilus_trader.model.identifiers import Venue  # noqa: PLC0415

        venue = Venue(venue_config.name)

        # Nautilus often returns Money-formatted strings like "100.00 USDT".
        # Strip the currency suffix to recover a pure float.
        def _to_float(v: object) -> float:
            s = str(v).split(" ", 1)[0]
            return float(s)

        # Equity curve from account reports
        account_report = engine.trader.generate_account_report(venue)
        if account_report is not None and len(account_report) > 0:
            result.equity_curve = [_to_float(v) for v in account_report["total"].values]

        # Trade statistics from position reports
        position_report = engine.trader.generate_positions_report()
        if position_report is not None and len(position_report) > 0:
            result.total_trades = len(position_report)
            pnls = position_report.get("realized_pnl", [])
            if len(pnls) > 0:
                pnl_arr = np.array([_to_float(p) for p in pnls], dtype=np.float64)
                result.total_pnl = float(np.sum(pnl_arr))
                wins = pnl_arr[pnl_arr > 0]
                losses = pnl_arr[pnl_arr < 0]
                result.win_rate = float(len(wins) / len(pnl_arr)) if len(pnl_arr) > 0 else 0.0

                gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
                gross_loss = abs(float(np.sum(losses))) if len(losses) > 0 else 0.0
                if gross_loss > 0:
                    result.profit_factor = gross_profit / gross_loss
                elif gross_profit > 0:
                    result.profit_factor = float("inf")

        # Compute Sharpe, Sortino, drawdown from equity curve
        if len(result.equity_curve) > 1:
            equity = np.array(result.equity_curve, dtype=np.float64)
            returns = np.diff(equity) / equity[:-1]
            returns = returns[np.isfinite(returns)]

            if len(returns) > 1:
                mean_ret = float(np.mean(returns))
                std_ret = float(np.std(returns, ddof=1))
                bars_per_year = 252

                if std_ret > 0:
                    result.sharpe_ratio = float(mean_ret / std_ret * np.sqrt(bars_per_year))

                downside = returns[returns < 0]
                if len(downside) > 0:
                    downside_std = float(np.std(downside, ddof=1))
                    if downside_std > 0:
                        result.sortino_ratio = float(
                            mean_ret / downside_std * np.sqrt(bars_per_year)
                        )

                peak = np.maximum.accumulate(equity)
                drawdown = (peak - equity) / peak
                result.max_drawdown = float(np.max(drawdown))

                ann_return = mean_ret * bars_per_year
                if result.max_drawdown > 0:
                    result.calmar_ratio = float(ann_return / result.max_drawdown)

            result.total_return = float((equity[-1] - equity[0]) / equity[0])

    except Exception:  # noqa: BLE001
        logger.exception("Failed to extract results from NautilusTrader engine")

    return result
