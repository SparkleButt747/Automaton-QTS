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
    instrument: object | None = None,
    custom_data: list | None = None,
) -> BacktestResult:
    """Run a strategy against a MarketTerrain via NautilusTrader BacktestNode.

    This is the equivalent of simulate_episode() in the racing lab:
    the terrain provides the data, Nautilus provides the physics engine.

    Args:
        terrain: MarketTerrain containing bars and regime annotations.
        strategy: QTS Strategy protocol implementation.
        venue_config: Venue configuration (defaults to standard BINANCE sim).
        log_level: Nautilus logging level.
        instrument: Optional pre-built Nautilus instrument. If None, the runner
            falls back to a small built-in registry of common crypto/FX
            instruments keyed by terrain.symbol; unknown symbols raise.

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
    from nautilus_trader.model.identifiers import Venue  # noqa: PLC0415
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

    # 2. Resolve instrument first — its asset class drives the venue setup.
    # Honour a caller-supplied instrument, otherwise look it up by symbol.
    if instrument is None:
        instrument = _get_instrument(terrain.symbol, vc.name)
    instrument_id = instrument.id

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

    if custom_data:
        from nautilus_trader.model.identifiers import ClientId  # noqa: PLC0415

        # Custom (non-builtin) data must be wrapped in CustomData and added with a
        # client_id; the engine interleaves it with bars by ts_init.
        engine.add_data(custom_data, client_id=ClientId("QTS_NEWS"))

    # 5. Create and configure the QTSStrategy actor
    actor_config = QTSStrategyConfig(
        instrument_id=str(instrument_id),
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


_CRYPTO_INSTRUMENTS: dict[str, str] = {
    "BTCUSDT": "btcusdt_binance",
    "ETHUSDT": "ethusdt_binance",
    "ADAUSDT": "adausdt_binance",
}


class UnsupportedSymbolError(ValueError):
    """Raised when no built-in instrument is available for a symbol."""


def _get_instrument(symbol: str, venue_name: str) -> object:
    """Return a Nautilus instrument for a known symbol.

    Resolves a small built-in registry of common crypto pairs and 6-letter
    FX tickers. Unknown symbols raise UnsupportedSymbolError — callers
    can pass a custom instrument to run_terrain_backtest(instrument=...).
    """
    from nautilus_trader.test_kit.providers import (  # noqa: PLC0415
        TestInstrumentProvider,
    )

    sym_upper = symbol.upper()
    factory_name = _CRYPTO_INSTRUMENTS.get(sym_upper)
    if factory_name is not None:
        return getattr(TestInstrumentProvider, factory_name)()

    # 6-letter FX pair (e.g. EURUSD, GBPJPY)
    if len(sym_upper) == 6 and sym_upper.isalpha():
        try:
            return TestInstrumentProvider.default_fx_ccy(sym_upper)
        except (ValueError, KeyError) as exc:
            raise UnsupportedSymbolError(
                f"FX pair '{sym_upper}' is not recognised by Nautilus: {exc}"
            ) from exc

    raise UnsupportedSymbolError(
        f"No built-in instrument for symbol '{symbol}'. Known crypto: "
        f"{sorted(_CRYPTO_INSTRUMENTS)}; pass instrument=<Instrument> to "
        f"run_terrain_backtest() for anything else."
    )


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

        # Equity curve from account reports. Multi-currency CASH accounts
        # produce a row per currency per snapshot — pick the dominant one
        # (the currency holding the largest starting balance) so the curve
        # tracks a single, sensible "equity in quote" series.
        account_report = engine.trader.generate_account_report(venue)
        if account_report is not None and len(account_report) > 0:
            if "currency" in account_report.columns and "total" in account_report.columns:
                starting = account_report.groupby("currency")["total"].first()
                starting_floats = {c: _to_float(v) for c, v in starting.items()}
                dominant_currency = max(starting_floats, key=lambda c: starting_floats[c])
                series = account_report[account_report["currency"] == dominant_currency]["total"]
                result.equity_curve = [_to_float(v) for v in series.values]
            else:
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

        # Compute Sharpe, Sortino, drawdown from equity curve.
        # Drop zero/negative balances first — they appear when the engine
        # halts (e.g. AccountBalanceNegative) and would otherwise produce
        # divide-by-zero warnings and a misleading -100% total_return.
        if len(result.equity_curve) > 1:
            equity_all = np.array(result.equity_curve, dtype=np.float64)
            equity = equity_all[equity_all > 0]

            if len(equity) > 1:
                with np.errstate(divide="ignore", invalid="ignore"):
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
