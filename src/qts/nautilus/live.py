"""Live trading via NautilusTrader TradingNode.

Same QTSStrategy actor, same signal pipeline, same risk logic —
zero strategy code changes from backtest. Only the venue changes.

This is Phase 7: the backtest-to-live path that NautilusTrader makes trivial.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveVenueConfig:
    """Venue configuration for live trading."""

    venue_name: str = "BINANCE"
    account_type: str = "MARGIN"
    base_currency: str = "USD"

    # API credentials (loaded from environment)
    api_key: str = ""
    api_secret: str = ""
    is_testnet: bool = True

    @property
    def base_url(self) -> str:
        if self.venue_name == "BINANCE":
            if self.is_testnet:
                return "https://testnet.binance.vision"
            return "https://api.binance.com"
        return ""


def run_live_strategy(
    strategy: Any,
    venue_config: LiveVenueConfig,
    symbols: list[str],
    log_level: str = "INFO",
) -> None:
    """Run a strategy live via NautilusTrader TradingNode.

    The same QTSStrategy actor used in backtesting runs here —
    only the venue configuration changes.

    Args:
        strategy: QTS Strategy protocol implementation.
        venue_config: Live venue configuration with API credentials.
        symbols: List of symbols to trade.
        log_level: Logging level for Nautilus.
    """
    from nautilus_trader.config import (  # noqa: PLC0415
        LiveExecEngineConfig,
        LoggingConfig,
        TradingNodeConfig,
    )
    from nautilus_trader.live.node import TradingNode  # noqa: PLC0415

    from qts.nautilus.actor import QTSStrategy, QTSStrategyConfig  # noqa: PLC0415

    logger.info(
        "Starting live trading: venue=%s testnet=%s symbols=%s",
        venue_config.venue_name,
        venue_config.is_testnet,
        symbols,
    )

    # Build node config
    node_config = TradingNodeConfig(
        logging=LoggingConfig(log_level=log_level),
        exec_engine=LiveExecEngineConfig(
            reconciliation=True,
        ),
    )

    node = TradingNode(config=node_config)

    # Add venue adapter based on venue type
    if venue_config.venue_name == "BINANCE":
        _add_binance_venue(node, venue_config)
    else:
        raise ValueError(f"Unsupported live venue: {venue_config.venue_name}")

    # Add strategy actors for each symbol
    for symbol in symbols:
        instrument_id = f"{symbol}.{venue_config.venue_name}"
        actor_config = QTSStrategyConfig(
            instrument_id=instrument_id,
            bar_window=50,
        )
        actor = QTSStrategy(config=actor_config)
        actor.set_qts_strategy(strategy)
        node.trader.add_strategy(actor)

    # Build and run
    node.build()

    try:
        node.run()
    except KeyboardInterrupt:
        logger.info("Live trading stopped by user")
    finally:
        node.dispose()


def _add_binance_venue(node: Any, config: LiveVenueConfig) -> None:
    """Add Binance venue adapter to the trading node."""
    try:
        from nautilus_trader.adapters.binance.config import (  # noqa: PLC0415
            BinanceDataClientConfig,
            BinanceExecClientConfig,
        )
        from nautilus_trader.adapters.binance.factories import (  # noqa: PLC0415
            BinanceLiveDataClientFactory,
            BinanceLiveExecClientFactory,
        )

        # Live client configs — pending wiring: these belong in the node's
        # TradingNodeConfig(data_clients=..., exec_clients=...) at build time;
        # add_*_client_factory below only registers the factory class.
        data_config = BinanceDataClientConfig(  # noqa: F841
            api_key=config.api_key,
            api_secret=config.api_secret,
            testnet=config.is_testnet,
        )
        exec_config = BinanceExecClientConfig(  # noqa: F841
            api_key=config.api_key,
            api_secret=config.api_secret,
            testnet=config.is_testnet,
        )

        node.add_data_client_factory(config.venue_name, BinanceLiveDataClientFactory)
        node.add_exec_client_factory(config.venue_name, BinanceLiveExecClientFactory)

    except ImportError:
        logger.error("Binance adapter not available. Install nautilus_trader with Binance support.")
        raise
