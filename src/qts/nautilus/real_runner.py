"""run_real_backtest — terrain backtest with news as interleaved custom data.

Real-data analogue of run_terrain_backtest. Each RealEpisode text event is
converted to a NewsDataPoint and added to the engine as CustomData, so Nautilus
interleaves it with the bar stream by timestamp. The QTSStrategy actor receives
each via on_data (events for ts <= T are delivered before bar T), reconstructs the
TextEvent, and forwards it to the strategy's on_text — a causal, look-ahead-free
dispatch. (v2's eager pre-dispatch is gone.)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qts.nautilus.runner import run_terrain_backtest

if TYPE_CHECKING:
    from qts.data.real_episode import RealEpisode
    from qts.nautilus.config import BacktestResult, VenueConfig
    from qts.strategies.base import Strategy

logger = logging.getLogger(__name__)


def run_real_backtest(
    episode: RealEpisode,
    strategy: Strategy,
    venue_config: VenueConfig | None = None,
    log_level: str = "WARNING",
    instrument: object | None = None,
) -> BacktestResult:
    """Run `strategy` against a RealEpisode, dispatching text events as custom data."""
    from nautilus_trader.model.data import CustomData, DataType  # noqa: PLC0415

    from qts.nautilus.converters import text_event_to_news_data  # noqa: PLC0415
    from qts.nautilus.news_data import NewsDataPoint  # noqa: PLC0415

    data_type = DataType(NewsDataPoint)
    custom_data = [
        CustomData(data_type=data_type, data=text_event_to_news_data(event))
        for event in sorted(episode.text_events, key=lambda e: e.timestamp)
    ]

    return run_terrain_backtest(
        episode.terrain,
        strategy,
        venue_config=venue_config,
        log_level=log_level,
        instrument=instrument,
        custom_data=custom_data,
    )
