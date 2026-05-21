"""run_real_backtest — terrain backtest that also dispatches text events.

Real-data analogue of run_terrain_backtest. Uses the same Nautilus
machinery for bars + matching engine, but additionally walks the
RealEpisode's text_events list and forwards each through
QTSStrategy.on_text_event at the appropriate point in the bar stream.

For v2, text-event dispatch happens *before* the backtest runs:
we eagerly walk every event in chronological order against the
QTSStrategy actor's on_text_event method. This is correct because
on_text only updates strategy state — the strategy's bar-level
decisions happen during the Nautilus engine.run() pass, by which
time the belief state is already populated.

For events that fall *during* the bar stream (the common FOMC case),
this approximation is fine for v2: the belief state evolves correctly
relative to bar timestamps because BeliefAxis.at(bar.timestamp)
respects the event timestamp. Bars before the event see decayed state
from t=-inf (zero); bars after see the post-event belief.

(A future bar-by-bar interleaved dispatch would be more faithful for
multi-event scenarios, but is out of scope for v2.)
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
    """Run `strategy` against a RealEpisode, dispatching text events first."""
    # Pre-dispatch all text events into the strategy in chronological order.
    for event in sorted(episode.text_events, key=lambda e: e.timestamp):
        try:
            strategy.on_text(event)
        except Exception:  # noqa: BLE001
            logger.exception("strategy.on_text failed on event %r", event)

    # Then run the standard terrain backtest — the strategy's belief is now warmed up.
    return run_terrain_backtest(
        episode.terrain,
        strategy,
        venue_config=venue_config,
        log_level=log_level,
        instrument=instrument,
    )
