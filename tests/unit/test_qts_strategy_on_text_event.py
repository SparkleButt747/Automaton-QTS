"""QTSStrategy.on_text_event now calls inner_strategy.on_text directly.

After Task 1 added on_text to the Strategy protocol, the actor no longer
needs hasattr / callable guards. This test pins the new direct-call shape.
"""

from __future__ import annotations

from datetime import UTC, datetime

from qts.world.events import TextEvent


def test_actor_forwards_text_to_inner_on_text() -> None:  # T-ACT-1
    from qts.nautilus.actor import QTSStrategy, QTSStrategyConfig

    captured: list[TextEvent] = []

    class _RecordingStrategy:
        params = None
        name = "recording"

        def on_bar(self, *_a: object, **_k: object) -> list:
            return []

        def on_fill(self, *_a: object, **_k: object) -> None:
            pass

        def on_text(self, event: TextEvent) -> None:
            captured.append(event)

    actor = QTSStrategy(config=QTSStrategyConfig(instrument_id="BTCUSDT.BINANCE", bar_window=10))
    actor.set_qts_strategy(_RecordingStrategy())

    evt = TextEvent(
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        source="powell",
        persona="powell",
        text="hawkish",
        metadata={},
    )
    actor.on_text_event(evt)

    assert captured == [evt]


def test_actor_noop_when_inner_strategy_unset() -> None:  # T-ACT-2
    from qts.nautilus.actor import QTSStrategy, QTSStrategyConfig

    actor = QTSStrategy(config=QTSStrategyConfig(instrument_id="BTCUSDT.BINANCE", bar_window=10))
    # No set_qts_strategy() call — _inner_strategy is None.
    evt = TextEvent(
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        source="x",
        persona=None,
        text="y",
        metadata={},
    )
    # Should not raise.
    actor.on_text_event(evt)
