"""Tests for QTSStrategy actor's on_text forwarding to inner strategy."""

from __future__ import annotations

from datetime import UTC, datetime


class _TextAwareStrategy:
    """Inner strategy that records every text event it sees."""

    def __init__(self) -> None:
        self.params = None  # avoid pipeline params path
        self.seen: list[object] = []
        self.name = "text_aware"

    def on_bar(self, *_a: object, **_k: object) -> list[object]:
        return []

    def on_fill(self, *_a: object, **_k: object) -> None:
        pass

    def on_text(self, event: object) -> None:
        self.seen.append(event)


class _TextBlindStrategy:
    """Inner strategy with no on_text method — must NOT raise."""

    def __init__(self) -> None:
        self.params = None
        self.name = "text_blind"

    def on_bar(self, *_a: object, **_k: object) -> list[object]:
        return []

    def on_fill(self, *_a: object, **_k: object) -> None:
        pass


def test_on_text_event_forwards_when_method_exists() -> None:  # T-NACT-1
    from qts.nautilus.actor import QTSStrategy, QTSStrategyConfig

    cfg = QTSStrategyConfig(instrument_id="BTCUSDT.BINANCE", bar_window=50)
    actor = QTSStrategy(config=cfg)
    inner = _TextAwareStrategy()
    actor.set_qts_strategy(inner)

    payload = {"timestamp": datetime(2025, 1, 1, tzinfo=UTC), "source": "powell", "text": "hawkish"}
    actor.on_text_event(payload)

    assert inner.seen == [payload]


def test_on_text_event_noop_when_method_missing() -> None:  # T-NACT-2
    from qts.nautilus.actor import QTSStrategy, QTSStrategyConfig

    cfg = QTSStrategyConfig(instrument_id="BTCUSDT.BINANCE", bar_window=50)
    actor = QTSStrategy(config=cfg)
    inner = _TextBlindStrategy()
    actor.set_qts_strategy(inner)

    # Must not raise
    actor.on_text_event({"any": "thing"})


def test_on_text_event_safe_before_strategy_set() -> None:  # T-NACT-3
    from qts.nautilus.actor import QTSStrategy, QTSStrategyConfig

    cfg = QTSStrategyConfig(instrument_id="BTCUSDT.BINANCE", bar_window=50)
    actor = QTSStrategy(config=cfg)
    # _inner_strategy is None — must not raise
    actor.on_text_event({"any": "thing"})
