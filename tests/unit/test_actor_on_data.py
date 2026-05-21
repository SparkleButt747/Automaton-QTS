"""Tests for QTSStrategy.on_data — interleaved news custom data → inner.on_text."""

from __future__ import annotations


class _RecordingStrategy:
    params = None
    name = "recording"

    def __init__(self) -> None:
        self.seen: list[object] = []

    def on_bar(self, *_a: object, **_k: object) -> list:
        return []

    def on_fill(self, *_a: object, **_k: object) -> None:
        pass

    def on_text(self, event: object) -> None:
        self.seen.append(event)


def _actor():
    from qts.nautilus.actor import QTSStrategy, QTSStrategyConfig

    return QTSStrategy(config=QTSStrategyConfig(instrument_id="BTCUSDT.BINANCE", bar_window=50))


def test_on_data_forwards_news_to_inner_on_text():  # T-V21-ACTOR-1
    from qts.nautilus.news_data import NewsDataPoint

    actor = _actor()
    inner = _RecordingStrategy()
    actor.set_qts_strategy(inner)
    actor.on_data(NewsDataPoint(source="fomc", persona="", text="dovish", ts_event=1, ts_init=1))

    assert len(inner.seen) == 1
    assert inner.seen[0].source == "fomc"
    assert inner.seen[0].text == "dovish"


def test_on_data_safe_before_strategy_set():  # T-V21-ACTOR-2
    from qts.nautilus.news_data import NewsDataPoint

    _actor().on_data(NewsDataPoint(source="x", persona="", text="y", ts_event=1, ts_init=1))


def test_on_data_swallows_classifier_keyerror():  # T-V21-ACTOR-3
    from qts.nautilus.news_data import NewsDataPoint

    class _Raises(_RecordingStrategy):
        def on_text(self, event: object) -> None:
            raise KeyError("not in cache")

    actor = _actor()
    actor.set_qts_strategy(_Raises())
    # Must not propagate — classifier cache miss is logged and swallowed.
    actor.on_data(NewsDataPoint(source="x", persona="", text="y", ts_event=1, ts_init=1))


def test_on_data_ignores_non_news_data():  # T-V21-ACTOR-4
    actor = _actor()
    inner = _RecordingStrategy()
    actor.set_qts_strategy(inner)
    actor.on_data(object())  # not a NewsDataPoint
    assert inner.seen == []
