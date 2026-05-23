"""T-PATHA-DATA-*: assemble EventSamples from universe + features + reactions."""

from __future__ import annotations

from datetime import date

import numpy as np

from qts.propagation.equity.dataset import assemble_event_samples
from qts.propagation.equity.earnings import EarningsEvent
from qts.propagation.equity.universe import EquityUniverse


def _uni() -> EquityUniverse:
    return EquityUniverse(
        tickers=("A", "B", "C"), sectors=("S", "S", "S"), _aliases=(("a",), ("b",), ("c",))
    )


def test_assemble_aligns_named_idx_features_reactions() -> None:  # T-PATHA-DATA-1
    uni = _uni()
    feats = np.arange(9.0).reshape(3, 3)
    reactions = np.array([0.05, -0.02, 0.0])
    ev = EarningsEvent(ticker="B", date=date(2022, 3, 1), sue=1.7)
    samples = assemble_event_samples(
        uni,
        events=[ev],
        features_by_date={ev.date: feats},
        reactions_by_event={(ev.ticker, ev.date): reactions},
    )
    assert len(samples) == 1
    s = samples[0]
    assert s.named_idx == 1
    assert s.merit == 1.7
    np.testing.assert_array_equal(s.features, feats)
    np.testing.assert_array_equal(s.reactions, reactions)


def test_assemble_skips_events_missing_data() -> None:  # T-PATHA-DATA-2
    uni = _uni()
    ev = EarningsEvent(ticker="A", date=date(2022, 3, 1), sue=1.0)
    assert (
        assemble_event_samples(uni, events=[ev], features_by_date={}, reactions_by_event={}) == []
    )
