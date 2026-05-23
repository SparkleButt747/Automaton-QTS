"""T-PATHA-SAMPLE-*: the Phase-3 training contract."""

from __future__ import annotations

from datetime import date

import numpy as np

from qts.propagation.equity.samples import EventSample


def test_event_sample_shapes_align() -> None:  # T-PATHA-SAMPLE-1
    feats = np.zeros((5, 8))
    reactions = np.zeros(5)
    s = EventSample(
        named_idx=2, merit=1.3, event_date=date(2022, 1, 1), features=feats, reactions=reactions
    )
    assert s.n_nodes == 5
    assert s.features.shape == (5, 8)
    assert s.reactions.shape == (5,)


def test_event_sample_rejects_mismatched_shapes() -> None:  # T-PATHA-SAMPLE-2
    import pytest

    with pytest.raises(ValueError):
        EventSample(
            named_idx=0,
            merit=1.0,
            event_date=date(2022, 1, 1),
            features=np.zeros((5, 8)),
            reactions=np.zeros(4),
        )


def test_public_api_exported() -> None:  # T-PATHA-SAMPLE-3
    import qts.propagation.equity as eq

    for name in (
        "EquityUniverse",
        "load_universe",
        "EarningsEvent",
        "compute_sue",
        "node_feature_vector",
        "market_model_abnormal_return",
        "EventSample",
    ):
        assert hasattr(eq, name), name
