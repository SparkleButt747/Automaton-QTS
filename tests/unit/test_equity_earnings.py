"""T-PATHA-SUE-*: standardised unexpected earnings (SUE) merit."""

from __future__ import annotations

from datetime import date

import pandas as pd

from qts.propagation.equity.earnings import (
    EarningsEvent,
    EarningsRow,
    compute_sue,
    normalize_yf_earnings,
)


def test_compute_sue_standardises_by_surprise_std() -> None:  # T-PATHA-SUE-1
    rows = [
        EarningsRow(date=date(2020, 1, 1), estimate=10.0, reported=11.0),
        EarningsRow(date=date(2020, 4, 1), estimate=10.0, reported=9.0),
        EarningsRow(date=date(2020, 7, 1), estimate=10.0, reported=11.0),
        EarningsRow(date=date(2020, 10, 1), estimate=10.0, reported=9.0),
        EarningsRow(date=date(2021, 1, 1), estimate=10.0, reported=12.0),
    ]
    events = compute_sue(rows, min_history=4)
    assert len(events) == 1
    e = events[0]
    assert isinstance(e, EarningsEvent)
    assert e.date == date(2021, 1, 1)
    assert e.sue == 2.0


def test_compute_sue_skips_when_insufficient_history() -> None:  # T-PATHA-SUE-2
    rows = [EarningsRow(date=date(2020, 1, 1), estimate=1.0, reported=2.0)]
    assert compute_sue(rows, min_history=4) == []


def test_normalize_yf_earnings_from_dataframe() -> None:  # T-PATHA-SUE-3
    df = pd.DataFrame(
        {"EPS Estimate": [1.0, 1.2], "Reported EPS": [1.1, 1.0]},
        index=pd.to_datetime(["2021-02-01", "2021-05-01"]),
    )
    rows = normalize_yf_earnings(df)
    assert [r.date for r in rows] == [date(2021, 2, 1), date(2021, 5, 1)]
    assert rows[0].estimate == 1.0 and rows[0].reported == 1.1
