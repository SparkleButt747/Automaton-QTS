"""Standardised unexpected earnings (SUE) — the do()-intervention merit (spec §2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EarningsRow:
    date: date
    estimate: float
    reported: float

    @property
    def surprise(self) -> float:
        return self.reported - self.estimate


@dataclass(frozen=True)
class EarningsEvent:
    ticker: str | None
    date: date
    sue: float  # standardised unexpected earnings (the merit signal)


def normalize_yf_earnings(df: pd.DataFrame) -> list[EarningsRow]:
    """yfinance ``Ticker.get_earnings_dates()`` DataFrame -> chronological EarningsRow list.

    Drops rows missing estimate or reported (future/unreported quarters).
    """
    rows: list[EarningsRow] = []
    for idx, row in df.sort_index().iterrows():
        est, rep = row.get("EPS Estimate"), row.get("Reported EPS")
        if pd.isna(est) or pd.isna(rep):
            continue
        d = idx.date() if isinstance(idx, (pd.Timestamp, datetime)) else idx
        rows.append(EarningsRow(date=d, estimate=float(est), reported=float(rep)))
    return rows


def compute_sue(
    rows: list[EarningsRow], *, ticker: str | None = None, min_history: int = 4
) -> list[EarningsEvent]:
    """SUE_i = surprise_i / std(prior surprises). Needs >= ``min_history`` prior events.

    Point-in-time: the denominator uses only surprises STRICTLY before event i.
    """
    ordered = sorted(rows, key=lambda r: r.date)
    surprises = [r.surprise for r in ordered]
    events: list[EarningsEvent] = []
    for i, r in enumerate(ordered):
        if i < min_history:
            continue
        denom = float(np.std(surprises[:i]))
        if denom == 0.0:
            continue
        events.append(EarningsEvent(ticker=ticker, date=r.date, sue=r.surprise / denom))
    return events


def fetch_earnings_yf(ticker: str, *, limit: int = 40) -> list[EarningsRow]:  # pragma: no cover
    """Thin yfinance wrapper (network; not unit-tested — logic lives in normalize_yf_earnings)."""
    import yfinance as yf

    df = yf.Ticker(ticker).get_earnings_dates(limit=limit)
    return normalize_yf_earnings(df) if df is not None else []
