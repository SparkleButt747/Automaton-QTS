"""Helpers for the broadened-trigger backtest gate: detected-event frequency and the
GO/NO-GO verdict (same bar as the v0 verdict in run_crypto_contagion_v0.py)."""

from __future__ import annotations

from datetime import datetime

from qts.propagation.crypto.events import ContagionEvent
from qts.propagation.crypto.gate import BacktestResult, CryptoGateReport, EventStudyReport

HOURS_PER_YEAR = 24 * 365.25


def events_per_year(events: list[ContagionEvent], start: datetime, end: datetime) -> float:
    """Annualised detected-event frequency over the panel span — replaces the hardcoded
    12.0 default in contagion_backtest, which would otherwise misscale the Sharpe."""
    span_hours = (end - start).total_seconds() / 3600.0
    if span_hours <= 0 or not events:
        return 0.0
    return len(events) * HOURS_PER_YEAR / span_hours


def broadened_verdict(es: EventStudyReport, gate: CryptoGateReport, bt: BacktestResult) -> bool:
    """GO only if linked drawdown is significant, the operator beats the pairwise
    baseline, and the costed market-neutral Sharpe clears 1.0."""
    return es.significant and gate.beats_pairwise and bt.market_neutral_sharpe > 1.0
