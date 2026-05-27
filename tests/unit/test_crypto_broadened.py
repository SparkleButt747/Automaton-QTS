"""T-BTEST-*: broadened-backtest event frequency + GO/NO-GO verdict."""

from __future__ import annotations

from datetime import UTC, datetime

from qts.propagation.crypto.broadened import broadened_verdict, events_per_year
from qts.propagation.crypto.events import ContagionEvent
from qts.propagation.crypto.gate import BacktestResult, CryptoGateReport, EventStudyReport


def _events(k: int) -> list[ContagionEvent]:
    ts = datetime(2022, 1, 1, tzinfo=UTC)
    return [
        ContagionEvent(source_token="ALT", timestamp=ts, event_type="drop", usd_severity=0.2)
    ] * k


def test_events_per_year_annualises_over_span() -> None:  # T-BTEST-1
    start = datetime(2021, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 1, tzinfo=UTC)  # ~3 years
    assert round(events_per_year(_events(36), start, end)) == 12
    assert events_per_year([], start, end) == 0.0


def _es(sig: bool) -> EventStudyReport:
    return EventStudyReport(
        n_linked=10,
        n_unlinked=10,
        mean_linked_car=-0.05,
        mean_unlinked_car=0.0,
        mann_whitney_p=0.01 if sig else 0.5,
        significant=sig,
    )


def _gate(beats: bool) -> CryptoGateReport:
    return CryptoGateReport(
        n_linked_obs=10,
        graph_mse=0.1,
        pairwise_mse=0.2,
        graph_hit=0.6,
        pairwise_hit=0.5,
        beats_pairwise=beats,
    )


def _bt(sharpe: float) -> BacktestResult:
    return BacktestResult(
        n_trades=20,
        market_neutral_mean=0.01,
        market_neutral_sharpe=sharpe,
        outright_mean=0.0,
        outright_sharpe=0.0,
    )


def test_verdict_requires_all_three_conditions() -> None:  # T-BTEST-2
    assert broadened_verdict(_es(True), _gate(True), _bt(1.5)) is True
    assert broadened_verdict(_es(False), _gate(True), _bt(1.5)) is False
    assert broadened_verdict(_es(True), _gate(False), _bt(1.5)) is False
    assert broadened_verdict(_es(True), _gate(True), _bt(0.5)) is False
