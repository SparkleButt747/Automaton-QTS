"""Assemble EventSamples (one do()-on-universe per earnings event) for the Path A gate.

Pure assembly (``assemble_event_samples``) is unit-tested with injected data;
``build_path_a_dataset`` is the thin network orchestrator (FNSPID + yfinance + Alpaca
via the Phase-1/2 wrappers) and is not unit-tested (``# pragma: no cover``) — its logic
lives in the tested assembler.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from qts.propagation.equity.earnings import EarningsEvent
from qts.propagation.equity.samples import EventSample
from qts.propagation.equity.universe import EquityUniverse


def assemble_event_samples(
    universe: EquityUniverse,
    *,
    events: list[EarningsEvent],
    features_by_date: dict[date, np.ndarray],
    reactions_by_event: dict[tuple[str, date], np.ndarray],
) -> list[EventSample]:
    """Join events to their point-in-time feature matrix + per-node reactions into EventSamples.

    Events missing either the feature matrix (by date) or the reaction vector (by ticker+date)
    are skipped (incomplete data), never raised.
    """
    out: list[EventSample] = []
    for ev in events:
        feats = features_by_date.get(ev.date)
        reactions = reactions_by_event.get((ev.ticker, ev.date)) if ev.ticker else None
        if feats is None or reactions is None:
            continue
        out.append(
            EventSample(
                named_idx=universe.index_of(ev.ticker),
                merit=ev.sue,
                event_date=ev.date,
                features=feats,
                reactions=reactions,
            )
        )
    return out


def build_path_a_dataset(universe: EquityUniverse) -> None:  # pragma: no cover
    """Network orchestrator: fetch FNSPID co-mentions, yfinance earnings, Alpaca bars.

    Produces features, abnormal-return reactions, and EventSamples.
    Wires the Phase-1/2 fetchers; see scripts/run_path_a_v2.py.
    Not unit-tested (network); the assembly logic is ``assemble_event_samples``.
    """
    raise NotImplementedError("wired in scripts/run_path_a_v2.py (end-to-end, needs live data)")
