"""T-SHOCK-*: idiosyncratic-drop shock detector (offline, on a constructed panel)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from qts.propagation.crypto.detect import IdiosyncraticDropDetector, ShockDetector

EST, WIN = 30, 3
N = EST + WIN + 3


def _grid(n: int) -> list[datetime]:
    base = datetime(2022, 1, 1, tzinfo=UTC)
    return [base + timedelta(hours=i) for i in range(n)]


def _series(idiosyncratic_drop: bool, market_drop: bool) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    btc_ret = rng.normal(0.0, 0.01, N)
    alt_ret = btc_ret.copy()  # beta ~1 through the estimation window
    if market_drop:
        btc_ret[EST:] = -0.10  # whole market dumps
        alt_ret = btc_ret.copy()  # alt moves WITH the market -> no idiosyncratic shock
    if idiosyncratic_drop:
        alt_ret[EST:] = -0.10  # alt drops on its own while BTC drifts
    btc = 100.0 * np.cumprod(1.0 + btc_ret)
    alt = 100.0 * np.cumprod(1.0 + alt_ret)
    return {"BTC": btc, "ALT": alt}


def test_detector_satisfies_protocol() -> None:  # T-SHOCK-0
    assert isinstance(IdiosyncraticDropDetector(), ShockDetector)


def test_fires_on_idiosyncratic_drop() -> None:  # T-SHOCK-1
    closes = _series(idiosyncratic_drop=True, market_drop=False)
    det = IdiosyncraticDropDetector(threshold=0.15, window=WIN, est_window=EST, cooldown=1)
    events = det.detect(closes, _grid(N), closes["BTC"])
    assert any(e.source_token == "ALT" for e in events)


def test_event_shape_is_well_formed() -> None:  # T-SHOCK-2
    closes = _series(idiosyncratic_drop=True, market_drop=False)
    det = IdiosyncraticDropDetector(threshold=0.15, window=WIN, est_window=EST, cooldown=1)
    ev = next(e for e in det.detect(closes, _grid(N), closes["BTC"]) if e.source_token == "ALT")
    assert ev.event_type == "drop"
    assert ev.usd_severity > 0.0
    assert ev.timestamp.tzinfo is not None


def test_does_not_fire_on_market_wide_dump() -> None:  # T-SHOCK-3
    closes = _series(idiosyncratic_drop=False, market_drop=True)
    det = IdiosyncraticDropDetector(threshold=0.15, window=WIN, est_window=EST, cooldown=1)
    events = det.detect(closes, _grid(N), closes["BTC"])
    assert not any(e.source_token == "ALT" for e in events)
