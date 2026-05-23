# Path A v2 — Phase 2: Event/Label Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the event/label layer for Path A v2: the equity **universe** (with the alias map that feeds Phase-1 ticker extraction), **SUE** (standardised unexpected earnings) merit from earnings data, point-in-time **node features** `ξ`, and **abnormal-return** reaction labels. Output: an `EventSample` (event → per-node features + per-node reaction) that the Phase-3 meta-trainer consumes.

**Architecture:** Four modules under the existing `src/qts/propagation/equity/` package + one config file. Pure-math cores (SUE, features, abnormal returns) take numpy arrays / normalised rows so they are unit-tested with synthetic fixtures and **never touch the network**; thin fetch wrappers (yfinance earnings, Alpaca bars) sit at the boundary and are tested only via a normalise function on a fixture DataFrame. This mirrors the Phase-1 FNSPID pattern (normalise at the boundary, test the normalised form).

**Tech Stack:** Python 3.11, numpy, pandas, `yfinance` (NEW dependency — earnings dates/estimates), the repo's `AlpacaBarAdapter` (`src/qts/data/market/alpaca_adapter.py`, async `get_historical_bars -> list[Bar]`), pydantic + `yaml.safe_load` for config (à la `src/qts/world/scenario.py::load_scenario_yaml`), pytest `--no-cov`.

**Spec:** `docs/specs/2026-05-23-path-a-v2-nhop-meta-feasibility.md` §4–§7. **Depends on:** Phase 1 (`src/qts/propagation/equity/` link graph — built, committed).

---

## File Structure (Phase 2)

- Create: `config/universe/path_a_v2.yaml` — v0 universe: per-ticker aliases + GICS sector
- Create: `src/qts/propagation/equity/universe.py` — `EquityUniverse`, `load_universe`
- Create: `src/qts/propagation/equity/earnings.py` — `EarningsRow`, `EarningsEvent`, `compute_sue`, `normalize_yf_earnings`
- Create: `src/qts/propagation/equity/features.py` — `realised_vol`, `momentum_12_1`, `market_beta`, `node_feature_vector`, `FEATURE_DIM`
- Create: `src/qts/propagation/equity/labels.py` — `market_model_abnormal_return`
- Create: `src/qts/propagation/equity/samples.py` — `EventSample` (the Phase-3 contract)
- Modify: `src/qts/propagation/equity/__init__.py` — export the new public names
- Modify: `pyproject.toml` — add `yfinance` dependency
- Test: `tests/unit/test_equity_universe.py`, `test_equity_earnings.py`, `test_equity_features.py`, `test_equity_labels.py` (T-PATHA-UNIV / SUE / FEAT / LABEL-*)

---

### Task 1: Equity universe + YAML loader

**Files:**
- Create: `config/universe/path_a_v2.yaml`
- Create: `src/qts/propagation/equity/universe.py`
- Test: `tests/unit/test_equity_universe.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_equity_universe.py
"""T-PATHA-UNIV-*: equity universe + alias map loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from qts.propagation.equity.universe import EquityUniverse, load_universe

_YAML = """\
tickers:
  NVDA:
    sector: Semiconductors
    aliases: [nvidia]
  AMD:
    sector: Semiconductors
    aliases: [advanced micro devices]
  TSM:
    sector: Semiconductors
    aliases: [tsmc, taiwan semiconductor]
"""


def _write(tmp_path: Path) -> Path:
    p = tmp_path / "u.yaml"
    p.write_text(_YAML)
    return p


def test_load_universe_tickers_and_sectors(tmp_path: Path) -> None:  # T-PATHA-UNIV-1
    u = load_universe(_write(tmp_path))
    assert isinstance(u, EquityUniverse)
    assert u.tickers == ("AMD", "NVDA", "TSM")  # sorted
    assert u.sector_of("NVDA") == "Semiconductors"


def test_alias_map_lowercased_includes_ticker_and_aliases(tmp_path: Path) -> None:  # T-PATHA-UNIV-2
    u = load_universe(_write(tmp_path))
    am = u.alias_map()
    assert am["nvidia"] == "NVDA"
    assert am["nvda"] == "NVDA"  # ticker itself is always an alias
    assert am["taiwan semiconductor"] == "TSM"
    assert all(k == k.lower() for k in am)


def test_index_of_is_stable(tmp_path: Path) -> None:  # T-PATHA-UNIV-3
    u = load_universe(_write(tmp_path))
    assert u.index_of("AMD") == 0 and u.index_of("TSM") == 2  # node index = position in sorted tickers


def test_unknown_ticker_raises(tmp_path: Path) -> None:  # T-PATHA-UNIV-4
    u = load_universe(_write(tmp_path))
    with pytest.raises(KeyError):
        u.sector_of("ZZZZ")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_universe.py --no-cov -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'qts.propagation.equity.universe'`

- [ ] **Step 3: Write the config + implementation**

Create `config/universe/path_a_v2.yaml`:

```yaml
# Path A v2 v0 universe — small S&P-500 subset across sectors with clear economic links.
# aliases feed Phase-1 ticker extraction (lowercased at load); sector feeds node features.
tickers:
  NVDA: {sector: Semiconductors, aliases: [nvidia]}
  AMD: {sector: Semiconductors, aliases: [advanced micro devices]}
  INTC: {sector: Semiconductors, aliases: [intel]}
  TSM: {sector: Semiconductors, aliases: [tsmc, taiwan semiconductor]}
  AVGO: {sector: Semiconductors, aliases: [broadcom]}
  QCOM: {sector: Semiconductors, aliases: [qualcomm]}
  AAPL: {sector: Technology, aliases: [apple]}
  MSFT: {sector: Technology, aliases: [microsoft]}
  WMT: {sector: Retail, aliases: [walmart]}
  TGT: {sector: Retail, aliases: [target]}
  COST: {sector: Retail, aliases: [costco]}
  F: {sector: Autos, aliases: [ford]}
  GM: {sector: Autos, aliases: [general motors]}
  DAL: {sector: Airlines, aliases: [delta air lines]}
  UAL: {sector: Airlines, aliases: [united airlines]}
```

Create `src/qts/propagation/equity/universe.py`:

```python
"""Equity universe: tickers, sectors, and the alias map that feeds ticker extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EquityUniverse:
    tickers: tuple[str, ...]  # sorted; position = node index
    sectors: tuple[str, ...]  # sector per ticker, aligned to ``tickers``
    _aliases: tuple[tuple[str, ...], ...]  # alias list per ticker, aligned

    def index_of(self, ticker: str) -> int:
        return self.tickers.index(ticker)

    def sector_of(self, ticker: str) -> str:
        return self.sectors[self.index_of(ticker)]

    def alias_map(self) -> dict[str, str]:
        """Lowercased alias/ticker -> ticker, for qts.propagation.equity.extract_tickers."""
        out: dict[str, str] = {}
        for tkr, aliases in zip(self.tickers, self._aliases, strict=True):
            out[tkr.lower()] = tkr
            for a in aliases:
                out[a.lower()] = tkr
        return out

    @property
    def unique_sectors(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.sectors)))


def load_universe(yaml_path: Path) -> EquityUniverse:
    raw = yaml.safe_load(Path(yaml_path).read_text())
    items = raw["tickers"]
    tickers = tuple(sorted(items))
    sectors = tuple(items[t]["sector"] for t in tickers)
    aliases = tuple(tuple(items[t].get("aliases", [])) for t in tickers)
    return EquityUniverse(tickers=tickers, sectors=sectors, _aliases=aliases)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_universe.py --no-cov -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add config/universe/path_a_v2.yaml src/qts/propagation/equity/universe.py tests/unit/test_equity_universe.py
git commit -m "feat(equity): v0 universe + alias-map loader (Path A v2 Phase 2)"
```

---

### Task 2: SUE earnings merit

**Files:**
- Create: `src/qts/propagation/equity/earnings.py`
- Test: `tests/unit/test_equity_earnings.py`
- Modify: `pyproject.toml` (add `yfinance`)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_equity_earnings.py
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
    # surprises: +1,-1,+1,-1 (std=1), then a +2 surprise -> SUE = 2.0
    rows = [
        EarningsRow(date=date(2020, 1, 1), estimate=10.0, reported=11.0),
        EarningsRow(date=date(2020, 4, 1), estimate=10.0, reported=9.0),
        EarningsRow(date=date(2020, 7, 1), estimate=10.0, reported=11.0),
        EarningsRow(date=date(2020, 10, 1), estimate=10.0, reported=9.0),
        EarningsRow(date=date(2021, 1, 1), estimate=10.0, reported=12.0),
    ]
    events = compute_sue(rows, min_history=4)
    # only the 5th event has >=4 prior surprises
    assert len(events) == 1
    e = events[0]
    assert isinstance(e, EarningsEvent)
    assert e.date == date(2021, 1, 1)
    assert e.sue == 2.0  # surprise +2 / std 1.0


def test_compute_sue_skips_when_insufficient_history() -> None:  # T-PATHA-SUE-2
    rows = [EarningsRow(date=date(2020, 1, 1), estimate=1.0, reported=2.0)]
    assert compute_sue(rows, min_history=4) == []


def test_normalize_yf_earnings_from_dataframe() -> None:  # T-PATHA-SUE-3
    # yfinance get_earnings_dates() shape: DatetimeIndex + 'EPS Estimate' / 'Reported EPS' columns
    df = pd.DataFrame(
        {"EPS Estimate": [1.0, 1.2], "Reported EPS": [1.1, 1.0]},
        index=pd.to_datetime(["2021-02-01", "2021-05-01"]),
    )
    rows = normalize_yf_earnings(df)
    assert [r.date for r in rows] == [date(2021, 2, 1), date(2021, 5, 1)]  # chronological
    assert rows[0].estimate == 1.0 and rows[0].reported == 1.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_earnings.py --no-cov -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Add the dependency + implementation**

In `pyproject.toml`, add `"yfinance>=0.2.40",` to the main `[project] dependencies` list (keep alphabetical/style consistent with neighbours). Then run `.venv/bin/pip install "yfinance>=0.2.40"`.

Create `src/qts/propagation/equity/earnings.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_earnings.py --no-cov -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/qts/propagation/equity/earnings.py tests/unit/test_equity_earnings.py
git commit -m "feat(equity): SUE merit from earnings + yfinance normaliser (Path A v2 Phase 2)"
```

---

### Task 3: Point-in-time node features

**Files:**
- Create: `src/qts/propagation/equity/features.py`
- Test: `tests/unit/test_equity_features.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_equity_features.py
"""T-PATHA-FEAT-*: point-in-time node features."""

from __future__ import annotations

import numpy as np

from qts.propagation.equity.features import (
    market_beta,
    momentum_12_1,
    node_feature_vector,
    realised_vol,
)


def test_market_beta_recovers_known_slope() -> None:  # T-PATHA-FEAT-1
    rng = np.random.default_rng(0)
    mkt = rng.normal(0, 0.01, 500)
    asset = 1.8 * mkt + rng.normal(0, 1e-4, 500)  # beta ~ 1.8
    assert abs(market_beta(asset, mkt) - 1.8) < 0.05


def test_realised_vol_positive_and_scales() -> None:  # T-PATHA-FEAT-2
    rng = np.random.default_rng(1)
    quiet = rng.normal(0, 0.005, 300)
    wild = rng.normal(0, 0.05, 300)
    assert realised_vol(wild) > realised_vol(quiet) > 0.0


def test_momentum_12_1_excludes_recent_month() -> None:  # T-PATHA-FEAT-3
    # prices: flat for 11 months then a spike in the last ~21 days -> 12-1 momentum ~ 0
    closes = np.concatenate([np.full(231, 100.0), np.linspace(100.0, 150.0, 21)])
    assert abs(momentum_12_1(closes)) < 1e-6


def test_node_feature_vector_dim_and_sector_onehot() -> None:  # T-PATHA-FEAT-4
    sectors = ("Autos", "Retail", "Semiconductors")
    rng = np.random.default_rng(2)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
    mkt = 100 * np.cumprod(1 + rng.normal(0, 0.01, 300))
    vec = node_feature_vector(
        sector="Retail", sectors=sectors, log_mktcap=25.0, closes=closes, market_closes=mkt
    )
    assert vec.shape == (len(sectors) + 4,)  # sector one-hot + [log_mktcap, beta, momentum, vol]
    # sector one-hot: Retail is index 1
    assert vec[1] == 1.0 and vec[0] == 0.0 and vec[2] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_features.py --no-cov -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write implementation**

```python
# src/qts/propagation/equity/features.py
"""Point-in-time node features xi — the basis the bilinear operator matches relations from."""

from __future__ import annotations

import numpy as np

TRADING_DAYS_MONTH = 21
TRADING_DAYS_YEAR = 252


def _returns(closes: np.ndarray) -> np.ndarray:
    closes = np.asarray(closes, dtype=float)
    return np.diff(closes) / closes[:-1]


def realised_vol(returns_or_closes: np.ndarray, *, are_returns: bool = True) -> float:
    r = np.asarray(returns_or_closes, dtype=float)
    if not are_returns:
        r = _returns(r)
    return float(np.std(r))


def momentum_12_1(closes: np.ndarray) -> float:
    """12-month minus most-recent-month return: closes[-21]/closes[-252] - 1 (skips last month)."""
    closes = np.asarray(closes, dtype=float)
    if len(closes) < TRADING_DAYS_YEAR:
        return 0.0
    return float(closes[-TRADING_DAYS_MONTH] / closes[-TRADING_DAYS_YEAR] - 1.0)


def market_beta(asset_returns: np.ndarray, market_returns: np.ndarray) -> float:
    a = np.asarray(asset_returns, dtype=float)
    m = np.asarray(market_returns, dtype=float)
    var = float(np.var(m))
    if var == 0.0:
        return 0.0
    return float(np.cov(a, m)[0, 1] / var)


def node_feature_vector(
    *,
    sector: str,
    sectors: tuple[str, ...],
    log_mktcap: float,
    closes: np.ndarray,
    market_closes: np.ndarray,
) -> np.ndarray:
    """[sector one-hot | log_mktcap, beta, momentum_12_1, realised_vol]. All from data < event t."""
    onehot = np.zeros(len(sectors), dtype=float)
    onehot[sectors.index(sector)] = 1.0
    ar, mr = _returns(closes), _returns(market_closes)
    n = min(len(ar), len(mr))
    extras = np.array(
        [
            float(log_mktcap),
            market_beta(ar[-n:], mr[-n:]),
            momentum_12_1(closes),
            realised_vol(ar),
        ],
        dtype=float,
    )
    return np.concatenate([onehot, extras])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_features.py --no-cov -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qts/propagation/equity/features.py tests/unit/test_equity_features.py
git commit -m "feat(equity): point-in-time node features (sector, size, beta, momentum, vol)"
```

---

### Task 4: Abnormal-return reaction labels

**Files:**
- Create: `src/qts/propagation/equity/labels.py`
- Test: `tests/unit/test_equity_labels.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_equity_labels.py
"""T-PATHA-LABEL-*: market-model abnormal-return reaction labels."""

from __future__ import annotations

import numpy as np

from qts.propagation.equity.labels import market_model_abnormal_return


def test_abnormal_return_isolates_idiosyncratic_shock() -> None:  # T-PATHA-LABEL-1
    rng = np.random.default_rng(0)
    n = 300
    mkt = rng.normal(0, 0.01, n)
    asset = 1.5 * mkt.copy()  # pure beta=1.5, zero alpha, no idio in estimation window
    # inject a +3% idiosyncratic shock on the event day (index = est_window)
    est_window = 250
    event_idx = est_window
    asset[event_idx] += 0.03
    car = market_model_abnormal_return(
        asset_returns=asset, market_returns=mkt, event_idx=event_idx, k=1, est_window=est_window
    )
    assert abs(car - 0.03) < 1e-3  # CAR over [t, t+1) recovers the shock, market move removed


def test_abnormal_return_zero_when_pure_market() -> None:  # T-PATHA-LABEL-2
    rng = np.random.default_rng(1)
    n = 300
    mkt = rng.normal(0, 0.01, n)
    asset = 0.8 * mkt  # pure market, no idio anywhere
    car = market_model_abnormal_return(
        asset_returns=asset, market_returns=mkt, event_idx=260, k=3, est_window=250
    )
    assert abs(car) < 1e-6  # nothing abnormal


def test_window_k_accumulates() -> None:  # T-PATHA-LABEL-3
    n = 300
    mkt = np.zeros(n)
    asset = np.zeros(n)
    asset[255:258] = 0.01  # +1% idio on each of 3 post-event days
    car = market_model_abnormal_return(
        asset_returns=asset, market_returns=mkt, event_idx=255, k=3, est_window=250
    )
    assert abs(car - 0.03) < 1e-9  # cumulative over [t, t+3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_labels.py --no-cov -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write implementation**

```python
# src/qts/propagation/equity/labels.py
"""Market-model abnormal returns: the prediction targets (spec §6, §7)."""

from __future__ import annotations

import numpy as np


def market_model_abnormal_return(
    *,
    asset_returns: np.ndarray,
    market_returns: np.ndarray,
    event_idx: int,
    k: int = 3,
    est_window: int = 250,
) -> float:
    """Cumulative abnormal return over ``[event_idx, event_idx + k)``.

    Estimate (alpha, beta) by OLS on the pre-event window ``[event_idx - est_window, event_idx)``
    (point-in-time), then CAR = sum of (r_asset - (alpha + beta * r_market)) over the event window.
    """
    a = np.asarray(asset_returns, dtype=float)
    m = np.asarray(market_returns, dtype=float)
    lo = event_idx - est_window
    if lo < 0:
        raise ValueError("not enough pre-event history for the estimation window")
    am, mm = a[lo:event_idx], m[lo:event_idx]
    var = float(np.var(mm))
    beta = float(np.cov(am, mm)[0, 1] / var) if var > 0 else 0.0
    alpha = float(np.mean(am) - beta * np.mean(mm))
    ev_a, ev_m = a[event_idx : event_idx + k], m[event_idx : event_idx + k]
    abnormal = ev_a - (alpha + beta * ev_m)
    return float(np.sum(abnormal))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_labels.py --no-cov -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/qts/propagation/equity/labels.py tests/unit/test_equity_labels.py
git commit -m "feat(equity): market-model abnormal-return reaction labels (Path A v2 Phase 2)"
```

---

### Task 5: EventSample contract + exports

**Files:**
- Create: `src/qts/propagation/equity/samples.py`
- Modify: `src/qts/propagation/equity/__init__.py`
- Test: `tests/unit/test_equity_samples.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_equity_samples.py
"""T-PATHA-SAMPLE-*: the Phase-3 training contract."""

from __future__ import annotations

from datetime import date

import numpy as np

from qts.propagation.equity.samples import EventSample


def test_event_sample_shapes_align() -> None:  # T-PATHA-SAMPLE-1
    feats = np.zeros((5, 8))  # 5 nodes, feature_dim 8
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
            reactions=np.zeros(4),  # mismatch
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_samples.py --no-cov -q`
Expected: FAIL — module not found.

- [ ] **Step 3: Write implementation**

```python
# src/qts/propagation/equity/samples.py
"""EventSample — one earnings do()-intervention on the universe graph (Phase-3 input contract)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np


@dataclass(frozen=True)
class EventSample:
    """A single event: named firm = the do() source (merit=SUE), targets = all nodes' reactions.

    ``features`` is the point-in-time node-feature matrix (n_nodes, feature_dim); ``reactions`` is the
    per-node abnormal return over the event window. ``named_idx`` indexes both, into the universe's
    sorted-ticker order.
    """

    named_idx: int
    merit: float
    event_date: date
    features: np.ndarray
    reactions: np.ndarray

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("features must be 2-D (n_nodes, feature_dim)")
        if self.reactions.shape != (self.features.shape[0],):
            raise ValueError("reactions must be 1-D of length n_nodes")
        if not (0 <= self.named_idx < self.features.shape[0]):
            raise ValueError("named_idx out of range")

    @property
    def n_nodes(self) -> int:
        return int(self.features.shape[0])
```

Then add to `src/qts/propagation/equity/__init__.py` (merge into the existing imports + `__all__`, keeping alphabetical order):

```python
from qts.propagation.equity.earnings import EarningsEvent, EarningsRow, compute_sue, normalize_yf_earnings
from qts.propagation.equity.features import (
    market_beta,
    momentum_12_1,
    node_feature_vector,
    realised_vol,
)
from qts.propagation.equity.labels import market_model_abnormal_return
from qts.propagation.equity.samples import EventSample
from qts.propagation.equity.universe import EquityUniverse, load_universe
```

Add these names to `__all__`: `"EarningsEvent"`, `"EarningsRow"`, `"Economic... (existing)"`, `"EquityUniverse"`, `"EventSample"`, `"compute_sue"`, `"load_universe"`, `"market_beta"`, `"market_model_abnormal_return"`, `"momentum_12_1"`, `"node_feature_vector"`, `"normalize_yf_earnings"`, `"realised_vol"` (alphabetical merge with the Phase-1 entries).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_samples.py --no-cov -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Full Phase-2 suite + ruff**

Run: `.venv/bin/python -m pytest tests/unit/test_equity_*.py --no-cov -q && .venv/bin/ruff check src/qts/propagation/equity/ tests/unit/test_equity_*.py && .venv/bin/ruff format src/qts/propagation/equity/ tests/unit/test_equity_*.py`
Expected: all equity tests pass (Phase 1 + Phase 2 = 13 + 17 = 30); ruff clean (line length 99 — wrap any long `__init__` import lines, no `noqa`).

- [ ] **Step 6: Commit**

```bash
git add src/qts/propagation/equity/samples.py src/qts/propagation/equity/__init__.py tests/unit/test_equity_samples.py
git commit -m "feat(equity): EventSample Phase-3 contract + Phase-2 exports"
```

---

## Phase 3 (outline — final follow-up plan)

- Real-data `CorrelationalBaseline` analog: `r̂_peer = β_{peer,named} · r_named` (β from pre-event history).
- **Real-data meta-training loop:** assemble `EventSample`s (one universe graph; train on training-link events, hold out links/firms); train `MetaPropagationGraph` (`qts.propagation.meta`); reuse `evaluate_meta_transfer` capture/win logic; `few_shot_adapt` per link.
- Feasibility gate (1-hop) + 2-hop extension + few-shot value-of-data curve; CLI `scripts/run_path_a_v2.py`; end-to-end run on real FNSPID + yfinance + Alpaca.

---

## Self-Review

**Spec coverage (Phase 2 scope):** §4 universe/yfinance/alias → Tasks 1–2. §2 SUE merit → Task 2. §5.3 node features → Task 3. §7 abnormal-return labels + point-in-time estimation window → Task 4. §3 mapping (EventSample = one do() on the universe) → Task 5. Phase 3 deferred.

**Placeholder scan:** none — every step has runnable code. Network code (`fetch_earnings_yf`, Alpaca fetch) is isolated behind normalise functions that ARE tested on fixtures; `fetch_earnings_yf` marked `# pragma: no cover` (network, not unit-tested) — its logic is `normalize_yf_earnings`, which is tested (T-PATHA-SUE-3).

**Type consistency:** `EarningsRow(date, estimate, reported)` and `EarningsEvent(ticker, date, sue)` are consistent across `earnings.py` + tests. `node_feature_vector` keyword signature matches its test. `market_model_abnormal_return` keyword signature (`asset_returns, market_returns, event_idx, k, est_window`) matches all three tests. `EventSample` fields match `samples.py` + tests. `EquityUniverse.alias_map()` output feeds Phase-1 `extract_tickers(text, alias_map)` (dict[str,str], lowercased) — verified against Phase-1's signature.

**Feature-dim note:** `node_feature_vector` returns `len(unique_sectors) + 4`. The Phase-3 trainer must set `MetaPropagationGraph(feature_dim=len(universe.unique_sectors) + 4)`. Documented here so Phase 3 wires it correctly.

---

## Execution Handoff

Phase 2 plan complete and saved. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks (as in Phase 1).
2. **Inline Execution** — execute in this session with checkpoints.
