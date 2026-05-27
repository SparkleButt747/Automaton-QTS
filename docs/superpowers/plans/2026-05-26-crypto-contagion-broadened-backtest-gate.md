# Crypto Contagion — Broadened-Trigger Backtest Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decide GO/NO-GO on whether the contagion strategy, fired by a *broadened* small-shock trigger (not just rare mega-cascades), has tradeable edge — before committing to the expensive live build.

**Architecture:** A pluggable `ShockDetector` turns the historical price panel into `ContagionEvent`s (idiosyncratic BTC-adjusted drawdowns). Those events feed the *existing* crypto contagion machinery (dataset → operator fit → Null A / Null B / costed backtest) unchanged, with the event frequency and 72h horizon corrected. The run prints a GO/NO-GO verdict. The detector built here is reused verbatim by the live build later.

**Tech Stack:** Python 3.12, numpy, PyYAML, pytest (`asyncio_mode=auto`); reuses `qts.propagation.crypto.*` and `qts.propagation.equity` (the shared `RelationTypedPropagation` operator).

**Scope note — this is Plan 1 of 2.** Plan 2 (model freeze + multi-instrument NautilusTrader live paper trading) is deliberately deferred: it is only worth writing if this gate returns GO. Per the spec, Phase 1 STOPs the whole effort if no edge is found.

**Spec:** `docs/specs/2026-05-26-crypto-contagion-live-paper-trading-v0.md`

---

## File Structure

| File | New/Modify | Responsibility |
|---|---|---|
| `src/qts/propagation/crypto/detect.py` | **new** | `ShockDetector` protocol + `IdiosyncraticDropDetector` (panel → `list[ContagionEvent]`) |
| `src/qts/propagation/crypto/broadened.py` | **new** | Pure helpers: `events_per_year`, `broadened_verdict` |
| `src/qts/propagation/crypto/universe.py` | **modify** | Add `build_live_universe` (union of per-event universe configs) |
| `src/qts/propagation/crypto/structural_links.py` | **modify** | Add `build_live_structural_links` (union of per-event structural seeds) |
| `scripts/build_live_universe.py` | **new** | CLI: write `crypto_contagion_live.yaml` + `crypto_structural_live.yaml` |
| `scripts/run_contagion_broadened_backtest.py` | **new** | The gate runner: detect → existing backtest → GO/NO-GO verdict |
| `tests/unit/test_crypto_live_union.py` | **new** | T-UNI-* |
| `tests/unit/test_crypto_shock_detector.py` | **new** | T-SHOCK-* |
| `tests/unit/test_crypto_broadened.py` | **new** | T-BTEST-* |

Reused verbatim (do **not** modify): `events.py` (`ContagionEvent`), `dataset.py` (`build_crypto_contagion_dataset`), `gate.py` (`fit_crypto_propagation`, `event_study_linked_vs_unlinked`, `evaluate_crypto_gate`, `contagion_backtest` + its report dataclasses), `reactions.py` (`btc_adjusted_car`), `prices.py` (`fetch_price_panel`), `equity/gate.py` (`train_holdout_adjacency`).

---

## Task 1: Live universe + structural-seed union builders

The broadened backtest needs one universe spanning every token where contagion has been observed, plus a matching structural-link seed. Both are the union of the existing per-event configs.

**Files:**
- Modify: `src/qts/propagation/crypto/universe.py`
- Modify: `src/qts/propagation/crypto/structural_links.py`
- Create: `scripts/build_live_universe.py`
- Test: `tests/unit/test_crypto_live_union.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_crypto_live_union.py
"""T-UNI-*: union builders for the live contagion universe + structural seed."""

from __future__ import annotations

from pathlib import Path

import yaml

from qts.propagation.crypto.structural_links import (
    build_live_structural_links,
    load_structural_links,
)
from qts.propagation.crypto.universe import build_live_universe, load_crypto_universe


def test_universe_union_merges_all_event_configs(tmp_path: Path) -> None:  # T-UNI-1
    (tmp_path / "crypto_contagion_ftx.yaml").write_text(
        yaml.safe_dump({"tokens": {"FTT": {"cluster": "exchange"}, "BTC": {"cluster": "major"}}})
    )
    (tmp_path / "crypto_contagion_terra.yaml").write_text(
        yaml.safe_dump({"tokens": {"LUNA": {"cluster": "l1"}, "BTC": {"cluster": "major"}}})
    )
    out = tmp_path / "crypto_contagion_live.yaml"
    merged = build_live_universe(tmp_path, out)
    assert set(merged) == {"FTT", "BTC", "LUNA"}
    uni = load_crypto_universe(out)
    assert set(uni.tokens) == {"FTT", "BTC", "LUNA"}


def test_structural_union_dedupes_keeping_higher_confidence(tmp_path: Path) -> None:  # T-UNI-2
    (tmp_path / "crypto_structural_ftx.yaml").write_text(
        yaml.safe_dump(
            {"links": [{"source": "FTT", "peer": "SOL", "relation": "entity_exposure", "confidence": 0.8}]}
        )
    )
    (tmp_path / "crypto_structural_terra.yaml").write_text(
        yaml.safe_dump(
            {
                "links": [
                    {"source": "FTT", "peer": "SOL", "relation": "entity_exposure", "confidence": 0.95},
                    {"source": "LUNA", "peer": "UST", "relation": "collateral_of", "confidence": 0.99},
                ]
            }
        )
    )
    out = tmp_path / "crypto_structural_live.yaml"
    links = build_live_structural_links(tmp_path, out)
    by_pair = {(l["source"], l["peer"]): l for l in links}
    assert by_pair[("FTT", "SOL")]["confidence"] == 0.95  # higher kept
    assert ("LUNA", "UST") in by_pair
    assert len(load_structural_links(out)) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_crypto_live_union.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_live_universe'`.

- [ ] **Step 3: Add `build_live_universe` to `universe.py`**

Append to `src/qts/propagation/crypto/universe.py`:

```python
def build_live_universe(src_dir: Path, out_path: Path) -> dict[str, dict]:
    """Merge every ``crypto_contagion_*.yaml`` token map in ``src_dir`` into one
    universe written to ``out_path``. First definition of a token wins."""
    merged: dict[str, dict] = {}
    for path in sorted(src_dir.glob("crypto_contagion_*.yaml")):
        if path.name == out_path.name:
            continue
        raw = yaml.safe_load(path.read_text()) or {}
        for token, meta in raw.get("tokens", {}).items():
            merged.setdefault(token, meta)
    out_path.write_text(yaml.safe_dump({"tokens": merged}, sort_keys=True))
    return merged
```

- [ ] **Step 4: Add `build_live_structural_links` to `structural_links.py`**

Append to `src/qts/propagation/crypto/structural_links.py`:

```python
def build_live_structural_links(src_dir: Path, out_path: Path) -> list[dict]:
    """Merge every ``crypto_structural_*.yaml`` link list in ``src_dir`` into one
    seed written to ``out_path``. De-dupe on (source, peer, relation), keep the
    highest confidence."""
    merged: dict[tuple[str, str, str], dict] = {}
    for path in sorted(src_dir.glob("crypto_structural_*.yaml")):
        if path.name == out_path.name:
            continue
        raw = yaml.safe_load(path.read_text()) or {}
        for link in raw.get("links", []):
            key = (str(link["source"]), str(link["peer"]), str(link["relation"]))
            prev = merged.get(key)
            if prev is None or float(link.get("confidence", 0.9)) > float(prev.get("confidence", 0.9)):
                merged[key] = link
    links = list(merged.values())
    out_path.write_text(yaml.safe_dump({"links": links}, sort_keys=True))
    return links
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/unit/test_crypto_live_union.py -v`
Expected: PASS (both T-UNI-1, T-UNI-2).

- [ ] **Step 6: Create the CLI wrapper**

```python
# scripts/build_live_universe.py
"""Build the live contagion universe + structural seed as the union of all per-event
configs — the live strategy watches everywhere contagion has been observed."""

from pathlib import Path

from qts.propagation.crypto.structural_links import build_live_structural_links
from qts.propagation.crypto.universe import build_live_universe

if __name__ == "__main__":
    uni = build_live_universe(
        Path("config/universe"), Path("config/universe/crypto_contagion_live.yaml")
    )
    links = build_live_structural_links(
        Path("config/links"), Path("config/links/crypto_structural_live.yaml")
    )
    print(f"universe: {len(uni)} tokens -> config/universe/crypto_contagion_live.yaml")
    print(f"structural: {len(links)} links -> config/links/crypto_structural_live.yaml")
```

- [ ] **Step 7: Generate the live configs and eyeball them**

Run: `python scripts/build_live_universe.py`
Expected: prints token + link counts; creates `config/universe/crypto_contagion_live.yaml` and `config/links/crypto_structural_live.yaml`. Confirm `BTC` is present in the universe (the panel needs `closes["BTC"]`).

- [ ] **Step 8: Commit**

```bash
git add src/qts/propagation/crypto/universe.py src/qts/propagation/crypto/structural_links.py scripts/build_live_universe.py tests/unit/test_crypto_live_union.py config/universe/crypto_contagion_live.yaml config/links/crypto_structural_live.yaml
git commit -m "feat(crypto): add live universe + structural-seed union builders"
```

---

## Task 2: `ShockDetector` protocol + `IdiosyncraticDropDetector`

The trigger. Scans the price panel; emits a `ContagionEvent` when a token's trailing BTC-adjusted abnormal return over `window` bars falls below `-threshold`. Reuses `btc_adjusted_car` (so the detection metric matches the labelling metric) by treating "`window` bars ago" as a pseudo-event. The `ShockDetector` Protocol is the seam a future news detector implements.

**Files:**
- Create: `src/qts/propagation/crypto/detect.py`
- Test: `tests/unit/test_crypto_shock_detector.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_crypto_shock_detector.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_crypto_shock_detector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qts.propagation.crypto.detect'`.

- [ ] **Step 3: Implement the detector**

```python
# src/qts/propagation/crypto/detect.py
"""Shock detectors: turn a price panel into ContagionEvents — the live trigger, run
offline here for the gate. v0 fires on an idiosyncratic (BTC-adjusted) drawdown, matching
how the operator is trained (BTC-adjusted CARs) and filtering market-wide beta sell-offs.
A news/keyword detector can later implement the same ShockDetector protocol."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np

from qts.propagation.crypto.events import ContagionEvent
from qts.propagation.crypto.reactions import btc_adjusted_car


@runtime_checkable
class ShockDetector(Protocol):
    def detect(
        self, closes: dict[str, np.ndarray], grid: list[datetime], btc_closes: np.ndarray
    ) -> list[ContagionEvent]: ...


class IdiosyncraticDropDetector:
    """Fire when a token's trailing BTC-adjusted abnormal return over ``window`` bars
    falls below ``-threshold``; at most one event per token per ``cooldown`` bars."""

    def __init__(
        self,
        *,
        threshold: float = 0.15,
        window: int = 24,
        est_window: int = 720,
        cooldown: int = 72,
    ) -> None:
        self.threshold = threshold
        self.window = window
        self.est_window = est_window
        self.cooldown = cooldown

    def detect(
        self, closes: dict[str, np.ndarray], grid: list[datetime], btc_closes: np.ndarray
    ) -> list[ContagionEvent]:
        events: list[ContagionEvent] = []
        n = len(grid)
        first = self.est_window + self.window
        for token, tc in closes.items():
            if token == "BTC":
                continue
            last_fire = -self.cooldown - 1
            for t in range(first, n):
                if t - last_fire <= self.cooldown:
                    continue
                ar = btc_adjusted_car(
                    token_closes=tc,
                    btc_closes=btc_closes,
                    event_close_idx=t - self.window,
                    horizon=self.window,
                    est_window=self.est_window,
                )
                if ar < -self.threshold:
                    events.append(
                        ContagionEvent(
                            source_token=token,
                            timestamp=grid[t],
                            event_type="drop",
                            usd_severity=float(abs(ar)),
                        )
                    )
                    last_fire = t
        return sorted(events, key=lambda e: e.timestamp)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_crypto_shock_detector.py -v`
Expected: PASS (T-SHOCK-0/1/2/3).

- [ ] **Step 5: Commit**

```bash
git add src/qts/propagation/crypto/detect.py tests/unit/test_crypto_shock_detector.py
git commit -m "feat(crypto): add pluggable ShockDetector + idiosyncratic-drop trigger"
```

---

## Task 3: Broadened-backtest helpers + gate runner

Two pure, unit-tested helpers (the event-frequency fix and the verdict), then the orchestration script that wires the detector into the existing backtest and prints GO/NO-GO.

**Files:**
- Create: `src/qts/propagation/crypto/broadened.py`
- Create: `scripts/run_contagion_broadened_backtest.py`
- Test: `tests/unit/test_crypto_broadened.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_crypto_broadened.py
"""T-BTEST-*: broadened-backtest event frequency + GO/NO-GO verdict."""

from __future__ import annotations

from datetime import UTC, datetime

from qts.propagation.crypto.broadened import broadened_verdict, events_per_year
from qts.propagation.crypto.events import ContagionEvent
from qts.propagation.crypto.gate import BacktestResult, CryptoGateReport, EventStudyReport


def _events(k: int) -> list[ContagionEvent]:
    ts = datetime(2022, 1, 1, tzinfo=UTC)
    return [ContagionEvent(source_token="ALT", timestamp=ts, event_type="drop", usd_severity=0.2)] * k


def test_events_per_year_annualises_over_span() -> None:  # T-BTEST-1
    start = datetime(2021, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 1, tzinfo=UTC)  # ~3 years
    assert round(events_per_year(_events(36), start, end)) == 12
    assert events_per_year([], start, end) == 0.0


def _es(sig: bool) -> EventStudyReport:
    return EventStudyReport(
        n_linked=10, n_unlinked=10, mean_linked_car=-0.05, mean_unlinked_car=0.0,
        mann_whitney_p=0.01 if sig else 0.5, significant=sig,
    )


def _gate(beats: bool) -> CryptoGateReport:
    return CryptoGateReport(
        n_linked_obs=10, graph_mse=0.1, pairwise_mse=0.2, graph_hit=0.6, pairwise_hit=0.5,
        beats_pairwise=beats,
    )


def _bt(sharpe: float) -> BacktestResult:
    return BacktestResult(
        n_trades=20, market_neutral_mean=0.01, market_neutral_sharpe=sharpe,
        outright_mean=0.0, outright_sharpe=0.0,
    )


def test_verdict_requires_all_three_conditions() -> None:  # T-BTEST-2
    assert broadened_verdict(_es(True), _gate(True), _bt(1.5)) is True
    assert broadened_verdict(_es(False), _gate(True), _bt(1.5)) is False
    assert broadened_verdict(_es(True), _gate(False), _bt(1.5)) is False
    assert broadened_verdict(_es(True), _gate(True), _bt(0.5)) is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/unit/test_crypto_broadened.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'qts.propagation.crypto.broadened'`.

- [ ] **Step 3: Implement the helpers**

```python
# src/qts/propagation/crypto/broadened.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/unit/test_crypto_broadened.py -v`
Expected: PASS (T-BTEST-1, T-BTEST-2).

- [ ] **Step 5: Create the gate runner**

```python
# scripts/run_contagion_broadened_backtest.py
"""Broadened-trigger contagion backtest GATE (Phase 1). Detect idiosyncratic-drop events
over history -> existing dataset/fit/Null A/Null B/costed backtest -> GO/NO-GO verdict on
whether the broadened variant is worth taking live. Reuses run_crypto_contagion_v0.py's
flow with detector-sourced events, the detected event frequency, and the 72h horizon.

Usage: python scripts/run_contagion_broadened_backtest.py [universe.yaml] [structural.yaml]"""

import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from qts.data.market.binance_adapter import BinanceBarAdapter
from qts.oversight.llm_client import LlamaCppClient
from qts.propagation.crypto.broadened import broadened_verdict, events_per_year
from qts.propagation.crypto.dataset import build_crypto_contagion_dataset
from qts.propagation.crypto.detect import IdiosyncraticDropDetector
from qts.propagation.crypto.gate import (
    contagion_backtest,
    evaluate_crypto_gate,
    event_study_linked_vs_unlinked,
    fit_crypto_propagation,
)
from qts.propagation.crypto.prices import fetch_price_panel
from qts.propagation.crypto.universe import load_crypto_universe
from qts.propagation.equity.gate import train_holdout_adjacency

logging.basicConfig(level=logging.INFO)
UNI = Path(sys.argv[1] if len(sys.argv) > 1 else "config/universe/crypto_contagion_live.yaml")
SEED = Path(sys.argv[2] if len(sys.argv) > 2 else "config/links/crypto_structural_live.yaml")
START, END = "2021-01-01", "2024-01-01"
HORIZON, EST_WINDOW = 72, 720  # 72h hold (feasibility-validated); 30-day (720h) beta estimation


async def main() -> None:
    uni = load_crypto_universe(UNI)
    adapter = BinanceBarAdapter()
    grid, closes = await fetch_price_panel(
        list(uni.tokens), bar_adapter=adapter, start=START, end=END, interval="1h"
    )
    detector = IdiosyncraticDropDetector(
        threshold=0.15, window=24, est_window=EST_WINDOW, cooldown=72
    )
    events = detector.detect(closes, grid, closes["BTC"])
    print(f"detector fired {len(events)} events over {START}..{END}")
    if len(events) < 4:
        print("too few detected events — lower the threshold or widen the universe")
        return

    ds = await build_crypto_contagion_dataset(
        uni,
        events,
        bar_adapter=adapter,
        structural_seed_path=SEED,
        llm=LlamaCppClient(base_url="http://localhost:8080"),
        cache_dir=Path("data/crypto/link_cache_live"),
        horizon=HORIZON,
        est_window=EST_WINDOW,
        start=START,
        end=END,
    )
    print(f"{len(ds.samples)} usable events, {int((ds.adj_type >= 0).sum())} typed edges")
    if len(ds.samples) < 4:
        print("too few usable events after panel alignment")
        return

    adj_train, _ = train_holdout_adjacency(ds.adj_type, holdout_frac=0.3)
    model = fit_crypto_propagation(ds.samples, adj_type=adj_train, feature_dim=ds.feature_dim)
    named = np.array([s.named_idx for s in ds.samples])
    merit = np.array([s.merit for s in ds.samples])
    pred = model.predict_np(ds.samples[0].features, ds.adj_type, named, merit)

    es = event_study_linked_vs_unlinked(ds.samples, adj_type=ds.adj_type)
    gate = evaluate_crypto_gate(pred, ds.samples, adj_type=ds.adj_type)
    epy = events_per_year(
        events,
        datetime.fromisoformat(START).replace(tzinfo=UTC),
        datetime.fromisoformat(END).replace(tzinfo=UTC),
    )
    bt = contagion_backtest(
        pred,
        ds,
        token_names=uni.tokens,
        top_k=3,
        cost_bps=7.5,
        horizon=HORIZON,
        events_per_year=epy,
    )
    print(
        f"\nNULL A: linked {es.mean_linked_car:+.4f} vs unlinked {es.mean_unlinked_car:+.4f} "
        f"| p={es.mann_whitney_p:.4f} {'SIG' if es.significant else 'n.s.'}"
    )
    print(
        f"NULL B: graph MSE {gate.graph_mse:.5f} vs pairwise {gate.pairwise_mse:.5f} "
        f"| {'BEATS' if gate.beats_pairwise else 'loses'}"
    )
    print(
        f"BACKTEST: MN mean {bt.market_neutral_mean:+.4f} Sharpe {bt.market_neutral_sharpe:+.2f} "
        f"| {bt.n_trades} trades | {epy:.0f} events/yr"
    )
    go = broadened_verdict(es, gate, bt)
    print(f"\nGATE: {'GO — broadened variant has edge; proceed to Plan 2 (live)' if go else 'NO-GO — stop'}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 6: Commit the helpers + runner**

```bash
git add src/qts/propagation/crypto/broadened.py scripts/run_contagion_broadened_backtest.py tests/unit/test_crypto_broadened.py
git commit -m "feat(crypto): add broadened-trigger backtest gate runner"
```

- [ ] **Step 7: Smoke-run the gate (integration; needs the local LLM at :8080 + network)**

Run: `python scripts/run_contagion_broadened_backtest.py`
Expected: prints the detected-event count, usable-event count, the Null A / Null B / backtest lines, and a final `GATE: GO ...` or `GATE: NO-GO — stop`. This is the deliverable — record the verdict; it decides whether Plan 2 is written.

---

## Final Verification

- [ ] Run the full new suite: `pytest tests/unit/test_crypto_live_union.py tests/unit/test_crypto_shock_detector.py tests/unit/test_crypto_broadened.py -v` — all pass.
- [ ] Lint: `ruff check src/qts/propagation/crypto/detect.py src/qts/propagation/crypto/broadened.py scripts/run_contagion_broadened_backtest.py scripts/build_live_universe.py` — clean.
- [ ] The gate run produced a recorded GO/NO-GO verdict.

---

## Self-Review (completed by plan author)

- **Spec coverage:** Phase 1 of the spec is fully covered — `ShockDetector` (§4), the broadened backtest reusing the existing machinery (§3 Phase 1), the `events_per_year` fix and 72h horizon (§9), and the GO/NO-GO gate. Phase 0 (freeze) and Phase 2 (Nautilus live paper) are intentionally out of this plan — they belong in Plan 2, written only on a GO (per the spec's gate semantics).
- **Placeholder scan:** none — every step carries complete code or an exact command + expected output. The smoke-run (Task 3 Step 7) is a real integration run, not a placeholder.
- **Type consistency:** report dataclass fields (`EventStudyReport`, `CryptoGateReport`, `BacktestResult`) and the `contagion_backtest`/`fit_crypto_propagation`/`predict_np` signatures match the verbatim definitions in `gate.py` and `equity/model.py`. `ContagionEvent` fields match `events.py`. `build_crypto_contagion_dataset` and `fetch_price_panel` calls match `dataset.py`.
- **Known modelling note (carried to Plan 2):** the backtest seeds propagation with each event's *forward* realised BTC-adjusted move (`merit`, `dataset.py:122`); the live strategy will only have the *observed-to-now* move. This train/live seed difference is inherited from the existing v0 design and does not affect the gate, but Plan 2 must surface it.
