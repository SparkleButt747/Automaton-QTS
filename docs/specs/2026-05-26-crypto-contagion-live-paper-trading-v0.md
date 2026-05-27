# Crypto Contagion — Live Paper Trading v0

- **Date:** 2026-05-26
- **Status:** Approved design → ready for planning
- **Related:**
  - `docs/specs/2026-05-24-crypto-contagion-propagation-v0.md` (the backtest spec)
  - `docs/research/2026-05-25-crypto-contagion-feasibility.md` (MN Sharpe ~1.49, 72h horizon)
  - `.grill/live-paper-trading-crypto-contagion.md` (decision log)

## 1. Goal

Forward-test the crypto contagion strategy with a **broadened trigger** that fires
on small idiosyncratic shocks — not just the rare FTX/Terra/Curve mega-cascades —
in order to:

1. generate **near-term paper P&L** (a forward track record starting now), and
2. **prove the end-to-end live pipeline** works (detect → predict → size → fill → P&L),

before the contagion edge decays as crypto market structure matures.

**Non-goal:** a statistically meaningful out-of-sample Sharpe. A rare-event
strategy cannot produce one in any reasonable paper window; broadening the
trigger is the deliberate response, at the cost of trading a variant the
mega-cascade backtest does not certify (hence the Phase-1 validation gate).

## 2. Locked decisions

| Axis | Decision |
|---|---|
| Trigger (v0) | Idiosyncratic BTC-adjusted drop threshold, behind a pluggable `ShockDetector` (news detector scaffolded for later) |
| Validation | Backtest the broadened trigger on the historical archive **first**, as a hard gate before live |
| Execution | Reuse the NautilusTrader integration (`src/qts/nautilus/`) |
| Paper venue | Local sandbox sim against **live mainnet** Binance data |
| Horizon | **72h** (feasibility-validated; not the v0 script's 24h) |
| Model | Fit-once on FTX/Terra/Curve → freeze to disk → load for live inference |

## 3. Architecture — one frozen model, two phases, two gates

- **Phase 0 — Freeze the model (fit-once).** Fit `RelationTypedPropagation`
  on the historical cascades, then persist the operator state + link graph +
  node features to a versioned artifact. Closes the gap where the operator is
  only ever re-fit in-process (`src/qts/propagation/crypto/train.py:80-86`).
- **Phase 1 — Broadened backtest (GATE).** Run the `ShockDetector` over the
  historical archive → `list[ContagionEvent]` → existing dataset/fit/nulls/backtest.
  **If Null A is insignificant, or it fails to beat the pairwise baseline, or the
  market-neutral Sharpe ≤ 0 → STOP.** Do not go live on a dead variant.
- **Phase 2 — Live paper (Nautilus).** Live perp bars → `ShockDetector` →
  frozen operator → rank predicted-to-drop peers → open market-neutral position
  (short peers + beta-weighted BTC hedge, perps) → hold 72h → close → P&L log.
  **GATE (acceptance):** replay FTX/Terra/Curve through the live code path and
  reconcile **signals** against the backtest.

## 4. Components

| Component | File | Status | Responsibility |
|---|---|---|---|
| `ShockDetector` protocol + `IdiosyncraticDropDetector` | `src/qts/propagation/crypto/detect.py` | **new** | `detect(bars) -> list[ContagionEvent]`. Emit an event when a universe token's BTC-adjusted residual return drops > X% over Y hours. Reuse `btc_adjusted_car` (`reactions.py:17-36`) / the regression at `dataset.py:95-101`. |
| `NewsKeywordDetector` | same | **stub** | Same protocol; not implemented — scaffold only. |
| Frozen artifact + inference | `src/qts/propagation/crypto/freeze.py`, `scripts/freeze_contagion_model.py` | **new** | Fit once (`fit_crypto_propagation`, `gate.py:23`), `torch.save` `state_dict` + adjacency + node features + universe/relation metadata → `models/contagion_v0.pt`. `load_and_predict(path, links, shock_seed) -> peer_reactions` callable outside the backtest loop. |
| Phase-1 runner | `scripts/run_contagion_broadened_backtest.py` | **new** | Detector events → `build_crypto_contagion_dataset` (`dataset.py:53`) → fit/predict → Null A (`gate.py:108`) + Null B (`gate.py:66`) + `contagion_backtest` (`gate.py:152`). Set `events_per_year` to the **detected** frequency. Emit a verdict. |
| `ContagionStrategy` | extends `src/qts/nautilus/actor.py` (`QTSStrategy`, lines 58-335) | **new** | Nautilus Strategy: `on_bar` → detect → `load_and_predict` → rank peers → submit MN perp orders → manage 72h lifecycle → close. |
| Live wiring fix + sandbox venue + instruments | `src/qts/nautilus/live.py` | **fix/extend** | Fix the bug at `live.py:126-135` (pass `data_config`/`exec_config` into `TradingNodeConfig(data_clients=…, exec_clients=…)`); add a `SandboxExecutionClient` simulating fills on live mainnet data; load real Binance **perp** instruments (replace `TestInstrumentProvider`, `runner.py:75-78`). |
| Paper P&L logger | `src/qts/propagation/crypto/paper_log.py` | **new** | Persist every signal, order, fill, and per-trade P&L to JSONL — including shadow signals when no trade fires. |

## 5. The `ContagionEvent` seam (one detector, both phases)

`ShockDetector` emits the exact dataclass the backtest already consumes, so the
same detector drives Phase 1 and Phase 2.

`ContagionEvent` (`src/qts/propagation/crypto/events.py:13-18`):

| Field | Type | Use |
|---|---|---|
| `source_token` | `str` | shock origin ticker |
| `timestamp` | tz-aware UTC `datetime` | when the shock is first observable |
| `event_type` | `str` | label only (`"drop"` for v0) |
| `usd_severity` | `float` | **dead downstream** — repurpose to carry the drop magnitude for logging |

- Phase 1: hand `list[ContagionEvent]` to `build_crypto_contagion_dataset` at
  `scripts/run_crypto_contagion_v0.py:36`, bypassing `load_contagion_events`.
- The operator's seed (`merit`) is the source token's **realized BTC-adjusted
  return** during the window (`dataset.py:122`), not `usd_severity`.
- Events outside `[FEATURE_WINDOW, len(grid)-horizon]` are dropped silently
  (`dataset.py:86`) → **log** dropped events in Phase 1.

## 6. Data flow (live)

```
BinanceDataClient (live perp bars)
  → ContagionStrategy.on_bar
    → ShockDetector.detect            # idiosyncratic BTC-adjusted drop?
      → (shock) load_and_predict      # frozen operator over hybrid link graph
        → rank predicted-to-drop peers
          → submit orders             # short peers (perp) + beta-weighted BTC hedge (perp)
            → SandboxExecutionClient   # simulated fill on live price
              → position held 72h → close
                → paper_log (JSONL)
```

## 7. Error handling

- **WS disconnect** → Nautilus auto-reconnect; the detector skips bars with
  stale/missing data (never fire on a data gap).
- **De-duplication** → one shock per token per cooldown window (avoid re-firing
  every bar on the same drawdown).
- **Artifact integrity** → `contagion_v0.pt` is version-stamped with its universe
  + relation set; the strategy refuses to start on a mismatch or a missing artifact.
- **Restart safety** → open positions + cumulative P&L persisted; the process
  resumes cleanly (it is long-running).
- **Boundary drops** → log any event discarded by `dataset.py:86`.

## 8. Testing strategy

No mocking of internal code (house rule). Feed = the archive adapter
(`binance_archive_adapter`) on a fixed historical window.

| ID | Test |
|---|---|
| T-SHOCK-1 | Detector fires exactly at the threshold boundary; not just below it |
| T-SHOCK-2 | Detector emits a well-formed `ContagionEvent` (tz-aware UTC ts, magnitude in `usd_severity`) |
| T-SHOCK-3 | A market-wide beta sell-off (all tokens down with BTC) does **not** fire (BTC-adjustment works) |
| T-FREEZE-1 | freeze → load round-trips: predictions identical pre/post `torch.save` |
| T-FREEZE-2 | Strategy refuses to start on a universe/relation mismatch |
| T-BTEST-1 | `events_per_year` is set from the detected count, not the hardcoded 12 |
| T-BTEST-2 | Phase-1 verdict gate returns STOP when Null A is insignificant |
| T-LIVE-1 | `live.py` wires `data_clients`/`exec_clients` into `TradingNodeConfig` (regression on the `live.py:126-135` bug) |
| T-LIVE-2 | MN position opens a peer short **and** a beta-weighted BTC hedge; both via perp |
| T-LIVE-3 | Position closes at 72h; restart mid-hold resumes the open position |
| T-PIPE-1 | **Acceptance:** replaying FTX/Terra/Curve through the live path re-discovers the events and reproduces the backtest's **signal-level** predictions (reconcile signals, not P&L) |

## 9. Surfaced design realities (honesty notes for the implementer)

- The existing backtest is a **return proxy** (`-reaction - cost`, `gate.py:179`;
  flat 7.5 bps, `gate.py:158/168`), not a position-level sim. The Nautilus paper
  path adds real positions, an explicit perp hedge, real fills, and funding — so
  **its P&L will not match the backtest 1:1, by design.** The acceptance test
  reconciles *signals*, not P&L.
- MN hedge is **implicit** in the backtest (BTC-beta regressed out); live it is an
  **explicit** perp short + BTC hedge.
- Horizon: the v0 script defaults to 24h (`run_crypto_contagion_v0.py:30`,
  `HORIZON, EST_WINDOW = 24, 720`) where the edge is weak; **live uses 72h**.

## 10. Out of scope (v0)

- News / keyword trigger (interface scaffolded only).
- Real-capital live trading.
- Rolling re-fit of the operator.
- Trigger-threshold tuning (Optuna).

## 11. Open questions (resolve during planning)

- Idiosyncratic-drop threshold X (drop %) and window Y (hours) — calibrate in Phase 1.
- Live universe / watchlist (default: union of the cascade universes + their
  structural links).
- Paper notional per trade + max concurrent positions.
- Cooldown window length for one-shock-per-token de-duplication.
