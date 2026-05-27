# Grill: Live paper trading — crypto contagion
Date: 2026-05-26

## Intent
Forward-test the crypto contagion strategy with a *broadened* trigger that fires
on small idiosyncratic shocks (not just rare FTX/Terra/Curve mega-cascades), to
generate near-term paper P&L and prove the end-to-end live pipeline — before the
contagion edge decays as crypto market structure matures. A meaningful
out-of-sample Sharpe is explicitly *not* the near-term goal (a rare-event
strategy can't produce one quickly; broadening the trigger is the deliberate
response to that).

## Constraints
- Validate the broadened trigger on history **before** going live — no
  paper-trading a variant with zero backtest evidence.
- Reuse the existing NautilusTrader integration for execution (not a bespoke broker).
- Trigger must be pluggable: idiosyncratic-drop now, news/keyword detector later.
- Paper only — no real capital.

## Key decisions
- **Trigger v0 = idiosyncratic BTC-adjusted drop threshold**, behind a pluggable
  `ShockDetector`. Reason: matches how the operator was trained (BTC-adjusted
  CARs) and filters market-wide beta sell-offs. Rejected: raw price drop
  (conflates beta with idiosyncratic contagion — the exact null the model strips);
  vol-adjusted z-score (deferred as an upgrade); news trigger (needs a scraper
  that doesn't exist — scaffolded only).
- **Validation = backtest the broadened trigger on the historical archive first,
  as a hard gate.** Reason: don't commit to live operation on an unvalidated
  variant. Rejected: straight-to-live (blind); both-in-parallel (more upfront work).
- **Execution = reuse NautilusTrader.** Reason: fidelity + consistency with the
  main system; a live skeleton already exists (`nautilus/live.py`). Rejected:
  thin paper-broker (faster but diverges from the main system); abstract/defer.
- **Paper venue = local sandbox sim on live mainnet data.** Reason: real prices +
  controllable fills, best for measuring true performance. Rejected: Binance
  testnet (distorted prices/liquidity).
- **Horizon = 72h.** Reason: the feasibility study found the market-neutral edge
  lives at 72h; 24h was weak (+1.4%/event). Rejected: 24h (the v0 script default).
- **Model = fit-once → freeze to disk → load for inference.** Reason: live
  inference needs a frozen artifact; today the operator is only re-fit in-process
  (`train.py:80-86`). Rejected: rolling re-fit (out of scope for v0).

## Surfaced assumptions
- The existing "backtest" is a **return proxy** (BTC-adjusted abnormal return
  minus a flat 7.5 bps, `gate.py:179`), *not* a position-level execution sim. The
  Nautilus paper path is the first time real positions / perp hedge / fills /
  funding exist — so its P&L will **not** match the backtest 1:1; it is
  deliberately higher fidelity.
- The market-neutral hedge is **implicit** in the backtest (BTC-beta is regressed
  out, `dataset.py:95-101`). Live, it must become an **explicit** perp short + BTC
  hedge.
- `events_per_year` is hardcoded to 12 (`gate.py:161`) → must be set to the
  *detected* frequency once the trigger is broadened, or the annualised Sharpe is
  meaningless.
- Events outside `[FEATURE_WINDOW, len(grid)-horizon]` are silently dropped
  (`dataset.py:86`) → must be logged for generated events.
- `usd_severity` (`events.py`) is a dead field downstream → repurpose for
  shock-magnitude logging.

## Open questions
- Exact X / Y threshold for the idiosyncratic-drop detector (calibrate in Phase 1).
- Live universe / watchlist composition (default: union of the cascade universes +
  their structural links; confirm during planning).
- Paper notional sizing + max concurrent positions.
- Cooldown window length for one-shock-per-token de-duplication.

## Out of scope
- News / keyword trigger (interface scaffolded only).
- Real-capital live trading.
- Rolling re-fit of the operator.
- Trigger-threshold tuning (Optuna).
