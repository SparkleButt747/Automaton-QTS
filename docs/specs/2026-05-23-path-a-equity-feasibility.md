# Spec: Path A v0 — Real-Equity Propagation Feasibility Cut

**Status:** SUPERSEDED (2026-05-23) by `2026-05-23-path-a-v2-nhop-meta-feasibility.md`. This 1-hop-only
cut pre-dates the n-hop meta-learning breakthrough (design §15); the v2 spec upgrades it to the
meta-trained operator over an FNSPID co-mention link graph, with 1-hop as the gate and 2-hop as an
extension. Kept for history.

First real-data test of the event-propagation thesis, on equities. Mirrors the sim feasibility cut
(design doc `docs/research/2026-05-22-event-propagation-graph-design.md`).

## 1. Goal

Test, on **real US-equity earnings events**, the core thesis the sim proved synthetically: **does a
propagation model predict the reaction of an *unnamed* economically-linked peer better than a
correlational baseline?** Scope is deliberately a feasibility cut — **1-hop only** (n-hop is banked,
§14), **earnings-surprise merit** (no LLM yet), **curated competitor edges** (no learned graph yet).

## 2. Locked decisions (from grilling, 2026-05-23)

| Decision | Choice | Why |
|----------|--------|-----|
| Market / event | US equities, **earnings announcements** | the documented economic-link alpha (Cohen-Frazzini) lives here; earnings are well-timed and quantifiable |
| Merit | **standardised earnings surprise (SUE)** — actual vs consensus EPS, scaled | clean quant merit; isolates *propagation* from LLM-extraction noise (LLM merit is the next layer) |
| Relation graph | **hand-curated competitor/sector peer pairs** | smallest fair test of "event on A → unnamed peer B"; must be **non-correlational** edges (see §5) |
| Hops | **1-hop only** | the proven thesis; multi-hop is banked pending an NBFNet-style operator (§14.5) |
| Test | **beat a correlational baseline on the peer's reaction** | identical bar to the sim gate |

## 3. Data

- **Bars:** reuse `src/qts/data/market/alpaca_adapter.py` (real US-equity OHLCV).
- **Earnings dates + consensus + surprise:** **adopt `yfinance`** (`Ticker.get_earnings_dates()` gives
  estimate / reported / surprise%) for v0 — free, no key, adequate large-cap coverage. *Search-first
  (CLAUDE.md r6): yfinance is the adopt; FMP/Alpha Vantage are fallbacks if coverage is poor.* Coverage
  / consensus quality is a known v0 risk (§6).
- **Universe (v0):** ~3-5 sectors, ~4-8 names each, chosen for clear competitor/peer relationships
  (e.g. semis, big-box retail, airlines, soft drinks). Hand-listed with sector + peer-edge metadata.

## 4. Components (build order)

1. `config/universe/path_a_v0.{yaml,py}` — the curated universe + peer-edge list (the relation graph).
2. **Earnings dataset builder** — for each (ticker, earnings date): SUE merit + the event timestamp.
   Reuse the episode-loader pattern (`src/qts/data/real_episode.py`) where possible.
3. **Reaction labels** — event-study **abnormal returns**: for the named firm A and each peer B, the
   cumulative abnormal return over a window [t+0, t+K] (K≈1-3 trading days), abnormal = raw − sector/
   market expectation. These are the prediction targets.
4. **Correlational baseline** — `r̂_B = β_{B,A} · r_A` (β from pre-event history), the bar to beat —
   the real-data analogue of `src/qts/propagation/baselines.py::CorrelationalBaseline`.
5. **Propagation model** — adapt the 1-hop graph to real features (peer node features = sector + a few
   fundamentals/price factors). v0 may even start with a learned-per-edge transfer coefficient before
   the full feature-conditioned bilinear; keep it minimal.
6. **Feasibility gate** — graph beats correlational baseline on the **peer (B)** abnormal reaction,
   out-of-sample (held-out events and/or held-out peer pairs). Mirror `evaluate_feasibility`.

## 5. The crux: a *fair* (non-correlational) test

When A beats earnings, peer B can move **up** (sector read-through — pure common factor, which the
correlational baseline already captures) or **down** (competitive share-shift — the *causal* effect
the thesis is about, which is orthogonal to / opposite the correlation). The graph must beat the
baseline on the **residual, non-correlational** component of B's reaction. This is the real-data
analogue of the sim's "substitute decorrelated from named" confound. **If peer reactions are purely
sector read-through, there is no alpha here — and that itself is the feasibility answer.**

## 6. Risks / open items

- **Earnings-consensus data quality** (yfinance coverage/accuracy) — the biggest v0 risk; validate on a
  few names by hand before scaling.
- **Reaction-window choice** (K) and **abnormal-return model** (market vs sector adjustment) — pick one,
  document it; don't sweep yet.
- **Small N** — a curated universe yields few events; may need a multi-year earnings history for power.
- **Look-ahead** — consensus/surprise must be point-in-time (known at the event), not restated.

## 7. Out of scope (v0)

- LLM-extracted merit (the next layer once propagation is shown on clean merit).
- Multi-hop / n-hop (banked, §14; revisit with an NBFNet-style operator).
- Learned relation graph, supply-chain links, live trading, position sizing / PnL.

## 8. Success criterion

On held-out earnings events (and/or held-out peer pairs), the propagation model's MSE on the **peer's
abnormal reaction** is **below the correlational baseline's**, robustly. PASS → the economic-link
propagation alpha survives contact with real data; proceed to the LLM-merit layer. FAIL (peer moves are
pure read-through) → honest negative, documented like the sim findings.
