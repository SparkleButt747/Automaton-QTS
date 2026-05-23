# Spec: Path A v2 — Real-Equity n-hop Meta-Propagation Feasibility Cut

**Status:** spec (design 2026-05-23; awaiting confirmation). **Supersedes**
`docs/specs/2026-05-23-path-a-equity-feasibility.md` (the 1-hop-only v1, written before the n-hop
breakthrough). First real-data test of the **meta-learned** propagation mechanism (design doc
`docs/research/2026-05-22-event-propagation-graph-design.md` §15).

## 1. Goal

Test, on **real US-equity earnings events**, whether the **meta-trained propagation operator**
(`qts.propagation.meta`) predicts the reaction of an *unnamed* economically-linked peer better than a
correlational baseline — **1-hop first (the feasibility GATE), then 2-hop (a documented extension)**.
Each economic link is a meta-training episode; the **feature-conditioned** operator generalises to
**held-out links** (zero-shot) and improves with **per-link history** (few-shot, `few_shot_adapt`).

## 2. Locked decisions (grilling, 2026-05-23)

| Decision | Choice | Why |
|----------|--------|-----|
| Diversity axis | **Each economic link = an episode**; one shared operator trained across all links | matches the sim's relation-diversity finding (design §15); the single real universe has hundreds of distinct links ≫ the ~10-link floor |
| Hops | **1-hop meta-transfer = GATE; 2-hop = extension** (built only if the gate passes) | de-risk the synthetic→real leap before stacking composition risk |
| Links source | **FNSPID news co-mention graph**, LLM-filtered | free, point-in-time (article-dated), reuses the Qwen pipeline; free peer APIs (Finnhub/FMP) are *correlational* sector peers the baseline already captures |
| Merit | **standardised earnings surprise (SUE)** | clean quant merit; isolates *propagation* from LLM-extraction noise (LLM merit is the next layer) |
| Model | **`MetaPropagationGraph` + `train_meta` + `few_shot_adapt`** | the committed, validated n-hop operator (design §15) |
| Test | beat the correlational baseline on the peer's **non-correlational** reaction | identical bar to the sim gate (§6) |

## 3. The real-data mapping (the key conceptual move)

In the sim, relation-diversity came from many distinct **worlds**. On real equities there is **one
market universe** with many nodes (firms) and many links. The diversity the mechanism needs comes from
the **hundreds of distinct co-mention links within that single universe**. Transfer works because the
operator is **feature-conditioned, not entity-indexed**: it scores edges as `xi_A^T M xi_B` from company
*features*, so a link it never saw at train time is predictable if its endpoints' features resemble
trained links. **Held-out links/firms are the real-data analog of held-out worlds.**

> **Build note:** the committed `train_meta` resamples synthetic *worlds*. Path A needs a thin
> real-data training loop — one universe graph, event batches across **training** links, **held-out**
> links/firms reserved for transfer eval — reusing `MetaPropagationGraph` + the `evaluate_meta_transfer`
> capture/win logic unchanged. `few_shot_adapt` is reused as-is per link.

## 4. Data

- **Bars:** reuse `src/qts/data/market/alpaca_adapter.py` (US-equity OHLCV).
- **Earnings + SUE:** `yfinance` (`Ticker.get_earnings_dates()` → estimate / reported / surprise%).
  Coverage/consensus quality is a known risk (§9); validate a few names by hand first.
- **Links — FNSPID** ([github.com/Zdong104/FNSPID_Financial_News_Dataset](https://github.com/Zdong104/FNSPID_Financial_News_Dataset)):
  15.7M ticker-tagged articles, ~4,775 S&P 500 firms, 1999–2023. Build co-mention edges
  `(ticker_A, ticker_B, article_date)`; **point-in-time by article date**.
- **Universe (v0):** S&P 500 names ∩ FNSPID coverage ∩ yfinance earnings coverage.

## 5. The link graph (the meta "relations")

1. **Co-mention edges:** from FNSPID, `(A, B)` co-mentioned in an article on date `d`, weighted by
   frequency / recency. Strictly point-in-time (only edges with `d < event_t`).
2. **LLM economic-link filter (Qwen):** classify each candidate co-mention as a genuine **economic
   link** (competitor / supplier / customer / partner) vs **incidental** co-mention, and tag the
   relation type + direction. This is the quality gate that turns noisy co-mention into causal links.
3. **Node features `xi`** (proposed default — the key tuning knob): GICS sector/industry embedding,
   log market cap, a few price factors (market beta, 12-1 momentum, realised vol), optionally a few
   fundamentals. The operator matches relations from these, so they must distinguish link types.
4. **2-hop chains (extension):** `A→B→C` where `A→B` and `B→C` are both filtered economic links and
   **C is not directly co-mentioned with A** — the decorrelated terminal, the real analog of the sim's
   2-hop terminal.

## 6. The crux: a *fair* (non-correlational) test  *(unchanged from v1 §5)*

When A beats earnings, peer B can move **up** (sector read-through — common factor, which the
correlational baseline already captures) or **down** (competitive share-shift — the *causal* effect,
orthogonal to / opposite the correlation). The operator must beat the baseline on the **residual,
non-correlational** component of B's reaction. **If peer reactions are pure sector read-through, there
is no alpha here — and that itself is the feasibility answer.**

## 7. Reaction labels

- Event = A's earnings announcement at `t`; merit = SUE.
- Target = **abnormal return** of peer B (and, for the extension, 2-hop terminal C) over `[t, t+K]`,
  `K ≈ 1–3` trading days; abnormal = raw − market/sector expectation (factor-residualised).
- **Point-in-time:** only co-mention edges and price history with timestamp `< t`; SUE must be the
  point-in-time surprise (not restated).

## 8. Components (build order)

1. **FNSPID ingestion + co-mention edge builder** (point-in-time, weighted).
2. **LLM economic-link filter** (Qwen classify+type each candidate edge) — reuse `NewsClassifier` /
   `LlamaCppClient`.
3. **Universe + node-feature builder** (`config/universe/path_a_v2.{yaml,py}`).
4. **Earnings/SUE dataset** (yfinance) — reuse the episode-loader pattern where possible.
5. **Reaction labels** — abnormal-return event study (named A + each peer B / terminal C).
6. **Correlational baseline** — `r̂_B = β_{B,A} · r_A` (β from pre-event history); real-data analog of
   `propagation/baselines.py::CorrelationalBaseline`.
7. **Real-data meta-training loop** — train `MetaPropagationGraph` over training-link events; held-out
   links/firms for transfer; `few_shot_adapt` per link with history.
8. **Feasibility gate** — `evaluate_meta_transfer`-style: operator beats correlational on **held-out
   links' peer (B)** abnormal reaction, zero-shot, + a few-shot value-of-data curve. **2-hop extension**
   = same on terminal C.

## 9. Risks / open items

- **Co-mention noise** — incidental co-mentions pollute the graph; the LLM filter (§5.2) quality is the
  pivotal v0 risk. Validate filtered edges by hand on a few names.
- **Earnings-consensus quality** (yfinance) — validate before scaling.
- **Look-ahead** — co-mention edges, price history, and SUE all strictly point-in-time.
- **Few events per link** — over 1999–2023 a given link fires on few earnings events → the *few-shot*
  regime is data-thin per link; lean on **cross-link meta-training** (zero-shot) as the primary gate.
- **The crux risk** — peer moves may be pure read-through (no alpha) → honest negative, documented like
  the sim findings.

## 10. Out of scope (v0)

- LLM-extracted *merit* (next layer once propagation is shown on clean SUE merit).
- 3-hop+ (synthetic work shows it is marginal zero-shot; revisit only with strong few-shot data).
- Live trading, position sizing, PnL.

## 11. Success criterion

- **GATE (1-hop):** on held-out links/events, the meta-operator's MSE on the **peer's abnormal
  reaction** is **below the correlational baseline's**, robustly; `few_shot_adapt` lifts capture
  (the value-of-data curve on real links). PASS → the economic-link propagation alpha survives real
  data; proceed to the 2-hop extension, then the LLM-merit layer.
- **2-hop extension:** same bar on the terminal C.
- **FAIL** (peer moves are pure read-through) → honest negative, documented like the sim findings.
