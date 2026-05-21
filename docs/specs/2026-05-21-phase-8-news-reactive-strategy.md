# Phase 8 v2 — News-Reactive Strategy + Real-Data Acceptance

**Status**: spec draft
**Date**: 2026-05-21
**Predecessor**: `docs/specs/2026-05-20-phase-8-world-simulator.md` (v1 shipped — multi-agent sim infrastructure)
**Companion log**: `.grill/phase-8-news-reactive-strategy.md`

## Why

Phase 8 v1 shipped a news-reactive **simulator**. v2 ships a news-reactive **strategy** and validates it against **real** historical market data.

The alpha hypothesis from the original Phase 8 grill is: *the strategy decodes news better than the crowd*. v1 enforced the asymmetry (anons use VADER + keyword regex; strategy uses Qwen) but didn't actually build the Qwen-driven strategy. v2 builds it and proves — on real BTC bars + real Powell text — that it adds value over buy-and-hold.

This is the bridge between sim infrastructure and live deployment. Live Binance, bulk historical sweeps, and Optuna tuning are explicitly deferred until v2 proves the strategy has edge.

## What v2 ships

A single coherent slice with one binary acceptance test: **NewsReactiveMomentum day-end equity > buy-and-hold day-end equity on real 2023-12-13 BTC bars + real Powell text**.

### Concrete pieces

- A new strategy `NewsReactiveMomentum` that wraps `MomentumStrategy` and adds a Qwen-driven multi-axis belief state.
- A formal `Strategy.on_text(event)` protocol method (with no-op default) replacing the duck-typed forward from v1.
- A Qwen-backed news classifier producing `NewsSignal(direction, confidence, relevance, magnitude)`.
- A `RealEpisode` type that mirrors `SimulatedEpisode` but carries real market data.
- A one-off data fetcher script + a curated dataset for 2023-12-13 committed to the repo.
- An integration test asserting strategy beats buy-and-hold on the curated day.

### Out of scope (explicit, per the grill)

- Live Binance integration (separate slice).
- Multiple historical FOMC days / bulk-data pipeline.
- Other event types (CPI, NFP, geopolitical, USDT depeg).
- Optuna sweeps over the new strategy's params.
- Multimodal (audio/video).
- Removing the spot-only constraint.

## Architecture

### Package layout

```
src/qts/
├── strategies/
│   ├── base.py              # MODIFIED — add on_text(event) to Strategy protocol with no-op default
│   ├── news_reactive.py     # NEW — NewsReactiveMomentum
│   └── ...                  # existing strategies inherit the no-op default
├── macro/
│   └── news_classifier.py   # NEW — Qwen-backed multi-axis news classifier with disk cache
├── data/
│   ├── __init__.py          # NEW package
│   └── real_episode.py      # NEW — RealEpisode wrapping real bars + text events
└── nautilus/
    ├── actor.py             # MODIFIED — QTSStrategy.on_text_event uses the formal protocol (drops duck-typing)
    └── real_runner.py       # NEW — run_real_backtest dispatches TextEvents at real timestamps

data/real/fomc/2023-12-13/
├── bars.csv                 # NEW — real BTCUSDT 1m bars (Binance kline export)
├── statement.txt            # NEW — FOMC statement text
└── press_conf.json          # NEW — Powell press conference paragraphs with timestamps

scripts/
└── fetch_fomc_data.py       # NEW — one-off fetcher (Binance klines + fed.gov)

tests/
├── unit/
│   ├── test_news_classifier.py        # NEW — Qwen classifier behaviour + caching
│   ├── test_news_reactive.py          # NEW — NewsReactiveMomentum belief state + decay
│   └── test_real_episode.py           # NEW — RealEpisode loader/structure
└── integration/
    └── test_news_reactive_2023_12_13.py   # NEW — acceptance test (beats buy-and-hold)
```

### Data model

```python
# qts/macro/news_classifier.py
@dataclass(frozen=True, slots=True)
class NewsSignal:
    direction: Literal["bull", "bear", "neutral"]
    confidence: float   # [0, 1]
    relevance: float    # [0, 1] — does this text matter for BTC?
    magnitude: float    # [0, 1] — how big a move does this imply?

class NewsClassifier:
    """Qwen-backed multi-axis classifier with on-disk content-hash cache."""

    def __init__(self, llm_client: LLMClient, cache_dir: Path) -> None: ...

    async def classify(self, event: TextEvent) -> NewsSignal: ...
        # 1. content_hash = sha256(event.text + event.source).hexdigest()
        # 2. if cache_dir / f"{content_hash}.json" exists: load and return
        # 3. else: call llm_client.query_json(...) with the structured-output prompt
        # 4. validate the response shape; persist; return
```

```python
# qts/strategies/news_reactive.py
@dataclass
class BeliefAxis:
    """One axis of the multi-axis sentiment belief — decays exponentially with time."""

    value: float       # most recent reading
    last_update: datetime
    half_life: timedelta

    def at(self, now: datetime) -> float:
        """Return the decayed value at `now`."""
        elapsed = (now - self.last_update).total_seconds()
        half_life_s = self.half_life.total_seconds()
        decay = 0.5 ** (elapsed / half_life_s)
        return self.value * decay

    def update(self, new_value: float, now: datetime) -> None:
        self.value = new_value
        self.last_update = now


class NewsReactiveMomentum:
    """MomentumStrategy + Qwen-driven news belief state.

    Composes MomentumStrategy: forwards on_bar, on_fill to inner.
    Adds on_text: each TextEvent updates a multi-axis BeliefAxis stack.
    On every on_bar, blends decayed belief into MomentumStrategy.sentiment_signal_input
    via a configurable weight.
    """

    def __init__(
        self,
        inner: MomentumStrategy,
        classifier: NewsClassifier,
        belief_half_life: timedelta = timedelta(hours=4),
        news_signal_weight: float = 0.5,   # how much the news belief biases the momentum sentiment signal
    ) -> None: ...

    def on_bar(self, bar, snapshot, positions) -> list[Order]: ...
    def on_fill(self, fill) -> None: ...
    def on_text(self, event: TextEvent) -> None: ...
```

```python
# qts/data/real_episode.py
@dataclass
class RealEpisode:
    """Real-data analogue of SimulatedEpisode.

    Same wrapping pattern (.terrain holds the MarketTerrain) so the existing
    run_terrain_backtest pipeline consumes it unchanged. Adds .text_events
    for the news classifier to chew on.
    """

    terrain: MarketTerrain
    text_events: list[TextEvent]
    name: str
    source: str   # "fomc:2023-12-13" or similar

    @classmethod
    def from_disk(cls, root: Path) -> RealEpisode:
        """Load bars.csv + statement.txt + press_conf.json from root."""
        ...
```

### Strategy protocol extension

```python
# qts/strategies/base.py
class Strategy(Protocol):
    def on_bar(self, bar: Bar, snapshot: SignalSnapshot, positions: list[Position]) -> list[Order]: ...
    def on_fill(self, fill: Fill) -> None: ...
    def on_text(self, event: TextEvent) -> None: ...   # FORMAL NOW, default no-op
```

Default implementation in `BaseStrategy.on_text` is `pass`. Existing strategies (`MomentumStrategy`, `MeanReversionStrategy`, `SMACrossoverStrategy`) inherit unchanged.

`QTSStrategy.on_text_event` (the Nautilus actor) stops checking `hasattr(strategy, "on_text")` — the protocol guarantees the method exists. Cleaner and slightly faster.

### Real-data runner

```python
# qts/nautilus/real_runner.py
def run_real_backtest(
    episode: RealEpisode,
    strategy: Strategy,
    venue_config: VenueConfig | None = None,
) -> BacktestResult:
    """Variant of run_terrain_backtest that also dispatches TextEvents.

    Identical bar-handling path to run_terrain_backtest (reuses the same actor).
    On top: walks episode.text_events in chronological order, and at each tick
    (or each bar boundary) forwards events with timestamp <= now to
    QTSStrategy.on_text_event.

    Implementation note: piggy-back on TextEventInjector from qts.world.text_injector
    if the cursor pattern fits; otherwise inline.
    """
```

### Qwen prompt + structured output

The classifier prompt is hand-tuned for v2; Optuna-tuning is deferred. Sketch:

```
SYSTEM:
You are an expert macro analyst. You read FOMC statements, Fed speeches, and
news headlines, and classify their likely impact on Bitcoin (BTC) price action.
Respond ONLY with a JSON object matching this schema, no extra text:

  {
    "direction": "bull" | "bear" | "neutral",
    "confidence": <float 0-1, how confident you are in the direction>,
    "relevance": <float 0-1, how relevant this text is to BTC price>,
    "magnitude": <float 0-1, how big a price move this implies>
  }

USER:
Source: {event.source}
Timestamp: {event.timestamp}
Text:
"""
{event.text}
"""
```

Validation: the classifier asserts the response parses as JSON, fields are present, types are correct, ranges are in bounds. On malformed output, log warning and return `NewsSignal("neutral", 0.0, 0.0, 0.0)` (treat as no signal).

### Belief → momentum signal blend

`MomentumStrategy.sentiment_signal_input` is currently driven by an external sentiment value (existing implementation, see `src/qts/signals/`). NewsReactiveMomentum overrides this input each bar:

```
sentiment_input_blended = (1 - news_signal_weight) * sentiment_input_base
                        + news_signal_weight * (direction_sign × confidence × magnitude × relevance × belief_decay)
```

Where `direction_sign` is +1 for bull, -1 for bear, 0 for neutral. `belief_decay` is computed per-call using `BeliefAxis.at(now)`.

This keeps MomentumStrategy's decision logic untouched — NewsReactiveMomentum just substitutes the sentiment input. Vanilla MomentumStrategy still works with its existing input source.

## Acceptance criteria for v2

1. `pytest tests/integration/test_news_reactive_2023_12_13.py` passes:
   - Loads `RealEpisode.from_disk(Path("data/real/fomc/2023-12-13"))`.
   - Constructs `NewsReactiveMomentum` with an Ollama-backed `NewsClassifier`.
   - Runs `run_real_backtest(episode, strategy)`.
   - Computes `strategy_pnl = result.equity_curve[-1] - result.equity_curve[0]`.
   - Computes `buy_and_hold_pnl = bars[-1].close - bars[0].close` (scaled to same initial capital).
   - **Asserts `strategy_pnl > buy_and_hold_pnl`** (strict greater-than; ties fail).

2. `pytest tests/unit/` passes — no regressions in the 1121-test baseline.

3. `NewsReactiveMomentum` is reproducible under the same seed + cached Qwen classifier outputs (cache hits, no live LLM calls during repeated test runs).

4. Vanilla `MomentumStrategy` continues to work unchanged — its existing tests pass.

5. The Qwen classifier produces valid `NewsSignal` outputs for every text event in the curated dataset (validated via `test_news_classifier.py`).

6. The Strategy protocol's new `on_text` method has a no-op default that existing text-blind strategies inherit transparently.

Out of scope for v2 acceptance: PnL on any day other than 2023-12-13. Stat-sig portfolio-level results.

## Plan for scaling beyond v2

(Documented here so the path is preserved; user flagged all three as important for future grilling.)

- **v2.5 — Optuna over news params.** Belief half-life, news_signal_weight, Qwen prompt variants, direction thresholds. Search space: ~5-10 dims. Requires SQLite-backed study to persist across runs.
- **v3 — Bulk historical FOMC days.** ~20 FOMC meetings since 2021. Fetch infrastructure, batch classifier with cache hits, stat-sig portfolio test (mean strategy PnL > mean buy-and-hold PnL across the set with p < 0.05).
- **v3.5 — Other event types.** CPI prints, NFP, Trump/Senator tweets, geopolitical headlines. Each needs its own corpus + relevance calibration.
- **v4 — Live Binance integration.** Scrapling for real-time Powell statements + Fed RSS + Twitter. Binance testnet for orders. Live `Strategy.on_text` path mirroring the v2 architecture.

## Risks

- **Qwen output quality on real Powell text.** Hand-tuned v1 corpus statements are simple; real FOMC language is hedge-laden ("we will continue to assess"). Qwen may under-classify confidence. **Mitigation:** hand-validate classifier on the curated dataset before running acceptance; tune the system prompt iteratively until classifications look right.
- **Beating buy-and-hold on a +7% day is a hard bar.** Spot-only + intraday timing must add value on top of market beta. **Mitigation:** if missed, investigate whether the gap is timing (entered late), sizing (took small position), or strategy logic (got blocked by momentum filter). Failure to beat hold is a **valid** result that informs the next slice — it doesn't fail v2 architecture, just suggests Qwen-grade sentiment alone isn't enough.
- **Data fetcher fragility.** fed.gov URLs and Binance API surfaces change over time. **Mitigation:** commit the fetched data into the repo. Fetcher is one-off; if it breaks in 6 months, fix it then.
- **Reproducibility under cached Qwen calls.** First run populates the cache; subsequent runs should hit cache and produce identical results. **Mitigation:** cache key is content-hash + prompt-version; bump prompt version invalidates cache deliberately. Acceptance test runs against pre-warmed cache.
- **Strategy protocol change is reversible but visible.** Adding `on_text` to the protocol is a one-way door for downstream consumers. **Mitigation:** no-op default means existing strategies don't break; consumers opt in.

## File-level impact estimate

- New: ~6-8 files (~1200-1500 LOC including tests)
- Modified: 2 files (`strategies/base.py`, `nautilus/actor.py`)
- New data: 3 files in `data/real/fomc/2023-12-13/` (~hundreds of KB)
- New scripts: 1 (`scripts/fetch_fomc_data.py`)

Total: ~1.5-2 weeks of focused work. Mid-effort, contained blast radius.
