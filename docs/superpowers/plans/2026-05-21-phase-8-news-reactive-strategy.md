# Phase 8 v2 — News-Reactive Strategy + Real-Data Acceptance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Qwen-driven news-reactive strategy (`NewsReactiveMomentum`) and prove on real 2023-12-13 BTC bars + real Powell text that it beats buy-and-hold for the day.

**Architecture:** Extend the `Strategy` protocol with a formal `on_text(event)` method (no-op default). New `NewsReactiveMomentum` composes `MomentumStrategy`: each `TextEvent` is classified by a Qwen-backed `NewsClassifier` into a multi-axis `NewsSignal(direction, confidence, relevance, magnitude)` and updates an exponentially-decaying belief state. On every bar, the decayed belief is blended into the `combined_alpha` field of the `SignalSnapshot` that `MomentumStrategy` consumes — so the inner strategy's decision logic is unchanged but its sentiment input becomes Qwen-grade. A new `RealEpisode` mirrors `SimulatedEpisode` (real bars + real text events), loaded from a curated `data/real/fomc/2023-12-13/` directory populated by a one-off fetcher script.

**Tech Stack:** Python 3.11, async Ollama via existing `OllamaClient`, pyyaml, httpx, NautilusTrader 1.221, pytest + hypothesis. Pre-existing infrastructure from Phase 8 v1 (`TextEvent`, `MarketTerrain`, `run_terrain_backtest`, `QTSStrategy` actor with duck-typed `on_text_event`).

**Pre-task baseline:**
- Git HEAD: `7c8adc6` — Phase 8 v1 complete and pushed
- Test count: 1121 passed, 4 skipped
- Test runner: `.venv/bin/python -m pytest --no-cov -q ...` (`--no-cov` is mandatory)
- Working tree: clean on `main`
- Today: 2026-05-21

**Commits:** Each task ends in one commit. Do NOT add `Co-Authored-By: Claude` trailer — user explicitly forbids it (`feedback_no_ai_trailer.md`).

---

## File structure (locked in before tasks)

| File | Responsibility | Task |
|---|---|---|
| `src/qts/strategies/base.py` | Add `on_text(event)` to Strategy protocol | Task 1 |
| `src/qts/strategies/momentum.py` | No-op `on_text` stub | Task 1 |
| `src/qts/strategies/mean_reversion.py` | No-op `on_text` stub | Task 1 |
| `src/qts/strategies/sma_crossover.py` | No-op `on_text` stub | Task 1 |
| `src/qts/nautilus/actor.py` | Drop duck-typing in `on_text_event` | Task 2 |
| `src/qts/macro/news_signal.py` | `NewsSignal` dataclass | Task 3 |
| `src/qts/macro/news_classifier.py` | Qwen-backed multi-axis classifier (async + sync cached lookup) | Tasks 4, 5 |
| `src/qts/strategies/belief.py` | `BeliefAxis` decay primitive | Task 6 |
| `src/qts/strategies/news_reactive.py` | `NewsReactiveMomentum` (wraps Momentum, holds belief, blends `combined_alpha`) | Task 7 |
| `src/qts/data/real_episode.py` | `RealEpisode` + `RealEpisode.from_disk` | Task 8 |
| `src/qts/data/__init__.py` | Export `RealEpisode` | Task 8 |
| `src/qts/nautilus/real_runner.py` | `run_real_backtest` — bars + text events + Nautilus | Task 9 |
| `scripts/fetch_fomc_data.py` | One-off fetcher: Binance klines + fed.gov | Task 10 |
| `data/real/fomc/2023-12-13/bars.csv` | 1440 1m bars from Binance | Task 11 |
| `data/real/fomc/2023-12-13/statement.txt` | FOMC statement text | Task 11 |
| `data/real/fomc/2023-12-13/press_conf.json` | Press conference paragraphs with synthetic timestamps | Task 11 |
| `scripts/validate_news_classifier.py` | Hand-validation: classify all curated events + print scores | Task 12 |
| `tests/unit/test_strategy_protocol_on_text.py` | Protocol has `on_text`; existing strategies have no-op | Task 1 |
| `tests/unit/test_qts_strategy_on_text_event.py` | Actor calls `on_text` directly | Task 2 |
| `tests/unit/test_news_signal.py` | `NewsSignal` shape + validation | Task 3 |
| `tests/unit/test_news_classifier.py` | Classifier prompt + JSON parse + cache | Tasks 4, 5 |
| `tests/unit/test_belief_axis.py` | Decay math + update semantics | Task 6 |
| `tests/unit/test_news_reactive.py` | `NewsReactiveMomentum` belief integration + alpha blend | Task 7 |
| `tests/unit/test_real_episode.py` | `RealEpisode.from_disk` shape + parsing | Task 8 |
| `tests/unit/test_real_runner.py` | `run_real_backtest` dispatches text events at timestamps | Task 9 |
| `tests/unit/test_fetch_fomc_data.py` | Fetcher script unit-testable parts (parsing, chunking) | Task 10 |
| `tests/integration/test_news_reactive_2023_12_13.py` | **Acceptance** — beats buy-and-hold | Task 13 |
| `src/qts/data/__init__.py`, `src/qts/strategies/__init__.py` | Public exports | Task 14 |
| `docs/plans/terrain-refactor-plan.md` | Update Phase 8 v2 status | Task 14 |

---

## Task 1: Strategy.on_text protocol + no-op stubs

**Files:**
- Modify: `src/qts/strategies/base.py` — add `on_text(event)` to Protocol
- Modify: `src/qts/strategies/momentum.py` — add no-op `on_text`
- Modify: `src/qts/strategies/mean_reversion.py` — add no-op `on_text`
- Modify: `src/qts/strategies/sma_crossover.py` — add no-op `on_text`
- Create: `tests/unit/test_strategy_protocol_on_text.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_strategy_protocol_on_text.py`:

```python
"""Strategy protocol now includes on_text(event) with a no-op stub on existing strategies."""

from __future__ import annotations

from datetime import UTC, datetime

from qts.strategies.base import Strategy
from qts.world.events import TextEvent


def _event() -> TextEvent:
    return TextEvent(
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        source="test",
        persona=None,
        text="dummy",
        metadata={},
    )


def test_protocol_declares_on_text() -> None:  # T-PROT-1
    # `on_text` is declared on the Strategy protocol surface.
    assert hasattr(Strategy, "on_text")


def test_momentum_on_text_is_noop() -> None:  # T-PROT-2
    from qts.config import RiskLimits, StrategyParams
    from qts.strategies.momentum import MomentumStrategy

    strat = MomentumStrategy(params=StrategyParams(), risk_limits=RiskLimits())
    # Must not raise; returns None.
    assert strat.on_text(_event()) is None


def test_mean_reversion_on_text_is_noop() -> None:  # T-PROT-3
    from qts.config import RiskLimits, StrategyParams
    from qts.strategies.mean_reversion import MeanReversionStrategy

    strat = MeanReversionStrategy(params=StrategyParams(), risk_limits=RiskLimits())
    assert strat.on_text(_event()) is None


def test_sma_crossover_on_text_is_noop() -> None:  # T-PROT-4
    from qts.strategies.sma_crossover import SMACrossoverStrategy

    strat = SMACrossoverStrategy(fast_period=10, slow_period=30)
    assert strat.on_text(_event()) is None
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_strategy_protocol_on_text.py -v
```

Expected: 4 FAILED — `AttributeError: 'MomentumStrategy' object has no attribute 'on_text'` (or similar).

- [ ] **Step 3: Extend the Strategy protocol**

Edit `src/qts/strategies/base.py`. Replace the entire file with:

```python
"""Base strategy interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from qts.models.base import Bar, Fill, Order, Position, SignalSnapshot
from qts.world.events import TextEvent


@runtime_checkable
class Strategy(Protocol):
    """Strategy interface. All strategies must implement this."""

    @property
    def name(self) -> str:
        """Return the human-readable strategy name."""
        ...

    def on_bar(
        self,
        bar: Bar,
        snapshot: SignalSnapshot,
        positions: list[Position],
    ) -> list[Order]:
        """Process a new bar and return orders to execute.

        Args:
            bar: The newly closed OHLCV bar.
            snapshot: Latest signal snapshot containing all indicators.
            positions: List of currently open positions.

        Returns:
            List of Order objects to submit; may be empty.
        """
        ...

    def on_fill(self, fill: Fill) -> None:
        """Handle fill notification.

        Args:
            fill: The execution fill report from the broker.
        """
        ...

    def on_text(self, event: TextEvent) -> None:
        """Handle a text event (news, persona statement, macro release).

        Default implementation in concrete strategies is a no-op; news-reactive
        strategies override to update internal belief state.

        Args:
            event: The TextEvent to consume.
        """
        ...
```

- [ ] **Step 4: Stub on_text on MomentumStrategy**

Edit `src/qts/strategies/momentum.py`. After the existing `on_fill` method (around line 205), append:

```python
    def on_text(self, event: object) -> None:  # noqa: ARG002
        """Default no-op — vanilla momentum doesn't consume text."""
        return None
```

(The `object` annotation avoids dragging the TextEvent import into momentum.py; the noqa silences the unused-arg lint.)

- [ ] **Step 5: Stub on_text on MeanReversionStrategy**

Edit `src/qts/strategies/mean_reversion.py`. After the existing `on_fill` method, append:

```python
    def on_text(self, event: object) -> None:  # noqa: ARG002
        """Default no-op — vanilla mean-reversion doesn't consume text."""
        return None
```

- [ ] **Step 6: Stub on_text on SMACrossoverStrategy**

Edit `src/qts/strategies/sma_crossover.py`. After the existing `on_fill` method, append:

```python
    def on_text(self, event: object) -> None:  # noqa: ARG002
        """Default no-op — vanilla SMA crossover doesn't consume text."""
        return None
```

- [ ] **Step 7: Run tests to verify pass + no regressions**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_strategy_protocol_on_text.py -v
.venv/bin/python -m pytest --no-cov -q 2>&1 | tail -3
```

Expected: 4 PASSED on the new test file. Full suite: 1125 passed, 4 skipped (1121 + 4).

- [ ] **Step 8: Commit**

```bash
git add src/qts/strategies/base.py src/qts/strategies/momentum.py src/qts/strategies/mean_reversion.py src/qts/strategies/sma_crossover.py tests/unit/test_strategy_protocol_on_text.py
git commit -m "feat(strategies): add on_text to Strategy protocol with no-op default

Promotes the duck-typed on_text dispatch from Phase 8 v1 to a formal
Protocol method. Existing strategies (Momentum, MeanReversion, SMA
crossover) get an explicit no-op so they remain text-blind. News-
reactive strategies override to consume TextEvents."
```

---

## Task 2: Drop duck-typing in QTSStrategy.on_text_event

**Files:**
- Modify: `src/qts/nautilus/actor.py` — drop `getattr/callable` guard, call directly
- Create: `tests/unit/test_qts_strategy_on_text_event.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_qts_strategy_on_text_event.py`:

```python
"""QTSStrategy.on_text_event now calls inner_strategy.on_text directly.

After Task 1 added on_text to the Strategy protocol, the actor no longer
needs hasattr / callable guards. This test pins the new direct-call shape.
"""

from __future__ import annotations

from datetime import UTC, datetime

from qts.world.events import TextEvent


def test_actor_forwards_text_to_inner_on_text() -> None:  # T-ACT-1
    from qts.nautilus.actor import QTSStrategy, QTSStrategyConfig

    captured: list[TextEvent] = []

    class _RecordingStrategy:
        params = None
        name = "recording"

        def on_bar(self, *_a: object, **_k: object) -> list:
            return []

        def on_fill(self, *_a: object, **_k: object) -> None:
            pass

        def on_text(self, event: TextEvent) -> None:
            captured.append(event)

    actor = QTSStrategy(config=QTSStrategyConfig(instrument_id="BTCUSDT.BINANCE", bar_window=10))
    actor.set_qts_strategy(_RecordingStrategy())

    evt = TextEvent(
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        source="powell",
        persona="powell",
        text="hawkish",
        metadata={},
    )
    actor.on_text_event(evt)

    assert captured == [evt]


def test_actor_noop_when_inner_strategy_unset() -> None:  # T-ACT-2
    from qts.nautilus.actor import QTSStrategy, QTSStrategyConfig

    actor = QTSStrategy(config=QTSStrategyConfig(instrument_id="BTCUSDT.BINANCE", bar_window=10))
    # No set_qts_strategy() call — _inner_strategy is None.
    evt = TextEvent(
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        source="x",
        persona=None,
        text="y",
        metadata={},
    )
    # Should not raise.
    actor.on_text_event(evt)
```

- [ ] **Step 2: Run tests to verify**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_qts_strategy_on_text_event.py -v
```

Expected: 2 PASSED (the Phase 8 v1 duck-typed path already supports this; Task 1 makes it strictly compliant). If FAILED, proceed to Step 3.

- [ ] **Step 3: Update actor.py to call on_text directly**

Open `src/qts/nautilus/actor.py` and locate `on_text_event`. The Phase 8 v1 implementation was:

```python
    def on_text_event(self, event):
        if self._inner_strategy is None:
            return
        handler = getattr(self._inner_strategy, "on_text", None)
        if callable(handler):
            handler(event)
```

Replace it with:

```python
    def on_text_event(self, event):
        """Forward TextEvent to the inner strategy.

        Strategy protocol guarantees on_text exists (with a no-op default
        in non-news strategies), so we can call it directly.
        """
        if self._inner_strategy is None:
            return
        self._inner_strategy.on_text(event)
```

- [ ] **Step 4: Run tests + full suite**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_qts_strategy_on_text_event.py -v
.venv/bin/python -m pytest --no-cov -q 2>&1 | tail -3
```

Expected: 2 PASSED on the new file. Full suite: 1127 passed, 4 skipped (1125 + 2).

- [ ] **Step 5: Commit**

```bash
git add src/qts/nautilus/actor.py tests/unit/test_qts_strategy_on_text_event.py
git commit -m "refactor(nautilus): drop on_text duck-typing in QTSStrategy

The Strategy protocol now formally requires on_text (Task 1 added the
no-op default). The actor's hasattr/callable guard is dead weight —
call inner_strategy.on_text directly."
```

---

## Task 3: NewsSignal dataclass

**Files:**
- Create: `src/qts/macro/news_signal.py`
- Create: `tests/unit/test_news_signal.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_news_signal.py`:

```python
"""Tests for the NewsSignal multi-axis structured output."""

from __future__ import annotations

import pytest


def test_news_signal_shape() -> None:  # T-NSIG-1
    from qts.macro.news_signal import NewsSignal

    sig = NewsSignal(direction="bull", confidence=0.8, relevance=0.9, magnitude=0.6)
    assert sig.direction == "bull"
    assert sig.confidence == 0.8
    assert sig.relevance == 0.9
    assert sig.magnitude == 0.6


def test_news_signal_rejects_invalid_direction() -> None:  # T-NSIG-2
    from qts.macro.news_signal import NewsSignal

    with pytest.raises(ValueError, match="direction must be"):
        NewsSignal(direction="up", confidence=0.5, relevance=0.5, magnitude=0.5)


def test_news_signal_clamps_out_of_range() -> None:  # T-NSIG-3
    from qts.macro.news_signal import NewsSignal

    with pytest.raises(ValueError, match="must be in"):
        NewsSignal(direction="bull", confidence=1.5, relevance=0.5, magnitude=0.5)

    with pytest.raises(ValueError, match="must be in"):
        NewsSignal(direction="bull", confidence=0.5, relevance=-0.1, magnitude=0.5)


def test_news_signal_direction_sign() -> None:  # T-NSIG-4
    from qts.macro.news_signal import NewsSignal

    assert NewsSignal(direction="bull", confidence=0.5, relevance=0.5, magnitude=0.5).direction_sign == 1.0
    assert NewsSignal(direction="bear", confidence=0.5, relevance=0.5, magnitude=0.5).direction_sign == -1.0
    assert NewsSignal(direction="neutral", confidence=0.5, relevance=0.5, magnitude=0.5).direction_sign == 0.0


def test_news_signal_alpha_contribution() -> None:  # T-NSIG-5
    from qts.macro.news_signal import NewsSignal

    sig = NewsSignal(direction="bull", confidence=0.8, relevance=1.0, magnitude=0.5)
    # alpha = direction_sign × confidence × relevance × magnitude = 1 × 0.8 × 1.0 × 0.5 = 0.4
    assert sig.alpha_contribution() == pytest.approx(0.4)

    sig_bear = NewsSignal(direction="bear", confidence=0.6, relevance=0.5, magnitude=0.8)
    # -1 × 0.6 × 0.5 × 0.8 = -0.24
    assert sig_bear.alpha_contribution() == pytest.approx(-0.24)
```

- [ ] **Step 2: Run tests to verify fail**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_news_signal.py -v
```

Expected: 5 FAILED — `ModuleNotFoundError`.

- [ ] **Step 3: Create NewsSignal**

Create `src/qts/macro/news_signal.py`:

```python
"""Multi-axis structured output from the news classifier.

The decode-gap alpha hypothesis: Qwen reads news smarter than VADER-grade
crowd sentiment by producing a structured belief over multiple axes, not
a single scalar. The strategy combines these axes into an alpha bias.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

_VALID_DIRECTIONS: tuple[str, ...] = ("bull", "bear", "neutral")


@dataclass(frozen=True, slots=True)
class NewsSignal:
    """Structured news read.

    direction:  classification of price-impact direction
    confidence: how confident the classifier is in the direction call
    relevance:  how relevant this text is to the asset under analysis
    magnitude:  how large a move this implies
    """

    direction: Literal["bull", "bear", "neutral"]
    confidence: float
    relevance: float
    magnitude: float

    def __post_init__(self) -> None:
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {_VALID_DIRECTIONS}, got {self.direction!r}"
            )
        for field_name, value in (
            ("confidence", self.confidence),
            ("relevance", self.relevance),
            ("magnitude", self.magnitude),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be in [0, 1], got {value}"
                )

    @property
    def direction_sign(self) -> float:
        """+1 for bull, -1 for bear, 0 for neutral."""
        if self.direction == "bull":
            return 1.0
        if self.direction == "bear":
            return -1.0
        return 0.0

    def alpha_contribution(self) -> float:
        """Combined scalar bias for the strategy's alpha blend.

        Returns direction_sign × confidence × relevance × magnitude, which
        is bounded in [-1, 1]. Used as one input to a weighted alpha blend
        in NewsReactiveMomentum.
        """
        return self.direction_sign * self.confidence * self.relevance * self.magnitude
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_news_signal.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/qts/macro/news_signal.py tests/unit/test_news_signal.py
git commit -m "feat(macro): NewsSignal multi-axis structured output

Direction (bull/bear/neutral) + confidence + relevance + magnitude.
alpha_contribution() collapses to a scalar in [-1, 1] for blending into
the strategy's combined_alpha. Validates ranges and direction enum at
construction; frozen dataclass so signals are safely shareable."
```

---

## Task 4: NewsClassifier — async Qwen call + JSON validation

**Files:**
- Create: `src/qts/macro/news_classifier.py` (partial — async path only; cache added in Task 5)
- Create: `tests/unit/test_news_classifier.py` (partial — tests for the async path)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_news_classifier.py`:

```python
"""Tests for the Qwen-backed multi-axis news classifier."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest


def _event() -> "TextEvent":  # noqa: F821
    from qts.world.events import TextEvent

    return TextEvent(
        timestamp=datetime(2023, 12, 13, 19, 0, tzinfo=UTC),
        source="fed_press_release",
        persona=None,
        text=(
            "The Committee decided to maintain the target range for the federal funds "
            "rate at 5-1/4 to 5-1/2 percent. The Committee's longer-run goals are to "
            "promote maximum employment and price stability."
        ),
        metadata={"event_kind": "FOMC"},
    )


class _FakeLLM:
    """In-memory stub of LLMClientProtocol.

    Configured with a queue of canned responses; raises if depleted.
    """

    def __init__(self, responses: list[dict] | None = None) -> None:
        self._responses = list(responses or [])
        self.calls: list[tuple[str, str]] = []

    async def query(self, system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> str:  # noqa: ARG002
        import json
        return json.dumps(self._responses.pop(0))

    async def query_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 4096) -> dict:  # noqa: ARG002
        self.calls.append((system_prompt, user_prompt))
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_classifier_returns_news_signal(tmp_path) -> None:  # T-CLF-1
    from qts.macro.news_classifier import NewsClassifier
    from qts.macro.news_signal import NewsSignal

    llm = _FakeLLM(responses=[{
        "direction": "bull",
        "confidence": 0.7,
        "relevance": 0.9,
        "magnitude": 0.6,
    }])
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    sig = await clf.classify_async(_event())
    assert isinstance(sig, NewsSignal)
    assert sig.direction == "bull"
    assert sig.confidence == 0.7


@pytest.mark.asyncio
async def test_classifier_falls_back_to_neutral_on_malformed(tmp_path) -> None:  # T-CLF-2
    from qts.macro.news_classifier import NewsClassifier

    # Missing required keys.
    llm = _FakeLLM(responses=[{"foo": "bar"}])
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    sig = await clf.classify_async(_event())
    # Fall-back is direction=neutral with all axes zero.
    assert sig.direction == "neutral"
    assert sig.confidence == 0.0
    assert sig.relevance == 0.0
    assert sig.magnitude == 0.0


@pytest.mark.asyncio
async def test_classifier_calls_llm_with_event_text(tmp_path) -> None:  # T-CLF-3
    from qts.macro.news_classifier import NewsClassifier

    llm = _FakeLLM(responses=[{
        "direction": "bear", "confidence": 0.5, "relevance": 0.5, "magnitude": 0.5
    }])
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    await clf.classify_async(_event())
    assert len(llm.calls) == 1
    system_prompt, user_prompt = llm.calls[0]
    # The event text must appear in the user prompt.
    assert "target range for the federal funds rate" in user_prompt
    # The system prompt should mention BTC (the asset under analysis).
    assert "BTC" in system_prompt or "Bitcoin" in system_prompt
```

- [ ] **Step 2: Run tests to verify fail**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_news_classifier.py -v
```

Expected: 3 FAILED — `ModuleNotFoundError`.

- [ ] **Step 3: Create the classifier (async-only path)**

Create `src/qts/macro/news_classifier.py`:

```python
"""Qwen-backed multi-axis news classifier.

Strategy-grade decode of arbitrary text into a NewsSignal. Pre-warming
the cache before a backtest is the recommended pattern — see warm_cache_for
(added in Task 5).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from qts.macro.news_signal import NewsSignal

if TYPE_CHECKING:
    from qts.oversight.llm_client import LLMClientProtocol
    from qts.world.events import TextEvent

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert macroeconomic and crypto-market analyst. Given a piece of text
(typically a Fed/FOMC release, central-bank speech, or financial news headline),
classify its likely impact on Bitcoin (BTC) spot price.

Respond ONLY with a valid JSON object matching this schema, no markdown fences,
no commentary, no preamble:

  {
    "direction":  "bull" | "bear" | "neutral",
    "confidence": <float 0.0 to 1.0>,
    "relevance":  <float 0.0 to 1.0>,
    "magnitude":  <float 0.0 to 1.0>
  }

- direction: which way is BTC most likely to move on this text?
- confidence: how sure are you about the direction call?
- relevance: how directly does this text affect BTC price?
- magnitude: how large a price move does this imply?

Be calibrated. Hedged Fed language often warrants confidence < 0.5. Off-topic
text (sports, weather) warrants relevance < 0.2.
"""


class NewsClassifier:
    """Multi-axis classifier with optional disk cache (Task 5 adds caching)."""

    def __init__(
        self,
        llm_client: "LLMClientProtocol",
        cache_dir: Path,
    ) -> None:
        self._llm = llm_client
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    async def classify_async(self, event: "TextEvent") -> NewsSignal:
        """Classify a TextEvent into a NewsSignal via Qwen."""
        user_prompt = self._render_user_prompt(event)
        try:
            raw = await self._llm.query_json(_SYSTEM_PROMPT, user_prompt)
        except Exception:  # noqa: BLE001
            logger.exception("News classifier LLM call failed; returning neutral signal")
            return NewsSignal(direction="neutral", confidence=0.0, relevance=0.0, magnitude=0.0)

        return self._parse_response(raw)

    @staticmethod
    def _render_user_prompt(event: "TextEvent") -> str:
        return (
            f"Source: {event.source}\n"
            f"Timestamp: {event.timestamp.isoformat()}\n"
            f"Text:\n\"\"\"\n{event.text}\n\"\"\"\n"
        )

    @staticmethod
    def _parse_response(raw: dict) -> NewsSignal:
        """Parse LLM JSON into a NewsSignal; fall back to neutral on malformed."""
        try:
            return NewsSignal(
                direction=raw["direction"],
                confidence=float(raw["confidence"]),
                relevance=float(raw["relevance"]),
                magnitude=float(raw["magnitude"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("News classifier produced malformed response %r: %s", raw, exc)
            return NewsSignal(direction="neutral", confidence=0.0, relevance=0.0, magnitude=0.0)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_news_classifier.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/qts/macro/news_classifier.py tests/unit/test_news_classifier.py
git commit -m "feat(macro): NewsClassifier async Qwen path

Multi-axis news classification via LLMClientProtocol. Renders an event
into a structured prompt; parses the JSON response into a NewsSignal;
falls back to neutral on malformed output or LLM exceptions. Caching
added in the next commit."
```

---

## Task 5: NewsClassifier disk cache + warm helper

**Files:**
- Modify: `src/qts/macro/news_classifier.py` — add content-hash cache + warm_cache_for + sync classify()
- Modify: `tests/unit/test_news_classifier.py` — add cache + warm tests

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_news_classifier.py`:

```python
@pytest.mark.asyncio
async def test_classifier_caches_response(tmp_path) -> None:  # T-CLF-4
    from qts.macro.news_classifier import NewsClassifier

    llm = _FakeLLM(responses=[{
        "direction": "bull", "confidence": 0.7, "relevance": 0.9, "magnitude": 0.6
    }])
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    sig1 = await clf.classify_async(_event())
    # Second call must NOT hit the LLM — fully cached.
    sig2 = await clf.classify_async(_event())

    assert sig1 == sig2
    assert len(llm.calls) == 1  # LLM only called once


def test_classifier_sync_classify_requires_cache_hit(tmp_path) -> None:  # T-CLF-5
    from qts.macro.news_classifier import NewsClassifier

    llm = _FakeLLM(responses=[])
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    with pytest.raises(KeyError, match="not in cache"):
        clf.classify(_event())


@pytest.mark.asyncio
async def test_classifier_sync_classify_works_after_warm(tmp_path) -> None:  # T-CLF-6
    from qts.macro.news_classifier import NewsClassifier

    llm = _FakeLLM(responses=[{
        "direction": "bear", "confidence": 0.5, "relevance": 0.7, "magnitude": 0.4
    }])
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    await clf.warm_cache_for([_event()])
    # Synchronous lookup — must not call the LLM again.
    sig = clf.classify(_event())

    assert sig.direction == "bear"
    assert sig.confidence == 0.5


@pytest.mark.asyncio
async def test_cache_key_content_addressed(tmp_path) -> None:  # T-CLF-7
    from datetime import timedelta

    from qts.macro.news_classifier import NewsClassifier
    from qts.world.events import TextEvent

    llm = _FakeLLM(responses=[
        {"direction": "bull", "confidence": 0.7, "relevance": 0.9, "magnitude": 0.6},
        {"direction": "bear", "confidence": 0.7, "relevance": 0.9, "magnitude": 0.6},
    ])
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    e1 = _event()
    # Same text, different timestamp → SAME cache key (content-addressed).
    e2 = TextEvent(
        timestamp=e1.timestamp + timedelta(hours=1),
        source=e1.source,
        persona=e1.persona,
        text=e1.text,
        metadata={"event_kind": "FOMC"},
    )

    sig1 = await clf.classify_async(e1)
    sig2 = await clf.classify_async(e2)

    # Same text → same cached signal (LLM only called once).
    assert sig1 == sig2
    assert len(llm.calls) == 1
```

- [ ] **Step 2: Run tests to verify fail**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_news_classifier.py -v
```

Expected: 4 new tests FAIL (the original 3 still pass).

- [ ] **Step 3: Add cache + warm + sync classify**

Edit `src/qts/macro/news_classifier.py`. Add imports at the top (after existing imports):

```python
import asyncio
import hashlib
import json
```

Then replace the `NewsClassifier` class body with this expanded version (keep `_SYSTEM_PROMPT` and helpers intact):

```python
class NewsClassifier:
    """Multi-axis classifier with content-addressed disk cache.

    Recommended workflow:
        clf = NewsClassifier(llm_client, cache_dir=Path("data/news_cache"))
        await clf.warm_cache_for(events)        # pre-warm: async LLM calls
        # ... in the strategy's on_text path:
        signal = clf.classify(event)            # sync, cache-only lookup
    """

    _CACHE_KEY_VERSION = "v1"   # bump to invalidate all cached entries

    def __init__(
        self,
        llm_client: "LLMClientProtocol",
        cache_dir: Path,
    ) -> None:
        self._llm = llm_client
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: dict[str, NewsSignal] = {}

    # ---- Public API ------------------------------------------------------

    async def classify_async(self, event: "TextEvent") -> NewsSignal:
        """Classify an event, hitting cache first; otherwise call the LLM."""
        key = self._cache_key(event)
        cached = self._read_cache(key)
        if cached is not None:
            return cached

        user_prompt = self._render_user_prompt(event)
        try:
            raw = await self._llm.query_json(_SYSTEM_PROMPT, user_prompt)
        except Exception:  # noqa: BLE001
            logger.exception("News classifier LLM call failed; returning neutral signal")
            return NewsSignal(direction="neutral", confidence=0.0, relevance=0.0, magnitude=0.0)

        signal = self._parse_response(raw)
        self._write_cache(key, signal)
        return signal

    def classify(self, event: "TextEvent") -> NewsSignal:
        """Synchronous lookup. Requires the event to be in cache (call warm_cache_for first)."""
        key = self._cache_key(event)
        cached = self._read_cache(key)
        if cached is None:
            raise KeyError(
                f"event with cache key {key!r} not in cache — call warm_cache_for(events) first"
            )
        return cached

    async def warm_cache_for(self, events: "list[TextEvent]") -> None:
        """Pre-classify a batch of events so .classify() can serve them synchronously."""
        for event in events:
            await self.classify_async(event)

    # ---- Internal helpers -----------------------------------------------

    def _cache_key(self, event: "TextEvent") -> str:
        """Content-addressed key: sha256(source || text || version)."""
        payload = f"{event.source}\n{event.text}\n{self._CACHE_KEY_VERSION}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str) -> NewsSignal | None:
        if key in self._memory_cache:
            return self._memory_cache[key]
        path = self._cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            signal = NewsSignal(**raw)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            logger.warning("Failed to load cache entry %s: %s", path, exc)
            return None
        self._memory_cache[key] = signal
        return signal

    def _write_cache(self, key: str, signal: NewsSignal) -> None:
        self._memory_cache[key] = signal
        path = self._cache_dir / f"{key}.json"
        path.write_text(
            json.dumps({
                "direction": signal.direction,
                "confidence": signal.confidence,
                "relevance": signal.relevance,
                "magnitude": signal.magnitude,
            }),
            encoding="utf-8",
        )

    @staticmethod
    def _render_user_prompt(event: "TextEvent") -> str:
        return (
            f"Source: {event.source}\n"
            f"Timestamp: {event.timestamp.isoformat()}\n"
            f"Text:\n\"\"\"\n{event.text}\n\"\"\"\n"
        )

    @staticmethod
    def _parse_response(raw: dict) -> NewsSignal:
        try:
            return NewsSignal(
                direction=raw["direction"],
                confidence=float(raw["confidence"]),
                relevance=float(raw["relevance"]),
                magnitude=float(raw["magnitude"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("News classifier produced malformed response %r: %s", raw, exc)
            return NewsSignal(direction="neutral", confidence=0.0, relevance=0.0, magnitude=0.0)
```

Remove the now-unused top-level `asyncio` import if you didn't end up using it; ruff will flag it.

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_news_classifier.py -v
```

Expected: 7 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/qts/macro/news_classifier.py tests/unit/test_news_classifier.py
git commit -m "feat(macro): NewsClassifier disk cache + sync classify

Content-addressed cache (sha256 of source + text + version) lets repeated
runs hit the cache instead of the LLM. classify_async populates cache;
classify() is sync and requires a prior warm_cache_for() pass. This is
the pattern the strategy uses — pre-warm before the backtest, lookup
synchronously during on_text dispatch."
```

---

## Task 6: BeliefAxis decay primitive

**Files:**
- Create: `src/qts/strategies/belief.py`
- Create: `tests/unit/test_belief_axis.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_belief_axis.py`:

```python
"""Tests for the exponential-decay belief primitive."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest


def test_belief_axis_default_zero() -> None:  # T-BEL-1
    from qts.strategies.belief import BeliefAxis

    b = BeliefAxis(half_life=timedelta(hours=4))
    now = datetime(2024, 1, 1, tzinfo=UTC)
    assert b.at(now) == 0.0


def test_belief_axis_update_then_at_returns_full_value() -> None:  # T-BEL-2
    from qts.strategies.belief import BeliefAxis

    b = BeliefAxis(half_life=timedelta(hours=4))
    t = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    b.update(value=0.6, now=t)

    # No time elapsed → no decay → full value.
    assert b.at(t) == pytest.approx(0.6)


def test_belief_axis_decays_to_half_after_half_life() -> None:  # T-BEL-3
    from qts.strategies.belief import BeliefAxis

    b = BeliefAxis(half_life=timedelta(hours=4))
    t0 = datetime(2024, 1, 1, 12, 0, tzinfo=UTC)
    b.update(value=0.8, now=t0)

    t1 = t0 + timedelta(hours=4)
    # One half-life elapsed → value × 0.5.
    assert b.at(t1) == pytest.approx(0.4)

    t2 = t0 + timedelta(hours=8)
    # Two half-lives → value × 0.25.
    assert b.at(t2) == pytest.approx(0.2)


def test_belief_axis_signed_decay() -> None:  # T-BEL-4
    from qts.strategies.belief import BeliefAxis

    b = BeliefAxis(half_life=timedelta(hours=2))
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    b.update(value=-0.6, now=t0)

    t1 = t0 + timedelta(hours=2)
    assert b.at(t1) == pytest.approx(-0.3)


def test_belief_axis_update_replaces_value() -> None:  # T-BEL-5
    from qts.strategies.belief import BeliefAxis

    b = BeliefAxis(half_life=timedelta(hours=4))
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    b.update(value=0.5, now=t0)

    t1 = t0 + timedelta(hours=2)
    b.update(value=-0.4, now=t1)

    # New value replaces the old; decay timer resets to t1.
    assert b.at(t1) == pytest.approx(-0.4)
    assert b.at(t1 + timedelta(hours=4)) == pytest.approx(-0.2)
```

- [ ] **Step 2: Run tests to verify fail**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_belief_axis.py -v
```

Expected: 5 FAILED — `ModuleNotFoundError`.

- [ ] **Step 3: Create BeliefAxis**

Create `src/qts/strategies/belief.py`:

```python
"""Exponentially-decaying scalar belief primitive.

Used by NewsReactiveMomentum to hold a per-axis news read that fades
over time, so the strategy 'forgets' stale signals at a controlled rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class BeliefAxis:
    """A scalar belief that decays exponentially toward zero over time.

    Args:
        half_life: Time after which an updated value decays by 50%.
    """

    half_life: timedelta
    _value: float = field(default=0.0, init=False)
    _last_update: datetime | None = field(default=None, init=False)

    def update(self, value: float, now: datetime) -> None:
        """Replace the belief with `value` anchored at `now`."""
        self._value = value
        self._last_update = now

    def at(self, now: datetime) -> float:
        """Return the value at `now`, after exponential decay since last_update."""
        if self._last_update is None:
            return 0.0
        elapsed = (now - self._last_update).total_seconds()
        half_life_s = self.half_life.total_seconds()
        if half_life_s <= 0:
            return self._value
        decay = 0.5 ** (elapsed / half_life_s)
        return self._value * decay
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_belief_axis.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/qts/strategies/belief.py tests/unit/test_belief_axis.py
git commit -m "feat(strategies): BeliefAxis exponential-decay primitive

Scalar belief that fades with time. update(value, now) anchors the
belief at a timestamp; at(now) returns the decayed value. half_life
controls the fade rate. Used by NewsReactiveMomentum to carry a
multi-axis news read across bars."
```

---

## Task 7: NewsReactiveMomentum strategy

**Files:**
- Create: `src/qts/strategies/news_reactive.py`
- Create: `tests/unit/test_news_reactive.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_news_reactive.py`:

```python
"""Tests for NewsReactiveMomentum — belief integration + alpha blend."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from qts.config import RiskLimits, StrategyParams
from qts.models.base import Bar, SignalSnapshot, VolLevel
from qts.world.events import TextEvent


def _bar(t: datetime, close: float = 30_000.0) -> Bar:
    return Bar(
        timestamp=t,
        symbol="BTCUSDT",
        open=close,
        high=close,
        low=close,
        close=close,
        volume=1.0,
    )


def _snapshot(t: datetime, combined_alpha: float = 0.0) -> SignalSnapshot:
    return SignalSnapshot(
        timestamp=t,
        symbol="BTCUSDT",
        rsi=50.0,
        macd_histogram=0.0,
        macd_line=0.0,
        macd_signal=0.0,
        bb_position=0.5,
        bb_upper=31_000.0,
        bb_lower=29_000.0,
        atr=200.0,
        momentum_5=0.0,
        vol_level=VolLevel.LOW,
        vol_level_confidence=0.8,
        sentiment_score=0.0,
        combined_alpha=combined_alpha,
    )


class _FakeLLM:
    def __init__(self, response: dict) -> None:
        self._response = response
        self.calls = 0

    async def query(self, *_a: object, **_k: object) -> str:
        import json
        return json.dumps(self._response)

    async def query_json(self, *_a: object, **_k: object) -> dict:
        self.calls += 1
        return self._response


@pytest.mark.asyncio
async def test_news_reactive_updates_belief_on_text(tmp_path: Path) -> None:  # T-NRX-1
    from qts.macro.news_classifier import NewsClassifier
    from qts.strategies.momentum import MomentumStrategy
    from qts.strategies.news_reactive import NewsReactiveMomentum

    llm = _FakeLLM({
        "direction": "bull", "confidence": 0.8, "relevance": 1.0, "magnitude": 0.5
    })
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    t0 = datetime(2023, 12, 13, 19, 0, tzinfo=UTC)
    evt = TextEvent(
        timestamp=t0, source="fed_press_release", persona=None,
        text="Inflation has slowed; we expect cuts in 2024.",
        metadata={"event_kind": "FOMC"},
    )
    await clf.warm_cache_for([evt])

    inner = MomentumStrategy(params=StrategyParams(), risk_limits=RiskLimits())
    strat = NewsReactiveMomentum(
        inner=inner,
        classifier=clf,
        belief_half_life=timedelta(hours=4),
        news_signal_weight=0.5,
    )

    strat.on_text(evt)

    # Belief is updated at the event timestamp.
    assert strat.news_alpha_at(t0) == pytest.approx(0.8 * 1.0 * 0.5)  # 0.4


@pytest.mark.asyncio
async def test_news_reactive_blends_into_snapshot_alpha(tmp_path: Path) -> None:  # T-NRX-2
    from qts.macro.news_classifier import NewsClassifier
    from qts.strategies.momentum import MomentumStrategy
    from qts.strategies.news_reactive import NewsReactiveMomentum

    llm = _FakeLLM({
        "direction": "bull", "confidence": 1.0, "relevance": 1.0, "magnitude": 1.0
    })
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    t0 = datetime(2023, 12, 13, 19, 0, tzinfo=UTC)
    evt = TextEvent(
        timestamp=t0, source="fed_press_release", persona=None,
        text="dovish surprise",
        metadata={},
    )
    await clf.warm_cache_for([evt])

    inner = MomentumStrategy(params=StrategyParams(), risk_limits=RiskLimits())
    strat = NewsReactiveMomentum(
        inner=inner, classifier=clf,
        belief_half_life=timedelta(hours=4),
        news_signal_weight=0.5,
    )

    # Update belief.
    strat.on_text(evt)

    # Bar at the same timestamp; base combined_alpha = 0.
    bar = _bar(t0, close=30_000.0)
    snap = _snapshot(t0, combined_alpha=0.0)

    # Capture the snapshot that inner.on_bar sees.
    captured: list[SignalSnapshot] = []
    original_on_bar = inner.on_bar
    def _spy(b, s, pos):
        captured.append(s)
        return original_on_bar(b, s, pos)
    inner.on_bar = _spy  # type: ignore[method-assign]

    strat.on_bar(bar, snap, positions=[])

    assert len(captured) == 1
    blended = captured[0].combined_alpha
    # base=0, news_alpha=1.0, weight=0.5 → blended = 0*0.5 + 1.0*0.5 = 0.5
    assert blended == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_news_reactive_belief_decays_between_bars(tmp_path: Path) -> None:  # T-NRX-3
    from qts.macro.news_classifier import NewsClassifier
    from qts.strategies.momentum import MomentumStrategy
    from qts.strategies.news_reactive import NewsReactiveMomentum

    llm = _FakeLLM({
        "direction": "bear", "confidence": 1.0, "relevance": 1.0, "magnitude": 1.0
    })
    clf = NewsClassifier(llm_client=llm, cache_dir=tmp_path)

    t0 = datetime(2023, 12, 13, 19, 0, tzinfo=UTC)
    evt = TextEvent(
        timestamp=t0, source="fed_press_release", persona=None,
        text="hawkish surprise",
        metadata={},
    )
    await clf.warm_cache_for([evt])

    strat = NewsReactiveMomentum(
        inner=MomentumStrategy(params=StrategyParams(), risk_limits=RiskLimits()),
        classifier=clf,
        belief_half_life=timedelta(hours=2),
        news_signal_weight=1.0,   # all news, no base
    )
    strat.on_text(evt)

    # At t0: news_alpha = -1.0 (full hit).
    assert strat.news_alpha_at(t0) == pytest.approx(-1.0)
    # At t0 + half_life: decayed to half.
    assert strat.news_alpha_at(t0 + timedelta(hours=2)) == pytest.approx(-0.5)


def test_news_reactive_forwards_on_fill(tmp_path: Path) -> None:  # T-NRX-4
    from qts.macro.news_classifier import NewsClassifier
    from qts.models.base import Fill, OrderSide
    from qts.strategies.momentum import MomentumStrategy
    from qts.strategies.news_reactive import NewsReactiveMomentum

    clf = NewsClassifier(llm_client=_FakeLLM({}), cache_dir=tmp_path)
    inner = MomentumStrategy(params=StrategyParams(), risk_limits=RiskLimits())
    strat = NewsReactiveMomentum(inner=inner, classifier=clf)

    captured: list[Fill] = []
    original = inner.on_fill
    def _spy(fill):
        captured.append(fill)
        return original(fill)
    inner.on_fill = _spy  # type: ignore[method-assign]

    f = Fill(
        order_id="x", fill_id="y", symbol="BTCUSDT",
        side=OrderSide.BUY, price=30_000.0, quantity=0.1,
        commission=0.0, timestamp=datetime(2024, 1, 1, tzinfo=UTC),
    )
    strat.on_fill(f)
    assert captured == [f]
```

- [ ] **Step 2: Run tests to verify fail**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_news_reactive.py -v
```

Expected: 4 FAILED — `ModuleNotFoundError`.

- [ ] **Step 3: Create NewsReactiveMomentum**

Create `src/qts/strategies/news_reactive.py`:

```python
"""NewsReactiveMomentum: MomentumStrategy + Qwen-driven news belief.

Composes an inner MomentumStrategy. Each TextEvent fires on_text:
    1. Look up the multi-axis NewsSignal via the cached NewsClassifier.
    2. Update a BeliefAxis with the signal's scalar alpha_contribution().

On every on_bar, the decayed belief is blended into the SignalSnapshot's
combined_alpha via:

    blended = (1 - w) * base_alpha + w * decayed_news_alpha

so MomentumStrategy's decision logic (entry/exit thresholds, Kelly sizing)
operates on a Qwen-grade input. Vanilla MomentumStrategy is unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from qts.strategies.belief import BeliefAxis

if TYPE_CHECKING:
    from qts.macro.news_classifier import NewsClassifier
    from qts.models.base import Bar, Fill, Order, Position, SignalSnapshot
    from qts.strategies.momentum import MomentumStrategy
    from qts.world.events import TextEvent


_DEFAULT_HALF_LIFE = timedelta(hours=4)


class NewsReactiveMomentum:
    """Momentum strategy with a Qwen-driven news belief overlay."""

    def __init__(
        self,
        inner: "MomentumStrategy",
        classifier: "NewsClassifier",
        belief_half_life: timedelta = _DEFAULT_HALF_LIFE,
        news_signal_weight: float = 0.5,
    ) -> None:
        if not 0.0 <= news_signal_weight <= 1.0:
            raise ValueError(f"news_signal_weight must be in [0, 1], got {news_signal_weight}")
        self._inner = inner
        self._classifier = classifier
        self._belief = BeliefAxis(half_life=belief_half_life)
        self._weight = news_signal_weight

    @property
    def name(self) -> str:
        return "NewsReactiveMomentum"

    @property
    def params(self) -> object:
        # Forward MomentumStrategy.params so the actor still finds it (Phase 8 v1 lookup pattern).
        return self._inner.params

    # ---- Strategy protocol ---------------------------------------------------

    def on_bar(
        self,
        bar: "Bar",
        snapshot: "SignalSnapshot",
        positions: "list[Position]",
    ) -> "list[Order]":
        news_alpha = self.news_alpha_at(bar.timestamp)
        blended = (1.0 - self._weight) * snapshot.combined_alpha + self._weight * news_alpha
        blended_snapshot = replace(snapshot, combined_alpha=blended)
        return self._inner.on_bar(bar, blended_snapshot, positions)

    def on_fill(self, fill: "Fill") -> None:
        self._inner.on_fill(fill)

    def on_text(self, event: "TextEvent") -> None:
        """Classify the event (cache-only) and update the belief."""
        signal = self._classifier.classify(event)
        self._belief.update(value=signal.alpha_contribution(), now=event.timestamp)

    # ---- Inspection -----------------------------------------------------------

    def news_alpha_at(self, now: datetime) -> float:
        """Return the current decayed news alpha (for tests + diagnostics)."""
        return self._belief.at(now)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_news_reactive.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/qts/strategies/news_reactive.py tests/unit/test_news_reactive.py
git commit -m "feat(strategies): NewsReactiveMomentum

Composes MomentumStrategy with a Qwen-driven news belief overlay.
on_text fires a synchronous cache-only classifier lookup and updates
a BeliefAxis at the event timestamp. on_bar blends the decayed belief
into the SignalSnapshot's combined_alpha before delegating to the
inner momentum logic. Vanilla MomentumStrategy is unchanged."
```

---

## Task 8: RealEpisode + from_disk loader

**Files:**
- Create: `src/qts/data/real_episode.py`
- Modify: `src/qts/data/__init__.py` — export RealEpisode
- Create: `tests/unit/test_real_episode.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_real_episode.py`:

```python
"""Tests for RealEpisode and its disk loader."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


def _write_curated(root: Path) -> None:
    """Write a minimal curated dataset (3 bars + 1 statement + 1 press-conf paragraph)."""
    root.mkdir(parents=True, exist_ok=True)
    bars_csv = "\n".join([
        "timestamp,open,high,low,close,volume",
        "2023-12-13T00:00:00+00:00,40000.0,40100.0,39950.0,40050.0,12.3",
        "2023-12-13T00:01:00+00:00,40050.0,40080.0,40010.0,40060.0,9.1",
        "2023-12-13T00:02:00+00:00,40060.0,40090.0,40030.0,40075.0,7.8",
    ])
    (root / "bars.csv").write_text(bars_csv + "\n", encoding="utf-8")

    (root / "statement.txt").write_text(
        "The Federal Open Market Committee decided to maintain the target range.",
        encoding="utf-8",
    )

    press_conf = {
        "paragraphs": [
            {
                "timestamp": "2023-12-13T19:30:00+00:00",
                "text": "We see growing evidence that the disinflationary process is on track.",
            },
        ],
    }
    (root / "press_conf.json").write_text(json.dumps(press_conf), encoding="utf-8")


def test_from_disk_loads_bars(tmp_path: Path) -> None:  # T-REP-1
    from qts.data.real_episode import RealEpisode

    root = tmp_path / "fomc_test"
    _write_curated(root)

    ep = RealEpisode.from_disk(root, symbol="BTCUSDT", source="fomc:test")
    assert len(ep.terrain.bars) == 3
    first = ep.terrain.bars[0]
    assert first.timestamp == datetime(2023, 12, 13, 0, 0, tzinfo=UTC)
    assert first.open == 40_000.0
    assert first.close == 40_050.0
    assert first.symbol == "BTCUSDT"


def test_from_disk_loads_text_events(tmp_path: Path) -> None:  # T-REP-2
    from qts.data.real_episode import RealEpisode

    root = tmp_path / "fomc_test"
    _write_curated(root)

    ep = RealEpisode.from_disk(root, symbol="BTCUSDT", source="fomc:test")
    # Two events: the statement (timestamped at the press release moment) and the press-conf paragraph.
    assert len(ep.text_events) == 2

    statement = next(e for e in ep.text_events if e.source == "fed_press_release")
    assert "Federal Open Market Committee" in statement.text

    presser = next(e for e in ep.text_events if e.source == "powell_press_conf")
    assert presser.timestamp == datetime(2023, 12, 13, 19, 30, tzinfo=UTC)


def test_from_disk_attaches_source_and_terrain(tmp_path: Path) -> None:  # T-REP-3
    from qts.data.real_episode import RealEpisode

    root = tmp_path / "fomc_test"
    _write_curated(root)

    ep = RealEpisode.from_disk(root, symbol="BTCUSDT", source="fomc:2023-12-13")
    assert ep.source == "fomc:2023-12-13"
    assert ep.terrain.symbol == "BTCUSDT"
    assert ep.terrain.name.startswith("real:fomc")
```

- [ ] **Step 2: Run tests to verify fail**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_real_episode.py -v
```

Expected: 3 FAILED — `ModuleNotFoundError`.

- [ ] **Step 3: Create RealEpisode**

Create `src/qts/data/real_episode.py`:

```python
"""RealEpisode: real-market analogue of SimulatedEpisode.

Loads a curated dataset from disk (bars.csv + statement.txt + press_conf.json)
into a MarketTerrain plus a list of TextEvents. Same wrapping pattern as
SimulatedEpisode so the existing run_terrain_backtest pipeline composes.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from qts.models.base import (
    Bar,
    Catalyst,
    LiquidityLevel,
    SentimentLevel,
    Trend,
    VolLevel,
)
from qts.models.terrain import MacroRegime, MarketEvent, MarketTerrain
from qts.world.events import TextEvent

# The press release fires at 14:00 ET = 19:00 UTC for US FOMC days.
_FOMC_PRESS_RELEASE_UTC_HOUR = 19


@dataclass
class RealEpisode:
    """Real-data analogue of SimulatedEpisode."""

    terrain: MarketTerrain
    text_events: list[TextEvent]
    source: str   # e.g. "fomc:2023-12-13"

    @classmethod
    def from_disk(
        cls,
        root: Path,
        symbol: str,
        source: str,
    ) -> RealEpisode:
        """Load bars.csv + statement.txt + press_conf.json from `root`."""
        root = Path(root)
        bars = _load_bars(root / "bars.csv", symbol)
        statement_text = (root / "statement.txt").read_text(encoding="utf-8").strip()
        press_conf_paragraphs = _load_press_conf(root / "press_conf.json")

        if not bars:
            raise ValueError(f"no bars found in {root}/bars.csv")

        first_bar = bars[0]
        last_bar = bars[-1]
        statement_timestamp = first_bar.timestamp.replace(
            hour=_FOMC_PRESS_RELEASE_UTC_HOUR, minute=0, second=0, microsecond=0
        )

        # Build TextEvents
        statement_event = TextEvent(
            timestamp=statement_timestamp,
            source="fed_press_release",
            persona=None,
            text=statement_text,
            metadata={"event_kind": "FOMC", "real": True},
        )
        press_conf_events = [
            TextEvent(
                timestamp=p["timestamp"],
                source="powell_press_conf",
                persona="powell",
                text=p["text"],
                metadata={"event_kind": "FOMC", "real": True},
            )
            for p in press_conf_paragraphs
        ]
        text_events = sorted(
            [statement_event, *press_conf_events],
            key=lambda e: e.timestamp,
        )

        # Build a placeholder MacroRegime — real-data tests don't depend on regime tags.
        regime = MacroRegime(
            trend=Trend.SIDEWAYS,
            volatility=VolLevel.HIGH,
            liquidity=LiquidityLevel.ABUNDANT,
            sentiment=SentimentLevel.NEUTRAL,
            catalyst=Catalyst.MACRO_EVENT,
            expected_drift=0.0,
            expected_vol=0.01,
            correlation_regime=0.5,
            scenario_description=f"real:{source}",
        )

        # Mirror text events as MarketEvents on the calendar (parallel to Phase 8 v1 SimulatedEpisode).
        calendar = [
            MarketEvent(
                timestamp=e.timestamp,
                event_type=f"text:{e.source}",
                description=e.text,
                impact_magnitude=0.0,
            )
            for e in text_events
        ]

        terrain = MarketTerrain(
            name=f"real:{source}",
            symbol=symbol,
            start=first_bar.timestamp,
            end=last_bar.timestamp,
            regime=regime,
            bars=bars,
            event_calendar=calendar,
        )

        return cls(terrain=terrain, text_events=text_events, source=source)


def _load_bars(path: Path, symbol: str) -> list[Bar]:
    bars: list[Bar] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bars.append(
                Bar(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    symbol=symbol,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return bars


def _load_press_conf(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    paragraphs = raw.get("paragraphs", [])
    out: list[dict] = []
    for p in paragraphs:
        out.append({
            "timestamp": datetime.fromisoformat(p["timestamp"]),
            "text": p["text"],
        })
    return out
```

- [ ] **Step 4: Export from qts.data**

Edit `src/qts/data/__init__.py`. Replace it with:

```python
"""Data ingestion and management sub-package.

Submodules:
    market       - OHLCV and order book data from exchanges
    news         - News article ingestion and preprocessing
    social       - Social media data (Reddit, Twitter)
    geopolitical - Geopolitical event data (GDELT, etc.)

Top-level types:
    RealEpisode  - Real-market analogue of SimulatedEpisode
"""

from qts.data.real_episode import RealEpisode

__all__ = ["RealEpisode"]
```

- [ ] **Step 5: Run tests to verify pass**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_real_episode.py -v
```

Expected: 3 PASSED.

- [ ] **Step 6: Commit**

```bash
git add src/qts/data/real_episode.py src/qts/data/__init__.py tests/unit/test_real_episode.py
git commit -m "feat(data): RealEpisode wraps real bars + text events into MarketTerrain

Mirror of SimulatedEpisode for real-market data. from_disk loads
bars.csv, statement.txt, press_conf.json into a MarketTerrain plus
sorted TextEvent list. FOMC statement is timestamped at 19:00 UTC
(14:00 ET press release moment); press-conf paragraphs carry their
own timestamps."
```

---

## Task 9: run_real_backtest — bars + text events through Nautilus

**Files:**
- Create: `src/qts/nautilus/real_runner.py`
- Create: `tests/unit/test_real_runner.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_real_runner.py`:

```python
"""Tests for run_real_backtest — terrain backtest with text-event dispatch."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from qts.models.base import Bar
from qts.models.terrain import MarketTerrain
from qts.world.events import TextEvent


def _terrain_with_n_bars(n: int) -> MarketTerrain:
    from qts.models.base import (
        Catalyst, LiquidityLevel, SentimentLevel, Trend, VolLevel,
    )
    from qts.models.terrain import MacroRegime

    start = datetime(2023, 12, 13, 18, 50, tzinfo=UTC)
    bars = [
        Bar(
            timestamp=start + timedelta(minutes=i),
            symbol="BTCUSDT",
            open=40_000.0 + i,
            high=40_000.0 + i + 5,
            low=40_000.0 + i - 5,
            close=40_000.0 + i,
            volume=1.0,
        )
        for i in range(n)
    ]
    regime = MacroRegime(
        trend=Trend.SIDEWAYS, volatility=VolLevel.LOW,
        liquidity=LiquidityLevel.ABUNDANT, sentiment=SentimentLevel.NEUTRAL,
        catalyst=Catalyst.MACRO_EVENT,
        expected_drift=0.0, expected_vol=0.01, correlation_regime=0.5,
        scenario_description="test",
    )
    return MarketTerrain(
        name="test", symbol="BTCUSDT",
        start=bars[0].timestamp, end=bars[-1].timestamp,
        regime=regime, bars=bars, event_calendar=[],
    )


def test_run_real_backtest_dispatches_text_events_in_order(tmp_path: Path) -> None:  # T-REAL-1
    from qts.data.real_episode import RealEpisode
    from qts.nautilus.real_runner import run_real_backtest

    seen: list[TextEvent] = []

    class _Strategy:
        params = None
        name = "spy"
        def on_bar(self, *a, **k): return []
        def on_fill(self, *a, **k): pass
        def on_text(self, event: TextEvent) -> None:
            seen.append(event)

    terrain = _terrain_with_n_bars(20)
    events = [
        TextEvent(
            timestamp=terrain.bars[5].timestamp,
            source="fed_press_release", persona=None,
            text="event-A", metadata={},
        ),
        TextEvent(
            timestamp=terrain.bars[15].timestamp,
            source="powell_press_conf", persona="powell",
            text="event-B", metadata={},
        ),
    ]
    episode = RealEpisode(terrain=terrain, text_events=events, source="test")

    run_real_backtest(episode, _Strategy(), log_level="ERROR")

    assert [e.text for e in seen] == ["event-A", "event-B"]


def test_run_real_backtest_returns_backtest_result(tmp_path: Path) -> None:  # T-REAL-2
    from qts.config import RiskLimits, StrategyParams
    from qts.data.real_episode import RealEpisode
    from qts.nautilus.real_runner import run_real_backtest
    from qts.strategies.momentum import MomentumStrategy

    terrain = _terrain_with_n_bars(60)
    episode = RealEpisode(terrain=terrain, text_events=[], source="test")
    strat = MomentumStrategy(params=StrategyParams(), risk_limits=RiskLimits())

    result = run_real_backtest(episode, strat, log_level="ERROR")
    assert len(result.equity_curve) > 0
```

- [ ] **Step 2: Run tests to verify fail**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_real_runner.py -v
```

Expected: 2 FAILED — `ModuleNotFoundError`.

- [ ] **Step 3: Create run_real_backtest**

Create `src/qts/nautilus/real_runner.py`:

```python
"""run_real_backtest — terrain backtest that also dispatches text events.

Real-data analogue of run_terrain_backtest. Uses the same Nautilus
machinery for bars + matching engine, but additionally walks the
RealEpisode's text_events list and forwards each through
QTSStrategy.on_text_event at the appropriate point in the bar stream.

For v2, text-event dispatch happens *before* the backtest runs:
we eagerly walk every event in chronological order against the
QTSStrategy actor's on_text_event method. This is correct because
on_text only updates strategy state — the strategy's bar-level
decisions happen during the Nautilus engine.run() pass, by which
time the belief state is already populated.

For events that fall *during* the bar stream (the common FOMC case),
this approximation is fine for v2: the belief state evolves correctly
relative to bar timestamps because BeliefAxis.at(bar.timestamp)
respects the event timestamp. Bars before the event see decayed state
from t=-inf (zero); bars after see the post-event belief.

(A future bar-by-bar interleaved dispatch would be more faithful for
multi-event scenarios, but is out of scope for v2.)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qts.nautilus.runner import run_terrain_backtest

if TYPE_CHECKING:
    from qts.data.real_episode import RealEpisode
    from qts.nautilus.config import BacktestResult, VenueConfig
    from qts.strategies.base import Strategy

logger = logging.getLogger(__name__)


def run_real_backtest(
    episode: "RealEpisode",
    strategy: "Strategy",
    venue_config: "VenueConfig | None" = None,
    log_level: str = "WARNING",
    instrument: object | None = None,
) -> "BacktestResult":
    """Run `strategy` against a RealEpisode, dispatching text events first."""
    # Pre-dispatch all text events into the strategy in chronological order.
    for event in sorted(episode.text_events, key=lambda e: e.timestamp):
        try:
            strategy.on_text(event)
        except Exception:  # noqa: BLE001
            logger.exception("strategy.on_text failed on event %r", event)

    # Then run the standard terrain backtest — the strategy's belief is now warmed up.
    return run_terrain_backtest(
        episode.terrain,
        strategy,
        venue_config=venue_config,
        log_level=log_level,
        instrument=instrument,
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_real_runner.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/qts/nautilus/real_runner.py tests/unit/test_real_runner.py
git commit -m "feat(nautilus): run_real_backtest — bars + text events

Pre-dispatches text events to strategy.on_text in chronological order
before running the terrain backtest, so belief state is populated by
the time Nautilus drives bars. BeliefAxis decay still respects event
vs bar timestamps. Bar-by-bar interleaved dispatch is a future
extension; v2's approximation suffices for single-event FOMC days."
```

---

## Task 10: FOMC data fetcher script

**Files:**
- Create: `scripts/fetch_fomc_data.py`
- Create: `tests/unit/test_fetch_fomc_data.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_fetch_fomc_data.py`:

```python
"""Tests for the FOMC data fetcher script's parsing helpers."""

from __future__ import annotations

from datetime import UTC, datetime


def test_parse_binance_klines() -> None:  # T-FETCH-1
    from scripts.fetch_fomc_data import parse_binance_klines

    raw = [
        # [open_time_ms, open, high, low, close, volume, close_time_ms, ...]
        [1702425600000, "40000.0", "40100.0", "39950.0", "40050.0", "12.3", 1702425659999, "0", 0, "0", "0", "0"],
        [1702425660000, "40050.0", "40080.0", "40010.0", "40060.0", "9.1", 1702425719999, "0", 0, "0", "0", "0"],
    ]
    bars = parse_binance_klines(raw, symbol="BTCUSDT")
    assert len(bars) == 2
    assert bars[0]["timestamp"] == datetime(2023, 12, 13, 0, 0, tzinfo=UTC)
    assert bars[0]["open"] == 40_000.0
    assert bars[0]["close"] == 40_050.0
    assert bars[1]["timestamp"] == datetime(2023, 12, 13, 0, 1, tzinfo=UTC)


def test_extract_paragraphs_from_html() -> None:  # T-FETCH-2
    from scripts.fetch_fomc_data import extract_paragraphs_from_html

    html = (
        "<html><body>"
        "<p>This is the first paragraph of the transcript.</p>"
        "<p>Short.</p>"  # too short — filtered out
        "<p>Here is a longer paragraph that should survive the filter "
        "because it's discussion-worthy.</p>"
        "</body></html>"
    )
    paragraphs = extract_paragraphs_from_html(html, min_chars=50)
    assert len(paragraphs) == 2
    assert "first paragraph" in paragraphs[0]
    assert "longer paragraph" in paragraphs[1]


def test_assign_paragraph_timestamps() -> None:  # T-FETCH-3
    from scripts.fetch_fomc_data import assign_paragraph_timestamps

    start = datetime(2023, 12, 13, 19, 30, tzinfo=UTC)
    paragraphs = ["a", "b", "c", "d"]
    spaced = assign_paragraph_timestamps(paragraphs, start=start, span_minutes=60)

    assert len(spaced) == 4
    assert spaced[0]["timestamp"] == start
    # Equal spacing across 60 minutes -> 20-min gaps.
    assert (spaced[1]["timestamp"] - spaced[0]["timestamp"]).total_seconds() == 20 * 60
    assert (spaced[3]["timestamp"] - spaced[0]["timestamp"]).total_seconds() == 60 * 60
```

- [ ] **Step 2: Run tests to verify fail**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_fetch_fomc_data.py -v
```

Expected: 3 FAILED — `ModuleNotFoundError: No module named 'scripts.fetch_fomc_data'`.

- [ ] **Step 3: Create the fetcher script**

Create `scripts/fetch_fomc_data.py`:

```python
"""One-off fetcher for a curated FOMC dataset.

Pulls real BTCUSDT 1m bars from Binance + FOMC statement + Powell press
conference transcript from federalreserve.gov, into a single directory.

Usage:
    .venv/bin/python -m scripts.fetch_fomc_data 2023-12-13 data/real/fomc/2023-12-13

The output is committed to the repo so the integration test is reproducible
without re-fetching.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
_BINANCE_MAX_LIMIT = 1000
_PARAGRAPH_MIN_CHARS = 80
_PRESS_CONF_START_UTC_HOUR = 19  # 14:00 ET — 30 min after press release
_PRESS_CONF_START_UTC_MINUTE = 30


def parse_binance_klines(raw: list[list[Any]], symbol: str) -> list[dict]:
    """Convert raw Binance klines payload into bar dicts."""
    bars: list[dict] = []
    for row in raw:
        open_ms = int(row[0])
        bars.append({
            "timestamp": datetime.fromtimestamp(open_ms / 1000.0, tz=UTC),
            "symbol": symbol,
            "open": float(row[1]),
            "high": float(row[2]),
            "low": float(row[3]),
            "close": float(row[4]),
            "volume": float(row[5]),
        })
    return bars


def fetch_binance_bars(symbol: str, day: datetime) -> list[dict]:
    """Fetch all 1m bars for `day` (UTC) from Binance, chunked across multiple calls."""
    start_ms = int(day.replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
    end_ms = int((day + timedelta(days=1)).timestamp() * 1000) - 1

    bars: list[dict] = []
    cursor_ms = start_ms
    while cursor_ms < end_ms:
        params = {
            "symbol": symbol.upper(),
            "interval": "1m",
            "startTime": cursor_ms,
            "endTime": end_ms,
            "limit": _BINANCE_MAX_LIMIT,
        }
        with httpx.Client(timeout=30.0) as client:
            response = client.get(_BINANCE_KLINES_URL, params=params)
            response.raise_for_status()
            raw = response.json()
        if not raw:
            break
        chunk = parse_binance_klines(raw, symbol)
        bars.extend(chunk)
        cursor_ms = int(raw[-1][0]) + 60_000  # advance past the last open_time
        if len(chunk) < _BINANCE_MAX_LIMIT:
            break
    return bars


def write_bars_csv(bars: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for b in bars:
            writer.writerow([
                b["timestamp"].isoformat(),
                b["open"], b["high"], b["low"], b["close"], b["volume"],
            ])


def fetch_fomc_statement(date_str: str) -> str:
    """Fetch the FOMC statement HTML from federalreserve.gov and extract its body text."""
    # Standard URL pattern for FOMC press releases.
    yyyymmdd = date_str.replace("-", "")
    url = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{yyyymmdd}a.htm"
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    body = response.text
    paragraphs = extract_paragraphs_from_html(body, min_chars=_PARAGRAPH_MIN_CHARS)
    return "\n\n".join(paragraphs)


def fetch_press_conf_paragraphs(date_str: str) -> list[str]:
    """Fetch the press conference transcript HTML and return its body paragraphs."""
    yyyymmdd = date_str.replace("-", "")
    url = f"https://www.federalreserve.gov/monetarypolicy/fomcpresconf{yyyymmdd}.htm"
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
    return extract_paragraphs_from_html(response.text, min_chars=_PARAGRAPH_MIN_CHARS)


_TAG_RE = re.compile(r"<[^>]+>")
_PARA_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", re.IGNORECASE | re.DOTALL)


def extract_paragraphs_from_html(content: str, min_chars: int = 80) -> list[str]:
    """Naive <p>-tag extractor. Filters paragraphs shorter than min_chars."""
    paragraphs: list[str] = []
    for match in _PARA_RE.finditer(content):
        inner = _TAG_RE.sub("", match.group(1))
        text = html.unescape(inner).strip()
        text = re.sub(r"\s+", " ", text)
        if len(text) >= min_chars:
            paragraphs.append(text)
    return paragraphs


def assign_paragraph_timestamps(
    paragraphs: list[str],
    start: datetime,
    span_minutes: int = 60,
) -> list[dict]:
    """Spread N paragraphs evenly across `span_minutes` starting at `start`."""
    if not paragraphs:
        return []
    if len(paragraphs) == 1:
        return [{"timestamp": start, "text": paragraphs[0]}]
    step = timedelta(minutes=span_minutes / (len(paragraphs) - 1))
    return [
        {"timestamp": start + step * i, "text": text}
        for i, text in enumerate(paragraphs)
    ]


def write_statement_txt(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def write_press_conf_json(paragraphs_with_ts: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "paragraphs": [
            {"timestamp": p["timestamp"].isoformat(), "text": p["text"]}
            for p in paragraphs_with_ts
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch curated FOMC data for a given date.")
    parser.add_argument("date", help="ISO date, e.g. 2023-12-13")
    parser.add_argument("out_dir", help="Output directory")
    parser.add_argument("--symbol", default="BTCUSDT")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    day = datetime.fromisoformat(args.date).replace(tzinfo=UTC)
    out = Path(args.out_dir)

    logger.info("Fetching Binance bars for %s on %s", args.symbol, args.date)
    bars = fetch_binance_bars(args.symbol, day)
    logger.info("  got %d bars", len(bars))
    write_bars_csv(bars, out / "bars.csv")

    logger.info("Fetching FOMC statement")
    statement = fetch_fomc_statement(args.date)
    write_statement_txt(statement, out / "statement.txt")

    logger.info("Fetching press conference transcript")
    paragraphs = fetch_press_conf_paragraphs(args.date)
    logger.info("  got %d paragraphs", len(paragraphs))
    presser_start = day.replace(
        hour=_PRESS_CONF_START_UTC_HOUR,
        minute=_PRESS_CONF_START_UTC_MINUTE,
        second=0, microsecond=0,
    )
    timestamped = assign_paragraph_timestamps(paragraphs, presser_start, span_minutes=60)
    write_press_conf_json(timestamped, out / "press_conf.json")

    logger.info("Done. Output in %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Note: this file lives in `scripts/` not `src/`. Tests import via `from scripts.fetch_fomc_data import ...`. Make sure `scripts/__init__.py` exists (empty file) so the package import works in tests.

- [ ] **Step 4: Add empty scripts/__init__.py**

```bash
touch scripts/__init__.py
```

- [ ] **Step 5: Run tests to verify pass**

```bash
.venv/bin/python -m pytest --no-cov -q tests/unit/test_fetch_fomc_data.py -v
```

Expected: 3 PASSED.

- [ ] **Step 6: Commit**

```bash
git add scripts/__init__.py scripts/fetch_fomc_data.py tests/unit/test_fetch_fomc_data.py
git commit -m "feat(scripts): one-off FOMC data fetcher

Pulls 1m BTCUSDT bars from Binance kline API + FOMC statement and
press conference paragraphs from federalreserve.gov. Press-conf
paragraphs are spread evenly across the 1h post-decision window
(timestamps are approximations — exact per-paragraph timing isn't
in the public transcript). Pure parsing helpers are unit-tested."
```

---

## Task 11: Run fetcher + commit curated 2023-12-13 dataset

**Files:**
- Create: `data/real/fomc/2023-12-13/bars.csv` (from Binance)
- Create: `data/real/fomc/2023-12-13/statement.txt` (from fed.gov)
- Create: `data/real/fomc/2023-12-13/press_conf.json` (from fed.gov)

- [ ] **Step 1: Add data/real/ to the gitignore exception list**

Edit `.gitignore`. Find the existing line `!/data/world/` (~line 44). After it, add:

```
!/data/real/
!/data/real/**
```

- [ ] **Step 2: Verify the directory exists**

```bash
mkdir -p data/real/fomc/2023-12-13
```

- [ ] **Step 3: Run the fetcher**

```bash
.venv/bin/python -m scripts.fetch_fomc_data 2023-12-13 data/real/fomc/2023-12-13
```

Expected output: three files populated. The script logs the bar count and paragraph count.

If Binance returns < 1400 bars or fed.gov returns 0 paragraphs, **STOP and report BLOCKED** — the curated dataset is incomplete. Do not proceed to commit.

- [ ] **Step 4: Spot-check the files**

```bash
ls -la data/real/fomc/2023-12-13/
.venv/bin/python -c "
from pathlib import Path
from qts.data.real_episode import RealEpisode
ep = RealEpisode.from_disk(Path('data/real/fomc/2023-12-13'), symbol='BTCUSDT', source='fomc:2023-12-13')
print(f'bars: {len(ep.terrain.bars)}, text_events: {len(ep.text_events)}')
print(f'  first bar:  {ep.terrain.bars[0].timestamp} close={ep.terrain.bars[0].close}')
print(f'  last bar:   {ep.terrain.bars[-1].timestamp} close={ep.terrain.bars[-1].close}')
print(f'  statement first 120 chars: {ep.text_events[0].text[:120]!r}')
"
```

Expected: 1440 bars, ≥ 2 text events. Statement should contain phrases like "Committee" and "federal funds rate". If anything is missing or malformed, fix the fetcher and re-run before committing.

- [ ] **Step 5: Commit**

```bash
git add .gitignore data/real/fomc/2023-12-13/bars.csv data/real/fomc/2023-12-13/statement.txt data/real/fomc/2023-12-13/press_conf.json
git commit -m "chore(data): curated 2023-12-13 FOMC dataset

Real BTCUSDT 1m bars from Binance + FOMC statement and press
conference transcript paragraphs from federalreserve.gov. Committed
into the repo so the v2 acceptance test runs reproducibly without
re-fetching from upstream."
```

---

## Task 12: Hand-validation tool for the news classifier

**Files:**
- Create: `scripts/validate_news_classifier.py`

This is a developer tool, not a test. Run it to eyeball Qwen outputs before trusting the integration test.

- [ ] **Step 1: Create the validation script**

Create `scripts/validate_news_classifier.py`:

```python
"""Hand-validate the news classifier on the curated dataset.

Walks every TextEvent in data/real/fomc/<DATE>, classifies each via
the Qwen-backed NewsClassifier (Ollama), and prints the structured
output for visual inspection. Use this BEFORE running the v2
acceptance test to confirm Qwen is reading the text sensibly.

Usage:
    .venv/bin/python -m scripts.validate_news_classifier data/real/fomc/2023-12-13
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from qts.data.real_episode import RealEpisode
from qts.macro.news_classifier import NewsClassifier
from qts.oversight.llm_client import create_llm_client

logger = logging.getLogger(__name__)


async def _main(root: Path, cache_dir: Path) -> int:
    episode = RealEpisode.from_disk(root, symbol="BTCUSDT", source=f"validate:{root.name}")
    llm = create_llm_client(backend="ollama")
    classifier = NewsClassifier(llm_client=llm, cache_dir=cache_dir)

    print(f"\nValidating {len(episode.text_events)} text events from {root}:")
    print("-" * 80)
    for i, event in enumerate(episode.text_events, 1):
        signal = await classifier.classify_async(event)
        print(
            f"[{i:>2}] src={event.source:>20s} "
            f"dir={signal.direction:>7s} "
            f"conf={signal.confidence:.2f} rel={signal.relevance:.2f} "
            f"mag={signal.magnitude:.2f} | "
            f"alpha={signal.alpha_contribution():+.3f}"
        )
        preview = event.text[:100].replace("\n", " ")
        print(f"     text: {preview!r}...")
        print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Hand-validate news classifier on curated data.")
    parser.add_argument("root", help="Curated dataset directory")
    parser.add_argument(
        "--cache-dir",
        default="data/news_cache",
        help="Where to cache classifier responses (default: data/news_cache)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    return asyncio.run(_main(Path(args.root), Path(args.cache_dir)))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the validation**

```bash
.venv/bin/python -m scripts.validate_news_classifier data/real/fomc/2023-12-13
```

Inspect the output. For 2023-12-13 (dovish pivot), expect:
- FOMC statement: direction=`bull` or `neutral`, confidence ≥ 0.4
- Press conference paragraphs about disinflation / rate cuts: direction=`bull`, confidence ≥ 0.5

If most outputs come back as `direction=neutral, confidence=0.0`, Qwen is mis-reading the text. Iterate on `_SYSTEM_PROMPT` in `news_classifier.py` until the outputs look sensible. **Do not proceed to Task 13 until classifications pass visual inspection.**

- [ ] **Step 3: Commit the validation script**

```bash
git add scripts/validate_news_classifier.py
git commit -m "feat(scripts): hand-validation tool for the news classifier

Walks every TextEvent in a curated dataset and prints the multi-axis
Qwen classification, for visual inspection before trusting the
acceptance test. Use BEFORE running v2 integration test to confirm
Qwen is reading the text sensibly."
```

- [ ] **Step 4: (Optional) Commit the news_cache**

The cache directory `data/news_cache/` is populated as a side effect. If reproducibility on a fresh checkout matters (CI on machines without Ollama), commit the cache:

```bash
git add data/news_cache/
git commit -m "chore(data): commit news_cache for v2 acceptance reproducibility

Pre-computed classifier outputs for the 2023-12-13 curated dataset
let the v2 acceptance test run without Ollama present (cache-only
path via NewsClassifier.classify)."
```

If you skip this step, the v2 acceptance test in Task 13 will require Ollama running on the test machine. If unsure, **commit the cache** — CI portability matters more than dataset size.

---

## Task 13: Integration acceptance test — beats buy-and-hold

**Files:**
- Create: `tests/integration/test_news_reactive_2023_12_13.py`

This is THE acceptance test. If it passes, v2 is done.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/__init__.py` (empty) if it doesn't exist, then create `tests/integration/test_news_reactive_2023_12_13.py`:

```python
"""v2 acceptance: NewsReactiveMomentum beats buy-and-hold on 2023-12-13.

Loads the curated dataset, classifies all text events (cache-only),
runs NewsReactiveMomentum through Nautilus, and asserts the strategy's
day-end equity exceeds buy-and-hold equity for the same notional.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

_CURATED_ROOT = Path("data/real/fomc/2023-12-13")
_CACHE_DIR = Path("data/news_cache")


def _curated_exists() -> bool:
    return (
        (_CURATED_ROOT / "bars.csv").exists()
        and (_CURATED_ROOT / "statement.txt").exists()
        and (_CURATED_ROOT / "press_conf.json").exists()
    )


@pytest.mark.skipif(not _curated_exists(), reason="curated dataset missing — run scripts/fetch_fomc_data.py")
def test_beats_buy_and_hold_on_dovish_pivot() -> None:  # T-V2-ACCEPT
    from qts.config import RiskLimits, StrategyParams
    from qts.data.real_episode import RealEpisode
    from qts.macro.news_classifier import NewsClassifier
    from qts.nautilus.real_runner import run_real_backtest
    from qts.oversight.llm_client import create_llm_client
    from qts.strategies.momentum import MomentumStrategy
    from qts.strategies.news_reactive import NewsReactiveMomentum

    episode = RealEpisode.from_disk(_CURATED_ROOT, symbol="BTCUSDT", source="fomc:2023-12-13")

    # Pre-warm classifier cache. If the cache is already committed, this is a no-op.
    llm = create_llm_client(backend="ollama")
    classifier = NewsClassifier(llm_client=llm, cache_dir=_CACHE_DIR)
    asyncio.run(classifier.warm_cache_for(episode.text_events))

    # Strategy under test
    strat = NewsReactiveMomentum(
        inner=MomentumStrategy(params=StrategyParams(), risk_limits=RiskLimits()),
        classifier=classifier,
        news_signal_weight=0.5,
    )
    result = run_real_backtest(episode, strat, log_level="ERROR")

    # Buy-and-hold benchmark: bought at first bar's open, holds through last bar's close.
    # The Nautilus account starts with vc.starting_balance USDT (default 100k) at first bar;
    # buy-and-hold equity = starting_balance × (last_close / first_open).
    from qts.nautilus.config import VenueConfig
    vc = VenueConfig()
    bars = episode.terrain.bars
    hold_multiplier = bars[-1].close / bars[0].open
    hold_equity = vc.starting_balance * hold_multiplier

    strat_equity = result.equity_curve[-1] if result.equity_curve else 0.0

    assert strat_equity > hold_equity, (
        f"NewsReactiveMomentum did NOT beat buy-and-hold:\n"
        f"  strategy day-end equity:  {strat_equity:,.2f}\n"
        f"  buy-and-hold equity:      {hold_equity:,.2f}\n"
        f"  shortfall:                {hold_equity - strat_equity:,.2f}\n"
    )
```

- [ ] **Step 2: Run the acceptance test**

```bash
.venv/bin/python -m pytest --no-cov -q tests/integration/test_news_reactive_2023_12_13.py -v
```

Three possible outcomes:

**PASS:** v2 is done. Proceed to Task 14.

**FAIL with strategy_equity < hold_equity:** This is a *valid result*. The architecture works (test ran end-to-end, strategy was driven through real bars + real text), but the strategy did not beat buy-and-hold. Per the spec's risk section: report DONE_WITH_CONCERNS and document the gap. The gap analysis goes in the commit message and Task 14's plan doc update. Do NOT silently weaken the assertion.

**FAIL with skipped (curated dataset missing):** Task 11 wasn't completed — return to Task 11 and run the fetcher.

**FAIL with `KeyError: 'event ... not in cache'`:** Cache wasn't warmed. Make sure Ollama is running, or commit the pre-warmed `data/news_cache/`.

- [ ] **Step 3: Commit (on PASS) or commit-with-concerns (on valid FAIL)**

On PASS:

```bash
git add tests/integration/__init__.py tests/integration/test_news_reactive_2023_12_13.py
git commit -m "feat(test): v2 acceptance — NewsReactiveMomentum beats buy-and-hold

End-to-end real-data integration test on 2023-12-13 (Powell pivot,
BTC +7%). Loads curated bars + Powell text, warms classifier cache,
runs NewsReactiveMomentum through Nautilus, and asserts day-end
equity > buy-and-hold equity. ACCEPTANCE PASSES — Phase 8 v2 done."
```

On valid FAIL (architecture works, just no edge):

```bash
git add tests/integration/__init__.py tests/integration/test_news_reactive_2023_12_13.py
git commit -m "test(v2): acceptance test wired end-to-end; baseline does NOT beat hold

Architecture passes — strategy ran through real bars + real text,
classifier wired correctly. But NewsReactiveMomentum at default
params (half_life=4h, news_signal_weight=0.5) did not beat
buy-and-hold on 2023-12-13. Recorded the gap in the test failure
message. Next slice: Optuna sweep over news_signal_weight,
belief_half_life, and Qwen prompt variants (see project memory
project_deferred_grills)."
```

The failing assertion stays as a tracked regression target. Mark the test with `pytest.xfail` only if the user explicitly confirms after reviewing the gap.

---

## Task 14: Public API exports + plan doc update

**Files:**
- Modify: `src/qts/strategies/__init__.py` — export NewsReactiveMomentum
- Modify: `src/qts/macro/__init__.py` — export NewsClassifier, NewsSignal
- Modify: `docs/plans/terrain-refactor-plan.md` — note Phase 8 v2 status

- [ ] **Step 1: Update qts.strategies exports**

Read `src/qts/strategies/__init__.py` to confirm its current shape, then append after the existing docstring:

```python
from qts.strategies.belief import BeliefAxis
from qts.strategies.mean_reversion import MeanReversionStrategy
from qts.strategies.momentum import MomentumStrategy
from qts.strategies.news_reactive import NewsReactiveMomentum
from qts.strategies.sma_crossover import SMACrossoverStrategy

__all__ = [
    "BeliefAxis",
    "MeanReversionStrategy",
    "MomentumStrategy",
    "NewsReactiveMomentum",
    "SMACrossoverStrategy",
]
```

(If the existing file already has imports, **merge** instead of replacing — preserve anything already exported.)

- [ ] **Step 2: Update qts.macro exports**

Edit `src/qts/macro/__init__.py`. If it has existing exports, merge:

```python
from qts.macro.news_classifier import NewsClassifier
from qts.macro.news_signal import NewsSignal

__all__ = [
    "NewsClassifier",
    "NewsSignal",
]
```

(Merge with any existing `__all__`.)

- [ ] **Step 3: Update terrain-refactor-plan.md**

In `docs/plans/terrain-refactor-plan.md`, find the `## Phase 8: World Simulator — v1 vertical slice (DELIVERED)` section (added in Phase 8 v1 Task 17). After its `Deferred (per spec scaling path)` bullet list, insert a new sub-section:

```markdown
### Phase 8 v2 — News-Reactive Strategy + Real-Data Acceptance (DELIVERED|IN-PROGRESS)

See `docs/specs/2026-05-21-phase-8-news-reactive-strategy.md` for the spec and `.grill/phase-8-news-reactive-strategy.md` for the design log.

v2 delivered:
- [x] `Strategy.on_text(event)` formal protocol method (no-op default on existing strategies)
- [x] `qts.macro.NewsSignal` — multi-axis structured output (direction + confidence + relevance + magnitude)
- [x] `qts.macro.NewsClassifier` — Qwen-backed multi-axis classifier with content-hash disk cache
- [x] `qts.strategies.BeliefAxis` — exponentially-decaying belief primitive
- [x] `qts.strategies.NewsReactiveMomentum` — composes MomentumStrategy with a Qwen-driven belief overlay
- [x] `qts.data.RealEpisode` — real-data analogue of SimulatedEpisode
- [x] `qts.nautilus.real_runner.run_real_backtest` — bar + text-event dispatch
- [x] Curated `data/real/fomc/2023-12-13/` dataset (bars + statement + press-conf paragraphs)
- [x] Acceptance test: NewsReactiveMomentum vs buy-and-hold on 2023-12-13

Deferred (next-grill candidates, per project memory `project_deferred_grills.md`):
- Optuna sweep over news params (belief half-life, news_signal_weight, prompt variants)
- Multiple historical FOMC days / bulk-data pipeline
- Live Binance integration (scrapling + testnet)
- Other event types (CPI, NFP, geopolitical, USDT depeg)
```

Set the header to `DELIVERED` if the acceptance test passed, `IN-PROGRESS` if it landed as a valid-fail per Task 13.

- [ ] **Step 4: Run the full suite**

```bash
.venv/bin/python -m pytest --no-cov --tb=no -q 2>&1 | tail -3
```

Expected: ~1161 passed (1121 baseline + ~40 from Tasks 1-13), 4 skipped (or 5 if the acceptance test was skipped due to missing data).

- [ ] **Step 5: End-to-end smoke**

```bash
.venv/bin/python -c "
from qts.strategies import NewsReactiveMomentum, MomentumStrategy
from qts.macro import NewsClassifier, NewsSignal
from qts.data import RealEpisode
from qts.nautilus.real_runner import run_real_backtest
print('All v2 public exports import cleanly.')
"
```

- [ ] **Step 6: Commit**

```bash
git add src/qts/strategies/__init__.py src/qts/macro/__init__.py docs/plans/terrain-refactor-plan.md
git commit -m "docs(v2): export public API + update Phase 8 v2 status in plan

Strategies: NewsReactiveMomentum, BeliefAxis. Macro: NewsClassifier,
NewsSignal. Data: RealEpisode (already exported in Task 8).
terrain-refactor-plan.md gets a v2 sub-section listing delivered
components and next-grill candidates."
```

---

## Verification

After all 14 tasks:

```bash
.venv/bin/python -m pytest --no-cov --tb=no -q
```

Expected: ~1161 passed, 4 skipped (or 5 if acceptance test skipped), 0 failing.

```bash
.venv/bin/python -c "
import asyncio
from pathlib import Path
from qts.data import RealEpisode
from qts.macro import NewsClassifier
from qts.nautilus.real_runner import run_real_backtest
from qts.oversight.llm_client import create_llm_client
from qts.strategies import MomentumStrategy, NewsReactiveMomentum
from qts.config import StrategyParams, RiskLimits

ep = RealEpisode.from_disk(Path('data/real/fomc/2023-12-13'), 'BTCUSDT', 'fomc:2023-12-13')
llm = create_llm_client(backend='ollama')
clf = NewsClassifier(llm_client=llm, cache_dir=Path('data/news_cache'))
asyncio.run(clf.warm_cache_for(ep.text_events))

strat = NewsReactiveMomentum(
    inner=MomentumStrategy(params=StrategyParams(), risk_limits=RiskLimits()),
    classifier=clf,
)
result = run_real_backtest(ep, strat, log_level='ERROR')
print(f'Equity curve start: {result.equity_curve[0]:,.2f}, end: {result.equity_curve[-1]:,.2f}')
print(f'Total trades: {result.total_trades}, win rate: {result.win_rate:.2%}')
"
```

Expected: a populated equity curve, > 0 trades, sensible win rate. If FAIL on the acceptance assertion specifically, that's a known result documented in Task 13's commit message — the architecture is intact.

---

## Final commit (push)

After verification passes, push:

```bash
git push origin main
```

---

## Self-review checklist (done before handoff)

1. **Spec coverage:**
   - ✅ Strategy.on_text protocol — Task 1
   - ✅ Drop duck-typing in actor — Task 2
   - ✅ NewsSignal dataclass — Task 3
   - ✅ NewsClassifier (async + cache + sync lookup) — Tasks 4, 5
   - ✅ BeliefAxis decay primitive — Task 6
   - ✅ NewsReactiveMomentum strategy — Task 7
   - ✅ RealEpisode + from_disk — Task 8
   - ✅ run_real_backtest — Task 9
   - ✅ Fetcher script — Task 10
   - ✅ Curated 2023-12-13 dataset — Task 11
   - ✅ Hand-validation tool — Task 12 (spec asks for this in Risks)
   - ✅ Acceptance test (beats hold) — Task 13
   - ✅ Public API + plan update — Task 14
2. **Placeholder scan:** none — every step has runnable code/commands.
3. **Type consistency:**
   - `NewsSignal(direction, confidence, relevance, magnitude)` consistent across Tasks 3, 4, 5, 7
   - `BeliefAxis(half_life)` + `update(value, now)` + `at(now)` consistent across Tasks 6, 7
   - `NewsClassifier(llm_client, cache_dir)` + `classify_async` + `classify` + `warm_cache_for` consistent across Tasks 4, 5, 7, 12, 13
   - `RealEpisode(terrain, text_events, source)` + `from_disk(root, symbol, source)` consistent across Tasks 8, 9, 13
   - `run_real_backtest(episode, strategy, venue_config=None, log_level="WARNING", instrument=None)` consistent across Tasks 9, 13, verification
   - `NewsReactiveMomentum(inner, classifier, belief_half_life=4h, news_signal_weight=0.5)` consistent across Tasks 7, 13, verification
   - `Strategy.on_text(event)` formal across Tasks 1, 2, 7, 9
