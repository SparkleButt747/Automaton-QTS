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
        inner: MomentumStrategy,
        classifier: NewsClassifier,
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
        bar: Bar,
        snapshot: SignalSnapshot,
        positions: list[Position],
    ) -> list[Order]:
        news_alpha = self.news_alpha_at(bar.timestamp)
        blended = (1.0 - self._weight) * snapshot.combined_alpha + self._weight * news_alpha
        blended_snapshot = replace(snapshot, combined_alpha=blended)
        return self._inner.on_bar(bar, blended_snapshot, positions)

    def on_fill(self, fill: Fill) -> None:
        self._inner.on_fill(fill)

    def on_text(self, event: TextEvent) -> None:
        """Classify the event (cache-only) and update the belief."""
        signal = self._classifier.classify(event)
        self._belief.update(value=signal.alpha_contribution(), now=event.timestamp)

    # ---- Inspection -----------------------------------------------------------

    def news_alpha_at(self, now: datetime) -> float:
        """Return the current decayed news alpha (for tests + diagnostics)."""
        return self._belief.at(now)
