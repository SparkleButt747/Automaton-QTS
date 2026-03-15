"""Trade attribution and failure-mode analysis engine."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from qts.models.base import (
    FailureMode,
    SignalSnapshot,
    TradeOutcome,
    TradeRecord,
    VolRegime,
)

logger = logging.getLogger(__name__)

# ── Classification thresholds (from spec) ────────────────────────────────────

# SENTIMENT_FADE: sentiment degraded by > 30% from entry to exit
_SENTIMENT_FADE_RATIO: float = 0.3
# NEWS_FRONT_RUN: entry price more than 2 ATR away from "fair" price
_NEWS_FRONT_RUN_ATR_MULTIPLIER: float = 2.0
# REGIME_MISMATCH: low volatility regime with large |sentiment| signal
_REGIME_MISMATCH_SENTIMENT_THRESHOLD: float = 0.5
# EXECUTION_SLIP: slippage exceeds the expected slippage budget
# (compared against zero expected slippage when not specified)
_DEFAULT_EXPECTED_SLIPPAGE: float = 0.0


# ── Report domain types ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FailureModeSummary:
    """Aggregated statistics for a single failure mode.

    Attributes:
        count: Number of trades classified with this failure mode.
        avg_pnl: Average P&L (in USD) across trades with this failure mode.
        pct_of_losses: Fraction of total losing trades attributed to this mode.
    """

    count: int
    avg_pnl: float
    pct_of_losses: float


@dataclass(frozen=True, slots=True)
class AttributionReport:
    """Full attribution analysis report for a collection of trades.

    Attributes:
        failure_mode_summary: Per-mode statistics keyed by :class:`~qts.models.base.FailureMode`.
        sentiment_hit_rate: Fraction of trades where sentiment sign matched trade direction.
        sentiment_pnl_correlation: Pearson correlation between ``sentiment_at_entry``
            and ``pnl_usd`` across all trades.
        total_trades: Total number of trades analysed.
        winning_trades: Number of trades with outcome WIN.
        losing_trades: Number of trades with outcome LOSS.
    """

    failure_mode_summary: dict[FailureMode, FailureModeSummary]
    sentiment_hit_rate: float
    sentiment_pnl_correlation: float
    total_trades: int
    winning_trades: int
    losing_trades: int


# ── Failure-mode classifier ────────────────────────────────────────────────────


class FailureModeClassifier:
    """Rule-based classifier that assigns a :class:`~qts.models.base.FailureMode` to a trade.

    Rules are evaluated in priority order; the first matching rule is returned.
    All losing trades receive a classification; winning trades are classified
    only when a rule explicitly matches.

    Classification rules (in priority order):

    1. **SENTIMENT_FADE**: ``sentiment_at_exit < -0.3 * sentiment_at_entry``
    2. **NEWS_FRONT_RUN**: ``|entry_price - fair_price| > 2 * atr``
       where *fair_price* is derived from the entry snapshot (``bb_upper + bb_lower) / 2``).
    3. **REGIME_MISMATCH**: ``vol_regime == LOW and |sentiment| > 0.5``
    4. **EXECUTION_SLIP**: ``slippage_usd > expected_slippage``
    5. **FALSE_ALPHA**: default fallback for losses with no other classification.
    """

    def classify(
        self,
        trade: TradeRecord,
        exit_snapshot: Optional[SignalSnapshot] = None,
    ) -> Optional[FailureMode]:
        """Classify the failure mode for a trade.

        Args:
            trade: Completed trade record to classify.
            exit_snapshot: Optional signal snapshot captured at trade exit.
                Required for SENTIMENT_FADE and NEWS_FRONT_RUN rules.

        Returns:
            :class:`~qts.models.base.FailureMode` for losing trades, or
            ``None`` for winning / breakeven trades with no rule match.
        """
        # ── Rule 1: SENTIMENT_FADE ──────────────────────────────────────────
        if exit_snapshot is not None:
            sentiment_at_exit = exit_snapshot.sentiment_score
            sentiment_at_entry = trade.sentiment_at_entry
            if sentiment_at_entry != 0.0:
                if sentiment_at_exit < -_SENTIMENT_FADE_RATIO * sentiment_at_entry:
                    logger.debug(
                        "SENTIMENT_FADE: trade=%s exit_sentiment=%.3f entry_sentiment=%.3f",
                        trade.trade_id,
                        sentiment_at_exit,
                        sentiment_at_entry,
                    )
                    return FailureMode.SENTIMENT_FADE

        # ── Rule 2: NEWS_FRONT_RUN ──────────────────────────────────────────
        if exit_snapshot is not None:
            # Approximate "fair price" as the midpoint of the Bollinger Bands
            # at entry (captured in exit snapshot as a proxy for at-entry state).
            fair_price = (exit_snapshot.bb_upper + exit_snapshot.bb_lower) / 2.0
            atr = exit_snapshot.atr
            if atr > 0.0:
                if abs(trade.entry_price - fair_price) > _NEWS_FRONT_RUN_ATR_MULTIPLIER * atr:
                    logger.debug(
                        "NEWS_FRONT_RUN: trade=%s |entry-fair|=%.3f > 2*atr=%.3f",
                        trade.trade_id,
                        abs(trade.entry_price - fair_price),
                        _NEWS_FRONT_RUN_ATR_MULTIPLIER * atr,
                    )
                    return FailureMode.NEWS_FRONT_RUN

        # ── Rule 3: REGIME_MISMATCH ─────────────────────────────────────────
        if (
            trade.vol_regime_at_entry == VolRegime.LOW
            and abs(trade.sentiment_at_entry) > _REGIME_MISMATCH_SENTIMENT_THRESHOLD
        ):
            logger.debug(
                "REGIME_MISMATCH: trade=%s regime=LOW sentiment=%.3f",
                trade.trade_id,
                trade.sentiment_at_entry,
            )
            return FailureMode.REGIME_MISMATCH

        # ── Rule 4: EXECUTION_SLIP ──────────────────────────────────────────
        if trade.slippage_usd > _DEFAULT_EXPECTED_SLIPPAGE:
            logger.debug(
                "EXECUTION_SLIP: trade=%s slippage=%.4f",
                trade.trade_id,
                trade.slippage_usd,
            )
            return FailureMode.EXECUTION_SLIP

        # ── Rule 5: FALSE_ALPHA (default for losses) ─────────────────────────
        if trade.outcome == TradeOutcome.LOSS:
            logger.debug("FALSE_ALPHA: trade=%s (default loss classification)", trade.trade_id)
            return FailureMode.FALSE_ALPHA

        # Winning / breakeven trades with no rule match get None
        return None


# ── Attribution engine ─────────────────────────────────────────────────────────


class AttributionEngine:
    """Runs post-trade attribution analysis over a collection of trades.

    Computes per-failure-mode statistics, sentiment hit rate, and the
    Pearson correlation between sentiment scores and P&L.

    Args:
        classifier: :class:`FailureModeClassifier` to use for per-trade
            classification.  Defaults to a fresh instance.
    """

    def __init__(self, classifier: Optional[FailureModeClassifier] = None) -> None:
        self._classifier = classifier or FailureModeClassifier()

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _pearson_correlation(xs: list[float], ys: list[float]) -> float:
        """Compute Pearson correlation coefficient for two equal-length lists.

        Args:
            xs: First variable list.
            ys: Second variable list.

        Returns:
            Pearson r in [-1, 1], or 0.0 if variance is zero / lists are empty.
        """
        n = len(xs)
        if n < 2 or len(ys) != n:
            return 0.0

        mean_x = sum(xs) / n
        mean_y = sum(ys) / n

        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)

        denominator = math.sqrt(var_x * var_y)
        if denominator == 0.0:
            return 0.0

        return cov / denominator

    @staticmethod
    def _sentiment_hit_rate(trades: list[TradeRecord]) -> float:
        """Compute fraction of trades where sentiment sign matched direction.

        A "hit" is defined as: a LONG trade with positive sentiment, or a SHORT
        trade with negative sentiment.  Trades with zero sentiment are excluded.

        Args:
            trades: List of :class:`~qts.models.base.TradeRecord` objects.

        Returns:
            Hit rate in [0, 1].  Returns 0.0 if all sentiments are zero.
        """
        from qts.models.base import TradeDirection

        eligible = [t for t in trades if t.sentiment_at_entry != 0.0]
        if not eligible:
            return 0.0

        hits = 0
        for trade in eligible:
            if trade.direction == TradeDirection.LONG and trade.sentiment_at_entry > 0.0:
                hits += 1
            elif trade.direction == TradeDirection.SHORT and trade.sentiment_at_entry < 0.0:
                hits += 1

        return hits / len(eligible)

    # ── Public API ────────────────────────────────────────────────────────────

    def run_attribution(
        self,
        trades: list[TradeRecord],
        snapshots: dict[str, SignalSnapshot],
    ) -> AttributionReport:
        """Run full attribution analysis over *trades*.

        Args:
            trades: List of completed :class:`~qts.models.base.TradeRecord` objects.
            snapshots: Mapping of ``trade_id -> exit SignalSnapshot``.  Trades
                whose IDs are absent from this dict are classified without an
                exit snapshot.

        Returns:
            :class:`AttributionReport` with per-mode summaries and aggregate metrics.
        """
        winning = [t for t in trades if t.outcome == TradeOutcome.WIN]
        losing = [t for t in trades if t.outcome == TradeOutcome.LOSS]
        n_losses = len(losing)

        # Classify each trade
        mode_pnls: dict[FailureMode, list[float]] = {m: [] for m in FailureMode}

        for trade in trades:
            exit_snap = snapshots.get(trade.trade_id)
            mode = self._classifier.classify(trade, exit_snapshot=exit_snap)
            if mode is not None:
                mode_pnls[mode].append(trade.pnl_usd)

        # Build per-mode summaries
        summaries: dict[FailureMode, FailureModeSummary] = {}
        for mode, pnls in mode_pnls.items():
            if not pnls:
                continue
            count = len(pnls)
            avg_pnl = sum(pnls) / count
            pct_of_losses = count / n_losses if n_losses > 0 else 0.0
            summaries[mode] = FailureModeSummary(
                count=count,
                avg_pnl=avg_pnl,
                pct_of_losses=pct_of_losses,
            )

        # Aggregate metrics
        sentiments = [t.sentiment_at_entry for t in trades]
        pnls = [t.pnl_usd for t in trades]
        corr = self._pearson_correlation(sentiments, pnls)
        hit_rate = self._sentiment_hit_rate(trades)

        return AttributionReport(
            failure_mode_summary=summaries,
            sentiment_hit_rate=hit_rate,
            sentiment_pnl_correlation=corr,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=n_losses,
        )
