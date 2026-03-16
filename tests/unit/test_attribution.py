"""Unit tests for qts.analytics.attribution.

Tests:
- Each failure mode rule triggers on synthetic trade data
- 100% of loss trades get classified (no None failure_mode for losses)
- Attribution report aggregates correctly (counts, percentages, metrics)
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime

import pytest

from qts.analytics.attribution import (
    AttributionEngine,
    FailureModeClassifier,
    FailureModeSummary,
)
from qts.models.base import (
    ExitReason,
    FailureMode,
    SignalSnapshot,
    TradeDirection,
    TradeOutcome,
    TradeRecord,
    VolRegime,
)

# ── Helpers / Factories ───────────────────────────────────────────────────────


def _make_trade(
    *,
    pnl_usd: float = -100.0,
    outcome: TradeOutcome = TradeOutcome.LOSS,
    sentiment_at_entry: float = 0.1,
    vol_regime_at_entry: VolRegime = VolRegime.HIGH,
    slippage_usd: float = 0.0,
    entry_price: float = 100.0,
    direction: TradeDirection = TradeDirection.LONG,
) -> TradeRecord:
    """Create a minimal TradeRecord for testing."""
    return TradeRecord(
        trade_id=str(uuid.uuid4()),
        symbol="AAPL",
        entry_time=datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC),
        exit_time=datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC),
        direction=direction,
        entry_price=entry_price,
        exit_price=entry_price * (1.0 + pnl_usd / (entry_price * 1.0)),
        size=1.0,
        pnl_usd=pnl_usd,
        pnl_pct=pnl_usd / (entry_price * 1.0),
        slippage_usd=slippage_usd,
        fees_usd=0.0,
        rsi_at_entry=50.0,
        macd_at_entry=0.0,
        sentiment_at_entry=sentiment_at_entry,
        social_velocity_at_entry=0.0,
        gdelt_intensity_at_entry=0.0,
        vol_regime_at_entry=vol_regime_at_entry,
        combined_alpha_at_entry=0.0,
        params_version="v1.0",
        outcome=outcome,
        exit_reason=ExitReason.STOP_LOSS,
        failure_mode=None,
    )


def _make_snapshot(
    *,
    sentiment_score: float = 0.0,
    bb_upper: float = 110.0,
    bb_lower: float = 90.0,
    atr: float = 5.0,
    vol_regime: VolRegime = VolRegime.HIGH,
) -> SignalSnapshot:
    """Create a minimal SignalSnapshot for testing."""
    return SignalSnapshot(
        timestamp=datetime(2024, 1, 1, 11, 0, 0, tzinfo=UTC),
        symbol="AAPL",
        rsi=50.0,
        macd_histogram=0.0,
        macd_line=0.0,
        macd_signal=0.0,
        bb_position=0.5,
        bb_upper=bb_upper,
        bb_lower=bb_lower,
        atr=atr,
        momentum_5=0.0,
        vol_regime=vol_regime,
        vol_regime_confidence=0.9,
        sentiment_score=sentiment_score,
        combined_alpha=0.0,
    )


# ── FailureModeClassifier — individual rule tests ─────────────────────────────


class TestSentimentFadeRule:
    """Rule: sentiment_at_exit < -0.3 * sentiment_at_entry -> SENTIMENT_FADE."""

    def test_triggers_when_sentiment_fades(self) -> None:
        classifier = FailureModeClassifier()
        # Entry sentiment=0.8, exit sentiment must be < -0.3 * 0.8 = -0.24
        trade = _make_trade(sentiment_at_entry=0.8)
        exit_snap = _make_snapshot(sentiment_score=-0.5)  # -0.5 < -0.24 -> triggers

        result = classifier.classify(trade, exit_snapshot=exit_snap)
        assert result == FailureMode.SENTIMENT_FADE

    def test_does_not_trigger_when_sentiment_stable(self) -> None:
        classifier = FailureModeClassifier()
        trade = _make_trade(sentiment_at_entry=0.8, slippage_usd=0.0)
        # exit = 0.0 which is NOT < -0.24
        exit_snap = _make_snapshot(sentiment_score=0.0)

        result = classifier.classify(trade, exit_snapshot=exit_snap)
        assert result != FailureMode.SENTIMENT_FADE

    def test_does_not_trigger_when_entry_sentiment_zero(self) -> None:
        """When entry_sentiment is 0, the fade rule cannot divide and should be skipped."""
        classifier = FailureModeClassifier()
        trade = _make_trade(sentiment_at_entry=0.0)
        exit_snap = _make_snapshot(sentiment_score=-0.9)

        result = classifier.classify(trade, exit_snapshot=exit_snap)
        assert result != FailureMode.SENTIMENT_FADE


class TestNewsFrontRunRule:
    """Rule: |entry_price - fair_price| > 2 * atr -> NEWS_FRONT_RUN."""

    def test_triggers_when_far_from_fair_price(self) -> None:
        classifier = FailureModeClassifier()
        # fair_price = (110 + 90) / 2 = 100.0
        # entry_price = 115.0 -> |115-100| = 15 > 2*5 = 10 -> triggers
        trade = _make_trade(entry_price=115.0, sentiment_at_entry=0.1)
        exit_snap = _make_snapshot(bb_upper=110.0, bb_lower=90.0, atr=5.0)

        result = classifier.classify(trade, exit_snapshot=exit_snap)
        assert result == FailureMode.NEWS_FRONT_RUN

    def test_does_not_trigger_when_close_to_fair_price(self) -> None:
        classifier = FailureModeClassifier()
        # entry_price = 101.0 -> |101-100| = 1 < 10 -> no trigger
        trade = _make_trade(entry_price=101.0, sentiment_at_entry=0.1)
        exit_snap = _make_snapshot(bb_upper=110.0, bb_lower=90.0, atr=5.0)

        result = classifier.classify(trade, exit_snapshot=exit_snap)
        assert result != FailureMode.NEWS_FRONT_RUN

    def test_does_not_trigger_when_atr_zero(self) -> None:
        """When ATR is zero, the rule cannot compare and should be skipped."""
        classifier = FailureModeClassifier()
        trade = _make_trade(entry_price=200.0, sentiment_at_entry=0.1)
        exit_snap = _make_snapshot(bb_upper=110.0, bb_lower=90.0, atr=0.0)

        result = classifier.classify(trade, exit_snapshot=exit_snap)
        assert result != FailureMode.NEWS_FRONT_RUN


class TestRegimeMismatchRule:
    """Rule: vol_regime == LOW and |sentiment| > 0.5 -> REGIME_MISMATCH."""

    def test_triggers_with_low_regime_and_high_sentiment(self) -> None:
        classifier = FailureModeClassifier()
        trade = _make_trade(
            vol_regime_at_entry=VolRegime.LOW,
            sentiment_at_entry=0.8,  # |0.8| > 0.5
        )
        result = classifier.classify(trade, exit_snapshot=None)
        assert result == FailureMode.REGIME_MISMATCH

    def test_triggers_with_low_regime_and_negative_high_sentiment(self) -> None:
        classifier = FailureModeClassifier()
        trade = _make_trade(
            vol_regime_at_entry=VolRegime.LOW,
            sentiment_at_entry=-0.7,  # |-0.7| > 0.5
        )
        result = classifier.classify(trade, exit_snapshot=None)
        assert result == FailureMode.REGIME_MISMATCH

    def test_does_not_trigger_with_high_regime(self) -> None:
        classifier = FailureModeClassifier()
        trade = _make_trade(
            vol_regime_at_entry=VolRegime.HIGH,
            sentiment_at_entry=0.8,
            slippage_usd=0.0,
        )
        result = classifier.classify(trade, exit_snapshot=None)
        assert result != FailureMode.REGIME_MISMATCH

    def test_does_not_trigger_with_low_sentiment(self) -> None:
        classifier = FailureModeClassifier()
        trade = _make_trade(
            vol_regime_at_entry=VolRegime.LOW,
            sentiment_at_entry=0.3,  # |0.3| < 0.5
            slippage_usd=0.0,
        )
        result = classifier.classify(trade, exit_snapshot=None)
        assert result != FailureMode.REGIME_MISMATCH


class TestExecutionSlipRule:
    """Rule: slippage > 0 -> EXECUTION_SLIP."""

    def test_triggers_when_slippage_positive(self) -> None:
        classifier = FailureModeClassifier()
        trade = _make_trade(
            slippage_usd=5.0,
            vol_regime_at_entry=VolRegime.HIGH,
            sentiment_at_entry=0.1,
        )
        result = classifier.classify(trade, exit_snapshot=None)
        assert result == FailureMode.EXECUTION_SLIP

    def test_does_not_trigger_when_zero_slippage(self) -> None:
        classifier = FailureModeClassifier()
        trade = _make_trade(slippage_usd=0.0, sentiment_at_entry=0.0)
        result = classifier.classify(trade, exit_snapshot=None)
        assert result != FailureMode.EXECUTION_SLIP


class TestFalseAlphaRule:
    """Rule: default fallback for losses with no other classification."""

    def test_triggers_as_default_for_loss(self) -> None:
        classifier = FailureModeClassifier()
        # Trade that matches no higher-priority rule
        trade = _make_trade(
            outcome=TradeOutcome.LOSS,
            sentiment_at_entry=0.0,
            vol_regime_at_entry=VolRegime.HIGH,
            slippage_usd=0.0,
        )
        result = classifier.classify(trade, exit_snapshot=None)
        assert result == FailureMode.FALSE_ALPHA

    def test_does_not_trigger_for_win_with_no_rules(self) -> None:
        classifier = FailureModeClassifier()
        trade = _make_trade(
            outcome=TradeOutcome.WIN,
            pnl_usd=100.0,
            sentiment_at_entry=0.0,
            vol_regime_at_entry=VolRegime.HIGH,
            slippage_usd=0.0,
        )
        result = classifier.classify(trade, exit_snapshot=None)
        assert result is None


# ── All losses get classified ─────────────────────────────────────────────────


class TestAllLossesClassified:
    """100% of loss trades must receive a non-None failure mode."""

    def test_single_loss_classified(self) -> None:
        classifier = FailureModeClassifier()
        trade = _make_trade(outcome=TradeOutcome.LOSS)
        assert classifier.classify(trade) is not None

    def test_batch_of_losses_all_classified(self) -> None:
        classifier = FailureModeClassifier()
        losses = [
            _make_trade(outcome=TradeOutcome.LOSS, sentiment_at_entry=0.0, slippage_usd=0.0),
            _make_trade(
                outcome=TradeOutcome.LOSS, sentiment_at_entry=0.8, vol_regime_at_entry=VolRegime.LOW
            ),
            _make_trade(outcome=TradeOutcome.LOSS, slippage_usd=10.0),
        ]
        for trade in losses:
            mode = classifier.classify(trade)
            assert mode is not None, f"Loss trade {trade.trade_id} was not classified"

    def test_diverse_loss_scenarios_all_classified(self) -> None:
        """Various loss scenarios — all must yield a non-None classification."""
        classifier = FailureModeClassifier()
        scenarios = [
            # Minimal loss — fallback to FALSE_ALPHA
            _make_trade(outcome=TradeOutcome.LOSS, sentiment_at_entry=0.0, slippage_usd=0.0),
            # Regime mismatch
            _make_trade(
                outcome=TradeOutcome.LOSS,
                vol_regime_at_entry=VolRegime.LOW,
                sentiment_at_entry=0.9,
            ),
            # Execution slip
            _make_trade(outcome=TradeOutcome.LOSS, slippage_usd=50.0, sentiment_at_entry=0.0),
        ]
        for t in scenarios:
            assert classifier.classify(t) is not None


# ── AttributionReport aggregation ────────────────────────────────────────────


class TestAttributionReport:
    def _build_engine_and_trades(
        self,
    ) -> tuple[AttributionEngine, list[TradeRecord], dict[str, SignalSnapshot]]:
        engine = AttributionEngine()

        # 3 winning trades (no special classification expected)
        wins = [
            _make_trade(
                pnl_usd=100.0,
                outcome=TradeOutcome.WIN,
                sentiment_at_entry=0.5,
                direction=TradeDirection.LONG,
            )
            for _ in range(3)
        ]

        # 2 losses: one REGIME_MISMATCH, one FALSE_ALPHA
        loss_regime = _make_trade(
            pnl_usd=-50.0,
            outcome=TradeOutcome.LOSS,
            vol_regime_at_entry=VolRegime.LOW,
            sentiment_at_entry=0.9,
        )
        loss_false_alpha = _make_trade(
            pnl_usd=-30.0,
            outcome=TradeOutcome.LOSS,
            sentiment_at_entry=0.0,
            slippage_usd=0.0,
        )

        trades = wins + [loss_regime, loss_false_alpha]
        snapshots: dict[str, SignalSnapshot] = {}
        return engine, trades, snapshots

    def test_report_total_trades(self) -> None:
        engine, trades, snapshots = self._build_engine_and_trades()
        report = engine.run_attribution(trades, snapshots)
        assert report.total_trades == 5

    def test_report_winning_trades(self) -> None:
        engine, trades, snapshots = self._build_engine_and_trades()
        report = engine.run_attribution(trades, snapshots)
        assert report.winning_trades == 3

    def test_report_losing_trades(self) -> None:
        engine, trades, snapshots = self._build_engine_and_trades()
        report = engine.run_attribution(trades, snapshots)
        assert report.losing_trades == 2

    def test_failure_mode_summary_contains_modes(self) -> None:
        engine, trades, snapshots = self._build_engine_and_trades()
        report = engine.run_attribution(trades, snapshots)
        # At least REGIME_MISMATCH and FALSE_ALPHA should appear
        assert FailureMode.REGIME_MISMATCH in report.failure_mode_summary
        assert FailureMode.FALSE_ALPHA in report.failure_mode_summary

    def test_failure_mode_pct_sums_to_one_for_all_losses(self) -> None:
        """Sum of pct_of_losses across all modes should equal 1.0 when all losses classified."""
        engine, trades, snapshots = self._build_engine_and_trades()
        report = engine.run_attribution(trades, snapshots)
        total_pct = sum(s.pct_of_losses for s in report.failure_mode_summary.values())
        # All losses should be accounted for
        assert math.isclose(total_pct, 1.0, abs_tol=1e-6)

    def test_failure_mode_counts_match_trades(self) -> None:
        engine, trades, snapshots = self._build_engine_and_trades()
        report = engine.run_attribution(trades, snapshots)
        total_classified = sum(s.count for s in report.failure_mode_summary.values())
        # 3 wins (possibly classified by a rule) + 2 losses (definitely classified)
        # At minimum the 2 losses must be classified
        assert total_classified >= 2

    def test_avg_pnl_per_mode(self) -> None:
        engine, trades, snapshots = self._build_engine_and_trades()
        report = engine.run_attribution(trades, snapshots)
        regime_summary = report.failure_mode_summary.get(FailureMode.REGIME_MISMATCH)
        if regime_summary:
            assert math.isclose(regime_summary.avg_pnl, -50.0, abs_tol=1e-6)

    def test_sentiment_hit_rate_in_range(self) -> None:
        engine, trades, snapshots = self._build_engine_and_trades()
        report = engine.run_attribution(trades, snapshots)
        assert 0.0 <= report.sentiment_hit_rate <= 1.0

    def test_sentiment_pnl_correlation_in_range(self) -> None:
        engine, trades, snapshots = self._build_engine_and_trades()
        report = engine.run_attribution(trades, snapshots)
        assert -1.0 <= report.sentiment_pnl_correlation <= 1.0

    def test_empty_trades_returns_empty_report(self) -> None:
        engine = AttributionEngine()
        report = engine.run_attribution([], {})
        assert report.total_trades == 0
        assert report.winning_trades == 0
        assert report.losing_trades == 0
        assert report.failure_mode_summary == {}

    def test_sentiment_fade_in_report(self) -> None:
        """SENTIMENT_FADE should appear in the report when that rule triggers."""
        engine = AttributionEngine()

        # Create a trade where sentiment fades: entry=0.8, exit=-0.5
        trade = _make_trade(
            pnl_usd=-100.0,
            outcome=TradeOutcome.LOSS,
            sentiment_at_entry=0.8,
        )
        exit_snap = _make_snapshot(sentiment_score=-0.5)

        report = engine.run_attribution([trade], {trade.trade_id: exit_snap})
        assert FailureMode.SENTIMENT_FADE in report.failure_mode_summary

    def test_execution_slip_in_report(self) -> None:
        """EXECUTION_SLIP should appear in the report when slippage > 0."""
        engine = AttributionEngine()
        trade = _make_trade(
            pnl_usd=-200.0,
            outcome=TradeOutcome.LOSS,
            slippage_usd=15.0,
            sentiment_at_entry=0.0,
        )
        report = engine.run_attribution([trade], {})
        assert FailureMode.EXECUTION_SLIP in report.failure_mode_summary


# ── Attribution report dataclass ──────────────────────────────────────────────


class TestAttributionReportDataclass:
    def test_report_is_frozen(self) -> None:
        engine = AttributionEngine()
        report = engine.run_attribution([], {})
        with pytest.raises(AttributeError):
            report.total_trades = 99  # type: ignore[misc]

    def test_failure_mode_summary_is_dataclass(self) -> None:
        engine = AttributionEngine()
        trade = _make_trade(outcome=TradeOutcome.LOSS)
        report = engine.run_attribution([trade], {})
        for summary in report.failure_mode_summary.values():
            assert isinstance(summary, FailureModeSummary)

    def test_failure_mode_summary_frozen(self) -> None:
        summary = FailureModeSummary(count=3, avg_pnl=-50.0, pct_of_losses=0.5)
        with pytest.raises(AttributeError):
            summary.count = 99  # type: ignore[misc]
