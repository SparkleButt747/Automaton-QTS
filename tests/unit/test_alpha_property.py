"""Property-based tests for combined alpha signal."""

from __future__ import annotations

from datetime import datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from qts.config import SentimentFusionWeights, SignalWeights, StrategyParams
from qts.models.base import SignalSnapshot, VolLevel
from qts.signals.alpha import combined_alpha

# ── Strategies ────────────────────────────────────────────────────────────────

_vol_level_st = st.sampled_from([VolLevel.HIGH, VolLevel.LOW, VolLevel.TRANSITIONING])


def _signal_snapshot_strategy() -> st.SearchStrategy:  # type: ignore[type-arg]
    """Generate random SignalSnapshots with valid field ranges."""
    return st.builds(
        SignalSnapshot,
        timestamp=st.just(datetime(2024, 1, 1)),
        symbol=st.just("PROP"),
        rsi=st.floats(min_value=0.0, max_value=100.0, allow_nan=False),
        macd_histogram=st.floats(
            min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        macd_line=st.floats(
            min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        macd_signal=st.floats(
            min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False
        ),
        bb_position=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        bb_upper=st.floats(
            min_value=1.0, max_value=200_000.0, allow_nan=False, allow_infinity=False
        ),
        bb_lower=st.floats(
            min_value=1.0, max_value=200_000.0, allow_nan=False, allow_infinity=False
        ),
        atr=st.floats(min_value=0.0, max_value=10_000.0, allow_nan=False, allow_infinity=False),
        momentum_5=st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        vol_level=_vol_level_st,
        vol_level_confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        sentiment_score=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        combined_alpha=st.just(0.0),
    )


def _weights_strategy() -> st.SearchStrategy:  # type: ignore[type-arg]
    """Generate random SignalWeights with all weights summing to 1.0."""

    # Draw 5 non-negative floats, then normalize them to sum to 1.
    def build_weights(raw: list[float]) -> SignalWeights:
        total = sum(raw)
        if total < 1e-9:
            raw = [0.2, 0.2, 0.2, 0.2, 0.2]
            total = 1.0
        normed = [v / total for v in raw]
        # Adjust last weight to ensure exact sum = 1.0; clamp to 0.0 to prevent
        # tiny negative values from floating-point rounding errors.
        normed[-1] = max(0.0, 1.0 - sum(normed[:-1]))
        return SignalWeights(
            w_rsi=normed[0],
            w_macd=normed[1],
            w_bb=normed[2],
            w_mom=normed[3],
            w_sentiment=normed[4],
        )

    return st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=5,
    ).map(build_weights)


def _strategy_params_strategy() -> st.SearchStrategy:  # type: ignore[type-arg]
    """Generate random StrategyParams with valid weights."""
    return st.builds(
        lambda weights: StrategyParams(
            version="prop_test",
            weights=weights,
            entry_threshold=0.25,
            exit_threshold=-0.10,
            max_hold_bars=48,
            sentiment_fusion_weights=SentimentFusionWeights(news=0.4, social=0.3, geopolitical=0.3),
        ),
        weights=_weights_strategy(),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestCombinedAlphaProperty:
    @given(
        snapshot=_signal_snapshot_strategy(),
        params=_strategy_params_strategy(),
    )
    @settings(max_examples=500)
    def test_combined_alpha_always_in_minus1_to_1(
        self, snapshot: SignalSnapshot, params: StrategyParams
    ) -> None:
        """combined_alpha must always return a value in [-1, 1] for any valid input."""
        result = combined_alpha(snapshot, params)
        assert -1.0 <= result <= 1.0, (
            f"combined_alpha out of [-1, 1]: {result}\n"
            f"  snapshot={snapshot}\n"
            f"  params.weights={params.weights}"
        )

    @given(
        snapshot=_signal_snapshot_strategy(),
        params=_strategy_params_strategy(),
    )
    @settings(max_examples=200)
    def test_combined_alpha_is_finite(
        self, snapshot: SignalSnapshot, params: StrategyParams
    ) -> None:
        """combined_alpha must always return a finite (not NaN, not inf) value."""
        import math

        result = combined_alpha(snapshot, params)
        assert math.isfinite(result), f"combined_alpha returned non-finite: {result}"

    @given(
        params=_strategy_params_strategy(),
    )
    @settings(max_examples=200)
    def test_neutral_snapshot_returns_near_zero(self, params: StrategyParams) -> None:
        """A perfectly neutral snapshot (RSI=50, all zeros) should return 0.0."""
        snapshot = SignalSnapshot(
            timestamp=datetime(2024, 1, 1),
            symbol="TEST",
            rsi=50.0,
            macd_histogram=0.0,
            macd_line=0.0,
            macd_signal=0.0,
            bb_position=0.5,
            bb_upper=110.0,
            bb_lower=90.0,
            atr=1.0,
            momentum_5=0.0,
            vol_level=VolLevel.HIGH,
            vol_level_confidence=0.9,
            sentiment_score=0.0,
            combined_alpha=0.0,
        )
        result = combined_alpha(snapshot, params)
        assert result == 0.0, f"Neutral snapshot returned {result}, expected 0.0"

    @given(
        snapshot=_signal_snapshot_strategy(),
        params=_strategy_params_strategy(),
    )
    @settings(max_examples=200)
    def test_low_regime_le_high_regime(
        self, snapshot: SignalSnapshot, params: StrategyParams
    ) -> None:
        """Signal in LOW regime must have absolute value <= HIGH regime signal."""
        snap_high = SignalSnapshot(
            timestamp=snapshot.timestamp,
            symbol=snapshot.symbol,
            rsi=snapshot.rsi,
            macd_histogram=snapshot.macd_histogram,
            macd_line=snapshot.macd_line,
            macd_signal=snapshot.macd_signal,
            bb_position=snapshot.bb_position,
            bb_upper=snapshot.bb_upper,
            bb_lower=snapshot.bb_lower,
            atr=snapshot.atr,
            momentum_5=snapshot.momentum_5,
            vol_level=VolLevel.HIGH,
            vol_level_confidence=snapshot.vol_level_confidence,
            sentiment_score=snapshot.sentiment_score,
            combined_alpha=0.0,
        )
        snap_low = SignalSnapshot(
            timestamp=snapshot.timestamp,
            symbol=snapshot.symbol,
            rsi=snapshot.rsi,
            macd_histogram=snapshot.macd_histogram,
            macd_line=snapshot.macd_line,
            macd_signal=snapshot.macd_signal,
            bb_position=snapshot.bb_position,
            bb_upper=snapshot.bb_upper,
            bb_lower=snapshot.bb_lower,
            atr=snapshot.atr,
            momentum_5=snapshot.momentum_5,
            vol_level=VolLevel.LOW,
            vol_level_confidence=snapshot.vol_level_confidence,
            sentiment_score=snapshot.sentiment_score,
            combined_alpha=0.0,
        )
        alpha_high = combined_alpha(snap_high, params)
        alpha_low = combined_alpha(snap_low, params)
        # LOW regime uses 0.5 scalar, HIGH uses 1.0, so |alpha_low| <= |alpha_high|
        assert abs(alpha_low) <= abs(alpha_high) + 1e-9, (
            f"|alpha_low|={abs(alpha_low)} > |alpha_high|={abs(alpha_high)}"
        )
