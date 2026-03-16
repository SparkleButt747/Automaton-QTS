"""Property-based tests for risk management."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from qts.config import RiskLimits
from qts.execution.risk import RiskManager
from qts.models.base import Position, TradeDirection

# ── Strategies ────────────────────────────────────────────────────────────────

_POSITIVE_FLOAT = st.floats(min_value=1e-6, max_value=1e9, allow_nan=False, allow_infinity=False)
_FRACTION_FLOAT = st.floats(min_value=1e-4, max_value=0.5, allow_nan=False, allow_infinity=False)


def _risk_limits_strategy() -> st.SearchStrategy:  # type: ignore[type-arg]
    """Generate valid RiskLimits instances."""
    return st.builds(
        RiskLimits,
        max_daily_drawdown_pct=_FRACTION_FLOAT,
        max_position_size_pct=_FRACTION_FLOAT,
        max_open_positions=st.integers(min_value=1, max_value=20),
        circuit_breaker_cooldown_seconds=st.integers(min_value=0, max_value=86400),
        sentiment_signal_max_scalar=st.floats(min_value=1.0, max_value=10.0, allow_nan=False),
    )


def _position_strategy() -> st.SearchStrategy:  # type: ignore[type-arg]
    """Generate a valid Position object."""
    return st.builds(
        Position,
        symbol=st.text(alphabet=st.characters(whitelist_categories=("Lu",)), min_size=1, max_size=6),
        direction=st.sampled_from([TradeDirection.LONG, TradeDirection.SHORT]),
        entry_price=_POSITIVE_FLOAT,
        quantity=_POSITIVE_FLOAT,
        entry_time=st.just(datetime(2024, 1, 1)),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestCircuitBreakerProperty:
    @given(
        limits=_risk_limits_strategy(),
        portfolio_value=_POSITIVE_FLOAT,
    )
    @settings(max_examples=300)
    def test_circuit_breaker_always_triggers_when_drawdown_exceeds_limit(
        self,
        limits: RiskLimits,
        portfolio_value: float,
    ) -> None:
        """Circuit breaker must always trigger when drawdown >= limit."""
        rm = RiskManager(limits)
        # Compute a loss that exactly meets or exceeds the drawdown limit
        drawdown_loss = portfolio_value * limits.max_daily_drawdown_pct
        result = rm.check_daily_drawdown(-drawdown_loss, portfolio_value)
        assert result is False, (
            f"Expected circuit breaker to trigger: "
            f"loss={drawdown_loss}, portfolio={portfolio_value}, "
            f"limit={limits.max_daily_drawdown_pct}"
        )
        assert rm._halted is True, "Expected rm._halted to be True after drawdown exceeded"

    @given(
        limits=_risk_limits_strategy(),
        portfolio_value=_POSITIVE_FLOAT,
        loss_multiplier=st.floats(min_value=1.0, max_value=10.0, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_circuit_breaker_triggers_for_any_excess_loss(
        self,
        limits: RiskLimits,
        portfolio_value: float,
        loss_multiplier: float,
    ) -> None:
        """Circuit breaker triggers for any loss >= max_daily_drawdown_pct * portfolio."""
        rm = RiskManager(limits)
        excess_loss = portfolio_value * limits.max_daily_drawdown_pct * loss_multiplier
        result = rm.check_daily_drawdown(-excess_loss, portfolio_value)
        assert result is False
        assert rm._halted is True

    @given(
        limits=_risk_limits_strategy(),
        portfolio_value=_POSITIVE_FLOAT,
        loss_fraction=st.floats(min_value=0.0, max_value=0.999, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_no_circuit_breaker_when_drawdown_below_limit(
        self,
        limits: RiskLimits,
        portfolio_value: float,
        loss_fraction: float,
    ) -> None:
        """Circuit breaker must NOT trigger when drawdown is strictly below limit."""
        assume(loss_fraction < limits.max_daily_drawdown_pct)
        rm = RiskManager(limits)
        small_loss = portfolio_value * loss_fraction
        result = rm.check_daily_drawdown(-small_loss, portfolio_value)
        assert result is True
        assert rm._halted is False


class TestPositionSizeProperty:
    @given(
        limits=_risk_limits_strategy(),
        portfolio_value=_POSITIVE_FLOAT,
        size_multiplier=st.floats(min_value=1.001, max_value=100.0, allow_nan=False),
    )
    @settings(max_examples=300)
    def test_check_position_size_always_rejects_when_exceeds_limit(
        self,
        limits: RiskLimits,
        portfolio_value: float,
        size_multiplier: float,
    ) -> None:
        """check_position_size must always reject when size > limit."""
        rm = RiskManager(limits)
        # size is strictly larger than allowed maximum
        over_limit_size = portfolio_value * limits.max_position_size_pct * size_multiplier
        result = rm.check_position_size(over_limit_size, portfolio_value)
        assert result is False, (
            f"Expected rejection: size={over_limit_size}, portfolio={portfolio_value}, "
            f"limit_pct={limits.max_position_size_pct}"
        )

    @given(
        limits=_risk_limits_strategy(),
        portfolio_value=_POSITIVE_FLOAT,
        size_fraction=st.floats(min_value=0.0, max_value=0.999, allow_nan=False),
    )
    @settings(max_examples=300)
    def test_check_position_size_allows_within_limit(
        self,
        limits: RiskLimits,
        portfolio_value: float,
        size_fraction: float,
    ) -> None:
        """check_position_size must allow when size is strictly within limit."""
        assume(size_fraction < limits.max_position_size_pct)
        rm = RiskManager(limits)
        under_limit_size = portfolio_value * size_fraction
        result = rm.check_position_size(under_limit_size, portfolio_value)
        assert result is True, (
            f"Expected approval: size_frac={size_fraction}, limit={limits.max_position_size_pct}"
        )

    @given(limits=_risk_limits_strategy())
    @settings(max_examples=100)
    def test_zero_portfolio_value_always_rejects(self, limits: RiskLimits) -> None:
        """check_position_size must always reject when portfolio_value <= 0."""
        rm = RiskManager(limits)
        result = rm.check_position_size(1_000.0, 0.0)
        assert result is False


class TestMaxPositionsProperty:
    @given(
        limits=_risk_limits_strategy(),
        positions=st.lists(_position_strategy(), min_size=0, max_size=20),
    )
    @settings(max_examples=300)
    def test_check_max_positions_rejects_at_or_above_max(
        self,
        limits: RiskLimits,
        positions: list[Position],
    ) -> None:
        """check_max_positions must reject when at or above max_open_positions."""
        assume(len(positions) >= limits.max_open_positions)
        rm = RiskManager(limits)
        result = rm.check_max_positions(positions)
        assert result is False, (
            f"Expected rejection: count={len(positions)}, max={limits.max_open_positions}"
        )

    @given(
        limits=_risk_limits_strategy(),
        positions=st.lists(_position_strategy(), min_size=0, max_size=19),
    )
    @settings(max_examples=300)
    def test_check_max_positions_allows_below_max(
        self,
        limits: RiskLimits,
        positions: list[Position],
    ) -> None:
        """check_max_positions must allow when count is strictly below max."""
        assume(len(positions) < limits.max_open_positions)
        rm = RiskManager(limits)
        result = rm.check_max_positions(positions)
        assert result is True, (
            f"Expected approval: count={len(positions)}, max={limits.max_open_positions}"
        )

    @given(limits=_risk_limits_strategy())
    @settings(max_examples=100)
    def test_empty_positions_always_allowed(self, limits: RiskLimits) -> None:
        """check_max_positions with empty list must always return True."""
        assume(limits.max_open_positions >= 1)
        rm = RiskManager(limits)
        assert rm.check_max_positions([]) is True
