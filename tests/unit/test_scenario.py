"""Unit tests for qts.simulation.scenario.

Covers:
- T-SCEN-1 to T-SCEN-20
- StressScenarioRunner.flash_crash (happy path, validation, empty input, edge cases)
- StressScenarioRunner.sentiment_spike (happy path, validation, empty input)
- StressScenarioRunner.data_gap (happy path, validation, gap-at-start edge case)
- Originals are not mutated
"""

from __future__ import annotations

from datetime import datetime

import pytest

from qts.models.base import Bar, SignalSnapshot, VolLevel
from qts.simulation.scenario import StressScenarioRunner

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_bars(n: int, base_close: float = 100.0) -> list[Bar]:
    return [
        Bar(
            timestamp=datetime(2024, 1, 1, 0, i, 0),
            symbol="SYNTH",
            open=base_close,
            high=base_close * 1.002,
            low=base_close * 0.998,
            close=base_close,
            volume=1000.0,
            bar_count=1,
        )
        for i in range(n)
    ]


def _make_snapshots(n: int, sentiment: float = 0.0) -> list[SignalSnapshot]:
    return [
        SignalSnapshot(
            timestamp=datetime(2024, 1, 1, 0, i, 0),
            symbol="SYNTH",
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
            vol_level_confidence=0.8,
            sentiment_score=sentiment,
            combined_alpha=0.0,
        )
        for i in range(n)
    ]


# ── flash_crash ────────────────────────────────────────────────────────────────


class TestFlashCrash:
    def test_T_SCEN_1_returns_new_list_not_original(self) -> None:
        """flash_crash must return a new list; original is not mutated."""
        runner = StressScenarioRunner()
        bars = _make_bars(20)
        result = runner.flash_crash(bars, magnitude=-0.10, duration_bars=3)
        assert result is not bars

    def test_T_SCEN_2_same_length(self) -> None:
        """Result has the same number of bars as input."""
        runner = StressScenarioRunner()
        bars = _make_bars(30)
        result = runner.flash_crash(bars, magnitude=-0.10, duration_bars=5)
        assert len(result) == 30

    def test_T_SCEN_3_prices_fall_during_crash_phase(self) -> None:
        """Bars in decline phase should have lower close than the baseline."""
        runner = StressScenarioRunner()
        bars = _make_bars(30, base_close=100.0)
        start = 5
        result = runner.flash_crash(bars, magnitude=-0.20, duration_bars=5, start_index=start)
        for i in range(start, start + 5):
            assert result[i].close < 100.0, f"bar {i} close should be below baseline"

    def test_T_SCEN_4_prices_recover_after_crash(self) -> None:
        """Bars in recovery phase should have close between nadir and baseline."""
        runner = StressScenarioRunner()
        bars = _make_bars(40, base_close=100.0)
        start = 5
        magnitude = -0.20
        duration = 5
        result = runner.flash_crash(
            bars, magnitude=magnitude, duration_bars=duration, start_index=start
        )
        nadir = 100.0 * (1.0 + magnitude)  # 80.0
        end_crash = start + duration  # 10
        end_recovery = end_crash + duration  # 15
        for i in range(end_crash, end_recovery):
            assert (
                nadir <= result[i].close <= 100.0 + 1e-9
            ), f"bar {i} close={result[i].close} not in recovery range [{nadir}, 100]"

    def test_T_SCEN_5_bars_outside_window_unchanged(self) -> None:
        """Bars before crash window and after recovery window are unchanged."""
        runner = StressScenarioRunner()
        bars = _make_bars(50, base_close=100.0)
        result = runner.flash_crash(bars, magnitude=-0.10, duration_bars=3, start_index=10)
        # Bars before start are untouched
        for i in range(0, 10):
            assert result[i].close == pytest.approx(100.0)
        # Bars after recovery (index >= 16) are untouched
        for i in range(16, 50):
            assert result[i].close == pytest.approx(100.0)

    def test_T_SCEN_6_positive_magnitude_raises(self) -> None:
        """Positive magnitude should raise ValueError."""
        runner = StressScenarioRunner()
        bars = _make_bars(20)
        with pytest.raises(ValueError, match="negative"):
            runner.flash_crash(bars, magnitude=0.10)

    def test_T_SCEN_7_zero_magnitude_raises(self) -> None:
        runner = StressScenarioRunner()
        bars = _make_bars(20)
        with pytest.raises(ValueError, match="negative"):
            runner.flash_crash(bars, magnitude=0.0)

    def test_T_SCEN_8_zero_duration_bars_raises(self) -> None:
        runner = StressScenarioRunner()
        bars = _make_bars(20)
        with pytest.raises(ValueError, match="duration_bars"):
            runner.flash_crash(bars, magnitude=-0.10, duration_bars=0)

    def test_T_SCEN_9_empty_bars_returns_empty(self) -> None:
        runner = StressScenarioRunner()
        result = runner.flash_crash([], magnitude=-0.10)
        assert result == []

    def test_T_SCEN_10_default_start_index_is_third(self) -> None:
        """Default start_index is n // 3."""
        runner = StressScenarioRunner()
        bars = _make_bars(30, base_close=100.0)
        # start_index == 10, duration_bars == 5 by default
        result = runner.flash_crash(bars, magnitude=-0.20)
        # bar at index 10 should be modified
        assert result[10].close < 100.0
        # bar at index 0 should be untouched
        assert result[0].close == pytest.approx(100.0)

    def test_T_SCEN_11_original_bars_not_mutated(self) -> None:
        runner = StressScenarioRunner()
        bars = _make_bars(20, base_close=100.0)
        original_closes = [b.close for b in bars]
        runner.flash_crash(bars, magnitude=-0.15, start_index=3)
        for original, bar in zip(original_closes, bars, strict=True):
            assert bar.close == pytest.approx(original)

    def test_T_SCEN_12_volume_unchanged_in_crash_phase(self) -> None:
        """_scale_bar should preserve volume."""
        runner = StressScenarioRunner()
        bars = _make_bars(20, base_close=100.0)
        result = runner.flash_crash(bars, magnitude=-0.20, duration_bars=3, start_index=5)
        for i in range(5, 8):
            assert result[i].volume == pytest.approx(bars[i].volume)


# ── sentiment_spike ───────────────────────────────────────────────────────────


class TestSentimentSpike:
    def test_T_SCEN_13_sets_sentiment_in_window(self) -> None:
        """Sentiment score in the spike window should equal magnitude."""
        runner = StressScenarioRunner()
        snaps = _make_snapshots(20, sentiment=0.5)
        result = runner.sentiment_spike(snaps, magnitude=-0.9, start_index=5, duration_bars=4)
        for i in range(5, 9):
            assert result[i].sentiment_score == pytest.approx(-0.9)

    def test_T_SCEN_14_combined_alpha_reset_to_zero(self) -> None:
        """combined_alpha should be reset to 0.0 for affected snapshots."""
        runner = StressScenarioRunner()
        snaps = _make_snapshots(20)
        result = runner.sentiment_spike(snaps, magnitude=-0.5, start_index=0, duration_bars=5)
        for i in range(5):
            assert result[i].combined_alpha == pytest.approx(0.0)

    def test_T_SCEN_15_snapshots_outside_window_unchanged(self) -> None:
        runner = StressScenarioRunner()
        snaps = _make_snapshots(20, sentiment=0.3)
        result = runner.sentiment_spike(snaps, magnitude=-0.9, start_index=5, duration_bars=3)
        for i in list(range(0, 5)) + list(range(8, 20)):
            assert result[i].sentiment_score == pytest.approx(0.3)

    def test_T_SCEN_16_magnitude_out_of_range_raises(self) -> None:
        runner = StressScenarioRunner()
        snaps = _make_snapshots(10)
        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            runner.sentiment_spike(snaps, magnitude=1.5)

    def test_T_SCEN_17_magnitude_minus_one_is_valid(self) -> None:
        runner = StressScenarioRunner()
        snaps = _make_snapshots(10)
        result = runner.sentiment_spike(snaps, magnitude=-1.0, start_index=0, duration_bars=2)
        assert result[0].sentiment_score == pytest.approx(-1.0)

    def test_T_SCEN_18_empty_snapshots_returns_empty(self) -> None:
        runner = StressScenarioRunner()
        result = runner.sentiment_spike([], magnitude=-0.5)
        assert result == []

    def test_T_SCEN_19_original_snapshots_not_mutated(self) -> None:
        runner = StressScenarioRunner()
        snaps = _make_snapshots(10, sentiment=0.1)
        runner.sentiment_spike(snaps, magnitude=-0.9, start_index=2, duration_bars=3)
        for s in snaps:
            assert s.sentiment_score == pytest.approx(0.1)

    def test_T_SCEN_20_returns_new_list(self) -> None:
        runner = StressScenarioRunner()
        snaps = _make_snapshots(10)
        result = runner.sentiment_spike(snaps, magnitude=-0.5)
        assert result is not snaps


# ── data_gap ──────────────────────────────────────────────────────────────────


class TestDataGap:
    def test_T_SCEN_21_gap_bars_have_stale_close(self) -> None:
        """Bars inside the gap should have open/high/low/close equal to pre-gap close."""
        runner = StressScenarioRunner()
        bars = _make_bars(20, base_close=100.0)
        # Slightly vary bar before gap so we can distinguish stale price
        bars[4] = Bar(
            timestamp=bars[4].timestamp,
            symbol=bars[4].symbol,
            open=99.0,
            high=101.0,
            low=98.0,
            close=99.5,
            volume=500.0,
            bar_count=1,
        )
        result = runner.data_gap(bars, gap_duration_bars=3, start_index=5)
        for i in range(5, 8):
            assert result[i].close == pytest.approx(99.5)
            assert result[i].open == pytest.approx(99.5)
            assert result[i].high == pytest.approx(99.5)
            assert result[i].low == pytest.approx(99.5)
            assert result[i].volume == pytest.approx(0.0)
            assert result[i].bar_count == 0

    def test_T_SCEN_22_gap_at_start_uses_first_bar(self) -> None:
        """When start_index is 0, the first bar itself is used as stale reference."""
        runner = StressScenarioRunner()
        bars = _make_bars(10, base_close=50.0)
        result = runner.data_gap(bars, gap_duration_bars=3, start_index=0)
        for i in range(3):
            assert result[i].close == pytest.approx(50.0)

    def test_T_SCEN_23_bars_outside_gap_unchanged(self) -> None:
        runner = StressScenarioRunner()
        bars = _make_bars(20, base_close=100.0)
        result = runner.data_gap(bars, gap_duration_bars=4, start_index=8)
        for i in list(range(0, 8)) + list(range(12, 20)):
            assert result[i].close == pytest.approx(100.0)

    def test_T_SCEN_24_same_length_as_input(self) -> None:
        runner = StressScenarioRunner()
        bars = _make_bars(15)
        result = runner.data_gap(bars, gap_duration_bars=5, start_index=3)
        assert len(result) == 15

    def test_T_SCEN_25_zero_duration_raises(self) -> None:
        runner = StressScenarioRunner()
        bars = _make_bars(10)
        with pytest.raises(ValueError, match="gap_duration_bars"):
            runner.data_gap(bars, gap_duration_bars=0)

    def test_T_SCEN_26_empty_bars_returns_empty(self) -> None:
        runner = StressScenarioRunner()
        result = runner.data_gap([], gap_duration_bars=5)
        assert result == []

    def test_T_SCEN_27_original_bars_not_mutated(self) -> None:
        runner = StressScenarioRunner()
        bars = _make_bars(15, base_close=100.0)
        original_closes = [b.close for b in bars]
        runner.data_gap(bars, gap_duration_bars=3, start_index=5)
        for original, bar in zip(original_closes, bars, strict=True):
            assert bar.close == pytest.approx(original)

    def test_T_SCEN_28_timestamps_preserved_in_gap(self) -> None:
        """Gap bars keep their original timestamps."""
        runner = StressScenarioRunner()
        bars = _make_bars(10)
        original_timestamps = [b.timestamp for b in bars]
        result = runner.data_gap(bars, gap_duration_bars=3, start_index=3)
        for i in range(3, 6):
            assert result[i].timestamp == original_timestamps[i]

    def test_T_SCEN_29_gap_clipped_at_end_of_sequence(self) -> None:
        """Gap extending beyond the sequence end should not raise."""
        runner = StressScenarioRunner()
        bars = _make_bars(10)
        result = runner.data_gap(bars, gap_duration_bars=100, start_index=7)
        assert len(result) == 10
