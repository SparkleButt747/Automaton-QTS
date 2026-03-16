"""Tests for the hftbacktest adapter module."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import numpy as np
import pytest

from qts.models.base import OrderSide, Tick
from qts.simulation.backtest import BacktestResult
from qts.simulation.hft_backtest import (
    BUY_SIDE,
    HAS_HFTBACKTEST,
    NUM_COLUMNS,
    SELL_SIDE,
    TRADE_EVENT,
    HFTBacktestAdapter,
    _compute_statistics,
    convert_ticks_to_hft_format,
    load_hft_data,
)

# ── T-HFT-1: Module imports without hftbacktest installed ────────────────────


class TestImportGuard:
    """Verify the module is importable and usable without hftbacktest."""

    def test_module_importable(self):
        """T-HFT-1: Module can be imported regardless of hftbacktest availability."""
        # If we got here, the import succeeded
        from qts.simulation import hft_backtest  # noqa: F401

    def test_has_hftbacktest_is_bool(self):
        """T-HFT-2: HAS_HFTBACKTEST flag is a boolean."""
        assert isinstance(HAS_HFTBACKTEST, bool)

    def test_adapter_raises_without_library(self):
        """T-HFT-3: HFTBacktestAdapter raises ImportError without hftbacktest."""
        with patch("qts.simulation.hft_backtest.HAS_HFTBACKTEST", False):
            with pytest.raises(ImportError, match="hftbacktest is required"):
                HFTBacktestAdapter(
                    tick_data_path="/fake/path.npz",
                    strategy_fn=lambda state: ([], []),
                )


# ── T-HFT-4..7: convert_ticks_to_hft_format ─────────────────────────────────


class TestConvertTicks:
    """Test conversion from QTS Tick objects to hftbacktest NumPy format."""

    def _make_tick(
        self, price: float = 100.0, qty: float = 1.0, side: OrderSide = OrderSide.BUY
    ) -> Tick:
        return Tick(
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC),
            symbol="BTCUSDT",
            price=price,
            quantity=qty,
            side=side,
        )

    def test_empty_input(self):
        """T-HFT-4: Empty tick list produces empty array with correct shape."""
        result = convert_ticks_to_hft_format([])
        assert result.shape == (0, NUM_COLUMNS)
        assert result.dtype == np.float64

    def test_single_buy_tick(self):
        """T-HFT-5: Single BUY tick converts correctly."""
        tick = self._make_tick(price=50_000.0, qty=0.1, side=OrderSide.BUY)
        result = convert_ticks_to_hft_format([tick])

        assert result.shape == (1, NUM_COLUMNS)
        assert result[0, 0] == TRADE_EVENT
        assert result[0, 3] == BUY_SIDE
        assert result[0, 4] == 50_000.0
        assert result[0, 5] == 0.1

    def test_single_sell_tick(self):
        """T-HFT-6: Single SELL tick maps to SELL_SIDE."""
        tick = self._make_tick(side=OrderSide.SELL)
        result = convert_ticks_to_hft_format([tick])

        assert result[0, 3] == SELL_SIDE

    def test_multiple_ticks_preserves_order(self):
        """T-HFT-7: Multiple ticks maintain input order."""
        ticks = [
            self._make_tick(price=100.0),
            self._make_tick(price=200.0),
            self._make_tick(price=300.0),
        ]
        result = convert_ticks_to_hft_format(ticks)

        assert result.shape == (3, NUM_COLUMNS)
        np.testing.assert_array_equal(result[:, 4], [100.0, 200.0, 300.0])

    def test_timestamp_is_nanoseconds(self):
        """T-HFT-8: Timestamps are converted to nanosecond epoch."""
        tick = self._make_tick()
        result = convert_ticks_to_hft_format([tick])

        expected_ns = tick.timestamp.timestamp() * 1_000_000_000
        assert result[0, 1] == pytest.approx(expected_ns)
        assert result[0, 2] == pytest.approx(expected_ns)


# ── T-HFT-9..11: load_hft_data ──────────────────────────────────────────────


class TestLoadHftData:
    """Test loading hftbacktest data from files."""

    def test_load_npy(self, tmp_path):
        """T-HFT-9: Load data from .npy file."""
        data = np.random.default_rng(42).random((10, NUM_COLUMNS))
        path = tmp_path / "test.npy"
        np.save(path, data)

        loaded = load_hft_data(path)
        np.testing.assert_array_almost_equal(loaded, data)

    def test_load_npz(self, tmp_path):
        """T-HFT-10: Load data from .npz file."""
        data = np.random.default_rng(42).random((10, NUM_COLUMNS))
        path = tmp_path / "test.npz"
        np.savez_compressed(path, data=data)

        loaded = load_hft_data(path)
        np.testing.assert_array_almost_equal(loaded, data)

    def test_file_not_found(self, tmp_path):
        """T-HFT-11: FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_hft_data(tmp_path / "nonexistent.npy")

    def test_wrong_shape(self, tmp_path):
        """T-HFT-12: ValueError for incorrect column count."""
        data = np.random.default_rng(42).random((10, 4))
        path = tmp_path / "bad.npy"
        np.save(path, data)

        with pytest.raises(ValueError, match="6 columns"):
            load_hft_data(path)


# ── T-HFT-13..14: Statistics helper ─────────────────────────────────────────


class TestComputeStatistics:
    """Test the standalone statistics computation."""

    def test_empty_equity_curve(self):
        """T-HFT-13: No stats computed for short equity curve."""
        result = BacktestResult(equity_curve=[100_000.0])
        _compute_statistics(result, 100_000.0)
        assert result.total_return == 0.0

    def test_basic_statistics(self):
        """T-HFT-14: Statistics computed correctly for simple equity curve."""
        equity = [100_000.0, 101_000.0, 100_500.0, 102_000.0]
        result = BacktestResult(equity_curve=equity)
        _compute_statistics(result, 100_000.0)

        assert result.total_return == pytest.approx(0.02)
        assert result.max_drawdown > 0.0
        assert result.sharpe_ratio != 0.0


# ── T-HFT-15: Integration tests (skip without hftbacktest) ──────────────────


@pytest.mark.skipif(not HAS_HFTBACKTEST, reason="hftbacktest not installed")
class TestHFTBacktestIntegration:
    """Integration tests requiring hftbacktest to be installed."""

    def test_adapter_constructs(self, tmp_path):
        """T-HFT-15: Adapter can be constructed with valid parameters."""
        # Create minimal valid data file
        data = np.zeros((100, NUM_COLUMNS), dtype=np.float64)
        data[:, 0] = TRADE_EVENT
        data[:, 4] = 50_000.0
        data[:, 5] = 0.01
        path = tmp_path / "test_data.npz"
        np.savez_compressed(path, data=data)

        adapter = HFTBacktestAdapter(
            tick_data_path=path,
            strategy_fn=lambda state: ([], []),
            initial_capital=100_000.0,
            symbol="BTCUSDT",
        )
        assert adapter._symbol == "BTCUSDT"
        assert adapter._initial_capital == 100_000.0
