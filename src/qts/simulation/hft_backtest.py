"""HFT backtesting adapter using hftbacktest.

Wraps the hftbacktest library (Rust-based, tick-level, queue-position-aware)
to provide sub-second strategy simulation. Returns results in the same
BacktestResult format as the bar-level BacktestEngine.

The module is importable without hftbacktest installed; actual usage raises
ImportError with install instructions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from qts.models.base import (
    ExitReason,
    OrderSide,
    Tick,
    TradeDirection,
    TradeOutcome,
    TradeRecord,
    VolRegime,
)
from qts.simulation.backtest import BacktestResult

logger = logging.getLogger(__name__)

# ── Lazy import guard ─────────────────────────────────────────────────────────

HAS_HFTBACKTEST = False
_hftbacktest: Any = None

try:
    import hftbacktest as _hftbacktest  # type: ignore[no-redef]

    HAS_HFTBACKTEST = True
except ImportError:
    pass

_INSTALL_MSG = (
    "hftbacktest is required for tick-level HFT simulation.\n"
    "Install it with:  pip install 'qts[hft]'  or  pip install hftbacktest"
)


def _require_hftbacktest() -> Any:
    """Return the hftbacktest module or raise ImportError."""
    if not HAS_HFTBACKTEST:
        raise ImportError(_INSTALL_MSG)
    return _hftbacktest


# ── hftbacktest event constants ───────────────────────────────────────────────

# Column indices in the hftbacktest NumPy data format
COL_EVENT = 0
COL_EXCH_TS = 1
COL_LOCAL_TS = 2
COL_SIDE = 3
COL_PRICE = 4
COL_QTY = 5

NUM_COLUMNS = 6

# Event type constants (mirrored from hftbacktest for use without import)
TRADE_EVENT = 2
DEPTH_EVENT = 1
BUY_SIDE = 1
SELL_SIDE = -1


# ── Data conversion ──────────────────────────────────────────────────────────


def convert_ticks_to_hft_format(ticks: Sequence[Tick]) -> np.ndarray:
    """Convert QTS Tick objects to hftbacktest's NumPy array format.

    The output array has columns:
    [event, exchange_timestamp_ns, local_timestamp_ns, side, price, qty]

    All ticks are mapped as TRADE_EVENT. Timestamps are converted to
    nanoseconds since epoch.

    Args:
        ticks: Sequence of Tick objects from the QTS domain model.

    Returns:
        NumPy float64 array of shape (len(ticks), 6).
    """
    if not ticks:
        return np.empty((0, NUM_COLUMNS), dtype=np.float64)

    n = len(ticks)
    data = np.empty((n, NUM_COLUMNS), dtype=np.float64)

    for i, tick in enumerate(ticks):
        ts_ns = int(tick.timestamp.timestamp() * 1_000_000_000)
        side = BUY_SIDE if tick.side == OrderSide.BUY else SELL_SIDE
        data[i] = [TRADE_EVENT, ts_ns, ts_ns, side, tick.price, tick.quantity]

    return data


def load_hft_data(path: Path) -> np.ndarray:
    """Load tick data from an npz/npy file in hftbacktest format.

    Args:
        path: Path to a .npz or .npy file containing the data array.

    Returns:
        NumPy float64 array of shape (N, 6).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the loaded array does not have 6 columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HFT data file not found: {path}")

    if path.suffix == ".npz":
        with np.load(path) as npz:
            # Use the first array key
            key = list(npz.files)[0]
            data = npz[key].astype(np.float64)
    else:
        data = np.load(path).astype(np.float64)

    if data.ndim != 2 or data.shape[1] != NUM_COLUMNS:
        raise ValueError(f"Expected array with {NUM_COLUMNS} columns, got shape {data.shape}")

    return data  # type: ignore[no-any-return]


# ── Statistics helper ─────────────────────────────────────────────────────────


def _compute_statistics(result: BacktestResult, initial_capital: float) -> None:
    """Compute performance statistics on a BacktestResult (in-place).

    Mirrors the logic from BacktestEngine._compute_statistics so that
    HFT results use the same metrics.
    """
    equity = np.array(result.equity_curve, dtype=np.float64)
    if len(equity) < 2:
        return

    result.total_return = float((equity[-1] - initial_capital) / initial_capital)

    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]
    if len(returns) == 0:
        return

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0

    # Annualise assuming ~252 * 6.5h * 3600s / tick_interval steps per year
    # For tick data this is approximate; use 252 as a conservative baseline.
    bars_per_year = 252

    if std_ret > 0:
        result.sharpe_ratio = float(mean_ret / std_ret * np.sqrt(bars_per_year))

    downside = returns[returns < 0]
    if len(downside) > 0:
        downside_std = float(np.std(downside, ddof=1))
        if downside_std > 0:
            result.sortino_ratio = float(mean_ret / downside_std * np.sqrt(bars_per_year))

    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak
    result.max_drawdown = float(np.max(drawdown))

    ann_return = mean_ret * bars_per_year
    if result.max_drawdown > 0:
        result.calmar_ratio = float(ann_return / result.max_drawdown)

    trades = result.trades
    if trades:
        wins = [t for t in trades if t.outcome == TradeOutcome.WIN]
        losses = [t for t in trades if t.outcome == TradeOutcome.LOSS]
        result.win_rate = float(len(wins) / len(trades))

        gross_profit = sum(t.pnl_usd for t in wins)
        gross_loss = abs(sum(t.pnl_usd for t in losses))
        if gross_loss > 0:
            result.profit_factor = float(gross_profit / gross_loss)
        elif gross_profit > 0:
            result.profit_factor = float("inf")
        else:
            result.profit_factor = 0.0


# ── Fill mapping ──────────────────────────────────────────────────────────────


@dataclass
class _OpenPosition:
    """Internal tracker for open HFT positions."""

    symbol: str
    direction: TradeDirection
    entry_price: float
    quantity: float
    entry_time: datetime
    entry_fee: float


def _ns_to_datetime(ts_ns: float) -> datetime:
    """Convert nanosecond timestamp to datetime (UTC)."""
    return datetime.fromtimestamp(ts_ns / 1_000_000_000, tz=UTC)


# ── Adapter ───────────────────────────────────────────────────────────────────


class HFTBacktestAdapter:
    """Adapter for hftbacktest tick-level simulation.

    Wraps the hftbacktest library's simulation engine to provide
    queue-position-aware, tick-by-tick backtesting for HFT strategies.
    Returns results in the same BacktestResult format as our bar-level engine.
    """

    def __init__(
        self,
        tick_data_path: str | Path,
        strategy_fn: Callable[[Any], tuple[list[Any], list[Any]]],
        initial_capital: float = 100_000.0,
        tick_size: float = 0.01,
        lot_size: float = 0.001,
        maker_fee: float = -0.00025,
        taker_fee: float = 0.00075,
        latency_ns: int = 50_000,
        symbol: str = "UNKNOWN",
    ) -> None:
        """Initialise the HFT backtest adapter.

        Args:
            tick_data_path: Path to .npz/.npy tick data in hftbacktest format.
            strategy_fn: Callable receiving backtester state, returning
                (submissions, cancellations) order lists.
            initial_capital: Starting portfolio cash.
            tick_size: Minimum price increment.
            lot_size: Minimum order quantity increment.
            maker_fee: Maker fee rate (negative = rebate).
            taker_fee: Taker fee rate.
            latency_ns: Simulated order-to-exchange latency in nanoseconds.
            symbol: Symbol name for trade records.
        """
        _require_hftbacktest()

        self._tick_data_path = Path(tick_data_path)
        self._strategy_fn = strategy_fn
        self._initial_capital = initial_capital
        self._tick_size = tick_size
        self._lot_size = lot_size
        self._maker_fee = maker_fee
        self._taker_fee = taker_fee
        self._latency_ns = latency_ns
        self._symbol = symbol

    def run(self) -> BacktestResult:
        """Execute the HFT backtest and return results.

        Loads tick data, configures hftbacktest's BacktestAsset and
        HashMapMarketDepthBacktest engine, runs the strategy function,
        then maps fills back to our BacktestResult format.

        Returns:
            BacktestResult with equity curve, trade list, and statistics.
        """
        hft = _require_hftbacktest()

        data = load_hft_data(self._tick_data_path)
        logger.info(f"Loaded {len(data)} ticks from {self._tick_data_path}")

        asset = hft.BacktestAsset()
        asset.data = data
        asset.tick_size = self._tick_size
        asset.lot_size = self._lot_size
        asset.maker_fee = self._maker_fee
        asset.taker_fee = self._taker_fee
        asset.order_latency = self._latency_ns

        bt = hft.HashMapMarketDepthBacktest([asset])

        # Run the user's strategy function
        cash = self._initial_capital
        equity_curve: list[float] = []
        completed_trades: list[TradeRecord] = []
        open_positions: list[_OpenPosition] = []

        while bt.elapse(1_000_000):  # Advance 1ms at a time
            submissions, cancellations = self._strategy_fn(bt)

            # Process cancellations
            for cancel in cancellations:
                bt.cancel(cancel)

            # Process submissions
            for order in submissions:
                bt.submit(order)

            # Check for fills
            fills = bt.get_fills()
            for fill_data in fills:
                fill_price = float(fill_data.price)
                fill_qty = float(fill_data.qty)
                fill_side = fill_data.side
                fill_ts = _ns_to_datetime(float(fill_data.timestamp))
                is_buy = fill_side == BUY_SIDE

                fee_rate = self._maker_fee if fill_data.is_maker else self._taker_fee
                fee = abs(fill_price * fill_qty * fee_rate)

                # Check if this closes an existing position
                close_idx = None
                for i, pos in enumerate(open_positions):
                    if (is_buy and pos.direction == TradeDirection.SHORT) or (
                        not is_buy and pos.direction == TradeDirection.LONG
                    ):
                        close_idx = i
                        break

                if close_idx is not None:
                    closed = open_positions.pop(close_idx)
                    if closed.direction == TradeDirection.LONG:
                        pnl = (fill_price - closed.entry_price) * closed.quantity
                    else:
                        pnl = (closed.entry_price - fill_price) * closed.quantity
                    pnl -= fee + closed.entry_fee

                    outcome = (
                        TradeOutcome.WIN
                        if pnl > 0
                        else (TradeOutcome.LOSS if pnl < 0 else TradeOutcome.BREAKEVEN)
                    )

                    trade = TradeRecord(
                        trade_id=TradeRecord.generate_id(),
                        symbol=self._symbol,
                        entry_time=closed.entry_time,
                        exit_time=fill_ts,
                        direction=closed.direction,
                        entry_price=closed.entry_price,
                        exit_price=fill_price,
                        size=closed.quantity,
                        pnl_usd=pnl,
                        pnl_pct=pnl / (closed.entry_price * closed.quantity),
                        slippage_usd=0.0,
                        fees_usd=fee + closed.entry_fee,
                        rsi_at_entry=50.0,
                        macd_at_entry=0.0,
                        sentiment_at_entry=0.0,
                        social_velocity_at_entry=0.0,
                        gdelt_intensity_at_entry=0.0,
                        vol_regime_at_entry=VolRegime.LOW,
                        combined_alpha_at_entry=0.0,
                        params_version="hft_backtest",
                        outcome=outcome,
                        exit_reason=ExitReason.ALPHA_FLIP,
                        failure_mode=None,
                    )
                    completed_trades.append(trade)
                    cash += pnl
                else:
                    # Open new position
                    direction = TradeDirection.LONG if is_buy else TradeDirection.SHORT
                    open_positions.append(
                        _OpenPosition(
                            symbol=self._symbol,
                            direction=direction,
                            entry_price=fill_price,
                            quantity=fill_qty,
                            entry_time=fill_ts,
                            entry_fee=fee,
                        )
                    )
                    if is_buy:
                        cash -= fill_price * fill_qty + fee
                    else:
                        cash += fill_price * fill_qty - fee

            # Mark-to-market
            mid_price = bt.mid_price if hasattr(bt, "mid_price") else 0.0
            mark_value = sum(
                (
                    (mid_price - p.entry_price) * p.quantity
                    if p.direction == TradeDirection.LONG
                    else (p.entry_price - mid_price) * p.quantity
                )
                for p in open_positions
            )
            equity_curve.append(cash + mark_value)

        logger.info(
            f"HFT backtest complete: {len(completed_trades)} trades, "
            f"{len(equity_curve)} equity samples"
        )

        result = BacktestResult(equity_curve=equity_curve, trades=completed_trades)
        _compute_statistics(result, self._initial_capital)
        return result
