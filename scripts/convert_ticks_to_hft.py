#!/usr/bin/env python3
"""Convert CSV/Parquet tick data to hftbacktest's NumPy format.

Usage::

    python scripts/convert_ticks_to_hft.py --input ticks.csv --output ticks.npz --symbol BTCUSDT
    python scripts/convert_ticks_to_hft.py --input ticks.parquet --output ticks.npz
"""

from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

from qts.simulation.hft_backtest import (
    BUY_SIDE,
    NUM_COLUMNS,
    SELL_SIDE,
    TRADE_EVENT,
)

logger = logging.getLogger(__name__)


def _read_tick_data(path: Path) -> pd.DataFrame:
    """Read tick data from CSV or Parquet.

    Expected columns: timestamp, price, quantity, side
    Timestamp should be ISO-format or epoch nanoseconds.
    Side should be 'BUY' or 'SELL' (case-insensitive).
    """
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".gz"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix} (expected .csv, .gz, .parquet)")

    required = {"timestamp", "price", "quantity", "side"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def _convert_dataframe(df: pd.DataFrame) -> np.ndarray:
    """Convert a tick DataFrame to hftbacktest NumPy format."""
    n = len(df)
    data = np.empty((n, NUM_COLUMNS), dtype=np.float64)

    # Parse timestamps to nanoseconds
    ts = pd.to_datetime(df["timestamp"])
    ts_ns = ts.astype(np.int64).values.astype(np.float64)

    sides = df["side"].str.upper().map({"BUY": BUY_SIDE, "SELL": SELL_SIDE}).values

    data[:, 0] = TRADE_EVENT
    data[:, 1] = ts_ns
    data[:, 2] = ts_ns  # local_ts = exchange_ts (no latency offset in raw data)
    data[:, 3] = sides
    data[:, 4] = df["price"].values.astype(np.float64)
    data[:, 5] = df["quantity"].values.astype(np.float64)

    return data


@click.command()
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True),
    help="Input CSV/Parquet tick file.",
)
@click.option(
    "--output", "output_path", required=True, type=click.Path(), help="Output .npz file path."
)
@click.option("--symbol", default="UNKNOWN", help="Symbol name (metadata only, stored in npz).")
def main(input_path: str, output_path: str, symbol: str) -> None:
    """Convert tick data to hftbacktest NumPy format."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    src = Path(input_path)
    dst = Path(output_path)

    logger.info(f"Reading tick data from {src}")
    df = _read_tick_data(src)
    logger.info(f"Loaded {len(df)} ticks for {symbol}")

    data = _convert_dataframe(df)

    np.savez_compressed(dst, data=data)
    logger.info(f"Saved {len(data)} rows to {dst} ({dst.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
