"""Parquet data catalog management for NautilusTrader.

Provides utilities to:
- Convert QTS Bar sequences to Parquet files
- Convert CSV market data to Parquet
- Load Parquet catalogs for Nautilus BacktestNode ingestion
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from qts.models.base import Bar

logger = logging.getLogger(__name__)

# Default catalog root
_DEFAULT_CATALOG_ROOT = Path("data/catalogs")


def bars_to_dataframe(bars: Sequence[Bar]) -> pd.DataFrame:
    """Convert a sequence of QTS Bars to a pandas DataFrame."""
    records = [
        {
            "timestamp": b.timestamp,
            "symbol": b.symbol,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        }
        for b in bars
    ]
    df = pd.DataFrame.from_records(records)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def dataframe_to_bars(df: pd.DataFrame) -> list[Bar]:
    """Convert a pandas DataFrame back to a list of QTS Bars."""
    bars: list[Bar] = []
    for _, row in df.iterrows():
        bars.append(
            Bar(
                timestamp=row["timestamp"].to_pydatetime(),
                symbol=str(row["symbol"]),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )
        )
    return bars


def write_bars_to_parquet(
    bars: Sequence[Bar],
    path: Path,
    symbol: str | None = None,
) -> Path:
    """Write QTS Bars to a Parquet file.

    Args:
        bars: Sequence of Bar objects to write.
        path: Output directory or file path. If directory, file is named
              ``{symbol}_bars.parquet``.
        symbol: Symbol name for the filename. Inferred from bars if not given.

    Returns:
        Path to the written Parquet file.
    """
    df = bars_to_dataframe(bars)
    if df.empty:
        raise ValueError("No bars to write")

    sym = symbol or df["symbol"].iloc[0]
    if path.is_dir() or not path.suffix:
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"{sym}_bars.parquet"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_path = path

    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)
    logger.info("Wrote %d bars to %s", len(bars), file_path)
    return file_path


def read_bars_from_parquet(path: Path) -> list[Bar]:
    """Read QTS Bars from a Parquet file."""
    table = pq.read_table(path)
    df = table.to_pandas()
    return dataframe_to_bars(df)


def csv_to_parquet(
    csv_path: Path,
    output_dir: Path | None = None,
    symbol: str | None = None,
    timestamp_col: str = "timestamp",
) -> Path:
    """Convert a CSV file with OHLCV data to Parquet.

    Expected columns: timestamp, open, high, low, close, volume.
    Symbol column is optional — if missing, uses the ``symbol`` argument.

    Args:
        csv_path: Path to the CSV file.
        output_dir: Directory for the output Parquet file.
        symbol: Symbol name (used if CSV has no 'symbol' column).
        timestamp_col: Name of the timestamp column.

    Returns:
        Path to the written Parquet file.
    """
    df = pd.read_csv(csv_path)
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
    df = df.rename(columns={timestamp_col: "timestamp"})

    if "symbol" not in df.columns:
        sym = symbol or csv_path.stem
        df["symbol"] = sym
    else:
        sym = df["symbol"].iloc[0]

    # Ensure standard column names
    expected = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    out_dir = output_dir or _DEFAULT_CATALOG_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{sym}_bars.parquet"

    table = pa.Table.from_pandas(df[list(expected)])
    pq.write_table(table, file_path)
    logger.info("Converted %s → %s (%d rows)", csv_path, file_path, len(df))
    return file_path


def ensure_catalog(
    bars: Sequence[Bar],
    catalog_root: Path | None = None,
) -> Path:
    """Ensure bars are written to a Parquet catalog, returning the path.

    Idempotent: if the file already exists with the same row count, skips writing.
    """
    root = catalog_root or _DEFAULT_CATALOG_ROOT
    if not bars:
        raise ValueError("No bars provided for catalog")

    symbol = bars[0].symbol
    file_path = root / f"{symbol}_bars.parquet"

    if file_path.exists():
        existing = pq.read_table(file_path)
        if len(existing) == len(bars):
            logger.debug("Catalog already exists and has correct row count: %s", file_path)
            return file_path

    return write_bars_to_parquet(bars, root, symbol)
