"""CSV-based bar data provider for backtesting.

Implements the BarProvider protocol by reading OHLCV data from CSV files.
Designed for backtesting workflows where historical data is pre-downloaded
and stored locally.

Expected CSV format (columns, case-insensitive):
    timestamp, open, high, low, close, volume[, bar_count]

The timestamp column must be parseable by pandas (ISO 8601 recommended).
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from qts.models.base import Bar

logger = logging.getLogger(__name__)

# Column name aliases to handle minor variations in CSV headers
_COLUMN_ALIASES: dict[str, str] = {
    "time": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vol": "volume",
}

_REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}


class CSVBarAdapter:
    """Bar provider that reads historical OHLCV data from CSV files.

    Supports loading a single CSV file or a directory of CSV files
    organised by symbol (e.g. BTCUSDT.csv). The adapter is read-only;
    it does not support real-time subscriptions beyond replaying historical
    data at configurable speed.

    Attributes:
        data_dir: Directory containing CSV files.
        symbol_col: Whether a 'symbol' column is present in the CSV.
    """

    def __init__(self, data_dir: str | Path) -> None:
        """Initialise the adapter with the directory containing CSV files.

        Args:
            data_dir: Path to the directory containing CSV files.
                Files should be named '<SYMBOL>.csv' (e.g. BTCUSDT.csv).

        Raises:
            FileNotFoundError: If data_dir does not exist.
        """
        self._data_dir = Path(data_dir)
        if not self._data_dir.exists():
            raise FileNotFoundError(f"CSV data directory not found: {self._data_dir}")
        self._cache: dict[str, pd.DataFrame] = {}
        logger.debug("CSVBarAdapter initialised with data_dir=%s", self._data_dir)

    def _load_symbol(self, symbol: str) -> pd.DataFrame:
        """Load and cache the DataFrame for a symbol.

        Looks for a file named <SYMBOL>.csv (case-insensitive) in the data directory.

        Args:
            symbol: Ticker symbol to load.

        Returns:
            DataFrame with normalised columns: timestamp, open, high, low, close, volume,
            bar_count.

        Raises:
            FileNotFoundError: If no CSV file is found for the symbol.
            ValueError: If the CSV is missing required columns after normalisation.
        """
        if symbol in self._cache:
            return self._cache[symbol]

        # Case-insensitive file search
        candidates = list(self._data_dir.glob(f"{symbol}.csv"))
        if not candidates:
            candidates = list(self._data_dir.glob(f"{symbol.upper()}.csv"))
        if not candidates:
            candidates = list(self._data_dir.glob(f"{symbol.lower()}.csv"))

        if not candidates:
            raise FileNotFoundError(
                f"No CSV file found for symbol '{symbol}' in {self._data_dir}. "
                f"Expected a file named '{symbol}.csv'."
            )

        csv_path = candidates[0]
        logger.info("Loading CSV for symbol %s from %s", symbol, csv_path)

        df = pd.read_csv(csv_path)

        # Normalise column names: strip whitespace, lowercase, apply aliases
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns=_COLUMN_ALIASES)

        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV file '{csv_path}' is missing required columns: {missing}. "
                f"Found: {list(df.columns)}"
            )

        # Parse timestamp to datetime, coerce errors to NaT and drop
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Ensure bar_count column exists
        if "bar_count" not in df.columns:
            df["bar_count"] = 0

        # Add symbol column
        df["symbol"] = symbol

        self._cache[symbol] = df
        logger.debug("Loaded %d bars for symbol %s", len(df), symbol)
        return df

    async def get_historical_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1m",
    ) -> list[Bar]:
        """Fetch historical OHLCV bars from CSV for a given time range.

        The interval parameter is accepted for protocol compatibility but is
        not used to filter data (the CSV file is assumed to contain a single
        interval). To use multiple intervals, provide separate CSV files.

        Args:
            symbol: Ticker symbol to load (must match a file in data_dir).
            start: Start datetime string (ISO 8601 recommended).
            end: End datetime string (ISO 8601 recommended).
            interval: Bar interval string (accepted but not used for filtering).

        Returns:
            List of Bar objects within [start, end] ordered by timestamp ascending.

        Raises:
            FileNotFoundError: If no CSV file exists for the symbol.
            ValueError: If start or end cannot be parsed as datetimes.
        """
        df = self._load_symbol(symbol)

        start_dt = pd.Timestamp(start, tz="UTC")
        end_dt = pd.Timestamp(end, tz="UTC")

        mask = (df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)
        filtered = df.loc[mask]

        bars: list[Bar] = []
        for row in filtered.itertuples(index=False):
            bars.append(
                Bar(
                    timestamp=row.timestamp.to_pydatetime(),
                    symbol=row.symbol,
                    open=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                    close=float(row.close),
                    volume=float(row.volume),
                    bar_count=int(row.bar_count),
                )
            )

        logger.debug(
            "CSVBarAdapter: returned %d bars for %s [%s, %s]",
            len(bars),
            symbol,
            start,
            end,
        )
        return bars

    async def subscribe(self, symbol: str, interval: str = "1m") -> AsyncIterator[Bar]:
        """Replay all available historical bars for a symbol as an async stream.

        Useful for simulating real-time data in backtests. Each bar is yielded
        with a short async sleep to allow cooperative scheduling.

        Args:
            symbol: Ticker symbol to replay.
            interval: Bar interval string (accepted for protocol compatibility).

        Yields:
            Bar objects in chronological order.

        Raises:
            FileNotFoundError: If no CSV file exists for the symbol.
        """
        df = self._load_symbol(symbol)
        logger.info(
            "CSVBarAdapter: subscribing to %d bars for symbol %s", len(df), symbol
        )
        for row in df.itertuples(index=False):
            yield Bar(
                timestamp=row.timestamp.to_pydatetime(),
                symbol=row.symbol,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
                bar_count=int(row.bar_count),
            )
            # Yield control to event loop between bars
            await asyncio.sleep(0)

    async def close(self) -> None:
        """Release the in-memory CSV cache.

        Clears all cached DataFrames. Subsequent calls to get_historical_bars
        or subscribe will reload from disk.
        """
        self._cache.clear()
        logger.debug("CSVBarAdapter closed and cache cleared")

    def list_available_symbols(self) -> list[str]:
        """Return the list of symbols for which CSV files are available.

        Returns:
            Sorted list of symbol names (derived from CSV filenames, uppercased).
        """
        return sorted(p.stem.upper() for p in self._data_dir.glob("*.csv"))
