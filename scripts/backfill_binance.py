#!/usr/bin/env python3
"""Download historical kline data from Binance and save as Parquet.

No API key required — uses the public /api/v3/klines endpoint.

Usage::

    python scripts/backfill_binance.py --symbol BTCUSDT --interval 1m \
        --start 2024-01-01 --end 2024-02-01

    python scripts/backfill_binance.py --symbol ETHUSDT --interval 1h \
        --start 2023-06-01 --output-dir data/eth/
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import click
import httpx
import pandas as pd
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

logger = logging.getLogger(__name__)

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000
REQUEST_DELAY_S = 0.1  # 100ms between requests — well within 1200 weight/min


def _iso_to_ms(date_str: str) -> int:
    """Convert an ISO date string (YYYY-MM-DD) to milliseconds since epoch (UTC)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _ms_to_dt(ms: int) -> pd.Timestamp:
    """Convert milliseconds since epoch to a UTC pandas Timestamp."""
    return pd.Timestamp(ms, unit="ms", tz="UTC")


def fetch_klines(
    client: httpx.Client,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> list[list]:
    """Fetch a single page of klines from Binance."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": MAX_LIMIT,
    }
    resp = client.get(BINANCE_KLINES_URL, params=params, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def download_all_klines(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Download all klines for the given symbol/interval/date range.

    Paginates automatically and shows a progress bar.
    """
    start_ms = _iso_to_ms(start_date)
    end_ms = _iso_to_ms(end_date)
    total_ms = end_ms - start_ms

    all_rows: list[list] = []
    current_ms = start_ms

    with (
        httpx.Client() as client,
        Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
        ) as progress,
    ):
        task = progress.add_task(f"{symbol} {interval}", total=total_ms)

        while current_ms < end_ms:
            klines = fetch_klines(client, symbol, interval, current_ms, end_ms)

            if not klines:
                logger.info("No more data returned at %s", _ms_to_dt(current_ms))
                break

            all_rows.extend(klines)

            last_close_time = klines[-1][6]  # close_time field
            elapsed = last_close_time - start_ms
            progress.update(task, completed=min(elapsed, total_ms))

            # Advance past the last bar
            current_ms = last_close_time + 1

            # Don't hammer the API
            time.sleep(REQUEST_DELAY_S)

    if not all_rows:
        logger.warning("No kline data retrieved for %s %s", symbol, interval)
        return pd.DataFrame()

    # Build DataFrame from raw kline arrays
    # Kline format: [open_time, open, high, low, close, volume, close_time, ...]
    df = pd.DataFrame(
        {
            "timestamp": [_ms_to_dt(row[0]) for row in all_rows],
            "open": [float(row[1]) for row in all_rows],
            "high": [float(row[2]) for row in all_rows],
            "low": [float(row[3]) for row in all_rows],
            "close": [float(row[4]) for row in all_rows],
            "volume": [float(row[5]) for row in all_rows],
            "bar_count": [int(row[8]) for row in all_rows],
        }
    )

    # Enforce dtypes
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype("float64")
    df["bar_count"] = df["bar_count"].astype("int64")

    # Drop duplicates (pagination overlap)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    logger.info("Downloaded %d bars for %s %s", len(df), symbol, interval)
    return df


@click.command()
@click.option("--symbol", "-s", default="BTCUSDT", show_default=True, help="Trading pair symbol")
@click.option(
    "--interval",
    "-i",
    default="1m",
    show_default=True,
    help="Kline interval (1m, 5m, 1h, 1d, ...)",
)
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", default=None, help="End date (YYYY-MM-DD, default: today)")
@click.option(
    "--output-dir",
    default="data/",
    show_default=True,
    type=click.Path(),
    help="Directory to save Parquet files",
)
def main(symbol: str, interval: str, start: str, end: str | None, output_dir: str) -> None:
    """Download historical Binance kline data and save as Parquet."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    if end is None:
        end = datetime.now(tz=UTC).strftime("%Y-%m-%d")

    logger.info("Backfilling %s %s from %s to %s", symbol, interval, start, end)

    df = download_all_klines(symbol, interval, start, end)

    if df.empty:
        logger.error("No data to write — exiting")
        raise SystemExit(1)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    filename = out_path / f"{symbol}_{interval}.parquet"

    df.to_parquet(filename, index=False, engine="pyarrow")
    logger.info("Saved %d rows to %s", len(df), filename)
    click.echo(f"Done: {len(df)} bars -> {filename}")


if __name__ == "__main__":
    main()
