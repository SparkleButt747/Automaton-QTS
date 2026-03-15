#!/usr/bin/env python3
"""Script to run a historical backtest.

Usage::

    python scripts/run_backtest.py --symbol BTCUSDT --start 2023-01-01 --end 2023-12-31
    # Or via CLI:
    qts backtest --symbol BTCUSDT --start 2023-01-01 --end 2023-12-31
"""

from __future__ import annotations

import sys

# Prefer using the CLI entry point for full arg parsing:
#   qts backtest ...
# This script is a convenience wrapper for direct execution.
if __name__ == "__main__":
    from qts.cli.main import main

    sys.exit(main(["backtest"] + sys.argv[1:]))  # type: ignore[call-arg]
