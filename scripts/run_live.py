#!/usr/bin/env python3
"""Script to start the live trading engine.

WARNING: Set QTS_DRY_RUN=false and provide valid exchange credentials
before running in live mode. All trades will be real.

Usage::

    # Paper trading (default):
    python scripts/run_live.py --symbol BTCUSDT

    # Live trading (requires confirmation):
    python scripts/run_live.py --symbol BTCUSDT --live
"""

from __future__ import annotations

import sys

if __name__ == "__main__":
    from qts.cli.main import main

    sys.exit(main(["live"] + sys.argv[1:]))  # type: ignore[call-arg]
