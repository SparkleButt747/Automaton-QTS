#!/usr/bin/env python3
"""Script to run a post-session debrief with LLM-assisted analysis.

Analyzes the previous trading session's performance, generates a P&L
attribution report, and surfaces lessons learned via Claude.

Usage::

    python scripts/run_debrief.py
    # Or:
    qts debrief
"""

from __future__ import annotations

import sys

if __name__ == "__main__":
    from qts.cli.main import main

    sys.exit(main(["debrief"] + sys.argv[1:]))  # type: ignore[call-arg]
