"""NautilusTrader integration — execution and simulation engine.

This package provides the bridge between QTS strategy/signal logic and
NautilusTrader's production-grade execution engine. Key components:

- ``actor``: QTSStrategy actor adapter (NautilusTrader Strategy subclass)
- ``converters``: Data type converters between QTS and Nautilus domains
- ``runner``: ``run_terrain_backtest()`` — the equivalent of ``simulate_episode()``
- ``config``: Strategy and venue configuration models
- ``catalog``: Parquet data catalog management
"""
