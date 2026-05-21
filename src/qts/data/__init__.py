"""Data ingestion and management sub-package.

Submodules:
    market       - OHLCV and order book data from exchanges
    news         - News article ingestion and preprocessing
    social       - Social media data (Reddit, Twitter)
    geopolitical - Geopolitical event data (GDELT, etc.)

Top-level types:
    RealEpisode  - Real-market analogue of SimulatedEpisode
"""

from qts.data.real_episode import RealEpisode

__all__ = ["RealEpisode"]
