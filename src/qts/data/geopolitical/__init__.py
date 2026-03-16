"""Geopolitical data sub-package: GDELT event ingestion and theme extraction.

Provides:
    - :class:`~qts.data.geopolitical.gdelt_client.GDELTClient`: async HTTP
      client for the GDELT Doc API (no API key required).
    - :class:`~qts.data.geopolitical.gdelt_client.GDELTClientProtocol`:
      dependency-injection protocol for the GDELT client.
"""
