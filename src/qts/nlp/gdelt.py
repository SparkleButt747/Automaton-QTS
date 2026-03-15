"""GDELT Global Knowledge Graph event processing."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ── CAMEO code filter sets ────────────────────────────────────────────────────

# Military / conflict events (CAMEO root 19x)
_CAMEO_MILITARY: frozenset[str] = frozenset(
    {
        "190",
        "191",
        "192",
        "193",
        "194",
        "195",
        "196",
    }
)

# Sanctions (CAMEO code 163)
_CAMEO_SANCTIONS: frozenset[str] = frozenset({"163"})

# Policy / diplomatic events (CAMEO root 12x)
_CAMEO_POLICY: frozenset[str] = frozenset(
    {
        "120",
        "121",
        "122",
        "123",
        "124",
        "125",
        "126",
        "127",
        "128",
        "129",
    }
)

RELEVANT_CAMEO_CODES: frozenset[str] = _CAMEO_MILITARY | _CAMEO_SANCTIONS | _CAMEO_POLICY

# ── Country → asset-sector mapping ───────────────────────────────────────────

# Maps a country name (lower-cased) to a list of asset-sector identifiers that
# are likely affected by geopolitical events involving that country.
COUNTRY_SECTOR_MAP: dict[str, list[str]] = {
    "russia": ["energy", "XOM", "CVX", "GAZP"],
    "ukraine": ["energy", "wheat", "corn"],
    "china": ["tech", "BABA", "BIDU", "JD", "semiconductors"],
    "taiwan": ["semiconductors", "TSM"],
    "iran": ["energy", "oil"],
    "saudi arabia": ["energy", "oil", "ARAMCO"],
    "usa": ["SPY", "QQQ", "usd"],
    "europe": ["EUR", "DAX"],
    "israel": ["defense", "tech"],
    "north korea": ["defense", "semiconductors"],
}

# ── Domain types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GeopoliticalEvent:
    """A single GDELT-derived geopolitical event record.

    Attributes:
        event_id: Unique GDELT event identifier.
        timestamp: UTC datetime of the event.
        cameo_code: CAMEO event-type code (e.g. "190" for military action).
        description: Human-readable event description.
        countries: Countries involved (lower-cased).
        intensity: Raw GDELT event intensity score (typically positive).
        source_url: Original news source URL.
    """

    event_id: str
    timestamp: datetime
    cameo_code: str
    description: str
    countries: tuple[str, ...]
    intensity: float
    source_url: str


# ── Protocol (for dependency injection) ──────────────────────────────────────


@runtime_checkable
class GDELTProtocol(Protocol):
    """Dependency-injection interface for GDELT-style event processors."""

    def compute_conflict_intensity(
        self, events: list[GeopoliticalEvent], symbol: str
    ) -> float:
        """Compute a normalised conflict intensity score for a given symbol.

        Args:
            events: List of geopolitical events to evaluate.
            symbol: Asset symbol or sector identifier to score.

        Returns:
            Float in [0.0, 1.0] representing weighted conflict intensity.
        """
        ...


# ── Implementation ────────────────────────────────────────────────────────────


@dataclass
class _SymbolRelevanceCache:
    """Internal cache for symbol relevance lookups."""

    cache: dict[tuple[str, str], float] = field(default_factory=dict)


class GDELTProcessor:
    """Processes GDELT events and derives asset-level conflict intensity scores.

    Filters events by CAMEO code relevance (military 19x, sanctions 163,
    policy 12x) and computes a relevance-weighted intensity score per symbol.

    The score reflects how strongly the filtered events are expected to affect
    a given asset, taking into account:
    - Whether the CAMEO code is in the relevant set.
    - Whether any involved country maps to the requested symbol.
    - The raw GDELT event intensity.

    The final score is normalised to [0.0, 1.0] via sigmoid compression.
    """

    # Intensity scaling factor before sigmoid — controls sensitivity
    _SCALE: float = 0.1
    # Maximum raw intensity we normalise against (used for simple clamp)
    _MAX_INTENSITY: float = 10.0

    def _country_relevance(self, countries: tuple[str, ...], symbol: str) -> float:
        """Return a relevance weight in [0, 1] for (countries, symbol).

        Returns 1.0 if any country in *countries* maps to *symbol* (exact or
        sector match), otherwise 0.5 for partial signal contamination.

        Args:
            countries: Tuple of lower-cased country names from the event.
            symbol: Asset symbol or sector identifier.

        Returns:
            Relevance weight in {0.5, 1.0}.
        """
        symbol_lower = symbol.lower()
        for country in countries:
            sectors = COUNTRY_SECTOR_MAP.get(country.lower(), [])
            for sector in sectors:
                if symbol_lower in sector.lower() or sector.lower() in symbol_lower:
                    return 1.0
        # No direct match — apply a small base weight for systemic spillover
        return 0.5

    def _cameo_weight(self, cameo_code: str) -> float:
        """Return an event-type weight based on the CAMEO code category.

        Military events get full weight; sanctions and policy events get
        partial weight to reflect lower immediate market impact.

        Args:
            cameo_code: CAMEO numeric code string.

        Returns:
            Weight in {1.0, 0.8, 0.6, 0.0}.
        """
        if cameo_code in _CAMEO_MILITARY:
            return 1.0
        if cameo_code in _CAMEO_SANCTIONS:
            return 0.8
        if cameo_code in _CAMEO_POLICY:
            return 0.6
        return 0.0  # irrelevant event type

    def compute_conflict_intensity(
        self, events: list[GeopoliticalEvent], symbol: str
    ) -> float:
        """Compute a normalised conflict intensity score for *symbol*.

        Steps:
        1. Filter events whose CAMEO code is in the relevant set.
        2. For each remaining event, compute:
           ``weight = cameo_weight * country_relevance * event.intensity``
        3. Sum weights and normalise to [0.0, 1.0] via ``min(sum / max_sum, 1.0)``.

        Args:
            events: GDELT events to process.
            symbol: Asset symbol or sector to score (e.g. ``"AAPL"`` or ``"energy"``).

        Returns:
            Float in [0.0, 1.0].  Returns 0.0 if *events* is empty or no
            relevant events exist.
        """
        if not events:
            return 0.0

        weighted_sum = 0.0
        for event in events:
            cameo_w = self._cameo_weight(event.cameo_code)
            if cameo_w == 0.0:
                continue  # irrelevant CAMEO code
            country_w = self._country_relevance(event.countries, symbol)
            weighted_sum += cameo_w * country_w * abs(event.intensity)

        if weighted_sum == 0.0:
            return 0.0

        # Normalise: a theoretical maximum of N events each at max intensity
        # We simply clip to 1.0 rather than using a fixed normalization constant.
        max_possible = len(events) * 1.0 * 1.0 * self._MAX_INTENSITY
        if max_possible <= 0.0:
            return 0.0

        return min(weighted_sum / max_possible, 1.0)
