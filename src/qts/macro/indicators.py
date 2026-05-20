"""Macro indicator ingestion and processing.

Ingests macro-economic indicators (VIX, yield curve, CPI, unemployment, etc.)
and normalises them for use in regime classification.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MacroSnapshot:
    """Point-in-time snapshot of macro-economic indicators."""

    timestamp: datetime

    # Volatility
    vix: float | None = None
    vix_percentile: float | None = None  # vs historical

    # Rates
    fed_funds_rate: float | None = None
    us_10y_yield: float | None = None
    us_2y_yield: float | None = None
    yield_curve_spread: float | None = None  # 10y - 2y

    # Inflation
    cpi_yoy: float | None = None  # year-over-year
    ppi_yoy: float | None = None

    # Employment
    unemployment_rate: float | None = None
    nonfarm_payrolls: float | None = None

    # Credit
    high_yield_spread: float | None = None  # OAS
    investment_grade_spread: float | None = None

    # Growth
    gdp_qoq: float | None = None  # quarter-over-quarter annualised

    @property
    def yield_curve_inverted(self) -> bool:
        """True if 2s10s spread is negative (recession signal)."""
        if self.yield_curve_spread is not None:
            return self.yield_curve_spread < 0
        if self.us_10y_yield is not None and self.us_2y_yield is not None:
            return self.us_10y_yield < self.us_2y_yield
        return False

    def to_dict(self) -> dict[str, float]:
        """Convert to a flat dict for MacroEngine consumption."""
        result: dict[str, float] = {}
        if self.vix is not None:
            result["vix"] = self.vix
        if self.yield_curve_spread is not None:
            result["yield_curve_spread"] = self.yield_curve_spread
        if self.cpi_yoy is not None:
            result["cpi_yoy"] = self.cpi_yoy
        if self.unemployment_rate is not None:
            result["unemployment_rate"] = self.unemployment_rate
        if self.high_yield_spread is not None:
            result["credit_spread"] = self.high_yield_spread
        if self.gdp_qoq is not None:
            result["gdp_qoq"] = self.gdp_qoq
        return result


class MacroIndicatorFetcher:
    """Fetches macro indicators from external data sources.

    Currently a placeholder — in production, connects to FRED, Alpha Vantage,
    or other macro data providers.
    """

    def fetch_latest(self) -> MacroSnapshot:
        """Fetch the latest macro indicator snapshot.

        Returns a snapshot with whatever data is available.
        """
        logger.info("MacroIndicatorFetcher: fetch_latest() — placeholder")
        return MacroSnapshot(timestamp=datetime.now())

    def fetch_historical(
        self,
        start: datetime,
        end: datetime,
    ) -> list[MacroSnapshot]:
        """Fetch historical macro snapshots for a date range.

        Returns one snapshot per available data point in the range.
        """
        logger.info(
            "MacroIndicatorFetcher: fetch_historical(%s, %s) — placeholder",
            start,
            end,
        )
        return []
