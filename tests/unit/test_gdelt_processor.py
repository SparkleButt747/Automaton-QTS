"""Unit tests for qts.nlp.gdelt.GDELTProcessor.

Covers:
- T-GDELT-1 to T-GDELT-15
- compute_conflict_intensity: empty events, irrelevant CAMEO codes, relevant codes
- _country_relevance: matched vs unmatched countries
- _cameo_weight: military / sanctions / policy / unknown codes
- Normalisation: clamp to [0, 1]
- Protocol conformance
"""

from __future__ import annotations

from datetime import datetime

import pytest

from qts.nlp.gdelt import (
    GDELTProcessor,
    GDELTProtocol,
    GeopoliticalEvent,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_event(
    event_id: str = "evt-001",
    cameo_code: str = "190",
    countries: tuple[str, ...] = ("russia",),
    intensity: float = 5.0,
) -> GeopoliticalEvent:
    return GeopoliticalEvent(
        event_id=event_id,
        timestamp=datetime(2024, 1, 1, 12, 0),
        cameo_code=cameo_code,
        description="Test event",
        countries=countries,
        intensity=intensity,
        source_url="https://example.com",
    )


# ── Protocol conformance ──────────────────────────────────────────────────────


class TestProtocolConformance:
    def test_T_GDELT_1_processor_implements_protocol(self) -> None:
        proc = GDELTProcessor()
        assert isinstance(proc, GDELTProtocol)

    def test_T_GDELT_2_has_compute_conflict_intensity(self) -> None:
        proc = GDELTProcessor()
        assert hasattr(proc, "compute_conflict_intensity")
        assert callable(proc.compute_conflict_intensity)


# ── compute_conflict_intensity — empty / irrelevant ───────────────────────────


class TestComputeConflictIntensityEmpty:
    def test_T_GDELT_3_empty_events_returns_zero(self) -> None:
        proc = GDELTProcessor()
        assert proc.compute_conflict_intensity([], "AAPL") == pytest.approx(0.0)

    def test_T_GDELT_4_all_irrelevant_cameo_codes_returns_zero(self) -> None:
        """Events with CAMEO codes not in the relevant set should score 0."""
        proc = GDELTProcessor()
        events = [_make_event(cameo_code="010"), _make_event(cameo_code="050")]
        assert proc.compute_conflict_intensity(events, "AAPL") == pytest.approx(0.0)


# ── compute_conflict_intensity — relevant codes ───────────────────────────────


class TestComputeConflictIntensityRelevant:
    def test_T_GDELT_5_military_event_produces_nonzero_score(self) -> None:
        proc = GDELTProcessor()
        events = [_make_event(cameo_code="190", intensity=5.0)]
        score = proc.compute_conflict_intensity(events, "energy")
        assert score > 0.0

    def test_T_GDELT_6_score_in_unit_interval(self) -> None:
        proc = GDELTProcessor()
        events = [_make_event(cameo_code="190", intensity=1000.0)]
        score = proc.compute_conflict_intensity(events, "energy")
        assert 0.0 <= score <= 1.0

    def test_T_GDELT_7_sanctions_code_produces_score(self) -> None:
        proc = GDELTProcessor()
        events = [_make_event(cameo_code="163", intensity=5.0, countries=("russia",))]
        score = proc.compute_conflict_intensity(events, "XOM")
        assert score > 0.0

    def test_T_GDELT_8_policy_code_produces_score(self) -> None:
        proc = GDELTProcessor()
        events = [_make_event(cameo_code="120", intensity=5.0, countries=("china",))]
        score = proc.compute_conflict_intensity(events, "BABA")
        assert score > 0.0

    def test_T_GDELT_9_direct_country_match_gives_higher_score_than_no_match(self) -> None:
        """Energy symbol with Russia event scores higher than unrelated symbol."""
        proc = GDELTProcessor()
        events = [_make_event(cameo_code="190", countries=("russia",), intensity=5.0)]
        score_energy = proc.compute_conflict_intensity(events, "XOM")
        score_other = proc.compute_conflict_intensity(events, "ZZZMADEUP")
        assert score_energy >= score_other

    def test_T_GDELT_10_higher_intensity_gives_higher_score(self) -> None:
        proc = GDELTProcessor()
        events_low = [_make_event(cameo_code="190", intensity=1.0)]
        events_high = [_make_event(cameo_code="190", intensity=8.0)]
        score_low = proc.compute_conflict_intensity(events_low, "energy")
        score_high = proc.compute_conflict_intensity(events_high, "energy")
        assert score_high > score_low

    def test_T_GDELT_11_zero_intensity_event_returns_zero(self) -> None:
        proc = GDELTProcessor()
        events = [_make_event(cameo_code="190", intensity=0.0)]
        score = proc.compute_conflict_intensity(events, "energy")
        assert score == pytest.approx(0.0)


# ── _cameo_weight ─────────────────────────────────────────────────────────────


class TestCameoWeight:
    def test_T_GDELT_12_military_codes_get_weight_one(self) -> None:
        proc = GDELTProcessor()
        for code in ["190", "191", "192", "193", "194", "195", "196"]:
            assert proc._cameo_weight(code) == pytest.approx(1.0), f"code={code}"

    def test_T_GDELT_13_sanctions_code_gets_weight_08(self) -> None:
        proc = GDELTProcessor()
        assert proc._cameo_weight("163") == pytest.approx(0.8)

    def test_T_GDELT_14_policy_codes_get_weight_06(self) -> None:
        proc = GDELTProcessor()
        for code in ["120", "121", "122", "123"]:
            assert proc._cameo_weight(code) == pytest.approx(0.6), f"code={code}"

    def test_T_GDELT_15_unknown_code_gets_zero_weight(self) -> None:
        proc = GDELTProcessor()
        assert proc._cameo_weight("999") == pytest.approx(0.0)
        assert proc._cameo_weight("010") == pytest.approx(0.0)


# ── _country_relevance ────────────────────────────────────────────────────────


class TestCountryRelevance:
    def test_T_GDELT_16_known_country_and_matching_symbol_returns_one(self) -> None:
        proc = GDELTProcessor()
        # Russia -> ["energy", "XOM", "CVX", "GAZP"]
        weight = proc._country_relevance(("russia",), "XOM")
        assert weight == pytest.approx(1.0)

    def test_T_GDELT_17_known_country_no_symbol_match_returns_half(self) -> None:
        proc = GDELTProcessor()
        weight = proc._country_relevance(("russia",), "ZZZMADEUP")
        assert weight == pytest.approx(0.5)

    def test_T_GDELT_18_unknown_country_returns_half(self) -> None:
        proc = GDELTProcessor()
        weight = proc._country_relevance(("atlantis",), "energy")
        assert weight == pytest.approx(0.5)

    def test_T_GDELT_19_sector_match_returns_one(self) -> None:
        proc = GDELTProcessor()
        # Ukraine -> ["energy", "wheat", "corn"]
        weight = proc._country_relevance(("ukraine",), "energy")
        assert weight == pytest.approx(1.0)
