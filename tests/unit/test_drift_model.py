"""T-DRIFT-1..6: SentimentDriftModel ramp/decay/direction/noise."""

from __future__ import annotations

from datetime import datetime, timedelta

from qts.world.drift_model import SentimentDriftModel

_EVENT = datetime(2023, 12, 13, 19, 0, 0)


def _model(direction: float, noise: float = 0.0) -> SentimentDriftModel:
    return SentimentDriftModel(
        direction=direction,
        onset_lag=timedelta(minutes=10),
        peak_bps=100.0,
        decay_halflife=timedelta(minutes=30),
        noise_std_bps=noise,
        event_time=_EVENT,
        seed=7,
    )


def test_zero_before_event() -> None:  # T-DRIFT-1
    m = _model(1.0, noise=5.0)
    assert m.value_at(_EVENT - timedelta(minutes=1)) == 0.0


def test_ramps_to_peak_at_onset_lag() -> None:  # T-DRIFT-2
    m = _model(1.0)
    assert abs(m.value_at(_EVENT + timedelta(minutes=10)) - 100.0) < 1e-9


def test_half_ramp_midway() -> None:  # T-DRIFT-3
    m = _model(1.0)
    assert abs(m.value_at(_EVENT + timedelta(minutes=5)) - 50.0) < 1e-9


def test_decays_by_half_after_one_halflife() -> None:  # T-DRIFT-4
    m = _model(1.0)
    # peak at +10min, then one halflife (30min) later -> 50.0
    assert abs(m.value_at(_EVENT + timedelta(minutes=40)) - 50.0) < 1e-9


def test_hawkish_direction_is_negative() -> None:  # T-DRIFT-5
    m = _model(-1.0)
    assert m.value_at(_EVENT + timedelta(minutes=10)) < 0.0


def test_noise_is_seed_deterministic() -> None:  # T-DRIFT-6
    m = _model(0.0, noise=20.0)
    t = _EVENT + timedelta(minutes=15)
    assert m.value_at(t) == m.value_at(t)
