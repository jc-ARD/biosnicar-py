"""Tests for the retrieval quality flag system (C1)."""

import numpy as np
import pytest

from biosnicar.sea_ice.quality_flags import (
    LOW_CONFIDENCE_THRESHOLD,
    POOR_FIT_THRESHOLD,
    QualityFlag,
    compute_quality_flags,
    describe_quality_flags,
)


_CLEAN = dict(
    cost=0.001,
    confidence=0.95,
    converged=True,
    parameters={"wind_speed_ms": 5.0},
    bounds={"wind_speed_ms": (0.0, 15.0)},
    surface_type="FYI_bare",
    cost_per_type={"FYI_bare": 0.001, "FYI_snow": 2.0},
)


def _flags(**overrides):
    kw = dict(_CLEAN)
    kw.update(overrides)
    return compute_quality_flags(**kw)


class TestIndividualFlags:
    def test_clean_retrieval_no_flags(self):
        assert _flags() == 0

    def test_poor_fit(self):
        assert _flags(cost=POOR_FIT_THRESHOLD * 2) & QualityFlag.POOR_FIT
        assert not _flags(cost=POOR_FIT_THRESHOLD / 2) & QualityFlag.POOR_FIT

    def test_low_confidence(self):
        f = _flags(confidence=LOW_CONFIDENCE_THRESHOLD / 2)
        assert f & QualityFlag.LOW_CONFIDENCE
        assert not _flags(confidence=0.5) & QualityFlag.LOW_CONFIDENCE

    def test_at_bounds_upper(self):
        f = _flags(parameters={"wind_speed_ms": 14.95})
        assert f & QualityFlag.AT_BOUNDS

    def test_at_bounds_lower(self):
        f = _flags(parameters={"wind_speed_ms": 0.05})
        assert f & QualityFlag.AT_BOUNDS

    def test_at_bounds_interior_clear(self):
        assert not _flags(parameters={"wind_speed_ms": 7.0}) & QualityFlag.AT_BOUNDS

    def test_at_bounds_unknown_param_ignored(self):
        f = _flags(parameters={"mystery": 1e9}, bounds={})
        assert not f & QualityFlag.AT_BOUNDS

    def test_spectrally_ambiguous(self):
        f = _flags(cost_per_type={"a": 0.95, "b": 1.0})
        assert f & QualityFlag.SPECTRALLY_AMBIGUOUS
        f = _flags(cost_per_type={"a": 0.5, "b": 1.0})
        assert not f & QualityFlag.SPECTRALLY_AMBIGUOUS

    def test_single_type_not_ambiguous(self):
        assert not _flags(cost_per_type={"a": 0.5}) & QualityFlag.SPECTRALLY_AMBIGUOUS

    def test_no_convergence(self):
        assert _flags(converged=False) & QualityFlag.NO_CONVERGENCE

    def test_open_water_informational(self):
        assert _flags(surface_type="open_water") & QualityFlag.OPEN_WATER_LIKELY

    def test_young_ice_informational(self):
        assert _flags(surface_type="young_ice") & QualityFlag.YOUNG_ICE_LIKELY

    def test_flags_combine(self):
        f = _flags(cost=1.0, confidence=0.05, converged=False)
        assert f & QualityFlag.POOR_FIT
        assert f & QualityFlag.LOW_CONFIDENCE
        assert f & QualityFlag.NO_CONVERGENCE


class TestDescribe:
    def test_describe_empty(self):
        d = describe_quality_flags(0)
        assert set(d.keys()) == {
            "poor_fit", "low_confidence", "at_bounds", "spectrally_ambiguous",
            "no_convergence", "open_water_likely", "young_ice_likely",
        }
        assert not any(d.values())

    def test_describe_roundtrip(self):
        f = QualityFlag.POOR_FIT | QualityFlag.AT_BOUNDS
        d = describe_quality_flags(f)
        assert d["poor_fit"] and d["at_bounds"]
        assert not d["low_confidence"]

    def test_fits_in_uint8(self):
        all_flags = 0
        for v in vars(QualityFlag).values():
            if isinstance(v, int):
                all_flags |= v
        assert 0 <= all_flags <= 255
        assert np.uint8(all_flags) == all_flags


class TestRetrievalIntegration:
    def test_result_has_flags_and_description(self):
        from biosnicar.sea_ice.open_water import OpenWaterModel
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice

        m = OpenWaterModel()
        obs = m.predict(solzen=60, wind_speed_ms=5.0)
        result = retrieve_sea_ice(
            observed=obs, emulators={"open_water": m}, solzen=60, direct=1,
        )
        # single-emulator fit of its own forward model: clean except the
        # informational open_water bit
        d = result.quality_flag_description()
        assert d["open_water_likely"]
        assert not d["poor_fit"]
        assert not d["no_convergence"]
        assert "open_water_likely" in result.summary()
