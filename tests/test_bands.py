"""Tests for biosnicar.bands platform band convolution module."""

import numpy as np
import pytest

from biosnicar.bands._core import (
    WVL,
    N_WVL,
    interval_average,
    srf_convolve,
    srf_convolve_stack,
    load_srf,
    load_srf_stack,
)
from biosnicar.bands import BandResult, to_platform


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def flat_albedo():
    """Flat albedo = 0.5 across all wavelengths."""
    return np.full(N_WVL, 0.5)


@pytest.fixture
def uniform_flux():
    """Uniform solar flux = 1.0 across all wavelengths."""
    return np.ones(N_WVL)


@pytest.fixture
def ramp_albedo():
    """Albedo ramp from 1.0 (short λ) to 0.0 (long λ)."""
    return np.linspace(1.0, 0.0, N_WVL)


# ── Core utilities ──────────────────────────────────────────────────

class TestWVLGrid:
    def test_length(self):
        assert N_WVL == 480

    def test_start(self):
        assert abs(WVL[0] - 0.205) < 1e-6

    def test_step(self):
        assert abs(WVL[1] - WVL[0] - 0.01) < 1e-8


class TestIntervalAverage:
    def test_flat_albedo(self, flat_albedo, uniform_flux):
        """Flat albedo should return 0.5 for any interval."""
        assert interval_average(flat_albedo, uniform_flux, 0.3, 0.7) == pytest.approx(0.5)

    def test_full_range(self, flat_albedo, uniform_flux):
        result = interval_average(flat_albedo, uniform_flux, 0.2, 5.0)
        assert result == pytest.approx(0.5)

    def test_empty_interval(self, flat_albedo, uniform_flux):
        """Interval with no wavelength bins should return NaN."""
        result = interval_average(flat_albedo, uniform_flux, 10.0, 11.0)
        assert np.isnan(result)


class TestSRFConvolve:
    def test_flat_albedo_tophat(self, flat_albedo, uniform_flux):
        """Flat albedo + tophat SRF → 0.5."""
        srf = np.zeros(N_WVL)
        srf[10:20] = 1.0
        assert srf_convolve(flat_albedo, uniform_flux, srf) == pytest.approx(0.5)

    def test_zero_srf(self, flat_albedo, uniform_flux):
        """Zero SRF should return NaN."""
        srf = np.zeros(N_WVL)
        assert np.isnan(srf_convolve(flat_albedo, uniform_flux, srf))


class TestSRFConvolveStack:
    """srf_convolve_stack is the vectorised equivalent of srf_convolve."""

    def test_matches_single_band(self, ramp_albedo, uniform_flux):
        """Row-wise stack result equals looping srf_convolve per band."""
        names = ["B2", "B3", "B4", "B8A"]
        srf = load_srf("sentinel2_msi")
        stack = load_srf_stack("sentinel2_msi", names)
        got = srf_convolve_stack(ramp_albedo, uniform_flux, stack)
        want = [srf_convolve(ramp_albedo, uniform_flux, srf[n]) for n in names]
        assert got == pytest.approx(want, rel=1e-12)

    def test_flat_albedo_tophat(self, flat_albedo, uniform_flux):
        """Flat albedo + tophat SRFs → 0.5 for every band."""
        stack = np.zeros((2, N_WVL))
        stack[0, 10:20] = 1.0
        stack[1, 30:50] = 1.0
        out = srf_convolve_stack(flat_albedo, uniform_flux, stack)
        assert out == pytest.approx(0.5)

    def test_zero_srf_row_is_nan(self, flat_albedo, uniform_flux):
        """A zero-SRF row is NaN; other rows are unaffected."""
        stack = np.zeros((2, N_WVL))
        stack[1, 10:20] = 1.0
        out = srf_convolve_stack(flat_albedo, uniform_flux, stack)
        assert np.isnan(out[0])
        assert out[1] == pytest.approx(0.5)


class TestLoadSRFStack:
    def test_order_and_shape(self):
        """Stack rows follow the requested band order and match load_srf."""
        names = ["B3", "B1", "B8A"]
        srf = load_srf("sentinel2_msi")
        stack = load_srf_stack("sentinel2_msi", names)
        assert stack.shape == (len(names), N_WVL)
        for i, name in enumerate(names):
            assert np.array_equal(stack[i], srf[name])


# ── GCM platforms ───────────────────────────────────────────────────

class TestCESM2Band:
    def test_flat_albedo(self, flat_albedo, uniform_flux):
        r = to_platform(flat_albedo, "cesm2band", uniform_flux)
        assert r.platform == "cesm2band"
        assert r.vis == pytest.approx(0.5)
        assert r.nir == pytest.approx(0.5)
        assert "vis" in r.band_names
        assert "nir" in r.band_names

    def test_ramp_vis_gt_nir(self, ramp_albedo, uniform_flux):
        """Ramp albedo: VIS (shorter λ) should be brighter than NIR."""
        r = to_platform(ramp_albedo, "cesm2band", uniform_flux)
        assert r.vis > r.nir


class TestCESMRRTMG:
    def test_flat_albedo(self, flat_albedo, uniform_flux):
        r = to_platform(flat_albedo, "cesmrrtmg", uniform_flux)
        assert r.platform == "cesmrrtmg"
        assert len(r.band_names) == 14
        for name in r.band_names:
            val = getattr(r, name)
            # Some bands may be NaN if outside model range
            if np.isfinite(val):
                assert val == pytest.approx(0.5, abs=0.01)


class TestMAR:
    def test_flat_albedo(self, flat_albedo, uniform_flux):
        r = to_platform(flat_albedo, "mar", uniform_flux)
        assert r.platform == "mar"
        assert len(r.band_names) == 4
        for name in r.band_names:
            assert getattr(r, name) == pytest.approx(0.5)


class TestHadCM3:
    def test_flat_albedo(self, flat_albedo, uniform_flux):
        r = to_platform(flat_albedo, "hadcm3", uniform_flux)
        assert r.platform == "hadcm3"
        assert len(r.band_names) == 6
        for name in r.band_names:
            assert getattr(r, name) == pytest.approx(0.5)


# ── Satellite platforms (tophat / SRF) ──────────────────────────────

class TestMODIS:
    def test_flat_albedo(self, flat_albedo, uniform_flux):
        r = to_platform(flat_albedo, "modis", uniform_flux)
        assert r.platform == "modis"
        assert r.B1 == pytest.approx(0.5)
        assert r.B4 == pytest.approx(0.5)
        assert r.VIS == pytest.approx(0.5)
        assert r.NIR == pytest.approx(0.5)

    def test_ndsi_flat(self, flat_albedo, uniform_flux):
        """Flat albedo → NDSI = 0."""
        r = to_platform(flat_albedo, "modis", uniform_flux)
        assert r.NDSI == pytest.approx(0.0, abs=1e-6)


class TestSentinel2:
    def test_flat_albedo(self, flat_albedo, uniform_flux):
        r = to_platform(flat_albedo, "sentinel2", uniform_flux)
        assert r.platform == "sentinel2"
        assert len(r.band_names) == 13
        for name in r.band_names:
            assert getattr(r, name) == pytest.approx(0.5, abs=0.01)

    def test_indices_flat(self, flat_albedo, uniform_flux):
        r = to_platform(flat_albedo, "sentinel2", uniform_flux)
        assert r.NDSI == pytest.approx(0.0, abs=1e-6)
        assert r.NDVI == pytest.approx(0.0, abs=1e-6)
        assert r.II == pytest.approx(1.0, abs=0.01)

    def test_ramp_ndvi_negative(self, ramp_albedo, uniform_flux):
        """Ramp albedo decreases with λ → NIR < Red → NDVI < 0."""
        r = to_platform(ramp_albedo, "sentinel2", uniform_flux)
        assert r.NDVI < 0


class TestSentinel3:
    def test_flat_albedo(self, flat_albedo, uniform_flux):
        r = to_platform(flat_albedo, "sentinel3", uniform_flux)
        assert r.platform == "sentinel3"
        assert len(r.band_names) == 21
        for name in r.band_names:
            assert getattr(r, name) == pytest.approx(0.5, abs=0.01)

    def test_ndci_flat(self, flat_albedo, uniform_flux):
        r = to_platform(flat_albedo, "sentinel3", uniform_flux)
        assert r.NDCI == pytest.approx(0.0, abs=1e-6)


class TestLandsat8:
    def test_flat_albedo(self, flat_albedo, uniform_flux):
        r = to_platform(flat_albedo, "landsat8", uniform_flux)
        assert r.platform == "landsat8"
        assert len(r.band_names) == 7
        for name in r.band_names:
            assert getattr(r, name) == pytest.approx(0.5, abs=0.01)

    def test_ndsi_flat(self, flat_albedo, uniform_flux):
        r = to_platform(flat_albedo, "landsat8", uniform_flux)
        assert r.NDSI == pytest.approx(0.0, abs=1e-6)


# ── BandResult ──────────────────────────────────────────────────────

class TestBandResult:
    def test_as_dict(self, flat_albedo, uniform_flux):
        r = to_platform(flat_albedo, "cesm2band", uniform_flux)
        d = r.as_dict()
        assert "vis" in d
        assert "nir" in d

    def test_repr(self, flat_albedo, uniform_flux):
        r = to_platform(flat_albedo, "cesm2band", uniform_flux)
        s = repr(r)
        assert "cesm2band" in s
        assert "vis" in s


# ── Error handling ──────────────────────────────────────────────────

class TestErrors:
    def test_unknown_platform(self, flat_albedo, uniform_flux):
        with pytest.raises(ValueError, match="Unknown platform"):
            to_platform(flat_albedo, "nonexistent", uniform_flux)


# ── Integration with run_model (optional, slow) ────────────────────

@pytest.mark.slow
class TestIntegration:
    def test_run_model_flx_slr(self):
        """run_model should populate flx_slr on Outputs."""
        from biosnicar import run_model
        o = run_model(solzen=50)
        assert o.flx_slr is not None
        assert len(o.flx_slr) == 480

    def test_full_pipeline(self):
        """Full pipeline: run_model → to_platform for each platform."""
        from biosnicar import run_model
        o = run_model(solzen=50)

        for plat in ["cesm2band", "cesmrrtmg", "mar", "hadcm3",
                      "modis", "sentinel2", "sentinel3", "landsat8"]:
            r = to_platform(o.albedo, plat, o.flx_slr)
            assert r.platform == plat
            assert len(r.band_names) > 0
            # At least one band should be finite
            assert any(np.isfinite(getattr(r, n)) for n in r.band_names)
