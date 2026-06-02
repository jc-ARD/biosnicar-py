"""Tests for brine complex refractive index."""

import numpy as np
import pytest
from biosnicar.sea_ice.brine_optics import (
    compute_brine_rfidx, build_brine_rfidx_lut, invalidate_lut_cache, WAVELENGTHS, _LUT_PATH
)


class TestBrineLUT:
    def test_lut_file_exists(self):
        assert _LUT_PATH.exists(), "brine_rfidx.npz not found; run build_brine_rfidx_lut()"

    def test_lut_shape(self):
        data = np.load(str(_LUT_PATH))
        assert data["n_re"].shape[2] == 480
        assert data["n_im"].shape[2] == 480
        assert data["n_re"].ndim == 3
        assert data["n_re"].shape == data["n_im"].shape

    def test_lut_physical_values(self):
        data = np.load(str(_LUT_PATH))
        assert np.all(data["n_re"] >= 1.0), "Real RI must be ≥ 1 everywhere"
        assert np.all(data["n_im"] >= 0.0), "Imaginary RI must be non-negative"


class TestComputeBrineRfidx:
    def test_returns_complex_array(self):
        ri = compute_brine_rfidx(10.0, -10.0)
        assert np.iscomplexobj(ri)
        assert ri.shape == (480,)

    def test_imaginary_part_nonnegative(self):
        ri = compute_brine_rfidx(20.0, -15.0)
        assert np.all(np.imag(ri) >= 0.0)

    def test_real_part_above_one(self):
        ri = compute_brine_rfidx(10.0, -10.0)
        assert np.all(np.real(ri) >= 1.0)

    def test_zero_salinity_near_pure_water(self):
        ri_zero = compute_brine_rfidx(0.0, -5.0)
        ri_low = compute_brine_rfidx(1.0, -5.0)
        # Real part should be close at low salinities
        np.testing.assert_allclose(
            np.real(ri_zero), np.real(ri_low), rtol=0.01,
            err_msg="Near-zero salinity should give near-pure-water RI"
        )

    def test_salinity_increases_real_ri(self):
        ri_low = compute_brine_rfidx(10.0, -10.0)
        ri_high = compute_brine_rfidx(100.0, -10.0)
        # Higher salinity → higher real RI
        assert np.mean(np.real(ri_high)) > np.mean(np.real(ri_low))

    def test_custom_wavelengths(self):
        wvl = np.array([0.5, 1.0, 2.0])
        ri = compute_brine_rfidx(10.0, -10.0, wavelengths_um=wvl)
        assert ri.shape == (3,)

    def test_no_negative_imaginary_part(self):
        for S in [0, 10, 50, 100]:
            for T in [-2, -10, -25]:
                ri = compute_brine_rfidx(S, T)
                assert np.all(np.imag(ri) >= 0), f"S={S}, T={T}"
