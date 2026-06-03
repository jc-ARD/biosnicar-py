"""Tests for the improved brine complex refractive index module.

Key physics changes in v0.2:
  - Brine RI now computed at the liquidus salinity S_b = -18.7 × T_C,
    not the bulk ice salinity.  At T=-10°C, S_b ≈ 187 psu.
  - LUT is T-only (2D: nT × 480), since S_b is determined by T.
  - NaCl has no absorption above 400 nm: _K_VIS_ALPHA = 0.
  - Liquidus salinity capped at 250 psu (eutectic limit).
"""

import numpy as np
import pytest
from biosnicar.sea_ice.brine_optics import (
    compute_brine_rfidx, build_brine_rfidx_lut, invalidate_lut_cache,
    WAVELENGTHS, _LUT_PATH,
)


class TestBrineLUT:
    def test_lut_file_exists(self):
        assert _LUT_PATH.exists(), "brine_rfidx.npz not found"

    def test_lut_shape(self):
        """LUT is now (nT, 480) — T-only, since brine S is determined by T."""
        data = np.load(str(_LUT_PATH))
        assert data["n_re"].ndim == 2, "Expect 2D LUT (nT × 480)"
        assert data["n_re"].shape[1] == 480
        assert data["n_re"].shape == data["n_im"].shape

    def test_lut_physical_values(self):
        data = np.load(str(_LUT_PATH))
        assert np.all(data["n_re"] >= 1.0), "Real RI must be ≥ 1 everywhere"
        assert np.all(data["n_im"] >= 0.0), "Imaginary RI must be non-negative"

    def test_lut_warmer_temp_gives_higher_real_ri(self):
        """At warmer T, S_brine is lower → smaller real-part correction."""
        data = np.load(str(_LUT_PATH))
        T_grid = data["T_grid"]
        # Colder T → higher S_b (up to cap) → larger Δn_re
        # Find indices for -5°C and -20°C
        idx_warm = np.argmin(np.abs(T_grid - (-5)))
        idx_cold = np.argmin(np.abs(T_grid - (-20)))
        # Average real RI at 500 nm
        n_warm = data["n_re"][idx_warm, 29]   # index 29 ≈ 0.5 µm
        n_cold = data["n_re"][idx_cold, 29]
        # Colder → higher liquidus S_b → higher real RI (capped at 250 psu)
        assert n_cold >= n_warm, "Colder brine should have higher real RI"


class TestComputeBrineRfidx:
    def test_returns_complex_array(self):
        ri = compute_brine_rfidx(-10.0)
        assert np.iscomplexobj(ri)
        assert ri.shape == (480,)

    def test_old_positional_api_still_works(self):
        """Legacy call compute_brine_rfidx(S, T) still returns a result."""
        ri = compute_brine_rfidx(10.0, -10.0)    # old signature
        assert ri.shape == (480,)

    def test_imaginary_part_nonnegative(self):
        for T in [-2, -10, -20, -25]:
            ri = compute_brine_rfidx(T)
            assert np.all(np.imag(ri) >= 0.0), f"T={T}"

    def test_real_part_above_one(self):
        for T in [-5, -10, -20]:
            ri = compute_brine_rfidx(T)
            assert np.all(np.real(ri) >= 1.0), f"T={T}"

    def test_colder_temp_increases_real_ri(self):
        """Colder T → higher liquidus S_b → larger Q&F real-part correction."""
        ri_warm = compute_brine_rfidx(-5.0)    # S_b ≈  94 psu
        ri_cold = compute_brine_rfidx(-15.0)   # S_b ≈ 250 psu (capped)
        assert np.mean(np.real(ri_cold)) > np.mean(np.real(ri_warm))

    def test_liquidus_correction_significant(self):
        """At T=-10°C, S_b=187 psu gives a real-RI correction > 0.03 (vs
        the v0.1 value of 0.0016 at bulk S=8 psu).
        """
        ri = compute_brine_rfidx(-10.0)
        import pandas as pd
        import biosnicar
        rowe = pd.read_csv(
            biosnicar.DATA_DIR / "OP_data" / "480band" / "refractive_index_water_273K_Rowe2020.csv"
        )
        n_water_500 = float(rowe["n"].iloc[29])   # index 29 ≈ 0.5 µm
        n_brine_500 = float(np.real(ri[29]))
        delta_n = n_brine_500 - n_water_500
        assert delta_n > 0.03, (
            f"Δn_re at -10°C should be > 0.03 (liquidus S_b=187 psu), got {delta_n:.4f}"
        )

    def test_nir_k_near_water(self):
        """In NIR (> 700 nm) brine k should be close to pure water k
        (NaCl has no absorption above 400 nm).
        """
        ri = compute_brine_rfidx(-10.0)
        import pandas as pd, biosnicar
        rowe = pd.read_csv(
            biosnicar.DATA_DIR / "OP_data" / "480band" / "refractive_index_water_273K_Rowe2020.csv"
        )
        # NIR band indices 50-200 (0.7-2.2 µm)
        k_brine = np.imag(ri[50:200])
        k_water = rowe["k"].values[50:200]
        # Brine k in NIR should be within 10% of pure water
        np.testing.assert_allclose(k_brine, k_water, rtol=0.10,
                                   err_msg="NIR brine k should be close to pure water")

    def test_custom_wavelengths(self):
        wvl = np.array([0.5, 1.0, 2.0])
        ri = compute_brine_rfidx(-10.0, wavelengths_um=wvl)
        assert ri.shape == (3,)

    def test_no_negative_imaginary(self):
        for T in [-2, -10, -25]:
            ri = compute_brine_rfidx(T)
            assert np.all(np.imag(ri) >= 0), f"T={T}"
