"""Tests for hyperspectral resampling utilities (D3)."""

import numpy as np
import pytest

from biosnicar.sea_ice.spectral_utils import (
    MODEL_WAVELENGTHS_NM,
    estimate_instrument_uncertainty,
    resample_to_model_grid,
    trim_to_vis,
    vis_wavelength_mask,
)


@pytest.fixture
def asd_grid():
    """ASD-like instrument grid: 350-2500 nm at 1 nm."""
    return np.arange(350.0, 2501.0)


class TestResample:
    @pytest.mark.parametrize("method", ["gaussian", "linear", "box"])
    def test_flat_spectrum_stays_flat(self, asd_grid, method):
        flat = np.full(len(asd_grid), 0.42)
        out = resample_to_model_grid(flat, asd_grid, method=method, fwhm=3.0)
        covered = np.isfinite(out)
        assert covered.any()
        np.testing.assert_allclose(out[covered], 0.42, atol=1e-9)

    def test_shape_and_nan_outside_coverage(self, asd_grid):
        out = resample_to_model_grid(np.ones(len(asd_grid)), asd_grid)
        assert out.shape == (480,)
        assert np.all(np.isnan(out[MODEL_WAVELENGTHS_NM < 350]))
        assert np.all(np.isnan(out[MODEL_WAVELENGTHS_NM > 2500]))
        in_range = (MODEL_WAVELENGTHS_NM >= 350) & (MODEL_WAVELENGTHS_NM <= 2500)
        assert np.all(np.isfinite(out[in_range]))

    def test_gaussian_smooths_spike(self, asd_grid):
        spec = np.zeros(len(asd_grid))
        spec[asd_grid == 855.0] = 1.0
        out = resample_to_model_grid(spec, asd_grid, method="gaussian", fwhm=10.0)
        i = int(np.argmin(np.abs(MODEL_WAVELENGTHS_NM - 855.0)))
        assert 0 < out[i] < 1.0

    def test_linear_recovers_gradient(self, asd_grid):
        spec = asd_grid / 2500.0
        out = resample_to_model_grid(spec, asd_grid, method="linear")
        i = int(np.argmin(np.abs(MODEL_WAVELENGTHS_NM - 1005.0)))
        assert out[i] == pytest.approx(1005.0 / 2500.0, rel=1e-6)

    def test_unsorted_input(self):
        wvl = np.array([600.0, 400.0, 500.0])
        spec = np.array([0.6, 0.4, 0.5])
        out = resample_to_model_grid(spec, wvl, method="linear")
        i = int(np.argmin(np.abs(MODEL_WAVELENGTHS_NM - 505.0)))
        assert out[i] == pytest.approx(0.505, rel=1e-3)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            resample_to_model_grid(np.ones(5), np.ones(6))

    def test_unknown_method_raises(self, asd_grid):
        with pytest.raises(ValueError, match="unknown method"):
            resample_to_model_grid(np.ones(len(asd_grid)), asd_grid, method="cubic")


class TestVis:
    def test_mask_range(self):
        mask = vis_wavelength_mask()
        assert mask.shape == (480,)
        sel = MODEL_WAVELENGTHS_NM[mask]
        assert sel.min() >= 400 and sel.max() <= 1000

    def test_trim_matches_mask(self):
        spec = np.arange(480.0)
        trimmed = trim_to_vis(spec)
        assert len(trimmed) == int(vis_wavelength_mask().sum())
        np.testing.assert_array_equal(trimmed, spec[vis_wavelength_mask()])

    def test_trim_wrong_length_raises(self):
        with pytest.raises(ValueError, match="480"):
            trim_to_vis(np.ones(100))


class TestUncertainty:
    def test_scalar_snr(self):
        sig = np.array([0.5, 1.0])
        unc = estimate_instrument_uncertainty(sig, 100.0)
        np.testing.assert_allclose(unc, [0.006, 0.011])

    def test_dict_with_floor(self):
        unc = estimate_instrument_uncertainty(
            np.array([0.5]), {"snr": 50, "floor": 0.01},
        )
        assert unc[0] == pytest.approx(0.02)

    def test_callable(self):
        unc = estimate_instrument_uncertainty(
            np.ones(4), lambda idx: 100.0 + idx,
        )
        assert unc[0] > unc[3]
