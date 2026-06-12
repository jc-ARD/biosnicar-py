"""Tests for per-emulator classification band masks (C2) and platform SNR
defaults (C3)."""

import numpy as np
import pytest

from biosnicar.bands._core import WVL
from biosnicar.sea_ice.emulator_configs import BAND_MASKS, SEA_ICE_EMULATOR_CONFIGS
from biosnicar.sea_ice.retrieve import (
    _band_centers_um,
    _band_mask_bool,
    _classification_cost,
)
from biosnicar.sea_ice.sensor_config import PLATFORM_SNR, default_obs_uncertainty


class _FakeFit:
    def __init__(self, predicted, best_fit=None, flx_slr=None):
        self.predicted_albedo = predicted
        self.best_fit = best_fit or {}
        self.flx_slr = flx_slr


class TestBandMasks:
    def test_all_configs_have_band_mask(self):
        for name, cfg in SEA_ICE_EMULATOR_CONFIGS.items():
            assert cfg.get("band_mask") in BAND_MASKS, name

    def test_vis_only_range(self):
        sel = _band_mask_bool("vis_only")
        assert WVL[sel].min() >= 0.4 and WVL[sel].max() <= 1.0

    def test_vis_swir_wider_than_vis_only(self):
        assert _band_mask_bool("vis_swir").sum() > _band_mask_bool("vis_only").sum()

    def test_band_centers_sentinel2(self):
        centers = _band_centers_um("sentinel2")
        assert 0.5 < centers["B3"] < 0.6
        assert 1.5 < centers["B11"] < 1.7


class TestClassificationCost:
    def test_swir_deviation_ignored_by_vis_only(self):
        observed = np.full(480, 0.5)
        predicted = observed.copy()
        predicted[WVL > 1.2] += 0.2  # SWIR-only error
        fit = _FakeFit(predicted)
        c_vis = _classification_cost(fit, observed, "vis_only", None, None,
                                     None, None, None, None)
        c_swir = _classification_cost(fit, observed, "vis_swir", None, None,
                                      None, None, None, None)
        assert c_vis == pytest.approx(0.0)
        assert c_swir > 1.0

    def test_uniform_residual_mask_invariant(self):
        # Rescaling makes a spectrally-flat residual cost identical
        # under any mask, so masks never bias against an emulator.
        observed = np.full(480, 0.5)
        fit = _FakeFit(observed + 0.01)
        costs = [
            _classification_cost(fit, observed, m, None, None, None, None,
                                 None, None)
            for m in (None, "vis_only", "vis_swir")
        ]
        assert costs[0] == pytest.approx(costs[1], rel=1e-9)
        assert costs[0] == pytest.approx(costs[2], rel=1e-9)

    def test_regularization_penalty_included(self):
        observed = np.full(480, 0.5)
        fit = _FakeFit(observed.copy(), best_fit={"sea_ice_temperature": -25.0})
        c = _classification_cost(fit, observed, None, None, None, None, None,
                                 {"sea_ice_temperature": (-4.0, 3.0)}, None)
        assert c == pytest.approx(49.0)

    def test_band_mode_vis_only(self):
        flx = np.ones(480)
        observed = np.array([0.5, 0.5])
        predicted = np.full(480, 0.5)
        predicted[WVL > 1.2] = 0.8  # corrupts B11 only
        fit = _FakeFit(predicted, flx_slr=flx)
        c_vis = _classification_cost(
            fit, observed, "vis_only", "sentinel2", ["B3", "B11"],
            None, None, None, None,
        )
        c_all = _classification_cost(
            fit, observed, "vis_swir", "sentinel2", ["B3", "B11"],
            None, None, None, None,
        )
        assert c_vis < c_all

    def test_nan_observations_excluded(self):
        observed = np.full(480, 0.5)
        observed[WVL > 2.5] = np.nan
        fit = _FakeFit(np.full(480, 0.51))
        c = _classification_cost(fit, observed, "vis_swir", None, None,
                                 None, None, None, None)
        assert np.isfinite(c)


class TestPlatformSNR:
    def test_known_platforms(self):
        for p in ("sentinel2", "landsat8", "planetscope", "modis"):
            assert p in PLATFORM_SNR

    def test_lookup_order_and_values(self):
        unc = default_obs_uncertainty("sentinel2", ["B3", "B11", "B12"])
        np.testing.assert_allclose(unc, [0.02, 0.03, 0.04])

    def test_unknown_platform_returns_none(self):
        assert default_obs_uncertainty("worldview3", ["B1"]) is None

    def test_unknown_band_gets_default(self):
        unc = default_obs_uncertainty("sentinel2", ["B99"])
        assert unc[0] == pytest.approx(0.03)
