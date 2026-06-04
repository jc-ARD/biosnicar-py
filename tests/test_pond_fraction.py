"""Tests for melt pond areal fraction blending."""

import numpy as np
import pytest

from biosnicar import run_model
from biosnicar.sea_ice.pond_fraction import blend_pond_fraction
from biosnicar.sea_ice.presets import FYI_SUMMER_BARE, FYI_POND_SHALLOW


@pytest.fixture(scope="module")
def ice():
    return run_model(preset=FYI_SUMMER_BARE, solzen=60)


@pytest.fixture(scope="module")
def pond():
    return run_model(preset=FYI_POND_SHALLOW, solzen=60)


class TestBlendPondFraction:
    def test_f0_returns_ice_albedo(self, ice, pond):
        result = blend_pond_fraction(ice, pond, f=0.0)
        np.testing.assert_array_equal(result.albedo, ice.albedo)

    def test_f1_returns_pond_albedo(self, ice, pond):
        result = blend_pond_fraction(ice, pond, f=1.0)
        np.testing.assert_array_equal(result.albedo, pond.albedo)

    def test_f_half_midpoint_spectrum(self, ice, pond):
        result = blend_pond_fraction(ice, pond, f=0.5)
        expected = 0.5 * ice.albedo + 0.5 * pond.albedo
        np.testing.assert_allclose(result.albedo, expected)

    def test_bba_between_ice_and_pond(self, ice, pond):
        result = blend_pond_fraction(ice, pond, f=0.3)
        lo, hi = min(ice.BBA, pond.BBA), max(ice.BBA, pond.BBA)
        assert lo <= result.BBA <= hi

    def test_bba_monotone_in_f(self, ice, pond):
        bbas = [blend_pond_fraction(ice, pond, f=f).BBA
                for f in [0.0, 0.25, 0.5, 0.75, 1.0]]
        # BBA should be monotonically ordered (ice BBA > pond BBA for summer)
        assert bbas == sorted(bbas, reverse=(ice.BBA > pond.BBA))

    def test_bba_aliases_correct(self, ice, pond):
        result = blend_pond_fraction(ice, pond, f=0.4)
        assert result.broadband == result.BBA
        assert result.visible == result.BBAVIS
        assert result.nir == result.BBANIR

    def test_bbavis_bbanir_physical(self, ice, pond):
        result = blend_pond_fraction(ice, pond, f=0.3)
        assert 0.0 <= result.BBAVIS <= 1.0
        assert 0.0 <= result.BBANIR <= 1.0

    def test_subsurface_fields_none(self, ice, pond):
        result = blend_pond_fraction(ice, pond, f=0.5)
        assert result.F_up is None
        assert result.F_dwn is None
        assert result.heat_rt is None

    def test_invalid_f_raises(self, ice, pond):
        with pytest.raises(ValueError):
            blend_pond_fraction(ice, pond, f=-0.1)
        with pytest.raises(ValueError):
            blend_pond_fraction(ice, pond, f=1.1)

    def test_spectrum_shape_preserved(self, ice, pond):
        result = blend_pond_fraction(ice, pond, f=0.3)
        assert result.albedo.shape == (480,)

    def test_to_platform_works_on_blended(self, ice, pond):
        result = blend_pond_fraction(ice, pond, f=0.3)
        bands = result.to_platform("sentinel2")
        assert hasattr(bands, "B3")


class TestRunModelPondFraction:
    def test_convenience_kwarg_returns_outputs(self):
        result = run_model(
            preset="FYI_SUMMER_BARE", solzen=60,
            pond_fraction=0.30, pond_depth=0.15,
        )
        assert 0.0 <= result.BBA <= 1.0
        assert result.albedo.shape == (480,)

    def test_pond_fraction_zero_matches_pure_ice(self, ice):
        result = run_model(preset="FYI_SUMMER_BARE", solzen=60, pond_fraction=0.0)
        np.testing.assert_allclose(result.albedo, ice.albedo, rtol=1e-6)

    def test_pond_fraction_darkens_ice(self, ice):
        mixed = run_model(
            preset="FYI_SUMMER_BARE", solzen=60,
            pond_fraction=0.40, pond_depth=0.15,
        )
        assert mixed.BBA < ice.BBA

    def test_custom_pond_depth_changes_result(self):
        shallow = run_model(preset="FYI_SUMMER_BARE", solzen=60,
                            pond_fraction=0.3, pond_depth=0.05)
        deep = run_model(preset="FYI_SUMMER_BARE", solzen=60,
                         pond_fraction=0.3, pond_depth=0.40)
        assert shallow.BBA != deep.BBA

    def test_custom_pond_preset_string(self):
        result = run_model(
            preset="FYI_SUMMER_BARE", solzen=60,
            pond_fraction=0.2, pond_preset="FYI_POND_DEEP",
        )
        assert 0.0 <= result.BBA <= 1.0

    def test_invalid_pond_fraction_raises(self):
        with pytest.raises(ValueError):
            run_model(preset="FYI_SUMMER_BARE", solzen=60, pond_fraction=1.5)
