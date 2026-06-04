"""Tests for the sea ice extension via the unified run_model() API."""

import numpy as np
import pytest

from biosnicar import run_model
from biosnicar.sea_ice.presets import FYI_WINTER_BARE, FYI_WINTER_SNOW, MYI_WINTER_BARE, ALL_PRESETS
from biosnicar.classes.outputs import Outputs


# ---------------------------------------------------------------------------
# Presets via run_model
# ---------------------------------------------------------------------------

class TestPresets:
    @pytest.mark.parametrize("preset", [FYI_WINTER_BARE, FYI_WINTER_SNOW, MYI_WINTER_BARE])
    def test_preset_dict_produces_valid_outputs(self, preset):
        outputs = run_model(preset=preset, solzen=60)
        assert isinstance(outputs, Outputs)
        assert outputs.albedo.shape == (480,)
        assert 0.0 <= outputs.BBA <= 1.0

    @pytest.mark.parametrize("name", ["FYI_WINTER_BARE", "FYI_WINTER_SNOW", "MYI_WINTER_BARE"])
    def test_preset_by_name(self, name):
        outputs = run_model(preset=name, solzen=60)
        assert isinstance(outputs, Outputs)
        assert 0.0 <= outputs.BBA <= 1.0

    @pytest.mark.parametrize("preset", [FYI_WINTER_BARE, FYI_WINTER_SNOW, MYI_WINTER_BARE])
    def test_preset_bba_in_physical_range(self, preset):
        outputs = run_model(preset=preset, solzen=60)
        assert 0.4 <= outputs.BBA <= 0.9, (
            f"BBA={outputs.BBA:.3f} outside [0.4, 0.9]"
        )

    def test_all_presets_in_registry(self):
        assert "FYI_WINTER_BARE" in ALL_PRESETS
        assert "FYI_WINTER_SNOW" in ALL_PRESETS
        assert "MYI_WINTER_BARE" in ALL_PRESETS

    def test_snow_covered_higher_than_bare(self):
        bare = run_model(preset=FYI_WINTER_BARE, solzen=60)
        snow = run_model(preset=FYI_WINTER_SNOW, solzen=60)
        assert snow.BBA > bare.BBA

    def test_preset_solzen_override(self):
        r60 = run_model(preset=FYI_WINTER_BARE, solzen=60)
        r75 = run_model(preset=FYI_WINTER_BARE, solzen=75)
        assert r75.BBA != r60.BBA   # SZA affects albedo

    def test_preset_impurity_override(self):
        clean = run_model(preset=FYI_WINTER_SNOW, solzen=60)
        dirty = run_model(preset=FYI_WINTER_SNOW, solzen=60, black_carbon=5000)
        assert dirty.BBA < clean.BBA   # black carbon darkens ice

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            run_model(preset="NOT_A_REAL_PRESET", solzen=60)


# ---------------------------------------------------------------------------
# Flat-kwargs sea ice (same style as terrestrial API)
# ---------------------------------------------------------------------------

class TestFlatKwargs:
    def test_single_sea_ice_layer(self):
        outputs = run_model(
            solzen=60,
            layer_type=4,
            dz=1.5,
            rds=500,
            rho=910,
            sea_ice_salinity=8,
            sea_ice_temperature=-10,
            sea_ice_bubble_radius=200,
        )
        assert isinstance(outputs, Outputs)
        assert 0.3 <= outputs.BBA <= 0.9

    def test_two_sea_ice_layers(self):
        outputs = run_model(
            solzen=60,
            layer_type=[4, 4],
            dz=[0.05, 1.45],
            rds=[500, 500],
            rho=[895, 895],
            sea_ice_salinity=[12, 8],
            sea_ice_temperature=[-25, -20],
            sea_ice_bubble_radius=[100, 200],
        )
        assert 0.3 <= outputs.BBA <= 0.9

    def test_snow_on_sea_ice_flat(self):
        outputs = run_model(
            solzen=60,
            layer_type=[0, 4, 4],
            dz=[0.15, 0.05, 1.45],
            rds=[200, 500, 500],
            rho=[300, 895, 895],
            sea_ice_salinity=[None, 12, 8],
            sea_ice_temperature=[None, -25, -20],
            sea_ice_bubble_radius=[None, 100, 200],
        )
        assert 0.5 <= outputs.BBA <= 0.98

    def test_rds_none_for_sea_ice_layers(self):
        """rds=None for layer_type=4 should be silently replaced with dummy."""
        outputs = run_model(
            solzen=60,
            layer_type=[0, 4],
            dz=[0.15, 1.5],
            rds=[200, None],    # None for sea ice layer
            rho=[300, 895],
            sea_ice_salinity=[None, 8],
            sea_ice_temperature=[None, -10],
            sea_ice_bubble_radius=[None, 200],
        )
        assert 0.3 <= outputs.BBA <= 1.0

    def test_diffuse_sky(self):
        outputs = run_model(preset=FYI_WINTER_BARE, solzen=60, direct=0)
        assert 0.3 <= outputs.BBA <= 0.9

    def test_different_atmospheres(self):
        for incoming in [0, 2, 4]:
            outputs = run_model(preset=FYI_WINTER_BARE, solzen=60, incoming=incoming)
            assert 0.0 <= outputs.BBA <= 1.0


# ---------------------------------------------------------------------------
# Output object attributes (same as terrestrial Outputs)
# ---------------------------------------------------------------------------

class TestOutputAttributes:
    def setup_method(self):
        self.outputs = run_model(preset=FYI_WINTER_BARE, solzen=60)

    def test_bba_attribute(self):
        assert 0.0 <= self.outputs.BBA <= 1.0

    def test_broadband_alias(self):
        assert self.outputs.broadband == self.outputs.BBA

    def test_albedo_attribute(self):
        assert self.outputs.albedo.shape == (480,)

    def test_spectrum_alias(self):
        np.testing.assert_array_equal(self.outputs.spectrum, self.outputs.albedo)

    def test_visible_alias(self):
        assert self.outputs.visible == self.outputs.BBAVIS

    def test_nir_alias(self):
        assert self.outputs.nir == self.outputs.BBANIR

    def test_wavelengths_property(self):
        wvl = self.outputs.wavelengths
        assert wvl.shape == (480,)
        assert abs(wvl[0] - 0.205) < 1e-6

    def test_spectrum_all_physical(self):
        assert np.all(self.outputs.albedo >= 0.0)
        assert np.all(self.outputs.albedo <= 1.0)

    def test_to_platform_works(self):
        result = self.outputs.to_platform("sentinel2")
        assert hasattr(result, "B3")

    def test_flx_slr_available(self):
        assert self.outputs.flx_slr is not None
        assert len(self.outputs.flx_slr) == 480


# ---------------------------------------------------------------------------
# Outputs attributes are identical between terrestrial and sea ice
# ---------------------------------------------------------------------------

class TestUnifiedOutputs:
    def test_same_attribute_names(self):
        """Terrestrial and sea ice outputs have identical attribute names."""
        terr = run_model(solzen=60, layer_type=1, rds=500, rho=700, dz=1.5)
        seai = run_model(preset=FYI_WINTER_BARE, solzen=60)
        for attr in ("BBA", "BBAVIS", "BBANIR", "albedo",
                     "broadband", "visible", "nir", "spectrum", "wavelengths"):
            assert hasattr(terr, attr), f"terrestrial missing .{attr}"
            assert hasattr(seai, attr), f"sea ice missing .{attr}"

    def test_to_platform_same_interface(self):
        terr = run_model(solzen=60, layer_type=1, rds=500, rho=700, dz=1.5)
        seai = run_model(preset=FYI_WINTER_BARE, solzen=60)
        for result in [terr.to_platform("modis"), seai.to_platform("modis")]:
            assert hasattr(result, "B1")


# ---------------------------------------------------------------------------
# Physics sanity checks
# ---------------------------------------------------------------------------

class TestSeaIcePhysics:
    def test_higher_salinity_changes_albedo(self):
        # FYI_WINTER_BARE has 2 layers [DL(4), IL(4)] — no SSL in winter.
        low  = run_model(preset="FYI_WINTER_BARE", solzen=60,
                         sea_ice_salinity=[1, 1])
        high = run_model(preset="FYI_WINTER_BARE", solzen=60,
                         sea_ice_salinity=[12, 12])
        assert low.BBA != high.BBA

    def test_myi_and_fyi_both_physical(self):
        for preset in [FYI_WINTER_BARE, MYI_WINTER_BARE]:
            out = run_model(preset=preset, solzen=60)
            assert 0.4 <= out.BBA <= 0.9


# ---------------------------------------------------------------------------
# Regression: terrestrial ice unchanged
# ---------------------------------------------------------------------------

class TestRegressionExistingIce:
    def test_layer_type_0_unchanged(self):
        out = run_model(layer_type=[0, 0], dz=[0.02, 1.0], rds=[500, 500], rho=[400, 400])
        assert 0.5 < out.BBA < 1.0

    def test_layer_type_1_unchanged(self):
        out = run_model(layer_type=[1, 1], dz=[0.02, 1.0], rds=[500, 500], rho=[700, 700])
        assert 0.3 < out.BBA < 1.0


# ---------------------------------------------------------------------------
# Backwards-compatibility: deprecated API still works (with warnings)
# ---------------------------------------------------------------------------

class TestDeprecatedAPIStillWorks:
    def test_seaicecolumn_from_preset(self):
        from biosnicar.sea_ice.api import SeaIceColumn
        with pytest.warns(DeprecationWarning):
            col = SeaIceColumn.from_preset(FYI_WINTER_BARE)
        with pytest.warns(DeprecationWarning):
            result = col.compute_albedo(sza_deg=60)
        # result is now an Outputs object
        assert hasattr(result, "BBA")
        assert 0.4 <= result.BBA <= 0.9

    def test_seaicecolumn_direct_layers(self):
        from biosnicar.sea_ice.api import SeaIceColumn, SeaIceLayer
        with pytest.warns(DeprecationWarning):
            col = SeaIceColumn(layers=[
                SeaIceLayer(thickness_m=1.5, temperature_C=-10,
                            salinity_psu=8, density_kg_m3=910,
                            bubble_radius_um=200),
            ])
        with pytest.warns(DeprecationWarning):
            result = col.compute_albedo(sza_deg=55)
        assert 0.3 <= result.BBA <= 0.9

    def test_deprecated_sza_deg_param(self):
        from biosnicar.sea_ice.api import SeaIceColumn
        with pytest.warns(DeprecationWarning):
            col = SeaIceColumn.from_preset(FYI_WINTER_BARE)
        with pytest.warns(DeprecationWarning):
            # both sza_deg and compute_albedo itself should warn
            result = col.compute_albedo(sza_deg=60)
        assert result.BBA > 0
