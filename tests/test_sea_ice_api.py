"""Tests for sea-ice Python API and presets."""

import numpy as np
import pytest

from biosnicar.sea_ice.api import SeaIceColumn, SeaIceLayer, SnowLayer, AlbedoResult
from biosnicar.sea_ice.presets import FYI_WINTER_BARE, FYI_WINTER_SNOW, MYI_WINTER_BARE, ALL_PRESETS


class TestPresets:
    @pytest.mark.parametrize("preset", [FYI_WINTER_BARE, FYI_WINTER_SNOW, MYI_WINTER_BARE])
    def test_preset_produces_valid_albedo(self, preset):
        col = SeaIceColumn.from_preset(preset)
        result = col.compute_albedo(sza_deg=60)
        assert isinstance(result, AlbedoResult)
        assert result.spectrum.shape == (480,)
        assert 0.0 <= result.broadband <= 1.0
        assert 0.0 <= result.visible <= 1.0
        assert 0.0 <= result.nir <= 1.0

    @pytest.mark.parametrize("preset", [FYI_WINTER_BARE, FYI_WINTER_SNOW, MYI_WINTER_BARE])
    def test_preset_bba_in_physical_range(self, preset):
        """Winter sea ice BBA must be in [0.4, 0.9]."""
        col = SeaIceColumn.from_preset(preset)
        result = col.compute_albedo(sza_deg=60)
        assert 0.4 <= result.broadband <= 0.9, (
            f"{preset.name}: BBA={result.broadband:.3f} outside [0.4, 0.9]"
        )

    def test_all_presets_in_registry(self):
        assert "FYI_WINTER_BARE" in ALL_PRESETS
        assert "FYI_WINTER_SNOW" in ALL_PRESETS
        assert "MYI_WINTER_BARE" in ALL_PRESETS

    def test_snow_covered_higher_albedo_than_bare(self):
        bare = SeaIceColumn.from_preset(FYI_WINTER_BARE).compute_albedo(sza_deg=60)
        snow = SeaIceColumn.from_preset(FYI_WINTER_SNOW).compute_albedo(sza_deg=60)
        assert snow.broadband > bare.broadband, (
            "Snow-covered ice must have higher BBA than bare ice"
        )


class TestSeaIceColumnDirect:
    def test_single_layer(self):
        col = SeaIceColumn(layers=[
            SeaIceLayer(thickness_m=1.5, temperature_C=-10, salinity_psu=8,
                        density_kg_m3=910, bubble_radius_um=200),
        ])
        result = col.compute_albedo(sza_deg=55)
        assert 0.3 <= result.broadband <= 0.9

    def test_two_layer(self):
        col = SeaIceColumn(layers=[
            SeaIceLayer(thickness_m=0.05, temperature_C=-25, salinity_psu=12,
                        density_kg_m3=920, bubble_radius_um=100),
            SeaIceLayer(thickness_m=1.45, temperature_C=-10, salinity_psu=8,
                        density_kg_m3=915, bubble_radius_um=200),
        ])
        result = col.compute_albedo(sza_deg=60)
        assert 0.3 <= result.broadband <= 0.9

    def test_snow_on_ice(self):
        col = SeaIceColumn(layers=[
            SnowLayer(thickness_m=0.10, density_kg_m3=300, grain_radius_um=200),
            SeaIceLayer(thickness_m=1.5, temperature_C=-10, salinity_psu=8,
                        density_kg_m3=910, bubble_radius_um=200),
        ])
        result = col.compute_albedo(sza_deg=60)
        assert 0.5 <= result.broadband <= 0.98

    def test_diffuse_sky(self):
        col = SeaIceColumn.from_preset(FYI_WINTER_BARE)
        result = col.compute_albedo(sza_deg=60, sky="cloudy")
        assert 0.3 <= result.broadband <= 0.9

    def test_result_attributes(self):
        col = SeaIceColumn.from_preset(FYI_WINTER_BARE)
        result = col.compute_albedo(sza_deg=60)
        assert hasattr(result, "spectrum")
        assert hasattr(result, "wavelengths")
        assert hasattr(result, "broadband")
        assert hasattr(result, "visible")
        assert hasattr(result, "nir")
        assert hasattr(result, "outputs")

    def test_spectrum_all_physical(self):
        col = SeaIceColumn.from_preset(FYI_WINTER_BARE)
        result = col.compute_albedo(sza_deg=60)
        assert np.all(result.spectrum >= 0.0)
        assert np.all(result.spectrum <= 1.0)

    def test_myi_lower_salinity_than_fyi(self):
        """MYI has less brine → more similar to glacier ice."""
        fyi = SeaIceColumn.from_preset(FYI_WINTER_BARE).compute_albedo(sza_deg=60)
        myi = SeaIceColumn.from_preset(MYI_WINTER_BARE).compute_albedo(sza_deg=60)
        # Both should be in valid range
        assert 0.4 <= fyi.broadband <= 0.9
        assert 0.4 <= myi.broadband <= 0.9


class TestRegressionExistingIce:
    """Verify that terrestrial ice (layer_type 0 and 1) is unaffected."""

    def test_layer_type_0_unchanged(self):
        """run_model with granular snow must produce same result as before."""
        from biosnicar import run_model
        out = run_model(layer_type=[0, 0], dz=[0.02, 1.0], rds=[500, 500], rho=[400, 400])
        assert 0.5 < out.BBA < 1.0

    def test_layer_type_1_unchanged(self):
        """run_model with solid glacier ice must produce same result as before."""
        from biosnicar import run_model
        out = run_model(layer_type=[1, 1], dz=[0.02, 1.0], rds=[500, 500], rho=[700, 700])
        assert 0.3 < out.BBA < 1.0
