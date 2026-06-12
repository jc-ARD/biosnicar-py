"""Tests for the analytical open water albedo model (B1)."""

import numpy as np
import pytest

from biosnicar.sea_ice.emulator_configs import (
    SEA_ICE_EMULATOR_CONFIGS,
    load_sea_ice_emulators,
    trained_emulator_names,
)
from biosnicar.sea_ice.open_water import OpenWaterModel
from biosnicar.sea_ice.retrieve import retrieve_sea_ice


@pytest.fixture(scope="module")
def model():
    return OpenWaterModel()


def _bba(model, albedo):
    flx = model.flx_slr
    return float(albedo @ flx / flx.sum())


class TestOpenWaterPhysics:
    def test_bba_range_typical_solzen(self, model):
        # Typical Arctic open water at moderate sun: BBA 0.02-0.10.
        # The guide's blanket [0.03, 0.10] cannot hold at the SZA extremes:
        # flat-water Fresnel gives ~0.022 at 20 deg, ~0.13 at 70 deg and
        # ~0.3 at 80 deg (cf. Briegleb / Jin et al. parameterisations).
        for solzen in (20, 30, 40, 50, 60):
            for wind in (0, 5, 15):
                bba = _bba(model, model.predict(solzen=solzen, wind_speed_ms=wind))
                assert 0.015 < bba < 0.12, f"solzen={solzen} wind={wind} BBA={bba}"
        for wind in (0, 5, 15):
            assert _bba(model, model.predict(solzen=70, wind_speed_ms=wind)) < 0.16
            assert _bba(model, model.predict(solzen=80, wind_speed_ms=wind)) < 0.40

    def test_bba_increases_with_solzen(self, model):
        bbas = [_bba(model, model.predict(solzen=s, wind_speed_ms=5))
                for s in (20, 40, 60, 80)]
        assert all(b2 > b1 for b1, b2 in zip(bbas, bbas[1:]))

    def test_wind_reduces_albedo_at_high_solzen(self, model):
        calm = _bba(model, model.predict(solzen=80, wind_speed_ms=0))
        windy = _bba(model, model.predict(solzen=80, wind_speed_ms=15))
        assert windy < calm

    def test_monotonic_nir_decline(self, model):
        a = model.predict(solzen=60, wind_speed_ms=5)
        # No 550 nm peak (unlike shallow ponds); blue VIS >= NIR >= SWIR
        i450, i550, i850, i1600 = 24, 34, 64, 139
        assert a[i450] > a[i850] > a[i1600]
        assert a[i550] < a[i450]  # decreasing through the VIS, no green peak

    def test_diffuse_mode(self, model):
        a = model.predict(solzen=60, wind_speed_ms=5, direct=0)
        bba = _bba(model, a)
        assert 0.04 < bba < 0.10  # isotropic-sky Fresnel ~0.066
        # diffuse albedo is solzen-independent
        b = model.predict(solzen=30, wind_speed_ms=5, direct=0)
        np.testing.assert_allclose(a, b)

    def test_albedo_physical_range(self, model):
        for solzen in (20, 50, 80):
            a = model.predict(solzen=solzen, wind_speed_ms=3)
            assert a.shape == (480,)
            assert np.all(a >= 0) and np.all(a <= 1)


class TestEmulatorInterface:
    def test_param_names_and_bounds(self, model):
        assert model.param_names == ["solzen", "wind_speed_ms"]
        assert model.bounds["solzen"] == (20.0, 80.0)
        assert model.bounds["wind_speed_ms"] == (0.0, 15.0)

    def test_flx_slr_shape(self, model):
        assert model.flx_slr.shape == (480,)
        assert np.all(model.flx_slr > 0)

    def test_ignores_extra_kwargs(self, model):
        a = model.predict(solzen=60, wind_speed_ms=3, direct=1, black_carbon=0)
        assert a.shape == (480,)

    def test_predict_platform(self, model):
        vals = model.predict_platform(
            "sentinel2", ["B3", "B8", "B11"], solzen=60, wind_speed_ms=3,
        )
        assert vals.shape == (3,)
        assert np.all(vals >= 0) and np.all(vals <= 1)
        assert vals[0] > vals[2]  # B3 (green) > B11 (SWIR)

    def test_registered_in_configs(self):
        assert "open_water" in SEA_ICE_EMULATOR_CONFIGS
        assert "open_water" not in trained_emulator_names()

    def test_load_sea_ice_emulators_includes_open_water(self):
        fleet = load_sea_ice_emulators(["open_water"])
        assert isinstance(fleet["open_water"], OpenWaterModel)


BUILT = all(
    __import__("pathlib").Path(SEA_ICE_EMULATOR_CONFIGS[n]["emulator_file"]).exists()
    for n in trained_emulator_names()
)


@pytest.mark.skipif(not BUILT, reason="pre-built sea ice emulators not found")
class TestOpenWaterClassification:
    def test_synthetic_open_water_classifies(self, model):
        """Acceptance criterion: open water spectrum at solzen=60 classifies
        as open_water with confidence > 0.90."""
        rng = np.random.default_rng(7)
        obs = model.predict(solzen=60, wind_speed_ms=5.0)
        obs = np.clip(obs + rng.normal(0, 0.003, 480), 0, 1)
        assert 0.03 < _bba(model, obs) < 0.09
        result = retrieve_sea_ice(observed=obs, solzen=60, direct=1)
        assert result.surface_type == "open_water"
        assert result.confidence > 0.90
        assert abs(result.parameters["wind_speed_ms"] - 5.0) < 3.0

    def test_wmo_and_sigrid3_mapping(self, model):
        obs = model.predict(solzen=60, wind_speed_ms=3.0)
        result = retrieve_sea_ice(observed=obs, solzen=60, direct=1)
        if result.surface_type == "open_water":
            wmo = result.to_wmo()
            assert "Ice free" in wmo.stage_of_development
            sig = result.to_sigrid3()
            assert sig.ice_type_code == "OW"
            assert not sig.melt_pond_present

    def test_bright_ice_not_misclassified_as_water(self):
        from biosnicar.drivers.run_model import run_model

        obs = np.asarray(run_model(
            **SEA_ICE_EMULATOR_CONFIGS["FYI_snow"]["transform_fn"](dict(
                tau_snow=500, snow_grain_radius=300,
                sea_ice_temperature=-15, black_carbon=100, solzen=60, direct=1,
            ))
        ).albedo)
        result = retrieve_sea_ice(observed=obs, solzen=60, direct=1)
        assert result.surface_type != "open_water"
