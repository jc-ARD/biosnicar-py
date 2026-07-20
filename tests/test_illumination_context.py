"""Tests for deriving retrieval geometry from scene metadata
(biosnicar.sea_ice.illumination_context) and feeding it per-pixel to the
vectorised batch retrieval."""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from biosnicar.sea_ice.illumination_context import (
    illumination_context, solar_zenith_angle,
)
from biosnicar.sea_ice.emulator_configs import (
    SEA_ICE_EMULATOR_CONFIGS, trained_emulator_names,
)

_BUILT = all(Path(SEA_ICE_EMULATOR_CONFIGS[n]["emulator_file"]).exists()
             for n in trained_emulator_names())


class TestSolarZenith:
    def test_equator_local_noon_is_small(self):
        # equinox, 0°N/0°E, 12:00 UTC → sun nearly overhead
        sza = solar_zenith_angle(0.0, 0.0, datetime(2020, 3, 20, 12))
        assert sza < 3.0

    def test_high_arctic_summer_is_grazing(self):
        # 85°N, solstice noon → sun low (SZA ~ 62°, i.e. 90-(23.45+~4))
        sza = solar_zenith_angle(85.0, 0.0, datetime(2020, 6, 21, 12))
        assert 55.0 < sza < 70.0

    def test_polar_night_sun_below_horizon(self):
        # 85°N, midwinter → sun well below the horizon (SZA > 90)
        sza = solar_zenith_angle(85.0, 0.0, datetime(2020, 12, 21, 12))
        assert sza > 95.0

    def test_vectorised_over_pixels(self):
        lat = np.array([70.0, 80.0, 88.0])
        lon = np.array([0.0, 30.0, -40.0])
        sza = solar_zenith_angle(lat, lon, datetime(2020, 7, 1, 10))
        assert sza.shape == (3,)
        assert np.all((sza >= 0) & (sza <= 180))

    def test_timezone_aware_normalised_to_utc(self):
        naive = solar_zenith_angle(75.0, 0.0, datetime(2020, 7, 1, 12))
        aware = solar_zenith_angle(
            75.0, 0.0, datetime(2020, 7, 1, 12, tzinfo=timezone.utc))
        assert naive == pytest.approx(aware, abs=1e-9)


class TestIlluminationContext:
    WHEN = datetime(2020, 7, 1, 12)

    def test_clear_daytime_is_direct(self):
        solzen, direct = illumination_context(75.0, 0.0, self.WHEN,
                                              cloud_fraction=0.1)
        assert direct == 1
        assert 20.0 <= solzen <= 89.0

    def test_cloudy_is_diffuse(self):
        _, direct = illumination_context(75.0, 0.0, self.WHEN,
                                         cloud_fraction=0.9)
        assert direct == 0

    def test_boolean_cloud_mask(self):
        _, d_clear = illumination_context(75.0, 0.0, self.WHEN, cloudy=False)
        _, d_cloud = illumination_context(75.0, 0.0, self.WHEN, cloudy=True)
        assert d_clear == 1 and d_cloud == 0

    def test_sun_down_forces_diffuse_even_if_clear(self):
        _, direct = illumination_context(85.0, 0.0, datetime(2020, 12, 21, 12),
                                         cloud_fraction=0.0)
        assert direct == 0                     # polar night → no direct beam

    def test_no_cloud_info_assumes_clear(self):
        _, direct = illumination_context(75.0, 0.0, self.WHEN)
        assert direct == 1

    def test_vectorised_scene(self):
        lat = np.full(4, 80.0)
        lon = np.zeros(4)
        cloud = np.array([0.0, 0.2, 0.7, 1.0])
        solzen, direct = illumination_context(lat, lon, self.WHEN,
                                              cloud_fraction=cloud)
        assert solzen.shape == (4,) and direct.shape == (4,)
        assert list(direct) == [1, 1, 0, 0]    # threshold 0.5

    def test_solzen_clamped_to_bounds(self):
        solzen, _ = illumination_context(89.5, 0.0, datetime(2020, 6, 21, 3),
                                         solzen_bounds=(20.0, 80.0))
        assert 20.0 <= solzen <= 80.0


@pytest.mark.skipif(not _BUILT, reason="pre-built emulators absent")
def test_per_pixel_geometry_through_batch():
    """A scene with per-pixel solzen/direct arrays retrieves via the
    vectorised engine; scalar and array geometry agree pixel-by-pixel."""
    import warnings
    from biosnicar.drivers.run_model import run_model
    from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
    from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch

    fleet = load_sea_ice_emulators()
    cfg = SEA_ICE_EMULATOR_CONFIGS["FYI_pond"]
    p = {k: (lo + hi) / 2 for k, (lo, hi) in cfg["params"].items()}
    p["solzen"], p["direct"] = 55, 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = np.asarray(run_model(**cfg["transform_fn"](p)).albedo)
    obs = np.clip(np.array([a, a]) + 0.0, 0, 1)      # two identical pixels

    solz = np.array([55.0, 55.0])
    dirc = np.array([1, 1])
    scene = retrieve_sea_ice_batch(obs, emulators=fleet, engine="vectorized",
                                   method="oe", solzen=solz, direct=dirc)
    scalar = retrieve_sea_ice_batch(obs, emulators=fleet, engine="vectorized",
                                    method="oe", solzen=55, direct=1)
    a_t = scene.to_xarray()["surface_type"].values
    b_t = scalar.to_xarray()["surface_type"].values
    assert list(a_t) == list(b_t)                    # per-pixel array == scalar


@pytest.mark.skipif(not _BUILT, reason="pre-built emulators absent")
def test_array_geometry_rejected_by_loop_engine():
    from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
    from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch
    fleet = load_sea_ice_emulators()
    obs = np.clip(np.random.default_rng(0).uniform(0.1, 0.9, (2, 480)), 0, 1)
    with pytest.raises(ValueError, match="per-pixel solzen/direct"):
        retrieve_sea_ice_batch(obs, emulators=fleet, engine="loop",
                               method="L-BFGS-B",
                               solzen=np.array([55.0, 60.0]), direct=1)
