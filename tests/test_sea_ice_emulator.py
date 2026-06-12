"""Tests for the sea ice emulator and retrieve_sea_ice() function."""

import numpy as np
import pytest

from biosnicar.emulator import Emulator
from biosnicar.inverse.result import RetrievalResult
from biosnicar.sea_ice.emulator_configs import SEA_ICE_EMULATOR_CONFIGS
from biosnicar.sea_ice.retrieve import SeaIceRetrievalResult, retrieve_sea_ice


# ── Shared tiny emulators (built once per session) ───────────────────────────

def _build_tiny(name, n=250, seed=0):
    cfg = SEA_ICE_EMULATOR_CONFIGS[name]
    return Emulator.build(
        params=cfg["params"],
        n_samples=n,
        transform_fn=cfg["transform_fn"],
        seed=seed,
        progress=False,
    )


@pytest.fixture(scope="module")
def emu_fyi_bare():
    return _build_tiny("FYI_bare")


@pytest.fixture(scope="module")
def emu_fyi_pond():
    return _build_tiny("FYI_pond")


@pytest.fixture(scope="module")
def emu_fyi_summer():
    return _build_tiny("FYI_summer")


# ── transform_fn mechanics ───────────────────────────────────────────────────

class TestTransformFn:
    def test_fyi_bare_transform_keys(self):
        fn = SEA_ICE_EMULATOR_CONFIGS["FYI_bare"]["transform_fn"]
        result = fn({
            "brine_volume_fraction": 0.033,
            "sea_ice_bubble_radius": 200.0,
            "black_carbon": 500.0,
            "rho_DL": 850.0,
            "solzen": 60,
            "direct": 1,
        })
        assert "layer_type" in result
        assert result["layer_type"] == [4, 4]
        assert result["rho"] == [850.0, 910]
        assert len(result["sea_ice_temperature"]) == 2

    def test_fyi_snow_transform_keys(self):
        fn = SEA_ICE_EMULATOR_CONFIGS["FYI_snow"]["transform_fn"]
        result = fn({
            "tau_snow": 1000.0 / 3.0,   # × 300 um grain = 0.10 m snow depth
            "snow_grain_radius": 300.0,
            "sea_ice_temperature": -15.0,
            "black_carbon": 0.0,
            "solzen": 60,
            "direct": 1,
        })
        assert result["layer_type"] == [0, 4, 4]
        assert result["dz"][0] == pytest.approx(0.10)
        assert result["rds"][0] == pytest.approx(300.0)

    def test_fyi_pond_transform_keys(self):
        fn = SEA_ICE_EMULATOR_CONFIGS["FYI_pond"]["transform_fn"]
        result = fn({
            "pond_depth": 0.20,
            "sea_ice_temperature": -5.0,
            "black_carbon": 1200.0,
            "solzen": 60,
            "direct": 1,
        })
        assert result["layer_type"] == [5, 4, 4]
        assert result["dz"][0] == pytest.approx(0.20)
        assert result["black_carbon"][0] == 0       # no BC in pond water
        assert result["black_carbon"][1] == pytest.approx(1200.0)  # BC in DL floor

    def test_transform_fn_stored_in_metadata(self, emu_fyi_bare):
        assert emu_fyi_bare._metadata.get("transform_fn") == "_transform_fyi_bare"


# ── Emulator build and predict ───────────────────────────────────────────────

class TestSeaIceEmulatorBuild:
    def test_param_names(self, emu_fyi_bare):
        cfg = SEA_ICE_EMULATOR_CONFIGS["FYI_bare"]
        assert set(emu_fyi_bare.param_names) == set(cfg["params"].keys())

    def test_training_score_is_float(self, emu_fyi_bare):
        # With 250 samples R² can be negative (underfitting is fine at test scale)
        assert isinstance(emu_fyi_bare.training_score, float)

    def test_pca_components_exist(self, emu_fyi_bare):
        assert emu_fyi_bare.n_pca_components >= 1

    def test_flx_slr_populated(self, emu_fyi_bare):
        assert emu_fyi_bare.flx_slr is not None
        assert emu_fyi_bare.flx_slr.shape == (480,)

    def test_pond_emulator_has_correct_params(self, emu_fyi_pond):
        expected = {"pond_depth", "sea_ice_temperature", "black_carbon", "solzen", "direct"}
        assert set(emu_fyi_pond.param_names) == expected


class TestSeaIceEmulatorPredict:
    def test_predict_shape(self, emu_fyi_bare):
        pred = emu_fyi_bare.predict(
            brine_volume_fraction=0.033, sea_ice_bubble_radius=200.0,
            black_carbon=0.0,
            rho_DL=850.0, solzen=60, direct=1,
        )
        assert pred.shape == (480,)

    def test_predict_physical_range(self, emu_fyi_bare):
        pred = emu_fyi_bare.predict(
            brine_volume_fraction=0.033, sea_ice_bubble_radius=200.0,
            black_carbon=0.0,
            rho_DL=850.0, solzen=60, direct=1,
        )
        assert np.all(pred >= 0.0)
        assert np.all(pred <= 1.0)

    def test_higher_bc_darkens(self, emu_fyi_bare):
        kw = dict(brine_volume_fraction=0.033, sea_ice_bubble_radius=200.0, rho_DL=850.0, solzen=60, direct=1)
        clean = emu_fyi_bare.predict(black_carbon=0.0, **kw)
        dirty = emu_fyi_bare.predict(black_carbon=3000.0, **kw)
        clean_bba = float(np.mean(clean))
        dirty_bba = float(np.mean(dirty))
        assert dirty_bba < clean_bba

    def test_deeper_pond_lower_nir(self, emu_fyi_pond):
        kw = dict(sea_ice_temperature=-5.0, black_carbon=1200.0, solzen=60, direct=1)
        shallow = emu_fyi_pond.predict(pond_depth=0.05, **kw)
        deep    = emu_fyi_pond.predict(pond_depth=0.40, **kw)
        # NIR (bands 50:) should be lower for deeper pond
        assert float(np.mean(deep[50:])) < float(np.mean(shallow[50:]))

    def test_pond_predict_shape(self, emu_fyi_pond):
        pred = emu_fyi_pond.predict(
            pond_depth=0.15, sea_ice_temperature=-5.0,
            black_carbon=1200.0, solzen=60, direct=1,
        )
        assert pred.shape == (480,)


# ── Save / load round-trip ───────────────────────────────────────────────────

class TestSeaIceEmulatorSaveLoad:
    def test_save_load_predictions_match(self, emu_fyi_bare, tmp_path):
        path = tmp_path / "fyi_bare_test.npz"
        emu_fyi_bare.save(path)
        loaded = Emulator.load(path)
        kw = dict(brine_volume_fraction=0.033, sea_ice_bubble_radius=300.0,
                  black_carbon=500.0,
                  rho_DL=850.0, solzen=60, direct=1)
        np.testing.assert_allclose(
            emu_fyi_bare.predict(**kw), loaded.predict(**kw), rtol=1e-5
        )

    def test_flx_slr_preserved(self, emu_fyi_bare, tmp_path):
        path = tmp_path / "fyi_bare_flx.npz"
        emu_fyi_bare.save(path)
        loaded = Emulator.load(path)
        np.testing.assert_array_equal(emu_fyi_bare.flx_slr, loaded.flx_slr)

    def test_transform_fn_name_in_metadata(self, emu_fyi_bare, tmp_path):
        path = tmp_path / "meta_test.npz"
        emu_fyi_bare.save(path)
        loaded = Emulator.load(path)
        assert loaded._metadata.get("transform_fn") == "_transform_fyi_bare"


# ── run_emulator wrapper ─────────────────────────────────────────────────────

class TestRunEmulatorWithSeaIce:
    def test_run_emulator_returns_outputs(self, emu_fyi_bare):
        from biosnicar.drivers.run_emulator import run_emulator
        from biosnicar.classes.outputs import Outputs
        out = run_emulator(
            emu_fyi_bare,
            brine_volume_fraction=0.033, sea_ice_bubble_radius=200.0,
            black_carbon=0.0,
            rho_DL=850.0, solzen=60, direct=1,
        )
        assert isinstance(out, Outputs)
        assert 0.0 <= out.BBA <= 1.0
        assert out.flx_slr is not None

    def test_to_platform_works(self, emu_fyi_bare):
        from biosnicar.drivers.run_emulator import run_emulator
        out = run_emulator(
            emu_fyi_bare,
            brine_volume_fraction=0.033, sea_ice_bubble_radius=200.0,
            black_carbon=0.0,
            rho_DL=850.0, solzen=60, direct=1,
        )
        bands = out.to_platform("sentinel2")
        assert hasattr(bands, "B3")
        assert 0.0 <= bands.B3 <= 1.0


# ── retrieve_sea_ice ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_emulator_fleet():
    """A minimal two-emulator fleet for fast retrieve_sea_ice tests."""
    return {
        "FYI_bare":  _build_tiny("FYI_bare",  n=300, seed=1),
        "FYI_pond":  _build_tiny("FYI_pond",  n=300, seed=2),
    }


class TestRetrieveSeaIce:
    def test_returns_sea_ice_result(self, tiny_emulator_fleet, emu_fyi_bare):
        obs = emu_fyi_bare.predict(
            brine_volume_fraction=0.033, sea_ice_bubble_radius=200.0, black_carbon=0.0, rho_DL=850.0,
            solzen=60, direct=1,
        )
        result = retrieve_sea_ice(
            observed=obs, emulators=tiny_emulator_fleet, solzen=60, direct=1,
        )
        assert isinstance(result, SeaIceRetrievalResult)

    def test_surface_type_is_known_name(self, tiny_emulator_fleet, emu_fyi_bare):
        obs = emu_fyi_bare.predict(
            brine_volume_fraction=0.033, sea_ice_bubble_radius=200.0, black_carbon=0.0, rho_DL=850.0,
            solzen=60, direct=1,
        )
        result = retrieve_sea_ice(
            observed=obs, emulators=tiny_emulator_fleet, solzen=60, direct=1,
        )
        assert result.surface_type in tiny_emulator_fleet

    def test_confidence_in_range(self, tiny_emulator_fleet, emu_fyi_bare):
        obs = emu_fyi_bare.predict(
            brine_volume_fraction=0.033, sea_ice_bubble_radius=200.0, black_carbon=0.0, rho_DL=850.0,
            solzen=60, direct=1,
        )
        result = retrieve_sea_ice(
            observed=obs, emulators=tiny_emulator_fleet, solzen=60, direct=1,
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_cost_per_type_populated(self, tiny_emulator_fleet, emu_fyi_bare):
        obs = emu_fyi_bare.predict(
            brine_volume_fraction=0.033, sea_ice_bubble_radius=200.0, black_carbon=0.0, rho_DL=850.0,
            solzen=60, direct=1,
        )
        result = retrieve_sea_ice(
            observed=obs, emulators=tiny_emulator_fleet, solzen=60, direct=1,
        )
        assert set(result.cost_per_type.keys()) == set(tiny_emulator_fleet.keys())
        assert result.cost == min(result.cost_per_type.values())

    def test_all_fits_populated(self, tiny_emulator_fleet, emu_fyi_bare):
        obs = emu_fyi_bare.predict(
            brine_volume_fraction=0.033, sea_ice_bubble_radius=200.0, black_carbon=0.0, rho_DL=850.0,
            solzen=60, direct=1,
        )
        result = retrieve_sea_ice(
            observed=obs, emulators=tiny_emulator_fleet, solzen=60, direct=1,
        )
        assert set(result.all_fits.keys()) == set(tiny_emulator_fleet.keys())
        for fit in result.all_fits.values():
            assert isinstance(fit, RetrievalResult)

    def test_to_outputs_works(self, tiny_emulator_fleet, emu_fyi_bare):
        from biosnicar.classes.outputs import Outputs
        obs = emu_fyi_bare.predict(
            brine_volume_fraction=0.033, sea_ice_bubble_radius=200.0, black_carbon=0.0, rho_DL=850.0,
            solzen=60, direct=1,
        )
        result = retrieve_sea_ice(
            observed=obs, emulators=tiny_emulator_fleet, solzen=60, direct=1,
        )
        out = result.to_outputs()
        assert isinstance(out, Outputs)
        assert out.flx_slr is not None

    def test_to_platform_works(self, tiny_emulator_fleet, emu_fyi_bare):
        obs = emu_fyi_bare.predict(
            brine_volume_fraction=0.033, sea_ice_bubble_radius=200.0, black_carbon=0.0, rho_DL=850.0,
            solzen=60, direct=1,
        )
        result = retrieve_sea_ice(
            observed=obs, emulators=tiny_emulator_fleet, solzen=60, direct=1,
        )
        bands = result.to_outputs().to_platform("sentinel2")
        assert hasattr(bands, "B3")
        assert 0.0 <= bands.B3 <= 1.0

    def test_pond_observation_favours_pond(self, tiny_emulator_fleet, emu_fyi_pond):
        # A clear pond spectrum should be classified as FYI_pond
        obs = emu_fyi_pond.predict(
            pond_depth=0.20, sea_ice_temperature=-5.0,
            black_carbon=1200.0, solzen=60, direct=1,
        )
        result = retrieve_sea_ice(
            observed=obs, emulators=tiny_emulator_fleet, solzen=60, direct=1,
        )
        assert result.surface_type == "FYI_pond"

    def test_summary_contains_surface_type(self, tiny_emulator_fleet, emu_fyi_bare):
        obs = emu_fyi_bare.predict(
            brine_volume_fraction=0.033, sea_ice_bubble_radius=200.0, black_carbon=0.0, rho_DL=850.0,
            solzen=60, direct=1,
        )
        result = retrieve_sea_ice(
            observed=obs, emulators=tiny_emulator_fleet, solzen=60, direct=1,
        )
        s = result.summary()
        assert result.surface_type in s
        assert "confidence" in s.lower()

    def test_surface_types_filter(self, tiny_emulator_fleet, emu_fyi_bare):
        obs = emu_fyi_bare.predict(
            brine_volume_fraction=0.033, sea_ice_bubble_radius=200.0, black_carbon=0.0, rho_DL=850.0,
            solzen=60, direct=1,
        )
        result = retrieve_sea_ice(
            observed=obs, emulators=tiny_emulator_fleet,
            surface_types=["FYI_bare"], solzen=60, direct=1,
        )
        assert result.surface_type == "FYI_bare"
        assert list(result.all_fits.keys()) == ["FYI_bare"]
