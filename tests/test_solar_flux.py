"""The flux that weights a spectrum must be that of the illumination described.

An emulator used to carry one solar-flux spectrum — that of its first training
sample — and every broadband albedo and satellite band derived from it was
weighted by that sample's sun angle and sky, whatever geometry was asked for.
"""

from pathlib import Path

import numpy as np
import pytest

from biosnicar.classes.illumination import solar_flux
from biosnicar.drivers.run_model import run_model
from biosnicar.sea_ice.emulator_configs import (
    SEA_ICE_EMULATOR_CONFIGS, load_sea_ice_emulators, trained_emulator_names,
)

_BUILT = all(Path(SEA_ICE_EMULATOR_CONFIGS[n]["emulator_file"]).exists()
             for n in trained_emulator_names())
needs_fleet = pytest.mark.skipif(not _BUILT, reason="pre-built emulators absent")


class TestSolarFluxFunction:
    @pytest.mark.parametrize("solzen,direct", [(30, 1), (60, 1), (75, 1), (60, 0)])
    def test_matches_the_forward_model(self, solzen, direct):
        out = run_model(solzen=solzen, direct=direct)
        assert np.array_equal(solar_flux(solzen, direct), out.flx_slr)

    def test_float_solzen_rounds_to_the_tabulated_degree(self):
        assert np.array_equal(solar_flux(59.6, 1), solar_flux(60, 1))

    def test_diffuse_sky_ignores_solzen(self):
        assert np.array_equal(solar_flux(20, 0), solar_flux(80, 0))

    def test_clear_and_overcast_differ(self):
        assert not np.allclose(solar_flux(60, 1), solar_flux(60, 0))

    def test_bad_atmosphere_raises(self):
        with pytest.raises(ValueError, match="Irradiance type"):
            solar_flux(60, 1, incoming=9)


@needs_fleet
class TestEmulatorSolarFlux:
    @pytest.fixture(scope="class")
    def fleet(self):
        return load_sea_ice_emulators()

    def test_every_fleet_member_gives_the_same_flux_for_a_geometry(self, fleet):
        ref = solar_flux(60, 1)
        for name, emu in fleet.items():
            assert np.array_equal(emu.solar_flux(60, 1), ref), name

    def test_stored_fluxes_are_not_a_common_geometry(self, fleet):
        """Documents the defect: the stored spectra disagree with each other."""
        stored = [np.asarray(e.flx_slr) for e in fleet.values()]
        assert any(not np.allclose(stored[0], s) for s in stored[1:])

    def test_falls_back_to_stored_flux_without_geometry(self, fleet):
        emu = fleet["FYI_pond"]
        assert np.array_equal(emu.solar_flux(), emu.flx_slr)
        assert np.array_equal(emu.solar_flux(solzen=60), emu.flx_slr)

    def test_out_of_table_solzen_is_clipped(self, fleet):
        assert np.array_equal(fleet["FYI_pond"].solar_flux(95, 1),
                              solar_flux(89, 1))

    @pytest.mark.parametrize("direct", [0, 1])
    def test_run_emulator_broadband_matches_the_forward_model(self, fleet, direct):
        from biosnicar.drivers.run_emulator import run_emulator
        p = dict(pond_depth=0.15, sea_ice_temperature=-4.0,
                 sea_ice_bubble_radius=500.0, black_carbon=300.0,
                 solzen=70, direct=direct)
        truth = run_model(**SEA_ICE_EMULATOR_CONFIGS["FYI_pond"]["transform_fn"](p))
        out = run_emulator(fleet["FYI_pond"], **p)
        assert np.array_equal(out.flx_slr, truth.flx_slr)
        # Emulator error only — with the stored (overcast) flux the clear-sky
        # case was off by several hundredths.
        assert out.BBA == pytest.approx(truth.BBA, abs=0.01)


@needs_fleet
class TestRetrievalUsesTheRetrievalGeometry:
    @pytest.fixture(scope="class")
    def fleet(self):
        return load_sea_ice_emulators()

    @staticmethod
    def _obs(solzen, direct):
        p = dict(pond_depth=0.20, sea_ice_temperature=-4.0,
                 sea_ice_bubble_radius=400.0, black_carbon=200.0,
                 solzen=solzen, direct=direct)
        return run_model(
            **SEA_ICE_EMULATOR_CONFIGS["FYI_pond"]["transform_fn"](p))

    def test_result_flux_and_broadband(self, fleet):
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        truth = self._obs(70, 1)
        r = retrieve_sea_ice(observed=np.asarray(truth.albedo), emulators=fleet,
                             solzen=70, direct=1, known_month=7)
        assert np.array_equal(r.flx_slr, solar_flux(70, 1))
        for fit in r.all_fits.values():
            assert np.array_equal(fit.flx_slr, solar_flux(70, 1))
        assert r.to_outputs().BBA == pytest.approx(truth.BBA, abs=0.015)

    def test_vectorized_band_mode_per_pixel_geometry_matches_loop(self, fleet):
        from biosnicar.bands import to_platform
        from biosnicar.sea_ice.retrieve import (
            retrieve_sea_ice, retrieve_sea_ice_batch,
        )
        bands = ["B2", "B3", "B4", "B8", "B8A"]
        geo = [(40, 1), (70, 1), (60, 0), (70, 1)]
        Y = []
        for sz, dr in geo:
            o = self._obs(sz, dr)
            br = to_platform(np.asarray(o.albedo), "sentinel2", flx_slr=o.flx_slr)
            Y.append([getattr(br, b) for b in bands])
        Y = np.array(Y)
        kw = dict(platform="sentinel2", observed_band_names=bands,
                  known_month=7, method="oe")
        scene = retrieve_sea_ice_batch(
            Y, emulators=fleet, engine="vectorized",
            solzen=np.array([g[0] for g in geo], float),
            direct=np.array([g[1] for g in geo]), **kw).to_xarray()
        for i, (sz, dr) in enumerate(geo):
            r = retrieve_sea_ice(observed=Y[i], emulators=fleet, solzen=sz,
                                 direct=dr, **kw)
            assert str(scene["surface_type"].values[i]) == r.surface_type
            assert float(scene["confidence"].values[i]) == pytest.approx(
                r.confidence, abs=2e-3)
