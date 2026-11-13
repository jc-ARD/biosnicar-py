"""A clear sky is a direct beam plus scattered light (biosnicar.sky)."""

import pickle
import warnings

import numpy as np
import pytest

from biosnicar import sky
from biosnicar.drivers.run_model import run_model
from biosnicar.sea_ice.emulator_configs import (
    SEA_ICE_EMULATOR_CONFIGS, load_sea_ice_emulators,
)

WVL = sky.WAVELENGTHS_UM
SOLAR = (WVL >= 0.3) & (WVL <= 2.5)
VIS = (WVL >= 0.4) & (WVL <= 1.0)


def _at(um):
    return int(np.argmin(np.abs(WVL - um)))


def _mid(type_name, **over):
    p = {k: 0.5 * (lo + hi)
         for k, (lo, hi) in SEA_ICE_EMULATOR_CONFIGS[type_name]["params"].items()}
    p.update(solzen=60, direct=1)
    p.update(over)
    return p


def _run(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(run_model(**kw).albedo, dtype=float)


# ── the diffuse fraction ─────────────────────────────────────────────────

class TestDiffuseFraction:
    def test_is_rayleigh_like(self):
        f = sky.clear_sky_diffuse_fraction(60)
        assert f.shape == (480,)
        assert 0.3 < f[_at(0.4)] < 0.6          # ~0.4 in the blue at 60 deg
        assert 0.15 < f[_at(0.5)] < 0.3
        assert f[_at(0.4)] > f[_at(0.8)] > f[_at(1.6)]
        assert f[_at(1.6)] < 0.03
        assert ((f >= 0) & (f <= 1)).all()

    def test_grows_as_the_sun_drops(self):
        f = [sky.clear_sky_diffuse_fraction(z)[_at(0.5)] for z in (20, 50, 70, 85)]
        assert f == sorted(f)

    def test_grows_with_aerosol_and_over_a_bright_surface(self):
        base = sky.clear_sky_diffuse_fraction(60)
        hazy = sky.clear_sky_diffuse_fraction(60, aod_500=0.2)
        snow = sky.clear_sky_diffuse_fraction(60, regional_albedo=0.9)
        assert (hazy >= base).all() and hazy[_at(0.5)] > base[_at(0.5)] + 0.05
        assert (snow >= base).all() and snow[_at(0.4)] > base[_at(0.4)] + 0.05
        assert np.allclose(
            sky.clear_sky_diffuse_fraction(60, regional_albedo=0.0), base)

    def test_vectorised_over_zenith(self):
        z = np.array([30.0, 60.0, 75.0])
        f = sky.clear_sky_diffuse_fraction(z)
        assert f.shape == (3, 480)
        for i, zi in enumerate(z):
            assert np.allclose(f[i], sky.clear_sky_diffuse_fraction(zi))

    def test_argument_forms(self):
        assert sky.resolve_diffuse_fraction(None, 60) is None
        # "clear_sky" assumes snow-covered surroundings unless told otherwise
        assert np.allclose(
            sky.resolve_diffuse_fraction("clear_sky", 60),
            sky.clear_sky_diffuse_fraction(
                60, regional_albedo=sky.DEFAULT_REGIONAL_ALBEDO))
        assert np.allclose(
            sky.resolve_diffuse_fraction("clear_sky", 60, regional_albedo=None),
            sky.clear_sky_diffuse_fraction(60))
        assert float(sky.resolve_diffuse_fraction(0.3, 60)) == 0.3
        assert float(sky.resolve_diffuse_fraction(1.7, 60)) == 1.0     # clipped
        arr = np.linspace(0, 1, 480)
        assert np.allclose(sky.resolve_diffuse_fraction(arr, 60), arr)
        with pytest.raises(ValueError, match="diffuse_fraction"):
            sky.resolve_diffuse_fraction("overcast", 60)


# ── in the forward model ─────────────────────────────────────────────────

class TestForwardModel:
    GRANULAR = dict(layer_type=[0, 0], dz=[0.1, 1.0], rds=[300, 800],
                    rho=[300, 500], solzen=70)

    def test_default_is_still_a_pure_beam(self):
        """No diffuse_fraction: nothing changes."""
        kw = SEA_ICE_EMULATOR_CONFIGS["FYI_pond"]["transform_fn"](_mid("FYI_pond"))
        assert np.array_equal(_run(**kw), _run(diffuse_fraction=None, **kw))

    @pytest.mark.parametrize("solver", ["adding-doubling", "toon"])
    def test_mixed_sky_is_the_blend_of_beam_and_diffuse(self, solver):
        """Radiative transfer is linear in the incident field, so splitting
        the incident flux inside the solver must equal blending two runs."""
        kw = dict(self.GRANULAR, solver=solver)
        a_dir = _run(direct=1, **kw)
        a_dif = _run(direct=0, **kw)
        mixed = _run(direct=1, diffuse_fraction="clear_sky", **kw)
        f = sky.resolve_diffuse_fraction("clear_sky", 70)
        assert np.allclose(mixed, (1 - f) * a_dir + f * a_dif, atol=1e-4)
        assert np.abs(mixed - a_dir)[VIS].max() > 1e-3

    @pytest.mark.parametrize("type_name", ["FYI_bare", "FYI_pond", "young_ice"])
    def test_blend_holds_with_a_refractive_surface_at_low_sun(self, type_name):
        """Including the bands where the beam is totally reflected, which the
        solver's smoothing has to keep its hands off."""
        kw = SEA_ICE_EMULATOR_CONFIGS[type_name]["transform_fn"](
            _mid(type_name, solzen=78))
        a_dir = _run(**kw)
        a_dif = _run(**{**kw, "direct": 0})
        mixed = _run(diffuse_fraction="clear_sky", **kw)
        f = sky.resolve_diffuse_fraction("clear_sky", 78)
        assert np.allclose(mixed, (1 - f) * a_dir + f * a_dif, atol=2e-4)

    def test_matters_most_for_specular_surfaces_at_low_sun(self):
        """At 78 degrees the beam's Fresnel reflectance off water is several
        times the diffuse value, and most blue light is diffuse."""
        tf = SEA_ICE_EMULATOR_CONFIGS["FYI_pond"]["transform_fn"]

        def effect(solzen):
            kw = tf(_mid("FYI_pond", solzen=solzen))
            return np.abs(_run(diffuse_fraction="clear_sky", **kw)
                          - _run(**kw))[VIS].mean()

        assert effect(78) > 0.03
        assert effect(78) > 5 * effect(50)

    def test_regional_albedo_reaches_the_solver(self):
        kw = dict(self.GRANULAR, direct=1, diffuse_fraction="clear_sky")
        bright = _run(**kw)                               # default 0.8
        dark = _run(regional_albedo=None, **kw)
        a_dir, a_dif = _run(**self.GRANULAR, direct=1), _run(**self.GRANULAR, direct=0)
        f = sky.clear_sky_diffuse_fraction(70)
        assert np.allclose(dark, (1 - f) * a_dir + f * a_dif, atol=1e-4)
        # more diffuse light over a bright surface: further from the beam
        assert np.abs(bright - a_dir)[VIS].mean() > np.abs(dark - a_dir)[VIS].mean()

    def test_limits_and_overcast(self):
        kw = dict(self.GRANULAR)
        a_dir, a_dif = _run(direct=1, **kw), _run(direct=0, **kw)
        assert np.allclose(_run(direct=1, diffuse_fraction=0.0, **kw), a_dir,
                           atol=1e-6)
        assert np.allclose(_run(direct=1, diffuse_fraction=1.0, **kw)[SOLAR],
                           a_dif[SOLAR], atol=1e-4)
        # overcast is already all diffuse: the argument is ignored
        assert np.array_equal(
            _run(direct=0, diffuse_fraction="clear_sky", **kw), a_dif)

    def test_broadband_albedo_uses_the_clear_sky_spectrum(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            beam = run_model(direct=1, **self.GRANULAR)
            mixed = run_model(direct=1, diffuse_fraction="clear_sky",
                              **self.GRANULAR)
        assert np.array_equal(beam.flx_slr, mixed.flx_slr)
        # (the solver integrates before its final smoothing, hence the tolerance)
        assert mixed.BBA == pytest.approx(
            np.sum(mixed.flx_slr * mixed.albedo) / np.sum(mixed.flx_slr),
            rel=1e-3)
        assert mixed.BBA != beam.BBA


# ── on an emulator ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fleet():
    return load_sea_ice_emulators()


class TestMixedSkyEmulator:
    def test_predict_is_the_blend(self, fleet):
        emu = fleet["FYI_pond"]
        p = {k: _mid("FYI_pond", solzen=70)[k] for k in emu.param_names}
        f = sky.resolve_diffuse_fraction("clear_sky", 70)
        want = ((1 - f) * emu.predict(**p)
                + f * emu.predict(**{**p, "direct": 0}))
        assert np.allclose(sky.mixed_sky(emu).predict(**p), want)

    def test_overcast_is_passed_through(self, fleet):
        emu = fleet["FYI_pond"]
        p = {k: _mid("FYI_pond", direct=0)[k] for k in emu.param_names}
        assert np.array_equal(sky.mixed_sky(emu).predict(**p), emu.predict(**p))

    def test_batch_matches_single_with_per_row_geometry(self, fleet):
        emu = fleet["FYI_bare"]
        m = sky.mixed_sky(emu)
        rows = [_mid("FYI_bare", solzen=z, direct=d)
                for z, d in ((30, 1), (65, 1), (78, 1), (60, 0))]
        pts = np.array([[r[k] for k in emu.param_names] for r in rows])
        got = m.predict_batch(pts)
        for i, r in enumerate(rows):
            want = m.predict(**{k: r[k] for k in emu.param_names})
            assert np.allclose(got[i], want, atol=1e-10)

    def test_agrees_with_the_forward_model(self, fleet):
        """Emulator blend against run_model's native mixed sky: as close as
        the emulator is to the model anyway."""
        for name in ("FYI_snow", "FYI_pond"):
            emu, cfg = fleet[name], SEA_ICE_EMULATOR_CONFIGS[name]
            p = _mid(name, solzen=72)
            truth = _run(diffuse_fraction="clear_sky", **cfg["transform_fn"](p))
            pred = sky.mixed_sky(emu).predict(
                **{k: p[k] for k in emu.param_names})
            beam = emu.predict(**{k: p[k] for k in emu.param_names})
            assert np.abs(pred - truth)[VIS].max() < 0.02
            assert np.abs(pred - truth)[VIS].mean() < \
                0.5 * np.abs(beam - truth)[VIS].mean()

    def test_wrapping(self, fleet):
        emu = fleet["FYI_bare"]
        m = sky.mixed_sky(emu)
        assert sky.mixed_sky(m) is m                         # idempotent
        assert sky.mixed_sky(emu, None) is emu               # off
        assert m.base is emu
        assert m.param_names == emu.param_names and m.bounds == emu.bounds
        assert np.array_equal(m.solar_flux(60, 1), emu.solar_flux(60, 1))
        back = pickle.loads(pickle.dumps(m))
        p = {k: _mid("FYI_bare")[k] for k in emu.param_names}
        assert np.allclose(back.predict(**p), m.predict(**p))

    def test_analytical_model(self, fleet):
        water = fleet["open_water"]
        m = sky.mixed_sky(water)
        assert not hasattr(m, "predict_batch")
        p = dict(solzen=75, wind_speed_ms=5.0)
        f = sky.resolve_diffuse_fraction("clear_sky", 75)
        want = ((1 - f) * water.predict(direct=1, **p)
                + f * water.predict(direct=0, **p))
        assert np.allclose(m.predict(direct=1, **p), want)

    def test_measured_fraction(self, fleet):
        emu = fleet["FYI_bare"]
        p = {k: _mid("FYI_bare", solzen=70)[k] for k in emu.param_names}
        a1, a0 = emu.predict(**p), emu.predict(**{**p, "direct": 0})
        assert np.allclose(sky.mixed_sky(emu, 0.25).predict(**p),
                           0.75 * a1 + 0.25 * a0)


# ── in the retrievals ────────────────────────────────────────────────────

class TestRetrieval:
    @pytest.fixture(scope="class")
    def low_sun_pond(self):
        """A pond under a real clear sky with the sun 15 degrees up."""
        truth = dict(pond_depth=0.20, sea_ice_temperature=-2.0,
                     sea_ice_bubble_radius=600.0, black_carbon=50.0,
                     solzen=75, direct=1)
        kw = SEA_ICE_EMULATOR_CONFIGS["FYI_pond"]["transform_fn"](truth)
        return truth, _run(diffuse_fraction="clear_sky", **kw)

    def _retrieve(self, y, **kw):
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return retrieve_sea_ice(
                observed=y, surface_types=["FYI_pond"], solzen=75, direct=1,
                use_priors=False, wavelength_mask=(WVL >= 0.4) & (WVL <= 2.5),
                **kw)

    def test_default_is_the_clear_sky_model(self, low_sun_pond):
        _, y = low_sun_pond
        a = self._retrieve(y)
        b = self._retrieve(y, diffuse_fraction="clear_sky")
        assert a.parameters == b.parameters

    def test_fits_a_real_sky_better_than_a_pure_beam_does(self, low_sun_pond):
        truth, y = low_sun_pond
        mask = (WVL >= 0.4) & (WVL <= 1.0)

        def rms(r):
            return np.sqrt(np.mean((r.predicted_albedo - y)[mask] ** 2))

        mixed = self._retrieve(y)
        beam = self._retrieve(y, diffuse_fraction=None)
        assert rms(mixed) < 0.5 * rms(beam)
        assert rms(mixed) < 0.01

    def test_overcast_is_unaffected(self, fleet):
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice
        emu = fleet["FYI_snow"]
        y = emu.predict(**{k: _mid("FYI_snow", direct=0)[k]
                           for k in emu.param_names})
        kw = dict(observed=y, surface_types=["FYI_snow"], solzen=60, direct=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = retrieve_sea_ice(**kw)
            b = retrieve_sea_ice(diffuse_fraction=None, **kw)
        assert a.parameters == b.parameters

    def test_both_engines_use_it(self, low_sun_pond):
        from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch
        _, y = low_sun_pond
        Y = np.tile(y, (2, 1))
        kw = dict(surface_types=["FYI_pond"], solzen=75, direct=1,
                  use_priors=False, method="oe",
                  wavelength_mask=(WVL >= 0.4) & (WVL <= 1.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = {}
            for engine in ("loop", "vectorized"):
                for spec in ("clear_sky", None):
                    ds = retrieve_sea_ice_batch(
                        Y, engine=engine, diffuse_fraction=spec, **kw).to_xarray()
                    out[engine, spec] = float(ds["param_pond_depth"].values.ravel()[0])
        for engine in ("loop", "vectorized"):
            assert abs(out[engine, "clear_sky"] - 0.20) < \
                abs(out[engine, None] - 0.20)
        assert out["loop", "clear_sky"] == pytest.approx(
            out["vectorized", "clear_sky"], rel=0.1)
