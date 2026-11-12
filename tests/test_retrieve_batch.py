"""Correctness gate for the vectorised OE fleet: retrieve_sea_ice_batch with
engine='vectorized' must reproduce the per-pixel loop path exactly."""

from pathlib import Path

import numpy as np
import pytest

from biosnicar.sea_ice.emulator_configs import (
    SEA_ICE_EMULATOR_CONFIGS, load_sea_ice_emulators, trained_emulator_names,
)

pytest.importorskip("xarray")

_BUILT = all(Path(SEA_ICE_EMULATOR_CONFIGS[n]["emulator_file"]).exists()
             for n in trained_emulator_names())

pytestmark = pytest.mark.skipif(not _BUILT, reason="pre-built emulators absent")


def _obs(stype, seed):
    from biosnicar.drivers.run_model import run_model
    cfg = SEA_ICE_EMULATOR_CONFIGS[stype]
    p = {k: (lo + hi) / 2 for k, (lo, hi) in cfg["params"].items()}
    p["solzen"], p["direct"] = 60, 1
    a = np.asarray(run_model(**cfg["transform_fn"](p)).albedo)
    return np.clip(a + np.random.default_rng(seed).normal(0, 0.004, 480), 0, 1)


@pytest.fixture(scope="module")
def fleet():
    return load_sea_ice_emulators()


@pytest.fixture(scope="module")
def scene_obs():
    return np.array([_obs("FYI_pond", i) for i in range(3)]
                    + [_obs("FYI_bare", 10 + i) for i in range(3)])


def _loop(obs, fleet, **kw):
    from biosnicar.sea_ice.retrieve import retrieve_sea_ice
    return [retrieve_sea_ice(observed=o, emulators=fleet, **kw) for o in obs]


def test_vectorized_matches_loop_spectral(fleet, scene_obs):
    from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch
    kw = dict(solzen=60, known_month=7, method="oe", model_error=True)
    scene = retrieve_sea_ice_batch(scene_obs, emulators=fleet,
                                   engine="vectorized", **kw)
    loop = _loop(scene_obs, fleet, **kw)
    ds = scene.to_xarray()

    assert list(ds["surface_type"].values) == [r.surface_type for r in loop]
    for i, r in enumerate(loop):
        assert float(ds["confidence"].values[i]) == pytest.approx(
            r.confidence, abs=2e-3)
        assert float(ds["dfs"].values[i]) == pytest.approx(r.dfs, abs=5e-3)
        for p, v in r.parameters.items():
            var = ds.get(f"param_{p}")
            if var is not None and np.isfinite(v):
                assert float(var.values[i]) == pytest.approx(v, rel=2e-3,
                                                             abs=2e-3)


def test_vectorized_matches_loop_band_mode(fleet, scene_obs):
    from biosnicar.bands import to_platform
    from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch
    bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A"]
    flx = fleet["FYI_pond"].solar_flux(60, 1)
    Y = np.array([[getattr(to_platform(o, "sentinel2", flx_slr=flx), b)
                   for b in bands] for o in scene_obs])
    kw = dict(platform="sentinel2", observed_band_names=bands,
              solzen=60, known_month=7, method="oe")
    scene = retrieve_sea_ice_batch(Y, emulators=fleet, engine="vectorized", **kw)
    loop = [__import__("biosnicar.sea_ice.retrieve", fromlist=["retrieve_sea_ice"])
            .retrieve_sea_ice(observed=Y[i], emulators=fleet, **kw)
            for i in range(len(Y))]
    ds = scene.to_xarray()
    vec_types = [str(v) for v in ds["surface_type"].values]
    loop_types = [r.surface_type for r in loop]

    # The engines used to disagree here by up to 9.8e-2 in class probability,
    # and this test tolerated it as "weakly constrained near-ties".  It was
    # two bugs in the vectorised path, both fixed: it skipped the platform
    # SNR default for obs_uncertainty (flat 0.02 instead of the per-band
    # table, so a different S_e), and it convolved every candidate with the
    # *first* emulator's solar flux instead of each emulator's own.  With the
    # same S_e and the same band operator the two engines solve the same
    # problem, so hold band mode to the spectral-mode tolerance.
    assert vec_types == loop_types
    for i, r in enumerate(loop):
        assert float(ds["confidence"].values[i]) == pytest.approx(
            r.confidence, abs=2e-3)
        for name, p_loop in r.class_probabilities.items():
            assert float(ds[f"prob_{name}"].values[i]) == pytest.approx(
                p_loop, abs=2e-3), (i, name)
        assert float(ds["dfs"].values[i]) == pytest.approx(r.dfs, abs=5e-3)


# Chunking cannot change the *mathematics* — pixels are independent — but it
# does change the shapes handed to the batched linear algebra (n_jobs=2 solves
# two 3-pixel chunks where the single-process path solves one of 6), and a
# different BLAS kernel accumulates in a different order.  Scene fields are
# then stored as float32.  So exact bit-identity is not available, and this
# used to assert atol=1e-9 on float32 — finer than float32 can represent, the
# ULP at a cost of 25.9 being 1.9e-6.  It passed by luck until the 2026-11-06
# retrain nudged one pixel onto the neighbouring ULP.
#
# Measured divergence splits along a physically meaningful line:
#
#   point estimates (param_*, cost)   <= 1.5e-6 relative — float32 ULP
#   covariance-derived (unc_*, dfs)   <= 1.0e-4 relative — ~100x worse
#
# Both of the latter come through inv(H), and H is ill-conditioned because of
# the sea-ice parameter degeneracies (bubble radius / density posterior
# correlation is 0.97-0.98, giving a condition number of order (1+r)/(1-r)).
# So the retrieval is stable and only its error bars amplify round-off.
# Tolerances below are set from those measurements with headroom, not tuned
# until green.
_POINT_RTOL, _POINT_ATOL = 1e-5, 1e-6     # measured max 1.5e-6
_COV_RTOL, _COV_ATOL = 1e-3, 1e-8         # measured max 1.0e-4


def _assert_scene_identical(a, b):
    """Scene fields must match the single-process result to the precision the
    storage and the conditioning allow.

    Surface type is discrete and must match exactly — that is the part a real
    divergence would break.  See the note above for why the float fields are
    compared at two different tolerances.
    """
    assert list(a["surface_type"].values) == list(b["surface_type"].values)

    def _close(name, rtol, atol):
        np.testing.assert_allclose(
            a[name].values, b[name].values, rtol=rtol, atol=atol,
            equal_nan=True, err_msg=f"field {name!r} diverged",
        )

    for v in ("confidence", "cost"):
        if v in a and v in b:
            _close(v, _POINT_RTOL, _POINT_ATOL)
    if "dfs" in a and "dfs" in b:
        _close("dfs", _COV_RTOL, _COV_ATOL)
    for pv in [k for k in a.data_vars if k.startswith("param_")]:
        _close(pv, _POINT_RTOL, _POINT_ATOL)
    for pv in [k for k in a.data_vars if k.startswith(("unc_", "prob_"))]:
        _close(pv, _COV_RTOL, _COV_ATOL)


def test_vectorized_parallel_matches_single(fleet, scene_obs):
    """vectorized_parallel=True matches the single-process vectorised path to
    float32 precision — it only fans the same solve out over pixel chunks.

    Not bit-identical: chunking changes the batched-linear-algebra shapes and
    hence the accumulation order.  See _assert_scene_identical.
    """
    from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch
    kw = dict(solzen=60, known_month=7, method="oe", model_error=True)
    single = retrieve_sea_ice_batch(scene_obs, emulators=fleet, engine="vectorized",
                                    vectorized_parallel=False, **kw).to_xarray()
    par = retrieve_sea_ice_batch(scene_obs, emulators=fleet, engine="vectorized",
                                 vectorized_parallel=True, n_jobs=2, **kw).to_xarray()
    _assert_scene_identical(par, single)


def test_vectorized_parallel_per_pixel_geometry(fleet, scene_obs):
    """Per-pixel solzen/direct arrays are sliced to each chunk's pixel range and
    stay aligned — parallel result identical to single-process."""
    from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch
    n = len(scene_obs)
    # model_error=None: this test is about slicing alignment, and needs a
    # well-conditioned problem to see it at float32 resolution.  Under the
    # default field covariance one of these pixels sits on a degenerate
    # ridge where round-off picks between neighbouring minima — covered by
    # test_chunking_agrees_within_posterior_under_default_covariance.
    kw = dict(solzen=np.linspace(50, 70, n), direct=(np.arange(n) % 2).astype(int),
              known_month=7, method="oe", model_error=None)
    single = retrieve_sea_ice_batch(scene_obs, emulators=fleet, engine="vectorized",
                                    vectorized_parallel=False, **kw).to_xarray()
    par = retrieve_sea_ice_batch(scene_obs, emulators=fleet, engine="vectorized",
                                 vectorized_parallel=True, n_jobs=3, **kw).to_xarray()
    _assert_scene_identical(par, single)


def test_chunking_agrees_within_posterior_under_default_covariance(fleet, scene_obs):
    """Chunking cannot change the mathematics, but it changes the shapes
    handed to BLAS, and so the round-off.  Where the posterior is a narrow
    degenerate ridge (bubble radius / density, |r| ~ 0.98) the damped
    Gauss-Newton path is sensitive to that round-off and can stop at a
    neighbouring point of the ridge: measured 0.24% in cost and ~1% in the
    parameters on one MYI_bare pixel, far inside its own error bars.

    So the honest invariant under the default (field-calibrated, strongly
    correlated) S_e is statistical, not bitwise: same class, and every
    parameter within a small fraction of its posterior sigma.
    """
    from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch
    n = len(scene_obs)
    kw = dict(solzen=np.linspace(50, 70, n), direct=(np.arange(n) % 2).astype(int),
              known_month=7, method="oe")
    single = retrieve_sea_ice_batch(scene_obs, emulators=fleet, engine="vectorized",
                                    vectorized_parallel=False, **kw).to_xarray()
    par = retrieve_sea_ice_batch(scene_obs, emulators=fleet, engine="vectorized",
                                 vectorized_parallel=True, n_jobs=3, **kw).to_xarray()
    assert list(par["surface_type"].values) == list(single["surface_type"].values)
    for name in single.data_vars:
        if not name.startswith("param_"):
            continue
        unc = single["unc_" + name[len("param_"):]].values
        a, b = par[name].values, single[name].values
        ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(unc)
        assert (np.abs(a - b)[ok] <= 0.25 * unc[ok] + 1e-9).all(), name


def test_fleet_reloadable_guard():
    """Standard-named fleets reload by name in workers; anything else (or empty)
    falls back to pickling the passed objects."""
    from biosnicar.sea_ice.retrieve import _fleet_reloadable
    assert _fleet_reloadable({"FYI_bare": object(), "FYI_pond": object()})
    assert not _fleet_reloadable({"a_custom_emulator": object()})
    assert not _fleet_reloadable({})


def test_class_priors_match_loop(fleet, scene_obs):
    from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch
    kw = dict(solzen=60, known_month=7, method="oe",
              class_priors={"MYI_bare": 2.0, "FYI_bare": -2.0})
    scene = retrieve_sea_ice_batch(scene_obs, emulators=fleet,
                                   engine="vectorized", **kw)
    loop = _loop(scene_obs, fleet, **kw)
    assert list(scene.to_xarray()["surface_type"].values) == \
        [r.surface_type for r in loop]


def test_auto_uses_loop_for_non_oe(fleet, scene_obs):
    """engine='auto' with L-BFGS-B must not use the vectorised OE path."""
    from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch
    scene = retrieve_sea_ice_batch(scene_obs[:2], emulators=fleet,
                                   engine="auto", method="L-BFGS-B",
                                   solzen=60, known_month=7)
    assert scene.to_xarray().sizes["pixel"] == 2


def test_vectorized_rejects_incompatible(fleet, scene_obs):
    from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch
    with pytest.raises(ValueError, match="method='oe'"):
        retrieve_sea_ice_batch(scene_obs[:2], emulators=fleet,
                               engine="vectorized", method="L-BFGS-B",
                               solzen=60)


def test_nodata_pixels_preserved(fleet, scene_obs):
    from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch
    obs = scene_obs.copy()
    obs[2] = np.nan                      # a no-data pixel
    scene = retrieve_sea_ice_batch(obs, emulators=fleet, engine="vectorized",
                                   solzen=60, known_month=7, method="oe")
    types = scene.to_xarray()["surface_type"].values
    assert types[2] == "no_data"
    assert types[0] != "no_data"


def test_2d_scene_shape(fleet, scene_obs):
    from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch
    img = scene_obs.reshape(2, 3, 480)
    scene = retrieve_sea_ice_batch(img, emulators=fleet, engine="vectorized",
                                   solzen=60, known_month=7, method="oe")
    assert scene.to_xarray().sizes == {"y": 2, "x": 3}
