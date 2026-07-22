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
    flx = fleet["FYI_pond"].flx_slr
    Y = np.array([[getattr(to_platform(o, "sentinel2", flx_slr=flx), b)
                   for b in bands] for o in scene_obs])
    kw = dict(platform="sentinel2", observed_band_names=bands,
              solzen=60, known_month=7, method="oe")
    scene = retrieve_sea_ice_batch(Y, emulators=fleet, engine="vectorized", **kw)
    loop = [__import__("biosnicar.sea_ice.retrieve", fromlist=["retrieve_sea_ice"])
            .retrieve_sea_ice(observed=Y[i], emulators=fleet, **kw)
            for i in range(len(Y))]
    assert list(scene.to_xarray()["surface_type"].values) == \
        [r.surface_type for r in loop]


def _assert_scene_identical(a, b):
    """Every field must match the single-process result exactly (pixels are
    independent, so chunk-then-concatenate changes nothing numerically)."""
    assert list(a["surface_type"].values) == list(b["surface_type"].values)
    for v in ("confidence", "cost", "dfs"):
        if v in a and v in b:
            np.testing.assert_allclose(a[v].values, b[v].values,
                                       rtol=0, atol=1e-9, equal_nan=True)
    for pv in [k for k in a.data_vars if k.startswith(("param_", "unc_", "prob_"))]:
        np.testing.assert_allclose(a[pv].values, b[pv].values,
                                   rtol=0, atol=1e-9, equal_nan=True)


def test_vectorized_parallel_matches_single(fleet, scene_obs):
    """vectorized_parallel=True is bit-identical to the single-process
    vectorised path — it only fans the same solve out over pixel chunks."""
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
    kw = dict(solzen=np.linspace(50, 70, n), direct=(np.arange(n) % 2).astype(int),
              known_month=7, method="oe")
    single = retrieve_sea_ice_batch(scene_obs, emulators=fleet, engine="vectorized",
                                    vectorized_parallel=False, **kw).to_xarray()
    par = retrieve_sea_ice_batch(scene_obs, emulators=fleet, engine="vectorized",
                                 vectorized_parallel=True, n_jobs=3, **kw).to_xarray()
    _assert_scene_identical(par, single)


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
