#!/usr/bin/env python3
"""Surface scattering layer: grain pile or porous crust, and where is the interface?

The melt-season SSL is modelled as granular (``layer_type=0``: ice grains in
air) with the smooth Fresnel interface underneath it, on top of the drained
layer.  Two things about that are assumptions rather than findings:

* **Morphology.**  A weathering crust is arguably bulk ice shot through with
  large interconnected voids — coral rather than a grain pile — which is what
  ``layer_type`` 1/2 (air inclusions in solid ice) describe for glacier ice.
* **Interface placement.**  A crust has a solid upper surface; a grain pile
  does not.  The default rule puts the interface *below* the grains.

This fits the SHEBA summer bare-ice spectra (Grenfell & Light 2007,
400-1000 nm) with the forward model itself under six structures — two
morphologies by three interface placements — with the same five free
parameters each (SSL radius, SSL density, SSL thickness, drained-layer bubble
radius, SSL black carbon).

Caveat carried from the roadmap: ``bubbly_air.npz`` is spherical-inclusion
Mie, so the voids-in-ice variants get the topology right and the void
*shape* wrong.

Usage::

    uv run python scripts/experiments/ssl_morphology_experiment.py
"""
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "validation_data"))
warnings.simplefilter("ignore")

import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution

import biosnicar
from biosnicar.drivers.run_model import run_model
from biosnicar.sea_ice.emulator_configs import _snap
from biosnicar.sea_ice.spectral_utils import MODEL_WAVELENGTHS_NM as WL

VIS = (WL >= 400) & (WL <= 1000)
_BUBBLE_RADII = np.load(
    biosnicar.DATA_DIR / "OP_data" / "480band" / "luts" / "bubbly_air.npz"
)["radii"]

# name -> (SSL layer_type, fresnel_layer, radius bounds um, density bounds)
VARIANTS = {
    "grains, interface below (shipped)": (0, "auto", (100, 5000), (200, 500)),
    "grains, no interface":              (0, None,   (100, 5000), (200, 500)),
    "grains, interface on top":          (0, 0,      (100, 5000), (200, 500)),
    "voids,  interface on top":          (2, 0,      (50, 5000),  (500, 880)),
    "voids,  interface below":           (2, "auto", (50, 5000),  (500, 880)),
    "voids,  no interface":              (2, None,   (50, 5000),  (500, 880)),
}


def _snap_bubble(r):
    return int(_BUBBLE_RADII[np.argmin(np.abs(_BUBBLE_RADII - r))])


def forward(x, sza, lt, fresnel):
    r, rho, log_dz, bbl, log_bc = x
    rds = _snap(r) if lt == 0 else _snap_bubble(r)
    return np.asarray(run_model(
        layer_type=[lt, 4, 4],
        dz=[10 ** log_dz, 0.05, 1.40],
        rds=[rds, 500, 500],
        rho=[rho, 850, 910],
        sea_ice_salinity=[None, 4, 2],
        sea_ice_temperature=[None, -2.5, -2.5],
        sea_ice_bubble_radius=[None, bbl, min(2000.0, 2 * bbl)],
        black_carbon=[10 ** log_bc - 1, 0, 0],
        solzen=sza, direct=1, fresnel_layer=fresnel,
    ).albedo)


def fit(date, obs, sza, name):
    lt, fresnel, r_b, rho_b = VARIANTS[name]
    m = np.isfinite(obs) & VIS
    res = differential_evolution(
        lambda x: float(np.mean((forward(x, sza, lt, fresnel)[m] - obs[m]) ** 2)),
        [r_b, rho_b, (-2.0, -0.8), (50, 1000), (0, 3.3)],
        seed=1, maxiter=30, popsize=8, tol=1e-8, polish=False,
    )
    return date, name, float(np.sqrt(res.fun)), res.x


def main():
    from Grenfell_light_2007.validate_grenfell_light_2007 import (
        catalogue, parse_albv,
    )
    _, summer = catalogue()
    jobs = []
    for e in summer:
        wl, alb, _ = parse_albv(e["path"])
        if wl is None:
            continue
        obs = interp1d(wl, alb, bounds_error=False, fill_value=np.nan)(WL)
        for name in VARIANTS:
            jobs.append((e["date"], obs, float(e["sza"]), name))
    out = Parallel(n_jobs=-1)(delayed(fit)(*j) for j in jobs)

    dates = sorted({o[0] for o in out})
    print(f"{len(dates)} SHEBA summer bare-ice spectra, RMSE over 400-1000 nm\n")
    print(f"{'structure':36} {'mean':>7} {'median':>7} {'worst':>7}  "
          f"{'best on':>7}   median r / rho / dz_cm")
    best = {d: min((o for o in out if o[0] == d), key=lambda o: o[2])[1]
            for d in dates}
    for name in VARIANTS:
        rows = [o for o in out if o[1] == name]
        rm = np.array([o[2] for o in rows])
        X = np.array([o[3] for o in rows])
        wins = sum(best[d] == name for d in dates)
        print(f"{name:36} {rm.mean():7.4f} {np.median(rm):7.4f} {rm.max():7.4f}  "
              f"{wins:>4}/{len(dates):<2}   {np.median(X[:,0]):5.0f} / "
              f"{np.median(X[:,1]):4.0f} / {100 * 10 ** np.median(X[:,2]):4.1f}")


if __name__ == "__main__":
    main()
