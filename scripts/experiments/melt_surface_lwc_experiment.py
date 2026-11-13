#!/usr/bin/env python3
"""D2: liquid water in the melting surface layer — what the model can and cannot say.

The melting-surface forward models carry no liquid water: the surface layer
of FYI_summer is ``layer_type=0`` — dry ice grains in air.  A melting surface
that holds no water is physically wrong, and the development plan (D2)
proposes adding liquid-water content there.

The repository has two wet-grain representations.  They are not equivalent,
and at 5% water by volume on 1000 um grains they disagree on the *sign* of the
near-infrared response (SZA 60, sea-ice column below):

    wavelength um         0.45   0.86   1.06   1.26
    dry grains            0.873  0.699  0.424  0.162
    layer_type=3          0.878  0.703  0.434  0.166   brighter everywhere
    water-coated spheres  0.880  0.704  0.423  0.157   darker in the NIR

``layer_type=3`` intersperses *separate water spheres* among the ice grains.
Every water sphere is an extra scatterer, so adding water brightens the layer.
That is the wrong picture for pore water held on and between grains, which
adds absorbing path and merges grains optically — wet snow is darker in the
NIR, not brighter.  The coated-sphere model (``water`` override, Mie on an
ice core with a water shell) has the physically expected sign, but it is a
runtime Mie calculation at ~24 s per spectrum and cannot be used to train an
emulator without first building a (radius, water-fraction) lookup table.

So the available fast representation is not one to build a retrieved
parameter on, and this script exists to put a number on the other half of
the argument: how much of the melt-season misfit liquid water accounts for
*even granting* ``layer_type=3``.  Each Smith et al. (2021) MOSAiC spectrum
is fitted twice with the forward model itself (no emulator), 400-2400 nm:

  dry   layer_type=0  grains in air                   (r, depth, BC)
  wet   layer_type=3  ice + water spheres, same size  (r, depth, BC, LWC)

Bulk density rises with the water added (``rho = rho_dry + 1000 * LWC``) so
the ice matrix is held fixed and LWC means water added to the pores.  ``wet``
nests ``dry`` exactly — ``layer_type=3`` at LWC = 0 reproduces
``layer_type=0`` to 2e-13.

Measured 2026-11-12, 22 spectra: SWIR RMS 0.0295 -> 0.0278 (5.5%), retrieved
LWC median 0.05, never at a bound.  Read that as a statement about
*identifiability* — the spectrum barely constrains water content through this
representation — and not as evidence that the surface is dry.  Field fits
with free radius, depth and impurity loading cannot arbitrate a 0.01 albedo
effect.

The physically correct route for D2 is therefore a coated-sphere LUT, not
this.

Usage::

    uv run python scripts/experiments/melt_surface_lwc_experiment.py [step]

``step`` subsamples the 44 spectra (default 2).
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
from scipy.optimize import differential_evolution

from biosnicar.drivers.run_model import run_model
from biosnicar.sea_ice.emulator_configs import _snap
from biosnicar.sea_ice.spectral_utils import (
    MODEL_WAVELENGTHS_NM as WL, resample_to_model_grid,
)

VIS = (WL >= 400) & (WL < 1000)
SWIR = (WL >= 1000) & (WL <= 2400)


def forward(x, sza, direct, wet):
    r, log_dz, log_bc = x[0], x[1], x[2]
    lwc = x[3] if wet else 0.0
    return np.asarray(run_model(
        layer_type=[3 if wet else 0, 4, 4],
        dz=[10 ** log_dz, 0.05, 1.40],
        rds=[_snap(r), 500, 500],
        rho=[400 + 1000.0 * lwc, 850, 910],
        lwc=[lwc, 0, 0],
        sea_ice_salinity=[None, 4, 2],
        sea_ice_temperature=[None, -2.5, -2.5],
        sea_ice_bubble_radius=[None, 300, 600],
        black_carbon=[10 ** log_bc - 1, 0, 0],
        solzen=sza, direct=direct,
    ).albedo)


def fit_one(date, obs, sza, direct):
    m = np.isfinite(obs) & (WL >= 400) & (WL <= 2400)
    row = dict(date=date)
    for wet in (False, True):
        bounds = [(50, 3000), (-2.3, -0.3), (0, 3.3)]
        if wet:
            bounds.append((0.0, 0.25))
        res = differential_evolution(
            lambda x: float(np.mean((forward(x, sza, direct, wet)[m] - obs[m]) ** 2)),
            bounds, seed=1, maxiter=30, popsize=8, tol=1e-7, polish=False,
        )
        p = forward(res.x, sza, direct, wet)
        k = "wet" if wet else "dry"
        row[k + "_vis"] = float(np.sqrt(np.mean((p[m & VIS] - obs[m & VIS]) ** 2)))
        row[k + "_swir"] = float(np.sqrt(np.mean((p[m & SWIR] - obs[m & SWIR]) ** 2)))
        row[k + "_x"] = res.x
    return row


def main():
    from run_global_validation import load_smith
    from smith_retrieval_validation import _direct_flag

    step = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    jobs = []
    for rec in list(load_smith())[::step]:
        obs = resample_to_model_grid(rec.alb, rec.wl_nm, method="linear")
        jobs.append((rec.date, obs, float(np.clip(rec.sza, 20, 85)),
                     _direct_flag(rec.sky)))
    rows = Parallel(n_jobs=-1)(delayed(fit_one)(*j) for j in jobs)

    print(f"{'date':12} | {'dry VIS':>7} {'SWIR':>7} | {'wet VIS':>7} {'SWIR':>7} "
          f"| {'r_dry':>6} {'r_wet':>6} {'LWC':>5}")
    for r in rows:
        print(f"{r['date']:12} | {r['dry_vis']:7.4f} {r['dry_swir']:7.4f} | "
              f"{r['wet_vis']:7.4f} {r['wet_swir']:7.4f} | "
              f"{r['dry_x'][0]:6.0f} {r['wet_x'][0]:6.0f} {r['wet_x'][3]:5.3f}")
    a = np.array([[r["dry_vis"], r["dry_swir"], r["wet_vis"], r["wet_swir"]]
                  for r in rows])
    lwc = np.array([r["wet_x"][3] for r in rows])
    print(f"\nmean RMS   dry VIS {a[:,0].mean():.4f}  SWIR {a[:,1].mean():.4f}   "
          f"wet VIS {a[:,2].mean():.4f}  SWIR {a[:,3].mean():.4f}")
    print(f"SWIR RMS reduction from LWC: "
          f"{100 * (1 - a[:,3].mean() / a[:,1].mean()):.1f}%  "
          f"(spectra improved by >10%: "
          f"{int((a[:,3] < 0.9 * a[:,1]).sum())}/{len(rows)})")
    print(f"retrieved LWC: median {np.median(lwc):.3f}, "
          f"at upper bound (>0.24): {int((lwc > 0.24).sum())}/{len(rows)}")


if __name__ == "__main__":
    main()
