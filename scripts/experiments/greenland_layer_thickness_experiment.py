"""Does the algae-layer thickness explain the 'needs more algae' result?

Physics (verified in code): glacier_algae is cells/mL (unit=1, cfactor=1); the algal
optical depth is tau_algae = conc * rho * dz * ext_xsc -- i.e. AREAL cell loading,
linear in layer thickness.  The field sample melts the top ~2 cm and counts cells, so
the measured concentration captures the *total* (areal) cells correctly; what is
unknown is the vertical distribution.  In reality the cells sit in the top ~2 mm.

Experiment: hold the measured AREAL loading fixed (conc * dz = C_meas * D_SAMPLE) and
sweep the algae-bearing top-layer thickness t.  Concentrating the same cells into a
thinner near-surface layer (higher conc, smaller t) puts them where the actinic flux
is highest, which can raise absorption efficiency.  For each t we re-run the
algae-perturbation profile and find the preferred multiplier m*(t).

  - if m*(t) -> 1 as t -> 2 mm : the earlier '3-5x too weak' was a layer-placement
    artifact (algae spread too thick / too deep), not a MAC error.
  - if m*(t) stays > 1 at 2 mm : the per-cell absorption (MAC) is genuinely too weak.

Model: two granular layers (layer_type=[0,0], no Fresnel/specular) -- an algae-bearing
top of thickness t over a thick clean weathering-crust interior, same grain radius
(the fitted nuisance) and density.  `Albedo_master` is ARF-corrected -> compared direct.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from greenland_asd_validation import FIT_MASK, load_data, rms, OUTDIR
from biosnicar.bands._core import WVL
from biosnicar.drivers.run_model import run_model

WVL_NM = WVL * 1000.0
GRIDCACHE = OUTDIR / "thickness_grid.npz"
D_SAMPLE = 0.02          # field sampling depth (m) the measured concentration averages over
RHO_CRUST = 600.0        # granular weathering-crust density (kg/m3)

DZ = np.array([0.002, 0.005, 0.01, 0.02, 0.05])                 # algae-layer thickness (m)
RDS = np.array([2000., 3000., 4000., 5000.])                   # granular grain radius (um)
CONC = np.array([0., 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7])  # layer concentration (cells/mL)
ZEN = np.array([40., 48., 55.])
LOGC = np.log10(CONC + 1.0)


def forward(dz, rds, conc, zen):
    return np.asarray(run_model(
        solver="adding-doubling", layer_type=[0, 0], dz=[dz, 3.0],
        rho=[RHO_CRUST, RHO_CRUST], rds=[rds, rds],
        glacier_algae=[conc, 0.0], solzen=zen).albedo)


def build_grid():
    if GRIDCACHE.exists():
        return np.load(GRIDCACHE)["spectra"]
    spectra = np.empty((len(DZ), len(RDS), len(CONC), len(ZEN), 480))
    tot = spectra.size // 480
    k = 0
    for i, dz in enumerate(DZ):
        for j, r in enumerate(RDS):
            for a, c in enumerate(CONC):
                for z, zn in enumerate(ZEN):
                    spectra[i, j, a, z] = forward(dz, r, c, zn)
                    k += 1
        print("  grid %d/%d" % (k, tot))
    np.savez(GRIDCACHE, spectra=spectra)
    return spectra


def main():
    interp = RegularGridInterpolator((DZ, RDS, LOGC, ZEN), build_grid(),
                                     bounds_error=False, fill_value=None)

    def predict(dz, rds, conc, zen):
        return interp((dz, rds, np.log10(conc + 1.0), zen))

    # interpolator accuracy check (valid LUT radii between nodes)
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(6):
        dz = float(rng.choice(DZ)); r = float(rng.choice([2000., 3000., 4000., 5000.]))
        c = float(10 ** rng.uniform(3, 6)); zn = rng.uniform(41, 54)
        errs.append(rms(forward(dz, r, c, zn), predict(dz, r, c, zn), FIT_MASK))
    print("interp check: median RMS %.4f max %.4f\n" % (np.median(errs), np.max(errs)))

    samples = load_data()
    present = [s for s in samples if s["measured_cells"] > 0]
    FR = np.linspace(RDS[0], RDS[-1], 20)
    mult = np.logspace(np.log10(0.1), np.log10(20.0), 17)

    print("Algae-layer thickness sweep (areal loading conserved = C_meas * %.0f cm):"
          % (D_SAMPLE * 100))
    print("%-10s %10s %12s %14s" % ("t (mm)", "med resid", "median m*", "%prefer >1x"))
    out = {}
    for dz_fixed in DZ:
        # concentration that reproduces the MEASURED areal loading at this thickness
        def conc_meas(s):
            return s["measured_cells"] * (D_SAMPLE / dz_fixed)
        # constrained-fit residual (algae at measured areal loading)
        res = []
        for s in samples:
            res.append(min(rms(predict(dz_fixed, r, conc_meas(s), s["zen"]),
                                s["observed"], FIT_MASK) for r in FR))
        # perturbation profile -> m*
        mstar = []
        prof = np.full((len(present), len(mult)), np.nan)
        for i, s in enumerate(present):
            for k, m in enumerate(mult):
                prof[i, k] = min(rms(predict(dz_fixed, r, conc_meas(s) * m, s["zen"]),
                                     s["observed"], FIT_MASK) for r in FR)
            mstar.append(mult[int(np.argmin(prof[i]))])
        mstar = np.array(mstar)
        out[dz_fixed] = dict(res=np.array(res), mstar=mstar, prof=prof)
        print("%-10.0f %10.4f %12.2f %14.0f%%" % (
            dz_fixed * 1000, np.median(res), np.median(mstar), 100 * np.mean(mstar > 1)))

    np.savez(OUTDIR / "thickness_experiment.npz", mult=mult,
             dz=DZ, D_SAMPLE=D_SAMPLE,
             **{f"res_{int(d*1000)}": out[d]["res"] for d in DZ},
             **{f"mstar_{int(d*1000)}": out[d]["mstar"] for d in DZ},
             **{f"prof_{int(d*1000)}": out[d]["prof"] for d in DZ})
    print("\nWrote", OUTDIR / "thickness_experiment.npz")


if __name__ == "__main__":
    main()
