"""Constrained forward validation of the terrestrial (glacier-ice) model against the
SW Greenland ASD field spectra.

Rationale (see conversation): a pure forward closure is ill-posed here -- too many
unmeasured structural unknowns shape the spectrum.  Instead we PROFILE over the
unmeasured nuisance parameters while FIXING the measured ones:

  fixed (field ground truth) : glacier_algae = measured_cells , solzen = metadata
  fitted (nuisance, bounded) : crust grain radius , crust density
  fixed to avoid degeneracy  : background LAP = 0 (dust would mimic the algae signal)

Surface model: granular weathered crust (layer_type=0, no air-ice Fresnel -> no
specular) over solid interior ice (layer_type=1).  `Albedo_master` is already
ARF-corrected, so model albedo is compared directly.

Two results:
  (1) constrained fit -- residual with algae pinned at the measured value, and whether
      the fitted grain/density land in physically plausible ranges.
  (2) ALGAE-PERTURBATION PROFILE (the real validation) -- for each algae-present
      sample, re-fit the nuisance parameters with algae fixed at m x measured for a
      range of m, and find where the residual is minimised.  If the spectrum prefers
      m ~ 1, the model's algae optics are right at field loadings.  A systematic
      preference for m > 1 means the per-cell absorption is too weak.

Speed: the forward model (`run_model`, adding-doubling) is evaluated on a coarse
(grain, density, algae, zenith) grid ONCE, then all fitting/profiling is done by
interpolation.  Interpolator accuracy is checked against direct `run_model`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from greenland_asd_validation import FIT_MASK, load_data, rms, OUTDIR
from biosnicar.bands._core import WVL
from biosnicar.drivers.run_model import run_model

WVL_NM = WVL * 1000.0
GRIDCACHE = OUTDIR / "crust_grid.npz"

# --- structural lookup grid axes ---
RDS = np.array([1500., 2500., 3500., 4500., 5000.])       # crust grain radius (um)
RHO = np.array([450., 600., 750.])                        # crust density (kg/m3)
ALGAE = np.array([0., 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 5e5])  # cells/mL
ZEN = np.array([40., 48., 55.])                           # solar zenith (deg)
CRUST_DZ = 0.08
ICE = dict(dz=3.0, rho=890.0, rds=1200.0)
LOGA = np.log10(ALGAE + 1.0)                              # interpolation axis for algae


def forward(rds, rho, algae, zen):
    return np.asarray(run_model(
        solver="adding-doubling", layer_type=[0, 1],
        dz=[CRUST_DZ, ICE["dz"]], rho=[rho, ICE["rho"]], rds=[rds, ICE["rds"]],
        glacier_algae=[algae, 0.0], dust=[0.0, 0.0], solzen=zen).albedo)


def build_grid():
    if GRIDCACHE.exists():
        d = np.load(GRIDCACHE)
        return d["spectra"]
    spectra = np.empty((len(RDS), len(RHO), len(ALGAE), len(ZEN), 480))
    n = spectra.size // 480
    k = 0
    for i, r in enumerate(RDS):
        for j, rho in enumerate(RHO):
            for a, al in enumerate(ALGAE):
                for z, zen in enumerate(ZEN):
                    spectra[i, j, a, z] = forward(r, rho, al, zen)
                    k += 1
        print("  grid %d/%d" % (k, n))
    np.savez(GRIDCACHE, spectra=spectra)
    return spectra


def make_interp(spectra):
    # interpolate over (rds, rho, log10(algae+1), zen) -> 480-band albedo
    return RegularGridInterpolator((RDS, RHO, LOGA, ZEN), spectra,
                                   bounds_error=False, fill_value=None)


def predict(interp, rds, rho, algae, zen):
    return interp((rds, rho, np.log10(algae + 1.0), zen))


def validate_interp(interp):
    # granular grain LUT is discrete; use radii valid in the LUT that sit BETWEEN
    # grid nodes (2000<2500|3500, 3000, 4000) so we test off-grid interpolation.
    rng = np.random.default_rng(0)
    errs = []
    for r in (2000., 3000., 4000.):
        for _ in range(2):
            rho = rng.uniform(480, 720)
            al = float(10 ** rng.uniform(3, 5.5)); zen = rng.uniform(41, 54)
            direct = forward(r, rho, al, zen)
            interpd = predict(interp, r, rho, al, zen)
            errs.append(rms(direct, interpd, FIT_MASK))
    print("interpolator vs direct run_model (off-grid): median RMS %.4f, max %.4f"
          % (float(np.median(errs)), float(np.max(errs))))


# fine nuisance grid for fitting (via interpolator, cheap)
FR = np.linspace(RDS[0], RDS[-1], 24)
FH = np.linspace(RHO[0], RHO[-1], 16)


def best_fit(interp, observed, algae, zen):
    """Return (min_rms, rds*, rho*) minimising residual over the nuisance grid."""
    best = (np.inf, np.nan, np.nan)
    for r in FR:
        for rho in FH:
            res = rms(predict(interp, r, rho, algae, zen), observed, FIT_MASK)
            if res < best[0]:
                best = (res, r, rho)
    return best


def main():
    samples = load_data()
    print("Building structural lookup grid (%d run_model calls)..."
          % (len(RDS) * len(RHO) * len(ALGAE) * len(ZEN)))
    interp = make_interp(build_grid())
    validate_interp(interp)

    # (1) constrained fit: algae pinned at measured value
    print("\n(1) CONSTRAINED FIT (algae = measured_cells, fit grain+density):")
    rows = []
    for s in samples:
        res, r, rho = best_fit(interp, s["observed"], s["measured_cells"], s["zen"])
        rows.append(dict(name=s["name"], cells=s["measured_cells"],
                         rms=res, rds=r, rho=rho))
    res_all = np.array([x["rms"] for x in rows])
    print("  residual RMS: median %.4f  (IQR %.4f-%.4f)"
          % (np.median(res_all), np.percentile(res_all, 25), np.percentile(res_all, 75)))
    print("  fitted grain radius: median %.0f um  (range %.0f-%.0f)"
          % (np.median([x["rds"] for x in rows]),
             min(x["rds"] for x in rows), max(x["rds"] for x in rows)))
    print("  fitted crust density: median %.0f  (range %.0f-%.0f)"
          % (np.median([x["rho"] for x in rows]),
             min(x["rho"] for x in rows), max(x["rho"] for x in rows)))

    # (2) algae-perturbation profile (algae-present samples only)
    print("\n(2) ALGAE-PERTURBATION PROFILE (re-fit nuisance at algae = m x measured):")
    mult = np.logspace(np.log10(0.1), np.log10(10.0), 15)
    present = [s for s in samples if s["measured_cells"] > 0]
    prof = np.full((len(present), len(mult)), np.nan)
    mstar = []
    for i, s in enumerate(present):
        for j, m in enumerate(mult):
            prof[i, j] = best_fit(interp, s["observed"], m * s["measured_cells"], s["zen"])[0]
        mstar.append(mult[int(np.argmin(prof[i]))])
    med_curve = np.nanmedian(prof, axis=0)
    m_at_min = mult[int(np.argmin(med_curve))]
    mstar = np.array(mstar)
    print("  n algae-present samples: %d" % len(present))
    print("  median residual minimised at m = %.2f x measured" % m_at_min)
    print("  per-sample preferred multiplier m*: median %.2f  (IQR %.2f-%.2f)"
          % (np.median(mstar), np.percentile(mstar, 25), np.percentile(mstar, 75)))
    frac_gt1 = float(np.mean(mstar > 1.0))
    print("  fraction of samples preferring MORE algae than measured: %.0f%%" % (100 * frac_gt1))
    if m_at_min > 1.3:
        print("  -> spectrum prefers MORE algae than measured => model per-cell absorption too WEAK")
    elif m_at_min < 0.77:
        print("  -> spectrum prefers LESS algae than measured => model per-cell absorption too STRONG")
    else:
        print("  -> spectrum prefers ~measured algae => algae optics consistent at field loadings")

    np.savez(OUTDIR / "constrained_validation.npz",
             names=[s["name"] for s in samples],
             cells=np.array([s["measured_cells"] for s in samples]),
             fit_rms=res_all,
             fit_rds=np.array([x["rds"] for x in rows]),
             fit_rho=np.array([x["rho"] for x in rows]),
             mult=mult, profile=prof, mstar=mstar,
             present_names=[s["name"] for s in present],
             present_cells=np.array([s["measured_cells"] for s in present]))
    print("\nWrote", OUTDIR / "constrained_validation.npz")


if __name__ == "__main__":
    main()
