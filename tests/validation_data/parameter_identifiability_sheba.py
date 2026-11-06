#!/usr/bin/env python3
"""Are the retrieved sea-ice parameters actually identified, on real spectra?

``scripts/audit_sea_ice_degeneracy.py`` measures posterior correlation on
*synthetic* states, each fit started at the truth.  That measures uncertainty
inflation but is blind to the failure mode where a degenerate ridge plus a hard
parameter bound slides the optimiser onto a rail, and it reports the rail with
an error bar.  This runs the real SHEBA spectra (Grenfell & Light 2007) and
checks for it.

Two diagnostics, and the combination is what matters:

``pinned``   the retrieved value sits within 2% of a bound.
``A``        averaging-kernel diagonal: 1 = the measurement determined this,
             0 = the prior/bound did.

  pinned + high A  -> the measurement is pushing hard against the bound; the
                      bound is binding and may be set too tight.
  pinned + low A   -> the likelihood is flat in this parameter; it drifted to
                      the wall and the reported value is meaningless.
  not pinned       -> fine.

Measured 2026-11-06 on 23 SHEBA spectra: 22/92 parameters (24%) at a bound and
19/23 spectra carrying a |r| > 0.8 pair.  ``sea_ice_temperature`` is the
standout -- averaging kernel 0.01-0.03 in several fits, i.e. reported but not
retrieved.

Usage::

    uv run python tests/validation_data/parameter_identifiability_sheba.py
"""
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
warnings.simplefilter("ignore")

import numpy as np
from scipy.interpolate import interp1d

from biosnicar.sea_ice.retrieve import retrieve_sea_ice
from biosnicar.sea_ice.emulator_configs import SEA_ICE_EMULATOR_CONFIGS as CFG
from biosnicar.sea_ice.spectral_utils import MODEL_WAVELENGTHS_NM as WL_NM

from Grenfell_light_2007.validate_grenfell_light_2007 import (  # noqa: E402
    catalogue, parse_albv,
)

VIS = (WL_NM >= 400) & (WL_NM <= 1000)
AT_BOUND = 2e-2      # within 2% of the range endpoint
LOW_INFO = 0.3       # averaging-kernel diagonal below this = prior-dominated


def load_sheba():
    spring, summer = catalogue()
    rows = []
    for entries, label in ((spring, "spring_snow"), (summer, "summer_bare")):
        for e in entries:
            wl, alb, _ = parse_albv(e["path"])
            if wl is None:
                continue
            obs = interp1d(wl, alb, kind="linear", bounds_error=False,
                           fill_value=np.nan)(WL_NM)
            obs = np.where(VIS & np.isfinite(obs), obs, np.nan)
            if np.isfinite(obs).sum() >= 30:
                rows.append((label, e["date"], e["sza"], obs))
    return rows


def main():
    rows = load_sheba()
    print(f"{len(rows)} SHEBA spectra, method='oe'\n")
    n_pin = n_par = n_hi = 0
    rails, low_info = {}, {}
    for label, date, sza, obs in rows:
        try:
            r = retrieve_sea_ice(observed=obs, solzen=int(round(sza)),
                                 direct=1, method="oe")
        except Exception as exc:
            print(f"{label:12} {date}  FAILED: {exc}")
            continue
        fit = r.all_fits.get(r.surface_type)
        if fit is None or not fit.posterior_covariance:
            continue
        P = CFG[r.surface_type]["params"]
        akd = fit.averaging_kernel_diag or {}
        notes = []
        for p, v in fit.best_fit.items():
            if p not in P:
                continue
            lo, hi = P[p]
            n_par += 1
            frac = (v - lo) / (hi - lo)
            a = akd.get(p, float("nan"))
            if a < LOW_INFO:
                low_info.setdefault(p, []).append(a)
            if frac < AT_BOUND or frac > 1 - AT_BOUND:
                n_pin += 1
                rails.setdefault((r.surface_type, p), []).append(a)
                kind = "binding" if a >= LOW_INFO else "FLAT"
                notes.append(f"{p}={v:.4g}@{'lo' if frac < 0.5 else 'hi'}"
                             f"(A={a:.2f},{kind})")
        worst = fit.degenerate_parameters(threshold=0.8)
        if worst:
            n_hi += 1
        w = (f"{worst[0][0][:13]}/{worst[0][1][:13]}={worst[0][2]:+.2f}"
             if worst else "-")
        print(f"{label:12} {date}  {r.surface_type:11} chi2={fit.cost:7.1f}  "
              f"worst={w:34} {' '.join(notes) if notes else ''}")

    print(f"\nparameters at a bound:            {n_pin}/{n_par} "
          f"({100 * n_pin / max(n_par, 1):.0f}%)")
    print(f"spectra with a |r| > 0.8 pair:    {n_hi}/{len(rows)}")
    if rails:
        print("\nrails (pinned parameter -> averaging kernels):")
        for (t, p), a in sorted(rails.items(), key=lambda kv: -len(kv[1])):
            tag = ("bound binding" if np.nanmean(a) >= LOW_INFO
                   else "flat likelihood - value meaningless")
            print(f"  {t:11} {p:24} {len(a):2}x  mean A={np.nanmean(a):.2f}"
                  f"  [{tag}]")
    if low_info:
        print(f"\nprior-dominated parameters (A < {LOW_INFO}) -- reported but "
              f"not retrieved:")
        for p, a in sorted(low_info.items(), key=lambda kv: -len(kv[1])):
            print(f"  {p:28} {len(a):2}x  mean A={np.nanmean(a):.3f}")


if __name__ == "__main__":
    main()
