"""Choose the weathering-crust template for the forward closure.

Not per-sample fitting: each template is a SINGLE fixed structure applied to all 31
samples; only algae (field cells) + solar zenith vary.  We evaluate a handful of
literature-plausible templates and report which best reproduces the measured spectra.

Physical rationale for the knobs:
  * background dust in the crust -- SW Greenland "dark zone" bare ice carries mineral
    dust + amorphous organic matter even where algal cells are absent (Wientjes et al.
    2011; Cook et al. 2020).  Needed to explain the VIS level of the brightest ice.
  * crust density / grain radius -- weathering crust is porous decayed ice, between
    snow and solid ice in brightness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from greenland_asd_validation import FIT_MASK, VIS_MASK, NIR_MASK, load_data, rms, OUTDIR
from biosnicar.drivers.run_model import run_model

# Each template: crust (layer_type=0) over solid ice (layer_type=1).
# dust/bc are background loadings in the crust layer (ppb).
TEMPLATES = {
    "v1_single (baseline)": None,  # handled specially
    "crust bright (no dust)":      dict(c_rho=600, c_rds=3000, dust=0,    bc=0),
    "crust + dust1500":            dict(c_rho=600, c_rds=3000, dust=1500, bc=0),
    "crust dense + dust1500":      dict(c_rho=750, c_rds=2000, dust=1500, bc=0),
    "crust dense fine + dust3000": dict(c_rho=800, c_rds=1200, dust=3000, bc=0),
    "crust dense + dust1000 + bc100": dict(c_rho=750, c_rds=2000, dust=1000, bc=100),
}
CRUST_DZ, ICE_DZ, ICE_RHO, ICE_RDS = 0.10, 3.0, 890.0, 1500.0


def predict(s, t):
    if t is None:
        out = run_model(solver="adding-doubling", layer_type=[1],
                        rds=[s["reff"]], rho=[s["rho"]], dz=[max(s["dz"], 0.02)],
                        solzen=s["zen"], glacier_algae=[s["measured_cells"]])
    else:
        out = run_model(
            solver="adding-doubling", layer_type=[0, 1], dz=[CRUST_DZ, ICE_DZ],
            rho=[t["c_rho"], ICE_RHO], rds=[t["c_rds"], ICE_RDS],
            glacier_algae=[s["measured_cells"], 0.0],
            dust=[t["dust"], 0.0], black_carbon=[t["bc"], 0.0],
            solzen=s["zen"])
    return np.asarray(out.albedo)


def main():
    samples = load_data()
    base_full = None
    print("Weathering-crust template sweep (median RMS over n=%d)\n" % len(samples))
    print("%-32s %8s %8s %8s %10s" % ("template", "full", "VIS", "NIR", "improved"))
    results = {}
    for label, t in TEMPLATES.items():
        ff, vv, nn = [], [], []
        per_full = []
        for s in samples:
            p = predict(s, t)
            per_full.append(rms(p, s["observed"], FIT_MASK))
            ff.append(per_full[-1])
            vv.append(rms(p, s["observed"], VIS_MASK))
            nn.append(rms(p, s["observed"], NIR_MASK))
        per_full = np.array(per_full)
        if base_full is None:
            base_full = per_full
            imp = "-"
        else:
            imp = "%d/%d" % (int((per_full < base_full).sum()), len(samples))
        results[label] = per_full
        print("%-32s %8.4f %8.4f %8.4f %10s" %
              (label, np.nanmedian(ff), np.nanmedian(vv), np.nanmedian(nn), imp))
    pd.DataFrame(results).to_csv(OUTDIR / "crust_sweep.csv", index=False)


if __name__ == "__main__":
    main()
