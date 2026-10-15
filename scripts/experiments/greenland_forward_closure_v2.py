"""Forward-model closure for SW Greenland ASD spectra — honest 3-way comparison.

Following the request to (a) pin the adding-doubling solver, (b) use a depth-varying
density/reff profile (weathering-crust / surface-scattering-layer analog), and
(c) use fixed literature crust parameters (true forward closure, no per-sample fit).

Measurement note: `Albedo_master.csv` is already ARF-corrected (verified
Albedo == HCRF / ARF, corr 1.000), i.e. the albedo-equivalent quantity the model
predicts — compared directly, no further BRDF correction.

Three forward models, all adding-doubling, algae = field-measured cells, sun angle
from metadata:

  A  single solid-ice layer, PER-SAMPLE S2-retrieved grain/density/dz
     -> reference, but SEMI-CIRCULAR: those inputs were fit to reproduce the same
        surfaces from Sentinel-2 bands that overlap the ASD VIS-NIR.
  B  single solid-ice layer, FIXED grain/density (blind — only algae + sun vary)
  C  2-layer weathering crust (porous solid-ice SSL over interior), FIXED literature
     structure, algae in the crust.

Finding: with fixed (non-circular) parameters, the single layer B (0.094) beats the
crust C (0.116); the crust does not improve the closure for these melt-season
dark-zone samples.  A's low RMS is inflated by circular S2 inputs.  Root cause:
the model's granular ice is intrinsically VIS-bright (~0.81) and its solid ice is
NIR-dark, so no fixed template spans the middle ground occupied by real weathering-
crust ice, and there is a residual VIS over-brightness for the cleanest ice.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from greenland_asd_validation import FIT_MASK, VIS_MASK, NIR_MASK, load_data, rms, OUTDIR
from biosnicar.drivers.run_model import run_model

# Fixed single-layer solid-ice (blind) params
B_RDS, B_RHO = 800.0, 850.0
# GRANULAR weathered-crust template: granular ice (layer_type=0, NO air-ice Fresnel
# top -> no specular peak) with coarse weathered grains, over solid interior ice.
# Grain radius at the LUT ceiling (5000 um = 5 mm) to represent decayed coarse crust.
C_CRUST = dict(dz=0.08, rho=600.0, rds=5000.0)   # granular weathering crust
C_ICE = dict(dz=3.0, rho=890.0, rds=1200.0)      # solid interior glacier ice


def model_A(s):
    return np.asarray(run_model(solver="adding-doubling", layer_type=[1],
        rds=[s["reff"]], rho=[s["rho"]], dz=[max(s["dz"], 0.02)],
        solzen=s["zen"], glacier_algae=[s["measured_cells"]]).albedo)


def model_B(s):
    return np.asarray(run_model(solver="adding-doubling", layer_type=[1],
        rds=[B_RDS], rho=[B_RHO], dz=[3.0],
        solzen=s["zen"], glacier_algae=[s["measured_cells"]]).albedo)


def model_C(s):
    # granular weathered crust (layer_type=0) over solid interior (layer_type=1)
    return np.asarray(run_model(solver="adding-doubling", layer_type=[0, 1],
        dz=[C_CRUST["dz"], C_ICE["dz"]], rho=[C_CRUST["rho"], C_ICE["rho"]],
        rds=[C_CRUST["rds"], C_ICE["rds"]],
        glacier_algae=[s["measured_cells"], 0.0], solzen=s["zen"]).albedo)


MODELS = {"A single solid (S2 per-sample)": model_A,
          "B single solid (fixed, blind)": model_B,
          "C granular crust (fixed, blind)": model_C}


def main():
    samples = load_data()
    preds = {k: [] for k in MODELS}
    rows = {k: {"full": [], "vis": [], "nir": []} for k in MODELS}
    for s in samples:
        for k, fn in MODELS.items():
            p = fn(s)
            preds[k].append(p)
            rows[k]["full"].append(rms(p, s["observed"], FIT_MASK))
            rows[k]["vis"].append(rms(p, s["observed"], VIS_MASK))
            rows[k]["nir"].append(rms(p, s["observed"], NIR_MASK))

    print("Forward closure, median spectral-fit RMS (albedo, n=%d)\n" % len(samples))
    print("%-28s %8s %8s %8s" % ("model", "full", "VIS", "NIR"))
    for k in MODELS:
        print("%-28s %8.4f %8.4f %8.4f" % (
            k, np.nanmedian(rows[k]["full"]),
            np.nanmedian(rows[k]["vis"]), np.nanmedian(rows[k]["nir"])))
    a = np.array(rows["A single solid (S2 per-sample)"]["full"])
    for k in ["B single solid (fixed, blind)", "C granular crust (fixed, blind)"]:
        b = np.array(rows[k]["full"])
        print("  %s beats A on %d/%d samples" % (k, int((b < a).sum()), len(samples)))
    bb = np.array(rows["B single solid (fixed, blind)"]["full"])
    cc = np.array(rows["C granular crust (fixed, blind)"]["full"])
    print("  C granular beats B solid on %d/%d samples" % (int((cc < bb).sum()), len(samples)))

    pd.DataFrame({f"{k}_{b}": rows[k][b] for k in MODELS for b in ("full", "vis", "nir")}
                 ).assign(name=[s["name"] for s in samples],
                          measured_cells=[s["measured_cells"] for s in samples]
                          ).to_csv(OUTDIR / "forward_closure_v2.csv", index=False)
    np.savez(OUTDIR / "forward_v2_spectra.npz",
             names=[s["name"] for s in samples],
             measured_cells=np.array([s["measured_cells"] for s in samples]),
             observed=np.array([s["observed"] for s in samples]),
             **{f"pred_{c}": np.array(preds[k]) for c, k in zip("ABC", MODELS)})
    print("\nWrote", OUTDIR / "forward_closure_v2.csv")


if __name__ == "__main__":
    main()
