"""Validate the terrestrial (glacier-ice) forward model and emulator retrieval
against real field spectra from the SW Greenland Ice Sheet (2016-2017).

Data (measured by J. Cook, ASD FieldSpec Pro 3):
    data/additional_data/Albedo_master.csv    -- 350-2500 nm @ 1 nm, one column per spectrum
    data/additional_data/Spectra_Metadata.csv -- per-sample physical state + field-measured
                                                  glacier-algae cell counts (`measured_cells`)

Two experiments
---------------
Exp 1  FORWARD CLOSURE (exact RT model, `run_model`):
       drive the model with each sample's physical state and compare the predicted
       spectrum to the measured spectrum.  Tests the forward radiative-transfer model.

Exp 2  EMULATOR RETRIEVAL (`Emulator` + `retrieve`):
       invert each *measured* spectrum and compare the retrieved glacier-algae load to
       the *field-measured* cell count -- the one genuinely independent ground truth.
       Also reports spectral-fit quality and physical-parameter consistency.

Honesty notes
-------------
* The `*_inv_model_S2` metadata columns are prior Sentinel-2 inverse-model retrievals,
  NOT independent measurements.  They are used only as (a) forward-model inputs and
  (b) a consistency reference -- never as retrieval ground truth.
* `measured_cells` (field microscopy, cells mL-1) IS independent ground truth and is the
  headline metric for the retrieval.
* The SWIR tail of the ASD is noisy over ice (albedo ~0, low signal); comparison is
  restricted to 350-1800 nm excluding the 1340-1470 nm water-vapour band.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from biosnicar.bands._core import WVL
from biosnicar.drivers.run_model import run_model
from biosnicar.emulator import Emulator
from biosnicar.inverse.optimize import retrieve
from biosnicar.sea_ice import spectral_utils as su

REPO = Path(__file__).resolve().parents[2]
ALBEDO_CSV = REPO / "data" / "additional_data" / "Albedo_master.csv"
META_CSV = REPO / "data" / "additional_data" / "Spectra_Metadata.csv"
EMULATOR = REPO / "data" / "emulators" / "glacier_ice_8_param_default.npz"
OUTDIR = REPO / "scripts" / "experiments" / "greenland_asd_results"
OUTDIR.mkdir(exist_ok=True)

WVL_NM = WVL * 1000.0  # model grid in nm

# Trusted comparison band: 350-1800 nm minus the 1340-1470 nm water-vapour absorption.
FIT_MASK = ((WVL_NM >= 350) & (WVL_NM <= 1800)) & ~((WVL_NM >= 1340) & (WVL_NM <= 1470))
VIS_MASK = FIT_MASK & (WVL_NM <= 700)
NIR_MASK = FIT_MASK & (WVL_NM > 700)


def rms(a, b, mask):
    d = (np.asarray(a) - np.asarray(b))[mask]
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d**2))) if d.size else np.nan


def load_data():
    alb = pd.read_csv(ALBEDO_CSV)
    meta = pd.read_csv(META_CSV)
    asd_wvl = alb["Wavelength"].to_numpy(dtype=float)
    samples = []
    for _, row in meta.iterrows():
        name = row["filename"]
        if name not in alb.columns:
            continue
        raw = alb[name].to_numpy(dtype=float)
        observed = np.asarray(
            su.resample_to_model_grid(raw, asd_wvl, method="gaussian", fwhm=3.0)
        )
        samples.append(
            dict(
                name=name,
                observed=observed,
                reff=float(row["reff_inv_model_S2"]),
                rho=float(row["density_inv_model_S2"]),
                dz=float(row["dz_inv_model_S2"]),
                zen=float(row["zen_inv_model_S2"]),
                measured_cells=float(row["measured_cells"]),
            )
        )
    return samples


def experiment_1_forward(samples):
    """Exact RT model driven by each sample's physical state vs measured spectrum."""
    rows = []
    for s in samples:
        out = run_model(
            layer_type=[1],
            rds=[s["reff"]],
            rho=[s["rho"]],
            dz=[max(s["dz"], 0.02)],
            solzen=s["zen"],
            glacier_algae=[s["measured_cells"]],
        )
        pred = np.asarray(out.albedo)
        rows.append(
            dict(
                name=s["name"],
                rms_full=rms(pred, s["observed"], FIT_MASK),
                rms_vis=rms(pred, s["observed"], VIS_MASK),
                rms_nir=rms(pred, s["observed"], NIR_MASK),
                predicted=pred,
            )
        )
    return rows


def experiment_2_retrieval(samples, emulator, config):
    """Invert each measured spectrum; compare retrieved algae to field cell counts."""
    rows = []
    for s in samples:
        if config == "independent":
            params = ["rds", "rho", "glacier_algae"]
            fixed = dict(
                black_carbon=0.0, dust=0.0, snow_algae=0.0, direct=0.9, solzen=s["zen"]
            )
            x0 = dict(rds=3000.0, rho=600.0, glacier_algae=5000.0)
        else:  # rho fixed to metadata density
            params = ["rds", "glacier_algae"]
            fixed = dict(
                rho=s["rho"], black_carbon=0.0, dust=0.0, snow_algae=0.0,
                direct=0.9, solzen=s["zen"],
            )
            x0 = dict(rds=3000.0, glacier_algae=5000.0)

        res = retrieve(
            s["observed"],
            parameters=params,
            emulator=emulator,
            fixed_params=fixed,
            x0=x0,
            wavelength_mask=FIT_MASK,
            method="L-BFGS-B",
        )
        bf = {k: float(v) for k, v in res.best_fit.items()}
        rows.append(
            dict(
                name=s["name"],
                measured_cells=s["measured_cells"],
                retrieved_algae=bf.get("glacier_algae", np.nan),
                retrieved_rds=bf.get("rds", np.nan),
                retrieved_rho=bf.get("rho", s["rho"]),
                meta_reff=s["reff"],
                meta_rho=s["rho"],
                rms_fit=rms(res.predicted_albedo, s["observed"], FIT_MASK),
                converged=bool(res.converged),
            )
        )
    return rows


def summarise(fwd, ret_ind, ret_fix):
    def med(xs):
        xs = [x for x in xs if np.isfinite(x)]
        return float(np.median(xs)) if xs else np.nan

    # --- forward ---
    print("\n=== EXPERIMENT 1: FORWARD MODEL vs MEASURED SPECTRA (n=%d) ===" % len(fwd))
    print("  spectral-fit RMS (albedo units), model driven by measured/S2 state:")
    print("    full 350-1800nm : median %.4f" % med([r["rms_full"] for r in fwd]))
    print("    VIS  350-700nm  : median %.4f" % med([r["rms_vis"] for r in fwd]))
    print("    NIR  700-1800nm : median %.4f" % med([r["rms_nir"] for r in fwd]))

    # --- retrieval: algae vs measured_cells (headline) ---
    for label, ret in [("INDEPENDENT (retrieve rds,rho,algae)", ret_ind),
                       ("RHO-FIXED (retrieve rds,algae; rho=S2)", ret_fix)]:
        m = np.array([r["measured_cells"] for r in ret], float)
        a = np.array([r["retrieved_algae"] for r in ret], float)
        fit = [r["rms_fit"] for r in ret]
        # correlation on log10(x+1) to handle the wide dynamic range + zeros
        lm, la = np.log10(m + 1), np.log10(a + 1)
        pear = float(np.corrcoef(lm, la)[0, 1])
        # Spearman via rank correlation
        rm = pd.Series(m).rank().to_numpy()
        ra = pd.Series(a).rank().to_numpy()
        spear = float(np.corrcoef(rm, ra)[0, 1])
        # detection skill: measured>0 vs retrieved>1000 cells/mL
        thr = 1000.0
        present = m > 0
        detected = a > thr
        tp = int(np.sum(present & detected)); fn = int(np.sum(present & ~detected))
        tn = int(np.sum(~present & ~detected)); fp = int(np.sum(~present & detected))
        print("\n=== EXPERIMENT 2: EMULATOR RETRIEVAL -- %s (n=%d) ===" % (label, len(ret)))
        print("  spectral-fit RMS: median %.4f  (max %.4f)" % (med(fit), np.nanmax(fit)))
        print("  retrieved glacier_algae vs FIELD measured_cells:")
        print("    Pearson r (log10 space): %.3f" % pear)
        print("    Spearman rank r        : %.3f" % spear)
        print("    detection @1000 cells/mL: TP=%d FN=%d TN=%d FP=%d  (present=%d absent=%d)"
              % (tp, fn, tn, fp, int(present.sum()), int((~present).sum())))
        zero_ret = a[~present]
        if zero_ret.size:
            print("    retrieved algae where measured==0: median %.0f  max %.0f cells/mL"
                  % (float(np.median(zero_ret)), float(np.max(zero_ret))))

    # consistency vs S2 physical params (independent config)
    rds_r = np.array([r["retrieved_rds"] for r in ret_ind], float)
    rds_m = np.array([r["meta_reff"] for r in ret_ind], float)
    print("\n  physical-parameter consistency (independent config, vs S2 inverse-model):")
    print("    grain radius: retrieved median %.0f um  vs  S2 median %.0f um" %
          (float(np.median(rds_r)), float(np.median(rds_m))))


def main():
    samples = load_data()
    emulator = Emulator.load(str(EMULATOR))
    print("Loaded %d matched Greenland ASD samples." % len(samples))

    fwd = experiment_1_forward(samples)
    ret_ind = experiment_2_retrieval(samples, emulator, "independent")
    ret_fix = experiment_2_retrieval(samples, emulator, "rho_fixed")
    summarise(fwd, ret_ind, ret_fix)

    # persist per-sample tables + arrays for the figure
    pd.DataFrame([{k: v for k, v in r.items() if k != "predicted"} for r in fwd]).to_csv(
        OUTDIR / "forward_closure.csv", index=False)
    pd.DataFrame(ret_ind).to_csv(OUTDIR / "retrieval_independent.csv", index=False)
    pd.DataFrame(ret_fix).to_csv(OUTDIR / "retrieval_rho_fixed.csv", index=False)
    np.savez(
        OUTDIR / "spectra.npz",
        wvl_nm=WVL_NM,
        fit_mask=FIT_MASK,
        names=[s["name"] for s in samples],
        observed=np.array([s["observed"] for s in samples]),
        predicted=np.array([r["predicted"] for r in fwd]),
        measured_cells=np.array([s["measured_cells"] for s in samples]),
    )
    print("\nWrote results to %s" % OUTDIR)


if __name__ == "__main__":
    main()
