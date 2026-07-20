#!/usr/bin/env python3
"""Perturbed-physics validation (V2) — bounding the inverse-crime overstatement.

Every synthetic accuracy figure in this project is an *inverse-crime* number:
the same forward model generates the test spectra and inverts them. Those
figures bound what is retrievable *in principle*; they say nothing about
performance when the forward model disagrees with reality. This experiment
breaks the crime in graded steps and measures how much each step costs, so the
headline synthetic numbers can be reported with an honest degradation bound.

The ladder (per surface type, N truth states, fixed geometry, no priors):

  C0  inverse crime   truth = emulator(params)          invert with emulator
  C1  surrogate gap   truth = run_model(full physics)   invert with emulator
  C2  model error     truth = run_model + a draw from   invert with emulator
                       the A7 field-residual covariance
                       (real magnitude & correlation)

C0→C1 isolates the emulator's own approximation error; C1→C2 adds realistic
forward-model mismatch at the magnitude actually measured against field spectra
(biosnicar.inverse.model_error). Each condition is inverted twice — with
instrument-only S_e and with model_error=True — so the experiment also tests
whether the A7 covariance keeps the retrieval honest under mismatch (does the
reduced chi-squared stay sane; do the right things get flagged).

Metrics per condition: fleet classification accuracy (winner == truth type),
parameter recovery for the type's best-constrained parameter on the correctly
classified subset, and median OE reduced chi-squared (a mismatch detector).

Usage::

    python tests/validation_data/perturbed_physics_validation.py [--n 30] [--json out.json]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from biosnicar.drivers.run_model import run_model  # noqa: E402
from biosnicar.inverse.model_error import ModelErrorCovariance  # noqa: E402
from biosnicar.sea_ice.emulator_configs import (  # noqa: E402
    SEA_ICE_EMULATOR_CONFIGS, load_sea_ice_emulators,
)
from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch  # noqa: E402

# The parameter each type constrains best (from E1) — what we score recovery on.
_KEY_PARAM = {
    "FYI_bare": "sea_ice_bubble_radius",
    "MYI_bare": "sea_ice_bubble_radius",
    "FYI_snow": "snow_grain_radius",
    "FYI_summer": "ssl_grain_radius",
    "FYI_pond": "pond_depth",
    "young_ice": "ice_thickness_cm",
    "open_water": "wind_speed_ms",
}
_KNOWN_MONTH = 7          # melt-season context, matching the deployment case


def _truth_params(cfg, n, rng):
    """N parameter sets sampled uniformly within the type's training bounds."""
    rows = []
    for _ in range(n):
        p = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in cfg["params"].items()}
        p["solzen"] = 60.0
        p["direct"] = 1
        rows.append(p)
    return rows


def _full_model_spectrum(cfg, p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(run_model(**cfg["transform_fn"](p)).albedo, dtype=float)


def _r2(truth, pred):
    truth, pred = np.asarray(truth), np.asarray(pred)
    ss_res = np.sum((truth - pred) ** 2)
    ss_tot = np.sum((truth - np.mean(truth)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def run(n=30, seed=2027):
    rng = np.random.default_rng(seed)
    fleet = load_sea_ice_emulators()
    types = list(SEA_ICE_EMULATOR_CONFIGS)
    model_err = ModelErrorCovariance.load()
    # Draw forward-model-error perturbations from the A7 covariance over the
    # full grid (EOF structure in the VIS + diagonal floor elsewhere).
    Se_full = model_err.dense(np.ones(480, dtype=bool))

    results = []
    for stype in types:
        cfg = SEA_ICE_EMULATOR_CONFIGS[stype]
        emu = fleet[stype]
        params = _truth_params(cfg, n, rng)

        # C0 truth: the emulator's own prediction (pure inverse crime).
        crime = np.array([emu.predict(**{k: p[k] for k in emu.param_names})
                          for p in params])
        # C1/C2 truth: the full forward model. open_water is analytical (no
        # transform_fn) — its "full model" is the analytical predictor itself,
        # so it has no surrogate gap (C0 == C1) by construction.
        has_tf = "transform_fn" in cfg
        full = (np.array([_full_model_spectrum(cfg, p) for p in params])
                if has_tf else crime.copy())
        pert = rng.multivariate_normal(np.zeros(480), Se_full, size=n)
        me = np.clip(full + pert, 0.0, 1.0)

        truth_key = np.array([p[_KEY_PARAM[stype]] for p in params])

        for cond, obs in (("C0_crime", np.clip(crime, 0, 1)),
                          ("C1_surrogate", np.clip(full, 0, 1)),
                          ("C2_model_error", me)):
            for se_label, me_arg in (("Se_inst", None), ("Se_model", True)):
                # use_priors=False: classify spectrum-only, so the measured
                # degradation is attributable to forward-model mismatch, not to
                # seasonal priors (which would independently penalise snow and
                # exclude young ice). Matches E1's no-prior baseline.
                scene = retrieve_sea_ice_batch(
                    obs, emulators=fleet, engine="vectorized", method="oe",
                    solzen=60, direct=1, use_priors=False, model_error=me_arg,
                )
                ds = scene.to_xarray()
                winners = ds["surface_type"].values
                correct = winners == stype
                acc = float(correct.mean())
                # parameter recovery on the correctly classified subset
                kp = _KEY_PARAM[stype]
                pv = ds.get(f"param_{kp}")
                if pv is not None and correct.sum() >= 4:
                    key_r2 = _r2(truth_key[correct], pv.values[correct])
                else:
                    key_r2 = float("nan")
                # median reduced chi2 proxy: cost / n_bands (OE cost is chi2)
                med_cost = float(np.nanmedian(ds["cost"].values))
                results.append(dict(
                    surface_type=stype, condition=cond, S_e=se_label,
                    n=n, accuracy=acc, key_param=kp, key_r2=key_r2,
                    median_cost=med_cost,
                ))
    return results


def summarise(results):
    print("\nPerturbed-physics validation (V2) — inverse-crime overstatement\n")
    types = sorted({r["surface_type"] for r in results})
    # Headline: classification accuracy per condition (Se_model), fleet mean.
    print("  Classification accuracy (fleet, model_error S_e):")
    print(f"    {'type':<12} {'C0 crime':>9} {'C1 surrog':>10} {'C2 mod-err':>11}")
    hdr = {}
    for stype in types:
        row = {}
        for c in ("C0_crime", "C1_surrogate", "C2_model_error"):
            m = [r for r in results if r["surface_type"] == stype
                 and r["condition"] == c and r["S_e"] == "Se_model"]
            row[c] = m[0]["accuracy"] if m else float("nan")
        hdr[stype] = row
        print(f"    {stype:<12} {row['C0_crime']:>9.0%} "
              f"{row['C1_surrogate']:>10.0%} {row['C2_model_error']:>11.0%}")
    mean = {c: np.nanmean([hdr[t][c] for t in types])
            for c in ("C0_crime", "C1_surrogate", "C2_model_error")}
    print(f"    {'MEAN':<12} {mean['C0_crime']:>9.0%} "
          f"{mean['C1_surrogate']:>10.0%} {mean['C2_model_error']:>11.0%}")

    # Key-parameter R² degradation (Se_model).
    print("\n  Best-parameter R² (correctly classified subset, model_error S_e):")
    print(f"    {'type':<12} {'param':<22} {'C0':>7} {'C1':>7} {'C2':>7}")
    for stype in types:
        cells = {}
        kp = _KEY_PARAM[stype]
        for c in ("C0_crime", "C1_surrogate", "C2_model_error"):
            m = [r for r in results if r["surface_type"] == stype
                 and r["condition"] == c and r["S_e"] == "Se_model"]
            cells[c] = m[0]["key_r2"] if m else float("nan")
        print(f"    {stype:<12} {kp:<22} {cells['C0_crime']:>7.2f} "
              f"{cells['C1_surrogate']:>7.2f} {cells['C2_model_error']:>7.2f}")

    # A7 mitigation: median cost under C2 with instrument vs model S_e.
    print("\n  A7 check — median OE cost under C2 (model error present):")
    for se in ("Se_inst", "Se_model"):
        vals = [r["median_cost"] for r in results
                if r["condition"] == "C2_model_error" and r["S_e"] == se]
        print(f"    {se:<10} median-of-medians cost = {np.nanmedian(vals):.1f}")
    print("    (instrument-only S_e should inflate cost / overconfidence; "
          "model_error S_e should temper it)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    results = run(n=args.n)
    summarise(results)
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\n  json: {args.json}")


if __name__ == "__main__":
    main()
