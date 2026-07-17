#!/usr/bin/env python3
"""Build the A7 forward-model error covariance from the field-residual library.

Calibration: SHEBA + Smith correct-model rows (is_expected & is_winner).
Hold-out:    Istomina (never used in the fit) — reports the whitened
             reduced chi-squared of its correct-model residuals under
             instrument-only vs instrument+model S_e. Honest S_e should
             bring it from >>1 toward ~1.

Writes data/model_error/field_se_v1.npz (small, committed — it is an input
to retrievals, like an emulator file, not a validation snapshot).

Usage::

    python scripts/build_model_error_covariance.py \
        [--library tests/validation_data/residual_library.npz] \
        [--out data/model_error/field_se_v1.npz] [--n-eofs 3]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "validation_data"))

from biosnicar.inverse.model_error import fit_model_error  # noqa: E402
from build_residual_library import load_residual_library  # noqa: E402

# Calibrate on two campaigns, hold a third out entirely: the held-out reduced
# chi2 is the only honest check that the covariance generalises rather than
# memorises. Istomina is the hold-out because it is the largest and the most
# out-of-distribution (late-melt / refrozen-pond), so it is the hardest test.
CAL_CAMPAIGNS = ("sheba", "smith")
HOLDOUT = "istomina"
# Must match the instrument-only default 1-sigma in optimize._run_oe, so the
# held-out chi2 here measures the same S_e a real OE retrieval would use.
SIG_INST = 0.02


def _correct_model_rows(lib, campaigns):
    sel = lib["is_expected"].astype(bool) & lib["is_winner"].astype(bool)
    sel &= np.isin(lib["campaign"], campaigns)
    return lib["residual"][sel], lib["mask"][sel]


def _reduced_chi2(residuals, masks, S_model=None):
    """Mean whitened reduced chi-squared over rows (VIS-SWIR as covered)."""
    out = []
    for r, m in zip(residuals, masks):
        y = r[m]
        S = np.diag(np.full(y.size, SIG_INST**2))
        if S_model is not None:
            S = S + S_model.dense(m)
        chi2 = float(y @ np.linalg.solve(S, y))
        out.append(chi2 / y.size)
    return float(np.mean(out)), float(np.median(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library",
                    default=str(ROOT / "tests/validation_data/residual_library.npz"))
    ap.add_argument("--out",
                    default=str(ROOT / "data/model_error/field_se_v1.npz"))
    ap.add_argument("--n-eofs", type=int, default=3)
    args = ap.parse_args()

    lib = load_residual_library(args.library)

    res_cal, mask_cal = _correct_model_rows(lib, CAL_CAMPAIGNS)
    print(f"calibration rows (correct-model, {'+'.join(CAL_CAMPAIGNS)}): "
          f"{len(res_cal)}")
    model = fit_model_error(
        res_cal, mask_cal, n_eofs=args.n_eofs,
        meta=dict(calibration_campaigns=list(CAL_CAMPAIGNS),
                  holdout_campaign=HOLDOUT,
                  population="is_expected & is_winner",
                  library=str(Path(args.library).name)),
    )
    print(f"EOF domain: {model.meta['eof_domain_bands']} bands, "
          f"{model.meta['n_eof_rows']} full-coverage rows")
    print(f"EOF sigmas: {np.sqrt(model.eof_var).round(4)}")
    vis = (np.arange(480) * 10 + 205 >= 400) & (np.arange(480) * 10 + 205 <= 1000)
    print(f"effective DOF over 60 VIS bands: {model.effective_dof(vis):.1f} "
          f"(instrument-only assumption: 60)")

    # Held-out coverage check (the A6-lite acceptance number)
    res_ho, mask_ho = _correct_model_rows(lib, (HOLDOUT,))
    mean0, med0 = _reduced_chi2(res_ho, mask_ho, None)
    mean1, med1 = _reduced_chi2(res_ho, mask_ho, model)
    print(f"\nheld-out ({HOLDOUT}, {len(res_ho)} correct-model rows) "
          f"reduced chi2 (mean/median):")
    print(f"  instrument-only S_e : {mean0:8.2f} / {med0:8.2f}")
    print(f"  + model-error S_e   : {mean1:8.2f} / {med1:8.2f}   "
          f"(honest S_e -> ~1)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
