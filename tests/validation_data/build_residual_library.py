#!/usr/bin/env python3
"""Build the field-residual library (A7 step 1).

For every real field spectrum in the three spectral validation campaigns
(SHEBA/Grenfell 1998, Smith/MOSAiC 2020, Istomina/IceArc 2012), fit the
retrieval fleet exactly as the campaign validation scripts do, and store the
post-fit residual vectors ``observed - predicted`` on the 480-band model grid.

These residuals are the empirical basis for the A7 forward-model error term
in the OE measurement covariance S_e: the misfit that persists after the
forward model has done its best against a real measurement. Two residual rows
are stored per spectrum where labels allow:

* the **winner** fit (whatever the classifier chose), and
* each **expected-type** fit (the campaign label's candidate set) — this is
  the row that characterises *correct-model* error and is the primary A7
  input. Rows carry ``is_winner`` / ``is_expected`` flags so analyses can
  select either population.

Honest scope notes (read before using the library):

* Residuals bundle forward-model error with footprint heterogeneity, SZA
  estimates and calibration effects. That bundle is exactly what S_e should
  represent for real retrievals, but it is not "pure" model error.
* Post-fit residuals slightly understate model error (the optimiser absorbs
  some misfit into 3-6 free parameters over ~60-480 bands).
* Campaign labels differ in quality: SHEBA labels are season-inferred,
  Smith positions are operator-labelled snow, Istomina comments are coarse
  field notes (see istomina_retrieval_validation.py).
* No field residuals exist for young_ice or open_water — S_e for those
  types must stay instrument-only or pool from neighbours, explicitly.
* Do NOT calibrate S_e and then quote validation numbers from the same
  campaign — split (e.g. calibrate SHEBA+Smith, hold out Istomina).

The output artifact is regenerable and git-ignored per the validation-data
source-of-truth policy. Regenerate with::

    python tests/validation_data/build_residual_library.py \
        [--out tests/validation_data/residual_library.npz] \
        [--campaigns sheba smith istomina]
"""

import argparse
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from biosnicar.sea_ice.retrieve import retrieve_sea_ice  # noqa: E402
from biosnicar.sea_ice.spectral_utils import (  # noqa: E402
    MODEL_WAVELENGTHS_NM, resample_to_model_grid,
)

WL_NM = MODEL_WAVELENGTHS_NM
VIS = (WL_NM >= 400) & (WL_NM <= 1000)
LIBRARY_VERSION = 1


# ── shared fitting + record assembly ─────────────────────────────────────────

def _fit(obs, mask, sza, direct, month):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return retrieve_sea_ice(
            observed=np.nan_to_num(obs), wavelength_mask=mask,
            solzen=float(np.clip(sza, 20, 89)), direct=int(direct),
            known_month=int(month),
        )


def _rows_for_spectrum(campaign, sid, date, label, expected, obs, mask,
                       sza, direct, month):
    """Fit the fleet once; emit one row per (winner ∪ expected) fit."""
    r = _fit(obs, mask, sza, direct, month)
    fit_types = {r.surface_type} | (set(expected) & set(r.all_fits))
    rows = []
    for t in sorted(fit_types):
        pred = np.asarray(r.all_fits[t].predicted_albedo, dtype=np.float32)
        resid = np.where(mask, obs - pred, np.nan).astype(np.float32)
        vis_m = mask & VIS
        rows.append(dict(
            campaign=campaign, spectrum_id=sid, date=date, label=label,
            expected_set="|".join(expected), fit_type=t,
            is_winner=(t == r.surface_type),
            is_expected=(t in expected),
            winner_type=r.surface_type, confidence=float(r.confidence),
            sza=float(sza), direct=int(direct), month=int(month),
            n_bands=int(mask.sum()),
            rms_vis=(float(np.sqrt(np.nanmean(resid[vis_m] ** 2)))
                     if vis_m.any() else np.nan),
            residual=resid, mask=mask.astype(bool),
        ))
    return rows


# ── campaign collectors (settings mirror the validation scripts) ─────────────

def collect_sheba():
    """SHEBA: spring snow (VIS) + summer bare ice (VIS, extended to VIS+SWIR
    where a paired ALBI file exists).

    Mirrors sheba_classification_validation.py: VIS from ALBV; for summer
    dates with a paired ALBI file the white-ice SWIR column (1100-2000 nm) is
    spliced on, giving full-range rows — the on-disk SWIR data the A7 SWIR
    covariance term needs. direct=1, known_month from date, noon SZA at 76N.
    """
    from Grenfell_light_2007.validate_grenfell_light_2007 import (
        catalogue, parse_albv, parse_albi,
    )
    GRENFELL = HERE / "Grenfell_light_2007"
    SWIR = (WL_NM >= 1100) & (WL_NM <= 2000)
    spring, summer = catalogue()
    rows = []
    for entries, expected, label, splice_swir in (
        (spring, ("FYI_snow",), "spring_snow", False),
        (summer, ("FYI_bare", "FYI_summer"), "summer_bare_ice", True),
    ):
        for e in entries:
            wl, alb, _ = parse_albv(e["path"])
            if wl is None:
                continue
            obs = interp1d(wl, alb, kind="linear", bounds_error=False,
                           fill_value=np.nan)(WL_NM)
            mask = VIS & np.isfinite(obs)
            # Splice the ALBI white-ice SWIR column onto summer dates.
            if splice_swir:
                tag = e["date"].replace("1998-", "").replace("-", "")
                albi = GRENFELL / f"ICEDATA_OPTICS_SPECALB_ALBI{tag}.CSV"
                if albi.exists():
                    wl_i, wi_alb, _, _ = parse_albi(albi)
                    if wl_i is not None:
                        obs_i = interp1d(wl_i, wi_alb, kind="linear",
                                         bounds_error=False,
                                         fill_value=np.nan)(WL_NM)
                        ir = SWIR & np.isfinite(obs_i)
                        obs[ir] = obs_i[ir]
                        mask = mask | ir
            if mask.sum() < 5:
                continue
            rows += _rows_for_spectrum(
                "sheba", e["date"], e["date"], label, expected,
                obs.astype(np.float32), mask, e["sza"], 1,
                int(e["date"][5:7]),
            )
    return rows


def collect_smith():
    """Smith/MOSAiC snow positions (expect snow-like: FYI_snow|FYI_summer).

    Mirrors smith_retrieval_validation.py: linear resample, >=30 finite
    bands, per-record sky-code direct flag, known_month, SZA clipped 20-89.
    """
    from run_global_validation import load_smith
    from smith_retrieval_validation import _direct_flag
    rows = []
    for rec in load_smith():
        obs = resample_to_model_grid(rec.alb, rec.wl_nm, method="linear")
        mask = np.isfinite(obs)
        if mask.sum() < 30:
            continue
        rows += _rows_for_spectrum(
            "smith", f"{rec.date}_{rec.notes or rec.n_spectra}", rec.date,
            "summer_snow", ("FYI_snow", "FYI_summer"),
            obs.astype(np.float32), mask, rec.sza, _direct_flag(rec.sky),
            int(rec.date[5:7]),
        )
    return rows


def collect_istomina():
    """Istomina/IceArc surface spectra (coarse field-note labels).

    Mirrors istomina_retrieval_validation.py: finite & [0,1] mask, >=100
    bands with >=20 in the VIS, per-station SZA/direct/month. Expected sets
    are coarse (labels are operator notes — see that script's docstring):
    ice -> the melt-season ice cluster; ponds are FOV-noisy so only the
    winner row is stored for them (expected set empty).
    """
    from istomina_retrieval_validation import load_istomina
    expected_by_label = {
        "ice": ("FYI_bare", "FYI_summer", "MYI_bare", "FYI_snow"),
        "open_pond": (),
        "frozen_pond": (),
    }
    rows = []
    for r in load_istomina():
        obs = resample_to_model_grid(r["alb"], r["wl_nm"], method="linear")
        mask = np.isfinite(obs) & (obs >= 0) & (obs <= 1.0)
        if mask.sum() < 100 or (mask & VIS).sum() < 20:
            continue
        rows += _rows_for_spectrum(
            "istomina", f"{r['station']}_{r['sid']}", r["station"],
            r["surface"], expected_by_label.get(r["surface"], ()),
            np.clip(obs, 0, 1).astype(np.float32), mask,
            r["sza"], r["direct"], r["month"],
        )
    return rows


COLLECTORS = {
    "sheba": collect_sheba,
    "smith": collect_smith,
    "istomina": collect_istomina,
}


# ── save / load ──────────────────────────────────────────────────────────────

_STR_FIELDS = ("campaign", "spectrum_id", "date", "label", "expected_set",
               "fit_type", "winner_type")
_NUM_FIELDS = ("is_winner", "is_expected", "confidence", "sza", "direct",
               "month", "n_bands", "rms_vis")


def save_library(rows, path):
    arrays = {
        "library_version": np.array(LIBRARY_VERSION),
        "created_utc": np.array(datetime.now(timezone.utc).isoformat()),
        "wavelength_nm": WL_NM.astype(np.float32),
        "residual": np.stack([r["residual"] for r in rows]),
        "mask": np.stack([r["mask"] for r in rows]),
    }
    for f in _STR_FIELDS:
        arrays[f] = np.array([r[f] for r in rows])
    for f in _NUM_FIELDS:
        arrays[f] = np.array([r[f] for r in rows])
    np.savez_compressed(path, **arrays)


def load_residual_library(path):
    """Load the library as ``{field: array}`` (rows share the first axis)."""
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


# ── summary ──────────────────────────────────────────────────────────────────

def summarise(rows):
    print(f"\nField-residual library — {len(rows)} residual rows")
    for campaign in COLLECTORS:
        sub = [r for r in rows if r["campaign"] == campaign]
        if not sub:
            continue
        n_spec = len({r["spectrum_id"] for r in sub})
        print(f"\n  {campaign}: {n_spec} spectra, {len(sub)} rows")
        for pop, flag in (("winner", "is_winner"), ("expected", "is_expected")):
            p = [r for r in sub if r[flag]]
            if not p:
                continue
            rms = np.array([r["rms_vis"] for r in p], dtype=float)
            rms = rms[np.isfinite(rms)]
            print(f"    {pop:<9} n={len(p):3d}  VIS RMS median "
                  f"{np.median(rms):.4f}  p90 {np.percentile(rms, 90):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "residual_library.npz"))
    ap.add_argument("--campaigns", nargs="+", choices=list(COLLECTORS),
                    default=list(COLLECTORS))
    args = ap.parse_args()

    rows = []
    for c in args.campaigns:
        print(f"collecting {c} ...", flush=True)
        rows += COLLECTORS[c]()
    if not rows:
        raise SystemExit("no residual rows collected — is the campaign "
                         "data downloaded? (see tests/validation_data/README.md)")
    summarise(rows)
    save_library(rows, args.out)
    print(f"\n  saved: {args.out}")


if __name__ == "__main__":
    main()
