#!/usr/bin/env python3
"""Surface-type classification checks against SHEBA and Morassutti spectra.

Tests retrieve_sea_ice() using five datasets.  IMPORTANT: label provenance
differs between datasets and must be considered when interpreting results.

Label provenance
----------------
Tests 1–2  SHEBA spring snow / summer bare ice (Grenfell & Light 2007)
           Labels are INFERRED FROM SEASON AND SPECTRAL CONTEXT — not from
           formal ice-chart (SIGRID-3) or ship-log (ASPeCt) observations.
           These are spectral-consistency checks, not held-out validation.

Test 3     ALBI melt-pond column (Grenfell & Light 2007)
           The ALBI file header identifies the column as "melt ponds" but
           provides no depth or formal classification.  Partial provenance.

Tests 4–5  Morassutti (1995) melt ponds
           Labels are DIRECTLY OBSERVED: "melt pond" recorded in field
           metadata; pond depth measured in situ.  This is the only
           genuinely independent validation in this script.

For rigorous held-out validation with formal surface-type labels, coincident
ASPeCt ship observations (Worby & Allison 1999) or NIC/AARI ice charts in
SIGRID-3 format would be needed.

Run:
    uv run python tests/validation_data/sheba_classification_validation.py
    uv run python ... --json results_classification.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

# ── paths ────────────────────────────────────────────────────────────────────
GRENFELL_DIR = Path(__file__).parent / "Grenfell_light_2007"
MORASSUTTI_DIR = Path(__file__).parent / "morassutti1995"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from biosnicar.sea_ice.retrieve import retrieve_sea_ice
from tests.validation_data.Grenfell_light_2007.validate_grenfell_light_2007 import (
    catalogue, parse_albv, parse_albi
)
from tests.validation_data.morassutti1995.validate_morassutti1995 import (
    load_data, DEPTH_BINS
)

# ── wavelength grid ───────────────────────────────────────────────────────────
WL_NM = np.arange(205, 4999, 10)          # 480-band SNICAR grid
MASK_VIS_NIR  = (WL_NM >= 400) & (WL_NM <= 1000)   # 400–1000 nm
MASK_VIS_SWIR = (WL_NM >= 400) & (WL_NM <= 2000)   # 400–2000 nm
MASK_SWIR     = (WL_NM >= 1100) & (WL_NM <= 2000)  # 1100–2000 nm only


def _interp(wl_obs, alb_obs):
    """Interpolate observed spectrum onto SNICAR grid, NaN outside range."""
    f = interp1d(wl_obs, alb_obs, kind="linear", bounds_error=False,
                 fill_value=np.nan)
    return f(WL_NM)


def _classify(obs, mask, sza, month, label="?"):
    """Run retrieve_sea_ice() and return a result dict."""
    m = mask & ~np.isnan(obs)
    if m.sum() < 5:
        return None
    r = retrieve_sea_ice(
        observed        = obs,
        wavelength_mask = m,
        solzen          = sza,
        direct          = 1,
        known_month     = month,
    )
    return {
        "label":        label,
        "surface_type": r.surface_type,
        "confidence":   float(r.confidence),
        "cost":         float(r.cost),
        "cost_per_type": {k: float(v) for k, v in r.cost_per_type.items()},
    }


def _correct(surface_type, expected_types):
    return surface_type in expected_types


# ── Test 1: Spring snow ───────────────────────────────────────────────────────

def test_spring_snow():
    """ALBV Apr–May 1998 → expect FYI_snow."""
    spring, _ = catalogue()
    rows = []
    for e in spring:
        wl, alb, _ = parse_albv(e["path"])
        if wl is None:
            continue
        obs   = _interp(wl, alb)
        month = int(e["date"][5:7])
        r = _classify(obs, MASK_VIS_NIR, e["sza"], month, e["date"])
        if r:
            r["expected"]   = ["FYI_snow"]
            r["obs_bba"]    = float(np.nanmean(obs[MASK_VIS_NIR]))
            r["correct"]    = _correct(r["surface_type"], r["expected"])
            rows.append(r)
    return rows


# ── Test 2: Summer white ice — VIS only ──────────────────────────────────────

def test_summer_white_ice_vis():
    """ALBV Aug–Sep 1998 (400–1000 nm) → expect FYI_bare or FYI_summer."""
    _, summer = catalogue()
    rows = []
    for e in summer:
        wl, alb, _ = parse_albv(e["path"])
        if wl is None:
            continue
        obs   = _interp(wl, alb)
        month = int(e["date"][5:7])
        r = _classify(obs, MASK_VIS_NIR, e["sza"], month, e["date"])
        if r:
            r["expected"]   = ["FYI_bare", "FYI_summer"]
            r["obs_bba"]    = float(np.nanmean(obs[MASK_VIS_NIR]))
            r["correct"]    = _correct(r["surface_type"], r["expected"])
            rows.append(r)
    return rows


# ── Test 3: Summer white ice — VIS + SWIR ────────────────────────────────────

def test_summer_white_ice_vis_swir():
    """ALBV+ALBI white-ice column (400–2000 nm) → expect FYI_bare or FYI_summer."""
    _, summer = catalogue()
    rows = []
    for e in summer:
        tag  = e["date"].replace("1998-", "").replace("-", "")
        albi = GRENFELL_DIR / f"ICEDATA_OPTICS_SPECALB_ALBI{tag}.CSV"
        if not albi.exists():
            continue
        wl_v, alb_v, _ = parse_albv(e["path"])
        wl_i, wi_alb, _, _ = parse_albi(albi)
        if wl_v is None or wl_i is None:
            continue
        obs      = _interp(wl_v, alb_v)
        # Overwrite SWIR (1100–2000 nm) with ALBI white-ice column
        obs_wi   = _interp(wl_i, wi_alb)
        ir_mask  = (WL_NM >= 1100) & (WL_NM <= 2000) & ~np.isnan(obs_wi)
        obs[ir_mask] = obs_wi[ir_mask]
        month = int(e["date"][5:7])
        mask  = MASK_VIS_SWIR & ~np.isnan(obs)
        r = _classify(obs, mask, e["sza"], month, e["date"])
        if r:
            r["expected"] = ["FYI_bare", "FYI_summer"]
            r["obs_bba"]  = float(np.nanmean(obs[MASK_VIS_NIR]))
            r["correct"]  = _correct(r["surface_type"], r["expected"])
            rows.append(r)
    return rows


# ── Test 4: ALBI melt ponds (IR only) ────────────────────────────────────────

def test_albi_melt_ponds():
    """ALBI melt-pond column (1100–2000 nm) → expect FYI_pond."""
    _, summer = catalogue()
    rows = []
    for e in summer:
        tag  = e["date"].replace("1998-", "").replace("-", "")
        albi = GRENFELL_DIR / f"ICEDATA_OPTICS_SPECALB_ALBI{tag}.CSV"
        if not albi.exists():
            continue
        wl_i, _, mp_alb, _ = parse_albi(albi)
        if wl_i is None or np.nanmean(mp_alb) < 0.01:
            continue
        obs   = _interp(wl_i, mp_alb)
        month = int(e["date"][5:7])
        r = _classify(obs, MASK_SWIR, e["sza"], month, e["date"])
        if r:
            r["expected"] = ["FYI_pond"]
            r["obs_nir"]  = float(np.nanmean(obs[MASK_SWIR & ~np.isnan(obs)]))
            r["correct"]  = _correct(r["surface_type"], r["expected"])
            rows.append(r)
    return rows


# ── Test 5: Morassutti melt ponds ─────────────────────────────────────────────

def test_morassutti_ponds():
    """Morassutti (1995) 6-band melt pond observations → expect FYI_pond.

    504 records, each with 6 × 100 nm bands (400–1000 nm).
    Band midpoints (450, 550, …, 950 nm) are interpolated onto the SNICAR grid.
    Known_month=7 (July — melt season in Canadian Arctic).
    """
    df = load_data()
    band_cols   = ["b1_400_500", "b2_500_600", "b3_600_700",
                   "b4_700_800", "b5_800_900", "b6_900_1000"]
    band_mids   = np.array([450, 550, 650, 750, 850, 950], dtype=float)
    sza_typical = 60   # representative Arctic summer mid-day

    rows = []
    for i, row in df.iterrows():
        vals = row[band_cols].values.astype(float)
        if np.any(np.isnan(vals)):
            continue
        obs     = _interp(band_mids, vals)
        # Rely on the 6 band midpoints; mask to 400–1000 nm
        mask    = MASK_VIS_NIR & ~np.isnan(obs)
        r = _classify(obs, mask, sza_typical, 7, f"pond_depth={row['pond_depth_m']:.2f}m")
        if r:
            r["expected"]     = ["FYI_pond"]
            r["pond_depth_m"] = float(row["pond_depth_m"])
            r["obs_bba"]      = float(row["bba_400_1000"])
            r["correct"]      = _correct(r["surface_type"], r["expected"])
            rows.append(r)
    return rows


# ── Accuracy summary ──────────────────────────────────────────────────────────

def summarise(test_name, rows, expected_types):
    n       = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    mean_conf = np.mean([r["confidence"] for r in rows]) if rows else float("nan")
    label   = " or ".join(expected_types)
    print(f"\n  {test_name}")
    print(f"  {'Date/record':28s}  BBA     {'Result':16s}  Conf   OK?")
    print("  " + "-" * 72)
    for r in rows:
        lbl = r.get("label", "?")
        bba = r.get("obs_bba", r.get("obs_nir", float("nan")))
        ok  = "✓" if r["correct"] else "✗"
        print(f"  {lbl:28s}  {bba:.3f}   {r['surface_type']:16s}  {r['confidence']:.3f}  {ok}")
    print(f"  Accuracy: {correct}/{n}  ({100*correct/n:.0f}%)   mean confidence: {mean_conf:.3f}")
    return {"test": test_name, "n": n, "correct": correct,
            "accuracy": correct / n if n else float("nan"),
            "mean_confidence": float(mean_conf)}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", metavar="FILE",
                        help="Save classification results to JSON")
    args = parser.parse_args()

    print("=" * 80)
    print("SHEBA + MORASSUTTI SURFACE-TYPE CLASSIFICATION VALIDATION")
    print("retrieve_sea_ice() with known_month seasonal priors")
    print("=" * 80)

    tests = [
        ("1. Spring snow (ALBV, Apr–May)      expect: FYI_snow",
         test_spring_snow,          ["FYI_snow"]),
        ("2. Summer white ice VIS only        expect: FYI_bare/FYI_summer",
         test_summer_white_ice_vis, ["FYI_bare", "FYI_summer"]),
        ("3. Summer white ice VIS+SWIR        expect: FYI_bare/FYI_summer",
         test_summer_white_ice_vis_swir, ["FYI_bare", "FYI_summer"]),
        ("4. ALBI melt ponds (SWIR only)      expect: FYI_pond",
         test_albi_melt_ponds,      ["FYI_pond"]),
        ("5. Morassutti ponds (400–1000 nm)   expect: FYI_pond",
         test_morassutti_ponds,     ["FYI_pond"]),
    ]

    all_summaries = []
    all_rows_by_test = {}
    for name, fn, expected in tests:
        print(f"\nRunning: {name.split('expect:')[0].strip()}…")
        rows = fn()
        s = summarise(name, rows, expected)
        all_summaries.append(s)
        all_rows_by_test[name] = rows

    # Overall table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  {'Test':50s}  {'N':>5}  {'Correct':>8}  {'Accuracy':>9}  {'Mean conf':>10}")
    print("  " + "-" * 78)
    total_n = total_correct = 0
    for s in all_summaries:
        name = s["test"].split("expect:")[0].strip()[:50]
        print(f"  {name:50s}  {s['n']:>5}  {s['correct']:>8}  "
              f"{s['accuracy']*100:>8.0f}%  {s['mean_confidence']:>10.3f}")
        total_n       += s["n"]
        total_correct += s["correct"]
    print("  " + "-" * 78)
    print(f"  {'TOTAL':50s}  {total_n:>5}  {total_correct:>8}  "
          f"{100*total_correct/total_n:>8.0f}%")
    print()
    print("  Notes:")
    print("  • Tests 2 and 3 compare VIS-only vs VIS+SWIR for summer bare ice")
    print("  • Test 4 (SWIR-only ponds) is intentionally challenging — no VIS signal")
    print("  • Test 5 uses 6 coarse bands (100 nm each); richer spectra would improve accuracy")
    print("  • All classifications use known_month to apply seasonal temperature priors")

    if args.json:
        out = {"summaries": all_summaries,
               "rows_by_test": {k: v for k, v in all_rows_by_test.items()}}
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
