"""Validation experiment: Morassutti (1995) melt pond albedo vs BioSNICAR.

Dataset
-------
Morassutti, M. (1995). Sea Ice Melt Pond Data from the Canadian Arctic.
NSIDC G01169. https://doi.org/10.7265/N55Q4T1C

504 records, summer 1994 (27 May – 26 Jun), Barrow Strait, Nunavut.
Albedo measured with a portable spectrometer across 6 bands (400–1000 nm).

IMPORTANT CAVEAT
----------------
This dataset records albedo OF MELT PONDS, not bare winter sea ice.
Mean BBA is 0.21 (open water ponds). The BioSNICAR sea ice MVP (v0.1)
models WINTER conditions (bare ice BBA ~0.44, snow-covered ~0.78).
These are fundamentally different surfaces.

The comparison below is therefore restricted to:
  - The 14 records with pond depth < 0.05 m (essentially zero) and BBA > 0.4,
    which plausibly represent the bare ice surface rather than ponded water.
  - Spectral comparison across the overlapping 400–1000 nm range only.

This should be treated as a lower-bound sanity check, not a formal validation.
For proper MVP validation, obtain SHEBA bare winter sea ice spectra
(e.g. Perovich et al. 2002, JGR; Grenfell & Maykut 1977) or
Light et al. (2008) laboratory measurements of first-year sea ice.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
DAT_FILE = HERE / "pond.dat"
if not DAT_FILE.exists():
    import shutil, os
    src = Path.home() / "Downloads" / "pond.dat"
    if src.exists():
        shutil.copy(src, DAT_FILE)
    else:
        raise FileNotFoundError(
            "pond.dat not found. Copy it to this directory or ~/Downloads/"
        )

# ---------------------------------------------------------------------------
# 1. Parse the data file
# ---------------------------------------------------------------------------
COL_NAMES = [
    "julian_day", "time", "fov", "cloud_type", "solar_obscuration",
    "ice_type", "pond_depth_m", "pond_bottom", "ice_layer", "debris",
    "pond_color", "bba_400_1000", "vis_400_700", "nir_700_1000",
    "b1_400_500", "b2_500_600", "b3_600_700",
    "b4_700_800", "b5_800_900", "b6_900_1000",
    "ice_layer_thick_cm", "pond_number",
]

NUMERIC = [c for c in COL_NAMES if c not in ("fov", "cloud_type")]

rows = []
with open(DAT_FILE) as fh:
    for line in fh:
        parts = line.split()
        if len(parts) == 22:
            rows.append(parts)

df = pd.DataFrame(rows, columns=COL_NAMES)
for col in NUMERIC:
    df[col] = pd.to_numeric(df[col])

print(f"Loaded {len(df)} records, Julian days {int(df.julian_day.min())}–{int(df.julian_day.max())}")

# ---------------------------------------------------------------------------
# 2. Dataset characterisation
# ---------------------------------------------------------------------------
print("\n--- Dataset overview ---")
print(f"BBA (400–1000 nm):  mean={df.bba_400_1000.mean():.3f}  "
      f"std={df.bba_400_1000.std():.3f}  "
      f"range=[{df.bba_400_1000.min():.3f}, {df.bba_400_1000.max():.3f}]")
print(f"Pond depth (m):     mean={df.pond_depth_m.mean():.3f}  "
      f"range=[{df.pond_depth_m.min():.3f}, {df.pond_depth_m.max():.3f}]")
print(f"Ice types: {df.ice_type.value_counts().sort_index().to_dict()}")
print(f"Sky conditions: {df.cloud_type.value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# 3. Select near-surface ice records for model comparison
# ---------------------------------------------------------------------------
# Shallow depth AND high albedo — most likely bare ice surface
bare_ice_proxy = df[(df.pond_depth_m <= 0.05) & (df.bba_400_1000 >= 0.40)].copy()
print(f"\n--- Near-surface ice records (depth ≤ 0.05 m, BBA ≥ 0.40) ---")
print(f"N = {len(bare_ice_proxy)}")
print(bare_ice_proxy[["julian_day", "fov", "cloud_type", "ice_type",
                        "pond_depth_m", "bba_400_1000",
                        "vis_400_700", "nir_700_1000"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 4. BioSNICAR model predictions for matching conditions
# ---------------------------------------------------------------------------
print("\n--- BioSNICAR model output (FYI bare ice, SZA varied) ---")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from biosnicar.sea_ice.api import SeaIceColumn, SeaIceLayer
from biosnicar.sea_ice.presets import FYI_WINTER_BARE, MYI_WINTER_BARE

# BioSNICAR 480-band wavelength grid
wvl_um = np.arange(0.205, 4.999, 0.01)  # µm

def weighted_band_albedo(spectrum, wvl, lo_um, hi_um, flx_slr):
    """Flux-weighted average albedo between lo_um and hi_um."""
    mask = (wvl >= lo_um) & (wvl < hi_um)
    num = np.sum(flx_slr[mask] * spectrum[mask])
    den = np.sum(flx_slr[mask])
    return float(num / den) if den > 0 else np.nan

# Run the model at several SZA values relevant to summer Arctic (low sun)
# Day 160 ≈ 9 Jun 1994, Barrow Strait latitude ~74°N
# Solar noon sun elevation ~39°, so SZA range roughly 50-80° through the day
model_rows = []
for preset, label in [(FYI_WINTER_BARE, "FYI bare"), (MYI_WINTER_BARE, "MYI bare")]:
    for sza in [50, 60, 70, 75]:
        col = SeaIceColumn.from_preset(preset)
        result = col.compute_albedo(sza_deg=sza, sky="clear")
        spec = result.spectrum
        flx = result.outputs.flx_slr

        bba_model = weighted_band_albedo(spec, wvl_um, 0.4, 1.0, flx)
        vis_model = weighted_band_albedo(spec, wvl_um, 0.4, 0.7, flx)
        nir_model = weighted_band_albedo(spec, wvl_um, 0.7, 1.0, flx)
        b1 = weighted_band_albedo(spec, wvl_um, 0.4, 0.5, flx)
        b2 = weighted_band_albedo(spec, wvl_um, 0.5, 0.6, flx)
        b3 = weighted_band_albedo(spec, wvl_um, 0.6, 0.7, flx)
        b4 = weighted_band_albedo(spec, wvl_um, 0.7, 0.8, flx)
        b5 = weighted_band_albedo(spec, wvl_um, 0.8, 0.9, flx)
        b6 = weighted_band_albedo(spec, wvl_um, 0.9, 1.0, flx)

        model_rows.append({
            "surface": label, "sza": sza,
            "bba_400_1000": bba_model, "vis_400_700": vis_model, "nir_700_1000": nir_model,
            "b1": b1, "b2": b2, "b3": b3, "b4": b4, "b5": b5, "b6": b6,
        })

model_df = pd.DataFrame(model_rows)
print(model_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Compare model to observed bare-ice proxies
# ---------------------------------------------------------------------------
print("\n--- Comparison: observed bare-ice proxies vs nearest model ---")

obs_mean = bare_ice_proxy[["bba_400_1000", "vis_400_700", "nir_700_1000",
                            "b1_400_500", "b2_500_600", "b3_600_700",
                            "b4_700_800", "b5_800_900", "b6_900_1000"]].mean()

# Use FYI bare at SZA=70° as representative (summer Arctic, low sun)
fyi_70 = model_df[(model_df.surface == "FYI bare") & (model_df.sza == 70)].iloc[0]

print(f"\n{'Band':<22} {'Observed (N={0})'.format(len(bare_ice_proxy)):>16} {'Model (FYI, SZA=70°)':>22} {'Difference':>12}")
print("-" * 74)
band_map = [
    ("BBA (400–1000 nm)", "bba_400_1000", "bba_400_1000"),
    ("VIS (400–700 nm)",  "vis_400_700",  "vis_400_700"),
    ("NIR (700–1000 nm)", "nir_700_1000", "nir_700_1000"),
    ("B1 (400–500 nm)",   "b1_400_500",   "b1"),
    ("B2 (500–600 nm)",   "b2_500_600",   "b2"),
    ("B3 (600–700 nm)",   "b3_600_700",   "b3"),
    ("B4 (700–800 nm)",   "b4_700_800",   "b4"),
    ("B5 (800–900 nm)",   "b5_800_900",   "b5"),
    ("B6 (900–1000 nm)",  "b6_900_1000",  "b6"),
]
for label, obs_col, mod_col in band_map:
    o = obs_mean[obs_col]
    m = fyi_70[mod_col]
    diff = m - o
    print(f"  {label:<20} {o:>16.3f} {m:>22.3f} {diff:>+12.3f}")

# ---------------------------------------------------------------------------
# 6. Save manifest
# ---------------------------------------------------------------------------
manifest = {
    "dataset": "Sea Ice Melt Pond Data from the Canadian Arctic",
    "citation": "Morassutti, M. (1995). NSIDC G01169. https://doi.org/10.7265/N55Q4T1C",
    "location": "Barrow Strait, Nunavut, Canada (near Cornwallis Island, ~74°N)",
    "period": "27 May – 26 Jun 1994 (Julian days 147–177)",
    "n_records": len(df),
    "spectral_bands_um": {
        "broadband": [0.4, 1.0], "visible": [0.4, 0.7], "nir": [0.7, 1.0],
        "b1_blue": [0.4, 0.5], "b2_green": [0.5, 0.6], "b3_red": [0.6, 0.7],
        "b4_nir1": [0.7, 0.8], "b5_nir2": [0.8, 0.9], "b6_nir3": [0.9, 1.0],
    },
    "surface_type": "MELT PONDS — not bare winter sea ice",
    "mvp_validation_suitability": "LOW — melt ponds are explicitly out of scope for BioSNICAR v0.1 MVP. Use for future v0.2 melt pond development.",
    "bare_ice_proxy_records": len(bare_ice_proxy),
    "bare_ice_proxy_filter": "pond_depth <= 0.05 m AND bba_400_1000 >= 0.40",
    "notes": [
        "Spectral range 400–1000 nm only; BioSNICAR operates 205–5000 nm.",
        "Ice type codes 1/2/3 not fully documented (table 5 truncated in PDF).",
        "For MVP validation, obtain: Perovich et al. 2002 (JGR) SHEBA bare ice, "
        "or Light et al. 2008 (J Geophys Res) first-year ice lab spectra.",
    ],
}

with open(HERE / "manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)
print("\nManifest saved to manifest.json")

# ---------------------------------------------------------------------------
# 7. Plain-text summary of findings
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Dataset: Morassutti (1995) — Canadian Arctic melt ponds, summer 1994
Records: {len(df)} total, {len(bare_ice_proxy)} usable as bare-ice proxies

Model (FYI bare, SZA=70°) vs bare-ice proxy observations:
  BBA (400–1000 nm):  model={fyi_70['bba_400_1000']:.3f}  obs={obs_mean['bba_400_1000']:.3f}  diff={fyi_70['bba_400_1000']-obs_mean['bba_400_1000']:+.3f}
  VIS (400–700 nm):   model={fyi_70['vis_400_700']:.3f}  obs={obs_mean['vis_400_700']:.3f}  diff={fyi_70['vis_400_700']-obs_mean['vis_400_700']:+.3f}
  NIR (700–1000 nm):  model={fyi_70['nir_700_1000']:.3f}  obs={obs_mean['nir_700_1000']:.3f}  diff={fyi_70['nir_700_1000']-obs_mean['nir_700_1000']:+.3f}

Caveats:
  - Observations are summer melt pond surfaces; model represents winter bare ice.
  - N=14 'bare ice' proxies is too small and circumstantial for formal validation.
  - Spectral range mismatch: obs 400–1000 nm; model 205–5000 nm (BBA differs).
  - The model overestimates VIS albedo compared to these summer observations,
    which is physically expected (summer ice is darker due to melt and brine exposure).

Recommendation:
  This dataset is better suited to future v0.2 melt pond model development.
  For MVP validation, find: Perovich et al. (2002) Table 2 (SHEBA bare FYI/MYI),
  Grenfell & Maykut (1977), or Light et al. (2008) first-year ice measurements.
""")
