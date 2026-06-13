#!/usr/bin/env python3
"""Independent retrieval validation against Smith et al. (2021) MOSAiC spectra.

This is the first test of retrieve_sea_ice() against a campaign DISTINCT from
the SHEBA/Grenfell data the system was developed and tuned on:

  * different year   — MOSAiC Leg 4, 2020 (vs SHEBA 1998)
  * different site   — ~82 N (vs SHEBA ~76 N)
  * different sensor — ASD FieldSpec, 350-2500 nm @ 1 nm (vs Grenfell
                       portable spectrometer, 400-1000 nm)
  * full SWIR        — first real-data exercise of the FYI_snow `vis_swir`
                       classification band mask (SHEBA ALBV is VIS-only)

44 snow-surface transect positions (surface_type='S', change-in-incident
<= 10%), each averaged across positions per the published processing, are
loaded via the existing global-validation loader.  Each observation is
resampled onto the 480-band model grid with the project's own
resample_to_model_grid (D3), then inverted with retrieve_sea_ice() using the
per-record solar zenith angle (from UTC + ship lat/lon), the per-record
illumination flag (diffuse when the solar disk was not visible — MOSAiC
"CO-SDNV" — which is the majority of this dataset), and known_month.

Honesty notes
-------------
  * These are SNOW positions, so the expected optical class is FYI_snow.
    Summer (Jun-Sep) snow at 82 N is heavily metamorphosed/melting and can
    be spectrally close to a melting surface-scattering layer, so FYI_summer
    is reported as a "snow-like / bright melting" near-miss, not silently
    counted as correct.
  * The FYI_snow emulator is a winter/spring snow parameterisation; large
    SWIR residuals on melting summer snow are expected and are the point of
    the test.
  * Labels are field surface-type codes (independent of our model), but the
    SZA and the diffuse/direct assignment are inferred — a genuine but not
    perfect hold-out.

Usage::

    python scripts/smith_retrieval_validation.py [--json out.json] [--out figdir]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "validation_data"))

from run_global_validation import load_smith  # noqa: E402

from biosnicar.sea_ice.retrieve import retrieve_sea_ice  # noqa: E402
from biosnicar.sea_ice.spectral_utils import (  # noqa: E402
    MODEL_WAVELENGTHS_NM, resample_to_model_grid,
)

TYPE_COLOR = {
    "FYI_snow": "#4575b4", "FYI_summer": "#fdae61", "FYI_bare": "#1b7837",
    "MYI_bare": "#762a83", "FYI_pond": "#35978f", "young_ice": "#d73027",
    "open_water": "#252525",
}
WL_UM = MODEL_WAVELENGTHS_NM / 1000.0


def _direct_flag(sky):
    """1 if the solar disk was available (direct beam), 0 if diffuse.

    MOSAiC nomenclature: SD = solar disk, NV = not visible, V = visible,
    CO = completely overcast.  Disk-not-visible / fog / overcast → diffuse.
    """
    s = (sky or "").lower()
    if "sdnv" in s or "fog" in s or "mist" in s or s.strip() in ("", "nan"):
        return 0
    if "co" in s and "sdv" not in s:
        return 0
    return 1


def run():
    rows = []
    for rec in load_smith():
        obs = resample_to_model_grid(rec.alb, rec.wl_nm, method="linear")
        mask = np.isfinite(obs)
        if mask.sum() < 30:
            continue
        month = int(rec.date[5:7])
        direct = _direct_flag(rec.sky)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = retrieve_sea_ice(
                observed=np.nan_to_num(obs), wavelength_mask=mask,
                solzen=float(np.clip(rec.sza, 20, 89)), direct=direct,
                known_month=month,
            )
        vis = mask & (MODEL_WAVELENGTHS_NM >= 400) & (MODEL_WAVELENGTHS_NM <= 1000)
        rms_vis = float(np.sqrt(np.mean((r.predicted_albedo[vis] - obs[vis]) ** 2)))
        rms_full = float(np.sqrt(np.mean((r.predicted_albedo[mask] - obs[mask]) ** 2)))
        rows.append(dict(
            date=rec.date, sza=float(rec.sza), direct=direct, sky=rec.sky,
            n_pos=rec.n_spectra, obs=obs, mask=mask, result=r,
            rms_vis=rms_vis, rms_full=rms_full,
            stype=r.surface_type, conf=float(r.confidence),
            flags=[k for k, v in r.quality_flag_description().items() if v],
        ))
    return rows


def summarise(rows):
    n = len(rows)
    snow = sum(r["stype"] == "FYI_snow" for r in rows)
    snowlike = sum(r["stype"] in ("FYI_snow", "FYI_summer") for r in rows)
    import collections
    breakdown = collections.Counter(r["stype"] for r in rows)
    rms_vis = np.array([r["rms_vis"] for r in rows])
    rms_full = np.array([r["rms_full"] for r in rows])
    n_diffuse = sum(r["direct"] == 0 for r in rows)

    print(f"\nSmith et al. (2021) MOSAiC — independent retrieval validation")
    print(f"  {n} snow positions; {n_diffuse} diffuse / {n - n_diffuse} direct")
    print(f"  classified FYI_snow (strict)        : {snow}/{n} ({snow/n:.0%})")
    print(f"  classified snow-like (snow+summer)  : {snowlike}/{n} ({snowlike/n:.0%})")
    print(f"  class breakdown: {dict(breakdown)}")
    print(f"  VIS RMS  (400-1000nm): median {np.median(rms_vis):.3f}  "
          f"p90 {np.percentile(rms_vis,90):.3f}  max {rms_vis.max():.3f}")
    print(f"  FULL RMS (350-2400nm): median {np.median(rms_full):.3f}  "
          f"p90 {np.percentile(rms_full,90):.3f}  max {rms_full.max():.3f}")
    flagged = sum(bool(r["flags"]) for r in rows)
    poor = sum("poor_fit" in r["flags"] for r in rows)
    print(f"  quality-flagged: {flagged}/{n}   poor_fit: {poor}/{n}")
    return dict(n=n, snow_strict=snow, snow_like=snowlike,
                breakdown=dict(breakdown),
                rms_vis_median=float(np.median(rms_vis)),
                rms_full_median=float(np.median(rms_full)),
                n_diffuse=n_diffuse, flagged=flagged, poor_fit=poor)


def plot(rows, out):
    ncols = 6
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for ax, row in zip(axes.ravel(), rows):
        res, m = row["result"], row["mask"]
        col = TYPE_COLOR.get(res.surface_type, "k")
        ax.plot(WL_UM[m], row["obs"][m], "-", color="0.25", lw=1.2)
        ax.plot(WL_UM[m], res.predicted_albedo[m], "--", color=col, lw=1.1)
        sun = "☀" if row["direct"] else "☁"
        ax.set_title(f"{row['date']} {sun}  {res.surface_type}",
                     fontsize=7.5, color=col)
        ax.text(0.04, 0.05,
                f"conf {row['conf']:.2f}\nRMS_vis {row['rms_vis']:.3f}\n"
                f"RMS_full {row['rms_full']:.3f}",
                transform=ax.transAxes, fontsize=5.5, va="bottom",
                bbox=dict(fc="white", alpha=0.8, ec="0.8"))
        ax.set_xlim(0.35, 2.4)
        ax.set_ylim(0, 1.05)
    for ax in axes.ravel()[len(rows):]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("wavelength (µm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("albedo")
    n_snow = sum(r["stype"] == "FYI_snow" for r in rows)
    fig.suptitle(
        f"Smith 2021 MOSAiC snow — observed vs retrieved (independent hold-out; "
        f"FYI_snow {n_snow}/{len(rows)}, ☀ direct / ☁ diffuse)", fontsize=12)
    fig.text(0.5, 0.004,
             "ASD 350–2400 nm resampled to the model grid; per-record SZA and "
             "diffuse/direct from MOSAiC sky codes; known_month set. Full-SWIR "
             "exercises the FYI_snow vis_swir classification mask on real data.",
             ha="center", fontsize=7, style="italic", color="0.35")
    fig.tight_layout(rect=(0, 0.012, 1, 1))
    fig.savefig(out / "smith_snow_fits.png", dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--out", default=str(ROOT / "figures" / "smith_validation"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = run()
    summary = summarise(rows)
    plot(rows, out)
    print(f"  figure: {out}/smith_snow_fits.png")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"summary": summary,
             "rows": [{k: r[k] for k in ("date", "sza", "direct", "stype",
                                         "conf", "rms_vis", "rms_full", "flags",
                                         "n_pos", "sky")} for r in rows]},
            indent=2))
        print(f"  json:   {args.json}")


if __name__ == "__main__":
    main()
