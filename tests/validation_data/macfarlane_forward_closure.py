#!/usr/bin/env python3
"""Forward-closure validation: MOSAiC microCT snow drivers → BioSNICAR → spectra.

Unlike ``smith_retrieval_validation.py`` (which *inverts* observed spectra to
retrieve parameters), this drives BioSNICAR *forward* from the **measured**
snow microphysics — density and specific surface area from Macfarlane et al.
(2022) microCT snowpits — and compares the predicted spectral albedo to the
coincident Smith/Light/Perovich observation. No parameters are fitted to the
target: measured-in → predicted-out. It is the strongest validation the suite
can make, because the snow state is observed, not inferred from the spectrum.

Pairing is the 26 co-located microCT↔albedo measurements verified in
``macfarlane_2022_microCT/COINCIDENCE_EVIDENCE.md`` (same named optics site,
same day, ship↔pit median 1.7 km, operator-confirmed).

STATUS: DRAFT. Two data-reality choices (flagged CONFIRM below) must be settled
against the dataset methods paper before the residuals are trusted:
  1. SSA_UNITS — mass-specific (m²/kg) vs volume-specific (mm⁻¹); swings the
     retrieved grain radius by ~ρ_ice (900×). A runtime assertion rejects
     implausible r_eff so a wrong choice fails loudly, not silently.
  2. SUBSTRATE — microCT profiles are snow-only; the sea-ice layer beneath is a
     fixed nominal FYI-summer assumption, NOT measured here. Its albedo leakage
     matters most in the NIR-transparent visible for thin snow.

Usage:
    python macfarlane_forward_closure.py --dry-run     # data loading only, no model
    python macfarlane_forward_closure.py --json out.json --plot figs/
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MCT = HERE / "macfarlane_2022_microCT"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# Reuse the audited Smith parser (spectrum + solar zenith + surface class).
from run_global_validation import _parse_smith_file  # noqa: E402

# ── data-reality configuration (see module docstring / README §Data reality) ──
RHO_ICE = 917.0            # kg/m³
SSA_UNITS = "m2/kg"        # CONFIRM: "m2/kg" (mass) or "mm-1" (volume, ÷ρ_ice)
R_EFF_PLAUSIBLE_UM = (50.0, 3000.0)   # summer snow / SSL grain-radius guard
SNOW_DENSITY_MAX = 600.0   # kg/m³ above this → treat as SSL/ice, not snow
N_SNOW_LAYERS = 5          # BioSNICAR snow layers to bin the profile into
LUT_RDS_MAX = 1500         # default granular-snow LUT ceiling (µm); coarser
                           # grains are clamped — count reported per run

# Melting-ice substrate beneath the snow: the project's SHEBA-calibrated
# FYI_SUMMER_BARE preset (SSL + drained + interior ice, BBA≈0.48–0.56). This
# is a substrate *class* selected from the observed surface type ('S' = snow on
# bare melting ice), not a tuned albedo — so the closure stays parameter-free.
# Prior run used a single clean nominal FYI layer (too bright); see README.
from biosnicar.sea_ice.presets import FYI_SUMMER_BARE  # noqa: E402

INCOMING = 3               # sub-Arctic summer irradiance spectrum
SNOW_THICKNESS_SOURCE = "measured"  # "measured" (Smith line) | "microct" (core)


def ssa_to_reff_um(ssa: np.ndarray) -> np.ndarray:
    """Optical-equivalent grain radius from SSA. r = 3 / (ρ_ice · SSA_mass)."""
    ssa_mass = ssa if SSA_UNITS == "m2/kg" else ssa * 1000.0 / RHO_ICE  # mm⁻¹→m²/kg
    with np.errstate(divide="ignore", invalid="ignore"):
        return 3.0 / (RHO_ICE * ssa_mass) * 1e6


def _event_to_cnumbers() -> dict[str, list[str]]:
    """Map each snowpit event (e.g. 'PS122/4_44-121') to its microCT sample
    C-numbers, from the staged index."""
    idx = pd.read_csv(MCT / "index_summer2020.csv")
    ev_col = "Event (Snow pit sampling)"
    xl_col = "XLSX"
    out: dict[str, list[str]] = {}
    for _, r in idx.iterrows():
        cnum = str(r[xl_col]).split("_")[0]  # C0000452_1.xlsx → C0000452
        out.setdefault(str(r[ev_col]), []).append(cnum)
    return {k: sorted(set(v)) for k, v in out.items()}


_EVENT_CNUM = _event_to_cnumbers()


def _read_profile_file(f: Path) -> pd.DataFrame | None:
    raw = pd.read_excel(f, header=None)
    tail = raw.iloc[1:, -8:].apply(pd.to_numeric, errors="coerce")
    tail.columns = ["density", "ssa", "y", "aniso", "eventdate",
                    "pos_sample_mm", "pos_snowpit_mm", "rel_depth"]
    if (tail["density"] > 50).sum() <= 5 or (tail["ssa"] > 0).sum() <= 5:
        return None  # geometry-only variant (the `_1` files)
    df = tail.dropna(subset=["density", "ssa", "rel_depth"])
    df = df[(df["density"] > 0) & (df["ssa"] > 0)]
    return df.sort_values("rel_depth", ascending=False) if len(df) > 5 else None


def load_microct_profile(event: str) -> pd.DataFrame | None:
    """Load THIS event's populated microCT profile (density, ssa, rel_depth),
    top→bottom. Resolves event → C-number(s) via the index, then picks the
    variant that actually carries data (the `_1` files are geometry-only).
    Columns read positionally from the right edge (header/data misalign)."""
    for cnum in _EVENT_CNUM.get(event, []):
        for f in sorted(MCT.glob(f"profiles/{cnum}_*.xlsx")):
            df = _read_profile_file(f)
            if df is not None:
                return df
    return None


def bin_to_layers(df: pd.DataFrame, n: int,
                  total_depth_m: float | None = None) -> dict | None:
    """Bin a top→bottom microCT profile into ``n`` snow layers over
    relative_depth → BioSNICAR (dz, rho, rds). Density/SSA/grain come from the
    core; the TOTAL snow depth is the measured Smith line thickness when given
    (``total_depth_m``), else the core's own extent. Dense SSL/ice excluded."""
    snow = df[df["density"] <= SNOW_DENSITY_MAX]
    if len(snow) < n:
        return None
    core_mm = snow["pos_snowpit_mm"].max() - snow["pos_snowpit_mm"].min()
    depth_m = total_depth_m if total_depth_m else core_mm / 1000.0
    if depth_m <= 0:
        return None
    edges = np.linspace(1.0, snow["rel_depth"].min(), n + 1)
    dz, rho, rds = [], [], []
    for i in range(n):
        m = (snow["rel_depth"] <= edges[i]) & (snow["rel_depth"] > edges[i + 1])
        seg = snow[m]
        if seg.empty:
            return None
        r_eff = ssa_to_reff_um(seg["ssa"].to_numpy())
        rho.append(float(seg["density"].mean()))
        rds.append(float(np.nanmean(r_eff)))
        dz.append(depth_m / n)  # measured (or core) depth split evenly
    lo, hi = R_EFF_PLAUSIBLE_UM
    assert all(lo <= r <= hi for r in rds), (
        f"grain radius {rds} µm outside {R_EFF_PLAUSIBLE_UM} — SSA_UNITS "
        f"({SSA_UNITS}) is probably wrong (see README §Data reality)")
    n_clamped = sum(r > LUT_RDS_MAX for r in rds)
    rds_snapped = [_snap_rds(min(r, LUT_RDS_MAX)) for r in rds]
    return dict(dz=dz, rho=rho, rds=rds_snapped, n_clamped=n_clamped,
                rds_raw=[int(round(r)) for r in rds])


def _snap_rds(v: float) -> int:
    """Snap to the granular-snow LUT grid: step 5 below 1500 µm."""
    return int(round(v / 5) * 5)


def measured_snow_depth_m(albedo_csv: Path) -> float | None:
    """Mean snow depth (m) over the snow ('S') positions of a Smith albedo
    line — the measured-in total snow thickness, from the per-position
    'Surface type' and 'Surface thickness (cm)' header rows."""
    types = thick = None
    for line in open(albedo_csv, encoding="utf-8", errors="replace"):
        if line.startswith("Surface type"):
            types = [t.strip() for t in line.split(",")[1:]]
        elif line.startswith("Surface thickness"):
            thick = line.split(",")[1:]
            break
    if not types or not thick:
        return None
    vals = []
    for t, x in zip(types, thick):
        if t.upper().startswith("S"):  # snow (incl. S/P mixed)
            try:
                v = float(x)
                if v > 0:
                    vals.append(v)
            except ValueError:
                pass
    return float(np.mean(vals)) / 100.0 if vals else None


def resample(wl_from, y, wl_to):
    return np.interp(wl_to, wl_from, y, left=np.nan, right=np.nan)


def _forward(obs, layers, substrate: dict):
    """Run BioSNICAR for a snow column (from microCT) over a substrate preset.
    Returns (wl_nm, albedo). Grain radii already snapped/clamped in `layers`."""
    from biosnicar import run_model  # deferred: heavy import

    ns = len(layers["dz"])
    sub_n = len(substrate["layer_type"])
    out = run_model(
        solzen=float(np.clip(obs.sza, 1, 89)),
        direct=0 if ("SDNV" in obs.sky or "diffuse" in obs.sky.lower()) else 1,
        incoming=INCOMING,
        layer_type=[0] * ns + list(substrate["layer_type"]),
        dz=layers["dz"] + list(substrate["dz"]),
        rho=layers["rho"] + list(substrate["rho"]),
        rds=layers["rds"] + list(substrate["rds"]),
        sea_ice_salinity=[None] * ns + list(substrate["sea_ice_salinity"]),
        sea_ice_temperature=[None] * ns + list(substrate["sea_ice_temperature"]),
        sea_ice_bubble_radius=[None] * ns + list(substrate["sea_ice_bubble_radius"]),
    )
    return out.wavelengths * 1000.0, np.asarray(out.albedo), float(out.BBA)


def _residuals(wl_mod, alb_mod, obs) -> dict:
    pred = resample(wl_mod, alb_mod, obs.wl_nm)
    ok = np.isfinite(pred) & np.isfinite(obs.alb)
    if ok.sum() < 50:
        return {}
    resid = pred[ok] - obs.alb[ok]
    wl = obs.wl_nm[ok]
    vis, nir = wl <= 750, wl > 750
    rmse = lambda m: float(np.sqrt(np.mean(resid[m] ** 2))) if m.any() else None
    return dict(rmse_all=rmse(np.ones_like(wl, bool)), rmse_vis=rmse(vis),
                rmse_nir=rmse(nir), bias=float(resid.mean()), n_bands=int(ok.sum()))


def run_pair(obs, layers) -> dict | None:
    wl, alb, bba = _forward(obs, layers, FYI_SUMMER_BARE)
    r = _residuals(wl, alb, obs)
    if not r:
        return None
    r["bba_pred"] = bba
    r["_spectra"] = dict(wl_obs=obs.wl_nm, alb_obs=obs.alb, wl_mod=wl, alb_mod=alb)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="load + bin data only, skip the forward model")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--plot", type=Path)
    args = ap.parse_args()

    ev = pd.read_csv(MCT / "coincidence_evidence.csv")
    smdir = (HERE / "smith_2021" / "resource_map_doi_10_18739_A2FT8DK8Z"
             / "data" / "SpectralAlbedoData")
    results, skipped = [], []
    for _, row in ev.iterrows():
        obs = _parse_smith_file(smdir / row["albedo_file"])
        prof = load_microct_profile(row["microct_event"])
        if obs is None or prof is None:
            skipped.append((row["albedo_file"], "obs" if obs is None else "profile"))
            continue
        depth_m = (measured_snow_depth_m(smdir / row["albedo_file"])
                   if SNOW_THICKNESS_SOURCE == "measured" else None)
        if SNOW_THICKNESS_SOURCE == "measured" and not depth_m:
            skipped.append((row["albedo_file"], "no-measured-snow-depth"))
            continue
        try:
            layers = bin_to_layers(prof, N_SNOW_LAYERS, total_depth_m=depth_m)
        except AssertionError as e:
            print(f"  PLAUSIBILITY FAIL {row['albedo_file']}: {e}")
            skipped.append((row["albedo_file"], "reff-implausible"))
            continue
        if layers is None:
            skipped.append((row["albedo_file"], "too-few-snow-layers"))
            continue
        rec = dict(date=row["date"], site=row["site"], file=row["albedo_file"],
                   sza=round(obs.sza, 1), n_snow=len(layers["dz"]),
                   snow_depth_cm=round((depth_m or sum(layers["dz"])) * 100, 1),
                   rds_um=layers["rds"], rds_raw_um=layers["rds_raw"],
                   n_clamped=layers["n_clamped"], rho=[round(r) for r in layers["rho"]])
        if not args.dry_run:
            r = run_pair(obs, layers)
            if r is None:
                skipped.append((row["albedo_file"], "model/overlap"))
                continue
            rec["_spectra"] = r.pop("_spectra")
            rec.update(r)
        results.append(rec)

    if args.plot and results and not args.dry_run:
        _plot(results, args.plot)

    print(f"\npairs attempted: {len(ev)} | usable: {len(results)} | skipped: {len(skipped)}")
    for f, why in skipped:
        print(f"  skip {f}: {why}")
    if results and not args.dry_run:
        for k in ("rmse_all", "rmse_vis", "rmse_nir"):
            vals = [r[k] for r in results if r.get(k) is not None]
            if vals:
                print(f"  {k}: median {np.median(vals):.3f}  (n={len(vals)})")
    if args.json:
        dump = [{k: v for k, v in r.items() if k != "_spectra"} for r in results]
        args.json.write_text(json.dumps(
            {"config": {"SSA_UNITS": SSA_UNITS, "substrate": "FYI_SUMMER_BARE",
                        "snow_thickness": SNOW_THICKNESS_SOURCE,
                        "n_snow_layers": N_SNOW_LAYERS},
             "results": dump, "skipped": skipped}, indent=2, default=str))
        print(f"→ {args.json}")
    return 0


def _plot(results: list[dict], outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    n = len(results)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.5 * rows),
                             sharex=True, sharey=True, squeeze=False)
    for ax, r in zip(axes.ravel(), results):
        sp = r["_spectra"]
        ax.plot(sp["wl_obs"], sp["alb_obs"], "k", lw=1.6, label="observed (ASD)")
        ax.plot(sp["wl_mod"], sp["alb_mod"], "C3", lw=1.3, label="BioSNICAR (microCT)")
        ax.set_title(f"{r['date']} {r['site']} · clamp {r['n_clamped']}/{r['n_snow']} · "
                     f"RMSE {r['rmse_all']:.3f}", fontsize=8)
        ax.set_xlim(350, 2000); ax.set_ylim(0, 1); ax.grid(alpha=.3)
    axes.ravel()[0].legend(fontsize=7)
    fig.supxlabel("wavelength (nm)"); fig.supylabel("albedo")
    fig.suptitle(f"MOSAiC forward closure (SSA={SSA_UNITS}): measured snow → predicted vs observed")
    fig.tight_layout()
    fig.savefig(outdir / "forward_closure_spectra.png", dpi=110)
    print(f"→ {outdir / 'forward_closure_spectra.png'}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    raise SystemExit(main())
