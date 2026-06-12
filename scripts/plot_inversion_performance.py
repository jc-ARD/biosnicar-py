#!/usr/bin/env python3
"""Demonstrate sea ice inversion performance with figures.

Generates noisy synthetic observations with the FORWARD MODEL (never the
emulators — no train/test recycling), runs the full-fleet retrieval as a user
would (known geometry + known_month), and plots:

  1_confusion_matrix.png   classification accuracy across the seven types
  2_parameter_scatter.png  retrieved vs true for the key parameters
  3_spectral_fits.png      observed vs retrieved spectra, one example per type
  4_young_ice_curve.png    albedo vs thickness physics + retrieved thicknesses

Usage::

    python scripts/plot_inversion_performance.py            # ~5 min, n=12/type
    python scripts/plot_inversion_performance.py --quick    # ~2 min, n=5/type
    python scripts/plot_inversion_performance.py --n 30 --noise 0.01

Figures are written to figures/inversion_performance/ (override with --out).
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biosnicar.drivers.run_model import run_model
from biosnicar.emulator import _LOG_SAMPLE_PARAMS
from biosnicar.sea_ice.emulator_configs import (
    SEA_ICE_EMULATOR_CONFIGS,
    load_sea_ice_emulators,
)
from biosnicar.sea_ice.open_water import OpenWaterModel
from biosnicar.sea_ice.retrieve import retrieve_sea_ice

WVL = np.arange(0.205, 4.999, 0.01)

# Month passed to the retrieval per true type — operational best practice.
KNOWN_MONTH = {
    "FYI_bare": 4, "FYI_snow": 4, "FYI_summer": 7, "MYI_bare": 4,
    "FYI_pond": 7, "young_ice": 11, "open_water": 9,
}
TYPE_COLOR = {
    "FYI_bare": "#1b7837", "FYI_snow": "#4575b4", "FYI_summer": "#fdae61",
    "MYI_bare": "#762a83", "FYI_pond": "#35978f", "young_ice": "#d73027",
    "open_water": "#252525",
}
TYPES = list(KNOWN_MONTH)


def sample_truth(stype, rng):
    bounds = SEA_ICE_EMULATOR_CONFIGS[stype]["params"]
    p = {}
    for name, (lo, hi) in bounds.items():
        if name == "direct":
            p[name] = 1
        elif name == "solzen":
            p[name] = int(rng.integers(30, 76))
        elif name in _LOG_SAMPLE_PARAMS:
            p[name] = float(10 ** rng.uniform(np.log10(lo + 1), np.log10(hi + 1)) - 1)
        else:
            p[name] = float(rng.uniform(lo, hi))
    return p


def forward(stype, truth):
    if stype == "open_water":
        return OpenWaterModel().predict(**truth)
    cfg = SEA_ICE_EMULATOR_CONFIGS[stype]
    return np.asarray(run_model(**cfg["transform_fn"](truth)).albedo)


def run_retrievals(n, noise, rng, fleet):
    """One full-fleet retrieval per synthetic observation."""
    rows = []
    for stype in TYPES:
        t0 = time.time()
        for _ in range(n):
            truth = sample_truth(stype, rng)
            try:
                clean = forward(stype, truth)
            except Exception:
                continue
            if not np.all((clean >= 0) & (clean <= 1.01)):
                continue
            obs = np.clip(clean + rng.normal(0, noise, 480), 0, 1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Supplying the known noise level puts the chi-squared on its
                # proper statistical scale, so the seasonal priors act as weak
                # regularisers rather than dominating the likelihood.
                result = retrieve_sea_ice(
                    observed=obs, emulators=fleet,
                    obs_uncertainty=np.full(480, max(noise, 1e-4)),
                    solzen=truth["solzen"], direct=truth.get("direct", 1),
                    known_month=KNOWN_MONTH[stype],
                )
            rows.append(dict(true_type=stype, truth=truth, obs=obs,
                             result=result))
        print(f"  {stype:11s} {n} retrievals in {time.time() - t0:.0f}s",
              flush=True)
    return rows


# ── Figures ──────────────────────────────────────────────────────────────────

def fig_confusion(rows, out):
    mat = np.zeros((len(TYPES), len(TYPES)), dtype=int)
    for r in rows:
        i = TYPES.index(r["true_type"])
        j = TYPES.index(r["result"].surface_type)
        mat[i, j] += 1
    frac = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(TYPES)):
        for j in range(len(TYPES)):
            if mat[i, j]:
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        color="white" if frac[i, j] > 0.5 else "black",
                        fontsize=9)
    ax.set_xticks(range(len(TYPES)), TYPES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(TYPES)), TYPES, fontsize=8)
    ax.set_xlabel("classified as")
    ax.set_ylabel("true surface type")
    acc = np.trace(mat) / mat.sum()
    ax.set_title(f"Surface-type classification — overall accuracy {acc:.0%}\n"
                 "(noisy forward-model spectra, known geometry + month)",
                 fontsize=10)
    fig.colorbar(im, label="row fraction")
    fig.tight_layout()
    fig.savefig(out / "1_confusion_matrix.png", dpi=180)
    plt.close(fig)
    return acc


def _true_and_fit(rows, stype, param):
    """True vs fitted values of the TRUE type's fit (parameter accuracy is
    assessed conditional on the surface type, like a single-type retrieval)."""
    t, f = [], []
    for r in rows:
        if r["true_type"] != stype or stype not in r["result"].all_fits:
            continue
        t.append(r["truth"][param])
        f.append(r["result"].all_fits[stype].best_fit[param])
    return np.array(t), np.array(f)


def fig_scatter(rows, out):
    panels = [
        ("FYI_pond", "pond_depth", "pond depth (m)", False),
        ("young_ice", "ice_thickness_cm", "ice thickness (cm)", False),
        ("FYI_snow", "snow_grain_radius", "snow grain radius (µm)", True),
        ("FYI_summer", "ssl_grain_radius", "SSL grain radius (µm)", True),
        ("FYI_bare", "sea_ice_bubble_radius", "bubble radius (µm)", True),
        ("open_water", "wind_speed_ms", "wind speed (m/s)", False),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (stype, param, label, logax) in zip(axes.ravel(), panels):
        t, f = _true_and_fit(rows, stype, param)
        if len(t) == 0:
            ax.set_axis_off()
            continue
        ax.scatter(t, f, s=28, color=TYPE_COLOR[stype], alpha=0.8,
                   edgecolor="k", linewidth=0.4, zorder=3)
        lo = min(t.min(), f.min()) * 0.9
        hi = max(t.max(), f.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.6)
        if logax:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ss = np.sum((t - t.mean()) ** 2)
        r2 = 1 - np.sum((f - t) ** 2) / ss if ss > 0 else np.nan
        rmse = float(np.sqrt(np.mean((f - t) ** 2)))
        ax.set_title(f"{stype}: {label}", fontsize=10,
                     color=TYPE_COLOR[stype])
        ax.text(0.04, 0.93, f"R² = {r2:.3f}\nRMSE = {rmse:.3g}\nn = {len(t)}",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(fc="white", alpha=0.8, ec="0.7"))
        ax.set_xlabel("true")
        ax.set_ylabel("retrieved")
    fig.suptitle("Parameter retrieval — true vs retrieved (true-type fit)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "2_parameter_scatter.png", dpi=180)
    plt.close(fig)


def fig_spectra(rows, out):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True, sharey=True)
    for ax, stype in zip(axes.ravel(), TYPES):
        r = next((r for r in rows if r["true_type"] == stype), None)
        if r is None:
            ax.set_axis_off()
            continue
        res = r["result"]
        ax.plot(WVL, r["obs"], color=TYPE_COLOR[stype], lw=1.2,
                label="observed (+noise)")
        ax.plot(WVL, res.predicted_albedo, "k--", lw=1.0, alpha=0.8,
                label="retrieved")
        ok = "✓" if res.surface_type == stype else f"✗ → {res.surface_type}"
        ax.set_title(f"{stype}  {ok}", fontsize=9, color=TYPE_COLOR[stype])
        ax.text(0.97, 0.93,
                f"χ² {res.cost:.0f} (480 bands)\nconf {res.confidence:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                bbox=dict(fc="white", alpha=0.8, ec="0.7"))
        ax.set_xlim(0.3, 2.5)
        ax.set_ylim(0, 1.02)
    axes[0, 0].legend(fontsize=7.5, loc="center right")
    axes[-1, -1].set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("wavelength (µm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("albedo")
    fig.suptitle("Observed vs retrieved spectra — one example per surface type",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "3_spectral_fits.png", dpi=180)
    plt.close(fig)


def fig_young_ice(rows, out):
    d_grid = np.linspace(0.005, 0.30, 30)
    bba = []
    flx = None
    for d in d_grid:
        o = run_model(layer_type=6, ice_thickness=float(d),
                      sea_ice_temperature=-10, sea_ice_salinity=25,
                      ocean_albedo=0.04, solzen=60, direct=1)
        bba.append(float(o.BBA))
        flx = np.asarray(o.flx_slr)

    t, f = _true_and_fit(rows, "young_ice", "ice_thickness_cm")
    obs_bba = [float(r["obs"] @ flx / flx.sum())
               for r in rows if r["true_type"] == "young_ice"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(d_grid * 100, bba, color=TYPE_COLOR["young_ice"], lw=1.5,
             label="forward model (T=−10 °C, S=25 psu)")
    for x, lab in [(1, "grease"), (5, "nilas"), (12, "grey"), (20, "grey-white")]:
        ax1.axvline(x, color="gray", lw=0.5, ls=":")
        ax1.text(x, 0.30, lab, rotation=90, fontsize=7, va="top", color="gray")
    ax1.scatter(t, obs_bba, s=22, color="k", alpha=0.6, zorder=3,
                label="synthetic observations")
    ax1.set_xlabel("ice thickness (cm)")
    ax1.set_ylabel("broadband albedo")
    ax1.set_title("Young ice: albedo grows with thickness", fontsize=10)
    ax1.legend(fontsize=8)

    ax2.scatter(t, f, s=28, color=TYPE_COLOR["young_ice"], alpha=0.8,
                edgecolor="k", linewidth=0.4)
    lim = max(31, (max(t.max(), f.max()) if len(t) else 31))
    ax2.plot([0, lim], [0, lim], "k--", lw=0.8)
    if len(t):
        rmse = float(np.sqrt(np.mean((f - t) ** 2)))
        ax2.text(0.04, 0.92, f"RMSE = {rmse:.2f} cm\nn = {len(t)}",
                 transform=ax2.transAxes, va="top", fontsize=8,
                 bbox=dict(fc="white", alpha=0.8, ec="0.7"))
    ax2.set_xlabel("true thickness (cm)")
    ax2.set_ylabel("retrieved thickness (cm)")
    ax2.set_title("Thickness retrieval (freeze-up prior applied)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "4_young_ice_curve.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12, help="observations per type")
    ap.add_argument("--quick", action="store_true", help="n=5")
    ap.add_argument("--noise", type=float, default=0.005,
                    help="1-sigma albedo noise")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(ROOT / "figures" / "inversion_performance"))
    args = ap.parse_args()
    n = 5 if args.quick else args.n

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    fleet = load_sea_ice_emulators()

    print(f"Running {n} retrievals per type (noise σ={args.noise}) ...")
    rows = run_retrievals(n, args.noise, rng, fleet)

    acc = fig_confusion(rows, out)
    fig_scatter(rows, out)
    fig_spectra(rows, out)
    fig_young_ice(rows, out)

    print(f"\nOverall classification accuracy: {acc:.0%} ({len(rows)} retrievals)")
    print(f"Figures written to {out}/")


if __name__ == "__main__":
    main()
