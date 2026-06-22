#!/usr/bin/env python3
"""Observed vs retrieved spectra for the SHEBA field campaign (empirical data).

Runs retrieve_sea_ice() on every Grenfell & Light (2007) SHEBA ALBV spectrum
(7 spring snow + 16 summer bare-ice dates, 400-1000 nm) with the same
configuration as the canonical validation suite (direct=1, known_month,
noon SZA at 76N), and plots observed vs retrieved albedo per date with the
classified type, confidence, unweighted RMS residual, and quality flags.

Usage::

    python scripts/plot_sheba_fits.py [--out figures/sheba_fits]

Writes sheba_spring_fits.png and sheba_summer_fits.png.
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # repo root (tests/validation_data -> repo)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from sea_ice_emulator_sheba_validation import (  # noqa: E402
    WL_NM, catalogue, obs_to_snicar, parse_albv,
)

from biosnicar.sea_ice.retrieve import retrieve_sea_ice  # noqa: E402

TYPE_COLOR = {
    "FYI_snow": "#4575b4", "FYI_summer": "#fdae61", "FYI_bare": "#1b7837",
    "MYI_bare": "#762a83", "FYI_pond": "#35978f", "young_ice": "#d73027",
    "open_water": "#252525",
}
WL_UM = WL_NM / 1000.0

# (label, unit, format) per retrievable parameter, for the panel text box
_PARAM_FMT = {
    "tau_snow":              ("τ_snow", "", "{:.0f}"),
    "snow_grain_radius":     ("r_grain", " µm", "{:.0f}"),
    "snow_depth":            ("d_snow*", " cm", "{:.1f}"),   # * derived
    "ssl_grain_radius":      ("r_SSL", " µm", "{:.0f}"),
    "sea_ice_temperature":   ("T_ice", " °C", "{:.1f}"),
    "sea_ice_bubble_radius": ("r_bbl", " µm", "{:.0f}"),
    "brine_volume_fraction": ("V_b", "", "{:.3f}"),
    "rho_DL":                ("ρ_DL", " kg/m³", "{:.0f}"),
    "black_carbon":          ("BC", " ppb", "{:.0f}"),
    "pond_depth":            ("d_pond*", " cm", "{:.1f}"),
    "ice_thickness":         ("d_ice*", " cm", "{:.1f}"),
    "ice_thickness_cm":      ("d_ice", " cm", "{:.1f}"),
    "wind_speed_ms":         ("wind", " m/s", "{:.1f}"),
    "ocean_albedo":          ("α_ocean", "", "{:.3f}"),
    "sea_ice_salinity":      ("S", " psu", "{:.0f}"),
}
_CM_SCALE = {"snow_depth", "pond_depth", "ice_thickness"}


def _fmt_params(result, sza, month):
    """Two-line retrieved / fixed parameter strings for a panel text box."""
    parts = []
    skip = {"ice_thickness_cm"} if "ice_thickness" in result.parameters else set()
    for name, val in result.parameters.items():
        if name in skip:
            continue
        label, unit, fmt = _PARAM_FMT.get(name, (name, "", "{:.3g}"))
        v = val * 100 if name in _CM_SCALE else val
        parts.append(f"{label}={fmt.format(v)}{unit}")
    lines = ["ret: " + "  ".join(parts[i:i + 3])
             for i in range(0, len(parts), 3)]
    lines.append(f"fix: SZA={sza:.0f}°  direct=1  month={month}")
    return "\n".join(lines)


def fit_all():
    spring, summer = catalogue()
    rows = []
    for season, entries, expected in (
        ("spring", spring, ("FYI_snow",)),
        ("summer", summer, ("FYI_summer", "FYI_bare")),
    ):
        for e in entries:
            wl, alb, _ = parse_albv(e["path"])
            if wl is None:
                continue
            obs = obs_to_snicar(wl, alb)
            mask = np.isfinite(obs) & (WL_NM >= 400) & (WL_NM <= 1000)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = retrieve_sea_ice(
                    observed=np.nan_to_num(obs), wavelength_mask=mask,
                    solzen=e["sza"], direct=1, known_month=e["mm"],
                )
            rms = float(np.sqrt(np.mean(
                (r.predicted_albedo[mask] - obs[mask]) ** 2
            )))
            rows.append(dict(date=e["date"], season=season, obs=obs, mask=mask,
                             result=r, rms=rms, sza=e["sza"], month=e["mm"],
                             ok=r.surface_type in expected))
    return rows


def plot_season(rows, season, ncols, out):
    rows = [r for r in rows if r["season"] == season]
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for ax, row in zip(axes.ravel(), rows):
        res = row["result"]
        m = row["mask"]
        col = TYPE_COLOR.get(res.surface_type, "k")
        ax.plot(WL_UM[m], row["obs"][m], "-", color="0.25", lw=1.6,
                label="observed (SHEBA)")
        ax.plot(WL_UM[m], res.predicted_albedo[m], "--", color=col, lw=1.4,
                label="retrieved")
        flags = [k for k, v in res.quality_flag_description().items() if v]
        mark = "✓" if row["ok"] else "✗"
        ax.set_title(f"{row['date']}  {mark} {res.surface_type}", fontsize=9,
                     color=col)
        ax.text(0.03, 0.04,
                f"conf {res.confidence:.2f}   RMS {row['rms']:.3f}   "
                f"{', '.join(flags) if flags else 'no flags'}\n"
                + _fmt_params(res, row["sza"], row["month"]),
                transform=ax.transAxes, fontsize=5.6, va="bottom",
                family="DejaVu Sans",
                bbox=dict(fc="white", alpha=0.8, ec="0.8"))
        ax.set_xlim(0.38, 1.02)
        ax.set_ylim(0, 1.05)
    for ax in axes.ravel()[len(rows):]:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xlabel("wavelength (µm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("albedo")
    axes.ravel()[0].legend(fontsize=7, loc="center left")
    n_ok = sum(r["ok"] for r in rows)
    fig.suptitle(
        f"SHEBA {season} — observed vs retrieved spectra "
        f"(Grenfell & Light 2007; classification {n_ok}/{len(rows)})",
        fontsize=12,
    )
    fig.text(0.5, 0.005,
             "Empirical field spectra, 400–1000 nm; retrieval uses direct=1, "
             "noon SZA at 76°N, known_month (canonical validation configuration). "
             "Starred parameters are derived (snow/pond/ice depth from the "
             "retrieved optical parameterisation).",
             ha="center", fontsize=7, style="italic", color="0.35")
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(out / f"sheba_{season}_fits.png", dpi=180)
    plt.close(fig)
    return n_ok, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "figures" / "sheba_fits"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = fit_all()
    for season, ncols in (("spring", 4), ("summer", 4)):
        ok, n = plot_season(rows, season, ncols, out)
        print(f"{season}: {ok}/{n} classified as expected")
    flagged = sum(
        1 for r in rows
        if any(r["result"].quality_flag_description().values())
    )
    print(f"quality-flagged: {flagged}/{len(rows)}")
    print(f"median RMS: {np.median([r['rms'] for r in rows]):.4f}")
    print(f"figures written to {out}/")


if __name__ == "__main__":
    main()
