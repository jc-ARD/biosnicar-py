#!/usr/bin/env python3
"""Visualise the field-residual error structure (A7 / V2 companion).

Turns the residual library (build_residual_library.py) into figures that show
how the forward-model error is distributed over wavelength, regime (campaign),
and surface type — the intuition behind the A7 covariance and the V2 /
gap-tolerant findings (SWIR error is large and regime-dependent).

Uses the WINNER fit per spectrum (is_winner): the residual of the surface type
the retrieval actually chose. This is the best-achievable forward-model error
per real spectrum and avoids the type-mismatch confound of is_expected — under
the coarse field labels a single spectrum is an "expected" candidate for
several types at once (e.g. MYI_bare, of which 0% actually classify as MYI), so
is_expected residuals mix forward-model error with wrong-model error. Caveat:
"winner" is the model's own choice, so without coincident structural ground
truth this cannot rule out a systematic labelling/typing bias — the reason the
program review's top field priority is measured structure, not more spectra.

    python tests/validation_data/plot_field_residuals.py [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_residual_library import load_residual_library  # noqa: E402

# regime colours: cold/spring -> deep blue, summer snow -> teal,
# late-melt/refrozen -> rust (the hard regime, matching the program review)
CAMP = {
    "sheba":    ("#2C5F8A", "SHEBA 1998 — spring snow / summer bare ice"),
    "smith":    ("#1E8E8E", "Smith/MOSAiC 2020 — summer melting snow"),
    "istomina": ("#B4552B", "Istomina/IceArc 2012 — late-melt ice / refrozen"),
}
_ORDER = ["sheba", "smith", "istomina"]
_MINCOV = 8

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#DCE3E8", "grid.linewidth": 0.7,
    "axes.edgecolor": "#5A6B78", "axes.labelcolor": "#10202B",
    "text.color": "#10202B", "xtick.color": "#38505E", "ytick.color": "#38505E",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def _band_rms(resid, mask, sel):
    """Per-band RMS over selected rows, NaN where coverage < _MINCOV."""
    R, M = resid[sel], mask[sel]
    cov = M.sum(0)
    sq = np.where(M, R, 0.0) ** 2
    with np.errstate(invalid="ignore"):
        rms = np.sqrt(sq.sum(0) / np.maximum(cov, 1))
    rms[cov < _MINCOV] = np.nan
    return rms


def _row_region_rms(resid, mask, band):
    """Per-row RMS within a band window, for rows covering >=5 of it."""
    out = []
    for r, m in zip(resid, mask):
        b = band & m
        if b.sum() >= 5:
            out.append(float(np.sqrt(np.mean(r[b] ** 2))))
    return np.array(out)


def fig_band_rms(lib, wl_um, out):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axvspan(0.40, 1.00, color="#EAF1F5", zorder=0)
    ax.axvspan(1.10, 2.40, color="#F6EEE8", zorder=0)
    ax.text(0.70, 0.001, "VIS–NIR", color="#5A6B78", ha="center", fontsize=9)
    ax.text(1.75, 0.001, "SWIR", color="#8A5A3A", ha="center", fontsize=9)
    wl = lib["wavelength_nm"]
    for c in _ORDER:
        sel = (lib["campaign"] == c) & (lib["is_winner"] == 1)
        rms = _band_rms(lib["residual"], lib["mask"], sel)
        ax.plot(wl_um, rms, color=CAMP[c][0], lw=2.2, label=CAMP[c][1])
    ax.set_xlim(0.35, 2.45)
    ax.set_ylim(0, None)
    ax.set_xlabel("wavelength (µm)")
    ax.set_ylabel("per-band residual RMS  (|obs − model|)")
    ax.set_title("Forward-model error grows into the SWIR — and depends on regime",
                 fontsize=12.5, fontweight="bold", loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "01_band_rms_by_regime.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_region_violin(lib, out):
    wl = lib["wavelength_nm"]
    regions = [("VIS\n0.4–0.7", (wl >= 400) & (wl <= 700)),
               ("NIR\n0.7–1.0", (wl >= 700) & (wl <= 1000)),
               ("SWIR\n1.1–2.4", (wl >= 1100) & (wl <= 2400))]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    positions, data, colors, ticks = [], [], [], []
    x = 0
    for rlabel, band in regions:
        for c in _ORDER:
            sel = (lib["campaign"] == c) & (lib["is_winner"] == 1)
            vals = _row_region_rms(lib["residual"][sel], lib["mask"][sel], band)
            if vals.size:
                data.append(vals); positions.append(x); colors.append(CAMP[c][0])
            x += 1
        ticks.append((x - 2, rlabel)); x += 1
    parts = ax.violinplot(data, positions=positions, widths=0.85,
                          showmedians=True, showextrema=False)
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col); body.set_alpha(0.75); body.set_edgecolor("none")
    parts["cmedians"].set_color("#10202B"); parts["cmedians"].set_linewidth(1.3)
    ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks])
    ax.set_ylabel("per-spectrum residual RMS")
    ax.set_title("Error distribution by spectral region and regime",
                 fontsize=12.5, fontweight="bold", loc="left", pad=12)
    handles = [plt.Line2D([0], [0], color=CAMP[c][0], lw=6) for c in _ORDER]
    ax.legend(handles, [CAMP[c][1].split(" — ")[0] for c in _ORDER],
              frameon=False, fontsize=9.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "02_region_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_residual_shape(lib, wl_um, out):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axhline(0, color="#5A6B78", lw=1)
    wl = lib["wavelength_nm"]
    for c in _ORDER:
        sel = (lib["campaign"] == c) & (lib["is_winner"] == 1)
        R, M = lib["residual"][sel], lib["mask"][sel]
        cov = M.sum(0)
        mean = np.where(M, R, np.nan)
        with np.errstate(invalid="ignore"):
            mu = np.nanmean(mean, axis=0)
            sd = np.nanstd(mean, axis=0)
        mu[cov < _MINCOV] = np.nan; sd[cov < _MINCOV] = np.nan
        col = CAMP[c][0]
        ax.fill_between(wl_um, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
        ax.plot(wl_um, mu, color=col, lw=2.2, label=CAMP[c][1].split(" — ")[0])
    ax.set_xlim(0.35, 2.45)
    ax.set_xlabel("wavelength (µm)")
    ax.set_ylabel("residual  (obs − model),  mean ± 1σ")
    ax.set_title("The error is a smooth, signed bias — not white noise",
                 fontsize=12.5, fontweight="bold", loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "03_residual_shape.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_correlation_and_eofs(lib, out):
    wl = lib["wavelength_nm"]
    vis = (wl >= 400) & (wl <= 1000)
    # empirical VIS correlation over all correct-model rows fully covering VIS
    sel = lib["is_winner"] == 1
    R = lib["residual"][sel][:, vis]
    M = lib["mask"][sel][:, vis]
    full = M.all(axis=1)
    Rv = R[full]
    C = np.corrcoef(Rv.T)
    wl_vis = wl[vis] / 1000.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8),
                             gridspec_kw={"width_ratios": [1, 1.05]})
    im = axes[0].imshow(C, origin="lower", cmap="RdBu_r", vmin=-1, vmax=1,
                        extent=[wl_vis[0], wl_vis[-1], wl_vis[0], wl_vis[-1]])
    axes[0].set_title("VIS residual correlation\n(smooth ⇒ few independent DOF)",
                      fontsize=11.5, fontweight="bold", loc="left")
    axes[0].set_xlabel("wavelength (µm)"); axes[0].set_ylabel("wavelength (µm)")
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cb.set_label("correlation", fontsize=9)

    try:
        from biosnicar.inverse.model_error import ModelErrorCovariance
        me = ModelErrorCovariance.load()
        dom = me.domain
        wl_d = wl[dom] / 1000.0
        cols = ["#B4552B", "#1E8E8E", "#2C5F8A"]
        for i in range(min(3, me.eof_basis.shape[0])):
            sig = float(np.sqrt(max(me.eof_var[i], 0)))
            axes[1].plot(wl_d, me.eof_basis[i][dom], lw=2.1, color=cols[i],
                         label=f"EOF {i+1}  (σ={sig:.3f})")
        axes[1].axhline(0, color="#5A6B78", lw=0.8)
        axes[1].set_title("Leading error modes (shipped covariance)",
                          fontsize=11.5, fontweight="bold", loc="left")
        axes[1].set_xlabel("wavelength (µm)"); axes[1].set_ylabel("mode amplitude")
        axes[1].legend(frameon=False, fontsize=9)
    except Exception as exc:  # noqa: BLE001
        axes[1].text(0.5, 0.5, f"covariance unavailable\n{exc}", ha="center")
    fig.tight_layout()
    fig.savefig(out / "04_correlation_and_eofs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_by_surface_type(lib, wl_um, out):
    # Group by the WINNING type (is_winner): the residual of the type the model
    # actually chose for each spectrum. Using the label-expected type instead
    # would count e.g. MYI_bare fitted to non-MYI spectra (0% of which classify
    # as MYI) — type-mismatch error, not forward-model error.
    wl = lib["wavelength_nm"]
    win = lib["is_winner"] == 1
    types = sorted(np.unique(lib["winner_type"][win]))
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axvspan(1.10, 2.40, color="#F6EEE8", zorder=0)
    for j, t in enumerate(types):
        sel = (lib["winner_type"] == t) & win
        if sel.sum() < _MINCOV:
            continue
        rms = _band_rms(lib["residual"], lib["mask"], sel)
        ax.plot(wl_um, rms, lw=2.0, color=cmap(j / max(len(types) - 1, 1)),
                label=f"{t}  (n={int(sel.sum())})")
    ax.set_xlim(0.35, 2.45); ax.set_ylim(0, None)
    ax.set_xlabel("wavelength (µm)")
    ax.set_ylabel("per-band residual RMS")
    ax.set_title("Forward-model error by surface type",
                 fontsize=12.5, fontweight="bold", loc="left", pad=12)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "05_by_surface_type.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "figures" / "field_residuals"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    lib = load_residual_library(str(HERE / "residual_library.npz"))
    wl_um = lib["wavelength_nm"] / 1000.0
    fig_band_rms(lib, wl_um, out)
    fig_region_violin(lib, out)
    fig_residual_shape(lib, wl_um, out)
    fig_correlation_and_eofs(lib, out)
    fig_by_surface_type(lib, wl_um, out)
    print(f"saved 5 figures to {out}")


if __name__ == "__main__":
    main()
