"""Figure for the Greenland ASD terrestrial validation.

Panel A: example spectral fits (measured vs forward-model vs retrieval) across algal load.
Panel B: retrieved glacier-algae vs field-measured cell counts (independent ground truth).
Panel C: spectral-fit RMS, forward model vs emulator retrieval.

Colours: Okabe-Ito CVD-safe palette, assigned by identity in fixed order.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biosnicar.emulator import Emulator

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "scripts" / "experiments" / "greenland_asd_results"
EMU = Emulator.load(str(REPO / "data" / "emulators" / "glacier_ice_8_param_default.npz"))

# Okabe-Ito (colourblind-safe), by identity
C_MEAS, C_FWD, C_RET = "#000000", "#D55E00", "#0072B2"
INK, MUTED, GRID = "#222222", "#666666", "#dddddd"

sp = np.load(OUT / "spectra.npz", allow_pickle=True)
wvl, mask = sp["wvl_nm"], sp["fit_mask"]
names = list(sp["names"])
observed, forward = sp["observed"], sp["predicted"]
cells = sp["measured_cells"]
ret = pd.read_csv(OUT / "retrieval_rho_fixed.csv")   # best rank-correlation config
fwd = pd.read_csv(OUT / "forward_closure.csv")


def retrieved_spectrum(row):
    return np.asarray(EMU.predict(
        rds=row["retrieved_rds"], rho=row["meta_rho"],
        black_carbon=0.0, dust=0.0, snow_algae=0.0,
        glacier_algae=max(row["retrieved_algae"], 0.0), direct=0.9, solzen=50.0))


plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "axes.labelcolor": INK})
fig = plt.figure(figsize=(12, 4.2))
gs = fig.add_gridspec(1, 3, wspace=0.42)
wmask = mask & (wvl <= 1800)

# --- Panel A: example fits spanning algal load ---
axA = fig.add_subplot(gs[0, 0])
order = np.argsort(cells)
picks = [order[0], order[len(order) // 2], order[-1]]  # none / mid / high algae
for k, idx in enumerate(picks):
    row = ret[ret["name"] == names[idx]].iloc[0]
    rspec = retrieved_spectrum(row)
    off = k * 0.0  # no offset; overplot
    lbl = "measured" if k == 0 else None
    axA.plot(wvl[wmask], observed[idx][wmask], color=C_MEAS, lw=1.4, label=lbl, zorder=3)
    axA.plot(wvl[wmask], forward[idx][wmask], color=C_FWD, lw=1.6, alpha=0.9,
             label="forward model" if k == 0 else None, zorder=2)
    axA.plot(wvl[wmask], rspec[wmask], color=C_RET, lw=1.6, ls=(0, (4, 2)),
             label="retrieval fit" if k == 0 else None, zorder=2)
    # label each measured curve near its 600 nm peak, spanning algal load
    j = np.argmin(np.abs(wvl[wmask] - 600))
    axA.annotate(f"{cells[idx]:.0f} cells/mL", xy=(600, observed[idx][wmask][j]),
                 xytext=(0, 5), textcoords="offset points", fontsize=6.5, color=MUTED, ha="center")
axA.set_xlim(350, 1850)
axA.set_ylim(0, 0.8)
axA.set_xlabel("wavelength (nm)")
axA.set_ylabel("albedo")
axA.set_title("A  Example spectral fits", loc="left", fontsize=10, color=INK)
axA.legend(frameon=False, fontsize=7.5, loc="upper right")
axA.grid(True, color=GRID, lw=0.6)

# --- Panel B: retrieved algae vs measured cells ---
axB = fig.add_subplot(gs[0, 1])
m = ret["measured_cells"].to_numpy(float)
a = ret["retrieved_algae"].to_numpy(float)
present = m > 0
FLOOR = 150.0  # plot algae-absent samples at a left-edge floor
mp = np.where(present, m, FLOOR)
axB.plot([FLOOR, 1e5], [FLOOR, 1e5], color=MUTED, lw=1.0, ls=":", zorder=1, label="1:1")
axB.scatter(mp[present], a[present], s=34, color=C_RET, edgecolor="white", lw=0.6,
            zorder=3, label="algae present")
axB.scatter(mp[~present], a[~present], s=40, facecolor="none", edgecolor=C_FWD, lw=1.3,
            zorder=3, label="measured = 0 (false +)")
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xlim(FLOOR * 0.7, 1e5); axB.set_ylim(FLOOR * 0.7, 1e5)
axB.set_xlabel("field-measured algae (cells/mL)")
axB.set_ylabel("retrieved glacier_algae (cells/mL)")
sp_r = pd.Series(m).rank().corr(pd.Series(a).rank())
axB.set_title("B  Retrieved vs measured algae", loc="left", fontsize=10, color=INK)
axB.text(0.04, 0.96, f"Spearman r = {sp_r:.2f}", transform=axB.transAxes,
         fontsize=8.5, color=INK, va="top")
axB.legend(frameon=False, fontsize=7.5, loc="lower right")
axB.grid(True, which="both", color=GRID, lw=0.5)

# --- Panel C: spectral-fit RMS, forward vs retrieval ---
axC = fig.add_subplot(gs[0, 2])
groups = [("forward\nmodel", fwd["rms_full"].to_numpy(float), C_FWD),
          ("emulator\nretrieval", ret["rms_fit"].to_numpy(float), C_RET)]
rng = np.random.default_rng(0)
for i, (lbl, vals, col) in enumerate(groups):
    x = i + (rng.random(len(vals)) - 0.5) * 0.18
    axC.scatter(x, vals, s=26, color=col, edgecolor="white", lw=0.5, alpha=0.9, zorder=3)
    med = np.median(vals)
    axC.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=2, zorder=4)
    axC.annotate(f"median\n{med:.3f}", xy=(i + 0.32, med), fontsize=7.5, color=INK, va="center")
axC.set_xticks([0, 1]); axC.set_xticklabels([g[0] for g in groups])
axC.set_xlim(-0.5, 1.7)
axC.set_ylim(0, max(fwd["rms_full"].max(), ret["rms_fit"].max()) * 1.1)
axC.set_ylabel("spectral-fit RMS (albedo, 350–1800 nm)")
axC.set_title("C  Goodness of fit", loc="left", fontsize=10, color=INK)
axC.grid(True, axis="y", color=GRID, lw=0.6)

fig.suptitle("BioSNICAR terrestrial validation — SW Greenland Ice Sheet ASD field spectra (n=31)",
             fontsize=11, color=INK, y=1.02, x=0.02, ha="left")
fig.savefig(OUT / "greenland_validation.png", dpi=150, bbox_inches="tight")
print("wrote", OUT / "greenland_validation.png")
