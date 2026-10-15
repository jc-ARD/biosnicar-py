"""Figure: algae-layer thickness / vertical placement does NOT explain the deficit."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent / "greenland_asd_results"
d = np.load(OUT / "thickness_experiment.npz")
mult = d["mult"]; DZmm = [2, 5, 10, 20, 50]

INK, MUTED, GRID, ALT = "#222", "#666", "#ddd", "#D55E00"
# sequential blues, light->dark with thickness
COLORS = ["#9ecae1", "#6baed6", "#3182bd", "#08519c", "#08306b"]
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"wspace": 0.28})

# Panel A: median perturbation profile per thickness (they overlap -> placement irrelevant)
for t, c in zip(DZmm, COLORS):
    med = np.nanmedian(d[f"prof_{t}"], axis=0)
    ax[0].plot(mult, med, color=c, lw=2.0, label=f"{t} mm")
ax[0].axvline(1.0, color=MUTED, ls=":", lw=1.2)
ax[0].annotate("measured\ntotal (m=1)", xy=(1.0, 0.145), fontsize=7.5, color=MUTED, ha="center")
ax[0].set_xscale("log")
ax[0].set_xlabel("algae multiplier  m  (× measured total cells)")
ax[0].set_ylabel("median best-fit residual RMS")
ax[0].set_title("A  Perturbation profile is the same at every thickness",
                loc="left", fontsize=9.5)
ax[0].legend(frameon=False, fontsize=7.5, title="algae layer", title_fontsize=7.5)
ax[0].grid(True, color=GRID, lw=0.5)

# Panel B: preferred multiplier m* vs thickness (flat)
mstar_med = [np.median(d[f"mstar_{t}"]) for t in DZmm]
ax[1].plot(DZmm, mstar_med, "o-", color="#08519c", lw=2, ms=8, mec="white")
ax[1].axhline(1.0, color=ALT, ls="--", lw=1.4)
ax[1].annotate("m*=1 (model consistent with measured cells)", xy=(2, 1.0),
               xytext=(2, 2.1), fontsize=7.5, color=ALT)
ax[1].set_xscale("log"); ax[1].set_xticks(DZmm); ax[1].set_xticklabels(DZmm)
ax[1].set_ylim(0, max(mstar_med) * 1.2)
ax[1].set_xlabel("algae-bearing layer thickness (mm)")
ax[1].set_ylabel("preferred multiplier  m*  (median)")
ax[1].set_title("B  Concentrating cells to 2 mm does not help", loc="left", fontsize=9.5)
ax[1].grid(True, color=GRID, lw=0.5)

fig.suptitle("Algae-layer thickness test — measured total (areal) cells held fixed, placement swept",
             fontsize=11, y=1.02, x=0.02, ha="left")
fig.savefig(OUT / "greenland_thickness.png", dpi=150, bbox_inches="tight")
print("wrote", OUT / "greenland_thickness.png")
