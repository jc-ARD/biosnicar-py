"""Figure for the 3-way forward-closure comparison (A S2-tuned, B fixed single, C fixed crust)."""
import numpy as np
import matplotlib.pyplot as plt
from biosnicar.bands._core import WVL
from pathlib import Path

OUT = Path(__file__).resolve().parent / "greenland_asd_results"
d = np.load(OUT / "forward_v2_spectra.npz", allow_pickle=True)
wl = WVL * 1000
names = list(d["names"]); cells = d["measured_cells"]
obs, pA, pB, pC = d["observed"], d["pred_A"], d["pred_B"], d["pred_C"]

# Okabe-Ito
C_OBS, C_A, C_B, C_C = "#000000", "#0072B2", "#009E73", "#D55E00"
INK, MUTED, GRID = "#222", "#666", "#ddd"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
m = (wl >= 350) & (wl <= 1800)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), gridspec_kw={"wspace": 0.3})

# Panels A/B: a clean and an algae-rich example
for ax, idx, tag in [(axes[0], int(np.argmin(cells)), "clean (0 cells/mL)"),
                     (axes[1], int(np.argmax(cells)), f"algae-rich ({cells.max():.0f} cells/mL)")]:
    ax.plot(wl[m], obs[idx][m], color=C_OBS, lw=1.6, label="measured", zorder=4)
    ax.plot(wl[m], pB[idx][m], color=C_B, lw=1.5, label="solid ice (layer_type=1)")
    ax.plot(wl[m], pC[idx][m], color=C_C, lw=1.6, label="granular weathered (layer_type=0)")
    ax.set_title(f"{names[idx]} — {tag}", loc="left", fontsize=9.5)
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("albedo")
    ax.set_ylim(0, 0.9); ax.grid(True, color=GRID, lw=0.6)
axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")

# Panel C: median RMS by band
axC = axes[2]
labels = ["A solid\n(S2, circular)", "B solid\n(fixed, blind)", "C granular\n(fixed, blind)"]
full = [0.0721, 0.0941, 0.1120]; vis = [0.0899, 0.0756, 0.1483]; nir = [0.0715, 0.0758, 0.0786]
x = np.arange(3); w = 0.26
axC.bar(x - w, full, w, color=INK, label="full 350–1800")
axC.bar(x, vis, w, color="#8c8c8c", label="VIS 350–700")
axC.bar(x + w, nir, w, color="#c7c7c7", label="NIR 700–1800")
axC.set_xticks(x); axC.set_xticklabels(labels, fontsize=8)
axC.set_ylabel("median spectral-fit RMS (albedo)")
axC.set_title("C  Closure error by band", loc="left", fontsize=9.5)
axC.legend(frameon=False, fontsize=7.5)
axC.grid(True, axis="y", color=GRID, lw=0.6)

fig.suptitle("Forward-model closure — SW Greenland ASD (n=31): adding-doubling, weathering-crust test",
             fontsize=11, y=1.02, x=0.02, ha="left")
fig.savefig(OUT / "greenland_closure_v2.png", dpi=150, bbox_inches="tight")
print("wrote", OUT / "greenland_closure_v2.png")
