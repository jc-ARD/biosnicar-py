"""Figure for the constrained validation / algae-perturbation profile."""
import numpy as np
import matplotlib.pyplot as plt
from biosnicar.bands._core import WVL

from greenland_constrained_validation import (
    build_grid, make_interp, predict, best_fit, FIT_MASK, OUTDIR)
from greenland_asd_validation import load_data

wl = WVL * 1000
d = np.load(OUTDIR / "constrained_validation.npz", allow_pickle=True)
mult, prof, mstar = d["mult"], d["profile"], d["mstar"]
present_names = list(d["present_names"]); present_cells = d["present_cells"]

C_MAIN, C_ALT, INK, MUTED, GRID = "#0072B2", "#D55E00", "#222", "#666", "#ddd"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, ax = plt.subplots(1, 3, figsize=(13, 4), gridspec_kw={"wspace": 0.3})

# Panel A: perturbation profile
med = np.nanmedian(prof, axis=0)
for i in range(prof.shape[0]):
    ax[0].plot(mult, prof[i], color=C_MAIN, lw=0.6, alpha=0.18)
ax[0].plot(mult, med, color=C_MAIN, lw=2.4, label="median over 25 samples", zorder=5)
mmin = mult[int(np.argmin(med))]
ax[0].axvline(1.0, color=MUTED, ls=":", lw=1.2)
ax[0].annotate("measured\n(m=1)", xy=(1.0, ax[0].get_ylim()[1]), xytext=(1.0, 0.045),
               fontsize=7.5, color=MUTED, ha="center")
ax[0].axvline(mmin, color=C_ALT, ls="--", lw=1.4)
ax[0].annotate(f"best fit\nm={mmin:.1f}", xy=(mmin, 0.045), fontsize=7.5, color=C_ALT, ha="center")
ax[0].set_xscale("log")
ax[0].set_xlabel("algae multiplier  m  (× field-measured cells)")
ax[0].set_ylabel("best-fit residual RMS (nuisance re-fit)")
ax[0].set_title("A  Algae-perturbation profile", loc="left", fontsize=10)
ax[0].legend(frameon=False, fontsize=7.5)
ax[0].grid(True, color=GRID, lw=0.5)

# Panel B: distribution of per-sample preferred multiplier m*
ax[1].hist(mstar, bins=np.logspace(np.log10(0.1), np.log10(10), 16),
           color=C_MAIN, edgecolor="white", lw=0.6)
ax[1].axvline(1.0, color=MUTED, ls=":", lw=1.2, label="measured (m=1)")
ax[1].axvline(np.median(mstar), color=C_ALT, ls="--", lw=1.6,
              label=f"median m*={np.median(mstar):.1f}")
ax[1].set_xscale("log")
ax[1].set_xlabel("per-sample preferred multiplier  m*")
ax[1].set_ylabel("number of samples")
ax[1].set_title("B  Every sample prefers more algae", loc="left", fontsize=10)
ax[1].legend(frameon=False, fontsize=7.5)
ax[1].grid(True, axis="y", color=GRID, lw=0.5)

# Panel C: example fit at m=1 vs m=m* for a representative sample
interp = make_interp(build_grid())
samples = {s["name"]: s for s in load_data()}
idx = int(np.argmin(np.abs(present_cells - np.median(present_cells))))
s = samples[present_names[idx]]
m = (wl >= 350) & (wl <= 1800)
_, r1, h1 = best_fit(interp, s["observed"], s["measured_cells"], s["zen"])
mstar_i = mstar[idx]
_, r2, h2 = best_fit(interp, s["observed"], mstar_i * s["measured_cells"], s["zen"])
ax[2].plot(wl[m], s["observed"][m], color=INK, lw=1.7, label="measured", zorder=4)
ax[2].plot(wl[m], predict(interp, r1, h1, s["measured_cells"], s["zen"])[m],
           color=C_MAIN, lw=1.5, label=f"model @ measured ({s['measured_cells']:.0f})")
ax[2].plot(wl[m], predict(interp, r2, h2, mstar_i * s["measured_cells"], s["zen"])[m],
           color=C_ALT, lw=1.5, ls=(0, (4, 2)),
           label=f"model @ {mstar_i:.1f}× ({mstar_i*s['measured_cells']:.0f})")
ax[2].set_xlabel("wavelength (nm)"); ax[2].set_ylabel("albedo"); ax[2].set_ylim(0, 0.8)
ax[2].set_title(f"C  Example: {s['name']}", loc="left", fontsize=10)
ax[2].legend(frameon=False, fontsize=7.5); ax[2].grid(True, color=GRID, lw=0.5)

fig.suptitle("Constrained validation — algae fixed at measured, grain+density profiled (n=25 algae-present)",
             fontsize=11, y=1.02, x=0.02, ha="left")
fig.savefig(OUTDIR / "greenland_constrained.png", dpi=150, bbox_inches="tight")
print("wrote", OUTDIR / "greenland_constrained.png")
