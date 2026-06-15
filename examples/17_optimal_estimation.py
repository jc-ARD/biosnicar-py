#!/usr/bin/env python3
"""Optimal estimation (method="oe") — uncertainty, averaging kernels, DFS.

Optimal estimation upgrades the retrieval from "best-fit point" to a full
Bayesian estimate. This example shows the three things that buys you:

  1. A posterior **uncertainty** on every retrieved parameter.
  2. An **averaging kernel** — how much of each answer came from the
     measurement vs the prior (honest provenance).
  3. **Degrees of freedom for signal (DFS)** — how many parameters the data
     genuinely constrained. This is what separates a rich hyperspectral
     measurement from a few-band satellite one, and the demo makes it visible.

It also shows OE classification, which returns calibrated per-surface-type
**probabilities** instead of a single best-guess label.

Observations are generated with the FORWARD MODEL (not the emulators), so the
retrieval faces genuine model error — no train/test recycling.

Prerequisites
-------------
    python scripts/build_sea_ice_emulators.py
    (matplotlib for the figure)

Usage
-----
    python examples/17_optimal_estimation.py [--out figures/oe_demo]
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biosnicar.drivers.run_model import run_model
from biosnicar.inverse.optimize import retrieve
from biosnicar.sea_ice.emulator_configs import (
    SEA_ICE_EMULATOR_CONFIGS,
    load_sea_ice_emulators,
)
from biosnicar.sea_ice.retrieve import retrieve_sea_ice

warnings.simplefilter("ignore")
rng = np.random.default_rng(0)
NOISE = 0.005


def forward_obs(stype, truth):
    """Honest observation: forward model + noise (never the emulator)."""
    a = np.asarray(run_model(**SEA_ICE_EMULATOR_CONFIGS[stype]["transform_fn"](truth)).albedo)
    return np.clip(a + rng.normal(0, NOISE, 480), 0.0, 1.0)


# ── 1. Single-type OE retrieval: posterior + averaging kernel + DFS ──────────

print("=" * 70)
print("1. Optimal-estimation retrieval of a melt pond (full 480-band spectrum)")
print("=" * 70)
emu = load_sea_ice_emulators(["FYI_pond"])["FYI_pond"]
truth = dict(pond_depth=0.25, sea_ice_temperature=-4.0, black_carbon=100.0,
             solzen=60, direct=1)
obs = forward_obs("FYI_pond", truth)
free = ["pond_depth", "sea_ice_temperature", "black_carbon"]

r = retrieve(observed=obs, parameters=free, emulator=emu,
             fixed_params={"solzen": 60, "direct": 1},
             obs_uncertainty=np.full(480, NOISE), method="oe")

truth_vals = {"pond_depth": 0.25, "sea_ice_temperature": -4.0, "black_carbon": 100.0}
print(f"  {'parameter':22s} {'retrieved':>12s} {'±1σ':>10s} {'truth':>8s}  info(A)")
for p in free:
    print(f"  {p:22s} {r.best_fit[p]:12.3f} {r.uncertainty[p]:10.3f} "
          f"{truth_vals[p]:8.2f}   {r.averaging_kernel_diag[p]:.2f}")
print(f"\n  DFS = {r.dfs:.2f} of {len(free)} parameters "
      f"(how many the spectrum genuinely constrained)")
print(f"  reduced chi-squared = {r.cost / np.isfinite(obs).sum():.2f}")


# ── 2. The DFS contrast: hyperspectral vs a few satellite bands ──────────────

print("\n" + "=" * 70)
print("2. Information content collapses with fewer bands (hyperspectral → S2)")
print("=" * 70)

# (a) full spectrum
r_hs = r
# (b) Sentinel-2 4-band (VIS–NIR only)
bands = ["B2", "B3", "B4", "B8"]
from biosnicar.bands import to_platform  # noqa: E402

band_obs = to_platform(obs, "sentinel2", flx_slr=emu.flx_slr)
y_s2 = np.array([getattr(band_obs, b) for b in bands])
r_s2 = retrieve(observed=y_s2, parameters=free, emulator=emu,
                platform="sentinel2", observed_band_names=bands,
                fixed_params={"solzen": 60, "direct": 1},
                obs_uncertainty=np.full(len(bands), 0.02), method="oe")

print(f"  full 480-band spectrum : DFS = {r_hs.dfs:.2f}")
print(f"  Sentinel-2 (4 bands)   : DFS = {r_s2.dfs:.2f}")
print("  per-parameter info content (1 = measured, 0 = prior-driven):")
print(f"    {'parameter':22s} {'hyperspectral':>14s} {'S2 4-band':>12s}")
for p in free:
    print(f"    {p:22s} {r_hs.averaging_kernel_diag[p]:14.2f} "
          f"{r_s2.averaging_kernel_diag[p]:12.2f}")
print("  → with fewer bands, parameters fall back on the prior; OE says so,")
print("    instead of reporting them as if measured.")


# ── 3. OE classification: posterior probabilities, not a bare label ──────────

print("\n" + "=" * 70)
print("3. OE classification returns calibrated surface-type probabilities")
print("=" * 70)
fleet = load_sea_ice_emulators()
res = retrieve_sea_ice(observed=obs, emulators=fleet, solzen=60, direct=1,
                       known_month=7, method="oe", obs_uncertainty=np.full(480, NOISE))
print(f"  winner: {res.surface_type}  (probability {res.confidence:.3f}), DFS {res.dfs:.2f}")
print("  class probabilities:")
for name, p in sorted(res.class_probabilities.items(), key=lambda kv: -kv[1]):
    bar = "█" * int(round(p * 30))
    print(f"    {name:12s} {p:6.3f} {bar}")


# ── 4. Figure: information content, hyperspectral vs satellite ───────────────

def make_figure(out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(free))
    w = 0.38
    ax.bar(x - w / 2, [r_hs.averaging_kernel_diag[p] for p in free], w,
           label=f"hyperspectral (480 band)  ·  DFS {r_hs.dfs:.2f}",
           color="#1b7837")
    ax.bar(x + w / 2, [r_s2.averaging_kernel_diag[p] for p in free], w,
           label=f"Sentinel-2 (4 band)  ·  DFS {r_s2.dfs:.2f}",
           color="#fdae61")
    ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
    ax.set_xticks(x, [p.replace("_", "\n") for p in free], fontsize=9)
    ax.set_ylabel("information content  (averaging-kernel diagonal)")
    ax.set_ylim(0, 1.1)
    ax.set_title("Optimal estimation: what each measurement can actually retrieve\n"
                 "(1 = determined by the data, 0 = determined by the prior)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    fig.text(0.5, 0.005,
             "Melt-pond retrieval; observation from the forward model + noise. "
             "Pond depth stays measured; temperature falls to the prior as bands drop.",
             ha="center", fontsize=7, style="italic", color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out.mkdir(parents=True, exist_ok=True)
    path = out / "oe_information_content.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print(f"\nFigure written to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "figures" / "oe_demo"))
    args = ap.parse_args()
    make_figure(Path(args.out))


if __name__ == "__main__":
    main()
