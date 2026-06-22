#!/usr/bin/env python3
"""Verification plot for the false-TIR fix (issue #111, PR fix-tir-criterion).

Replicates the ice configuration from the original bug report and plots
the spectrum before and after the fix so the artefacts can be seen to be gone.

Bug report config
-----------------
  layer_type : 1   (solid glacier ice with Fresnel surface)
  dz         : 0.02 m
  rds        : 100 µm  (fine-grained ice)
  rho        : 300 kg m⁻³
  SZA        : 55° and 89°

Root cause
----------
calc_correction_fresnel_layer() used arcsin(n+ik).real < beam_angle to detect
TIR.  For n_re > 1 the imaginary part k pulls the real part of arcsin below
π/2, so the override Rf=1.0 fired even when there was no physical TIR.

Fix
---
Gate the override on n_re < 1.0 (physically the only case where external TIR
can occur).  See docs/TIR_CRITERION_BUG.md for full analysis.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from biosnicar.drivers.setup_snicar import setup_snicar
from biosnicar.optical_properties.column_OPs import get_layer_OPs, mix_in_impurities
from biosnicar.rt_solvers.adding_doubling_solver import adding_doubling_solver

# rt_solvers/__init__.py shadows the submodule name with the imported function,
# so `import ... as mod` gives the function, not the module.  Reach the module
# through sys.modules after the package has been loaded.
_ads_module = sys.modules["biosnicar.rt_solvers.adding_doubling_solver"]


# ── Helpers ──────────────────────────────────────────────────────────────────

INPUT = "biosnicar/inputs.yaml"
WVL = np.arange(0.205, 4.999, 0.01)   # 480 wavelengths in µm


def _build_inputs(solzen):
    """Set up the bug-report ice configuration at the given SZA."""
    ice, illumination, rt_config, model_config, _, impurities = setup_snicar(INPUT)
    ice.layer_type   = [1]
    ice.dz           = [0.02]
    ice.rds          = [100]
    ice.rho          = [300]
    ice.nbr_lyr      = 1
    ice.lwc          = [0]
    ice.lwc_pct_bbl  = [0]
    ice.calculate_refractive_index(INPUT)
    illumination.solzen = solzen
    illumination.calculate_irradiance()
    for imp in impurities:
        imp.conc = [0]
    ssa_snw, g_snw, mac_snw = get_layer_OPs(ice, model_config)
    tau, ssa, g, L_snw = mix_in_impurities(
        ssa_snw, g_snw, mac_snw, ice, impurities, model_config
    )
    return tau, ssa, g, L_snw, ice, illumination, model_config


def run_fixed(solzen, smooth=True):
    """Run with the fixed TIR criterion (current (fixed) code)."""
    tau, ssa, g, L, ice, illum, mc = _build_inputs(solzen)
    mc.smooth = smooth
    return np.array(adding_doubling_solver(tau, ssa, g, L, ice, illum, mc).albedo)


def run_buggy(solzen, smooth=True):
    """Run with the OLD (buggy) TIR criterion restored by monkeypatching - pasting in from old version."""
    mod = _ads_module
    original = mod.calc_correction_fresnel_layer

    def buggy_fresnel(model_config, ice, illumination, mu0n, mu0, nr,
                      rdif_a, rdif_b, tdif_a, tdif_b, trnlay, lyr, rdir, tdir):
        ref_indx = ice.ref_idx_re + 1j * ice.ref_idx_im
        critical_angle = np.arcsin(ref_indx)
        nbr_wvl = model_config.nbr_wvl
        beam_angle = np.arccos(illumination.mu_not)

        R1 = (mu0 - nr * mu0n) / (mu0 + nr * mu0n)
        R2 = (nr * mu0 - mu0n) / (nr * mu0 + mu0n)
        T1 = 2 * mu0 / (mu0 + nr * mu0n)
        T2 = 2 * mu0 / (nr * mu0 + mu0n)
        Rf_dir_a = 0.5 * (R1**2 + R2**2)
        Tf_dir_a = 0.5 * (T1**2 + T2**2) * nr * mu0n / mu0

        # OLD criterion — no n_re < 1 gate
        tir = beam_angle >= critical_angle[:nbr_wvl].real
        Rf_dir_a[tir] = 1.0
        Tf_dir_a[tir] = 0.0

        Rf_dif_a = ice.fl_r_dif_a[:nbr_wvl]
        Tf_dif_a = 1 - Rf_dif_a
        Rf_dif_b = ice.fl_r_dif_b[:nbr_wvl]
        Tf_dif_b = 1 - Rf_dif_b

        rdif_a_old = rdif_a[:nbr_wvl, lyr].copy()
        tdif_a_old = tdif_a[:nbr_wvl, lyr].copy()
        tdif_b_old = tdif_b[:nbr_wvl, lyr].copy()
        rdir_old   = rdir[:nbr_wvl, lyr].copy()
        tdir_old   = tdir[:nbr_wvl, lyr].copy()

        rintfc = 1 / (1 - Rf_dif_b * rdif_a_old)
        tdir[:nbr_wvl, lyr] = (
            Tf_dir_a * tdir_old
            + Tf_dir_a * rdir_old * Rf_dif_b * rintfc * tdif_a_old
        )
        rdir[:nbr_wvl, lyr]  = Rf_dir_a + Tf_dir_a * rdir_old * rintfc * Tf_dif_b
        rdif_a[:nbr_wvl, lyr] = Rf_dif_a + Tf_dif_a * rdif_a_old * rintfc * Tf_dif_b
        rdif_b[:nbr_wvl, lyr] = (
            rdif_b[:nbr_wvl, lyr] + tdif_b_old * Rf_dif_b * rintfc * tdif_a_old
        )
        tdif_a[:nbr_wvl, lyr] = tdif_a_old * rintfc * Tf_dif_a
        tdif_b[:nbr_wvl, lyr] = tdif_b_old * rintfc * Tf_dif_b
        trnlay[:nbr_wvl, lyr] = Tf_dir_a * trnlay[:nbr_wvl, lyr]
        return rdif_a, rdif_b, tdif_a, tdif_b, trnlay, rdir, tdir

    mod.calc_correction_fresnel_layer = buggy_fresnel
    try:
        tau, ssa, g, L, ice, illum, mc = _build_inputs(solzen)
        mc.smooth = smooth
        result = np.array(adding_doubling_solver(tau, ssa, g, L, ice, illum, mc).albedo)
    finally:
        mod.calc_correction_fresnel_layer = original
    return result


# ── Compute spectra ───────────────────────────────────────────────────────────

print("Running model... ", end="", flush=True)
spectra = {}
for sza in (55, 89):
    spectra[sza] = {
        "buggy":  run_buggy(sza, smooth=True),
        "fixed":  run_fixed(sza, smooth=True),
    }
print("done.")

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
fig.suptitle(
    "False-TIR artefact fix (issue #111)\n"
    "Config: layer_type=1, rds=100 µm, rho=300 kg m⁻³, dz=0.02 m",
    fontsize=11,
)

panels = [
    (axes[0], 55,  0.2, 4.0,
     [(3.00, 3.10, "#ffaaaa", "Former false spike\n(3.0–3.1 µm, n_re>1)")]),
    (axes[1], 89,  0.2, 5.0,
     [(2.75, 3.50, "#ffe0aa", "Genuine TIR region\n(2.75–3.50 µm)"),
      (4.10, 4.90, "#ffaaaa", "Former false block\n(4.1–4.9 µm, n_re≈1.35)")]),
]

for ax, sza, xlo, xhi, regions in panels:
    mask = (WVL >= xlo) & (WVL <= xhi)
    w = WVL[mask]
    buggy = spectra[sza]["buggy"][mask]
    fixed = spectra[sza]["fixed"][mask]

    for lo, hi, colour, label in regions:
        ax.axvspan(lo, hi, color=colour, alpha=0.35, label=label, zorder=0)

    ax.plot(w, buggy, color="#cc3333", linewidth=1.2, linestyle="--",
            label="Before fix", zorder=2)
    ax.plot(w, fixed, color="#1a6fbd", linewidth=1.8,
            label="After fix", zorder=3)

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-0.02, 1.08)
    ax.set_ylabel("Albedo")
    ax.set_title(f"SZA = {sza}°", fontsize=10)
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle=":")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

axes[1].set_xlabel("Wavelength (µm)")

fig.tight_layout()
_out = Path(__file__).resolve().parents[1] / "images" / "tir_fix_verification.png"
plt.savefig(_out, dpi=150, bbox_inches="tight")
print(f"Plot saved to {_out}")
plt.show()
