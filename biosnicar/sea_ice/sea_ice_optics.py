"""Per-layer optical properties for sea ice (τ, ω, g) at 480 bands.

Physics pipeline
----------------
1. Brine volume fraction ν_b and gas volume fraction ν_air from the layer's
   temperature, salinity and bulk density — Cox & Weeks (1983) below -2 °C,
   Leppäranta & Manninen (1988) from -2 to 0 °C.  Two halves of one density
   balance, so they are mutually consistent by construction.
2. Brine complex RI at the liquidus salinity of the same phase model
   (`brine_optics.compute_brine_rfidx`).
3. Pure-ice complex RI from the Picard 2016 variant in rfidx_ice.npz.
4. Effective ice+brine RI by the symmetric Bruggeman rule, valid at any
   brine fraction.
5. Absorption coefficient from the imaginary part of the effective RI,
   weighted by the condensed-phase fraction (air does not absorb).
6. Scattering, the sum of four independent populations:
     - air bubbles, from the bubbly_air LUT × ν_air;
     - brine inclusions (pockets and tubes) in the ice;
     - mirabilite crystals in the brine, below -8.2 °C;
     - hydrohalite crystals in the brine, below -22.9 °C.
   The last three follow Light, Maykut & Grenfell (2004) — see
   :mod:`biosnicar.sea_ice.inclusion_scattering`.
7. Combined (τ, ω, g) per metre; g is the scattering-weighted mean of the
   populations' asymmetry parameters.

Why the brine terms matter
--------------------------
Brine inclusions scatter strongly but almost entirely forward (g ≈ 0.99), so
what counts is the similarity parameter s = σ(1 − g).  For interior
first-year ice at -15 °C and 4.7 ppt they contribute s ≈ 3 m⁻¹.  Air bubbles
of 300 µm at 1% porosity contribute about 7 m⁻¹, and at the 7% porosity of a
drained layer about 50.  So in drained, bubbly ice the bubbles dominate and
this term is a correction; in dense interior ice it is comparable to the
bubbles, and leaving it out forces the retrieved bubble radius to stand in
for it.

Approximations documented here
-------------------------------
- Air bubble scattering uses the pure-ice bubbly_air LUT.  The correct
  calculation would re-run Mie for bubbles in the ice+brine effective medium.
  For typical sea ice conditions (ν_b < 0.15, moderate bubble sizes) the
  error in scattering efficiency is < 5%.
- Bubbles are the layer's whole gas volume.  Light et al. distinguish
  bubbles inside brine inclusions, which grow as the ice warms and escape
  above a critical size, from bubbles in the ice matrix; that distinction is
  not made here, where the gas volume comes from the bulk density.
- The four scattering populations are independent and add.
- Gas (air) inside brine pockets is neglected as a separate scatterer.
- Bubble scattering is taken linear in ν_air (single-bubble LUT × volume
  fraction), i.e. bubbles scatter independently.  At the LUT's 700 kg/m³ floor
  the layer is ~21.5% air; inter-bubble clearance there is still many
  wavelengths, so the independent-scatterer picture holds, but it is untested
  above ~15% air.  Beyond ~30–35% air (ρ ≈ 600–640 kg/m³) the gas percolates
  and the bubbles-in-a-continuous-matrix topology itself fails — such ice
  belongs in ``layer_type=0`` (grains in air), not here.
- The inclusion microstructure is that of interior first-year ice and is
  applied to all ice; above a brine fraction of 0.5 it is held fixed.

References
----------
Cox & Weeks (1983); Leppäranta & Manninen (1988) — brine and gas volume.
Light, Maykut & Grenfell (2004) — brine-inclusion and salt-crystal scattering.
Maykut & Light (1995); Quan & Fry (1995) — brine refractive index.
Rowe et al. (2020) — water absorption.
Picard et al. (2016) — ice complex RI (variant 'Pic16').
"""

from typing import Tuple

import numpy as np

import biosnicar
from biosnicar.optical_properties.op_lookup import get_lut
from biosnicar.sea_ice.brine_volume import compute_brine_volume, compute_gas_volume
from biosnicar.sea_ice.brine_optics import compute_brine_rfidx
from biosnicar.sea_ice.effective_medium import effective_ri_ice_brine
from biosnicar.sea_ice.inclusion_scattering import inclusion_scattering

_DATA_DIR = biosnicar.DATA_DIR
_RFIDX_PATH = _DATA_DIR / "OP_data" / "480band" / "rfidx_ice.npz"
_BUBBLY_AIR_PATH = _DATA_DIR / "OP_data" / "480band" / "luts" / "bubbly_air.npz"

# BioSNICAR wavelength grid (μm)
WAVELENGTHS = np.arange(0.205, 4.999, 0.01)

# Minimum air volume fraction (avoids negative values from rounding).
# compute_gas_volume already clips at 0; this is belt-and-braces.
_MIN_AIR_FRAC = 0.0

_rfidx_cache = None


def _load_ice_ri(ri_variant: str = "Pic16") -> np.ndarray:
    """Return complex RI of pure ice at the 480-band grid."""
    global _rfidx_cache
    if _rfidx_cache is None:
        _rfidx_cache = np.load(str(_RFIDX_PATH))
    data = _rfidx_cache
    n_re = data[f"re_{ri_variant}"]
    n_im = data[f"im_{ri_variant}"]
    return n_re + 1j * n_im


def _nearest_lut_radius(radius_um: float, lut) -> int:
    """Round bubble radius to nearest entry in bubbly_air LUT.

    Requests outside the LUT grid are clamped to the nearest endpoint
    with a warning rather than silently snapped.
    """
    import warnings

    radii = lut.data["radii"]
    lo, hi = float(radii.min()), float(radii.max())
    if radius_um < lo or radius_um > hi:
        clamped = lo if radius_um < lo else hi
        warnings.warn(
            f"sea_ice_bubble_radius={radius_um:g} um is outside the "
            f"bubbly_air LUT range [{lo:g}, {hi:g}] um — clamped to "
            f"{clamped:g} um.",
            stacklevel=2,
        )
        radius_um = clamped
    idx = np.argmin(np.abs(radii - radius_um))
    return int(radii[idx])


def compute_sea_ice_optics(
    thickness_m: float,
    salinity_psu: float,
    temperature_C: float,
    density_kg_m3: float,
    bubble_radius_um: float,
    ri_variant: str = "Pic16",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute optical properties of a sea-ice layer.

    Args:
        thickness_m: Layer thickness in metres (> 0).
        salinity_psu: Bulk salinity in psu (0–15 typical).
        temperature_C: Temperature in °C (−44 to 0).
        density_kg_m3: Bulk sea-ice density in kg/m³.  Cold interior ice sits
            at 870–920; weathered, drained and gas-rich ice can fall to
            700–850 (Timco & Frederking 1996 report 720–940 for sea ice as a
            whole).  The shipped LUT spans 700–920 and *clamps* outside that,
            so values below 700 are silently treated as 700.
        bubble_radius_um: Effective radius of air bubbles in μm (typically
            100–1000 μm; controls scattering via the bubbly_air LUT).
        ri_variant: Pure-ice refractive index variant.  One of 'Pic16'
            (default), 'Wrn84', 'Wrn08'.

    Returns:
        (tau, ssa, g) — each a numpy array of length 480.
        tau: spectral optical depth of the layer (dimensionless).
        ssa: single-scattering albedo per wavelength (0–1).
        g:   asymmetry parameter per wavelength (−1 to 1).

    Raises:
        ValueError: If temperature or salinity are out of the valid range.
    """
    rho_si = density_kg_m3   # kg/m³

    # --- 1. Brine volume fraction ---
    # Pass the layer's bulk density: Cox & Weeks give brine and gas volume as
    # two halves of one density balance, so computing brine volume without the
    # density (i.e. assuming bubble-free ice) and then deriving the air
    # fraction from that same density is self-inconsistent.  It overestimated
    # nu_b by the gas fraction — 1–9% across the shipped presets.
    nu_b = compute_brine_volume(
        salinity_psu, temperature_C, bulk_density_kg_m3=rho_si
    )

    # Gas half of the same Cox & Weeks density balance.  Computed here rather
    # than after the absorption step, because both the brine concentration and
    # the absorption path length depend on it.
    nu_air = max(float(compute_gas_volume(salinity_psu, temperature_C, rho_si)),
                 _MIN_AIR_FRAC)

    # --- 2. Brine complex RI at the liquidus salinity for this temperature ---
    # The brine inside sea ice pockets is at the phase-equilibrium (liquidus)
    # salinity, determined by temperature alone.  Do NOT pass bulk salinity.
    ri_brine = compute_brine_rfidx(temperature_C)

    # --- 3. Pure-ice complex RI ---
    ri_ice = _load_ice_ri(ri_variant)

    # --- 4. Effective ice+brine RI (symmetric Bruggeman rule) ---
    # The rule mixes a *two-phase* condensed medium, so it needs the
    # brine fraction OF THAT MEDIUM, not of the whole layer.  nu_b is a
    # fraction of total volume, so divide out the air to get the brine
    # concentration within the ice+brine phase.
    nu_b_condensed = nu_b / (1.0 - nu_air) if nu_air < 1.0 else nu_b
    nu_b_condensed = min(nu_b_condensed, 1.0)
    ri_eff = effective_ri_ice_brine(ri_ice, ri_brine, nu_b_condensed)
    n_eff_im = np.maximum(np.imag(ri_eff), 0.0)

    # --- 5. Absorption coefficient of the layer (m^-1) ---
    # α = 4π k / λ is the absorption of the *condensed* phase.  Only
    # (1 - nu_air) of the layer is that phase, and air does not absorb, so the
    # layer's absorption per metre is reduced in proportion.
    #
    # Air is deliberately NOT mixed into the effective medium: effective-medium
    # rules assume inclusions small compared to the wavelength, and the bubbles are
    # 100–1000 µm against 0.2–5 µm light.  That is why they are treated as
    # discrete Mie scatterers (step 7).  For large non-absorbing inclusions the
    # correct treatment of absorption is path-length volume weighting, which is
    # also what layer_type 1/2/3 do via their mass absorption coefficients.
    lam_m = WAVELENGTHS * 1e-6     # μm → m
    abs_coeff = 4.0 * np.pi * n_eff_im / lam_m * (1.0 - nu_air)   # m^-1

    # --- 6. Air bubble volume fraction (computed in step 1) ---
    # nu_air is the gas half of the same Cox & Weeks density balance that
    # produced nu_b:  Va = 1 - ρ_si/ρ_i + ρ_si·S·F2/F1.
    #
    # This replaces an independent density balance
    #   ν_air = (ρ_ice - ρ_si + (ρ_brine - ρ_ice)·ν_b) / ρ_ice
    # which used brine_density() — a Frankenstein & Garner-derived liquidus —
    # while nu_b came from Cox & Weeks.  The two agreed to within ~0.6
    # percentage points, but they were different phase models describing the
    # same air, and only one of them was the one nu_b assumed.

    # --- 7. Scattering from air bubbles via bubbly_air LUT ---
    bubbly_lut = get_lut(str(_BUBBLY_AIR_PATH))
    r_lut = _nearest_lut_radius(bubble_radius_um, bubbly_lut)
    sca_cff_vlm = bubbly_lut.get(r_lut, "sca_cff_vlm")   # m^-1 per unit ν_air
    g_bubble = bubbly_lut.get(r_lut, "asm_prm")

    sca_bubble = sca_cff_vlm * nu_air   # m^-1

    # --- 7b. Scattering from brine inclusions and precipitated salts ---
    populations = inclusion_scattering(
        salinity_psu, temperature_C, rho_si, nu_b,
        np.real(ri_ice), np.real(ri_brine), WAVELENGTHS,
    )
    sca_coeff = sca_bubble.copy()
    sca_g = sca_bubble * g_bubble
    for sigma_i, g_i in populations.values():
        sca_coeff = sca_coeff + sigma_i
        sca_g = sca_g + sigma_i * g_i

    # --- 8. Total extinction and derived quantities ---
    ext_coeff = abs_coeff + sca_coeff   # m^-1

    # ssa: scattering / extinction
    with np.errstate(invalid="ignore", divide="ignore"):
        ssa = np.where(ext_coeff > 0, sca_coeff / ext_coeff, 0.0)
        # Scattering-weighted asymmetry parameter of the four populations
        # (Light et al. 2004, Eq. 6).
        g = np.where(sca_coeff > 0, sca_g / sca_coeff, g_bubble)

    # --- 9. Optical depth: τ = ext_coeff × thickness ---
    tau = ext_coeff * thickness_m

    # Physical constraints
    tau = np.maximum(tau, 0.0)
    ssa = np.clip(ssa, 1e-8, 1.0 - 1e-8)
    g = np.clip(g, -0.9999, 0.9999)

    return tau, ssa, g


def compute_sea_ice_optics_per_metre(
    salinity_psu: float,
    temperature_C: float,
    density_kg_m3: float,
    bubble_radius_um: float,
    ri_variant: str = "Pic16",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-metre optical properties (for LUT storage).

    Returns (tau_per_m, ssa, g) where tau_per_m = τ / thickness_m.
    ssa and g are intensive (independent of thickness).
    """
    tau, ssa, g = compute_sea_ice_optics(
        thickness_m=1.0,
        salinity_psu=salinity_psu,
        temperature_C=temperature_C,
        density_kg_m3=density_kg_m3,
        bubble_radius_um=bubble_radius_um,
        ri_variant=ri_variant,
    )
    return tau, ssa, g   # tau here == tau_per_m since thickness=1.0
