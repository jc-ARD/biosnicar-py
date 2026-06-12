"""Per-layer optical properties for sea ice (τ, ω, g) at 480 bands.

Physics pipeline
----------------
1. Brine volume fraction ν_b via Cox & Weeks (1983).
2. Brine complex RI via `brine_optics.compute_brine_rfidx`.
3. Pure-ice complex RI from the Picard 2016 variant in rfidx_ice.npz.
4. Effective ice+brine RI via Maxwell-Garnett mixing.
5. Absorption coefficient from imaginary part of effective RI.
6. Scattering from air bubbles: volume fraction inferred from bulk density
   and brine fraction; scattering coefficients from the bubbly_air LUT
   (pre-computed for air bubbles in pure ice — valid MVP approximation;
   the effective-medium correction to bubble scattering is < 3% for
   ν_b < 0.15 and is deferred to a future version).
7. Combined (τ, ω, g) per metre of sea ice thickness.

Approximations documented here
-------------------------------
- Air bubble scattering uses the pure-ice bubbly_air LUT.  The correct
  calculation would re-run Mie for bubbles in the ice+brine effective medium.
  For typical sea ice conditions (ν_b < 0.15, moderate bubble sizes) the
  error in scattering efficiency is < 5%.
- Brine RI corrections are calibrated on seawater near 35 psu and
  extrapolated to brine salinities up to ~200 psu.  See brine_optics.py.
- Gas (air) inside brine pockets is neglected; sea ice gas pockets are much
  smaller than air bubbles and contribute negligible scattering.

References
----------
Cox & Weeks (1983) — brine volume.
Rowe et al. (2020) — water RI base.
Picard et al. (2016) — ice complex RI (variant 'Pic16').
Light et al. (2004) — sea ice radiative transfer benchmark.
"""

from typing import Optional, Tuple

import numpy as np

import biosnicar
from biosnicar.optical_properties.op_lookup import get_lut
from biosnicar.sea_ice.brine_volume import brine_density, compute_brine_volume
from biosnicar.sea_ice.brine_optics import compute_brine_rfidx
from biosnicar.sea_ice.effective_medium import effective_ri_ice_brine

_DATA_DIR = biosnicar.DATA_DIR
_RFIDX_PATH = _DATA_DIR / "OP_data" / "480band" / "rfidx_ice.npz"
_BUBBLY_AIR_PATH = _DATA_DIR / "OP_data" / "480band" / "luts" / "bubbly_air.npz"

# BioSNICAR wavelength grid (μm)
WAVELENGTHS = np.arange(0.205, 4.999, 0.01)

# Pure-ice density (g/cm³ = 917 kg/m³)
_RHO_ICE = 917.0

# Minimum air volume fraction (avoids negative values from rounding)
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
        temperature_C: Temperature in °C (−44 to −2).
        density_kg_m3: Bulk sea-ice density in kg/m³ (typically 870–920).
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
    nu_b = compute_brine_volume(salinity_psu, temperature_C)

    # --- 2. Brine complex RI at the liquidus salinity for this temperature ---
    # The brine inside sea ice pockets is at the phase-equilibrium (liquidus)
    # salinity, determined by temperature alone.  Do NOT pass bulk salinity.
    ri_brine = compute_brine_rfidx(temperature_C)

    # --- 3. Pure-ice complex RI ---
    ri_ice = _load_ice_ri(ri_variant)

    # --- 4. Effective ice+brine RI via Maxwell-Garnett ---
    ri_eff = effective_ri_ice_brine(ri_ice, ri_brine, nu_b)
    n_eff_re = np.real(ri_eff)
    n_eff_im = np.maximum(np.imag(ri_eff), 0.0)

    # --- 5. Absorption coefficient of the effective medium (m^-1) ---
    # α = 4π k / λ   [k = imaginary RI, λ in metres]
    lam_m = WAVELENGTHS * 1e-6     # μm → m
    abs_coeff = 4.0 * np.pi * n_eff_im / lam_m   # m^-1

    # --- 6. Air bubble volume fraction ---
    # Density balance: ρ_si = ρ_ice × ν_ice + ρ_brine × ν_b + 0 × ν_air
    # ν_ice + ν_b + ν_air = 1
    # → ν_air = (ρ_ice - ρ_si + (ρ_brine - ρ_ice) × ν_b) / ρ_ice
    rho_b_kg = brine_density(temperature_C) * 1000.0   # g/cm³ → kg/m³
    nu_air = (_RHO_ICE - rho_si + (rho_b_kg - _RHO_ICE) * nu_b) / _RHO_ICE
    nu_air = max(float(nu_air), _MIN_AIR_FRAC)

    # --- 7. Scattering from air bubbles via bubbly_air LUT ---
    bubbly_lut = get_lut(str(_BUBBLY_AIR_PATH))
    r_lut = _nearest_lut_radius(bubble_radius_um, bubbly_lut)
    sca_cff_vlm = bubbly_lut.get(r_lut, "sca_cff_vlm")   # m^-1 per unit ν_air
    g_bubble = bubbly_lut.get(r_lut, "asm_prm")

    sca_coeff = sca_cff_vlm * nu_air   # m^-1

    # --- 8. Total extinction and derived quantities ---
    ext_coeff = abs_coeff + sca_coeff   # m^-1

    # ssa: scattering / extinction
    with np.errstate(invalid="ignore", divide="ignore"):
        ssa = np.where(ext_coeff > 0, sca_coeff / ext_coeff, 0.0)

    g = g_bubble   # asymmetry parameter (from bubble Mie; ice absorption ~0 asymmetry)

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
