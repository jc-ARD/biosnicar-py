"""Brine complex refractive index as a function of salinity and temperature.

Approximation method
--------------------
Starting point: pure liquid water RI at 273 K (Rowe et al. 2020), already
interpolated to the BioSNICAR 480-band grid (0.205–4.995 μm).

Salinity corrections applied per band:
  Real part:  Δn_re = +2.0e-4 × S   (Quan & Fry 1995, linear fit to their
              Eq. 9 evaluated at 589 nm, extended across all wavelengths;
              negligible wavelength dependence for our purposes)
  Imaginary:  k_brine = k_water × (1 + α_abs × S)  with α_abs = 5e-4 /psu
              (order-of-magnitude estimate; absorption of seawater salts
              becomes significant mainly at UV wavelengths)

Temperature corrections applied per band:
  Real part:  Δn_re = -1.0e-5 × (T_C - 0)   (very small; seawater dnT/dT
              is ~-1e-4 per °C in visible, but we start from 0°C reference)
  Imaginary:  Δk = +2.5e-4 × (0 - T_C) × k_water / k_water_ref
              simplified from Pegau et al. (1997) for 400–700 nm; extended
              to NIR with no additional correction (NIR k dominated by
              O-H bond overtones, weakly T-dependent in this range)

Known limitations
-----------------
- Corrections calibrated on seawater at ~35 psu; brine salinity can reach
  100–200 psu in cold sea ice.  Extrapolation at high S is uncertain.
- Temperature correction calibrated for liquid seawater (> 0°C); we
  extrapolate to sub-zero temperatures where brine remains liquid.
- Refractive index of brine differs from that of dilute seawater due to ion
  effects not captured by the linear scaling used here.
- These approximations are adequate for the MVP target (match SHEBA reference
  spectra within broadband albedo error < 0.05); a more rigorous treatment
  requires direct measurement of brine optical constants.

References
----------
Rowe, P. M. et al. (2020): refractive index of liquid water 0.2–5 μm at 0°C.
Quan, X. & Fry, E. S. (1995): Empirical equation for the refractive index of
    seawater. *Applied Optics*, 34, 3477.
Pegau, W. S., Gray, D. & Zaneveld, J. R. V. (1997): Absorption and
    attenuation of visible and near-infrared light in water. *Limnology and
    Oceanography*, 42(3), 443–452.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

import biosnicar

_DATA_DIR = biosnicar.DATA_DIR
_ROWE_PATH = _DATA_DIR / "OP_data" / "480band" / "refractive_index_water_273K_Rowe2020.csv"
_LUT_PATH = _DATA_DIR / "OP_data" / "brine_rfidx.npz"

# BioSNICAR 480-band wavelength grid (μm)
WAVELENGTHS = np.arange(0.205, 4.999, 0.01)

# Pre-computation grids
_S_GRID = np.array([0, 10, 20, 50, 100, 150, 200], dtype=float)   # psu
_T_GRID = np.array([-2, -5, -10, -15, -20, -25, -30], dtype=float)  # °C

# Salinity corrections
_ALPHA_RE = 2.0e-4   # Δn_re per psu (Quan & Fry 1995, extended)
_ALPHA_ABS = 5.0e-4  # multiplicative k correction per psu

# Temperature corrections (relative to 0°C reference)
_BETA_RE = -1.0e-5   # Δn_re per °C
_BETA_ABS = 2.5e-4   # Δk per °C × (k/k_ref) — applied as additive shift

_lut_interp_re: Optional[RegularGridInterpolator] = None
_lut_interp_im: Optional[RegularGridInterpolator] = None


def _load_water_ri():
    """Load Rowe2020 pure water RI at the 480-band grid. Returns (n_re, n_im)."""
    df = pd.read_csv(_ROWE_PATH)
    # columns: wvl, k, n
    n_re = df["n"].values.astype(float)
    n_im = df["k"].values.astype(float)
    return n_re, n_im


def _compute_brine_ri_raw(salinity_psu, temperature_C, n_re_water, n_im_water):
    """Compute brine RI for scalar S, T given base water RI arrays."""
    S = float(salinity_psu)
    T = float(temperature_C)

    # Salinity correction
    delta_re_s = _ALPHA_RE * S
    k_factor = 1.0 + _ALPHA_ABS * S

    # Temperature correction (reference is 0°C)
    delta_re_t = _BETA_RE * T          # T is negative → small positive shift
    delta_k_t = _BETA_ABS * (-T)       # -T > 0 → slight k increase at colder T

    n_re = n_re_water + delta_re_s + delta_re_t
    n_im = n_im_water * k_factor + delta_k_t * n_im_water

    # Physical constraints
    n_re = np.maximum(n_re, 1.0)
    n_im = np.maximum(n_im, 0.0)

    return n_re, n_im


def build_brine_rfidx_lut():
    """Pre-compute brine RI over (S, T) grid and save to brine_rfidx.npz.

    Output arrays:
        S_grid:  shape (nS,)
        T_grid:  shape (nT,)
        n_re:    shape (nS, nT, 480)
        n_im:    shape (nS, nT, 480)
    """
    n_re_water, n_im_water = _load_water_ri()
    nS, nT, nW = len(_S_GRID), len(_T_GRID), len(WAVELENGTHS)

    lut_re = np.zeros((nS, nT, nW))
    lut_im = np.zeros((nS, nT, nW))

    for i, S in enumerate(_S_GRID):
        for j, T in enumerate(_T_GRID):
            re, im = _compute_brine_ri_raw(S, T, n_re_water, n_im_water)
            lut_re[i, j, :] = re
            lut_im[i, j, :] = im

    np.savez_compressed(
        str(_LUT_PATH),
        S_grid=_S_GRID,
        T_grid=_T_GRID,
        n_re=lut_re,
        n_im=lut_im,
    )
    return str(_LUT_PATH)


def _load_lut():
    """Load (or build) the brine RI LUT and cache interpolators."""
    global _lut_interp_re, _lut_interp_im
    if _lut_interp_re is not None:
        return

    if not _LUT_PATH.exists():
        build_brine_rfidx_lut()

    data = np.load(str(_LUT_PATH))
    S_grid = data["S_grid"]
    T_grid = data["T_grid"]
    lut_re = data["n_re"]  # (nS, nT, 480)
    lut_im = data["n_im"]

    # Note: T_grid is in decreasing order (−2 to −30).
    # RegularGridInterpolator requires strictly increasing axes, so flip T.
    T_asc = T_grid[::-1]
    lut_re_asc = lut_re[:, ::-1, :]
    lut_im_asc = lut_im[:, ::-1, :]
    wvl = WAVELENGTHS

    _lut_interp_re = RegularGridInterpolator(
        (S_grid, T_asc, wvl), lut_re_asc, method="linear", bounds_error=False,
        fill_value=None,
    )
    _lut_interp_im = RegularGridInterpolator(
        (S_grid, T_asc, wvl), lut_im_asc, method="linear", bounds_error=False,
        fill_value=None,
    )


def compute_brine_rfidx(
    salinity_psu: float,
    temperature_C: float,
    wavelengths_um: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute complex refractive index of brine at given S and T.

    Interpolates the pre-computed brine RI LUT. If the LUT does not exist it
    is built on first call (adds ~0.5 s; subsequent calls are fast).

    Args:
        salinity_psu: Brine salinity in psu (0–200).
        temperature_C: Temperature in °C (−30 to −2).
        wavelengths_um: Wavelength grid in μm. Defaults to the BioSNICAR
            480-band grid (0.205–4.995 μm).

    Returns:
        Complex refractive index array of shape (n_wavelengths,):
            n_real + 1j * n_imag.  Imaginary part is non-negative (absorbing
            medium convention: n + ik, k ≥ 0).
    """
    if wavelengths_um is None:
        wavelengths_um = WAVELENGTHS

    S = float(salinity_psu)
    T = float(temperature_C)

    # Fast path: direct computation (avoids LUT for scalar calls in scripts)
    if S in _S_GRID and T in _T_GRID:
        _load_lut()
        pts = np.column_stack([
            np.full(len(wavelengths_um), S),
            np.full(len(wavelengths_um), T),
            wavelengths_um,
        ])
        n_re = _lut_interp_re(pts)
        n_im = _lut_interp_im(pts)
    else:
        # Clamp to LUT bounds, then interpolate
        _load_lut()
        S_c = np.clip(S, _S_GRID[0], _S_GRID[-1])
        T_c = np.clip(T, _T_GRID[-1], _T_GRID[0])   # T_GRID[-1] is most negative
        pts = np.column_stack([
            np.full(len(wavelengths_um), S_c),
            np.full(len(wavelengths_um), T_c),
            wavelengths_um,
        ])
        n_re = _lut_interp_re(pts)
        n_im = _lut_interp_im(pts)

    n_im = np.maximum(n_im, 0.0)
    return n_re + 1j * n_im


def invalidate_lut_cache():
    """Reset cached LUT interpolators (useful when LUT is rebuilt)."""
    global _lut_interp_re, _lut_interp_im
    _lut_interp_re = None
    _lut_interp_im = None
