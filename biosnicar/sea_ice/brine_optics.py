"""Brine complex refractive index as a function of temperature.

Physics
-------
Brine in sea ice pockets is at the **liquidus** (phase-equilibrium) salinity,
which is determined by temperature alone:

    S_brine ≈ -18.7 × T_C  (psu)           (linear liquidus approximation)

At T = -10 °C this gives S_brine ≈ 187 psu — roughly 23× the typical bulk
ice salinity of 8 psu.  Using the bulk salinity in RI calculations (as in
v0.1) underestimates the brine-ice optical contrast by up to 50×.

The corrected implementation computes brine RI at the liquidus salinity,
then applies Maxwell-Garnett mixing at the appropriate brine volume fraction
(which still depends on both bulk salinity and temperature).

Method
------
Starting point: pure liquid water RI at 0 °C from Rowe et al. (2020),
on the BioSNICAR 480-band grid (0.205–4.995 µm).

Real part (refractive index):
  400–700 nm:  Full Quan & Fry (1995) formula for the salt contribution
               at the liquidus brine salinity.  Calibrated for T ∈ [0, 30 °C]
               and S ∈ [0, 40 psu]; extrapolated to sub-zero T and high S.
               The formula gives the TOTAL n including water; we subtract the
               pure-water value to isolate the salt contribution, then add it
               to the Rowe2020 water RI (which is more accurate at 0 °C).
  >700 nm:     Salt contribution fixed at its 700-nm value from Q&F; water RI
               from Rowe2020.  (Salt ions contribute only weakly to NIR real RI.)

Imaginary part (absorption index k):
  0–400 nm:    Additional ionic UV absorption (Cl⁻ electronic band) decaying
               exponentially into the visible.
  400–700 nm:  Very small salt absorption; linear correction ∝ S_brine.
  >700 nm:     Dominated by O-H overtone bands of water; essentially independent
               of salt concentration.  k_brine ≈ k_water in this range.

Temperature correction (Pegau et al. 1997):
  Small additive correction to k applied across the visible range,
  scaled by (-T) relative to the 0 °C reference.

Known limitations
-----------------
- Quan & Fry (1995) calibrated to S ≤ 40 psu; extrapolated to 40–500 psu
  (typical brine range).  Error in real-part correction estimated < 10% based
  on the Lorentz-Lorenz mixing-rule comparison.
- Liquidus formula S_b = -18.7 T is a linear approximation; real seawater
  liquidus is slightly nonlinear.
- Imaginary-part ionic correction is an order-of-magnitude estimate.
- A rigorous treatment requires direct laboratory measurement of brine RI
  at sub-zero temperatures and high salinities (e.g. Friedlander et al. 2022).

References
----------
Rowe, P. M. et al. (2020). *J. Geophys. Res.*, 125, e2019JD031822.
Quan, X. & Fry, E. S. (1995). *Applied Optics*, 34, 3477.
Pegau, W. S., Gray, D. & Zaneveld, J. R. V. (1997). *Limnol. Oceanogr.*, 42(3).
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import biosnicar
from biosnicar.sea_ice.brine_volume import brine_salinity_at_temp

_DATA_DIR   = biosnicar.DATA_DIR
_ROWE_PATH  = _DATA_DIR / "OP_data" / "480band" / "refractive_index_water_273K_Rowe2020.csv"
_LUT_PATH   = _DATA_DIR / "OP_data" / "brine_rfidx.npz"

# BioSNICAR 480-band wavelength grid (µm)
WAVELENGTHS    = np.arange(0.205, 4.999, 0.01)
WAVELENGTHS_NM = WAVELENGTHS * 1000.0   # nm, for Quan & Fry

# LUT temperature grid
_T_GRID = np.array([-2, -5, -10, -15, -20, -25, -30], dtype=float)   # °C

# Quan & Fry (1995) coefficients — real part
_QF = dict(n0=1.31405, n1=1.779e-4, n2=-1.05e-6, n3=1.6e-8,
           n4=-2.02e-6, n5=15.868, n6=0.01155, n7=-0.00423,
           n8=-4382.0, n9=1.1455e6)

# Imaginary-part parameters
_K_IONIC_AMPLITUDE = 1.0e-8    # Cl⁻ UV absorption at 0.2 µm per psu
_K_IONIC_DECAY     = 25.0      # e-folding scale (µm⁻¹) — Cl⁻ band zero above ~0.4 µm
# NaCl has NO absorption above 400 nm — _K_VIS_ALPHA is intentionally 0.
# A non-zero value was found to introduce unphysically large visible absorption
# when extrapolated to liquidus brine salinities (100–400 psu).
_K_VIS_ALPHA       = 0.0

# Cap on liquidus salinity for optical calculations.
# Beyond ~16°C below freezing the NaCl eutectic is approached; the linear
# liquidus S_b = -18.7T overestimates actual brine salinity.  Capping at
# 250 psu (≈ T = -13°C) limits extrapolation error in the Q&F formula.
_S_BRINE_CAP = 250.0   # psu

_lut_interp_re: Optional[object] = None
_lut_interp_im: Optional[object] = None


# ---------------------------------------------------------------------------
# Quan & Fry (1995) real-part formula
# ---------------------------------------------------------------------------

def _qf_real(S_psu: float, T_C: float, lam_nm: np.ndarray) -> np.ndarray:
    """Full Quan & Fry (1995) real RI formula for saline water.

    Returns n(S, T, λ).  Valid range: T ∈ [0,30]°C, S ∈ [0,40] psu,
    λ ∈ [400,700] nm; extrapolated outside calibration range.
    """
    c = _QF
    return (c["n0"]
            + (c["n1"] + c["n2"]*T_C + c["n3"]*T_C**2) * S_psu
            + c["n4"] * T_C**2
            + (c["n5"] + c["n6"]*S_psu + c["n7"]*T_C) / lam_nm
            + c["n8"] / lam_nm**2
            + c["n9"] / lam_nm**3)


def _salt_delta_re(S_brine: float, T_C: float,
                   wvl_um: np.ndarray) -> np.ndarray:
    """Salt contribution to real RI: Δn = n_saline - n_pure_water.

    Uses Quan & Fry in 400–700 nm.  Outside that range, uses the value at
    the boundary wavelength (salt contribution is weakly wavelength-dependent
    in NIR, and extrapolation below 400 nm is less reliable).
    """
    lam = wvl_um * 1000.0   # µm → nm

    # Q&F at brine salinity and temperature
    n_brine_qf = _qf_real(S_brine, T_C, lam)
    # Q&F at S=0 (pure water)
    n_water_qf = _qf_real(0.0, T_C, lam)
    # Salt contribution (removing pure-water RI to avoid double-counting)
    delta = n_brine_qf - n_water_qf

    # Clamp to the 700-nm value above 700 nm; clamp to 400-nm value below 400 nm
    vis_mask = (wvl_um >= 0.4) & (wvl_um <= 0.7)
    if vis_mask.any():
        idx_700 = np.argmin(np.abs(wvl_um - 0.70))
        idx_400 = np.argmin(np.abs(wvl_um - 0.40))
        delta[wvl_um > 0.70] = delta[idx_700]
        delta[wvl_um < 0.40] = delta[idx_400]

    return delta


# ---------------------------------------------------------------------------
# Imaginary part correction
# ---------------------------------------------------------------------------

def _salt_delta_im(S_brine: float, T_C: float,
                   k_water: np.ndarray, wvl_um: np.ndarray) -> np.ndarray:
    """Salt + temperature contribution to imaginary RI.

    Components:
      1. UV ionic absorption (Cl⁻ band, decays exponentially above 0.2 µm)
      2. Small residual visible absorption (negligible in NIR)
      3. Temperature correction relative to 0 °C reference
    """
    # 1. UV ionic absorption (Cl⁻ electronic band)
    uv_factor = np.exp(-_K_IONIC_DECAY * (wvl_um - 0.2))
    uv_factor = np.clip(uv_factor, 0.0, 1.0)
    k_ionic = _K_IONIC_AMPLITUDE * S_brine * uv_factor

    # 2. Residual visible (VIS only, ~0 in NIR)
    vis_mask = (wvl_um <= 0.7).astype(float)
    k_vis = _K_VIS_ALPHA * S_brine * vis_mask

    # 3. Temperature correction: water absorbs more at sub-zero temperatures
    #    in visible range (Pegau et al. 1997); effect is small in NIR
    # NB: no explicit temperature correction — Pegau et al. (1997) report
    # near-zero visible dα/dT outside narrow shoulder bands, and the term
    # previously here had the wrong sign at <0.8% magnitude.
    return k_ionic + k_vis


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _load_water_ri():
    df = pd.read_csv(_ROWE_PATH)
    return df["n"].values.astype(float), df["k"].values.astype(float)


def _compute_brine_ri(temperature_C: float,
                      n_re_water: np.ndarray,
                      n_im_water: np.ndarray) -> tuple:
    """Compute brine (n_re, n_im) at the liquidus brine salinity."""
    T = float(temperature_C)
    # Brine is at the phase-boundary (liquidus) salinity, capped at eutectic
    S_b = min(float(brine_salinity_at_temp(T)), _S_BRINE_CAP)

    n_re = n_re_water + _salt_delta_re(S_b, T, WAVELENGTHS)
    n_im = n_im_water + _salt_delta_im(S_b, T, n_im_water, WAVELENGTHS)

    n_re = np.maximum(n_re, 1.0)
    n_im = np.maximum(n_im, 0.0)
    return n_re, n_im


# ---------------------------------------------------------------------------
# LUT build and load
# ---------------------------------------------------------------------------

def build_brine_rfidx_lut() -> str:
    """Pre-compute brine RI over the temperature grid and save to brine_rfidx.npz.

    The LUT stores brine RI at the liquidus salinity for each temperature in
    _T_GRID.  Shape: T_grid(nT,), n_re(nT, 480), n_im(nT, 480).
    """
    n_re_water, n_im_water = _load_water_ri()
    nT, nW = len(_T_GRID), len(WAVELENGTHS)

    lut_re = np.zeros((nT, nW))
    lut_im = np.zeros((nT, nW))

    for j, T in enumerate(_T_GRID):
        re, im = _compute_brine_ri(T, n_re_water, n_im_water)
        lut_re[j, :] = re
        lut_im[j, :] = im

    np.savez_compressed(
        str(_LUT_PATH),
        T_grid=_T_GRID,
        n_re=lut_re,
        n_im=lut_im,
    )
    return str(_LUT_PATH)


def _load_lut():
    global _lut_interp_re, _lut_interp_im
    if _lut_interp_re is not None:
        return

    if not _LUT_PATH.exists():
        build_brine_rfidx_lut()

    data = np.load(str(_LUT_PATH))
    T_grid = data["T_grid"]

    # Ensure ascending for interpolation
    if T_grid[0] > T_grid[-1]:
        T_grid = T_grid[::-1]
        lut_re = data["n_re"][::-1, :]
        lut_im = data["n_im"][::-1, :]
    else:
        lut_re = data["n_re"]
        lut_im = data["n_im"]

    from scipy.interpolate import RegularGridInterpolator
    _lut_interp_re = RegularGridInterpolator(
        (T_grid, WAVELENGTHS), lut_re,
        method="linear", bounds_error=False, fill_value=None,
    )
    _lut_interp_im = RegularGridInterpolator(
        (T_grid, WAVELENGTHS), lut_im,
        method="linear", bounds_error=False, fill_value=None,
    )


def compute_brine_rfidx(
    temperature_C: float,
    salinity_psu: Optional[float] = None,   # deprecated; ignored
    wavelengths_um: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute complex refractive index of brine at given temperature.

    Brine in sea ice is at the liquidus (phase-equilibrium) salinity,
    determined by temperature alone.  The ``salinity_psu`` argument is
    accepted for backwards compatibility but is **ignored** — it referred
    to bulk ice salinity, not brine salinity, and caused a 23–47× error
    in the optical correction.

    Args:
        temperature_C: Ice temperature in °C (−30 to −2).
        salinity_psu: Deprecated; ignored.  Brine salinity is computed
            internally from the liquidus relationship.
        wavelengths_um: Wavelength grid in µm.  Defaults to the BioSNICAR
            480-band grid (0.205–4.995 µm).

    Returns:
        Complex RI array n + ik, shape (n_wavelengths,).
        Imaginary part ≥ 0 (absorbing medium convention).
    """
    if wavelengths_um is None:
        wavelengths_um = WAVELENGTHS

    T = float(temperature_C)
    T_c = float(np.clip(T, _T_GRID.min(), _T_GRID.max()))

    _load_lut()

    pts = np.column_stack([
        np.full(len(wavelengths_um), T_c),
        wavelengths_um,
    ])
    n_re = _lut_interp_re(pts)
    n_im = np.maximum(_lut_interp_im(pts), 0.0)

    # If wavelengths differ from the LUT grid, fall back to direct calculation
    if (wavelengths_um.shape != WAVELENGTHS.shape
            or not np.allclose(wavelengths_um, WAVELENGTHS, atol=1e-4)):
        n_re_w, n_im_w = _load_water_ri()
        f_re = interp1d(WAVELENGTHS, n_re_w, kind="linear",
                        bounds_error=False, fill_value="extrapolate")
        f_im = interp1d(WAVELENGTHS, n_im_w, kind="linear",
                        bounds_error=False, fill_value="extrapolate")
        n_re_wvl = f_re(wavelengths_um)
        n_im_wvl = np.maximum(f_im(wavelengths_um), 0.0)
        S_b = min(float(brine_salinity_at_temp(T)), _S_BRINE_CAP)
        n_re = n_re_wvl + _salt_delta_re(S_b, T, wavelengths_um)
        n_im = np.maximum(
            n_im_wvl + _salt_delta_im(S_b, T, n_im_wvl, wavelengths_um), 0.0
        )

    return n_re + 1j * n_im


def invalidate_lut_cache():
    global _lut_interp_re, _lut_interp_im
    _lut_interp_re = None
    _lut_interp_im = None
