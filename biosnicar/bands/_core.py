"""Shared utilities for platform band convolution.

Provides the authoritative 480-band wavelength grid, SRF loading/caching,
SRF-weighted convolution, and flux-weighted interval averaging.
"""

import csv
import numpy as np
from pathlib import Path

from biosnicar import DATA_DIR

# Authoritative 480-band wavelength grid (µm), mirrors model_config.wavelengths
WVL = np.arange(0.205, 4.999, 0.01)  # shape (480,)
N_WVL = len(WVL)

# Module-level SRF cache: {sensor_name: {band_name: np.array(480)}}
_srf_cache = {}

# Stacked-SRF cache: {(sensor_name, band_order): np.array (n_bands, 480)}
_srf_stack_cache = {}


def load_srf(sensor_name):
    """Load a CSV spectral response function and cache it.

    The CSV must live at ``data/band_srfs/{sensor_name}.csv`` with columns
    ``wavelength_um,B1,B2,...``.  Values are interpolated onto :data:`WVL` if
    needed (though generated tophats are already on-grid).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of band name → SRF array (480,).
    """
    if sensor_name in _srf_cache:
        return _srf_cache[sensor_name]

    path = DATA_DIR / "band_srfs" / f"{sensor_name}.csv"
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    band_names = header[1:]

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    wvl_csv = data[:, 0]
    srf_dict = {}
    for i, bname in enumerate(band_names):
        raw = data[:, i + 1]
        if len(wvl_csv) == N_WVL and np.allclose(wvl_csv, WVL, atol=1e-6):
            srf_dict[bname] = raw
        else:
            srf_dict[bname] = np.interp(WVL, wvl_csv, raw, left=0.0, right=0.0)

    _srf_cache[sensor_name] = srf_dict
    return srf_dict


def srf_convolve(albedo, flx_slr, srf):
    """Flux-weighted SRF convolution for a single band.

    Parameters
    ----------
    albedo : np.ndarray (480,)
        Spectral albedo from BioSNICAR.
    flx_slr : np.ndarray (480,)
        Spectral solar flux (W m-2 per band).
    srf : np.ndarray (480,)
        Spectral response function (dimensionless, 0–1).

    Returns
    -------
    float
        Band-averaged albedo: Σ(albedo × SRF × flx) / Σ(SRF × flx).
    """
    weight = srf * flx_slr
    denom = np.sum(weight)
    if denom == 0:
        return np.nan
    return float(np.sum(albedo * weight) / denom)


def load_srf_stack(sensor_name, band_names):
    """Load several bands' SRFs as one stacked ``(n_bands, 480)`` array.

    Cached per ``(sensor_name, band_names)`` so the stack is built once.  Rows
    follow the order of *band_names*; pass the result to
    :func:`srf_convolve_stack` to convolve every band in a single matmul.

    Returns
    -------
    np.ndarray (n_bands, 480)
    """
    key = (sensor_name, tuple(band_names))
    if key not in _srf_stack_cache:
        srf = load_srf(sensor_name)
        _srf_stack_cache[key] = np.array([srf[b] for b in band_names])
    return _srf_stack_cache[key]


def srf_convolve_stack(albedo, flx_slr, srf_stack):
    """Flux-weighted SRF convolution for a stack of bands at once.

    Vectorised equivalent of :func:`srf_convolve`, applied row-wise to
    ``srf_stack`` of shape ``(n_bands, 480)``.  Bands with zero flux-weight are
    returned as NaN, matching the single-band case.

    Parameters
    ----------
    albedo : np.ndarray (480,)
    flx_slr : np.ndarray (480,)
    srf_stack : np.ndarray (n_bands, 480)

    Returns
    -------
    np.ndarray (n_bands,)
        Band-averaged albedo per row.
    """
    weight = srf_stack * flx_slr
    denom = weight.sum(axis=1)
    num = weight @ np.asarray(albedo, dtype=float)
    return np.divide(num, denom, out=np.full(denom.shape, np.nan), where=denom != 0)


def interval_average(albedo, flx_slr, lo_um, hi_um):
    """Flux-weighted mean albedo over [lo_um, hi_um).

    Parameters
    ----------
    albedo : np.ndarray (480,)
    flx_slr : np.ndarray (480,)
    lo_um, hi_um : float
        Wavelength bounds in µm.

    Returns
    -------
    float
        Flux-weighted mean albedo in the interval.
    """
    mask = (WVL >= lo_um) & (WVL < hi_um)
    flx = flx_slr[mask]
    denom = np.sum(flx)
    if denom == 0:
        return np.nan
    return float(np.sum(albedo[mask] * flx) / denom)
