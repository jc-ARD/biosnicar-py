"""Spectral resampling utilities for field and drone spectrometer data.

Converts spectra from an instrument-native wavelength grid (e.g. ASD
350-2500 nm at 1 nm) onto BioSNICAR's 480-band model grid so they can be
passed to :func:`~biosnicar.sea_ice.retrieve.retrieve_sea_ice`.

Model bands outside the instrument's wavelength coverage are returned as
NaN — pass the matching boolean mask (``np.isfinite(spectrum)`` combined
with :func:`vis_wavelength_mask` if desired) as ``wavelength_mask`` to the
retrieval.
"""

import numpy as np

from biosnicar.bands._core import WVL

# Model grid in nm (WVL is µm)
MODEL_WAVELENGTHS_NM = WVL * 1000.0
_MODEL_SPACING_NM = 10.0


def resample_to_model_grid(spectrum, instrument_wavelengths,
                           method="gaussian", fwhm=None):
    """Resample an observed spectrum onto the 480-band model grid.

    Parameters
    ----------
    spectrum : np.ndarray
        Instrument albedo/reflectance values.
    instrument_wavelengths : np.ndarray
        Wavelengths in nm, same length as *spectrum*.
    method : str
        ``"gaussian"`` — spectral response convolution with the instrument
        FWHM (most physically correct for field spectrometers);
        ``"linear"`` — interpolation (adequate for narrow uniform bands);
        ``"box"`` — mean over each model band's ±5 nm half-width.
    fwhm : float or None
        Instrument FWHM in nm for the Gaussian method.  Defaults to the
        model band spacing (10 nm).

    Returns
    -------
    np.ndarray of shape (480,)
        Albedo on the model grid; NaN where the instrument has no coverage.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    wvl = np.asarray(instrument_wavelengths, dtype=float)
    if spectrum.shape != wvl.shape:
        raise ValueError(
            f"spectrum {spectrum.shape} and instrument_wavelengths "
            f"{wvl.shape} must have the same shape"
        )
    order = np.argsort(wvl)
    wvl, spectrum = wvl[order], spectrum[order]
    finite = np.isfinite(spectrum)
    wvl, spectrum = wvl[finite], spectrum[finite]
    if len(wvl) == 0:
        return np.full(480, np.nan)

    centers = MODEL_WAVELENGTHS_NM
    out = np.full(480, np.nan)
    covered = (centers >= wvl[0]) & (centers <= wvl[-1])

    if method == "linear":
        out[covered] = np.interp(centers[covered], wvl, spectrum)
        return out

    if method == "gaussian":
        sigma = (fwhm or _MODEL_SPACING_NM) / 2.355
        half_window = 3.0 * sigma
    elif method == "box":
        half_window = _MODEL_SPACING_NM / 2.0
    else:
        raise ValueError(f"unknown method {method!r}; use gaussian/linear/box")

    for i in np.flatnonzero(covered):
        lo = np.searchsorted(wvl, centers[i] - half_window, side="left")
        hi = np.searchsorted(wvl, centers[i] + half_window, side="right")
        if hi <= lo:
            continue
        seg_w, seg_s = wvl[lo:hi], spectrum[lo:hi]
        if method == "gaussian":
            w = np.exp(-0.5 * ((seg_w - centers[i]) / sigma) ** 2)
            out[i] = float(np.sum(w * seg_s) / np.sum(w))
        else:
            out[i] = float(np.mean(seg_s))
    return out


def vis_wavelength_mask():
    """Boolean (480,) mask selecting the 400-1000 nm bands."""
    return (MODEL_WAVELENGTHS_NM >= 400.0) & (MODEL_WAVELENGTHS_NM <= 1000.0)


def trim_to_vis(spectrum_480):
    """Return the VIS-only (400-1000 nm) subset of a 480-band spectrum.

    Use when preparing bare ice observations for classification — adding
    SWIR bands degrades bare ice accuracy (see docs/SEA_ICE_EMULATOR.md).
    For retrieval, prefer passing ``vis_wavelength_mask()`` as the
    ``wavelength_mask`` argument so band alignment is preserved.
    """
    spectrum_480 = np.asarray(spectrum_480)
    if spectrum_480.shape[-1] != 480:
        raise ValueError(f"expected 480 bands, got {spectrum_480.shape}")
    return spectrum_480[..., vis_wavelength_mask()]


def estimate_instrument_uncertainty(spectrum, snr_model):
    """Per-band 1-sigma obs_uncertainty from a simple SNR model.

    Parameters
    ----------
    spectrum : np.ndarray
        Albedo values (any length).
    snr_model : float or dict or callable
        Scalar signal-to-noise ratio; or ``{"snr": value, "floor": value}``
        adding an additive noise floor (albedo units, default 0.001);
        or a callable ``snr(wavelength_index) -> float`` array-compatible.

    Returns
    -------
    np.ndarray
        1-sigma uncertainty per band: ``|albedo| / snr + floor``.
    """
    spectrum = np.asarray(spectrum, dtype=float)
    floor = 0.001
    if callable(snr_model):
        snr = np.asarray(snr_model(np.arange(len(spectrum))), dtype=float)
    elif isinstance(snr_model, dict):
        snr = float(snr_model["snr"])
        floor = float(snr_model.get("floor", floor))
    else:
        snr = float(snr_model)
    return np.abs(spectrum) / snr + floor
