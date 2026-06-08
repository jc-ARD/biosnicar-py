"""Platform band convolution for BioSNICAR spectral albedo.

Maps 480-band spectral albedo onto satellite or GCM band spaces via
SRF convolution (satellites) or flux-weighted interval averaging (GCMs).

Example
-------
>>> from biosnicar import run_model
>>> from biosnicar.bands import to_platform
>>> o = run_model(solzen=50, rds=1000)
>>> s2 = to_platform(o.albedo, "sentinel2", flx_slr=o.flx_slr)
>>> print(s2.B3, s2.NDSI)
"""

import numpy as np


class BandResult:
    """Container for platform-convolved band albedos and spectral indices.

    Attributes are set dynamically by each platform module.  Use
    :meth:`as_dict` for a flat dict or iterate ``band_names`` / ``index_names``.
    """

    def __init__(self, platform):
        self.platform = platform
        self.band_names = []
        self.index_names = []

    def _set_band(self, name, value):
        setattr(self, name, value)
        self.band_names.append(name)

    def _set_index(self, name, value):
        setattr(self, name, value)
        self.index_names.append(name)

    def as_dict(self):
        """Return all bands and indices as a flat dict."""
        d = {}
        for n in self.band_names:
            d[n] = getattr(self, n)
        for n in self.index_names:
            d[n] = getattr(self, n)
        return d

    def __repr__(self):
        lines = [f"BandResult(platform={self.platform!r})"]
        if self.band_names:
            lines.append("  Bands:")
            for n in self.band_names:
                lines.append(f"    {n}: {getattr(self, n):.4f}")
        if self.index_names:
            lines.append("  Indices:")
            for n in self.index_names:
                v = getattr(self, n)
                lines.append(f"    {n}: {v:.4f}" if np.isfinite(v) else f"    {n}: {v}")
        return "\n".join(lines)


# Registry: platform name → callable(albedo, flx_slr) → BandResult
_PLATFORMS = {}


def _register(name, func):
    _PLATFORMS[name] = func


def to_platform(albedo, platform, flx_slr=None):
    """Convolve 480-band spectral albedo onto a platform's band space.

    Parameters
    ----------
    albedo : np.ndarray (480,)
        Spectral albedo from ``run_model()``.
    platform : str
        One of: ``"sentinel2"``, ``"sentinel3"``, ``"landsat8"``, ``"modis"``,
        ``"cesm2band"``, ``"cesmrrtmg"``, ``"mar"``, ``"hadcm3"``.
    flx_slr : np.ndarray (480,), optional
        Spectral solar flux.  Required for flux-weighted convolution.
        Available as ``outputs.flx_slr`` after running the model.

    Returns
    -------
    BandResult
    """
    albedo = np.asarray(albedo, dtype=float)
    if flx_slr is not None:
        flx_slr = np.asarray(flx_slr, dtype=float)

    if platform not in _PLATFORMS:
        # Lazy-load platform modules to populate the registry
        _lazy_load()
        if platform not in _PLATFORMS:
            raise ValueError(
                f"Unknown platform {platform!r}. "
                f"Available: {sorted(_PLATFORMS.keys())}"
            )
    return _PLATFORMS[platform](albedo, flx_slr)


def _lazy_load():
    """Import all platform modules so they register themselves."""
    from biosnicar.bands.gcm import cesm, mar, hadcm3  # noqa: F401
    from biosnicar.bands.platforms import (  # noqa: F401
        sentinel2,
        sentinel3,
        landsat8,
        modis,
        planetscope,
    )
