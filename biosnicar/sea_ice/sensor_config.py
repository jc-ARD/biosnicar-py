"""Per-platform observation uncertainty defaults for satellite retrievals.

Approximate 1-sigma surface-albedo uncertainties for L2A-grade products,
used to weight the retrieval chi-squared when the caller supplies a
``platform`` but no explicit ``obs_uncertainty``.  SWIR bands carry larger
uncertainty (lower SNR over dark/bright extremes, atmospheric residuals).
"""

import numpy as np

PLATFORM_SNR = {
    # {platform: {band_name: typical_1sigma_albedo_uncertainty}}
    "sentinel2": {
        "B2": 0.02, "B3": 0.02, "B4": 0.02,
        "B8": 0.025, "B8A": 0.025, "B11": 0.03, "B12": 0.04,
    },
    "landsat8": {
        "B2": 0.02, "B3": 0.02, "B4": 0.02, "B5": 0.025,
        "B6": 0.03, "B7": 0.04,
    },
    "planetscope": {
        "B1": 0.025, "B2": 0.025, "B3": 0.025, "B4": 0.03,
    },
    "modis": {
        "B1": 0.02, "B2": 0.025, "B3": 0.02, "B4": 0.02,
        "B5": 0.03, "B6": 0.03, "B7": 0.04,
    },
    # Add Maxar WorldView-3 when band uncertainty estimates are available.
}

# Fallback for bands missing from the table above.
_DEFAULT_UNCERTAINTY = 0.03


def default_obs_uncertainty(platform, band_names):
    """1-sigma uncertainty array for *band_names* of *platform*.

    Returns None when the platform has no entry in :data:`PLATFORM_SNR`
    (the retrieval then falls back to unweighted chi-squared).
    """
    table = PLATFORM_SNR.get(platform)
    if table is None:
        return None
    return np.array(
        [table.get(b, _DEFAULT_UNCERTAINTY) for b in band_names], dtype=float,
    )
