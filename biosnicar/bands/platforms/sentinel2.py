"""Sentinel-2 MSI: 13 bands, NDSI / NDVI / Impurity Index.

Initial SRFs use tophat approximation from published band centres/widths
(ESA S2 MSI User Guide).  CSV at ``data/band_srfs/sentinel2_msi.csv`` can
be replaced with manufacturer SRFs later.

Band   Centre (nm)  Width (nm)   Centre (µm)  Half-width (µm)
 B1       443          20         0.443         0.010
 B2       490          65         0.490         0.0325
 B3       560          35         0.560         0.0175
 B4       665          30         0.665         0.015
 B5       705          15         0.705         0.0075
 B6       740          15         0.740         0.0075
 B7       783          20         0.783         0.010
 B8       842          115        0.842         0.0575
 B8A      865          20         0.865         0.010
 B9       945          20         0.945         0.010
 B10     1375          30         1.375         0.015
 B11     1610          90         1.610         0.045
 B12     2190         180         2.190         0.090
"""

from biosnicar.bands import BandResult, _register
from biosnicar.bands._core import load_srf_stack, srf_convolve_stack

SRF_NAME = "sentinel2_msi"

# Band names in CSV column order
BAND_NAMES = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7",
    "B8", "B8A", "B9", "B10", "B11", "B12",
]


def _sentinel2(albedo, flx_slr):
    r = BandResult("sentinel2")

    bands = srf_convolve_stack(albedo, flx_slr, load_srf_stack(SRF_NAME, BAND_NAMES))
    for name, value in zip(BAND_NAMES, bands):
        r._set_band(name, float(value))

    # NDSI = (B3 − B11) / (B3 + B11)
    denom = r.B3 + r.B11
    r._set_index("NDSI", (r.B3 - r.B11) / denom if denom != 0 else float("nan"))

    # NDVI = (B8 − B4) / (B8 + B4)
    denom = r.B8 + r.B4
    r._set_index("NDVI", (r.B8 - r.B4) / denom if denom != 0 else float("nan"))

    # Impurity Index II = B3 / B8A
    r._set_index("II", r.B3 / r.B8A if r.B8A != 0 else float("nan"))

    return r


_register("sentinel2", _sentinel2)
