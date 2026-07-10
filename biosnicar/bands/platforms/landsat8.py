"""Landsat 8 OLI: 7 bands, NDSI / NDVI indices.

Initial SRFs use tophat approximation from published band centres/widths
(USGS Landsat 8 Data Users Handbook).  CSV at ``data/band_srfs/landsat8_oli.csv``.

Band   Centre (nm)  Range (nm)
 B1       443        435–451   (Coastal aerosol)
 B2       482        452–512   (Blue)
 B3       561        533–590   (Green)
 B4       655        636–673   (Red)
 B5       865        851–879   (NIR)
 B6      1609       1567–1651  (SWIR-1)
 B7      2201       2107–2294  (SWIR-2)
"""

from biosnicar.bands import BandResult, _register
from biosnicar.bands._core import load_srf_stack, srf_convolve_stack

SRF_NAME = "landsat8_oli"

BAND_NAMES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]


def _landsat8(albedo, flx_slr):
    r = BandResult("landsat8")

    bands = srf_convolve_stack(albedo, flx_slr, load_srf_stack(SRF_NAME, BAND_NAMES))
    for name, value in zip(BAND_NAMES, bands):
        r._set_band(name, float(value))

    # NDSI = (B3 − B6) / (B3 + B6)
    denom = r.B3 + r.B6
    r._set_index("NDSI", (r.B3 - r.B6) / denom if denom != 0 else float("nan"))

    # NDVI = (B5 − B4) / (B5 + B4)
    denom = r.B5 + r.B4
    r._set_index("NDVI", (r.B5 - r.B4) / denom if denom != 0 else float("nan"))

    return r


_register("landsat8", _landsat8)
