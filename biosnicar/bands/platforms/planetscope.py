"""PlanetScope PSB.SD 8-band SuperDove: flux-weighted interval averaging.

Band specifications from Planet STAC metadata (FWHM-defined tophat intervals):

  Band  Name           Centre (nm)  FWHM (nm)   Interval (µm)
  B1    Coastal Blue      442         21          0.431 – 0.452
  B2    Blue              490         50          0.465 – 0.515
  B3    Green I           531         36          0.513 – 0.549
  B4    Green             565         36          0.547 – 0.583
  B5    Yellow            610         20          0.600 – 0.620
  B6    Red               665         30          0.650 – 0.680
  B7    Red Edge          705         16          0.697 – 0.713
  B8    NIR               865         40          0.845 – 0.885

All bands fall within BioSNICAR's 0.205–4.995 µm solar window, so interval
averaging (``interval_average``) is used rather than SRF CSV convolution.
"""

from biosnicar.bands import BandResult, _register
from biosnicar.bands._core import interval_average

# Tophat intervals derived from centre ± FWHM/2 (µm)
PLANETSCOPE_BANDS = {
    "B1": (0.431, 0.452),  # Coastal Blue
    "B2": (0.465, 0.515),  # Blue
    "B3": (0.513, 0.549),  # Green I
    "B4": (0.547, 0.583),  # Green
    "B5": (0.600, 0.620),  # Yellow
    "B6": (0.650, 0.680),  # Red
    "B7": (0.697, 0.713),  # Red Edge
    "B8": (0.845, 0.885),  # NIR
}


def _planetscope(albedo, flx_slr):
    r = BandResult("planetscope")
    for name, (lo, hi) in PLANETSCOPE_BANDS.items():
        r._set_band(name, interval_average(albedo, flx_slr, lo, hi))

    # NDSI  = (Green − NIR) / (Green + NIR)  — ice/snow vs water/vegetation
    denom = r.B4 + r.B8
    r._set_index("NDSI", (r.B4 - r.B8) / denom if denom != 0 else float("nan"))

    # NDVI  = (NIR − Red) / (NIR + Red)
    denom = r.B8 + r.B6
    r._set_index("NDVI", (r.B8 - r.B6) / denom if denom != 0 else float("nan"))

    # NDRE  = (NIR − Red Edge) / (NIR + Red Edge)  — algal/biological signal
    denom = r.B8 + r.B7
    r._set_index("NDRE", (r.B8 - r.B7) / denom if denom != 0 else float("nan"))

    return r


_register("planetscope", _planetscope)
