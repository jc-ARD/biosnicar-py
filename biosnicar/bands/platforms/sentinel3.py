"""Sentinel-3 OLCI: 21 bands, I2DBA / I3DBA / NDCI / MCI / NDVI indices.

Initial SRFs use tophat approximation from published band centres/widths
(ESA S3 OLCI User Guide).  CSV at ``data/band_srfs/sentinel3_olci.csv``.

Band  Centre (nm)  Width (nm)
Oa01    400          15
Oa02    412.5        10
Oa03    442.5        10
Oa04    490          10
Oa05    510          10
Oa06    560          10
Oa07    620          10
Oa08    665          10
Oa09    673.75        7.5
Oa10    681.25        7.5
Oa11    708.75       10
Oa12    753.75       10
Oa13    761.25        2.5
Oa14    764.375       3.75
Oa15    767.5         2.5
Oa16    778.75       15
Oa17    865          20
Oa18    885          10
Oa19    900          10
Oa20    940          20
Oa21   1020          40
"""

from biosnicar.bands import BandResult, _register
from biosnicar.bands._core import load_srf_stack, srf_convolve_stack

SRF_NAME = "sentinel3_olci"

BAND_NAMES = [
    "Oa01", "Oa02", "Oa03", "Oa04", "Oa05", "Oa06", "Oa07",
    "Oa08", "Oa09", "Oa10", "Oa11", "Oa12", "Oa13", "Oa14",
    "Oa15", "Oa16", "Oa17", "Oa18", "Oa19", "Oa20", "Oa21",
]


def _sentinel3(albedo, flx_slr):
    r = BandResult("sentinel3")

    bands = srf_convolve_stack(albedo, flx_slr, load_srf_stack(SRF_NAME, BAND_NAMES))
    for name, value in zip(BAND_NAMES, bands):
        r._set_band(name, float(value))

    # I2DBA = Oa12 / Oa08  (2-band diagnostic absorption)
    r._set_index("I2DBA", r.Oa12 / r.Oa08 if r.Oa08 != 0 else float("nan"))

    # I3DBA = (Oa08 − Oa11) / Oa13  (3-band diagnostic absorption)
    r._set_index("I3DBA", (r.Oa08 - r.Oa11) / r.Oa13 if r.Oa13 != 0 else float("nan"))

    # NDCI = (Oa11 − Oa08) / (Oa11 + Oa08)
    denom = r.Oa11 + r.Oa08
    r._set_index("NDCI", (r.Oa11 - r.Oa08) / denom if denom != 0 else float("nan"))

    # MCI = Oa11 − Oa10 − (Oa12 − Oa10) * (708.75 − 681.25) / (753.75 − 681.25)
    r._set_index("MCI", r.Oa11 - r.Oa10 - (r.Oa12 - r.Oa10) * (708.75 - 681.25) / (753.75 - 681.25))

    # NDVI = (Oa17 − Oa08) / (Oa17 + Oa08)
    denom = r.Oa17 + r.Oa08
    r._set_index("NDVI", (r.Oa17 - r.Oa08) / denom if denom != 0 else float("nan"))

    return r


_register("sentinel3", _sentinel3)
