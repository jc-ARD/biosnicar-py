"""Sea ice radiative transfer extension for BioSNICAR.

Adds support for layer_type=2 (sea ice) using:
- Cox & Weeks (1983) brine volume
- Maxwell-Garnett effective medium for ice+brine
- Pre-computed LUT for fast runtime interpolation

Quick start::

    from biosnicar.sea_ice.api import SeaIceColumn, SeaIceLayer
    from biosnicar.sea_ice.presets import FYI_WINTER_BARE

    col = SeaIceColumn.from_preset(FYI_WINTER_BARE)
    result = col.compute_albedo(sza_deg=60)
    print(result.broadband)
"""
