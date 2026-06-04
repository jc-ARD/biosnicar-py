"""Sea ice radiative transfer extension for BioSNICAR.

Adds support for layer_type=4 (sea ice) and layer_type=5 (melt pond) using:
- Cox & Weeks (1983) brine volume with liquidus salinity correction
- Maxwell-Garnett effective medium for ice+brine
- Pre-computed LUT for fast runtime interpolation
- Melt pond areal fraction blending

Quick start::

    from biosnicar import run_model
    from biosnicar.sea_ice.presets import FYI_SUMMER_BARE, FYI_POND_SHALLOW
    from biosnicar.sea_ice.pond_fraction import blend_pond_fraction

    # Pure white ice
    ice = run_model(preset=FYI_SUMMER_BARE, solzen=60)

    # Mixed surface: 30% melt pond cover
    mixed = run_model(preset="FYI_SUMMER_BARE", solzen=60, pond_fraction=0.30)

    # Or blend pre-computed results
    pond = run_model(preset=FYI_POND_SHALLOW, solzen=60)
    mixed = blend_pond_fraction(ice, pond, f=0.30)
    print(mixed.BBA)
"""

from biosnicar.sea_ice.pond_fraction import blend_pond_fraction
from biosnicar.sea_ice.retrieve import retrieve_sea_ice, SeaIceRetrievalResult
from biosnicar.sea_ice.emulator_configs import (
    SEA_ICE_EMULATOR_CONFIGS,
    load_sea_ice_emulators,
)

__all__ = [
    "blend_pond_fraction",
    "retrieve_sea_ice",
    "SeaIceRetrievalResult",
    "SEA_ICE_EMULATOR_CONFIGS",
    "load_sea_ice_emulators",
]
