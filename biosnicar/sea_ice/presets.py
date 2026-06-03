"""Sea ice presets — ready-made parameter dicts for run_model().

Each preset is a plain dict of keyword arguments that can be passed
directly to :func:`biosnicar.run_model`.  Overrides can be layered on top:

    from biosnicar import run_model, FYI_WINTER_BARE
    outputs = run_model(preset=FYI_WINTER_BARE, solzen=70, black_carbon=500)

Or by name string:

    outputs = run_model(preset="FYI_WINTER_BARE", solzen=70)

Parameter notes
---------------
  layer_type : 4 = sea ice (brine inclusions), 0 = granular snow.
  rds        : grain radius (µm) for type 0 (snow), unused for type 4.
               Sea ice layers carry 500 as a harmless dummy value.
  rho        : bulk density (kg/m³).  895 for FYI, 860–870 for MYI.
  sea_ice_salinity    : psu per layer (None for snow layers).
  sea_ice_temperature : °C per layer (None for snow layers).
  sea_ice_bubble_radius : µm per layer (None for snow layers).

FYI ice density (895 kg/m³)
  Gives ν_air ≈ 2–4%, fixing the near-zero scattering in the original
  920/915 values (which produced negative air fraction).

Snow density (250 kg/m³)
  Lower than fresh-snow default (300); better matches Arctic sea-ice snow
  (published values 200–280 kg/m³, Sturm et al. 2002).

References
  Timco & Frederking (1996). Cold Reg. Sci. Tech., 24(1), 1–6.
  Sturm et al. (2002). J. Climate, 15.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Preset dicts  (the primary interface)
# ---------------------------------------------------------------------------

FYI_WINTER_BARE: Dict = {
    "layer_type": [4, 4],
    "dz":         [0.05, 1.45],
    "rds":        [500, 500],       # unused for layer_type=4
    "rho":        [895, 895],
    "sea_ice_salinity":      [12, 8],
    "sea_ice_temperature":   [-25, -20],
    "sea_ice_bubble_radius": [100, 200],
}
"""Bare first-year ice, no snow cover.  Winter conditions.  BBA ≈ 0.55."""

FYI_WINTER_SNOW: Dict = {
    "layer_type": [0, 4, 4],
    "dz":         [0.15, 0.05, 1.45],
    "rds":        [200, 500, 500],  # 200 µm for snow; 500 dummy for sea ice
    "rho":        [250, 895, 895],
    "sea_ice_salinity":      [None, 12, 8],
    "sea_ice_temperature":   [None, -25, -20],
    "sea_ice_bubble_radius": [None, 100, 200],
}
"""First-year ice with 15 cm snow cover.  Winter conditions.  BBA ≈ 0.78."""

MYI_WINTER_BARE: Dict = {
    "layer_type": [4, 4],
    "dz":         [0.30, 2.70],
    "rds":        [500, 500],
    "rho":        [860, 870],
    "sea_ice_salinity":      [1, 3],
    "sea_ice_temperature":   [-20, -8],
    "sea_ice_bubble_radius": [500, 700],
}
"""Bare multiyear ice, no snow cover.  Winter conditions.  BBA ≈ 0.52."""

FYI_POND_SHALLOW: Dict = {
    "layer_type": [5, 4, 4],
    "dz":         [0.10, 0.05, 1.45],
    "rds":        [500, 500, 500],
    "rho":        [1000, 895, 895],            # 1000 kg/m³ for liquid water
    "sea_ice_salinity":      [None, 12, 8],
    "sea_ice_temperature":   [None, -2, -2],   # near-melting FYI below pond
    "sea_ice_bubble_radius": [None, 100, 200],
}
"""Shallow melt pond (10 cm) on first-year ice.  Summer conditions.  BBA ≈ 0.25."""

FYI_POND_DEEP: Dict = {
    "layer_type": [5, 4, 4],
    "dz":         [0.40, 0.05, 1.45],
    "rds":        [500, 500, 500],
    "rho":        [1000, 895, 895],
    "sea_ice_salinity":      [None, 12, 8],
    "sea_ice_temperature":   [None, -2, -2],
    "sea_ice_bubble_radius": [None, 100, 200],
}
"""Deep melt pond (40 cm) on first-year ice.  Summer conditions.  BBA ≈ 0.18."""

ALL_PRESETS: Dict[str, Dict] = {
    "FYI_WINTER_BARE":  FYI_WINTER_BARE,
    "FYI_WINTER_SNOW":  FYI_WINTER_SNOW,
    "MYI_WINTER_BARE":  MYI_WINTER_BARE,
    "FYI_POND_SHALLOW": FYI_POND_SHALLOW,
    "FYI_POND_DEEP":    FYI_POND_DEEP,
}


def _resolve_preset(preset) -> Dict:
    """Return a copy of the preset kwargs dict for use in run_model().

    Args:
        preset: A preset dict (e.g. FYI_WINTER_BARE), or a string name
            ("FYI_WINTER_BARE", "FYI_WINTER_SNOW", "MYI_WINTER_BARE").

    Raises:
        TypeError:  If preset is not a str or dict.
        ValueError: If the string name is not found in ALL_PRESETS.
    """
    if isinstance(preset, dict):
        return preset.copy()
    if isinstance(preset, str):
        if preset not in ALL_PRESETS:
            raise ValueError(
                f"Unknown preset {preset!r}. Available: {sorted(ALL_PRESETS)}"
            )
        return ALL_PRESETS[preset].copy()
    raise TypeError(
        f"preset must be a string name or a dict of run_model kwargs, "
        f"not {type(preset).__name__}"
    )


# ---------------------------------------------------------------------------
# Legacy dataclasses — kept for backwards compatibility only.
# New code should use the dict presets and run_model() directly.
# ---------------------------------------------------------------------------

@dataclass
class SeaIceLayerSpec:
    """Deprecated. Use run_model() with sea_ice_* kwargs directly."""
    thickness_m: float
    temperature_C: float
    salinity_psu: float
    density_kg_m3: float = 915.0
    bubble_radius_um: float = 200.0
    layer_class: str = "FYI"

    def __post_init__(self):
        warnings.warn(
            "SeaIceLayerSpec is deprecated and will be removed in v0.3. "
            "Use run_model() with sea_ice_temperature, sea_ice_salinity, "
            "sea_ice_bubble_radius kwargs directly.",
            DeprecationWarning,
            stacklevel=2,
        )


@dataclass
class SnowLayerSpec:
    """Deprecated. Use run_model() with rds/rho/dz kwargs directly."""
    thickness_m: float
    density_kg_m3: float
    grain_radius_um: float

    def __post_init__(self):
        warnings.warn(
            "SnowLayerSpec is deprecated and will be removed in v0.3. "
            "Use run_model() with dz, rho, rds kwargs directly.",
            DeprecationWarning,
            stacklevel=2,
        )


@dataclass
class SeaIcePreset:
    """Deprecated. Use the preset dicts (FYI_WINTER_BARE etc.) directly."""
    name: str
    ice_layers: List[SeaIceLayerSpec]
    snow_layer: Optional[SnowLayerSpec] = None

    def __post_init__(self):
        warnings.warn(
            "SeaIcePreset is deprecated and will be removed in v0.3. "
            "Use the preset dicts (FYI_WINTER_BARE, FYI_WINTER_SNOW, "
            "MYI_WINTER_BARE) with run_model(preset=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
