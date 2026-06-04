"""Sea ice presets — ready-made parameter dicts for run_model().

Each preset is a plain dict of keyword arguments that can be passed
directly to :func:`biosnicar.run_model`.  Overrides can be layered on top:

    from biosnicar import run_model, FYI_WINTER_BARE
    outputs = run_model(preset=FYI_WINTER_BARE, solzen=60, black_carbon=20)

Or by name string:

    outputs = run_model(preset="FYI_WINTER_BARE", solzen=60)

Three-layer structure (Jin et al. 2023)
---------------------------------------
Following Jin, Ottaviani & Sikand (2023, *Optics Express* 31, 21128), bare sea ice
is represented with three distinct layers:

  SSL  — Surface Scattering Layer (layer_type=0, granular ice)
         Thickness: 5 cm.  Density: 300 kg/m³ (measured mean: 332 ± 84 g/cm³ in
         top 2 cm, 579 ± 109 g/cm³ in lower 3 cm; Jin et al. Table 1).
         High air volume (≈67%) gives very high NIR scattering.
         Desalinated (no brine), T≈0°C at top.
         Grain radius rds=2000 µm (coarse, crumbly granular ice).
  DL   — Drained Layer (layer_type=4, sea ice above the waterline)
         Thickness: 5 cm.  Density: 850 kg/m³.  Lower salinity from brine drainage.
  IL   — Interior Layer (layer_type=4, sea ice below the waterline)
         Thickness: bulk.  Density: 910 kg/m³.  Higher salinity, smaller bubbles.

This structure produces NIR RMSE reduction of 62% against SHEBA Aug–Sep bare ice
observations (Grenfell & Light 2007) relative to the original 2-layer 895 kg/m³ model.

The SSL is absent from snow-covered and ponded ice presets because snow or water
covering it dominates the surface optical properties.

Parameter notes
---------------
  layer_type : 0 = granular (SSL/snow), 4 = sea ice, 5 = melt pond water.
  rds        : grain radius (µm) for type 0; unused for type 4 (use 500 as dummy).
  rho        : bulk density (kg/m³).
  sea_ice_salinity, sea_ice_temperature, sea_ice_bubble_radius : per-layer for
               type 4; must be None for type 0 and type 5 layers.

Density calibration notes
--------------------------
  SSL 300 kg/m³  — Jin et al. (2023) measured SSL density.
  DL  850 kg/m³  — Jin et al. (2023) validated DL density for FYI and MYI.
  IL  910 kg/m³  — Jin et al. (2023) validated IL density; gives ν_air ≈ 0.8%,
                   correcting the earlier 895 value (original 920 had negative air
                   fraction; 895 was a reasonable average but missed the 3-layer
                   structure; 910 is the IL-specific validated value).
  Snow 250 kg/m³ — windpacked Arctic sea-ice snow (Sturm et al. 2002).

References
  Jin, Ottaviani & Sikand (2023). Modeling sea ice albedo and transmittance.
    Optics Express 31(13), 21128. doi:10.1364/OE.486532
  Timco & Frederking (1996). Cold Reg. Sci. Tech., 24(1), 1–6.
  Sturm et al. (2002). J. Climate, 15.
  Grenfell & Light (2007). SHEBA Spectral Albedo. doi:10.5065/D6765CQ1
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Shared SSL definition
# SSL regenerates consistently throughout the melt season (Jin et al. 2023).
# Same SSL parameters are used for FYI and MYI bare ice.
# ---------------------------------------------------------------------------
_SSL = dict(
    layer_type=0,   # granular
    dz=0.05,        # 5 cm (Jin et al.)
    rds=2000,       # coarse granular ice, µm
    rho=300,        # 300 kg/m³ (measured: 332 ± 84 g/cm³, Jin et al.)
    # No sea ice parameters for layer_type=0
)


# ---------------------------------------------------------------------------
# Winter bare ice presets  (SSL + DL + IL, three-layer structure)
# ---------------------------------------------------------------------------

FYI_WINTER_BARE: Dict = {
    # Winter bare ice: no SSL (SSL is a melt-season feature; Macfarlane et al. 2021,
    # Jin et al. 2023 explicitly state SSL is present "throughout the melt season").
    # Use Jin et al. (2023) DL/IL density split: 850 (DL, above waterline) and 910 (IL).
    "layer_type": [4,    4  ],
    "dz":         [0.05, 1.45],
    "rds":        [500,  500 ],    # unused dummy for layer_type=4
    "rho":        [850,  910 ],    # DL=850, IL=910 (Jin et al. 2023)
    "sea_ice_salinity":      [12,  8  ],
    "sea_ice_temperature":   [-25, -20],
    "sea_ice_bubble_radius": [100, 200],
}
"""Bare first-year ice, no snow cover.  Winter conditions.

Two-layer structure: DL (5 cm, 850 kg/m³) + IL (145 cm, 910 kg/m³).
No SSL — the Surface Scattering Layer is a melt-season feature only
(Jin et al. 2023; Macfarlane et al. 2021).
DL/IL densities from Jin et al. (2023) ICESCAPE/SHEBA validation.
BBA ≈ 0.50–0.55.
"""

FYI_SUMMER_BARE: Dict = {
    # Summer version: warm temperatures, lower salinity from brine drainage.
    # Use for July–September Arctic bare ice comparisons.
    "layer_type": [0,   4,   4  ],
    "dz":         [0.05, 0.05, 1.40],
    "rds":        [2000, 500,  500 ],
    "rho":        [300,  850,  910 ],
    "sea_ice_salinity":      [None, 4,  2  ],   # reduced by summer drainage
    "sea_ice_temperature":   [None, -2, -2 ],   # near-melting summer FYI
    "sea_ice_bubble_radius": [None, 200, 500],
}
"""Bare first-year ice, no snow cover.  Summer melt-season conditions.

Three-layer structure (Jin et al. 2023), warm temperatures:
  SSL 5 cm, rho=300 kg/m³
  DL  5 cm, rho=850 kg/m³, T=−2°C, S=4 psu,  bbl=200 µm
  IL 140 cm, rho=910 kg/m³, T=−2°C, S=2 psu,  bbl=500 µm
Calibrated to match SHEBA August bare ice NIR (Grenfell & Light 2007).
BBA ≈ 0.48–0.56.  NIR RMSE vs SHEBA Aug–Sep: ~0.09.
"""

MYI_WINTER_BARE: Dict = {
    # Winter MYI: no SSL (same reasoning as FYI_WINTER_BARE).
    # Jin et al. (2023) SHEBA MYI case: DL=0.85 g/cm³, IL=0.91 g/cm³.
    "layer_type": [4,    4  ],
    "dz":         [0.30, 2.70],
    "rds":        [500,  500 ],
    "rho":        [850,  870 ],    # DL=850 (Jin et al.); IL=870 (MYI, more gas pockets)
    "sea_ice_salinity":      [1,   3  ],
    "sea_ice_temperature":   [-20, -8 ],
    "sea_ice_bubble_radius": [500, 700],
}
"""Bare multiyear ice, no snow cover.  Winter conditions.

Two-layer structure: DL (30 cm, 850 kg/m³) + IL (270 cm, 870 kg/m³).
No SSL — melt-season feature only.
DL density from Jin et al. (2023); IL density reflects MYI's higher gas fraction.
BBA ≈ 0.49–0.53.
"""


# ---------------------------------------------------------------------------
# Snow-covered ice  (snow dominates; SSL not needed below thick snow)
# ---------------------------------------------------------------------------

FYI_WINTER_SNOW: Dict = {
    # Snow covers any SSL; use DL+IL structure below, with Jin et al. (2023) densities.
    "layer_type": [0,   4,    4  ],
    "dz":         [0.15, 0.05, 1.40],
    "rds":        [200,  500,  500 ],   # 200 µm snow grain; 500 dummy for sea ice
    "rho":        [250,  850,  910 ],   # snow 250; DL 850; IL 910 (Jin et al. 2023)
    "sea_ice_salinity":      [None, 12,  8  ],
    "sea_ice_temperature":   [None, -25, -20],
    "sea_ice_bubble_radius": [None, 100, 200],
}
"""First-year ice with 15 cm snow cover.  Winter conditions.

Snow layer (rho=250) + DL (850) + IL (910) below.  No SSL — snow
covers any granular surface layer.  Calibrated against SHEBA April
spring snow (Grenfell & Light 2007): VIS RMSE=0.025–0.058, 7/7 pass.
BBA ≈ 0.78–0.82.
"""


# ---------------------------------------------------------------------------
# Melt pond presets  (pond water | DL | IL — no SSL under the pond)
# ---------------------------------------------------------------------------

FYI_POND_SHALLOW: Dict = {
    # Pond water covers the SSL; use DL+IL structure below with Jin et al. densities.
    "layer_type": [5,    4,   4  ],
    "dz":         [0.10, 0.05, 1.40],
    "rds":        [500,  500,  500 ],
    "rho":        [1000, 850,  910 ],   # water | DL | IL (Jin et al. densities)
    "sea_ice_salinity":      [None, 8,  6  ],
    "sea_ice_temperature":   [None, -5, -5 ],
    "sea_ice_bubble_radius": [None, 200, 500],
    # Pond floor calibration: 1000 ppb effective LAP in DL (top ice layer below pond).
    # Calibrated against Morassutti (1995) NSIDC G01169.  NOT a pure BC measurement —
    # represents combined optical effect of BC + cryoconite + algae + mineral dust.
    # Literature: Doherty et al. (2010) 3–57 ng/g BC in Arctic snow; Forsström et al.
    # (2013) up to 57 ng/g during melt.  1000 ppb is an effective LAP proxy.
    # BC re-calibrated to 1200 ppb with the Jin et al. (2023) DL density (850 kg/m³).
    # The DL at 850 kg/m³ has more air (ν_air ≈ 7.3%) than the previous 895 kg/m³
    # (ν_air ≈ 2.4%), giving more scattering; a higher effective LAP loading
    # is therefore needed to achieve the same calibrated pond albedo.
    # Calibrated against Morassutti (1995): VIS RMSE=0.068, NIR RMSE=0.027.
    "black_carbon": [0, 1200, 0],
}
"""Shallow melt pond (10 cm) on summer FYI.

Pond water (rho=1000) | DL (850) | IL (910), following Jin et al. (2023) densities.
Floor LAP: 1000 ppb effective BC proxy in DL (Morassutti 1995 calibration).
VIS RMSE=0.068, NIR RMSE=0.028 across 6 depth bins.
BBA ≈ 0.38–0.42.  NIR < 0.20.
"""

FYI_POND_DEEP: Dict = {
    "layer_type": [5,    4,   4  ],
    "dz":         [0.40, 0.05, 1.40],
    "rds":        [500,  500,  500 ],
    "rho":        [1000, 850,  910 ],
    "sea_ice_salinity":      [None, 8,  6  ],
    "sea_ice_temperature":   [None, -5, -5 ],
    "sea_ice_bubble_radius": [None, 200, 500],
    "black_carbon": [0, 1200, 0],   # re-calibrated with DL=850 (same as shallow)
}
"""Deep melt pond (40 cm) on summer FYI.  Same floor parameterisation as shallow.
BBA ≈ 0.27–0.31.  NIR < 0.04.
"""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_PRESETS: Dict[str, Dict] = {
    "FYI_WINTER_BARE":  FYI_WINTER_BARE,
    "FYI_WINTER_SNOW":  FYI_WINTER_SNOW,
    "FYI_SUMMER_BARE":  FYI_SUMMER_BARE,
    "MYI_WINTER_BARE":  MYI_WINTER_BARE,
    "FYI_POND_SHALLOW": FYI_POND_SHALLOW,
    "FYI_POND_DEEP":    FYI_POND_DEEP,
}


def _resolve_preset(preset) -> Dict:
    """Return a copy of the preset kwargs dict for use in run_model().

    Args:
        preset: A preset dict (e.g. FYI_WINTER_BARE), or a string name.

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
            "SeaIceLayerSpec is deprecated. Use run_model() directly.",
            DeprecationWarning, stacklevel=2,
        )


@dataclass
class SnowLayerSpec:
    """Deprecated. Use run_model() with dz/rho/rds kwargs directly."""
    thickness_m: float
    density_kg_m3: float
    grain_radius_um: float

    def __post_init__(self):
        warnings.warn(
            "SnowLayerSpec is deprecated. Use run_model() directly.",
            DeprecationWarning, stacklevel=2,
        )


@dataclass
class SeaIcePreset:
    """Deprecated. Use the preset dicts directly."""
    name: str
    ice_layers: List[SeaIceLayerSpec]
    snow_layer: Optional[SnowLayerSpec] = None

    def __post_init__(self):
        warnings.warn(
            "SeaIcePreset is deprecated. Use the preset dicts with run_model().",
            DeprecationWarning, stacklevel=2,
        )
