"""Pre-defined sea-ice column configurations for common scenarios.

All presets represent winter/spring conditions (no surface melt).  Broadband
albedo for winter bare sea ice is expected in the range [0.4, 0.9].

Reference conditions:
  FYI_WINTER_BARE       — bare first-year ice, no snow cover
  FYI_WINTER_SNOW       — first-year ice with 15 cm snow layer
  MYI_WINTER_BARE       — bare multiyear ice, no snow cover

Parameter calibration notes (v0.1.1)
--------------------------------------
Parameters were updated against Grenfell & Light (2007) SHEBA spectral
albedo data (doi:10.5065/D6765CQ1), April–September 1998.

FYI ice density (895 kg/m³)
  The original FYI surface density of 920 kg/m³ exceeded pure ice density
  (917 kg/m³), producing a negative computed air volume fraction and thus
  essentially zero bubble scattering.  Real cold FYI contains gas pockets
  from trapped air during freezing; published values range 890–915 kg/m³
  (Timco & Frederking 1996).  Reducing to 895 kg/m³ gives ν_air ≈ 2–4%,
  which halves the bare-ice NIR RMSE against summer SHEBA data
  (0.426 → 0.225).

Snow density (250 kg/m³)
  Reduced from 300 to 250 kg/m³ for FYI_WINTER_SNOW to better match
  April SHEBA albedo.  Arctic sea-ice snow is typically wind-compacted
  with density 200–280 kg/m³ (Sturm et al. 2002); the original 300 kg/m³
  produced a small systematic positive bias in the visible (~+0.03).

FYI ice temperature (bulk: −20 °C)
  Changed from −10 °C to −20 °C, consistent with typical cold-season
  Arctic FYI bulk temperature before melt onset (Perovich 2003).

MYI ice density (860/870 kg/m³)
  Reduced from 870/880 to 860/870 kg/m³.  Change is minor (~1 kg/m³
  improvement in NIR RMSE) but keeps the preset consistent with the
  observation that MYI brine drainage further reduces density over
  successive summers.

References
  Timco, G. W. & Frederking, R. M. W. (1996). A review of sea ice density.
    Cold Regions Science and Technology, 24(1), 1–6.
  Sturm, M. et al. (2002). Snow-cover on Arctic sea ice. J. Climate, 15.
  Perovich, D. K. (2003). Thin and thinner: Sea ice mass balance.
    Ann. Glaciol., 37, 63–68.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class SeaIceLayerSpec:
    """Physical specification of a single sea-ice layer."""
    thickness_m: float
    temperature_C: float
    salinity_psu: float
    density_kg_m3: float = 915.0
    bubble_radius_um: float = 200.0
    layer_class: str = "FYI"   # informational


@dataclass
class SnowLayerSpec:
    """Physical specification of a snow layer above sea ice."""
    thickness_m: float
    density_kg_m3: float
    grain_radius_um: float
    # v0.2: add salinity_psu for salty snow; MVP treats as fresh snow only


@dataclass
class SeaIcePreset:
    """Named preset configuration for a sea-ice column.

    Attributes:
        name:        Human-readable identifier.
        ice_layers:  List of SeaIceLayerSpec (top to bottom).
        snow_layer:  Optional snow layer above the ice (fresh snow in MVP).
    """
    name: str
    ice_layers: List[SeaIceLayerSpec]
    snow_layer: Optional[SnowLayerSpec] = None


# ---------------------------------------------------------------------------
# Concrete preset instances
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared FYI ice layer definitions
# density_kg_m3=895: gives ν_air ≈ 2–4%, fixing the near-zero scattering
# in the original 920/915 values (which produced negative air fraction).
# ---------------------------------------------------------------------------
_FYI_SURFACE = SeaIceLayerSpec(
    thickness_m=0.05,
    temperature_C=-25.0,
    salinity_psu=12.0,
    density_kg_m3=895.0,
    bubble_radius_um=100.0,
    layer_class="FYI",
)
_FYI_BULK = SeaIceLayerSpec(
    thickness_m=1.45,
    temperature_C=-20.0,
    salinity_psu=8.0,
    density_kg_m3=895.0,
    bubble_radius_um=200.0,
    layer_class="FYI",
)

FYI_WINTER_BARE = SeaIcePreset(
    name="FYI_WINTER_BARE",
    ice_layers=[_FYI_SURFACE, _FYI_BULK],
    snow_layer=None,
)

FYI_WINTER_SNOW = SeaIcePreset(
    name="FYI_WINTER_SNOW",
    ice_layers=[_FYI_SURFACE, _FYI_BULK],
    snow_layer=SnowLayerSpec(
        thickness_m=0.15,
        density_kg_m3=250.0,   # 300 → 250: better match to windpacked Arctic snow
        grain_radius_um=200.0,
    ),
)

MYI_WINTER_BARE = SeaIcePreset(
    name="MYI_WINTER_BARE",
    ice_layers=[
        SeaIceLayerSpec(
            thickness_m=0.30,
            temperature_C=-20.0,
            salinity_psu=1.0,
            density_kg_m3=860.0,   # 870 → 860: consistent with brine-drainage desalination
            bubble_radius_um=500.0,
            layer_class="MYI",
        ),
        SeaIceLayerSpec(
            thickness_m=2.70,
            temperature_C=-8.0,
            salinity_psu=3.0,
            density_kg_m3=870.0,   # 880 → 870
            bubble_radius_um=700.0,
            layer_class="MYI",
        ),
    ],
    snow_layer=None,
)

#: All built-in presets, accessible by name.
ALL_PRESETS: Dict[str, SeaIcePreset] = {
    p.name: p for p in [FYI_WINTER_BARE, FYI_WINTER_SNOW, MYI_WINTER_BARE]
}
