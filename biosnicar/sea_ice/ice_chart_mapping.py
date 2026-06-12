"""Map retrieve_sea_ice() results to standard sea-ice classification schemes.

Provides two mappings:

  map_to_wmo()      WMO Sea Ice Nomenclature (WMO No. 259)
  map_to_sigrid3()  SIGRID-3 — the WMO standard format used by operational
                    ice centres (NIC, AARI, DMI, MET Norway, etc.)

The emulator classifies surfaces by their spectral-optical state (albedo
signature), while ice charts classify by ice structure (age, thickness, stage
of melt).  The two systems are correlated but not equivalent.  The mapping
below is approximate and should be treated as a best-effort translation:

  ┌────────────────┬──────────────────────────────┬────────────────────────┐
  │ Our type       │ WMO nomenclature              │ SIGRID-3 partial code  │
  ├────────────────┼──────────────────────────────┼────────────────────────┤
  │ FYI_snow       │ First-year ice, snow cover    │ SM/SN, SG 1–2          │
  │ FYI_bare       │ First-year ice, bare          │ SM/SN, SG 2–3          │
  │ FYI_summer     │ First-year ice, ablating, SSL │ SM/SN, SG 3            │
  │ MYI_bare       │ Multiyear ice                 │ SQ,    SG 1–3          │
  │ FYI_pond       │ Any ice, melt-pond stage      │ SM/SN/SQ, SG 4         │
  └────────────────┴──────────────────────────────┴────────────────────────┘

What we cannot provide (requires independent measurement):
  - Ice concentration (CT) — we classify a surface, not a scene
  - Ice thickness / stage of development (SM vs SN vs SL)
  - Floe size
  - Topography (ridged, rafted, hummocked)

References
----------
  WMO (2014). WMO Sea-Ice Nomenclature, WMO No. 259.
  JCOMM (2014). SIGRID-3: A vector archive format for sea ice georeferenced
    information, JCOMM Technical Report No. 23.
  Worby, A. P. & Allison, I. (1999). A technique for making ship-based
    observations of Antarctic sea ice thickness and characteristics.
    Antarctic CRC Research Report 14.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class WMOIceClass:
    """Result of mapping a retrieved surface type to WMO nomenclature.

    Attributes
    ----------
    stage_of_development : str
        WMO stage-of-development term (e.g. "First-year ice").
    surface_description : str
        Human-readable description of the inferred surface state.
    melt_stage : str
        Qualitative melt stage (one of: "pre-melt", "melt-onset",
        "ablating", "peak-melt", "refreezing").
    notes : str
        Caveats about what cannot be determined from albedo alone.
    confidence : float
        Emulator classification confidence (0–1) from the retrieval.
    """

    stage_of_development: str
    surface_description: str
    melt_stage: str
    notes: str
    confidence: float


@dataclass
class SIGRID3IceClass:
    """Result of mapping a retrieved surface type to SIGRID-3 codes.

    SIGRID-3 uses a three-layer ice description (partial, total, thin).
    We populate only what can be inferred from albedo; fields that require
    independent structural measurements are marked ``None``.

    Attributes
    ----------
    ice_type_code : str
        Primary ice type code (e.g. "SM" = medium first-year ice, "SQ" = MYI).
        See WMO No. 259 / SIGRID-3 spec for full code table.
    stage_of_melt : int or None
        SG field: 1=dry snow, 2=wet snow/bare ice, 3=onset pooling/SSL,
        4=peak melt/deep ponds, 5=refreezing.  Estimated from surface type
        and retrieved parameters.
    melt_pond_present : bool
        Whether melt ponds are inferred on the surface.
    ice_age : str
        "first-year" or "multiyear".
    full_code : str
        Partial SIGRID-3 code string showing what can be populated.
        Format: ``<type>|SG<n>|<pond>`` where unknown fields are ``??``.
    notes : str
        Caveats about fields that cannot be determined from albedo.
    confidence : float
        Emulator classification confidence (0–1).
    """

    ice_type_code: str
    stage_of_melt: Optional[int]
    melt_pond_present: bool
    ice_age: str
    full_code: str
    notes: str
    confidence: float


# ── WMO mapping ───────────────────────────────────────────────────────────────

def map_to_wmo(result) -> WMOIceClass:
    """Map a SeaIceRetrievalResult to WMO Sea Ice Nomenclature (WMO No. 259).

    The mapping is approximate — albedo-based classification captures the
    optical surface state but not all structural properties required by the
    full WMO nomenclature.  See module docstring for what cannot be inferred.

    Parameters
    ----------
    result : SeaIceRetrievalResult
        Output of :func:`~biosnicar.sea_ice.retrieve.retrieve_sea_ice`.

    Returns
    -------
    WMOIceClass
    """
    stype  = result.surface_type
    params = result.parameters
    conf   = result.confidence

    _CANNOT_DETERMINE = (
        "Ice concentration, thickness, floe size, and topography cannot be "
        "determined from albedo alone and require independent measurement."
    )

    if stype == "FYI_snow":
        T_ice = params.get("sea_ice_temperature", -15.0)
        grain = params.get("snow_grain_radius", 300.0)
        if T_ice < -10 and grain < 200:
            melt = "pre-melt"
            desc = "First-year ice with dry snow cover (fine-grained, cold)"
        elif grain >= 400:
            melt = "melt-onset"
            desc = "First-year ice with metamorphosed/coarse snow cover"
        else:
            melt = "pre-melt"
            desc = "First-year ice with snow cover"
        return WMOIceClass(
            stage_of_development="First-year ice",
            surface_description=desc,
            melt_stage=melt,
            notes=_CANNOT_DETERMINE,
            confidence=conf,
        )

    if stype == "FYI_bare":
        Vb = params.get("brine_volume_fraction", 0.04)
        if Vb < 0.04:
            melt = "pre-melt"
            desc = "First-year ice, bare surface, low brine volume (cold)"
        else:
            melt = "melt-onset"
            desc = "First-year ice, bare surface, elevated brine volume"
        return WMOIceClass(
            stage_of_development="First-year ice",
            surface_description=desc,
            melt_stage=melt,
            notes=_CANNOT_DETERMINE,
            confidence=conf,
        )

    if stype == "FYI_summer":
        ssl_r = params.get("ssl_grain_radius", 2000.0)
        if ssl_r > 3000:
            melt = "ablating"
            desc = ("First-year ice, ablating, coarse-grained Surface Scattering "
                    "Layer (SSL) — advanced melt season")
        else:
            melt = "ablating"
            desc = "First-year ice, ablating, Surface Scattering Layer (SSL) present"
        return WMOIceClass(
            stage_of_development="First-year ice",
            surface_description=desc,
            melt_stage=melt,
            notes=_CANNOT_DETERMINE,
            confidence=conf,
        )

    if stype == "MYI_bare":
        Vb = params.get("brine_volume_fraction", 0.015)
        melt = "pre-melt" if Vb < 0.02 else "melt-onset"
        return WMOIceClass(
            stage_of_development="Old ice (multiyear ice)",
            surface_description=(
                "Multiyear ice, bare surface, desalinated — low brine volume "
                "and large air inclusions give characteristic high NIR scattering"
            ),
            melt_stage=melt,
            notes=_CANNOT_DETERMINE,
            confidence=conf,
        )

    if stype == "FYI_pond":
        depth = params.get("pond_depth", 0.15)
        if depth < 0.10:
            melt = "melt-onset"
            desc = "Ice surface, shallow melt ponds forming (depth < 10 cm)"
        elif depth < 0.30:
            melt = "peak-melt"
            desc = f"Ice surface, established melt ponds (depth ≈ {depth * 100:.0f} cm)"
        else:
            melt = "peak-melt"
            desc = "Ice surface, deep melt ponds (depth > 30 cm) — advanced melt"
        return WMOIceClass(
            stage_of_development="First-year or multiyear ice, melt-pond stage",
            surface_description=desc,
            melt_stage=melt,
            notes=(
                "Ice age (FY vs MY) under the pond cannot be determined from "
                "pond albedo alone.  " + _CANNOT_DETERMINE
            ),
            confidence=conf,
        )

    if stype == "young_ice":
        d = params.get("ice_thickness", 0.05)
        if d < 0.01:
            term, melt = "Grease ice / frazil ice", "pre-freeze"
        elif d < 0.05:
            term, melt = "Dark nilas", "pre-freeze"
        elif d < 0.10:
            term, melt = "Light nilas", "pre-freeze"
        elif d < 0.15:
            term, melt = "Grey ice", "pre-freeze"
        else:
            term, melt = "Grey-white ice (young ice)", "pre-freeze"
        return WMOIceClass(
            stage_of_development=term,
            surface_description=(
                f"{term} — semi-transparent ice ({d*100:.0f} cm thick), "
                "ocean below contributes to observed albedo"
            ),
            melt_stage=melt,
            notes=(
                "Ice thickness is the primary retrieved parameter.  "
                + _CANNOT_DETERMINE
            ),
            confidence=conf,
        )

    if stype == "open_water":
        wind = params.get("wind_speed_ms", None)
        desc = "Open water — no ice present at the observed location"
        if wind is not None:
            desc += f" (retrieved wind speed ≈ {wind:.1f} m/s)"
        return WMOIceClass(
            stage_of_development="Ice free / open water",
            surface_description=desc,
            melt_stage="n/a",
            notes=(
                "Classification applies to the observed pixel footprint only; "
                "nearby ice concentration cannot be inferred from albedo."
            ),
            confidence=conf,
        )

    # Unknown type
    return WMOIceClass(
        stage_of_development="Unknown",
        surface_description=f"Unrecognised surface type: {stype!r}",
        melt_stage="unknown",
        notes=_CANNOT_DETERMINE,
        confidence=conf,
    )


# ── SIGRID-3 mapping ──────────────────────────────────────────────────────────

# SIGRID-3 ice-type codes relevant to Arctic surface classification
# (from WMO No. 259 / JCOMM TR No. 23)
# young_ice codes (SA=new ice, SB=nilas, SI=grey, SJ=grey-white) are
# resolved by ice_thickness in map_to_sigrid3() below.
_SIGRID3_TYPE = {
    "FYI_snow":   "SM/SN",  # medium/thick FY (thickness unknown from albedo)
    "FYI_bare":   "SM/SN",
    "FYI_summer": "SM/SN",
    "MYI_bare":   "SQ",     # multiyear ice
    "FYI_pond":   "SM/SN/SQ",  # pond can be on any ice type
    "young_ice":  None,     # resolved from ice_thickness — see map_to_sigrid3()
    "open_water": "OW",     # ice free (SIGRID-3 CT=00)
}

# SIGRID-3 stage-of-melt (SG field)
# 1 = dry snow;  2 = wet snow / bare ice onset;  3 = onset of pooling / SSL
# 4 = peak melt / established ponds;  5 = refreezing
_SIGRID3_MELT = {
    "FYI_snow":   None,   # determined by parameters — see below
    "FYI_bare":   2,
    "FYI_summer": 3,
    "MYI_bare":   None,   # determined by Vb
    "FYI_pond":   None,   # determined by pond depth
}


def map_to_sigrid3(result) -> SIGRID3IceClass:
    """Map a SeaIceRetrievalResult to a partial SIGRID-3 ice-class description.

    Populates ice type code, stage of melt, and melt-pond flag.  Fields
    requiring structural measurement (concentration, thickness, floe size)
    are left as ``??`` in the code string.

    Parameters
    ----------
    result : SeaIceRetrievalResult
        Output of :func:`~biosnicar.sea_ice.retrieve.retrieve_sea_ice`.

    Returns
    -------
    SIGRID3IceClass
    """
    stype  = result.surface_type
    params = result.parameters
    conf   = result.confidence

    ice_type  = _SIGRID3_TYPE.get(stype, "??")
    ice_age   = "multiyear" if stype == "MYI_bare" else "first-year"
    pond      = stype == "FYI_pond"

    # Infer stage of melt from type and parameters
    if stype == "FYI_snow":
        grain = params.get("snow_grain_radius", 300.0)
        T_ice = params.get("sea_ice_temperature", -15.0)
        sg = 1 if (T_ice < -5 and grain < 300) else 2

    elif stype == "FYI_bare":
        Vb = params.get("brine_volume_fraction", 0.04)
        sg = 2 if Vb < 0.06 else 3

    elif stype == "FYI_summer":
        sg = 3

    elif stype == "MYI_bare":
        Vb = params.get("brine_volume_fraction", 0.015)
        sg = 1 if Vb < 0.015 else 2

    elif stype == "FYI_pond":
        depth = params.get("pond_depth", 0.15)
        sg = 4 if depth >= 0.10 else 3

    elif stype == "open_water":
        return SIGRID3IceClass(
            ice_type_code="OW",
            stage_of_melt=None,
            melt_pond_present=False,
            ice_age="n/a",
            full_code="CT=00|ice free",
            notes=(
                "Open water at the observed pixel.  Scene-level ice "
                "concentration requires aggregating many pixels."
            ),
            confidence=conf,
        )

    elif stype == "young_ice":
        # young_ice has its own code resolution — return early
        d = params.get("ice_thickness", 0.05)
        if d < 0.01:
            ycode = "SA"
        elif d < 0.10:
            ycode = "SB"
        elif d < 0.15:
            ycode = "SI"
        else:
            ycode = "SJ"
        return SIGRID3IceClass(
            ice_type_code=ycode,
            stage_of_melt=1,
            melt_pond_present=False,
            ice_age="new",
            full_code=(f"CT=??|CA={ycode}|SG1|EV=0"
                       f"|thick={d*100:.0f}cm|floe=??"),
            notes=(
                "Ice type code (SA/SB/SI/SJ) resolved from retrieved "
                "ice_thickness.  Concentration and floe size cannot be "
                "inferred from albedo alone."
            ),
            confidence=conf,
        )

    else:
        sg = None

    sg_str = str(sg) if sg is not None else "?"
    pond_str = "EV>0" if pond else "EV=0"
    full_code = f"CT=??|CA={ice_type}|SG{sg_str}|{pond_str}|thick=??|floe=??"

    _notes = (
        "Partial SIGRID-3 code — fields requiring independent structural "
        "measurement (CT: concentration, ice thickness, floe size) cannot "
        "be inferred from albedo and are marked ??."
    )

    return SIGRID3IceClass(
        ice_type_code=ice_type,
        stage_of_melt=sg,
        melt_pond_present=pond,
        ice_age=ice_age,
        full_code=full_code,
        notes=_notes,
        confidence=conf,
    )


# ── Convenience: print a formatted summary ───────────────────────────────────

def classification_summary(result) -> str:
    """Return a formatted string summarising the retrieval in WMO/SIGRID-3 terms.

    Parameters
    ----------
    result : SeaIceRetrievalResult
        Output of :func:`~biosnicar.sea_ice.retrieve.retrieve_sea_ice`.

    Returns
    -------
    str
    """
    wmo  = map_to_wmo(result)
    sig3 = map_to_sigrid3(result)

    lines = [
        f"Surface type       : {result.surface_type}  "
        f"(confidence={result.confidence:.3f})",
        f"WMO stage          : {wmo.stage_of_development}",
        f"WMO description    : {wmo.surface_description}",
        f"Melt stage         : {wmo.melt_stage}",
        f"SIGRID-3 (partial) : {sig3.full_code}",
        f"Ice age            : {sig3.ice_age}",
        f"Melt ponds         : {'present' if sig3.melt_pond_present else 'absent'}",
        f"Caveats            : {wmo.notes}",
    ]
    return "\n".join(lines)
