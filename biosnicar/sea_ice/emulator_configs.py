"""Sea ice emulator configurations — parameter spaces and transform functions.

Each entry in ``SEA_ICE_EMULATOR_CONFIGS`` defines one surface type:

  * ``params``        — free parameters and their (min, max) ranges
  * ``transform_fn``  — maps scalar sampled params → ``run_model()`` kwargs
  * ``description``   — human-readable label
  * ``n_samples``     — recommended training sample count
  * ``emulator_file`` — relative path under ``data/emulators/``

The transform functions handle broadcasting scalar parameters to per-layer
lists and mapping structural parameters (e.g. ``pond_depth``) to ``dz``.
They are only called during training; inference (``emulator.predict()``) runs
the neural network directly and does not need the transform.

Surface types
-------------
``FYI_bare``    Winter/spring bare first-year ice (DL + IL).
``FYI_snow``    Snow-covered first-year ice (snow + DL + IL).
``FYI_summer``  Melt-season bare ice with Surface Scattering Layer (SSL + DL + IL).
``MYI_bare``    Bare multiyear ice (DL + IL, lower salinity, larger bubbles).
``FYI_pond``    Melt pond on FYI (pond water + DL + IL).
``open_water``  Ice-free ocean (analytical Fresnel + subsurface model — no
                training; see :mod:`biosnicar.sea_ice.open_water`).
"""

from pathlib import Path

import numpy as np

# ── Absolute path to the emulator data directory ────────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "emulators"

# Classification band masks (C2).  Applied ONLY to the classification
# chi-squared in retrieve_sea_ice() — never to parameter fitting.
# "vis_only" (400–1000 nm): bare-ice types — white ice SWIR overlaps coarse
# snow and degrades summer bare-ice accuracy (100% → 42%, see
# docs/SEA_ICE_EMULATOR.md).  "vis_swir" (400–2500 nm): types whose SWIR
# signature is discriminative (snow grain size, pond/water absorption,
# thin-ice transmittance).
BAND_MASKS = {
    "vis_only": (0.400, 1.000),   # µm, inclusive
    "vis_swir": (0.400, 2.500),
}

# Reference bulk salinities used when inverting brine volume → temperature.
# These are representative of each ice type's drained-layer composition.
# Post-hoc temperature recovery: T = invert_brine_volume(Vb, S_ref).
FYI_BARE_S_REF = 6.0   # psu — typical FYI drained-layer salinity
MYI_BARE_S_REF = 2.0   # psu — typical desalinated MYI


def _snap(rds):
    """Snap grain radius to nearest granular-ice LUT grid entry.

    The granular-ice (layer_type=0) LUT grid uses:
      step 5  for [30, 1500)
      step 20 for [1500, 5000]
    """
    v = float(rds)
    if v < 1500:
        return int(round(v / 5) * 5)
    else:
        return int(round(v / 20) * 20)


# ============================================================================
# Transform functions  (sampled scalar dict → run_model kwargs)
# ============================================================================

def _transform_fyi_bare(p):
    """Winter/spring bare FYI: two sea-ice layers (DL + IL).

    Primary parameter is ``brine_volume_fraction`` (Vb) rather than the
    degenerate (T, S) pair.  Temperature is recovered by inverting the
    Cox & Weeks equation at the fixed reference salinity FYI_BARE_S_REF=6 psu.
    The IL salinity is set to half the reference (drainage gradient).
    """
    from biosnicar.sea_ice.brine_volume import invert_brine_volume
    T   = invert_brine_volume(p["brine_volume_fraction"], FYI_BARE_S_REF)
    bbl = p["sea_ice_bubble_radius"]
    return dict(
        layer_type=[4, 4],
        dz=[0.05, 1.45],
        rds=[500, 500],
        rho=[p["rho_DL"], 910],
        sea_ice_salinity=[FYI_BARE_S_REF, FYI_BARE_S_REF * 0.5],
        sea_ice_temperature=[T, T],
        sea_ice_bubble_radius=[bbl, min(1000.0, bbl * 2.0)],
        black_carbon=[p["black_carbon"], 0],
        solzen=p["solzen"],
        direct=p["direct"],
    )


def _transform_fyi_snow(p):
    """Snow-covered FYI: fresh snow layer on top of DL + IL.

    Parameterised by ``tau_snow`` (snow depth in units of grain radius —
    a dimensionless optical-depth proxy) and ``snow_grain_radius``, rather
    than the degenerate (snow_depth, snow_grain_radius) pair: the radiative
    response depends on depth/grain ratio, not on depth independently, so
    sampling depth directly piles retrievals onto the depth bounds.
    Physical depth is recovered as ``tau_snow * snow_grain_radius * 1e-6``.
    """
    T = p["sea_ice_temperature"]
    r = p["snow_grain_radius"]
    depth = float(np.clip(p["tau_snow"] * r * 1e-6, 0.003, 1.5))
    return dict(
        layer_type=[0, 4, 4],
        dz=[depth, 0.05, 1.45],
        rds=[_snap(r), 500, 500],
        rho=[300, 850, 910],
        sea_ice_salinity=[None, 8, 5],
        sea_ice_temperature=[None, T, T],
        sea_ice_bubble_radius=[None, 200, 400],
        black_carbon=[p["black_carbon"], 0, 0],
        solzen=p["solzen"],
        direct=p["direct"],
    )


def _transform_fyi_summer(p):
    """Summer bare FYI: SSL (granular) on top of DL + IL."""
    T   = p["sea_ice_temperature"]
    bbl = p["sea_ice_bubble_radius"]
    return dict(
        layer_type=[0, 4, 4],
        dz=[0.05, 0.05, 1.40],
        rds=[_snap(p["ssl_grain_radius"]), 500, 500],
        rho=[300, 850, 910],
        sea_ice_salinity=[None, 4, 2],
        sea_ice_temperature=[None, T, T],
        sea_ice_bubble_radius=[None, bbl, min(1000.0, bbl * 2.0)],
        black_carbon=[p["black_carbon"], 0, 0],
        solzen=p["solzen"],
        direct=p["direct"],
    )


def _transform_myi_bare(p):
    """Bare MYI: lower salinity, larger bubbles, slightly lower IL density.

    Primary parameter is ``brine_volume_fraction`` (Vb) rather than (T, S).
    Temperature is recovered via Cox & Weeks inversion at MYI_BARE_S_REF=2 psu.
    """
    from biosnicar.sea_ice.brine_volume import invert_brine_volume
    T   = invert_brine_volume(p["brine_volume_fraction"], MYI_BARE_S_REF)
    bbl = p["sea_ice_bubble_radius"]
    return dict(
        layer_type=[4, 4],
        dz=[0.30, 2.70],
        rds=[500, 500],
        rho=[850, 870],
        sea_ice_salinity=[MYI_BARE_S_REF, max(0.3, MYI_BARE_S_REF * 0.4)],
        sea_ice_temperature=[T, T],
        sea_ice_bubble_radius=[bbl, min(2000.0, bbl * 1.5)],
        black_carbon=[p["black_carbon"], 0],
        solzen=p["solzen"],
        direct=p["direct"],
    )


def _transform_young_ice(p):
    """Young ice: thin slab over dark ocean (layer_type=6).

    The sampled parameter is ``ice_thickness_cm`` (0.5-30 cm) so the
    log10(x+1) sampling/optimisation machinery conditions it properly
    (on the metre scale the +1 makes the transform a no-op).
    """
    return dict(
        layer_type=6,
        ice_thickness=p["ice_thickness_cm"] / 100.0,
        sea_ice_temperature=p["sea_ice_temperature"],
        sea_ice_salinity=p["sea_ice_salinity"],
        ocean_albedo=p["ocean_albedo"],
        solzen=p["solzen"],
        direct=p["direct"],
    )


def _transform_fyi_pond(p):
    """Melt pond on FYI: liquid water layer over DL + IL."""
    T = p["sea_ice_temperature"]
    return dict(
        layer_type=[5, 4, 4],
        dz=[p["pond_depth"], 0.05, 1.40],
        rds=[500, 500, 500],
        rho=[1000, 850, 910],
        sea_ice_salinity=[None, 8, 6],
        sea_ice_temperature=[None, T, T],
        sea_ice_bubble_radius=[None, 200, 500],
        black_carbon=[0, p["black_carbon"], 0],
        solzen=p["solzen"],
        direct=p["direct"],
    )


# ============================================================================
# Config registry
# ============================================================================

SEA_ICE_EMULATOR_CONFIGS = {
    "FYI_bare": {
        "description": "Winter/spring bare first-year ice (no snow, no SSL)",
        "band_mask": "vis_only",
        # (T, S) replaced by brine_volume_fraction to eliminate degeneracy.
        # Vb bounds correspond to T in [-22, -2.1]°C at S_ref=FYI_BARE_S_REF=6 psu.
        # Post-hoc T recovery: T = invert_brine_volume(Vb, FYI_BARE_S_REF)
        "params": {
            "brine_volume_fraction":  (0.019, 0.141),
            "sea_ice_bubble_radius":  (50.0,  1000.0),
            "black_carbon":           (0.0,   5000.0),
            "rho_DL":                 (820.0, 900.0),
            "solzen":                 (20,    80),
            "direct":                 (0,     1),
        },
        "transform_fn":       _transform_fyi_bare,
        "n_samples":          30000,
        "hidden_layer_sizes": (256, 256, 128, 64),
        "emulator_file": str(_DATA_DIR / "sea_ice_FYI_bare_6param.npz"),
    },
    "FYI_snow": {
        "description": "Snow-covered first-year ice",
        "band_mask": "vis_swir",
        # (snow_depth, grain_radius) replaced by (tau_snow, grain_radius):
        # tau_snow = depth / grain_radius (dimensionless; depth and radius
        # both in um).  tau 50-3000 with r 50-2000 um spans ~3 mm to 1.5 m
        # of physical snow depth.  Recover depth as tau * r * 1e-6 m.
        # Grain lower bound 50 um: SHEBA spring fits slammed the old 100 um
        # bound (fresh fine-grained snow); granular LUT supports >= 30 um.
        # T upper bound stays at -5: widening to -2 lets warm-snow fits
        # mimic summer bare ice (SHEBA summer accuracy 16/16 -> 9/16).
        # Late-spring retrievals saturate T at -5 because the melt-season
        # prior (mean -4) pushes against this bound — informative, not a
        # degeneracy.
        "params": {
            "tau_snow":               (50.0,  3000.0),
            "snow_grain_radius":      (50.0,  2000.0),
            "sea_ice_temperature":    (-30.0, -5.0),
            "black_carbon":           (0.0,   5000.0),
            "solzen":                 (20,    80),
            "direct":                 (0,     1),
        },
        "transform_fn":  _transform_fyi_snow,
        "n_samples":     12000,
        "emulator_file": str(_DATA_DIR / "sea_ice_FYI_snow_tau_6param.npz"),
    },
    "FYI_summer": {
        "description": "Melt-season bare FYI with Surface Scattering Layer",
        "band_mask": "vis_only",
        "params": {
            "ssl_grain_radius":       (500.0, 5000.0),
            "sea_ice_temperature":    (-10.0, -2.0),
            "sea_ice_bubble_radius":  (50.0,  500.0),
            "black_carbon":           (0.0,   5000.0),
            "solzen":                 (20,    80),
            "direct":                 (0,     1),
        },
        "transform_fn":  _transform_fyi_summer,
        "n_samples":     12000,
        "emulator_file": str(_DATA_DIR / "sea_ice_FYI_summer_6param.npz"),
    },
    "MYI_bare": {
        "description": "Bare multiyear ice (lower salinity, larger bubbles)",
        "band_mask": "vis_only",
        # (T, S) replaced by brine_volume_fraction.
        # Vb bounds correspond to T in [-22, -2.1]°C at S_ref=MYI_BARE_S_REF=2 psu.
        "params": {
            "brine_volume_fraction":  (0.0065, 0.045),  # T in [-21, -2.2]°C at S=2 psu
            "sea_ice_bubble_radius":  (200.0, 2000.0),
            "black_carbon":           (0.0,   5000.0),
            "solzen":                 (20,    80),
            "direct":                 (0,     1),
        },
        "transform_fn":       _transform_myi_bare,
        "n_samples":          30000,
        "hidden_layer_sizes": (256, 256, 128, 64),
        "emulator_file": str(_DATA_DIR / "sea_ice_MYI_bare_5param.npz"),
    },
    "FYI_pond": {
        "description": "Melt pond on first-year ice",
        "band_mask": "vis_swir",
        "params": {
            "pond_depth":             (0.02,  0.60),
            "sea_ice_temperature":    (-10.0, -2.0),
            "black_carbon":           (0.0,   3000.0),
            "solzen":                 (20,    80),
            "direct":                 (0,     1),
        },
        "transform_fn":  _transform_fyi_pond,
        "n_samples":     10000,
        "emulator_file": str(_DATA_DIR / "sea_ice_FYI_pond_5param.npz"),
    },
    "young_ice": {
        "description": "Young ice — grease ice, nilas, grey ice (0.5–30 cm)",
        "band_mask": "vis_swir",
        "params": {
            "ice_thickness_cm":       (0.5,   30.0),
            "sea_ice_temperature":    (-20.0, -2.0),
            "sea_ice_salinity":       (10.0,  35.0),
            "ocean_albedo":           (0.03,  0.08),
            "solzen":                 (20,    80),
            "direct":                 (0,     1),
        },
        "transform_fn":       _transform_young_ice,
        "n_samples":          8000,
        "hidden_layer_sizes": (128, 128, 64),  # thin-ice physics is smooth
        "emulator_file": str(_DATA_DIR / "sea_ice_young_ice_6param.npz"),
    },
    "open_water": {
        "description": "Open water (ice-free) — analytical Fresnel + subsurface",
        "band_mask": "vis_swir",
        "params": {
            "solzen":                 (20,    80),
            "wind_speed_ms":          (0.0,   15.0),
        },
        # Analytical model — no training data, no emulator file.
        "model_factory": "biosnicar.sea_ice.open_water:OpenWaterModel",
    },
}


def trained_emulator_names():
    """Names of surface types backed by a trained neural-network emulator.

    Excludes analytical models (e.g. ``open_water``) that have no
    training data or ``emulator_file``.
    """
    return [n for n, c in SEA_ICE_EMULATOR_CONFIGS.items() if "emulator_file" in c]


def load_sea_ice_emulators(names=None):
    """Load pre-built sea ice emulators from disk.

    Parameters
    ----------
    names : list of str or None
        Subset of emulator names to load (default: all five).

    Returns
    -------
    dict
        ``{name: Emulator}`` for each requested surface type.

    Raises
    ------
    FileNotFoundError
        If an emulator file has not yet been built.  Run
        ``scripts/build_sea_ice_emulators.py`` to generate them.
    """
    from importlib import import_module

    from biosnicar.emulator import Emulator

    if names is None:
        names = list(SEA_ICE_EMULATOR_CONFIGS)

    emulators = {}
    for name in names:
        cfg = SEA_ICE_EMULATOR_CONFIGS[name]
        factory = cfg.get("model_factory")
        if factory is not None:
            mod_name, cls_name = factory.split(":")
            emulators[name] = getattr(import_module(mod_name), cls_name)()
            continue
        path = cfg["emulator_file"]
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Sea ice emulator '{name}' not found at {path}.\n"
                "Run  python scripts/build_sea_ice_emulators.py  to build it."
            )
        emulators[name] = Emulator.load(path)
    return emulators
