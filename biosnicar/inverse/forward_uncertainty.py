"""Forward-model error from the uncertainty of what the model holds fixed.

A retrieval inverts ``y = F(x, b) + e``.  ``x`` is the state it solves for;
``b`` is everything else the forward model needs and does not solve for: a
refractive-index dataset, the number density of brine inclusions, the
thickness of the ice under the layer being retrieved.  ``b`` is uncertain,
and to first order that puts an error

    S_b = K_b  S_bb  K_b^T ,        K_b = dF/db

into the measurement space (Rodgers 2000, eq. 3.18 and section 3.2.3).  It
adds to instrument noise in ``S_e``.  This module evaluates it.

Every entry of :data:`FACTORS` names one element of ``b``, a one-sigma range
for it and where that range comes from.  For each surface type the forward
model is run at states drawn across the emulator's parameter box with every
factor moved one sigma each way, and the outer products of the resulting
spectral differences are averaged:

    S_type = < sum_i  d_i d_i^T >_states ,   d_i = [F(b_i + s_i) - F(b_i - s_i)] / 2

so the covariance is the *state-averaged* ``K_b S_bb K_b^T`` with ``S_bb``
diagonal (the factors are taken as independent).  Two terms that are not
parameter uncertainty but are also errors of the forward model as it is
actually used are added as realised differences: the emulator against the
model it emulates, and the sea-ice optics table against the physics it
tabulates.  The forward model here is the one the retrievals use: under a
clear sky, a direct beam plus its modelled diffuse part (:mod:`biosnicar.sky`).

What this does and does not contain
-----------------------------------
It contains what is listed in :data:`FACTORS`, at the ranges given there.
It contains nothing that is not listed.  Physics the model does not
represent — liquid water in a melting surface layer, absorbers other than
black carbon, surface roughness, horizontal heterogeneity inside a footprint
— contributes no variance here, so this is a *lower bound* on the true
forward-model error.  Sensor-side error (calibration, atmospheric
correction, the difference between a directional reflectance and an albedo)
is not forward-model error and belongs in ``obs_uncertainty``.

Nothing here is fitted to field measurements.  Field residuals are used
afterwards, to ask whether the list is complete
(``tests/validation_data/physics_budget_vs_field.py``).

State averaging is an approximation: the true ``K_b`` depends on the state,
so the covariance is too wide for some states and too narrow for others.
The average is the marginal covariance of the forward-model error under a
uniform prior over the emulator box.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np

from biosnicar import perturbation

WAVELENGTHS_UM = np.arange(0.205, 4.999, 0.01)


@dataclass(frozen=True)
class Factor:
    """One uncertain forward-model input.

    kind
        ``"additive"``        b -> b +/- sigma
        ``"multiplicative"``  b -> b * sigma, b / sigma   (sigma is a factor)
        ``"alternative"``     b -> ``sigma`` (a second defensible choice).
                              One sigma is half the difference between the
                              two choices.
    where
        ``"physics"``  a switch in :mod:`biosnicar.perturbation`
        ``"fixed"``    a key of the surface type's ``fixed`` dict
                       (:mod:`biosnicar.sea_ice.emulator_configs`)
        ``"state"``    a parameter the retrieval is told rather than solves
                       for (solar zenith)
    """

    name: str
    where: str
    key: str
    kind: str
    sigma: float
    basis: str
    source: str = ""
    types: tuple = field(default=())      # empty = every type the key exists for


# One-sigma ranges.  PROPOSED VALUES: each is a judgement from the cited
# source, written down so it can be argued with.  Changing one and rerunning
# scripts/build_physics_model_error.py is the whole procedure.
FACTORS = (
    # ── optical constants ────────────────────────────────────────────────
    Factor(
        "ice_absorption", "physics", "ice_absorption_blend", "alternative", 1.0,
        "Absorption of pure ice, 320-600 nm. The model uses Picard et al. "
        "(2016); the alternative is Warren & Brandt (2008), up to an order "
        "of magnitude lower near 400 nm. One sigma = half the difference.",
        "Picard et al. 2016 (TC 10:2655); Warren & Brandt 2008 (JGR 113:D14220)",
    ),
    Factor(
        "brine_index", "physics", "brine_index_offset", "additive", 0.002,
        "Real refractive index of brine. Sets the index contrast with ice "
        "(about 0.03) that brine inclusions scatter by, so 0.002 is ~7% of "
        "the contrast. Covers the spread between the Maykut & Light anchors "
        "and the Quan & Fry extrapolation beyond their salinity range.",
        "Maykut & Light 1995 (Appl. Opt. 34:950); Quan & Fry 1995",
    ),
    Factor(
        "pond_water_absorption", "physics", "water_absorption_scale",
        "multiplicative", 1.1,
        "Absorption of melt-pond water: pure-water absorption is known to "
        "~10% outside the blue; dissolved and particulate matter in real "
        "ponds is not represented at all (not included here).",
        "Pope & Fry 1997; Rowe et al. 2020",
        types=("FYI_pond",),
    ),
    # ── sea-ice microstructure ───────────────────────────────────────────
    Factor(
        "brine_inclusion_density", "physics", "brine_inclusion_scale",
        "multiplicative", 1.5,
        "Scattering cross-section of brine inclusions per unit brine volume. "
        "The size distribution is one set of measurements on interior "
        "first-year ice at -15 degC, applied to all ice.",
        "Light, Maykut & Grenfell 2003 (JGR 108:3051), 2004 (JGR 109:C06013)",
    ),
    Factor(
        "mirabilite_size", "physics", "mirabilite_scale", "multiplicative", 2.0,
        "Scattering by mirabilite crystals (below -8.2 degC). Cross-section "
        "per unit mass goes as 1/size; a single 10 um crystal size is assumed.",
        "Light, Maykut & Grenfell 2004",
    ),
    Factor(
        "hydrohalite_size", "physics", "hydrohalite_scale", "multiplicative", 3.0,
        "Scattering by hydrohalite crystals (below -22.9 degC). The model "
        "gives about a third of the scattering Light et al. infer from "
        "observation; the range spans that gap.",
        "Light, Maykut & Grenfell 2004",
    ),
    Factor(
        "bubble_backscatter", "physics", "bubble_backscatter_scale",
        "multiplicative", 1.1,
        "(1 - g) of air bubbles: Mie theory for monodisperse spheres; real "
        "bubbles are polydisperse and not spherical.",
        "judgement",
    ),
    Factor(
        "young_ice_scattering", "physics", "young_ice_scattering_scale",
        "multiplicative", 1.5,
        "Scattering coefficient of frazil / young ice, a single constant "
        "set from broadband albedo brackets.",
        "Grenfell & Maykut 1977; Perovich 1996",
        types=("young_ice",),
    ),
    # ── interfaces and boundaries ────────────────────────────────────────
    Factor(
        "fresnel_diffuse", "physics", "fresnel_diffuse_offset", "alternative",
        0.02,
        "Diffuse reflectance of the ice interface from below: the inherited "
        "table is ~0.02 below the value the n^2 radiance law gives.",
        "this repository, docs/sea_ice_validation.md",
    ),
    Factor(
        "ocean_reflectance", "physics", "ocean_reflectance_scale",
        "multiplicative", 1.5,
        "Reflectance of the water under the ice (pure-seawater Morel model; "
        "real under-ice water carries chlorophyll and CDOM).",
        "Morel & Prieur 1977",
    ),
    # ── column structure the retrieval does not solve for ────────────────
    Factor(
        "drained_layer_thickness", "fixed", "dl_thickness", "multiplicative",
        2.0,
        "Thickness of the drained layer above the waterline.",
        "Light, Grenfell & Perovich 2008 (JGR 113:C03023)",
    ),
    Factor(
        "interior_thickness", "fixed", "il_thickness", "multiplicative", 1.5,
        "Thickness of the ice interior (fixed at 1.4-1.45 m for first-year "
        "ice, 2.7 m for multiyear).",
        "judgement: level-ice thickness range",
    ),
    Factor(
        "drained_layer_density", "fixed", "dl_density", "additive", 30.0,
        "Bulk density of the drained layer, where it is not retrieved.",
        "Timco & Frederking 1996 (CRST 24:1)",
    ),
    Factor(
        "interior_density", "fixed", "il_density", "additive", 10.0,
        "Bulk density of the ice interior.",
        "Timco & Frederking 1996",
    ),
    Factor(
        "salinity", "fixed", "salinity_scale", "multiplicative", 1.3,
        "Bulk salinity of the ice layers (fixed per surface type). Where "
        "brine volume is the retrieved parameter it is held and the "
        "temperature moves instead.",
        "judgement: first-year 4-8, multiyear 1-3 ppt",
    ),
    Factor(
        "interior_bubble_size", "fixed", "il_bubble_ratio", "multiplicative",
        1.5,
        "Bubble radius of the interior relative to the drained layer.",
        "judgement",
    ),
    Factor(
        "drained_layer_bubble_size", "fixed", "dl_bubble_radius",
        "multiplicative", 2.0,
        "Bubble radius of the ice under snow, where it is not retrieved.",
        "judgement",
    ),
    Factor(
        "grain_shape", "fixed", "grain_shape", "alternative", 1,
        "Grains of snow and of the surface scattering layer are spheres in "
        "the model; the alternative is spheroids (He et al. 2017), the "
        "mildest non-spherical shape available. One sigma = half the "
        "difference.",
        "He et al. 2017 (J. Climate 30:10019); Libois et al. 2013",
    ),
    # ── illumination ─────────────────────────────────────────────────────
    Factor(
        "solar_zenith", "state", "solzen", "additive", 2.0,
        "Angle between sun and surface normal: surface slope and roughness.",
        "judgement",
    ),
    Factor(
        "diffuse_sky_fraction", "physics", "diffuse_fraction_scale",
        "multiplicative", 1.3,
        "Diffuse fraction of a clear sky (biosnicar.sky): aerosol optical "
        "depth 0.02-0.15 moves it by about this much at 500 nm, and the "
        "albedo of the surroundings (assumed 0.8; anything from open water "
        "to fresh snow) moves it by up to 25% in the blue.",
        "Tomasi et al. 2015 (Earth-Sci. Rev. 140:108); Hansen & Travis 1974",
    ),
)

# Realised (not perturbed) terms, appended after FACTORS in every budget.
LUT = "optics_table_interpolation"
EMULATOR = "emulator"
REALISED = (LUT, EMULATOR)
REALISED_BASIS = {
    LUT: "Sea-ice layer optics are interpolated from a table; the term is "
         "the table against the physics it was built from.",
    EMULATOR: "The neural-network emulator against the forward model.",
}


def factor_names():
    return [f.name for f in FACTORS] + list(REALISED)


def factor_table():
    """Every term of the budget with its range, basis and source, as plain
    dicts — what a built artifact records about how it was built."""
    table = [dict(name=f.name, where=f.where, key=f.key, kind=f.kind,
                  sigma=f.sigma, basis=f.basis, source=f.source)
             for f in FACTORS]
    table += [dict(name=n, where="realised", key="", kind="realised",
                   sigma=None, basis=REALISED_BASIS[n], source="")
              for n in REALISED]
    return table


# ── forward model at a perturbed b ───────────────────────────────────────

def _config(type_name):
    from biosnicar.sea_ice.emulator_configs import SEA_ICE_EMULATOR_CONFIGS
    return SEA_ICE_EMULATOR_CONFIGS[type_name]


_ANALYTICAL = {}


def forward(type_name, params, fixed=None, **switches):
    """480-band albedo of *type_name* at state *params* and perturbed ``b``.

    *fixed* overrides entries of the type's fixed-input dict; *switches* are
    :mod:`biosnicar.perturbation` switches.  A clear sky (``direct=1``) is a
    beam plus its modelled diffuse part, as in the retrievals.
    """
    from biosnicar.sky import CLEAR_SKY, mixed_sky

    cfg = _config(type_name)
    with perturbation.perturbed(**switches), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "transform_fn" in cfg:
            from biosnicar.drivers.run_model import run_model
            kwargs = cfg["transform_fn"](dict(params), fixed=fixed)
            return np.asarray(
                run_model(diffuse_fraction=CLEAR_SKY, **kwargs).albedo,
                dtype=float)
        if fixed:
            raise KeyError(f"{type_name} has no fixed inputs")
        model = _ANALYTICAL.get(type_name)
        if model is None:
            from importlib import import_module
            mod, cls = cfg["model_factory"].split(":")
            model = _ANALYTICAL[type_name] = mixed_sky(
                getattr(import_module(mod), cls)())
        return np.asarray(model.predict(**params), dtype=float)


def sample_states(type_name, n, seed=0):
    """*n* states across the type's parameter box, sampled the way the
    emulator's training set is (Latin hypercube; log for log parameters;
    integer solar zenith; binary ``direct``)."""
    from biosnicar.emulator import _LOG_SAMPLE_PARAMS, _latin_hypercube

    params = dict(_config(type_name)["params"])
    params.setdefault("direct", (0, 1))        # analytical models take it too
    names = list(params)
    unit = _latin_hypercube(n, len(names), seed=seed)
    states = []
    for row in unit:
        p = {}
        for u, name in zip(row, names):
            lo, hi = params[name]
            if name in _LOG_SAMPLE_PARAMS:
                llo, lhi = np.log10(lo + 1), np.log10(hi + 1)
                v = 10 ** (llo + u * (lhi - llo)) - 1
            else:
                v = lo + u * (hi - lo)
            if name == "direct":
                v = int(round(v))
            elif name == "solzen":
                v = int(round(v))
            p[name] = float(v) if name not in ("direct", "solzen") else v
        states.append(p)
    return states


def applicable_factors(type_name):
    """The entries of :data:`FACTORS` that exist for *type_name*."""
    cfg = _config(type_name)
    fixed = cfg.get("fixed", {})
    out = []
    for f in FACTORS:
        if f.types and type_name not in f.types:
            continue
        if f.where == "fixed" and f.key not in fixed:
            continue
        if f.where == "state" and f.key not in cfg["params"]:
            continue
        out.append(f)
    return out


def _plus_minus(f, base):
    if f.kind == "additive":
        return base + f.sigma, base - f.sigma
    if f.kind == "multiplicative":
        return base * f.sigma, base / f.sigma
    raise ValueError(f.kind)


def factor_delta(type_name, params, f, baseline=None):
    """One-sigma spectral response of *type_name* to factor *f* at *params*.

    *baseline* is ``forward(type_name, params, sea_ice_direct=True)``; it is
    needed only for ``"alternative"`` factors and computed if not given.
    """
    direct = dict(sea_ice_direct=True)

    def run(value):
        if f.where == "physics":
            return forward(type_name, params, **direct, **{f.key: value})
        if f.where == "fixed":
            return forward(type_name, params, fixed={f.key: value}, **direct)
        p = dict(params)
        p[f.key] = value
        return forward(type_name, p, **direct)

    if f.kind == "alternative":
        if baseline is None:
            baseline = forward(type_name, params, **direct)
        return 0.5 * (run(f.sigma) - baseline)

    if f.where == "physics":
        base = perturbation.NEUTRAL[f.key]
    elif f.where == "fixed":
        base = _config(type_name)["fixed"][f.key]
    else:
        base = params[f.key]
    hi, lo = _plus_minus(f, base)
    if f.where == "state" and f.key == "solzen":
        # The irradiance archive is per whole degree: difference over the
        # integer step actually taken and rescale to one sigma.
        hi, lo = int(round(min(hi, 89))), int(round(max(lo, 1)))
        if hi == lo:
            return np.zeros(WAVELENGTHS_UM.size)
        return (run(hi) - run(lo)) * f.sigma / (hi - lo)
    return 0.5 * (run(hi) - run(lo))


def state_deltas(type_name, params, emulator=None):
    """All one-sigma responses at one state.

    Returns ``{name: (480,) array}`` over :func:`factor_names`; a factor that
    does not exist for the type is all zeros.
    """
    nb = WAVELENGTHS_UM.size
    out = {name: np.zeros(nb) for name in factor_names()}
    base_direct = forward(type_name, params, sea_ice_direct=True)
    for f in applicable_factors(type_name):
        out[f.name] = factor_delta(type_name, params, f, baseline=base_direct)

    # The model as shipped (table lookup) against the physics.
    base_lut = forward(type_name, params)
    out[LUT] = base_lut - base_direct

    if emulator is not None and hasattr(emulator, "_forward_pass"):
        from biosnicar.sky import mixed_sky
        pred = np.asarray(mixed_sky(emulator).predict(
            **{k: params[k] for k in emulator.param_names}), dtype=float)
        out[EMULATOR] = pred - base_lut
    return out


def _state_deltas_row(type_name, params, emulator):
    d = state_deltas(type_name, params, emulator)
    return np.stack([d[n] for n in factor_names()]).astype(np.float32)


def type_deltas(type_name, n_states=300, seed=0, n_jobs=1, emulator=None):
    """One-sigma responses at *n_states* states: ``(n_states, n_factors, 480)``.

    The factor axis follows :func:`factor_names`.
    """
    states = sample_states(type_name, n_states, seed=seed)
    if n_jobs == 1:
        rows = [_state_deltas_row(type_name, p, emulator) for p in states]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_state_deltas_row)(type_name, p, emulator) for p in states)
    return np.stack(rows), states


def covariance_from_deltas(deltas):
    """State-averaged ``sum_i d_i d_i^T`` and its per-factor diagonal.

    Returns ``(S (480, 480), factor_var (n_factors, 480))``.
    """
    d = np.asarray(deltas, dtype=float)
    n_states, n_factors, nb = d.shape
    flat = d.reshape(n_states * n_factors, nb)
    S = flat.T @ flat / n_states
    factor_var = (d ** 2).mean(axis=0)
    return S, factor_var


def build_physics_model_error(types=None, n_states=300, seed=0, n_jobs=1,
                              max_rank=40, energy=0.999, progress=None):
    """Evaluate the budget for every surface type.

    Returns a :class:`~biosnicar.inverse.model_error.PhysicsModelError`.
    """
    from biosnicar.inverse.model_error import (
        ModelErrorCovariance, PhysicsModelError,
    )
    from biosnicar.sea_ice.emulator_configs import (
        SEA_ICE_EMULATOR_CONFIGS, load_sea_ice_emulators,
    )

    types = list(types or SEA_ICE_EMULATOR_CONFIGS)
    fleet = load_sea_ice_emulators(types)
    by_type, factor_var = {}, {}
    for i, name in enumerate(types):
        deltas, _ = type_deltas(name, n_states=n_states, seed=seed + i,
                                n_jobs=n_jobs, emulator=fleet[name])
        S, fv = covariance_from_deltas(deltas)
        by_type[name] = ModelErrorCovariance.from_dense(
            S, max_rank=max_rank, energy=energy,
            meta=dict(surface_type=name, n_states=int(n_states)))
        factor_var[name] = fv
        if progress:
            progress(name, by_type[name], fv)
    return PhysicsModelError(
        by_type=by_type, factor_names=factor_names(), factor_var=factor_var,
        meta=dict(source="physics", n_states=int(n_states), seed=int(seed),
                  factors=factor_table()),
    )
