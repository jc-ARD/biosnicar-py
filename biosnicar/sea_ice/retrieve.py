"""Retrieve sea ice surface properties from spectral or satellite albedo.

Provides :func:`retrieve_sea_ice`, a high-level inversion function that fits
a fleet of surface-type forward models against an observed spectrum and
classifies the most likely ice surface type by residual comparison.

The seven surface types are:

    FYI_bare    — winter/spring bare first-year ice
    FYI_snow    — snow-covered first-year ice (tau_snow parameterisation)
    FYI_summer  — melt-season bare ice with Surface Scattering Layer
    MYI_bare    — bare multiyear ice
    FYI_pond    — melt pond on first-year ice
    young_ice   — semi-transparent thin ice, 0.5–30 cm (layer_type=6)
    open_water  — ice-free ocean (analytical, no training data)

Physical parameters (bubble radius, temperature, pond depth, snow depth, …)
are retrieved from the winning emulator's parameter space.  The surface type
label is assigned retroactively from which emulator produced the lowest
chi-squared residual against the observation.

Usage::

    from biosnicar.sea_ice.retrieve import retrieve_sea_ice

    # Full spectral inversion (480-band observed albedo)
    result = retrieve_sea_ice(observed=spectrum, solzen=60)
    print(result.surface_type)       # e.g. "FYI_summer"
    print(result.confidence)         # 0–1, how decisively it won
    print(result.parameters)         # {"sea_ice_bubble_radius": 220, ...}
    out = result.to_outputs()
    out.to_platform("sentinel2")     # works — flx_slr is preserved

    # Satellite band mode
    result = retrieve_sea_ice(
        observed=[0.82, 0.65, 0.08],
        platform="sentinel2",
        observed_band_names=["B3", "B8", "B11"],
        solzen=60,
    )
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from biosnicar.inverse.result import RetrievalResult
from biosnicar.sea_ice.scene_result import SURFACE_TYPE_CODES  # noqa: F401  (public API)


# ── SeaIceRetrievalResult ────────────────────────────────────────────────────

# Human-readable descriptions for each internal surface-type label.
# These are the optical-state meanings of the emulator types, independent
# of WMO/SIGRID-3 structural classification.
_SURFACE_TYPE_DESCRIPTIONS = {
    "FYI_bare":   "First-year ice, bare surface (winter/spring, no snow, no SSL)",
    "FYI_snow":   "First-year ice, snow-covered",
    "FYI_summer": "First-year ice, ablating — Surface Scattering Layer present",
    "MYI_bare":   "Multiyear ice, bare surface (desalinated, large bubbles)",
    "FYI_pond":   "Ice surface with melt ponds (any ice age)",
    "young_ice":  "Young/new ice — semi-transparent, 0.5–30 cm thick",
    "open_water": "Open water (ice-free) — Fresnel reflection + subsurface scattering",
}


@dataclass
class SeaIceRetrievalResult:
    """Result returned by :func:`retrieve_sea_ice`.

    Attributes
    ----------
    surface_type : str
        Internal emulator label for the best-fitting surface type
        (e.g. ``"FYI_summer"``).  These labels describe the **optical
        state** of the surface, not WMO structural categories.  Use
        ``surface_description`` for a human-readable explanation, or call
        ``to_wmo()`` / ``to_sigrid3()`` for approximate WMO/SIGRID-3 codes.

        The internal labels are kept as primary because they encode
        distinctions that WMO codes collapse: for example, ``FYI_bare``,
        ``FYI_snow``, and ``FYI_summer`` all map to ``SM/SN`` in SIGRID-3
        but have completely different spectral signatures and forward-model
        physics.  Switching to WMO codes as primary would discard real,
        retrievable information.
    surface_description : str
        One-line human-readable explanation of what ``surface_type`` means
        in physical terms (e.g. ``"First-year ice, ablating — Surface
        Scattering Layer present"``).
    confidence : float
        How decisively the winner outperformed the next-best candidate:
        ``(second_best_cost - best_cost) / second_best_cost``.
        0 = tied; approaches 1 as the winner dominates.
    parameters : dict
        ``{param_name: best_value}`` from the winning emulator.
    uncertainty : dict
        ``{param_name: 1_sigma}`` from the winning emulator.
    predicted_albedo : np.ndarray
        480-band spectral albedo at the best-fit point.
    observed : np.ndarray
        Input observations.
    cost : float
        Classification chi-squared of the winner (band-masked, rescaled
        to the full observation band count — see ``_classification_cost``).
        Per-emulator fitting costs are in ``all_fits[name].cost``.
    converged : bool
        Whether the winning optimiser reported convergence.
    flx_slr : np.ndarray or None
        Solar flux spectrum (from the winning emulator).
    cost_per_type : dict
        ``{surface_type: chi_squared}`` for every fitted surface type.
    all_fits : dict
        ``{surface_type: RetrievalResult}`` — full result for each type.
    quality_flags : int
        uint8 bitmask of quality conditions — see
        :class:`~biosnicar.sea_ice.quality_flags.QualityFlag`.  Unpack with
        :meth:`quality_flag_description`.
    """

    surface_type: str
    surface_description: str   # auto-populated from _SURFACE_TYPE_DESCRIPTIONS
    confidence: float
    parameters: Dict[str, float]
    uncertainty: Dict[str, float]
    predicted_albedo: np.ndarray
    observed: np.ndarray
    cost: float
    converged: bool
    flx_slr: Optional[np.ndarray]
    cost_per_type: Dict[str, float] = field(default_factory=dict)
    all_fits: Dict[str, "RetrievalResult"] = field(default_factory=dict)
    quality_flags: int = 0

    # Optimal-estimation outputs (populated only when method="oe").
    # class_probabilities: posterior probability per surface type from the
    #   Laplace model evidence (proper Bayesian classification). For "oe",
    #   `confidence` is the winning type's probability.
    # dfs / averaging_kernel_diag: information content of the winning fit —
    #   how much each parameter (and the retrieval overall) came from the
    #   measurement vs the prior.
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    dfs: Optional[float] = None
    averaging_kernel_diag: Dict[str, float] = field(default_factory=dict)

    # A3 provenance — priors-on vs spectrum-only (populated only when
    # retrieve_sea_ice is called with flag_prior_influence=True and the
    # automatic metadata priors are active):
    #   prior_resolved: True if the metadata prior (known_month) changed the
    #     winning surface type relative to a spectrum-only retrieval — i.e. the
    #     classification is resolved by the prior, not the spectrum.  None when
    #     not diagnosed or no metadata prior was active.
    #   spectrum_only_*: the spectrum-only winner and its class distribution,
    #     kept alongside prior_resolved so the disagreement is transparent.
    prior_resolved: Optional[bool] = None
    spectrum_only_surface_type: Optional[str] = None
    spectrum_only_class_probabilities: Dict[str, float] = field(default_factory=dict)

    def prior_dominated_parameters(self, threshold: float = 0.5) -> Dict[str, bool]:
        """Per-parameter provenance from the averaging kernel (``method="oe"``).

        Returns ``{param: prior_dominated}`` where the value is True when the
        parameter's averaging-kernel diagonal is below *threshold* — i.e. its
        retrieved value came mostly from the prior rather than the measurement
        (low information content).  Empty unless ``method="oe"`` populated
        ``averaging_kernel_diag``.
        """
        return {p: (a < threshold) for p, a in self.averaging_kernel_diag.items()}

    def quality_flag_description(self) -> Dict[str, bool]:
        """Unpack the quality bitmask into ``{flag_name: bool}``."""
        from biosnicar.sea_ice.quality_flags import describe_quality_flags
        return describe_quality_flags(self.quality_flags)

    def to_wmo(self):
        """Map to WMO Sea Ice Nomenclature (WMO No. 259).

        Returns a :class:`~biosnicar.sea_ice.ice_chart_mapping.WMOIceClass`
        with stage of development, surface description, melt stage, and
        caveats about what cannot be inferred from albedo alone.
        """
        from biosnicar.sea_ice.ice_chart_mapping import map_to_wmo
        return map_to_wmo(self)

    def to_sigrid3(self):
        """Map to a partial SIGRID-3 ice-class description.

        Returns a :class:`~biosnicar.sea_ice.ice_chart_mapping.SIGRID3IceClass`
        with ice-type code, stage of melt, melt-pond flag, and the partial
        SIGRID-3 code string.  Fields requiring structural measurement
        (concentration, thickness, floe size) are marked ``??``.
        """
        from biosnicar.sea_ice.ice_chart_mapping import map_to_sigrid3
        return map_to_sigrid3(self)

    def classification_summary(self) -> str:
        """Formatted WMO + SIGRID-3 classification summary string."""
        from biosnicar.sea_ice.ice_chart_mapping import classification_summary
        return classification_summary(self)

    def to_outputs(self):
        """Wrap the winning predicted spectrum as an ``Outputs`` object.

        Returns an :class:`~biosnicar.classes.outputs.Outputs` instance with
        ``.BBA``, ``.albedo``, ``.to_platform()``, and all other attributes
        populated.  Requires ``flx_slr`` to be present (i.e. the winning
        emulator must have been built from a pre-run forward model).
        """
        # Delegate to the underlying RetrievalResult machinery
        winner = self.all_fits[self.surface_type]
        return winner.to_outputs()

    def summary(self) -> str:
        """Human-readable summary including optical-state description and WMO stage."""
        from biosnicar.sea_ice.ice_chart_mapping import map_to_wmo
        wmo = map_to_wmo(self)
        lines = [
            "SeaIceRetrievalResult",
            f"  Surface type   : {self.surface_type}  (confidence={self.confidence:.3f})",
            f"  Description    : {self.surface_description}",
            f"  WMO stage      : {wmo.stage_of_development} — {wmo.melt_stage}",
            f"  Cost: {self.cost:.4f}  converged={self.converged}",
        ]
        set_flags = [n for n, on in self.quality_flag_description().items() if on]
        lines.append(f"  Quality flags  : {', '.join(set_flags) if set_flags else 'none'}")
        lines.append("  Retrieved parameters:")
        for name, val in self.parameters.items():
            unc = self.uncertainty.get(name, float("nan"))
            lines.append(f"    {name:28s} = {val:12.4f}  ±  {unc:.4f}")
        lines.append("  Cost per surface type:")
        ranked = sorted(self.cost_per_type.items(), key=lambda kv: kv[1])
        for stype, cost in ranked:
            desc = _SURFACE_TYPE_DESCRIPTIONS.get(stype, "")
            marker = " ←" if stype == self.surface_type else ""
            lines.append(f"    {stype:14s}  {cost:.4f}{marker}  {desc}")
        return "\n".join(lines)


# ── Classification cost (C2: per-emulator band masks) ───────────────────────

_BAND_CENTER_CACHE: Dict[str, Dict[str, float]] = {}


def _band_mask_bool(mask_name):
    """Resolve a named band mask to a boolean (480,) wavelength selector."""
    from biosnicar.bands._core import WVL
    from biosnicar.sea_ice.emulator_configs import BAND_MASKS

    lo, hi = BAND_MASKS[mask_name]
    return (WVL >= lo - 1e-9) & (WVL <= hi + 1e-9)


def _band_centers_um(platform):
    """SRF-weighted centre wavelength (µm) per band of *platform*.

    Convolving the wavelength grid itself through ``to_platform`` (with
    flat flux) yields each band's response-weighted mean wavelength —
    works for every platform regardless of how its SRFs are defined.
    """
    if platform not in _BAND_CENTER_CACHE:
        from biosnicar.bands import to_platform
        from biosnicar.bands._core import WVL

        result = to_platform(WVL, platform, flx_slr=np.ones(480))
        _BAND_CENTER_CACHE[platform] = {
            b: float(getattr(result, b)) for b in result.band_names
        }
    return _BAND_CENTER_CACHE[platform]


def _classification_cost(fit, observed, mask_name, platform,
                         observed_band_names, obs_uncertainty,
                         wavelength_mask, regularization):
    """Rank-and-diagnose metrics for one fitted surface type.

    Returns ``(cost, rms)``: the classification chi-squared (band-masked,
    rescaled to the full observation band count so costs remain comparable
    across emulators with different masks, plus the Gaussian prior penalty
    so seasonal priors keep steering classification) and the *unweighted*
    RMS albedo residual over the observation, which is scale-free and feeds
    the POOR_FIT quality flag.
    """
    observed = np.asarray(observed, dtype=float)

    if platform is not None:
        from biosnicar.bands import to_platform

        band_result = to_platform(
            fit.predicted_albedo, platform, flx_slr=fit.flx_slr,
        )
        predicted = np.array(
            [getattr(band_result, b) for b in observed_band_names]
        )
        residual = predicted - observed
        rms = float(np.sqrt(np.mean(residual ** 2)))
        if obs_uncertainty is not None:
            residual = residual / np.asarray(obs_uncertainty, dtype=float)
        sel = np.ones(len(observed), dtype=bool)
        if mask_name is not None:
            from biosnicar.sea_ice.emulator_configs import BAND_MASKS

            lo, hi = BAND_MASKS[mask_name]
            centers = _band_centers_um(platform)
            sel = np.array(
                [lo <= centers[b] <= hi for b in observed_band_names]
            )
            if not sel.any():
                sel[:] = True
        n_ref = len(observed)
    else:
        residual = fit.predicted_albedo - observed
        base = np.isfinite(observed)
        if wavelength_mask is not None:
            base &= np.asarray(wavelength_mask, dtype=bool)
        rms = float(np.sqrt(np.mean(residual[base] ** 2))) if base.any() else np.inf
        if obs_uncertainty is not None:
            unc = np.asarray(obs_uncertainty, dtype=float)
            residual = np.where(base, residual / unc, 0.0)
        sel = base.copy()
        if mask_name is not None:
            sel &= _band_mask_bool(mask_name)
            if not sel.any():
                sel = base
        n_ref = int(base.sum())

    cost = float(np.sum(residual[sel] ** 2)) * (n_ref / max(int(sel.sum()), 1))

    if regularization:
        for name, (mu, sigma) in regularization.items():
            if name in fit.best_fit:
                cost += ((fit.best_fit[name] - mu) / sigma) ** 2
    return cost, rms


# ── retrieve_sea_ice ─────────────────────────────────────────────────────────

def retrieve_sea_ice(
    observed,
    emulators=None,
    surface_types=None,
    platform=None,
    observed_band_names=None,
    obs_uncertainty=None,
    method="L-BFGS-B",
    solzen=None,
    direct=None,
    fixed_params=None,
    bounds=None,
    x0=None,
    regularization=None,
    wavelength_mask=None,
    known_month=None,
    use_priors=True,
    flag_prior_influence=False,
    model_error=None,
    class_priors=None,
    prior_sources=None,
    mcmc_walkers=32,
    mcmc_steps=2000,
    mcmc_burn=500,
) -> SeaIceRetrievalResult:
    """Retrieve sea ice physical properties and classify surface type.

    Fits each of the seven sea ice surface-type models against *observed*
    and returns the best-fit parameters together with the most likely surface
    type (the emulator that achieved the lowest chi-squared residual).

    Parameters
    ----------
    observed : array-like
        Observed albedo.  Either a 480-element spectral array or an
        N-element array of satellite band albedos (with *platform* and
        *observed_band_names* set).
    emulators : dict or None
        ``{surface_type: Emulator}`` to use.  If None, loads the seven
        pre-built emulators from ``data/emulators/``.  Pass a subset to
        restrict which surface types are considered.
    surface_types : list of str or None
        Restrict fitting to this subset of surface type names.
    platform : str or None
        Satellite platform key (e.g. ``"sentinel2"``).  Required when
        *observed* is a band array rather than a full spectrum.
    observed_band_names : list of str or None
        Band names corresponding to *observed* entries (e.g. ``["B3","B8"]``).
    obs_uncertainty : array-like or None
        Per-observation 1-sigma uncertainty for chi-squared weighting.
        When None and *platform* is set, defaults are taken from the
        platform SNR table in :mod:`biosnicar.sea_ice.sensor_config`.
    method : str
        Optimisation method: ``"L-BFGS-B"`` (default), ``"Nelder-Mead"``,
        ``"differential_evolution"``, or ``"mcmc"``.

        ``"mcmc"`` (requires ``emcee``) samples the full posterior per
        emulator and populates ``uncertainty`` with the posterior standard
        deviation instead of the Hessian approximation.  Expect minutes per
        observation (vs seconds for L-BFGS-B) — use it on selected pixels
        for publication-grade uncertainty, not for scene processing.
        Tune with *mcmc_walkers*, *mcmc_steps*, *mcmc_burn*; the winner's
        ``all_fits[surface_type].chains`` holds the post-burn-in chain.
    solzen : float or None
        Solar zenith angle (degrees).  When provided, fixed for all
        emulators rather than retrieved.
    direct : int or None
        Illumination flag (1=direct, 0=diffuse).  Binary, so it is always
        fixed, never retrieved.  **Defaults to 1 (direct beam) when omitted**
        — pass ``direct=0`` explicitly for overcast/diffuse conditions.
    fixed_params : dict or None
        Additional parameters to fix across all emulators.  Merged with
        *solzen* and *direct* when those are provided.
    bounds, x0, regularization, wavelength_mask
        Passed through to each :func:`~biosnicar.inverse.optimize.retrieve`
        call.
    known_month : int or None
        Calendar month (1–12) of the observation.  When provided, physically
        motivated Gaussian priors are automatically added to prevent emulators
        from exploiting temperature values that are impossible in the given
        season.  This is the most important correction for summer bare ice
        classification: without it, FYI_snow can fit August spectra using
        T = −25°C (physically impossible) and win over FYI_summer/FYI_bare.

        **Summer months (5–9):** ``sea_ice_temperature`` is constrained to
        (−4°C ± 3°C) for emulators that have a temperature parameter.  This
        rules out the T < −10°C regime that is unreachable in the melt season.
        ``brine_volume_fraction`` for bare-ice emulators is constrained to
        (0.07 ± 0.04), consistent with T ≈ −5°C at the reference salinity.

        **Winter months (11–3):** ``sea_ice_temperature`` is constrained to
        (−15°C ± 8°C), preventing the optimizer from using near-melting
        temperatures that would be physically unrealistic in deep winter.

        Any explicit *regularization* dict passed by the caller is merged on
        top of the season priors — caller values take precedence.
    use_priors : bool
        When True (default), the automatic *known_month*-derived priors are
        applied: the seasonal temperature/thickness priors, their per-emulator
        brine-volume translation, and the melt-season young-ice exclusion.
        When False, none of these are applied — the retrieval and
        classification are **spectrum-only** (an explicit *regularization* dict
        you pass is still honoured; only the metadata-derived priors are
        dropped).  Use this to see what the spectrum alone constrains.
    model_error : None, True, or ModelErrorCovariance
        A7 forward-model error term for ``method="oe"``, spectral mode only.
        ``True`` loads the shipped field-calibrated covariance
        (``biosnicar.inverse.model_error``, SHEBA+Smith correct-model
        residuals, Istomina held out); or pass a
        :class:`~biosnicar.inverse.model_error.ModelErrorCovariance`.
        Adds the empirical low-rank residual covariance to S_e so spectral
        evidence is not over-counted (field residuals have ~2 effective DOF
        over 60 VIS bands, not 60). Default None = instrument-noise-only
        (the historical behaviour, known to yield overconfident evidence —
        see docs/sea_ice_validation.md §12).
    class_priors : dict or None
        Caller-supplied per-surface-type **class log-priors** (relative,
        0 = neutral, negative = disfavoured), composed with the spectral
        evidence / classification cost via the C1 metadata-prior adapter
        (:mod:`biosnicar.sea_ice.metadata_priors`). This is how location,
        ice-age or skin-temperature context enters the classification —
        e.g. ``{"MYI_bare": 1.5, "FYI_bare": -1.5}`` in an MYI zone. Ignored
        under ``use_priors=False``.
    prior_sources : sequence of str or None
        Which built-in metadata-prior providers to run (default: all;
        currently just ``"season"``, driven by *known_month*). Pass a subset
        (or ``()``) so an ensemble can withhold providers whose signal the
        fusion layer already owns, avoiding double-counting.
    flag_prior_influence : bool
        When True and the metadata priors are actually active, run one extra
        spectrum-only pass and populate the result's provenance fields
        (``prior_resolved``, ``spectrum_only_surface_type``,
        ``spectrum_only_class_probabilities``).  ``prior_resolved`` is True if
        the prior changed the winning surface type.  Costs a second emulator-
        fleet fit, so it is opt-in (off by default).

    Returns
    -------
    SeaIceRetrievalResult

    Raises
    ------
    FileNotFoundError
        If pre-built emulators are not found.  Run
        ``python scripts/build_sea_ice_emulators.py`` to generate them.
    """
    from biosnicar.inverse.optimize import retrieve
    from biosnicar.sea_ice.emulator_configs import (
        SEA_ICE_EMULATOR_CONFIGS,
        load_sea_ice_emulators,
    )

    observed = np.asarray(observed, dtype=float)

    # Non-finite handling: resampled field spectra legitimately carry NaN
    # outside instrument coverage — fold them into the wavelength mask so
    # the fitting cost (which does not mask NaN itself) stays finite.
    finite = np.isfinite(observed)
    if not finite.all():
        if platform is not None:
            raise ValueError(
                "observed contains non-finite band values — band mode "
                "cannot mask individual bands; drop them from `observed` "
                "and `observed_band_names` instead."
            )
        wavelength_mask = (
            finite if wavelength_mask is None
            else np.asarray(wavelength_mask, dtype=bool) & finite
        )
        if not wavelength_mask.any():
            raise ValueError("observed has no finite values.")
        observed = np.nan_to_num(observed)

    # Default per-band uncertainty from the platform SNR table (C3)
    if (obs_uncertainty is None and platform is not None
            and observed_band_names is not None):
        from biosnicar.sea_ice.sensor_config import default_obs_uncertainty
        obs_uncertainty = default_obs_uncertainty(platform, observed_band_names)

    # Load emulators if not supplied
    if emulators is None:
        names = surface_types or list(SEA_ICE_EMULATOR_CONFIGS)
        emulators = load_sea_ice_emulators(names)
    elif surface_types is not None:
        emulators = {k: v for k, v in emulators.items() if k in surface_types}

    # Snapshot the candidate fleet before any prior-driven exclusion, so the
    # spectrum-only shadow pass (flag_prior_influence) sees the full fleet.
    emulators_full = dict(emulators)

    # C1 metadata->prior adapter: compose class log-priors, parameter priors
    # and hard exclusions from the active providers (season, driven by
    # known_month) plus the caller's class_priors channel. All metadata priors
    # are skipped under spectrum-only (use_priors=False).
    from biosnicar.sea_ice.metadata_priors import build_prior_set
    if use_priors:
        prior_set = build_prior_set(
            context={"month": known_month},
            sources=prior_sources,
            extra_class_priors=class_priors,
        )
    else:
        prior_set = build_prior_set(sources=())   # empty
    class_log_prior = prior_set.class_log_prior

    # Apply hard physical exclusions to the candidate fleet, but never empty
    # it — if exclusions would remove every candidate, keep the fleet (the
    # caller restricted it deliberately) and let the soft priors rank instead.
    if prior_set.excluded and len(emulators) > 1:
        kept = {k: v for k, v in emulators.items()
                if k not in prior_set.excluded}
        if kept:
            emulators = kept

    # Build the fixed_params dict that applies to all emulators
    shared_fixed = dict(fixed_params) if fixed_params else {}
    if solzen is not None:
        shared_fixed["solzen"] = float(solzen)
    if direct is not None:
        shared_fixed["direct"] = int(direct)
    # `direct` is binary and cannot be retrieved: if the caller does not fix
    # it, it would land in every trained emulator's retrieve list, fail each
    # fit with a swallowed ValueError, and leave open_water (the only model
    # without a `direct` param) as the silent winner.  Default to direct
    # beam (1) — pass direct=0 explicitly for overcast conditions.
    shared_fixed.setdefault("direct", 1)

    # Catch typo'd fixed-parameter names early: a key unknown to every
    # candidate emulator would otherwise be silently dropped and its
    # intended parameter retrieved instead.
    _known = {"solzen", "direct"}
    for emu in emulators.values():
        _known.update(emu.param_names)
    _unknown = sorted(set(shared_fixed) - _known)
    if _unknown:
        raise ValueError(
            f"fixed_params {_unknown} are not parameters of any candidate "
            f"emulator ({sorted(emulators)})."
        )

    # Season-aware physical parameter priors (from the season provider) prevent
    # emulators from fitting physically impossible temperatures — the primary
    # cause of summer bare-ice misclassification.
    season_priors: Dict[str, tuple] = dict(prior_set.param_prior)
    # Caller-supplied regularization overrides season priors on a key-by-key basis
    effective_regularization = {**season_priors, **(regularization or {})}

    # Fit each emulator
    all_fits: Dict[str, RetrievalResult] = {}
    # Per-emulator regularization actually used in fitting, kept so the
    # classification cost applies the SAME prior penalty each candidate was
    # fitted under (bare-ice types carry a translated brine-volume prior,
    # not the raw temperature prior).
    emu_regs: Dict[str, dict] = {}
    for name, emu in emulators.items():
        cfg = SEA_ICE_EMULATOR_CONFIGS.get(name, {})

        # Parameters to retrieve: all emulator params that are not in fixed
        emu_params = list(emu.param_names)
        retrieve_params = [p for p in emu_params if p not in shared_fixed]

        # Emulator-specific fixed params (those in shared_fixed that the
        # emulator actually knows about)
        emu_fixed = {k: v for k, v in shared_fixed.items()
                     if k in emu_params or k in ("solzen", "direct")}

        emu_reg = _emulator_regularization(
            emu_params, cfg, effective_regularization, season_priors,
            regularization,
        )
        emu_regs[name] = dict(emu_reg)

        try:
            fit = retrieve(
                observed=observed,
                parameters=retrieve_params,
                emulator=emu,
                platform=platform,
                observed_band_names=observed_band_names,
                obs_uncertainty=obs_uncertainty,
                bounds=bounds,
                x0=x0,
                regularization=emu_reg or None,
                wavelength_mask=wavelength_mask,
                method=method,
                model_error=model_error,
                mcmc_walkers=mcmc_walkers,
                mcmc_steps=mcmc_steps,
                mcmc_burn=mcmc_burn,
                fixed_params=emu_fixed if emu_fixed else None,
            )
            all_fits[name] = fit
        except ValueError:
            # ValueError signals a configuration/caller error (binary param
            # in the retrieve list, unknown parameter name, invalid bounds).
            # Swallowing it once turned a missing `direct` into a scene of
            # silent open_water classifications — fail loudly instead.
            raise
        except Exception as exc:  # noqa: BLE001
            # Numerical failure in one emulator (e.g. LinAlgError): skip it
            # rather than aborting the whole retrieval.
            import warnings
            warnings.warn(
                f"retrieve_sea_ice: emulator '{name}' failed — {exc}",
                RuntimeWarning, stacklevel=2,
            )

    if not all_fits:
        raise RuntimeError("All emulator fits failed.")

    # Classify: winner = lowest classification chi-squared.  Classification
    # uses per-emulator band masks (cfg["band_mask"]) — parameter fitting
    # above always used the full observation.
    cost_per_type = {}
    rms_per_type = {}
    for name, fit in all_fits.items():
        mask_name = SEA_ICE_EMULATOR_CONFIGS.get(name, {}).get("band_mask")
        try:
            # Use the SAME per-emulator regularization the fit was performed
            # under (emu_regs), not the generic season prior: bare-ice types
            # are fitted under a translated brine-volume prior and would
            # otherwise pay zero prior penalty at classification while
            # T-parameterised types pay in full — an asymmetric ranking.
            cost_per_type[name], rms_per_type[name] = _classification_cost(
                fit, observed, mask_name, platform, observed_band_names,
                obs_uncertainty, wavelength_mask,
                emu_regs.get(name, effective_regularization),
            )
        except Exception as exc:  # noqa: BLE001
            # Excluded rather than ranked on the (differently scaled)
            # fitting cost, which would corrupt the comparison.
            import warnings
            warnings.warn(
                f"retrieve_sea_ice: classification cost failed for "
                f"'{name}' — excluded from ranking ({exc})",
                RuntimeWarning, stacklevel=2,
            )
    if not cost_per_type:  # all failed — fall back to fitting costs
        cost_per_type = {n: f.cost for n, f in all_fits.items()}
        rms_per_type = {n: float("inf") for n in all_fits}
    # C1: fold class log-priors into the ranking. The classification cost is a
    # chi-squared (lower = better), and posterior ∝ exp(-cost/2)·P(class), so
    # -2·ln posterior = cost - 2·class_log_prior. Rank on that; report the raw
    # fit cost (which the quality flags relate to) unchanged.
    rank_cost = {n: c - 2.0 * class_log_prior.get(n, 0.0)
                 for n, c in cost_per_type.items()}
    ranked = sorted(rank_cost.items(), key=lambda kv: kv[1])
    winner_name = ranked[0][0]
    best_cost = cost_per_type[winner_name]
    winner_fit = all_fits[winner_name]

    # Confidence: how much better is the winner than the next candidate?
    # (on the prior-adjusted ranking cost, so it reflects the decision made)
    if len(ranked) > 1:
        best_rank, second_rank = ranked[0][1], ranked[1][1]
        confidence = float(
            (second_rank - best_rank) / second_rank
            if second_rank > 0 else 0.0
        )
    else:
        confidence = 1.0  # only one emulator ran

    # Optimal estimation: classify by posterior model probability from the
    # Laplace log-evidence (Bayesian model selection), not lowest cost.
    # `confidence` becomes the winning type's probability (0–1).
    class_probabilities: Dict[str, float] = {}
    if method == "oe":
        # NOTE (audit B6, measured 2026-07-15): evidence-based selection
        # cannot apply the per-type classification band masks (evidences are
        # only comparable when every candidate scores the SAME observation).
        # The SHEBA SWIR experiment (oe_swir_classification_experiment.py)
        # measured the cost on real melt-season spectra: default method
        # 12/12 (VIS and VIS+SWIR); OE 7/12 VIS-only, 2/12 with SWIR, with
        # failures at posterior probability ~1.00 (overconfident FYI_snow).
        # Until repaired, classify with the default method and use OE for
        # parameter posteriors — see SEA_ICE_RETRIEVAL.md §2.5 and
        # sea_ice_validation.md §12.
        # C1: combine evidence with class log-priors (uniform when none set):
        # posterior(class) ∝ P(y | class) · P(class) -> log-add the prior.
        ev = {n: (f.log_evidence + class_log_prior.get(n, 0.0))
              for n, f in all_fits.items() if f.log_evidence is not None}
        if ev:
            mx = max(ev.values())
            w = {n: float(np.exp(e - mx)) for n, e in ev.items()}
            tot = sum(w.values())
            class_probabilities = {n: w[n] / tot for n in w}
            winner_name = max(class_probabilities, key=class_probabilities.get)
            winner_fit = all_fits[winner_name]
            best_cost = cost_per_type.get(winner_name, winner_fit.cost)
            confidence = class_probabilities[winner_name]

    winner_params = dict(winner_fit.best_fit)
    # Physical snow depth is derived from the (tau_snow, grain_radius)
    # parameterisation, clipped identically to _transform_fyi_snow so the
    # reported depth matches what the forward model actually saw.
    if "tau_snow" in winner_params:
        grain = winner_params.get(
            "snow_grain_radius", shared_fixed.get("snow_grain_radius")
        )
        if grain is not None:
            winner_params["snow_depth"] = float(
                np.clip(winner_params["tau_snow"] * grain * 1e-6, 0.003, 1.5)
            )
    # Young ice is retrieved in cm (log conditioning) — derive metres.
    if "ice_thickness_cm" in winner_params:
        winner_params["ice_thickness"] = winner_params["ice_thickness_cm"] / 100.0

    from biosnicar.sea_ice.quality_flags import compute_quality_flags

    flags = compute_quality_flags(
        cost=best_cost,
        rms_residual=rms_per_type.get(winner_name, float("inf")),
        confidence=min(confidence, 1.0),
        converged=winner_fit.converged,
        parameters=dict(winner_fit.best_fit),
        bounds=emulators[winner_name].bounds,
        surface_type=winner_name,
        cost_per_type=cost_per_type,
    )

    result = SeaIceRetrievalResult(
        surface_type=winner_name,
        surface_description=_SURFACE_TYPE_DESCRIPTIONS.get(winner_name, winner_name),
        confidence=min(confidence, 1.0),
        parameters=winner_params,
        uncertainty=dict(winner_fit.uncertainty),
        predicted_albedo=winner_fit.predicted_albedo,
        observed=observed,
        cost=best_cost,
        converged=winner_fit.converged,
        flx_slr=winner_fit.flx_slr,
        cost_per_type=cost_per_type,
        all_fits=all_fits,
        quality_flags=flags,
        class_probabilities=class_probabilities,
        dfs=winner_fit.dfs,
        averaging_kernel_diag=dict(winner_fit.averaging_kernel_diag or {}),
    )

    # A3 provenance: when requested and the metadata priors are actually
    # active, re-run spectrum-only and flag whether the prior changed the
    # classification.  Skipped (prior_resolved stays None) when no metadata
    # prior was active — there is then nothing for the prior to resolve.
    if flag_prior_influence and use_priors and not prior_set.is_empty():
        shadow = retrieve_sea_ice(
            observed=observed,
            emulators=emulators_full,
            platform=platform,
            observed_band_names=observed_band_names,
            obs_uncertainty=obs_uncertainty,
            method=method,
            solzen=solzen,
            direct=direct,
            fixed_params=fixed_params,
            bounds=bounds,
            x0=x0,
            regularization=regularization,
            wavelength_mask=wavelength_mask,
            known_month=known_month,
            use_priors=False,
            flag_prior_influence=False,
            model_error=model_error,
            class_priors=class_priors,
            prior_sources=prior_sources,
            mcmc_walkers=mcmc_walkers,
            mcmc_steps=mcmc_steps,
            mcmc_burn=mcmc_burn,
        )
        result.spectrum_only_surface_type = shadow.surface_type
        result.spectrum_only_class_probabilities = dict(shadow.class_probabilities)
        result.prior_resolved = shadow.surface_type != result.surface_type

    return result


# ── Batch retrieval ──────────────────────────────────────────────────────────

# Minimum finite bands for a spectral pixel to be worth retrieving; resampled
# field/drone spectra legitimately carry NaN outside instrument coverage.
_MIN_FINITE_BANDS = 20

# Sentinel record for a pixel that was attempted but failed (distinct from
# None = masked/no-data input that was never attempted).
_FAILED_RECORD = {"__failed__": True}


def _emulator_regularization(emu_params, cfg, effective_regularization,
                             season_priors, regularization):
    """Per-emulator fitting regularization, shared by the loop and vectorized
    OE paths so they cannot diverge.

    Brine volume scales with the reference salinity, so the seasonal
    temperature prior is translated into a per-emulator Vb prior via Cox &
    Weeks at that emulator's S_ref — a single shared Vb prior is wrong for MYI
    (Vb(T=-5) at S=2 is a third of its value at S=6).
    """
    emu_reg = effective_regularization
    s_ref = cfg.get("vb_s_ref")
    if (s_ref is not None
            and "brine_volume_fraction" in emu_params
            and "sea_ice_temperature" in season_priors
            and "brine_volume_fraction" not in (regularization or {})):
        from biosnicar.sea_ice.brine_volume import compute_brine_volume

        t_mu, t_sig = season_priors["sea_ice_temperature"]
        clip_t = lambda t: float(np.clip(t, -30.0, -2.1))  # noqa: E731
        vb_mu = float(compute_brine_volume(s_ref, clip_t(t_mu)))
        vb_hi = float(compute_brine_volume(s_ref, clip_t(t_mu + t_sig)))
        vb_lo = float(compute_brine_volume(s_ref, clip_t(t_mu - t_sig)))
        emu_reg = {**effective_regularization,
                   "brine_volume_fraction":
                       (vb_mu, max((vb_hi - vb_lo) / 2.0, 1e-3))}
    return emu_reg


def _result_to_record(result: SeaIceRetrievalResult) -> dict:
    """Reduce a retrieval result to the lightweight per-pixel record kept
    in batch output (drops spectra and per-emulator fits)."""
    return {
        "surface_type": result.surface_type,
        "confidence": float(result.confidence),
        "cost": float(result.cost),
        "quality_flags": int(result.quality_flags),
        "parameters": {k: float(v) for k, v in result.parameters.items()},
        "uncertainty": {k: float(v) for k, v in result.uncertainty.items()},
        # OE / provenance outputs (None / empty outside method="oe" or
        # flag_prior_influence) — carried so batch users don't pay for
        # them and get nothing back.
        "dfs": None if result.dfs is None else float(result.dfs),
        "class_probabilities": {k: float(v) for k, v
                                in result.class_probabilities.items()},
        "prior_resolved": result.prior_resolved,
        "spectrum_only_surface_type": result.spectrum_only_surface_type,
    }


def _retrieve_chunk(chunk_obs, emulators, kwargs):
    """Worker: run retrieve_sea_ice on each pixel of a chunk."""
    import warnings

    band_mode = kwargs.get("platform") is not None
    records = []
    for obs in chunk_obs:
        finite = np.isfinite(obs)
        # Band mode needs every band; spectral mode tolerates partial
        # coverage (retrieve_sea_ice folds NaN into the wavelength mask)
        # but needs enough bands to constrain anything.
        usable = finite.all() if band_mode else finite.sum() >= _MIN_FINITE_BANDS
        if not usable:
            records.append(None)                 # no-data input, not attempted
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = retrieve_sea_ice(observed=obs, emulators=emulators, **kwargs)
            records.append(_result_to_record(result))
        except ValueError:
            # Configuration/caller error — deterministic per scene, so
            # swallowing it would silently fail every pixel.  Fail loudly.
            raise
        except Exception:  # noqa: BLE001 — one bad pixel must not kill the scene
            records.append(dict(_FAILED_RECORD))  # attempted, failed
    return records


def _vectorized_compatible(engine, kwargs):
    """(use_vectorized, reason) — is the batched OE engine usable for these
    kwargs?  Returns (False, why) so engine='vectorized' can raise the why."""
    if engine not in ("auto", "vectorized"):
        return False, "engine != auto/vectorized"
    if kwargs.get("method") != "oe":
        return False, "vectorized engine requires method='oe'"
    if kwargs.get("flag_prior_influence"):
        return False, "flag_prior_influence needs the per-pixel shadow pass"
    if kwargs.get("bounds") is not None or kwargs.get("x0") is not None:
        return False, "caller bounds/x0 overrides are not vectorised yet"
    km = kwargs.get("known_month")
    if km is not None and np.ndim(km) > 0:
        return False, "per-pixel known_month is not vectorised yet"
    return True, ""


def _run_vectorized_fleet(flat, emulators, max_iter, kwargs):
    """Build the fleet prior/exclusion context (mirroring retrieve_sea_ice's
    scalar setup) and run the vectorised OE fleet; return per-pixel records
    with None for no-data pixels and a failed marker on engine error."""
    from biosnicar.sea_ice.emulator_configs import SEA_ICE_EMULATOR_CONFIGS
    from biosnicar.sea_ice.metadata_priors import build_prior_set
    from biosnicar.sea_ice.retrieve_batch import vectorized_oe_fleet

    platform = kwargs.get("platform")
    band_mode = platform is not None
    obs_band_names = kwargs.get("observed_band_names")
    obs_unc = kwargs.get("obs_uncertainty")
    wavelength_mask = kwargs.get("wavelength_mask")
    use_priors = kwargs.get("use_priors", True)
    known_month = kwargs.get("known_month")

    # Shared fixed params (+ direct default), matching retrieve_sea_ice.
    # solzen/direct may be per-pixel arrays (length N) — e.g. from
    # illumination_context() — or scalars; keep arrays as-is (subset to the
    # valid pixels below) and coerce scalars.
    shared_fixed = dict(kwargs.get("fixed_params") or {})
    if kwargs.get("solzen") is not None:
        sz = kwargs["solzen"]
        shared_fixed["solzen"] = (np.asarray(sz, dtype=float) if np.ndim(sz)
                                  else float(sz))
    if kwargs.get("direct") is not None:
        dr = kwargs["direct"]
        shared_fixed["direct"] = (np.asarray(dr, dtype=int) if np.ndim(dr)
                                  else int(dr))
    shared_fixed.setdefault("direct", 1)

    # Priors: exclusions -> fleet; season params -> emu_regs; class -> evidence.
    if use_priors:
        prior_set = build_prior_set(
            context={"month": known_month},
            sources=kwargs.get("prior_sources"),
            extra_class_priors=kwargs.get("class_priors"),
        )
    else:
        prior_set = build_prior_set(sources=())
    if prior_set.excluded and len(emulators) > 1:
        kept = {k: v for k, v in emulators.items()
                if k not in prior_set.excluded}
        if kept:
            emulators = kept

    season_priors = dict(prior_set.param_prior)
    effective_regularization = {**season_priors, **(kwargs.get("regularization") or {})}
    emu_regs = {
        name: _emulator_regularization(
            list(emu.param_names), SEA_ICE_EMULATOR_CONFIGS.get(name, {}),
            effective_regularization, season_priors, kwargs.get("regularization"),
        )
        for name, emu in emulators.items()
    }

    # No-data mask (same rule as the loop path).
    finite = np.isfinite(flat)
    if band_mode:
        valid = finite.all(axis=1)
    else:
        sel = (np.ones(flat.shape[1], dtype=bool) if wavelength_mask is None
               else np.asarray(wavelength_mask, dtype=bool))
        valid = (finite & sel).sum(axis=1) >= _MIN_FINITE_BANDS

    records = [None] * len(flat)
    if not valid.any():
        return records

    # Subset per-pixel array fixed params (e.g. per-pixel solzen/direct) to the
    # valid pixels so they align with the observation rows passed downstream.
    sf = {}
    for k, v in shared_fixed.items():
        if isinstance(v, np.ndarray) and v.shape[:1] == (len(flat),):
            sf[k] = v[valid]
        else:
            sf[k] = v
    obs_unc_v = (obs_unc[valid] if (isinstance(obs_unc, np.ndarray)
                 and obs_unc.shape[:1] == (len(flat),)) else obs_unc)

    try:
        recs = vectorized_oe_fleet(
            flat[valid], emulators, emu_regs, sf, obs_unc_v,
            wavelength_mask, platform, obs_band_names,
            kwargs.get("model_error"), prior_set.class_log_prior,
            _SURFACE_TYPE_DESCRIPTIONS, max_iter=max_iter,
        )
    except ValueError:
        raise                          # config error — fail loudly, all pixels
    for idx, rec in zip(np.flatnonzero(valid), recs):
        records[idx] = rec
    return records


def _run_vectorized_fleet_parallel(flat, emulators, max_iter, kwargs, n_jobs):
    """Chunked-parallel wrapper around :func:`_run_vectorized_fleet` (roadmap G3+).

    The vectorised engine is single-process; in band mode its per-pixel matrices
    are tiny (a handful of bands/params), so the batched linear algebra sees no
    BLAS thread scaling and is bound to a single core. Because every pixel's OE
    is independent and the per-emulator setup (x_a / S_a / S_e, priors, regs) is
    pixel-independent, we split the scene into a few *large* contiguous chunks
    (≈ one per worker — big enough that each chunk's batched solve stays
    efficient), run :func:`_run_vectorized_fleet` on each in a separate process
    via joblib, and concatenate the per-pixel records in order.

    The result is **bit-identical** to the single-process path (no cross-pixel
    coupling to break); it just uses all ``n_jobs`` cores. joblib's default loky
    backend spawns fresh interpreters, so this is fork-safe with the torch/GDAL
    stack. Per-pixel array inputs (``solzen`` / ``direct`` / ``obs_uncertainty``
    and any per-pixel ``fixed_params``, all length N) are sliced to each chunk's
    pixel range so they stay aligned with the chunk's observation rows.
    """
    from joblib import Parallel, cpu_count, delayed

    n = len(flat)
    workers = cpu_count() if n_jobs in (-1, None) else max(1, int(n_jobs))
    n_chunks = min(workers, n)
    if n_chunks <= 1:
        return _run_vectorized_fleet(flat, emulators, max_iter, kwargs)

    bounds = np.linspace(0, n, n_chunks + 1).astype(int)
    ranges = [(int(bounds[i]), int(bounds[i + 1]))
              for i in range(n_chunks) if bounds[i + 1] > bounds[i]]

    def _chunk_kwargs(i0, i1):
        ck = dict(kwargs)
        for key in ("solzen", "direct", "obs_uncertainty"):
            v = ck.get(key)
            if isinstance(v, np.ndarray) and v.shape[:1] == (n,):
                ck[key] = v[i0:i1]
        fp = ck.get("fixed_params")
        if isinstance(fp, dict):
            ck["fixed_params"] = {
                k: (v[i0:i1] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v)
                for k, v in fp.items()
            }
        return ck

    chunk_records = Parallel(n_jobs=n_jobs)(
        delayed(_run_vectorized_fleet)(
            flat[i0:i1], emulators, max_iter, _chunk_kwargs(i0, i1)
        )
        for (i0, i1) in ranges
    )
    return [rec for chunk in chunk_records for rec in chunk]


def retrieve_sea_ice_batch(
    observed,
    n_jobs=-1,
    chunksize=500,
    spatial_coords=None,
    crs=None,
    transform=None,
    engine="auto",
    max_iter=20,
    vectorized_parallel=False,
    **kwargs,
):
    """Run :func:`retrieve_sea_ice` over a scene of pixels in parallel.

    Parameters
    ----------
    observed : array-like
        ``(N, bands)`` pixel list or ``(H, W, bands)`` image of albedo
        spectra/band values.  In band mode every band must be finite; in
        spectral mode pixels with partial coverage (NaN outside instrument
        range, e.g. resampled field spectra) are retrieved through the
        wavelength mask as long as at least 20 bands are finite.  Pixels
        below that are no-data; pixels that fail during retrieval carry the
        ``NO_RETRIEVAL`` quality flag (distinct from no-data input).
    n_jobs : int
        joblib parallelism; ``-1`` uses all CPUs.
    chunksize : int
        Pixels per parallel job.
    spatial_coords : array-like or None
        ``(N, 2)`` or ``(H, W, 2)`` per-pixel (lat, lon) — required for
        :meth:`SeaIceSceneResult.to_h3_geojson`.
    crs : str or None
        EPSG string (e.g. ``"EPSG:32633"``) for GeoTIFF export.
    transform : affine.Affine or None
        Raster affine transform for GeoTIFF export (image input only).
    engine : {"auto", "vectorized", "loop"}
        Retrieval engine.  ``"vectorized"`` solves all pixels of each fleet
        emulator at once with the batched Gauss-Newton OE engine (10-100x
        faster than per-pixel; requires ``method="oe"``).  ``"loop"`` is the
        per-pixel joblib path (any method).  ``"auto"`` (default) uses
        vectorized when compatible, else loop.  ``"vectorized"`` raises if
        the kwargs are incompatible (non-OE method, caller bounds/x0,
        per-pixel known_month, flag_prior_influence).
    max_iter : int
        Gauss-Newton iteration cap for the vectorized engine.
    vectorized_parallel : bool
        Opt-in (default ``False`` → single process, unchanged behaviour). When
        ``True`` *and* the vectorized engine is used, fan the solve out over
        ``n_jobs`` processes by splitting the scene into large contiguous pixel
        chunks and concatenating the per-pixel records. Bit-identical to the
        single-process result (pixels are independent); it just uses more cores.
        Useful on many-core hosts where band-mode matrices are too small for
        BLAS threading to help the single-process path. No effect on the loop
        engine (already per-pixel parallel) or on single-spectrum callers.
    **kwargs
        Passed through to :func:`retrieve_sea_ice` (``platform``,
        ``solzen``, ``known_month``, ``model_error``, ``class_priors``, ...).
        With the vectorised engine, ``solzen`` and ``direct`` may be per-pixel
        arrays (length N, in flattened row order) as well as scalars — e.g.
        from :func:`biosnicar.sea_ice.illumination_context.illumination_context`
        to set illumination geometry from each pixel's location, time and cloud
        state. Per-pixel arrays require ``engine="vectorized"`` (``method="oe"``).

    Returns
    -------
    SeaIceSceneResult

    Notes
    -----
    The vectorized engine (roadmap G3) reproduces the per-pixel OE result to
    numerical tolerance (``tests/test_retrieve_batch.py``) while replacing N
    Python optimiser loops with batched linear algebra, making scene-scale
    retrieval practical.  The loop path remains for non-OE methods and as the
    correctness reference.
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        raise ImportError(
            "retrieve_sea_ice_batch requires joblib.  "
            "Install the geo extras with:  pip install biosnicar[geo]"
        )

    from biosnicar.sea_ice.emulator_configs import SEA_ICE_EMULATOR_CONFIGS
    from biosnicar.sea_ice.emulator_configs import load_sea_ice_emulators
    from biosnicar.sea_ice.scene_result import SeaIceSceneResult

    observed = np.asarray(observed, dtype=float)
    if observed.ndim == 3:
        shape = observed.shape[:2]
        flat = observed.reshape(-1, observed.shape[2])
    elif observed.ndim == 2:
        shape = None
        flat = observed
    else:
        raise ValueError(
            f"observed must be (N, bands) or (H, W, bands); got {observed.shape}"
        )

    # Load the fleet once in the parent so workers don't each hit disk
    emulators = kwargs.pop("emulators", None)
    if emulators is None:
        names = kwargs.pop("surface_types", None) or list(SEA_ICE_EMULATOR_CONFIGS)
        emulators = load_sea_ice_emulators(names)
    elif kwargs.get("surface_types") is not None:
        keep = kwargs.pop("surface_types")
        emulators = {k: v for k, v in emulators.items() if k in keep}

    # ── vectorised OE engine dispatch (roadmap G3) ──
    use_vec, why = _vectorized_compatible(engine, kwargs)
    if use_vec:
        if vectorized_parallel and len(flat) > 1:
            # Opt-in: fan the (single-process) vectorised solve out over pixel
            # chunks. Each pixel's OE is independent and the per-emulator setup
            # is pixel-independent, so chunk-then-concatenate is bit-identical to
            # the single-process path — it just uses more cores. See
            # _run_vectorized_fleet_parallel.
            records = _run_vectorized_fleet_parallel(
                flat, emulators, max_iter, kwargs, n_jobs
            )
        else:
            records = _run_vectorized_fleet(flat, emulators, max_iter, kwargs)
        n_failed = sum(1 for r in records if r is not None and r.get("__failed__"))
        latlon = (np.asarray(spatial_coords, dtype=float).reshape(-1, 2)
                  if spatial_coords is not None else None)
        return SeaIceSceneResult.from_records(
            records, shape=shape, latlon=latlon, crs=crs, transform=transform,
        )
    if engine == "vectorized":
        raise ValueError(f"engine='vectorized' unavailable: {why}")

    # Per-pixel geometry arrays are only supported by the vectorised engine;
    # the per-pixel loop path takes scalar solzen/direct.
    if any(np.ndim(kwargs.get(k)) for k in ("solzen", "direct")
           if kwargs.get(k) is not None):
        raise ValueError(
            "per-pixel solzen/direct arrays require the vectorised engine "
            f"(method='oe', engine in {{'auto','vectorized'}}); {why}"
        )

    chunks = [flat[i:i + chunksize] for i in range(0, len(flat), chunksize)]
    chunk_records = Parallel(n_jobs=n_jobs)(
        delayed(_retrieve_chunk)(chunk, emulators, kwargs) for chunk in chunks
    )
    records = [r for chunk in chunk_records for r in chunk]

    # Surface systematic failures: workers run with warnings suppressed, so
    # without this a scene-wide problem is indistinguishable from cloud mask.
    n_failed = sum(1 for r in records if r is not None and r.get("__failed__"))
    if n_failed:
        import warnings
        warnings.warn(
            f"retrieve_sea_ice_batch: {n_failed}/{len(records)} pixels were "
            f"attempted but failed retrieval (NO_RETRIEVAL quality flag set).",
            RuntimeWarning, stacklevel=2,
        )

    latlon = None
    if spatial_coords is not None:
        latlon = np.asarray(spatial_coords, dtype=float).reshape(-1, 2)

    return SeaIceSceneResult.from_records(
        records, shape=shape, latlon=latlon, crs=crs, transform=transform,
    )
