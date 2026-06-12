"""Retrieve sea ice surface properties from spectral or satellite albedo.

Provides :func:`retrieve_sea_ice`, a high-level inversion function that fits
five surface-type emulators against an observed spectrum and classifies the
most likely ice surface type by residual comparison.

The five surface types are:

    FYI_bare    — winter/spring bare first-year ice
    FYI_snow    — snow-covered first-year ice
    FYI_summer  — melt-season bare ice with Surface Scattering Layer
    MYI_bare    — bare multiyear ice
    FYI_pond    — melt pond on first-year ice

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
from typing import Dict, List, Optional

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
        ``{surface_type: chi_squared}`` for all five emulators.
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
            f"SeaIceRetrievalResult",
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
                         wavelength_mask, regularization, emulator):
    """Chi-squared used to *rank* surface types (not to fit parameters).

    Applies the emulator's classification band mask, rescaling the masked
    sum of squares to the full observation band count so costs remain
    comparable across emulators with different masks.  The Gaussian prior
    penalty is included, matching the fitting cost, so seasonal priors keep
    steering classification.
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
    return cost


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
    mcmc_walkers=32,
    mcmc_steps=2000,
    mcmc_burn=500,
) -> SeaIceRetrievalResult:
    """Retrieve sea ice physical properties and classify surface type.

    Fits each of the five sea ice surface-type emulators against *observed*
    and returns the best-fit parameters together with the most likely surface
    type (the emulator that achieved the lowest chi-squared residual).

    Parameters
    ----------
    observed : array-like
        Observed albedo.  Either a 480-element spectral array or an
        N-element array of satellite band albedos (with *platform* and
        *observed_band_names* set).
    emulators : dict or None
        ``{surface_type: Emulator}`` to use.  If None, loads the five
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
        Illumination flag (1=direct, 0=diffuse).  When provided, fixed.
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

    # Young ice cannot exist in the melt season — exclude it from the
    # candidate fleet when the month is known (unless it is the only
    # candidate, in which case the caller asked for it explicitly).
    if (known_month is not None and 5 <= int(known_month) <= 9
            and "young_ice" in emulators and len(emulators) > 1):
        emulators = {k: v for k, v in emulators.items() if k != "young_ice"}

    # Build the fixed_params dict that applies to all emulators
    shared_fixed = dict(fixed_params) if fixed_params else {}
    if solzen is not None:
        shared_fixed["solzen"] = float(solzen)
    if direct is not None:
        shared_fixed["direct"] = int(direct)

    # Build season-aware physical priors from known_month
    # These prevent emulators from fitting with physically impossible temperatures,
    # which is the primary cause of summer bare ice misclassification.
    season_priors: Dict[str, tuple] = {}
    if known_month is not None:
        m = int(known_month)
        if 5 <= m <= 9:      # melt season — May through September
            season_priors["sea_ice_temperature"]   = (-4.0, 3.0)   # near-melting
            season_priors["brine_volume_fraction"] = (0.07, 0.04)  # ≈T=−5°C at S_ref=6
        elif m in (11, 12, 1, 2, 3):  # deep winter — Nov through March
            season_priors["sea_ice_temperature"]   = (-15.0, 8.0)  # well below freezing
            season_priors["brine_volume_fraction"] = (0.03, 0.015) # cold ice
        # Apr is transitional — no prior applied
        if m in (10, 11, 12, 1, 2):  # freeze-up — young ice plausible
            season_priors["ice_thickness_cm"] = (5.0, 8.0)  # thin, not grease
            season_priors.setdefault("sea_ice_temperature", (-12.0, 6.0))
    # Caller-supplied regularization overrides season priors on a key-by-key basis
    effective_regularization = {**season_priors, **(regularization or {})}

    # Fit each emulator
    all_fits: Dict[str, RetrievalResult] = {}
    for name, emu in emulators.items():
        cfg = SEA_ICE_EMULATOR_CONFIGS.get(name, {})

        # Parameters to retrieve: all emulator params that are not in fixed
        emu_params = list(emu.param_names)
        retrieve_params = [p for p in emu_params if p not in shared_fixed]

        # Emulator-specific fixed params (those in shared_fixed that the
        # emulator actually knows about)
        emu_fixed = {k: v for k, v in shared_fixed.items()
                     if k in emu_params or k in ("solzen", "direct")}

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
                regularization=effective_regularization or None,
                wavelength_mask=wavelength_mask,
                method=method,
                mcmc_walkers=mcmc_walkers,
                mcmc_steps=mcmc_steps,
                mcmc_burn=mcmc_burn,
                fixed_params=emu_fixed if emu_fixed else None,
            )
            all_fits[name] = fit
        except Exception as exc:  # noqa: BLE001
            # If one emulator fails (e.g. numerical instability), skip it
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
    for name, fit in all_fits.items():
        mask_name = SEA_ICE_EMULATOR_CONFIGS.get(name, {}).get("band_mask")
        try:
            cost_per_type[name] = _classification_cost(
                fit, observed, mask_name, platform, observed_band_names,
                obs_uncertainty, wavelength_mask, effective_regularization,
                emulators[name],
            )
        except Exception:  # noqa: BLE001 — fall back to the fitting cost
            cost_per_type[name] = fit.cost
    ranked = sorted(cost_per_type.items(), key=lambda kv: kv[1])
    winner_name, best_cost = ranked[0]
    winner_fit = all_fits[winner_name]

    # Confidence: how much better is the winner than the next candidate?
    if len(ranked) > 1:
        second_cost = ranked[1][1]
        confidence = float(
            (second_cost - best_cost) / second_cost
            if second_cost > 0 else 0.0
        )
    else:
        confidence = 1.0  # only one emulator ran

    winner_params = dict(winner_fit.best_fit)
    # Physical snow depth is derived from the (tau_snow, grain_radius)
    # parameterisation — see _transform_fyi_snow.
    if "tau_snow" in winner_params and "snow_grain_radius" in winner_params:
        winner_params["snow_depth"] = float(
            winner_params["tau_snow"]
            * winner_params["snow_grain_radius"] * 1e-6
        )
    # Young ice is retrieved in cm (log conditioning) — derive metres.
    if "ice_thickness_cm" in winner_params:
        winner_params["ice_thickness"] = winner_params["ice_thickness_cm"] / 100.0

    from biosnicar.sea_ice.quality_flags import compute_quality_flags

    flags = compute_quality_flags(
        cost=best_cost,
        confidence=min(confidence, 1.0),
        converged=winner_fit.converged,
        parameters=dict(winner_fit.best_fit),
        bounds=emulators[winner_name].bounds,
        surface_type=winner_name,
        cost_per_type=cost_per_type,
    )

    return SeaIceRetrievalResult(
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
    )


# ── Batch retrieval ──────────────────────────────────────────────────────────

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
    }


def _retrieve_chunk(chunk_obs, emulators, kwargs):
    """Worker: run retrieve_sea_ice on each pixel of a chunk."""
    import warnings

    records = []
    for obs in chunk_obs:
        if not np.all(np.isfinite(obs)):
            records.append(None)
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = retrieve_sea_ice(observed=obs, emulators=emulators, **kwargs)
            records.append(_result_to_record(result))
        except Exception:  # noqa: BLE001 — one bad pixel must not kill the scene
            records.append(None)
    return records


def retrieve_sea_ice_batch(
    observed,
    n_jobs=-1,
    chunksize=500,
    spatial_coords=None,
    crs=None,
    transform=None,
    **kwargs,
):
    """Run :func:`retrieve_sea_ice` over a scene of pixels in parallel.

    Parameters
    ----------
    observed : array-like
        ``(N, bands)`` pixel list or ``(H, W, bands)`` image of albedo
        spectra/band values.  Non-finite pixels are skipped (no-data).
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
    **kwargs
        Passed through to :func:`retrieve_sea_ice` (``platform``,
        ``solzen``, ``known_month``, ...).

    Returns
    -------
    SeaIceSceneResult

    Notes
    -----
    Parallelised per-pixel L-BFGS-B is practical up to ~100k pixels.  For
    regional mosaics, the retrieval engine will be replaced by a vectorised
    inverse network behind the same :class:`SeaIceSceneResult` interface.
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

    chunks = [flat[i:i + chunksize] for i in range(0, len(flat), chunksize)]
    chunk_records = Parallel(n_jobs=n_jobs)(
        delayed(_retrieve_chunk)(chunk, emulators, kwargs) for chunk in chunks
    )
    records = [r for chunk in chunk_records for r in chunk]

    latlon = None
    if spatial_coords is not None:
        latlon = np.asarray(spatial_coords, dtype=float).reshape(-1, 2)

    return SeaIceSceneResult.from_records(
        records, shape=shape, latlon=latlon, crs=crs, transform=transform,
    )
