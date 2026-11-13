"""Clear skies are not a pure direct beam.

Under a cloudless sky part of the sunlight reaching the surface has been
scattered on the way down and arrives from the whole hemisphere.  That part
is large in the blue (around 40% at 400 nm with the sun 30 degrees above the
horizon) and negligible beyond about 1.5 um.  It matters wherever the
albedo depends on the angle of incidence — most of all on specular surfaces
at low sun, where the Fresnel reflectance of the direct beam is several
times that of diffuse light.

The albedo under a mixed sky is exactly

    albedo = (1 - f) * albedo_direct(solzen) + f * albedo_diffuse

per wavelength, with ``f`` the diffuse fraction of the downwelling
irradiance, because radiative transfer is linear in the incident field.
Two things follow:

* the forward model needs no new physics, only the right split of the
  incident flux (``run_model(diffuse_fraction=...)``, handled in
  :class:`biosnicar.classes.illumination.Illumination`);
* an emulator trained on pure direct and pure diffuse skies already holds
  everything needed — :class:`MixedSky` blends its two predictions, with no
  retraining and for any ``f``, including a measured one.

The model of ``f``
------------------
:func:`clear_sky_diffuse_fraction` is single scattering by molecules and a
background aerosol, with no gas absorption: Rayleigh optical depth from
Hansen & Travis (1974), aerosol optical depth ``aod_500 * (lambda/0.5)^-a``.
Half of the Rayleigh-scattered and 85% of the aerosol-scattered light
continues downward.  It is meant to be right to a few hundredths, not to
replace a radiative-transfer calculation of the sky: pass a measured or
modelled ``f`` (a number or a 480-band array) where one is available.

Light reflected by a bright surface and scattered back down adds to the
diffuse part — over snow, about a quarter more of it in the blue.  That
depends on the albedo of the surroundings, not of the target, so it is an
explicit argument, ``regional_albedo``.  ``"clear_sky"`` assumes
:data:`DEFAULT_REGIONAL_ALBEDO` (0.8, snow-covered ice), which is the common
case for sea ice and wrong for a scene that is mostly open water or ponds:
pass ``regional_albedo`` (``None`` or 0 for a dark surrounding) where that is
known.  The regional albedo and the aerosol loading are the main
uncertainties of ``f`` and are carried in the forward-model error budget
(:mod:`biosnicar.inverse.forward_uncertainty`).
"""

import numpy as np

from biosnicar import perturbation

WAVELENGTHS_UM = np.arange(0.205, 4.999, 0.01)

CLEAR_SKY = "clear_sky"

# Background Arctic aerosol (Tomasi et al. 2015): optical depth at 500 nm
# and Angstrom exponent.
DEFAULT_AOD_500 = 0.05
DEFAULT_ANGSTROM = 1.3

# Albedo of the surroundings assumed by "clear_sky": snow-covered sea ice.
DEFAULT_REGIONAL_ALBEDO = 0.8

_RAYLEIGH_DOWN = 0.5      # fraction of Rayleigh-scattered light going down
_AEROSOL_DOWN = 0.85      # same for aerosol, asymmetry parameter ~0.7
_DIFFUSIVITY = 1.66       # effective slant factor for diffuse light


def rayleigh_optical_depth(wavelengths_um):
    """Rayleigh optical depth of the atmosphere at sea level
    (Hansen & Travis 1974)."""
    lam = np.asarray(wavelengths_um, dtype=float)
    return 0.008569 * lam ** -4 * (1 + 0.0113 * lam ** -2 + 0.00013 * lam ** -4)


def clear_sky_diffuse_fraction(solzen_deg, wavelengths_um=None,
                               aod_500=DEFAULT_AOD_500,
                               angstrom=DEFAULT_ANGSTROM,
                               regional_albedo=None):
    """Diffuse fraction of downwelling irradiance under a cloudless sky.

    Parameters
    ----------
    solzen_deg : float or array (N,)
        Solar zenith angle in degrees.
    wavelengths_um : array, optional
        Wavelengths in um (default: the 480-band grid).
    aod_500, angstrom : float
        Aerosol optical depth at 500 nm and its Angstrom exponent.
    regional_albedo : float or array, optional
        Albedo of the surroundings.  When given, light reflected by the
        surface and scattered back down by the atmosphere is added to the
        diffuse part.

    Returns
    -------
    array (n_wavelengths,) for a scalar zenith angle, else (N, n_wavelengths)
    """
    lam = (WAVELENGTHS_UM if wavelengths_um is None
           else np.asarray(wavelengths_um, dtype=float))
    sz = np.asarray(solzen_deg, dtype=float)
    mu0 = np.maximum(np.cos(np.radians(sz)), 0.05)[..., None]
    tau_r = rayleigh_optical_depth(lam)
    tau_a = aod_500 * (lam / 0.5) ** -angstrom
    tau = tau_r + tau_a
    t_dir = np.exp(-tau / mu0)
    down = ((1.0 - t_dir)
            * (_RAYLEIGH_DOWN * tau_r + _AEROSOL_DOWN * tau_a) / tau)
    f = down / (down + t_dir)
    if regional_albedo is not None:
        # Spherical albedo of the atmosphere seen from below: the part of
        # upwelling diffuse light scattered back down.
        back = ((1 - _RAYLEIGH_DOWN) * tau_r + (1 - _AEROSOL_DOWN) * tau_a) / tau
        s = back * (1.0 - np.exp(-_DIFFUSIVITY * tau))
        f = 1.0 - (1.0 - f) * (1.0 - np.asarray(regional_albedo, float) * s)
    return f


def resolve_diffuse_fraction(spec, solzen_deg,
                             regional_albedo=DEFAULT_REGIONAL_ALBEDO):
    """Turn a ``diffuse_fraction`` argument into a fraction, or ``None``.

    ``None`` -> ``None`` (pure direct beam); ``"clear_sky"`` ->
    :func:`clear_sky_diffuse_fraction` over surroundings of
    *regional_albedo*; a number or 480-band array is used as given.  The
    result is clipped to [0, 1].
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec != CLEAR_SKY:
            raise ValueError(
                f"diffuse_fraction={spec!r}; expected None, 'clear_sky', a "
                "number or a 480-band array."
            )
        f = clear_sky_diffuse_fraction(solzen_deg,
                                       regional_albedo=regional_albedo)
        f = f * perturbation.get("diffuse_fraction_scale")
    else:
        f = np.asarray(spec, dtype=float)
    return np.clip(f, 0.0, 1.0)


class MixedSky:
    """An emulator (or analytical model) seen under a mixed sky.

    Wraps anything with ``predict(**params)`` that takes ``solzen`` and
    ``direct``.  Where ``direct`` is 1 the prediction is the blend of the
    direct-beam and diffuse-sky predictions by the diffuse fraction; where it
    is 0 (overcast) the model is passed through.  Every other attribute is
    the wrapped model's.
    """

    def __init__(self, model, diffuse_fraction=CLEAR_SKY,
                 regional_albedo=DEFAULT_REGIONAL_ALBEDO):
        self._base = model
        self._spec = diffuse_fraction
        self._regional = regional_albedo

    def _fraction(self, solzen):
        return resolve_diffuse_fraction(self._spec, solzen, self._regional)

    def __getattr__(self, name):
        if name.startswith("__") or name in ("_base", "_spec", "_regional"):
            raise AttributeError(name)
        return getattr(self._base, name)

    def __repr__(self):
        return f"MixedSky({self._base!r}, diffuse_fraction={self._spec!r})"

    @property
    def base(self):
        """The wrapped model."""
        return self._base

    def predict(self, **params):
        if not int(round(float(params.get("direct", 1)))):
            return self._base.predict(**params)
        f = self._fraction(params["solzen"])
        a_dir = np.asarray(self._base.predict(**{**params, "direct": 1}))
        a_dif = np.asarray(self._base.predict(**{**params, "direct": 0}))
        return (1.0 - f) * a_dir + f * a_dif


class _MixedSkyBatch(MixedSky):
    """:class:`MixedSky` for models that also have ``predict_batch``."""

    def predict_batch(self, points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        names = list(self._base.param_names)
        i_dir, i_sz = names.index("direct"), names.index("solzen")
        out = np.asarray(self._base.predict_batch(points), dtype=float)
        clear = points[:, i_dir] >= 0.5
        if not clear.any():
            return out
        pts = points[clear].copy()
        pts[:, i_dir] = 1.0
        a_dir = np.asarray(self._base.predict_batch(pts), dtype=float)
        pts[:, i_dir] = 0.0
        a_dif = np.asarray(self._base.predict_batch(pts), dtype=float)
        f = self._fraction(points[clear, i_sz])
        out[clear] = (1.0 - f) * a_dir + f * a_dif
        return out


def mixed_sky(model, diffuse_fraction=CLEAR_SKY,
              regional_albedo=DEFAULT_REGIONAL_ALBEDO):
    """Wrap *model* for a mixed sky; a no-op for ``None`` and for a model
    that is already wrapped."""
    if diffuse_fraction is None or isinstance(model, MixedSky):
        return model
    has_batch = (hasattr(model, "predict_batch")
                 and {"direct", "solzen"} <= set(model.param_names))
    return (_MixedSkyBatch if has_batch else MixedSky)(
        model, diffuse_fraction, regional_albedo)


def mixed_sky_fleet(emulators, diffuse_fraction=CLEAR_SKY,
                    regional_albedo=DEFAULT_REGIONAL_ALBEDO):
    """:func:`mixed_sky` over a ``{name: model}`` dict."""
    return {k: mixed_sky(v, diffuse_fraction, regional_albedo)
            for k, v in emulators.items()}
