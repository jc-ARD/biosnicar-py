"""Derive retrieval illumination geometry from scene metadata.

For satellite retrievals the illumination inputs should come from the
acquisition, not a hardcoded default. This module turns location + time (+
optional cloud information) into the two inputs the sea-ice retrieval needs:

* ``solzen`` — solar zenith angle from latitude, longitude and UTC time;
* ``direct`` — the binary direct/diffuse illumination flag: direct beam when
  the sun is above the horizon *and* the sky is clear, diffuse otherwise.

Both are vectorised over per-pixel ``lat``/``lon`` arrays so a whole scene can
be prepared in one call and fed to
:func:`biosnicar.sea_ice.retrieve.retrieve_sea_ice_batch` (vectorised engine),
which accepts per-pixel ``solzen``/``direct`` arrays.

Solar geometry uses the standard declination + hour-angle formula (no equation
of time), good to ~1-2° — comfortably below the model's integer-degree solar
grid. Times are treated as UTC (naive datetimes are assumed UTC).
"""

from datetime import datetime, timezone

import numpy as np


def solar_zenith_angle(lat, lon, when):
    """Solar zenith angle in degrees.

    Parameters
    ----------
    lat, lon : float or array
        Latitude / longitude in degrees (east-positive longitude). Vectorised.
    when : datetime.datetime
        Acquisition time. Naive datetimes are assumed UTC; aware datetimes are
        converted to UTC.

    Returns
    -------
    float or np.ndarray
        Solar zenith angle in [0, 180] degrees (values > 90 mean the sun is
        below the horizon). Shape follows *lat*/*lon*.
    """
    if not isinstance(when, datetime):
        raise TypeError("`when` must be a datetime.datetime (UTC)")
    when = when.astimezone(timezone.utc) if when.tzinfo else when
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    doy = when.timetuple().tm_yday
    hour_utc = when.hour + when.minute / 60.0 + when.second / 3600.0
    decl = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (doy - 81))))
    ha = np.radians((hour_utc + lon / 15.0 - 12.0) * 15.0)
    latr = np.radians(lat)
    cos_z = (np.sin(latr) * np.sin(decl)
             + np.cos(latr) * np.cos(decl) * np.cos(ha))
    sza = np.degrees(np.arccos(np.clip(cos_z, -1.0, 1.0)))
    return float(sza) if sza.ndim == 0 else sza


def illumination_context(lat, lon, when, *, cloud_fraction=None, cloudy=None,
                         cloud_threshold=0.5, horizon_sza=85.0,
                         solzen_bounds=(20.0, 89.0)):
    """Retrieval ``(solzen, direct)`` from location, time and cloud state.

    ``direct`` is 1 (direct beam) only where the sun is above the horizon
    (``solar_zenith_angle < horizon_sza``) **and** the sky is clear; it is 0
    (diffuse) under cloud or when the sun is down. ``solzen`` is the computed
    solar zenith clamped to *solzen_bounds*; it is the physically meaningful
    geometry for the direct-beam pixels (for diffuse pixels the incoming is
    treated as isotropic, so the exact value matters little, but a valid clamped
    angle is still returned).

    Parameters
    ----------
    lat, lon : float or array
        Per-pixel latitude / longitude (degrees). Vectorised.
    when : datetime.datetime
        Acquisition time (UTC; see :func:`solar_zenith_angle`). One time for the
        scene; pass separate calls for tiles acquired at different times.
    cloud_fraction : float or array, optional
        Per-pixel cloud fraction in [0, 1]. Pixels with fraction above
        *cloud_threshold* are treated as diffuse.
    cloudy : bool or array, optional
        Alternative to *cloud_fraction*: a boolean cloud mask (True = cloudy →
        diffuse). Ignored if *cloud_fraction* is given.
    cloud_threshold : float
        Cloud-fraction cutoff for the direct/diffuse decision (default 0.5).
    horizon_sza : float
        Solar zenith above which the sun is treated as effectively down, so no
        direct beam is possible regardless of cloud (default 85°).
    solzen_bounds : (float, float)
        Clamp range for the returned ``solzen``. Default (20, 89) matches the
        solar-flux archive; note the emulators are trained to 80° — angles
        between 80 and 89 retrieve but emit an out-of-bounds warning.

    Returns
    -------
    (solzen, direct)
        Floats for scalar input; ``(float ndarray, int ndarray)`` for arrays,
        ready to pass to ``retrieve_sea_ice_batch(..., solzen=, direct=)``.
    """
    sza = solar_zenith_angle(lat, lon, when)
    sza_arr = np.asarray(sza, dtype=float)
    sun_up = sza_arr < horizon_sza

    if cloud_fraction is not None:
        clear = np.asarray(cloud_fraction, dtype=float) <= cloud_threshold
    elif cloudy is not None:
        clear = ~np.asarray(cloudy, dtype=bool)
    else:
        clear = True                       # no cloud info -> assume clear sky

    direct = np.asarray(sun_up & clear)
    solzen = np.clip(sza_arr, solzen_bounds[0], solzen_bounds[1])

    if sza_arr.ndim == 0:
        return float(solzen), int(bool(direct))
    return solzen.astype(float), direct.astype(np.int64)
