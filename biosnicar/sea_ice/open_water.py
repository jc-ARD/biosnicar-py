"""Analytical open-water (ice-free ocean) albedo model.

Open water albedo is the sum of two components:

1. **Specular Fresnel reflection** at the air-water interface.  Computed
   from the spectral complex refractive index of water (Rowe 2020, 273 K)
   with surface roughness from the Cox & Munk (1954) wind-slope statistics:
   facets tilted toward a low sun see less grazing incidence, so albedo at
   high solar zenith angle decreases with wind speed.

2. **Diffuse subsurface upwelling** from volume scattering in the water
   column.  Pure-seawater irradiance reflectance R(0-) = f*b_b/(a+b_b)
   (Morel & Prieur 1977, f = 0.33) with Rayleigh-like molecular
   backscattering (Morel 1974) and absorption a = 4*pi*k/lambda.  Essentially
   zero beyond 700 nm, non-zero in the blue.

No training data is required — the model is fully analytical and exposes the
same duck-typed interface as :class:`biosnicar.emulator.Emulator`
(``param_names``, ``bounds``, ``predict``, ``flx_slr``) so it can be passed
to :func:`~biosnicar.sea_ice.retrieve.retrieve_sea_ice` alongside the
neural-network emulators.

References
----------
  Cox, C. & Munk, W. (1954). Measurement of the roughness of the sea surface
    from photographs of the sun's glitter. J. Opt. Soc. Am. 44, 838-850.
  Morel, A. (1974). Optical properties of pure water and pure sea water.
  Morel, A. & Prieur, L. (1977). Analysis of variations in ocean color.
  Rowe, P. et al. (2020). Temperature-dependent optical properties of
    liquid water from 240 to 298 K.
"""

import numpy as np

import biosnicar

_WATER_RI_PATH = (
    biosnicar.DATA_DIR / "OP_data" / "480band"
    / "refractive_index_water_273K_Rowe2020.csv"
)
_FSDS_PATH = biosnicar.DATA_DIR / "OP_data" / "480band" / "fsds.npz"

# Morel & Prieur (1977): R(0-) ≈ f * b_b / (a + b_b)
_MOREL_F = 0.33
# Morel (1974): pure seawater scattering b_w(500 nm) ≈ 0.00288 m-1, λ^-4.32,
# backscatter ratio 0.5 for molecular scattering.
_BW_500 = 0.00288
# Mean internal Fresnel reflectance of diffuse upwelling light at the
# water-air interface (Morel & Gentili 1996).
_R_INTERNAL = 0.48


def _fresnel_unpolarized(cos_theta, m):
    """Unpolarized Fresnel reflectance for complex refractive index *m*.

    cos_theta : (N,) array of incidence cosines; m : (480,) complex array.
    Returns (N, 480).
    """
    cos_i = np.asarray(cos_theta, dtype=float).reshape(-1, 1)
    sin_i2 = 1.0 - cos_i**2
    m = m.reshape(1, -1)
    cos_t = np.sqrt(1.0 - sin_i2 / m**2 + 0j)
    rs = (cos_i - m * cos_t) / (cos_i + m * cos_t)
    rp = (m * cos_i - cos_t) / (m * cos_i + cos_t)
    return 0.5 * (np.abs(rs) ** 2 + np.abs(rp) ** 2)


class OpenWaterModel:
    """Analytical open water albedo model.

    Parameters: ``solzen`` (20-80 deg) and ``wind_speed_ms`` (0-15 m/s).
    No training required — the forward model is analytical.
    """

    _PARAM_BOUNDS = {
        "solzen": (20.0, 80.0),
        "wind_speed_ms": (0.0, 15.0),
    }

    def __init__(self):
        import pandas as pd

        df = pd.read_csv(_WATER_RI_PATH)
        wvl_um = df["wvl"].to_numpy()
        n = df["n"].to_numpy()
        k = df["k"].to_numpy()
        self._wvl_um = wvl_um
        self._m = n + 1j * k

        # Subsurface irradiance reflectance R(0-), pure seawater
        a = 4.0 * np.pi * k / (wvl_um * 1e-6)            # m-1
        bb = 0.5 * _BW_500 * (wvl_um / 0.5) ** -4.32     # m-1
        r_sub = _MOREL_F * bb / (a + bb)
        # Water-leaving irradiance albedo for unit sub-surface downwelling:
        # upward transmission with internal multiple reflection.
        self._water_leaving = (
            (1.0 - _R_INTERNAL) * r_sub / (1.0 - _R_INTERNAL * r_sub)
        )

        # Fresnel LUT on a fine incidence-angle grid (used for the
        # rough-surface quadrature) — (n_chi, 480)
        self._chi_deg = np.arange(0.0, 90.0, 0.5)
        self._rf_lut = _fresnel_unpolarized(
            np.cos(np.radians(self._chi_deg)), self._m
        )

        # Diffuse-sky Fresnel albedo: 2 * ∫ R(θ) cosθ sinθ dθ
        th = np.radians(self._chi_deg)
        w = 2.0 * np.cos(th) * np.sin(th)
        self._rf_diffuse = (self._rf_lut * w.reshape(-1, 1)).sum(axis=0) / w.sum()

        # Cox & Munk slope quadrature nodes (unit-variance Gaussian grid)
        q = np.linspace(-3.5, 3.5, 21)
        gx, gy = np.meshgrid(q, q)
        self._slope_x = gx.ravel()
        self._slope_y = gy.ravel()
        self._slope_p = np.exp(-0.5 * (gx**2 + gy**2)).ravel()

        self._flx_slr = None

    # ── Emulator-compatible interface ───────────────────────────────────

    @property
    def param_names(self):
        return list(self._PARAM_BOUNDS)

    @property
    def bounds(self):
        return dict(self._PARAM_BOUNDS)

    @property
    def training_score(self):
        return None

    @property
    def flx_slr(self):
        """Representative solar flux (mid-latitude winter, clear, SZA 60)."""
        if self._flx_slr is None:
            with np.load(str(_FSDS_PATH)) as fsds:
                flx = fsds["swnb_480bnd_mlw_clr_SZA60"].copy()
            flx[flx <= 0] = 1e-30
            self._flx_slr = flx
        return self._flx_slr

    # ── Forward model ───────────────────────────────────────────────────

    def _fresnel_direct_rough(self, solzen, wind_speed_ms):
        """Slope-averaged direct-beam Fresnel reflectance, (480,)."""
        mss = 0.003 + 5.12e-3 * float(wind_speed_ms)
        sigma = np.sqrt(mss / 2.0)  # per-component slope std
        zx = self._slope_x * sigma
        # cosχ·secβ — the sqrt(1+z²) facet-area factor cancels against the
        # facet-normal normalisation, leaving (cosθ0 − zx·sinθ0).
        th0 = np.radians(float(solzen))
        proj = np.maximum(np.cos(th0) - zx * np.sin(th0), 0.0)
        w = self._slope_p * proj
        # Local incidence angle on each facet
        zy = self._slope_y * sigma
        norm = np.sqrt(1.0 + zx**2 + zy**2)
        cos_chi = np.clip((np.cos(th0) - zx * np.sin(th0)) / norm, 0.0, 1.0)
        chi_deg = np.degrees(np.arccos(cos_chi))
        # Linear interpolation into the Fresnel LUT
        idx = np.clip(chi_deg / 0.5, 0, len(self._chi_deg) - 1.001)
        i0 = idx.astype(int)
        f = (idx - i0).reshape(-1, 1)
        rf = (1.0 - f) * self._rf_lut[i0] + f * self._rf_lut[i0 + 1]
        wsum = w.sum()
        if wsum <= 0:  # sun below the slope-shadowed horizon
            return self._rf_lut[-1].copy()
        return (w.reshape(-1, 1) * rf).sum(axis=0) / wsum

    def predict(self, solzen, wind_speed_ms=3.0, **kwargs):
        """Return 480-band open water albedo spectrum, clipped to [0, 1]."""
        direct = int(kwargs.get("direct", 1))
        if direct:
            r_f = self._fresnel_direct_rough(solzen, wind_speed_ms)
        else:
            r_f = self._rf_diffuse
        albedo = r_f + (1.0 - r_f) * self._water_leaving
        return np.clip(albedo, 0.0, 1.0)

    def predict_platform(self, platform, band_names, **kwargs):
        """Band-averaged albedo for *platform*, ordered as *band_names*."""
        from biosnicar.bands import to_platform

        albedo = self.predict(**kwargs)
        band_result = to_platform(albedo, platform, flx_slr=self.flx_slr)
        return np.array([getattr(band_result, b) for b in band_names])

    def __repr__(self):
        return "OpenWaterModel(params=['solzen', 'wind_speed_ms'], analytical)"
