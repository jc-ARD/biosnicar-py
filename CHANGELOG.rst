Changelog
==========
All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
----------

Fixed
~~~~~
- **Package data now ships with the wheel/sdist** (``pyproject.toml``,
  ``MANIFEST.in``, ``biosnicar/__init__.py``).

  The ``data/`` tree (optical properties, band SRFs, pigments, pre-built
  emulators) lived as a *sibling* of the package, so ``pip install biosnicar``
  produced a distribution containing **no data files**. At runtime the loader
  resolved ``site-packages/data/emulators/*.npz`` — a path that did not exist —
  and calls such as ``load_sea_ice_emulators()`` failed with
  ``FileNotFoundError``. It only worked from editable/source checkouts (where
  ``data/`` happened to sit one level up), which is why CI (``pip install -e .``)
  never caught it.

  ``data/`` now lives inside the package at ``biosnicar/data/`` and is declared
  as package-data. ``biosnicar.DATA_DIR`` / ``PROJECT_ROOT`` resolve within the
  installed package, so source, editable, and installed-wheel layouts all work.
  Added ``tests/test_packaged_data.py`` to guard the invariant.

v0.2-sea-ice (2026-06-03)
--------------------------

Changed
~~~~~~~
- **Brine refractive index: critical physics fix** (``biosnicar/sea_ice/brine_optics.py``).

  The v0.1 implementation computed brine RI at the *bulk ice salinity* (e.g. 8 psu)
  when it should be computed at the *liquidus brine salinity* (e.g. 187 psu at −10 °C).
  This underestimated the brine-ice optical contrast by 23–47× at typical sea ice
  temperatures.  Consequences of the fix:

  - ``compute_brine_rfidx()`` now uses ``brine_salinity_at_temp(T)`` (liquidus constraint)
    internally.  The ``salinity_psu`` argument is accepted for backwards compatibility
    but is ignored — it no longer has a physical role.
  - Real part computed via the full wavelength-dependent Quan & Fry (1995) formula at
    the liquidus salinity (400–700 nm), with the salt contribution held fixed at its
    700-nm value in NIR.  Δn_re at −10 °C increases from 0.0016 to 0.037.
  - Imaginary part spurious visible correction (``_K_VIS_ALPHA``) removed — NaCl has
    no absorption above 400 nm.  UV ionic absorption term given a steeper decay
    (25 µm⁻¹) so Cl⁻ contribution is negligible above 0.4 µm.
  - Liquidus salinity capped at 250 psu (near NaCl eutectic at ~−21 °C).
  - ``brine_rfidx.npz`` LUT restructured from (S, T, 480) to (T, 480) — S dimension
    removed since brine RI is now a function of T only.
  - Both LUTs (``brine_rfidx.npz`` and ``sea_ice.npz``) rebuilt.
  - FYI_WINTER_BARE BBA: 0.555 → 0.506; MYI_WINTER_BARE: 0.514 → 0.488.  Changes
    are physically expected and consistent with SHEBA observations.
  - ``tests/test_brine_optics.py`` updated for new LUT shape and physics.

Added
~~~~~
- **Melt pond support** (``layer_type=5``).

  - ``biosnicar/optical_properties/column_OPs.py``: new ``layer_type=5`` branch
    computes liquid water optical properties analytically from Rowe et al. (2020) k
    at 0 °C.  Absorption α = 4π k/λ; near-zero scattering (Rayleigh-like term for
    numerical stability); ω → 0 in NIR, ~0.01 in visible.  Set ``rho=1000 kg/m³``.
  - ``biosnicar/sea_ice/presets.py``: two new pond presets
    ``FYI_POND_SHALLOW`` (10 cm) and ``FYI_POND_DEEP`` (40 cm).
  - ``tests/test_melt_pond.py``: 16 new tests covering physical correctness, NIR
    comparison against Morassutti (1995), and regression checks.
  - ``tests/validation_data/morassutti1995/validate_morassutti1995.py``: rewritten
    to compare the melt pond model against Morassutti (1995) observations, with
    per-depth-bin spectral and broadband comparisons and figures.
  - ``examples/13_sea_ice.py`` updated with melt pond demonstration.
  - ``docs/sea_ice.md`` and ``docs/METHODS.md`` updated with full physics documentation.

Known limitations of melt pond model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  - Clear, particle-free water assumed: VIS overestimated by ~0.2–0.4 vs real summer
    ponds (dark bottoms from algae, sediment, DOM).  NIR is well-predicted.
  - No Fresnel correction at air-water surface (~2% reflectance, acceptable for MVP).

v0.1-sea-ice-mvp (2026-06-02)
------------------------------

Added
~~~~~
- **Sea ice extension** (``biosnicar.sea_ice``): new ``layer_type=4`` for
  sea ice, wired through the full adding-doubling RT pipeline.

  - ``biosnicar/sea_ice/brine_volume.py``: Cox & Weeks (1983) brine volume
    fraction calculator, valid for −44 ≤ T ≤ −2 °C; vectorised.
  - ``biosnicar/sea_ice/brine_optics.py``: salinity- and temperature-corrected
    complex refractive index of brine, pre-computed to
    ``data/OP_data/brine_rfidx.npz`` over a (S, T, 480-band) grid.
  - ``biosnicar/sea_ice/effective_medium.py``: Maxwell-Garnett effective
    permittivity for spherical brine inclusions in ice; RI↔permittivity
    converters.
  - ``biosnicar/sea_ice/sea_ice_optics.py``: per-layer (τ, ω, g) at 480
    bands; combines effective medium absorption with air-bubble scattering
    from the existing ``bubbly_air.npz`` LUT.
  - ``scripts/build_sea_ice_lut.py``: offline pre-computation of
    ``data/OP_data/480band/luts/sea_ice.npz`` over 7 × 6 × 4 × 4 = 672
    (T, S, density, bubble_radius) combinations; runtime interpolation via
    ``RegularGridInterpolator``.
  - ``biosnicar/sea_ice/presets.py``: three built-in winter presets
    (``FYI_WINTER_BARE``, ``FYI_WINTER_SNOW``, ``MYI_WINTER_BARE``).
  - ``biosnicar/sea_ice/api.py``: high-level ``SeaIceColumn``,
    ``SeaIceLayer``, ``SnowLayer``, ``AlbedoResult`` classes; ``from_preset``
    constructor; ``compute_albedo(sza_deg, atmosphere, sky)`` entry point.
  - ``docs/sea_ice.md``: primer covering physics, API, presets, validation,
    known limitations, and references.
  - ``notebooks/sea_ice_mvp.ipynb``: end-to-end demo notebook.
  - 65 new tests across ``test_brine_volume.py``, ``test_brine_optics.py``,
    ``test_effective_medium.py``, ``test_sea_ice_optics.py``,
    ``test_sea_ice_api.py``.

- ``biosnicar/classes/ice.py``: added optional per-layer sea-ice attributes
  ``sea_ice_salinity``, ``sea_ice_temperature``, ``sea_ice_bubble_radius``
  (default ``None``; no effect on existing layer_type 0/1/2/3 paths).
- ``biosnicar/optical_properties/column_OPs.py``: added ``layer_type=4``
  branch; lazy-loaded 4-D LUT interpolator cached at module level.
- ``biosnicar/rt_solvers/adding_doubling_solver.py``: ``layer_type=4`` now
  triggers Fresnel reflection at the air–ice interface (same as
  ``layer_type=1``).
- ``biosnicar/drivers/run_model.py``: ``sea_ice_salinity``,
  ``sea_ice_temperature``, ``sea_ice_bubble_radius`` added as valid
  ``run_model`` override keys.

Known limitations
~~~~~~~~~~~~~~~~~
- Snow on sea ice treated as fresh water (salty snow deferred to v0.2).
- Brine RI corrections calibrated at ~35 psu; extrapolated to brine
  concentrations up to ~200 psu.
- Maxwell-Garnett sphere-inclusion only; valid for ν_b < 0.3.
- No sea-ice algae, melt ponds, or vertical T/S profiles in MVP.
- Validation is qualitative (broadband comparison against SHEBA summary
  statistics); formal spectral RMSE validation against digitised SHEBA
  data is planned.

Unreleased
----------

Fixed
~~~~~
- **Adding-doubling solver: incorrect albedo at SZA > ~55° for ice layers**
  (`#111 <https://github.com/jmcook1186/biosnicar-py/issues/111>`_).
  At oblique solar angles the direct beam can exceed the critical angle at
  ice's anomalous-dispersion bands (~2.93–3.09 µm), triggering total internal
  reflection (TIR) and producing a physically correct albedo of 1.0 in those
  bands. The Savitzky-Golay smoothing filter, applied as a post-processing
  step, treated these step discontinuities as noise and produced large ringing
  artefacts: values inside the TIR region were undershooting to 0.67–0.95, and
  the adjacent non-TIR transition bands were falsely elevated to 0.05–0.33.
  The fix identifies TIR bands before smoothing (``np.isclose(albedo, 1.0)``)
  and expands that mask outward by ``window_size // 2`` bands using
  ``scipy.ndimage.binary_dilation``. The raw solver output is preserved in
  this guard zone so the SG polynomial is never fitted across the
  discontinuity. The physical step at the TIR boundary—present in the Matlab
  reference (Whicker et al. 2022)—is reproduced correctly; the only
  transitions introduced at guard-zone edges are O(10⁻⁴) in magnitude and
  visually indistinguishable from the surrounding smoothed spectrum.

2.1.0 - (2023-07-20)
-------------

Changed
~~~~~~
- rename driver.py -> main.py
- remove read-the-docs in favour of custom site

Added
~~~~~~
- add wrapper func `get_albedo()` for one-line albedo prediction


