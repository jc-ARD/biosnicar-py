Changelog
==========
All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

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


