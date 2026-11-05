Changelog
==========
All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
-----------

Changed
~~~~~~~
- **Sea-ice gas (air) content now follows Cox & Weeks (1983).**
  ``compute_brine_volume`` takes a new ``porosity`` *or* ``bulk_density_kg_m3``
  argument (mutually exclusive, as in SMRT), and a new ``compute_gas_volume``
  supplies the gas half of the same density balance.  ``compute_sea_ice_optics``
  passes the layer's bulk density into both.

  Previously brine volume was computed bubble-free and the air fraction was
  then derived from that same density through a *separate* balance using a
  Frankenstein & Garner-derived brine density — two phase models describing
  the same air, only one of which the brine volume had assumed.  Brine volume
  was overestimated by the gas fraction, 1–9% across the shipped presets.

  Defaults are unchanged: omitting both arguments reproduces the previous
  bubble-free result exactly.  Verified against SMRT v1.7.3 — the porosity
  path to 0.033%, the bulk-density path exactly.

- **Sea-ice absorption now accounts for air content.**  ``abs_coeff`` was
  ``4π·n_eff_im/λ`` — the absorption of the ice+brine phase, applied across the
  whole layer — so a layer that is half air absorbed as though it were solid
  ice.  Absorption per metre measured ~82 m⁻¹ from 920 down to 500 kg/m³,
  independent of density.  It is now scaled by ``(1 - ν_air)``, bringing sea
  ice into line with every other layer type, which already excludes air by
  working in mass absorption coefficients.

  Relatedly, the Maxwell-Garnett mix now receives the brine fraction *of the
  condensed phase* (``ν_b/(1-ν_air)``) rather than of the whole layer, which is
  what a two-phase ice+brine mix requires.

  Air is deliberately **not** mixed into the effective medium: Maxwell-Garnett
  assumes inclusions small compared to the wavelength, and the bubbles are
  100–1000 µm against 0.2–5 µm light.  That is why they are treated as discrete
  Mie scatterers.  For large non-absorbing inclusions, path-length volume
  weighting is the correct limit.

- **Sea-ice LUT density axis extended to 700–920 kg/m³** (was 870–920), and the
  LUT rebuilt.  ``_interpolate_sea_ice_lut`` *clamps* rather than extrapolates,
  so every density below 870 was silently replaced by 870: the shipped presets'
  drained-layer density of 850 (Jin et al. 2023) was wrong, and the FYI_bare
  emulator's sampled ``rho_DL`` range of (820, 900) was inert over roughly 60%
  of its span.  This was an independent pre-existing bug, found while wiring
  density through for the porosity fix, and it is the larger of the two
  effects: max ΔBBA +0.026, against +0.009 for the porosity term and +0.006 for
  the absorption weighting; +0.039 combined (+6.5% on FYI_WINTER_BARE, +10% on
  the pond presets).

  The floor of 700 is set from the literature range rather than from what the
  code happened to request.  Timco & Frederking (1996) give sea-ice densities of
  720–940 kg/m³, with multi-year ice above the waterline at **720–910** — so the
  old 870 floor silently excluded most of the MYI range, and the MYI preset's
  own ``rho=[850, 870]`` sat on the grid edge.  720 kg/m³ is 21.5% air, still
  dispersed gas in a continuous ice matrix; the bubbly-ice topology holds until
  air percolates near 30–35% air (~600–640 kg/m³).  Ice below that is over 30%
  air and belongs in ``layer_type=0``.

  The absorption correction above is what makes this floor usable: without it,
  absorption was independent of porosity — a 27% overestimate at 720 kg/m³ — so
  a lower floor would have produced confident nonsense.

- **All six sea-ice emulators retrained** against the corrected forward model.
  ``invert_brine_volume`` now accepts ``bulk_density_kg_m3`` and the emulator
  transforms pass the layer density, so ``brine_volume_fraction`` still names
  the brine volume the forward model uses instead of becoming degenerate with
  ``rho_DL``.  The ``brine_volume_fraction`` training ranges were corrected to
  what is reachable in ice that holds air: FYI_bare (0.019, 0.123) from
  (0.019, 0.141); MYI_bare (0.0065, 0.042) from (0.0065, 0.045).

  Emulator accuracy is unchanged by the rebuild.  On a freshly generated
  2000-spectrum holdout (seed 777, disjoint from the training seed 42),
  FYI_bare scores pooled spectral R² **0.9977** and BBA MAE **0.0024**, against
  0.9978 / 0.0021 for the pre-rebuild emulator on the equivalent holdout — the
  same to within the noise of one draw.  The worst-case tail is slightly
  better: 5th-percentile per-spectrum R² 0.9842 → 0.9881, because the density
  axis no longer clamps and the forward model is smoother in ``rho_DL``.  All
  six emulators sit at 0.998–1.000 held-out spectral R².

  **Retrieval, classification and field-validation numbers in ROADMAP.md
  predate this change and still need regenerating.**

- **``rho_DL`` and ``brine_volume_fraction`` are now substantially degenerate**
  — a real consequence of the LUT density fix, not a regression.  The Jacobian
  collinearity between the two rose from mean ``|cos|`` 0.42 (min 0.03) to **0.73**
  (min 0.43, max 0.90).  This is expected: with the grid clamping at 870 kg/m³,
  most of the sampled ``rho_DL`` range was inert, so its Jacobian was close to
  numerical noise and uncorrelated with anything.  Now that density genuinely
  acts — through ν_air scattering and the (1 − ν_air) absorption weighting — it
  pushes the spectrum in much the same direction brine volume does.

  Forward accuracy is unaffected.  *Retrieval* is: joint retrievals of the two
  constrain a combination rather than two independent parameters, and should
  not be reported as the latter.  Reparameterising onto the along-ridge and
  across-ridge directions is logged in ``ROADMAP.md`` and needs a retrain.

  On what these corrections buy: fitting six SHEBA bare-ice spectra with the
  poorly-constrained parameters free, the corrected and original forward models
  fit **equally well** (mean RMSE 0.0107 both) once the three-layer SSL
  structure is used.  At the 850–920 kg/m³ the sea-ice layers actually want,
  air is only 0.6–8%.  What does change is parameter behaviour: retrieved
  drained-layer densities are 899/900/891/900/900/895 corrected, against
  785/828/890/900/897/900 before — same fit, tighter parameters.  These
  corrections are landed for correctness and parameter interpretation, not for
  fit quality, and should not be advertised as improving accuracy.

Verified
~~~~~~~~
- **Brine volume fraction cross-checked against SMRT v1.7.3.**
  ``compute_brine_volume`` agrees with SMRT's independent implementation of
  Cox & Weeks (1983) to **0.0326–0.0331%** over T ∈ [−3, −30] °C and
  S ∈ [1, 15] psu, the entire residual being the pure-ice density constant
  (0.917 vs 0.9167 g cm⁻³).  SMRT is a microwave model and shares no radiative
  observable with BioSNICAR, but brine volume is frequency-independent, which
  makes this a genuine external check on the sea-ice geophysics.
  ``tests/test_brine_volume.py`` now pins the implementation against
  hard-coded SMRT values (no SMRT dependency added).

Fixed
~~~~~
- **Removed an incorrect docstring claim** that ``compute_brine_volume`` ran
  "~25–35% higher" than Cox & Weeks Table 1 "due to unit/convention ambiguity
  in the original paper".  No such discrepancy exists: the published values are
  reproduced to within 3%, and SMRT departs from them by the same margin, so
  the residual is rounding in the published table.  The accompanying
  reference-value test tolerance, set to 40% on the strength of that claim, is
  tightened to 5%.
- **``Emulator.training_score`` was documented as "R² score on held-out
  validation set".**  It is neither: it is sklearn's multioutput R² over the
  retained **PCA coefficients**, scored on the training samples themselves.  It
  averages uniformly across components, so near-zero-variance tail components
  count as much as the first and drag the mean down independently of spectral
  accuracy — which is why FYI_bare reports 0.70 while its held-out spectral R²
  is 0.998.  The property docstring, the build-time comment and
  ``docs/EMULATOR.md`` now say what it measures, and record the measured
  FYI_bare example.

  The docstring also warns about the sampling trap that goes with it: audit
  samples must be drawn the way ``build`` draws them, Latin hypercube with
  ``_LOG_SAMPLE_PARAMS`` in log10(x+1) space.  Sampling those uniformly in
  linear space over-weights the sparse high end and understates FYI_bare at
  R² ≈ 0.80 / RMSE 0.063 — an artefact of the measurement, not the emulator.

- **The clamp below −30 °C is no longer silent.**  ``compute_brine_volume``
  accepts temperatures to −44 °C but holds the result at its −30 °C value below
  the Cox & Weeks fitted range; it now emits a ``RuntimeWarning`` saying so, and
  the docstring records that these are not Cox & Weeks values.  Unclamped, they
  would diverge from SMRT by up to ~35% at −38 °C.  Configured retrieval ranges
  stop at −30 °C, so no shipped code path reaches the clamp.

Added
~~~~~
- **``scripts/audit_sea_ice_emulators.py``** — held-out spectral audit for all
  six sea-ice emulators in one run.  It replicates the build's Latin hypercube
  and log-space sampling (getting this wrong is what makes an excellent
  emulator look broken), scores reconstructed spectra against the forward
  model, and reports the PCA reconstruction ceiling alongside the network
  score, so compression error can be told apart from network error.

- **Chunked-parallel vectorised OE engine** (``vectorized_parallel=`` on
  ``retrieve_sea_ice_batch``).

  The vectorised engine is single-process, and in band mode its per-pixel
  matrices are small enough that BLAS threading gains little, so on a many-core
  host most cores sit idle.  Passing ``vectorized_parallel=True`` splits the
  scene into large contiguous pixel chunks — roughly one per worker — solves
  each in a separate process via joblib/loky, and concatenates the per-pixel
  records in order.

  Opt-in: the default ``False`` preserves single-process behaviour, so existing
  callers and all non-batch paths are unchanged.  Every pixel's OE is
  independent, so the result is **bit-identical** to the single-process path;
  two tests assert this for scalar and for per-pixel ``solzen``/``direct``
  geometry.  Measured 3.1× on a 1500-pixel scene with 8 workers on a 10-core
  box.

  Workers are shipped emulator *names* and load the fleet once into a
  per-process cache, reused across chunks and across successive calls while the
  loky pool stays warm, which removes per-chunk marshalling of the fleet.  A
  custom fleet that ``load_sea_ice_emulators`` cannot reconstruct falls back to
  pickling the passed objects, so correctness holds for any fleet.

  Note: do **not** pin ``OMP_NUM_THREADS=1`` around this call.  On
  Linux/OpenBLAS the batched chunk solve genuinely threads across cores and
  pinning roughly halves throughput; it appears free only on macOS, whose
  Accelerate backend ignores the variable.

Changed
~~~~~~~
- **``miepython`` is no longer a core dependency** (`#116
  <https://github.com/jmcook1186/biosnicar-py/issues/116>`_).

  It is only needed to *generate* single scattering property files (the
  bio-optical model and ``ssps_spheres_generator``); the forward model reads the
  precomputed files in ``data/``.  ``mie_coated_water_spheres.py`` imported it
  but never used it, which dragged it — and therefore numba and llvmlite — onto
  the core import path.  That dead import is removed and the two genuine call
  sites now import lazily via ``biosnicar.utils.optional_deps.import_miepython``,
  raising an ImportError with install instructions if it is missing.

  ``miepython`` moved to a new ``mie`` extra (``pip install biosnicar[mie]``).
  This unblocks installation on platforms with no numba/llvmlite wheels, notably
  Intel (x86_64) macOS, where llvmlite ≥ 0.46 and numba ≥ 0.63 are arm64-only.

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


