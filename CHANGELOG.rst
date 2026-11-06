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

- **Field validation re-measured after the physics work.**  Snow and bare ice
  are unchanged; melt ponds regressed, were diagnosed, and are fixed below.

  Unchanged — SHEBA (Grenfell & Light 2007)::

      spring snow       6/7 full-spectrum,  7/7 S2,  7/7 L8
      summer bare ice  16/16 full-spectrum, 16/16 S2, 15/16 L8

  Regressed — Morassutti (1995) melt ponds::

      classification (504 records)   77%  ->  62%  (-> 74.6% after the fix)
      classification (120-record block)   ->  51%  (-> 63%)
      depth within +/-25%            ~22% ->  15%  (-> 23.8%)

  Three explanations ruled out.  It is **not** model flexibility from
  MYI_bare's new density parameter: the OE path, whose Laplace evidence
  carries an Occam factor, scores 62.5% against the default's 61.9%.  It is
  **not** near-ties: FYI_pond's chi-squared runs ~284% above the winning
  type.  And it is **not** bound-railing: only 5 of 109 misses railed on
  ``pond_depth`` or ``black_carbon``.  FYI_pond simply cannot fit those
  spectra.

- **Melt-pond regression diagnosed and fixed: the pond floor's bubble radius
  was hard-coded at 200 um.**  Not a calibration drift, which is where I
  looked first.

  Fixed at 200 um — a winter-ice value — the emulator bottomed out at
  BBA 0.144, while **21% of the Morassutti observations are darker than that
  at any parameter setting**.  Those spectra went to MYI_bare and FYI_bare,
  which reach down to ~0.11.  The other two knobs could only reach the dark
  end through physically indefensible settings: black carbon at
  12000-25000 ppb against a ~2000 ng/g literature maximum (and 1000 ppb is
  already an effective-LAP proxy), or a floor density of 916 kg/m3, which is
  bubble-free pure ice.  Coarseness was the remaining candidate and the
  physically right one — melt-season pond floors are deteriorated ice with
  large voids, and over 200-2000 um the floor's BBA moves 0.178 -> 0.066, a
  factor of 2.7.  Fixing it at any single value was a strong assumption.

  ``sea_ice_bubble_radius`` is now retrieved for FYI_pond over (200, 2000),
  making it a 6-parameter emulator written to
  ``sea_ice_FYI_pond_6param.npz``::

      reach (darkest BBA)      0.144  ->  0.067
      unreachable observations  107/504 -> 16/504
      classification (504)       62%   ->  74.6%   (77% before the physics work)
      depth within +/-25%        15%   ->  23.8%   (~22% before)

  Depth retrieval now exceeds the pre-existing baseline.  Snow and bare ice
  are untouched — SHEBA spring snow 6/7, summer bare ice 16/16, S2 7/7 and
  16/16, L8 7/7 and 15/16, all identical.

  Note for anyone chasing this later: the BC = 1000 ppb / rho = 895 figures in
  ``docs/sea_ice.md`` are for the *preset*, not this emulator, which samples
  BC freely and uses rho = 850.  That preset sweep is still stale.

- **A fourth frozen parameter region, this time in the transforms.**
  ``_transform_fyi_bare`` and ``_transform_fyi_summer`` set the interior
  layer's bubble radius as ``min(1000.0, bbl * 2.0)``.  That ``1000.0`` is the
  *old* LUT ceiling hard-coded into the transform, so for ``bbl >= 500`` the
  interior layer's radius froze at 1000 um regardless of what was sampled —
  over roughly 23% of FYI_bare's sampled log range, in the 1.45 m layer that
  dominates optical depth.  Both now cap at 2000, the current LUT ceiling.
  (MYI_bare's ``min(2000.0, bbl * 1.5)`` was already at the real ceiling and
  is unchanged.)

  Same pattern a fourth time — density below 870, bubble radius below 100,
  bubble radius above 1000, and now a transform cap left behind at a limit
  that had since been raised.  Each silently froze a parameter the config
  claimed to be sampling.

- **FYI_summer's bubble-radius ceiling raised from 500 to 1000 um.**  On the
  23 SHEBA spectra the old ceiling was pinned by 9 retrievals at a mean
  averaging kernel of 0.87 — the measurement pushing hard against it, not
  drifting there — and a forward-model scan put the optimum beyond 500 in half
  the pinned cases.  Matched to FYI_bare now the LUT reaches 2000 um.

  FYI_bare and FYI_summer rebuilt.  FYI_bare improves on every measure:
  held-out spectral R2 0.9983 -> **0.9987**, RMSE 0.0089 -> 0.0078, vis-only
  0.0056 -> 0.0047, and its PCA-coefficient training score 0.697 -> 0.845 —
  removing the frozen interior layer makes the function markedly easier to
  fit.  FYI_summer is marginally worse (vis-only 0.0024 -> 0.0033), which is
  expected: its radius range doubled at the same sample count.

- **All six emulators retrained** on the corrected LUT (21 min).  Held-out
  spectral accuracy is unchanged to slightly better::

      emulator      R2 before -> after    vis-only RMSE before -> after
      FYI_bare       0.9981     0.9983        0.0050      0.0056
      MYI_bare       0.9977     0.9979        0.0069      0.0054
      FYI_summer     1.0000     1.0000        0.0031      0.0024
      FYI_snow       0.9999     0.9999        0.0030      0.0035
      FYI_pond       0.9999     0.9999        0.0017      0.0017
      young_ice      0.9990     0.9990        0.0024      0.0024

  MYI_bare *improved* despite gaining a sixth parameter at the same 30k
  sample count, so no larger rebuild was needed: removing the inert radius
  region makes the function smoother to learn, which more than offsets the
  extra dimension.

- **The radius/density degeneracy got worse, and that is the correct
  outcome.**  Posterior correlation for ``sea_ice_bubble_radius``/``rho_DL``
  went from 0.837 to **0.972** for FYI_bare, and MYI_bare — where density was
  previously not retrieved at all — comes in at **0.982**.  Marginal sigma is
  inflated **7-8x**, up to 20x at worst, with the same sign at all 24 base
  points.

  Same lesson as the density axis: a clamped parameter has a Jacobian that is
  mostly numerical noise, and noise is uncorrelated with everything.  With
  both axes genuinely active the two parameters are what the code says they
  are — two knobs on the single product ``sca_cff_vlm(r) * nu_air`` — and are
  very nearly interchangeable.  The fix did not create the degeneracy, it
  stopped hiding it.

  **At |r| = 0.97-0.98, reporting bubble radius and density as separately
  determined quantities is not defensible.**  Report ``RetrievalResult.ssa``.
  On a live MYI retrieval with truth r = 800 um, rho = 780 kg/m3
  (SSA = 0.7183): radius comes back 589 +/- 327 um (56% uncertainty, 26% from
  truth) and density 811 +/- 55, while **SSA comes back 0.7220 +/- 0.031 —
  4.3% uncertainty and 0.5% from truth**.  SSA is 13x better determined than
  the radius and accurate where the components are not.

- **Sea-ice LUT bubble-radius axis extended to 50-2000 um** (was
  [100, 300, 500, 1000]), and the LUT rebuilt — 11 log-spaced points,
  3696 grid points, ~30 s.

  Same defect as the density axis, on a different axis and found later.
  ``_interpolate_sea_ice_lut`` clamps rather than extrapolating, and the
  emulators sample 50-1000 (FYI_bare), 50-500 (FYI_summer) and 200-2000
  (MYI_bare).  So ``tau`` and ``ssa`` were **bit-identical** for every radius
  below 100 and above 1000: the bottom of FYI_bare's range and roughly 30% of
  MYI_bare's log range were inert, exactly like ``rho_DL`` below 870 before.

  It was not merely flat, it was wrong: at r = 50 um the clamped ``tau`` is
  1153 against a correct 2238 — a factor of 2.  Verified bit-identical at the
  three shared grid radii (100, 300, 1000), so the rebuild adds resolution
  without changing physics.  ``tests/test_sea_ice_optics.py`` now guards both
  edges across the whole fleet, so a new or widened emulator cannot reopen it.

- **MYI_bare retrieves drained-layer density.**  ``_transform_myi_bare``
  hard-coded ``rho=[850, 870]``, so the parameter that most distinguishes
  multi-year ice — its low density, from drained brine and trapped gas —
  could not be retrieved at all.  ``rho_DL`` is now sampled over
  **(720, 900) kg/m3**, the Timco & Frederking (1996) range for MYI above the
  waterline, which the extended LUT density axis (700-920) now covers.

  ``brine_volume_fraction`` bounds tighten from (0.0065, 0.042) to
  (0.0063, 0.0346): the old range was computed at the pinned density of 850,
  and the intersection over the sampled range is what keeps every
  (Vb, rho_DL) pair invertible.  The emulator becomes 6-parameter and is
  written to ``sea_ice_MYI_bare_6param.npz``.

  This also makes ``RetrievalResult.ssa`` available for MYI retrievals, which
  need a retrieved radius *and* density to form it.

- **``run_model`` accepts the optical parameterisation as an alternative to
  the physical one.**  ``ssa=`` in place of ``rds``/``sea_ice_bubble_radius``,
  and ``brine_volume_fraction=`` in place of ``sea_ice_temperature``.  Either
  set is accepted per layer; supplying both for the same layer raises rather
  than applying a silent precedence rule, matching the mutually-exclusive
  style of ``compute_brine_volume(porosity=... | bulk_density_kg_m3=...)``.

  The rationale is the degeneracy work above: radius and density are
  individually degenerate and the spectrum constrains their combination,
  while temperature acts on the optics only through brine volume.  Both
  parameterisations give identical spectra (verified to 1e-7 across the 480
  bands; the residual is the brine-volume root-find tolerance, equivalent to
  1.2e-5 degC).

  **SSA has two conventions and the layer type selects between them.**
  Granular layers (``layer_type=0``, snow and the SSL) use the standard
  snow-science form ``3/(r*rho_ice)``, independent of bulk density.  Bubbly
  layers (1, 2, 4) use ``3*(1 - rho/917)/(r*rho)``.  They differ by ~2x at a
  snow density of 300 kg/m3, and the absolute values differ by two orders of
  magnitude between snow (~10 m2/kg) and dense sea ice (~0.5 m2/kg), so the
  conversion refuses to guess the topology.  ``layer_type`` 3, 5 and 6 have
  no single inclusion radius and raise.

  New module ``biosnicar.sea_ice.parameterisation`` exports
  ``specific_surface_area``, ``radius_from_ssa``, ``topology_for_layer_type``
  and ``ssa_density_sensitivity`` (the ``1/phi`` = 9-30x error amplification
  at sea-ice densities), replacing the formula previously duplicated across
  three call sites.

Fixed
~~~~~
- **``test_vectorized_parallel_matches_single`` asserted bit-identity that is
  not available.**  It compared float32 scene fields at ``atol=1e-9``, finer
  than float32 can represent — the ULP at a cost of 25.9 is 1.9e-6.  It passed
  by luck until the 2026-11-06 retrain nudged one pixel's cost onto the
  neighbouring ULP (measured difference: exactly 1.0 ULP, and zero on a
  different scene, i.e. data-dependent rounding rather than a logic
  divergence).

  Chunking cannot change the mathematics — pixels are independent — but it
  changes the shapes handed to the batched linear algebra, and a different
  BLAS kernel accumulates in a different order.  Measured divergence splits
  along a physically meaningful line::

      point estimates (param_*, cost)   <= 1.5e-6 relative   float32 ULP
      covariance-derived (unc_*, dfs)   <= 1.0e-4 relative   ~100x worse

  Both of the latter come through ``inv(H)``, and ``H`` is ill-conditioned
  because of the parameter degeneracies — the bubble-radius/density posterior
  correlation of 0.97-0.98 gives a condition number of order ``(1+r)/(1-r)``.
  So the retrieval is stable and only its error bars amplify round-off.
  Tolerances are now set from those measurements with headroom, and surface
  type — the discrete field a real divergence would break — is still required
  to match exactly.

- **``LUT.get`` truncated the radius instead of rounding it.**
  ``idx = self._radius_to_idx[int(radius)]`` turns a radius of
  799.9999999999999 — which is what an SSA conversion or any other float
  arithmetic can produce for 800 — into the **799 um** entry, silently
  selecting different optical properties.  Found because the two
  parameterisations above disagreed by 2.2e-4 in albedo when they should have
  been identical; the temperature path accounted for only 3e-9 of it.  Now
  ``int(round(radius))``, which fixes float noise while still raising
  KeyError for genuinely off-grid radii.  The hexagonal-column lookup
  (``side``/``length``) had the same issue and is fixed the same way.

- **``RetrievalResult.ssa`` now works for sea-ice retrievals, and its
  uncertainty uses the posterior covariance.**  Two bugs, one policy.

  The ``ssa`` property keyed only off ``rds``/``rho``, the glacier-ice
  parameter names.  The sea-ice emulators call the same physical quantities
  ``sea_ice_bubble_radius``/``rho_DL``, so ``ssa`` silently returned ``None``
  for every sea-ice retrieval — even though ``_compute_ssa``'s formula is
  derived for bubbly ice and its docstring says so.

  ``_ssa_uncertainty`` assumed radius and density were independent.  They are
  strongly anti-correlated along the scattering ridge (posterior r = -0.85
  mean, -0.99 at worst), and both partials of SSA are negative, so the
  cross-term subtracts.  Including it shrinks sigma_SSA **3x**.  The
  covariance is rebuilt from the correlation times the linear-space sigmas,
  because ``posterior_covariance`` is stored in optimiser space where
  ``sea_ice_bubble_radius`` is log-sampled — mixing the two spaces reduces the
  cross-term to nothing.

  The policy behind it: radius and density are individually degenerate, their
  combination is not.  This follows from the code — scattering is
  ``sca_cff_vlm(r) * nu_air`` and absorption ``(1 - nu_air) * alpha_ice``, so
  in the single-scattering albedo the density cancels and omega is a function
  of SSA alone.  Measured along true iso-SSA curves, holding SSA fixed removes
  **88% of NIR and 61% of VIS** spectral variation; in the posterior,
  sigma_SSA/SSA beats sigma_r/r in **22 of 24** states (0.44 vs 0.58 mean).

  Four limits, recorded so the policy is not over-applied: (1) it is weaker in
  the VIS, where omega -> 1 and the asymmetry parameter — a function of radius
  alone — carries the signal, which is why the pair correlates at 0.85 not
  0.999 and why SSA is reported *alongside* the components rather than
  replacing them; (2) SSA amplifies density error by 1/phi = 9-30x at sea-ice
  densities, since the air fraction is a small difference of large numbers;
  (3) only FYI_bare retrieves both components — the other five hard-code
  density, so SSA is computable for 0 of the 23 SHEBA spectra, and making it
  primary fleet-wide needs a retrain of each; (4) it addresses one of the six
  measured degeneracies and none of the others.

- **``sea_ice_temperature`` is reported but not retrieved.**  Across 22 of the
  23 SHEBA spectra its averaging-kernel diagonal averages **0.048** — roughly
  95% prior, 5% measurement.  The fleet sweep agrees (mean 0.16 for
  FYI_summer, 0.25 for FYI_snow, 0.29 for young_ice).

  Expected on reflection: temperature acts only through brine volume and brine
  salinity, and the bare-ice and SSL emulators are ``vis_only``-masked, where
  bubble scattering swamps brine absorption.  There is no temperature signal
  in the VIS.  It is currently returned with an uncertainty that implies more
  than it should.  Logged in ``ROADMAP.md`` as the highest-value open item
  from this work; no behaviour change here beyond documenting it and making
  the averaging kernel easy to check.

- **Every emulator carries a degeneracy, not only FYI_bare.**  Worst pair per
  emulator, mean posterior ``|r|``: FYI_bare 0.89
  (``sea_ice_bubble_radius``/``rho_DL``), MYI_bare 0.85
  (``sea_ice_bubble_radius``/``black_carbon``), young_ice 0.85
  (``ice_thickness_cm``/``ocean_albedo``), FYI_pond 0.79
  (``sea_ice_temperature``/``black_carbon``), FYI_summer 0.76
  (``sea_ice_bubble_radius``/``black_carbon``), FYI_snow 0.49.

  Three keep a stable sign across all base points — FYI_bare, FYI_summer and
  young_ice — so a reparameterisation can help there; each needs its own
  retrain.  FYI_snow is the evidence the technique works: ``tau_snow`` is
  itself a reparameterisation (depth / grain radius), and the pair it was
  introduced to fix is now the *least* degenerate in that emulator (0.24).

- **24% of parameters land on a bound in real SHEBA retrievals** (22 of 92),
  with 19 of 23 spectra carrying a pair above ``|r|`` = 0.8.  Pairing the pin with
  the averaging kernel separates a binding bound (high A — the measurement is
  pushing against it) from a flat likelihood (low A — the value drifted to the
  wall and means nothing).  A forward-model scan past ``FYI_summer``'s 500 um
  bubble-radius ceiling — half what the sea-ice LUT supports and half what
  FYI_bare allows — finds the optimum beyond 500 in only 4 of 8 pinned cases,
  with marginal gain, so that ceiling is a tidiness issue rather than an
  accuracy bug.

- **The dominant FYI_bare degeneracy is ``sea_ice_bubble_radius``/``rho_DL``,
  not the pair that was being tracked.**  Measuring all six parameter pairs
  instead of one gives posterior correlation **0.84 mean, 0.995 max**, the same
  sign at all 24 base points, inflating the marginal 1-sigma by **4.2x typical
  and 9.8x worst** — against 0.31 / 1.10x for
  ``rho_DL``/``brine_volume_fraction``.
  It had never been measured because the audit only ever looked at the latter.

  The mechanism is in ``sea_ice_optics.py``: layer scattering is formed as
  ``sca_cff_vlm(bubble_radius) * nu_air`` and ``nu_air`` is set by density, so
  the two parameters are knobs on one product.  They are not perfectly
  interchangeable — the asymmetry parameter depends on radius but not air
  fraction, absorption on air fraction but not radius — which is why the
  correlation is 0.84 and not 0.999 (temperature is also tied to (Vb,
  ``rho_DL``) by the parameterisation).

  Unlike the ``brine_volume_fraction`` pair, this one's sign is stable, so
  reparameterising onto the combination the measurement constrains is viable,
  with the granular-ice ``rds``/``rho`` -> SSA treatment as precedent.  It
  needs a retrain and is logged in ``ROADMAP.md`` rather than done here.

- **``rho_DL``/``brine_volume_fraction`` collinearity rose with the LUT
  density fix — measured, and smaller than the first diagnostic suggested.**
  The Jacobian cosine went from mean 0.42 to 0.73, which is expected rather
  than a regression: with the grid clamping at 870 kg/m³ most of the sampled
  ``rho_DL`` range was inert, so its Jacobian was close to numerical noise and
  uncorrelated with anything.  Now that density genuinely acts it pushes the
  spectrum much the way brine volume does.

  The cosine is a prior-free, noise-free proxy and overstates the problem.
  The quantity that governs a retrieval is the posterior correlation from the
  OE path, with all fitted parameters, the measurement noise and the
  calibrated forward-model error: **0.31 mean, 0.77 max**, inflating the
  marginal 1-sigma by 1/sqrt(1-r^2) = only **1.10x typical, 1.56x worst**.  Both
  parameters remain measurement-determined; ``rho_DL``'s averaging kernel
  never drops below 0.84, so it is not prior-dominated.

  **Reparameterising onto along-/across-ridge coordinates was considered and
  rejected.**  The correlation *changes sign* across the parameter space (17 of
  24 base points positive, 7 negative, spanning -0.60 to +0.77), so no global
  linear rotation can decorrelate it; one would trade a modest state-dependent
  correlation for a fixed one that is wrong over half the domain.  No retrain
  is planned on this account.

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

- **The degeneracy diagnostic in ``scripts/experiments/fyi_bare_audit.py``
  had stale hardcoded ranges** — it sampled ``brine_volume_fraction`` over
  (0.03, 0.12) and ``rho_DL`` over (830, 890), and normalised by the retired
  0.141 upper bound, none of which match the live config.  It now derives
  every range from ``SEA_ICE_EMULATOR_CONFIGS`` so it cannot drift again, and
  reports the posterior correlation and sigma inflation alongside the cosine,
  with a note that only the former should be quoted.

- **The clamp below −30 °C is no longer silent.**  ``compute_brine_volume``
  accepts temperatures to −44 °C but holds the result at its −30 °C value below
  the Cox & Weeks fitted range; it now emits a ``RuntimeWarning`` saying so, and
  the docstring records that these are not Cox & Weeks values.  Unclamped, they
  would diverge from SMRT by up to ~35% at −38 °C.  Configured retrieval ranges
  stop at −30 °C, so no shipped code path reaches the clamp.

Added
~~~~~
- **``scripts/audit_sea_ice_degeneracy.py``** — posterior degeneracy audit for
  the whole fleet: every parameter pair of every emulator, with sign stability
  (which decides whether a reparameterisation can help) and averaging kernels.
  Generalises the FYI_bare-only, one-pair-only diagnostic that let the
  dominant degeneracies go unmeasured.

- **``tests/validation_data/parameter_identifiability_sheba.py``** — the same
  question on real spectra: which retrieved parameters sit on a bound, and
  whether the averaging kernel says the bound is binding or the likelihood is
  flat.  Catches the rail-pinning failure mode that a synthetic sweep started
  at the truth cannot see.

- **``RetrievalResult.parameter_correlation()`` and ``.degenerate_parameters()``**
  (``method="oe"``).  The OE engine already stored the full posterior
  covariance; these turn it into correlations and flag pairs above a
  threshold, so callers can read the measured collinearity of their own
  retrieval rather than relying on a blanket documented warning.  This matters
  because sea-ice collinearity is strongly state-dependent: for the FYI_bare
  ``rho_DL``/``brine_volume_fraction`` pair it varies from 0.07 to 0.77 *and
  changes sign*, so any global statement about it is wrong somewhere.

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


