# BioSNICAR — roadmap

Forward-model BioSNICAR is mature and published. This roadmap is about the
**sea-ice extension and retrieval framework** (`biosnicar.sea_ice`,
`biosnicar.inverse`), which is research in progress and where the open questions
are.

Source material, all in the repository: the full development plan with
workstream labels (`A1`–`A8`, `B-HS*`, `B-MS*`, `C*`, `D*`, `G*`, `V1`–`V7`) is
[docs/archive/SEA_ICE_DEVELOPMENT_PLAN.md](docs/archive/SEA_ICE_DEVELOPMENT_PLAN.md);
the model-error arc is
[docs/archive/OE_MODEL_ERROR_EXPERIMENT.md](docs/archive/OE_MODEL_ERROR_EXPERIMENT.md);
validation reports are [docs/sea_ice_validation.md](docs/sea_ice_validation.md)
and the `sea_ice_validation_*` documents in `docs/archive/`.

> **Branch note.** The sea-ice line and the chunked-parallel vectorised engine
> were both merged into `master` in November 2026, so `biosnicar.sea_ice` and
> `vectorized_parallel=` are available on a default clone.
>
> **Do not merge `perf/vectorized-parallel-thread-pin` (PR #6). It is a known
> regression and is deliberately reverted in this history.** Pinning one
> BLAS/OpenMP thread per worker took a staging scene to ~298 s from 76–160 s.
> The benchmark that justified it ran on macOS Accelerate, which **ignores
> `OMP_NUM_THREADS`**, so pinning appeared free; on Linux/OpenBLAS the batched
> chunk solve genuinely threads across cores, so `OMP=1` serialised it. Any
> future threading change here must be A/B tested on Linux, never on a local
> Mac. See PR #7 for the full post-mortem.
>
> **`fix/physics-core-bugs` is a stale duplicate — do not re-merge it.** It is
> based on the pre-merge `master` and every fix it carries is already on
> `master`: the TIR criterion restricted to the Reststrahlen band
> (`adding_doubling_solver.py`, see [docs/archive/TIR_CRITERION_BUG.md](docs/archive/TIR_CRITERION_BUG.md)),
> the SZA flux-LUT key fix (`illumination.py`), the 914 → 917 kg/m³ ice density
> correction (`geometric_optics_ice.py`) and the TIR regression test
> (`tests/test_snicar.py`). A trial merge resolves to a tree identical to
> `master`, conflicting only on one comment line where the branch still points
> at the pre-archive `docs/TIR_CRITERION_BUG.md` path. It can be deleted, as can
> the now-merged `fix-tir-criterion` and the four vectorisation branches.
>
> Still genuinely unmerged: `planet-data` and `fix/miepython-optional`.

---

## Honest baseline

What is validated, and in what regime — this is the part that matters most, and
the numbers do not generalise as far as they first appear.

| | |
|---|---|
| **Validated, spring/cold Arctic, VIS–NIR** | SHEBA spring snow 6/7; summer bare ice 16/16 (inferred labels); FYI_bare emulator held-out spectral R² 0.998 |
| **Only independent ground-truth metric** | Morassutti pond depth — ~22% within ±25% over 504 records, and reliable only above 50 cm |
| **Independent hold-out (Smith/MOSAiC, full SWIR)** | Summer melting snow is **optically degenerate** with SSL and bare ice; fit RMS roughly doubles out of distribution. **Spring numbers do not generalise to the melt season** |
| **Synthetic metrics** | Inverse-crime numbers — the same forward model generates and inverts. Treat as plumbing checks, not performance |
| **Untested** | Non-Arctic surfaces; open water and young ice against real spectra (synthetic only); scene-scale retrieval *accuracy* (throughput is solved — see Phase 3); atmospheric/TOA coupling; BRDF and angular effects; confidence calibration; drone and hyperspectral-satellite modalities |

Two significant fixes have shipped: the **liquidus correction** (`D1`, derived
from Frankenstein & Garner 1967 brine volume) removed a ~30% cold-ice
brine-salinity error, and the **forward-model error covariance** (`A7`) is in
place for the VIS. Note that the brine *volume* itself is Cox & Weeks (1983),
not Frankenstein & Garner — see the SMRT cross-check below, which is where that
mismatch matters.

Still deferred in the physics: **no liquid-water content in the melting-surface
forward models**, which is precisely the regime where the hold-out degrades.

### Cross-check against SMRT (2026-11-05)

SMRT ([Picard et al.](https://smrt-model.science/)) is the community sea-ice RT
model and the natural benchmark to aim at, but it is a **microwave** model —
brightness temperature and backscatter, roughly 5–243 GHz — where BioSNICAR is
optical. They share no radiative observable, so a like-for-like albedo
comparison does not exist and never will.

What they *do* share is the frequency-independent geophysical state model. Both
implement **Cox & Weeks (1983)** brine volume, so ν_b is directly comparable.
Result: **BioSNICAR's `compute_brine_volume` agrees with SMRT v1.7.3 to
0.0326–0.0331%** over T ∈ [−3, −30] °C, S ∈ [1, 15] psu — the whole residual
being the pure-ice density constant (0.917 vs 0.9167 g cm⁻³). The
implementation is confirmed correct, and `tests/test_brine_volume.py` now pins
it against hard-coded SMRT values.

Two pieces of documentation were wrong and have been corrected: a docstring
claiming a "~25–35%" discrepancy against Cox & Weeks Table 1 "due to
unit/convention ambiguity in the original paper" (there is none — the published
values are reproduced to 3%, and SMRT misses them by the same margin), and a
reference-value test whose 40% tolerance was wide enough to pass almost any
implementation. Tolerance is now 5%.

Three genuine gaps the cross-check exposed. The first is **now closed**; the
other two still need a decision:

| | |
|---|---|
| ~~**No gas porosity**~~ | **Done** — Cox & Weeks carry an air-volume term we omitted, i.e. we assumed bubble-free ice. `compute_brine_volume` now takes `porosity` or `bulk_density_kg_m3`, `compute_gas_volume` supplies the gas half of the same balance, and the optics pass the layer density into both. See the porosity section below |
| **Mixed phase model** | ν_b comes from Cox & Weeks, but the liquidus S_b feeding brine optics is derived from the *Frankenstein & Garner (1967)* brine-volume relation. The two disagree by 1.7% at −2 °C rising to **13.9% at −22.9 °C**, so brine volume and brine salinity are not drawn from the same phase model. Standardise on one. (The air fraction no longer depends on F&G — that half is now pure Cox & Weeks — so this is confined to the brine optics) |
| **Melt season refused** | `compute_brine_volume` raises above −2 °C. SMRT covers −2 to 0 °C via Leppäranta & Manninen (1988). This is exactly the regime where the hold-out above degrades, and where microwave would break the optical degeneracy |

### Closing the porosity gap (2026-11-05)

Cox & Weeks give brine and gas volume as two halves of one density balance.
BioSNICAR computed ν_b *without* the density (bubble-free) and then derived
ν_air from that same density via a separate balance that used a
Frankenstein & Garner-derived brine density — two different phase models
describing the same air, only one of which ν_b had assumed.

Both halves now come from Cox & Weeks, driven by the layer's bulk density.
`compute_brine_volume` accepts either `porosity` or `bulk_density_kg_m3`
(mutually exclusive, as in SMRT); the new `compute_gas_volume` is its
companion. Verified against SMRT v1.7.3: the porosity path agrees to 0.033%
(the ice-density constant again), and **the bulk-density path agrees exactly**,
since it never evaluates ρ_ice. Defaults are unchanged, so bubble-free callers
see identical numbers.

Chasing this exposed a larger, independent bug. The sea-ice LUT is
interpolated with **clamping, not extrapolation**, and its density axis started
at 870 kg/m³ — while the shipped presets use a drained-layer density of 850
(Jin et al. 2023) and the FYI_bare emulator samples `rho_DL` over (820, 900).
So roughly 60% of that sampled range was clamped flat to 870: the preset
density was silently wrong, and the emulator was training on a parameter that
did nothing over most of its range. The LUT grid now runs 820–920.

A third defect surfaced from the same thread: **absorption ignored air
entirely**. `abs_coeff = 4π·n_eff_im/λ` was the absorption of the ice+brine
phase applied across the whole layer, so a layer that is half air absorbed as
though it were solid ice. Measured: absorption per metre sat at ~82 m⁻¹ from
920 down to 500 kg/m³. Every other layer type already handles this correctly,
via mass absorption coefficients that exclude air by construction — sea ice was
the lone exception. Absorption is now scaled by `(1 - ν_air)`, and the
Maxwell-Garnett mix receives the brine fraction *of the condensed phase*
(`ν_b/(1-ν_air)`) rather than of the whole layer.

Air is deliberately **not** mixed into the effective medium: Maxwell-Garnett
assumes sub-wavelength inclusions, and the bubbles are 100–1000 µm against
0.2–5 µm light. That is why they are discrete Mie scatterers. For large
non-absorbing inclusions, path-length volume weighting is the correct limit.

Measured effect on broadband albedo across the six presets:

| | max ΔBBA |
|---|---|
| LUT density clamp (the pre-existing bug) | **+0.026** |
| Cox & Weeks porosity term | +0.009 |
| Absorption air-weighting | +0.006 |
| Combined | **+0.039** (+6.5% on FYI_WINTER_BARE, +10% on ponds) |

### What this does and does not buy (measured, not asserted)

A fixed-preset comparison initially suggested these corrections made agreement
*worse* — summer bare-ice BBA bias rising from +0.028 to +0.068. That reading
was wrong, and the reason is instructive: a fixed preset confounds "the physics
is wrong" with "the assumed inputs are wrong", and the reference spectra
constrain density, bubble radius and impurity loading very poorly.

The right test is the **best achievable fit** with those parameters free.
Fitting six SHEBA bare-ice spectra (400–1000 nm, differential evolution):

| forward model | mean RMSE |
|---|---|
| Original | 0.0828 |
| Corrected, density floor 820 | 0.0464 |
| Corrected, density floor 500 | 0.0158 |

So with configuration free the corrected physics fits far better, not worse.

But that improvement is **almost entirely the density range, not the air
physics**, and it is an artefact of layer structure. Fitting a two-layer
column, `rho_DL` pins at whatever floor it is given (500, then 350) because in
a bubbly-ice layer *lower density means more scattering* — density is the
brightness knob. Add a surface scattering layer and `rho_DL` jumps to ~900 on
every date, none at the floor, with the best fit of all (RMSE 0.0107). The
ultra-low densities were the optimiser imitating a missing SSL.

With the correct three-layer structure, the corrected and original forward
models fit **identically well (0.0107 both)**. At the 850–920 kg/m³ the sea-ice
layers actually want, air is only 0.6–8%, which nine free parameters absorb
trivially.

The corrections are therefore landed for **correctness and parameter
interpretation, not fit quality**. The retrieved densities are visibly better
behaved: corrected gives 899/900/891/900/900/895, the original
785/828/890/900/897/900 — same fit, tighter parameters. For a retrieval model
whose output is meant to mean something, that is the argument.

**Density floor: 700 kg/m³.** Set from the literature range, not from what the
code happened to ask for. [Timco & Frederking (1996)](https://openpolar.no/Record/ftnrccanada:oai:cisti-icist.nrc-cnrc.ca:cistinparc:12339267/Details)
give sea-ice densities of 720–940 kg/m³, with **multi-year ice above the
waterline at 720–910**. The old 870 floor therefore excluded most of the MYI
range, and the 820 floor this work first used still excluded 720–820 — in both
cases silently, because the interpolator clamps.

720 kg/m³ is 21.5% air: dispersed gas in a continuous ice matrix, nothing like
grain-supported. The bubbly-ice topology holds until air percolates, somewhere
near 30–35% air (~600–640 kg/m³), so 700 sits inside the valid regime with
margin while covering the full documented range.

This bit hardest exactly where it mattered most: the MYI preset is
`rho=[850, 870]`, commented "IL=870 (MYI, more gas pockets)" — the ice type
defined by low density was pinned at the old grid floor. (The MYI emulator also
does not sample density at all, holding `rho=[850, 870]` fixed. That is a
separate gap and would need its own retrain.)

The absorption correction above is what makes this floor usable: without it
absorption was constant regardless of porosity, a 27% overestimate at
720 kg/m³, so a lower floor would only have produced confident nonsense. That
is the concrete payoff of a change that showed no fit-quality benefit.

Weathered ice below ~640 kg/m³ is real but is over 30% air, past where
"bubbles in a continuous matrix" holds; it belongs in `layer_type=0`, whose
grains-in-air topology is the better of two imperfect pictures there.

The `brine_volume_fraction` training ranges were corrected — their old upper
bounds (0.141 FYI, 0.045 MYI) were the bubble-free image and are not reachable
in ice that holds air. `invert_brine_volume` now takes the layer density too,
so the sampled `brine_volume_fraction` still names the brine volume the forward
model actually uses rather than becoming degenerate with `rho_DL`.

**Emulator accuracy has been re-measured after the rebuild; the retrieval and
classification numbers have not.** On a fresh 2000-spectrum holdout (seed 777)
FYI_bare scores pooled spectral R² 0.9977 / BBA MAE 0.0024, against 0.9978 /
0.0021 before — unchanged to within the noise of one holdout draw, with the
worst-case tail slightly better (5th-percentile per-spectrum R² 0.984 → 0.988)
because the density axis no longer clamps. All six emulators sit at 0.998–1.000
held-out spectral R² (`scripts/audit_sea_ice_emulators.py`). The SHEBA,
Morassutti and MOSAiC numbers in the baseline table above were produced under
the old physics and should be regenerated before being quoted again.

### Open, found along the way

| | |
|---|---|
| **SSL morphology** | The SSL is modelled as granular (`layer_type=0`, grains in air). Weathering-crust structure is arguably better described as bulk ice with large interconnected voids — coral or karst rather than a grain pile — which is what `layer_type` 1/2 already provide for glacier ice. Worth testing whether a voids-in-ice SSL fits better, and whether the Fresnel interface belongs at the SSL top (a crust is a solid ice surface; a grain pile is not). Caveat: `bubbly_air.npz` is spherical-inclusion Mie, so it would get the topology right and the morphology wrong |
| **Melt-pond Fresnel** | Pond columns are `[5,4,4]`, so the Fresnel interface lands on the ice *under* the pond and is computed as air→ice, ignoring the water above. It should be a strong interface at the pond top (air→water) and almost none at the pond bottom (water→ice, 1.33→1.31). Suppressing it moves pond BBA by ~0.06–0.07, ~30% relative. There is no way to disable it for sea ice — `fresnel_types = {1, 4}` with no type-2 analogue for type 4 |
| **No brine outside types 4/6** | Granular and bubbly-ice layers carry no salinity, so genuinely saline weathered ice has no correct representation. Drained crust is largely desalinated, so this is probably tolerable — but it is a silent assumption |
| **Layer-type discontinuity** | Granular and bubbly-ice layers respond to density with *opposite sign* (lower density is darker for grains, brighter for bubbles). Sweeping density across the regime boundary without switching layer type gives nonsense |
| **Vectorised/loop divergence in band mode** | The two OE engines do not agree to machine precision with only nine Sentinel-2 bands: per-class confidences differ by up to 9.8e-2 before these corrections, ~2e-2 after. Full-spectral mode agrees to 2e-3. `test_vectorized_matches_loop_band_mode` previously asserted exact class agreement and passed only because both engines were confidently *wrong* on the same pixels; it now asserts that confident pixels agree and tolerates one marginal near-tie. The divergence itself should be fixed |
| **MYI emulator holds density fixed** | `_transform_myi_bare` hard-codes `rho=[850, 870]`, so MYI retrievals cannot adjust the parameter that most distinguishes multi-year ice — and the literature range reaches 720 kg/m³ |
| **`sea_ice_temperature` is reported but not retrieved** | Averaging kernel **0.048** across 22 of 23 SHEBA spectra — roughly 95% prior, 5% measurement; the fleet sweep agrees (0.16 FYI_summer, 0.25 FYI_snow, 0.29 young_ice). Temperature acts only through brine volume and brine salinity, and those emulators are `vis_only`-masked where bubble scattering swamps brine absorption, so there is no VIS signal to retrieve. It is currently returned with an uncertainty that implies more than it should. Options: drop it from the reported parameter set, surface a quality flag from the averaging kernel, or extend those emulators into the SWIR where brine absorbs. **Highest-value open item from this work** |
| **Every emulator has a degeneracy, not just FYI_bare** | Worst pair per emulator spans mean \|r\| 0.49–0.89: FYI_bare `bubble_radius`/`rho_DL` 0.89, MYI_bare `bubble_radius`/`black_carbon` 0.85, young_ice `ice_thickness_cm`/`ocean_albedo` 0.85, FYI_pond `T`/`black_carbon` 0.79, FYI_summer `bubble_radius`/`black_carbon` 0.76, FYI_snow `tau_snow`/`black_carbon` 0.49. Three keep a stable sign (FYI_bare, FYI_summer, young_ice) and are reparameterisable; each needs its own retrain. FYI_snow is the proof the technique works — its already-reparameterised `tau_snow`/`snow_grain_radius` pair is now the *least* degenerate in the fleet. Measure with `scripts/audit_sea_ice_degeneracy.py` |
| **24% of parameters land on a bound in real SHEBA retrievals** | 22/92, with 19/23 spectra carrying a \|r\| > 0.8 pair. Pin + high averaging kernel means the bound is binding; pin + low means a flat likelihood and a meaningless reported value. `FYI_summer`'s `sea_ice_bubble_radius` ceiling (500 µm) is half the sea-ice LUT's support and half FYI_bare's; a forward-model scan past it finds the optimum beyond 500 in 4 of 8 pinned cases but with marginal gain, so this is tidiness rather than an accuracy bug. Check with `tests/validation_data/parameter_identifiability_sheba.py` |
| **`sea_ice_bubble_radius`/`rho_DL` is the dominant FYI_bare degeneracy** | Found while measuring the Vb pair, and far bigger: posterior correlation **0.84 mean, 0.995 max**, the same sign at all 24 base points, inflating marginal σ by **4.2× typical and 9.8× worst**. Never previously measured — the audit only looked at `rho_DL` vs Vb. Mechanism: `sea_ice_optics.py` forms scattering as `sca_cff_vlm(bubble_radius) × ν_air` with `ν_air` set by density, so the two are knobs on one product (not perfectly interchangeable — g depends on radius only, absorption on air only, and T is tied to (Vb, `rho_DL`), hence 0.84 not 0.999). Because the sign is stable, reparameterising onto the constrained combination **is** viable, with the granular-ice `rds`/`rho` → SSA treatment as precedent. Needs a retrain; **not done — decision pending** |
| **`rho_DL`/brine-volume collinearity rose with the fixes** | Expected, not a regression: with the LUT clamping at 870 most of the sampled `rho_DL` range was inert, so its Jacobian was near-noise and looked uncorrelated. Measured properly it is minor — posterior correlation **0.31 mean / 0.77 max** (the Jacobian cosine, 0.73, overstates it badly), inflating marginal σ by only **1.10× typical, 1.56× worst**. Both parameters stay measurement-determined (`rho_DL` averaging kernel ≥ 0.84). **Reparameterising was considered and rejected**: the correlation changes sign across the parameter space (17 of 24 positive, 7 negative), so no global linear rotation can decorrelate it. Exposed per-retrieval instead via `RetrievalResult.parameter_correlation()` / `.degenerate_parameters()`. **Closed 2026-11-06** |

| **Pond floor calibration is stale** | The BC = 1000 ppb / rho = 895 kg/m³ pond-floor parameterisation in `docs/sea_ice.md` was fitted against Morassutti (1995) under the old forward model. The fixes move pond BBA by ~+10%, so both values are an optimum of a model that no longer exists. The sweeps need re-running; the argument for an explicit floor absorber is unaffected |

The clamp below −30 °C is now documented and warns, but is latent in practice:
configured retrieval ranges stop at −30 °C, so it is unreachable except by
direct call. Unclamped, it would diverge from SMRT by up to 35% at −38 °C.

The higher-value programme this opens is a **shared-column closure experiment**
— drive both models from one ice state and validate each in its own domain
against coincident optical and microwave measurements (SHEBA, MOSAiC). Microwave
is sensitive to brine volume and wetness, which is precisely what the optical
signal cannot separate in the melt season.

---

## Phase 0 — probabilistic core and correctness

Mostly complete. The optimal-estimation core (`A1`–`A3`) is in place with
per-class posteriors, provenance, and the `use_priors` / `flag_prior_influence`
/ `prior_resolved` toggles; `D1` has shipped; `V2` perturbed-physics validation
has bounded the inverse-crime overstatement.

Remaining:

- **Finish `A7` — the SWIR model-error term.** The VIS term generalises well
  (held-out χ² 0.13). The SWIR does not, and the reason is now diagnosed and is
  **physical rather than methodological**: SWIR forward-model error is strongly
  regime-dependent, with calibration residual RMS ≈ 0.067 against held-out
  ≈ 0.134, so a two-regime covariance is about 4× too tight for a third. The
  gap-tolerant fit works — the domain reaches ~1995 nm with healthy low-rank
  structure — so **this is data-limited, not blocked**. The shipped covariance
  stays VIS-only (`domain_max_nm=1000`, per-band diagonal in the SWIR), and
  `build_model_error_covariance.py --gap-tolerant` is the ready path to switch
  the SWIR term on as soon as regime-spanning full-range residuals exist.
- **`V1` — freeze the independent test corpus.** Stop tuning against test data.
  No new data required; this is a discipline decision.
- **The default-behaviour decision.** Should `method="oe"` default to
  `model_error=True`? It widens uncertainties, which is honest, but it changes
  `class_probabilities` for downstream consumers (IceNav). Coordinate before
  flipping rather than after.

**Gate:** OE per-class posteriors with provenance and DFS; inverse-crime
degradation bounded; test corpus frozen.

## Phase 1 — both standalone modalities to first class

The standalone product is the hyperspectral and satellite paths, and this phase
brings both to the same standard.

- **`B-MS1` — the atmospheric-correction contract.** Define it early; it is
  cheap to specify and the satellite mandate is not credible without it. This is
  the single highest-risk item in the whole plan (see the risk register below).
- **`B-HS1`–`2`, `B-MS2`** — the two standalone paths end to end.
- **`A5`, `A6`, `A8`** — DFS-aware retrieval and calibration. Note that until
  `A6` produces reliability diagrams on a held-out corpus, **the word
  "calibrated" should not be used** about these probabilities.
- **`G1`, `G4`, `V3`, `V5`** — engineering and per-modality benchmarks.

**Gate:** both paths validated end to end, including an atmospheric-correction
contract for satellite; calibrated, DFS-aware retrieval; published per-modality
benchmarks.

## Phase 2 — priors, ensemble interface, empirical expansion

- **`C1`–`C2` — metadata adapter and temperature prior.** Now worth attaching:
  the evidence gaps sit in the few-nat regime where a 5:1 month/location prior
  has real leverage. This is also the modular toggle that lets the same code
  serve standalone use (priors on) and ensemble use (priors withheld) without
  forking.
- **`C3`, `C4`, `C6`** — the rest of the prior and ensemble interface.
- **Empirical data collection** (plan §5) — including the full-range,
  regime-spanning residuals that Phase 0's SWIR term is waiting on. More MOSAiC
  legs, Antarctic data, or a dedicated campaign.
- **`V4`, `V6`** — independent multi-campaign validation plus at least one
  non-Arctic site.
- **`D2`** — liquid-water content in the melting-surface forward models, the
  deferred physics behind the melt-season degradation.

**Gate:** honestly-attributed prior lift on the dark cluster and the FY/MY split
*on real data*; published ensemble contract; melt-season SWIR fit improved.

## Phase 3 — real-sensor fidelity and scale

`B-MS3`–`4`, `B-HS3`, `D4`–`D5`, `G2`: BRDF and glint, mixed-pixel handling,
scene and mosaic-scale benchmarking, hyperspectral-satellite matchups.

**`G3` (scene-scale retrieval speed) is done** and no longer blocks this phase.
`engine="vectorized"` batches the OE solve across pixels, and
`vectorized_parallel=True` fans that solve over worker processes, bit-identical
to single-process (measured 3.1× on 8 workers of a 10-core box, with a warm
per-worker fleet cache across repeated scenes). What remains under "scale" is
the *benchmarking* of retrieval quality at scene and mosaic scale, which is an
accuracy question, not a throughput one.

## Phase 4 — operational deployment

`G5`–`G6`, `V7`: standalone applications released, ensemble lift demonstrated,
expert sign-off, reproducible deployment.

---

## Risk register

From the development plan §9, with the decision rule for each — these are
rescope triggers, not worries.

| Risk | Likelihood | Decision rule |
|---|---|---|
| **Atmospheric correction dominates satellite error** | medium-high | Quantify in `V6`. If it swamps the signal, the satellite mandate rescopes to L2A-only / hyperspectral-satellite, and field/drone becomes the primary standalone use |
| **Optical degeneracies irreducible even with priors** | medium | Accept the ensemble role. If SAR/thermal/age cannot lift `V7`, the value is parameter retrieval and classification claims scope down accordingly |
| **No coincident structural ground truth obtainable** | medium | Provenance (snow vs ice) stays unvalidated — ship optical state and parameters, not structural labels |
| **Few-band satellite DFS too low** | medium | Report class and prior-dominated parameters honestly; position multispectral as classification-first, hyperspectral as retrieval-first |
| **Cubic-liquidus / LWC rebuild shifts validated results** | low-med | The Phase-0 gate re-runs `V1`–`V2` before accepting |
| **Calibration infeasible at the available data volume** | medium | Report uncalibrated with an explicit warning; fusion down-weights |

### Kill / rescope criterion

If Phase 1 cannot get the satellite path past the atmospheric-correction
barrier **and** Phase 2 cannot obtain independent validation, ship as a
**hyperspectral field/drone retrieval tool** — where the physics is strongest
and best validated — plus an ensemble likelihood emitter. Not an operational
satellite classifier.

Writing that down in advance is the point: it is a decision rule, so that the
outcome is a scoping call rather than a disappointment.

---

## Forward model

The forward model itself is stable and published, and nothing above changes it.
Ongoing work is maintenance rather than roadmap: keeping the optional `mie`
extra working across platforms (numba/llvmlite publish no wheels for Intel
macOS from llvmlite 0.46 / numba 0.63 onward), platform band definitions as
sensors are added, and the interactive classroom.
