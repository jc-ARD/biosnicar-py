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
| **In-sample, spring/cold Arctic, VIS–NIR** | Re-measured 2026-11-12. SHEBA spring snow 6/7 full-spectrum, 7/7 on S2 and L8 bands; summer bare ice 16/16 full-spectrum, 16/16 S2, 16/16 L8. SHEBA is a **development** set (see the corpus freeze below): priors, masks and bounds were set against it, so these say the model *can* be made to fit, not that it generalises |
| **Melt ponds (in-sample)** | Re-measured 2026-11-12 after the pond-surface interface fix. Morassutti classification **88.5%** (446/504; 74.6% before) and depth within ±25% **22.6%** (23.8% before). The classification gain was not sought: it is a side effect of giving the pond a water surface, which the old model lacked entirely. Depth is unchanged and still only usable for deep ponds (81% within ±25% beyond 50 cm, 5–17% below 20 cm) — the new audit shows why: pond depth and floor coarseness are now the most degenerate pair in that emulator (mean \|r\| 0.84) |
| **Synthetic metrics** | Inverse-crime numbers — the same forward model generates and inverts. Treat as plumbing checks, not performance |
| **Untested** | Non-Arctic surfaces; open water and young ice against real spectra (synthetic only); scene-scale retrieval *accuracy* (throughput is solved — see Phase 3); atmospheric/TOA coupling; BRDF and angular effects; confidence calibration; drone and hyperspectral-satellite modalities |

Two significant fixes have shipped: the **liquidus correction** (`D1`, derived
from Frankenstein & Garner 1967 brine volume) removed a ~30% cold-ice
brine-salinity error, and the **forward-model error covariance** (`A7`) is in
place for the VIS. Note that the brine *volume* itself is Cox & Weeks (1983),
not Frankenstein & Garner — see the SMRT cross-check below, which is where that
mismatch matters.

Still deferred in the physics: **no liquid-water content in the melting-surface
forward models**. See `D2` under Phase 0 below — the fast wet-grain
representation in the code has the wrong sign in the near-infrared, so this
needs a coated-sphere lookup table rather than a parameter added to an
emulator.

### How to read the field numbers

Every field dataset here has uncertain inputs: solar zenith is inferred from
date or ship position, direct/diffuse from sky notes, labels are season-inferred
or coarse field comments, and the structure of the ice was not measured. With
five or six free parameters, several of them degenerate, a forward model can be
made to fit most of these spectra whether or not it is right — the SSL
experiment below fits six structurally different columns to within 0.001 of
each other. So the rule for this roadmap is: **decide physics on physical and
mathematical correctness; use the field data to expose a wrong model, never to
choose between defensible ones or to tune a bound.** Agreement is weak
evidence. The numbers in the table above are reported on that footing.

### Corpus freeze (`V1`, done 2026-11-12)

`tests/validation_data/corpus_manifest.json` assigns each dataset one role and
hashes every data file; `corpus.require_role` refuses to tune or calibrate on a
test set, and the suite fails if a frozen file changes.

| Dataset | Role |
|---|---|
| SHEBA (Grenfell & Light 2007), Morassutti (1995), Macfarlane (2022) | development — in-sample |
| Smith (2021) MOSAiC | calibration — feeds the model-error covariance |
| Istomina (2016) IceArc | **test — frozen, evaluate only** |

One campaign of 121 coarsely-labelled spectra is a thin test set. It is what
exists. A second independent campaign remains the most valuable thing this
project could acquire.

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
| **SSL morphology** | **Tested 2026-11-12; the spectra cannot decide it.** `scripts/experiments/ssl_morphology_experiment.py` fits the 16 SHEBA summer spectra with grains-in-air and voids-in-ice surface layers, each with the interface below, on top, or absent — six structures, five free parameters each. Mean RMSE spans 0.0057–0.0067 across all six; the shipped structure is nominally lowest and has the best worst case, but the spread is under 0.001 and below measurement noise. So VIS albedo carries no usable information on SSL topology or interface placement, and the choice has to rest on structural evidence (micro-CT), not on fit. `run_model(fresnel_layer=...)` now makes the placement an explicit, switchable assumption. One thing every variant agreed on: the fitted layer is ~1.2–1.5 cm thick, against the 5 cm the FYI_summer emulator fixes |
| ~~**Melt-pond Fresnel**~~ | **Fixed 2026-11-12.** The refractive boundary of a ponded column is now the pond surface, air→water with water's refractive index and exact diffuse reflectances; the water→ice step at the floor (1.33→1.31, ~6×10⁻⁵) is neglected. The old placement gave a pond no surface reflection at all — NIR albedo exactly zero where the water is opaque. BBA moves about −0.01 at SZA 50°, +0.06 at SZA 70°. FYI_pond retrained (held-out R² 0.9993). See `docs/sea_ice.md` |
| **Inherited ice diffuse-Fresnel table is ~0.02 low from below** | Found while doing the above. `fl_reflection_diffuse.npz` (SNICAR-ADv4) matches an exact calculation to 2×10⁻⁷ for diffuse light from above, but its from-below column violates the `n²` radiance relation: 0.427 against 0.451 at 0.7 µm (CICE uses 0.455 for n = 1.31). Applying the exact value lowers albedo by 0.003–0.005 on sea-ice presets and glacier ice alike. **Not changed** — it would move the published glacier-ice benchmarks — and it is **on hold until confirmed with the SNICAR-ADv4 (MATLAB) authors** (decided 2026-11-12): how the table was generated is not known, so a deliberate convention cannot be ruled out. Do not change it before then |
| **No brine outside types 4/6** | Still true, no longer silent: `sea_ice_salinity` on a granular or bubbly-ice layer now warns that the layer is computed as fresh ice (2026-11-12). A real representation of saline weathered ice or saline snow remains unbuilt |
| **Layer-type discontinuity** | Granular and bubbly-ice layers respond to density with *opposite sign* (lower density is darker for grains, brighter for bubbles). Sweeping density across the regime boundary without switching layer type gives nonsense |
| ~~**Vectorised/loop divergence in band mode**~~ | **Fixed 2026-11-12.** Not weak conditioning — two defects in the vectorised path. It skipped the per-band platform SNR default for `obs_uncertainty` (flat 0.02 instead, so a different `S_e`), and it convolved every candidate with the *first* emulator's solar flux rather than each emulator's own. The engines now agree to 2×10⁻³ in band mode as in spectral mode, and the test is strict again. `engine="vectorized"` band-mode results therefore change for callers that do not pass `obs_uncertainty` — they now get the SNR-table uncertainties the per-pixel path always used. (BioSNICAR is not currently served in IceNav, so no live integration is affected) |
| ~~**Each emulator carries an arbitrary solar flux**~~ | **Fixed 2026-11-12.** `Emulator.flx_slr` is the flux of the *first training sample*: for the shipped fleet that was clear sky at SZA 53° (FYI_bare), 32° (FYI_snow, FYI_summer), 60° (young_ice) and an overcast sky (MYI_bare, FYI_pond). Everything derived from an emulator spectrum was weighted by it regardless of the geometry asked for. Satellite bands are narrow and barely moved (≤ 0.001), but **broadband albedo was wrong by 0.007–0.027 on average and up to 0.08**, since clear and overcast spectra weight the visible and near-infrared very differently. `Emulator.solar_flux(solzen, direct)` now supplies the flux of the stated illumination to `run_emulator`, band convolution (both OE engines, per pixel where geometry is per pixel) and retrieval results. No retrain: albedo itself does not depend on the flux spectrum. The atmosphere profile is still the training default (mid-latitude winter) for every retrieval; making that selectable is a small follow-up |
| ~~**MYI emulator holds density fixed**~~ | **Done 2026-11-06.** `rho_DL` is now sampled over (720, 900) kg/m³ — the Timco & Frederking range for MYI above the waterline — and the emulator is 6-parameter. Accuracy improved despite the extra dimension (vis-only RMSE 0.0069 → 0.0054). This also makes `RetrievalResult.ssa` available for MYI retrievals |
| **Report SSA fleet-wide, not radius and density separately** | Policy decision: the individually degenerate pair should be reported as its combination, `SSA = 3(1-ρ/917)/(rρ)`. Justified from the code — ω is a function of SSA alone — and measured: holding SSA fixed removes **88% of NIR and 61% of VIS** spectral variation, and σ(SSA)/SSA beats σ_r/r in 22 of 24 states (0.44 vs 0.58). `RetrievalResult.ssa` now reaches sea-ice parameter names and propagates with the posterior covariance (3× tighter). `run_model` now accepts `ssa=` and `brine_volume_fraction=` as alternatives to the physical inputs (either set, never both for a layer), so the forward model is done. **Retrieval is blocked for the rest of the fleet**: 5 of 6 emulators hard-code density, so SSA is computable for 0 of 23 SHEBA spectra. Making it primary requires those emulators to sample density — a retrain each. Caveats: SSA amplifies density error by 1/φ ≈ 9–30× at sea-ice densities, and it addresses only the radius/density pair, not the other four degeneracies measured |
| **`sea_ice_temperature` is reported but not retrieved** | Unchanged in substance: averaging kernel 0.016 across 22 of 23 SHEBA spectra under the default covariance, flagged as not retrieved in every output. The SWIR extension stays blocked behind `A7`'s SWIR term, which is now understood to be a physics problem rather than a data-volume one (Phase 0). One nuance from the pond fix: under the pond surface, light recycling gives temperature a small real VIS signal through the floor's brine volume (0.006 albedo RMS per 3 °C); it is over-counted to an averaging kernel of ~0.6 by an instrument-only `S_e` and is ~0.2 under the calibrated one |
| **Prior-dominated parameters were indistinguishable from measured ones** | **Closed 2026-11-06.** The averaging kernel was computed and stored but nothing consumed it by default, so `summary()`, `plot()` and the scene rasters gave every free parameter the same authority. `uncertainty` cannot substitute — a narrow prior and a sharp measurement both give a small σ. `biosnicar.inverse.information` now defines one threshold convention (measured ≥ 0.5, not retrieved < 0.1) shared by both result classes, both summaries, the plot and new `akd_<name>` scene rasters; `_result_to_record` no longer drops the kernel on the per-pixel loop path, which had made provenance depend on which engine ran. This matters beyond temperature: `ocean_albedo` (0.42 mean, 0.03 min), `brine_volume_fraction` (0.49) and `black_carbon` (0.62–0.97, minima near 0.01) sit in the same band. Caveat now documented: the kernel is relative to the prior supplied, not to truth — a wide prior inflates it for free, so read it with `AT_BOUNDS` and `parameter_correlation()` |
| **Every emulator has a degeneracy, not just FYI_bare** | Re-audited 2026-11-12 (`scripts/audit_sea_ice_degeneracy.py`). Worst pair per emulator: MYI_bare `bubble_radius`/`rho_DL` 0.99, FYI_bare `bubble_radius`/`rho_DL` 0.95, FYI_pond `pond_depth`/`bubble_radius` 0.84 (new — light recycling under the pond surface couples depth to the floor), young_ice `ice_thickness_cm`/`ocean_albedo` 0.83, FYI_summer `ssl_grain_radius`/`bubble_radius` 0.60 synthetic but `bubble_radius`/`black_carbon` −0.98 on real SHEBA spectra, FYI_snow `tau_snow`/`black_carbon` 0.55. **Retraining on rotated coordinates was considered and not done**: a rotation re-expresses the same posterior and adds no information, and it would rename retrieved parameters. Instead `constrained_combinations()` returns the posterior's principal axes for any retrieval — the combination the spectrum fixed, in units of the marginal error bars (σ = √(1−\|r\|), 7× tighter at r = −0.98) — and `summary()` names every pair with \|r\| ≥ 0.8. Where a combination has a physical name it should be reported by name, as SSA is |
| ~~**24% of parameters land on a bound in real SHEBA retrievals**~~ | **5/92 as of 2026-11-12, one of them a genuine rail.** Three things changed. `FYI_summer`'s bubble-radius ceiling went 1000 → 2000 µm (the LUT limit), where 4 retrievals had been pinned with averaging kernel 0.83 (23 → 19 pinned). Spectral OE then moved to the field-calibrated covariance by default, under which the over-counted evidence no longer drives parameters into walls (19 → 5). And the audit now separates bounds that are *meant* to bind — black carbon at zero is a physical floor; FYI_snow's `tau_snow` floor and −5 °C ceiling are class boundaries with FYI_summer — from rails: 4 of the 5 are the former. The one left is `ssl_grain_radius` at its 500 µm floor on one spectrum. No bound was moved to improve a score |
| **`sea_ice_bubble_radius`/`rho_DL` — report SSA, do not report the components** | After the 2026-11-06 retrain the posterior correlation is **0.961 (FYI_bare) and 0.982 (MYI_bare)**, σ inflated 6–8× (20× worst), same sign at all 24 base points. It rose from 0.837 because fixing the LUT bubble clamping made radius genuinely active — a clamped parameter's Jacobian is noise, and noise correlates with nothing. At these values the components are not separately determined: a live MYI retrieval gives radius 589±327 µm (56%, 26% off truth) against **SSA 0.7220±0.031 (4.3%, 0.5% off truth)**. Reparameterising the emulator onto `(ssa, ρ)` was considered and **rejected**: that box is ill-posed (usable SSA span collapses to 6.4× vs 40× in radius; a full box demands 8 µm bubbles), and a rotation adds no information anyway. `RetrievalResult.ssa` already reports the combination with covariance. **Closed 2026-11-06** |
| **`rho_DL`/brine-volume collinearity rose with the fixes** | Expected, not a regression: with the LUT clamping at 870 most of the sampled `rho_DL` range was inert, so its Jacobian was near-noise and looked uncorrelated. Measured properly it is minor — posterior correlation **0.31 mean / 0.77 max** (the Jacobian cosine, 0.73, overstates it badly), inflating marginal σ by only **1.10× typical, 1.56× worst**. Both parameters stay measurement-determined (`rho_DL` averaging kernel ≥ 0.84). **Reparameterising was considered and rejected**: the correlation changes sign across the parameter space (17 of 24 positive, 7 negative), so no global linear rotation can decorrelate it. Exposed per-retrieval instead via `RetrievalResult.parameter_correlation()` / `.degenerate_parameters()`. **Closed 2026-11-06** |

| ~~**Pond floor calibration**~~ | **Resolved 2026-11-06, and it was not a calibration problem.** The BC = 1000 ppb / ρ = 895 figures in `docs/sea_ice.md` are for the *preset*, not the FYI_pond emulator, which samples BC freely and uses ρ = 850. The actual defect was the floor's hard-coded 200 µm bubble radius. Black carbon could only reach the dark end at 12,000–25,000 ppb (literature max ~2000 ng/g) and density only at 916 kg/m³ (bubble-free pure ice) — both indefensible, which is what pointed at coarseness. The preset sweep in `docs/sea_ice.md` is still stale and still should not be quoted |

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

- **`A7` SWIR model-error term — reframed: this is physics-limited, not
  data-limited.** Correct-model SWIR residuals are 0.07 (Smith) to 0.12
  (SHEBA) albedo RMS, against ~0.02 in the VIS. Residuals that large and that
  regime-dependent are structural forward-model misfit, and a covariance
  cannot stand in for missing physics: a low-rank SWIR term fitted on
  SHEBA+Smith makes the frozen test set *worse* (SWIR-band reduced χ² 11.8
  against 5.2 with the per-band diagonal alone; measured on the residual
  library as it stood before the 2026-11-12 rebuild), and the inflation needed to
  cover it (~4×) amounts to telling the retrieval to ignore the SWIR. So the
  shipped covariance stays VIS-only by decision rather than for want of data,
  and the SWIR term waits on the SWIR physics (`D2`, surface-layer
  thickness), not on another campaign. The covariance itself was rebuilt
  2026-11-12 from residuals of the current emulators — the previous artifact
  had been fitted to a fleet since retrained twice — and holds out at reduced
  χ² 3.7 mean / 2.4 median on Istomina (39.6 / 33.1 instrument-only).
- **`D2` liquid water — needs a coated-sphere LUT.** A melting surface with no
  liquid water is physically wrong, but the fast wet-grain model in the code
  is not the fix. `layer_type=3` adds water as *separate spheres*, i.e. extra
  scatterers, so water brightens the layer at every wavelength; water held on
  grains should darken the NIR, which is what the coated-sphere model
  (`water=`) gives (1.26 µm, 5% water, 1000 µm grains: dry 0.162, type 3
  0.166, coated 0.157). The coated-sphere path is runtime Mie at ~24 s per
  spectrum, so an emulator needs a (radius, water-fraction) LUT built first.
  For scale, `scripts/experiments/melt_surface_lwc_experiment.py` shows that
  even granting `layer_type=3`, free water content moves the Smith SWIR misfit
  by 5.5% — the spectrum barely constrains it through that representation.
  That is a statement about identifiability, not evidence that the surface
  is dry.
- ~~**Surface-layer thickness in FYI_summer.**~~ **Done 2026-11-12.** The SSL
  was a fixed 5 cm. It is now parameterised by `tau_ssl` = thickness ÷ grain
  radius (1–100, log-sampled), with `ssl_grain_radius` kept and thickness
  reported as a derived quantity. The form follows from the optics rather
  than from a fit: scattering optical depth goes as ρ·dz/(ρ_ice·r), so
  thickness and grain size trade off and the spectrum constrains their ratio;
  grain size keeps a separate handle through absorption per scattering event;
  and density is *exactly* interchangeable with thickness (only ρ·dz enters,
  verified to machine precision), so it is not a third parameter. No clip is
  applied, so no part of the range is inert. FYI_summer is now 7-parameter
  (held-out R² 0.9999). What moved, for the record and not as justification:

  | | Before | After |
  |---|---|---|
  | SHEBA winner-fit VIS RMS, median | 0.017 | 0.006 |
  | Smith VIS / full-range RMS, median | 0.017 / 0.027 | 0.008 / 0.017 |
  | Smith classified FYI_summer or FYI_snow | 27/44 | 34/44 |
  | SHEBA summer, full / S2 / L8 | 16/16, 16/16, 16/16 | 16/16, 16/16, 15/16 |
  | SHEBA SWIR experiment, default method VIS+SWIR | 9/12 | 12/12 |
  | SHEBA SWIR experiment, OE + model error, VIS | 11/12 | 9/12 |
  | SHEBA parameters at a bound (spectral OE) | 5/92 | 21/103, 9 genuine |
  | Model-error held-out reduced χ² (Istomina), mean / median | 3.7 / 2.4 | 7.4 / 5.0 |

  Two of those rows are new problems, below.
- **`ssl_grain_radius` now sits on its 500 µm floor in 6 of 23 SHEBA
  retrievals** (averaging kernel 0.83, so the bound is setting the value).
  With thickness free, the fits ask for a thin layer of finer grains. Whether
  the floor should move is a physical question — what optical grain size a
  real SSL has, and where FYI_summer ends and coarse snow begins — and has not
  been settled by lowering it until the pins go away.
- **The model-error covariance under-covers, and a better-fitting model makes
  it worse.** It is calibrated on post-fit residuals of the winning model on
  SHEBA and Smith. A more flexible FYI_summer halves those residuals, so the
  fitted covariance tightens (effective DOF over the VIS 2.7 → 7.5) while the
  frozen test set's misfit does not shrink — held-out reduced χ² rises from
  3.7 to 7.4. This is the known weakness of calibrating `S_e` from in-sample
  post-fit residuals (the optimiser absorbs misfit into the parameters), now
  visible. The fix is methodological — calibrate on residuals of spectra not
  used to develop the model, or inflate by the fitted degrees of freedom —
  and it matters more than any single physics item for whether OE
  uncertainties can be believed.
- ~~**`V1` — freeze the independent test corpus.**~~ Done; see above.
- ~~**The default-behaviour decision.**~~ **Taken 2026-11-12:** spectral `method="oe"` now defaults to the calibrated covariance
  (`model_error="auto"`). The coordination worry no longer applies:
  BioSNICAR is not currently served in IceNav, and in any case `model_error`
  is refused in band mode, so the default resolves to off there. On the
  SHEBA SWIR experiment the default now gives 11/12 VIS-only and 8/12 with
  SWIR; instrument-only gives 7/12 and 8/12.
- **OE uncertainties on a degenerate ridge are noisy.** Seen while testing
  the above: the finite-difference Jacobian of a ReLU emulator is piecewise
  constant, so along a |r| ≈ 0.98 ridge a 1% move in the solution can change
  a marginal σ by 2×. Point estimates are stable; error bars on degenerate
  components are not. Another reason to report the constrained combination.

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
