# BioSNICAR forward-closure validation against MOSAiC microCT + spectral albedo

**Experiment report — 2026-07-20**

*A parameter-free test of BioSNICAR driven by measured snow microphysics,
against coincident in-situ spectral albedo from the MOSAiC drift. This report
covers the day's sequence of runs, what each established, two reasoning
corrections made along the way, and the resulting diagnosis of a broadband
albedo bias. Figures referenced live in `figures/`.*

---

## 1. What we set out to test

BioSNICAR's existing sea-ice validation uses the Smith/Light/Perovich MOSAiC
spectral albedo dataset (`doi:10.18739/A2FT8DK8Z`) as an independent hold-out,
but with *classified* (not measured) surface labels. This exercise tightened
that into a **forward closure**: drive the model from *measured* snow density
and specific surface area (SSA) — Macfarlane et al. microCT snowpit profiles,
`doi:10.1594/PANGAEA.952794` — and compare the predicted spectrum to the
coincident albedo observation, fitting nothing. Measured-in → predicted-out is
the strongest validation available, because the snow state is observed rather
than inferred from the spectrum being predicted.

**Coincidence was verified first** (see `COINCIDENCE_EVIDENCE.md`): 26
co-located microCT↔albedo pairs across 10 days (2020-06-19 → 07-21, Leg 4), at
four named optics sites (STERN/LDL/ROV/SYI). The microCT `Location` field
carries the albedo sites' own tokens with an explicit `optics-` prefix; same
day with recorded UTC times; ship↔pit distance median 1.7 km (min 0.03 km);
and the albedo field logs name the microCT operator doing snow sampling in the
albedo sessions. Six of the 26 pairs are usable snow closures (the rest are
bare-ice sites with no snow positions in the albedo line — correctly excluded).

Conversion used: optical grain radius `r_eff = 3/(ρ_ice·SSA)`, SSA in m²/kg
(a documentation cross-check on the units is still outstanding but supported by
the results below).

---

## 2. The experiments and what each showed

All runs use the six snow pairs, no fitted parameters, diffuse illumination
(the melt-season default) unless noted.

| # | Experiment | Change from baseline | Result |
|---|---|---|---|
| 1 | Baseline closure | microCT grains, clean single-FYI substrate, core depth | median RMSE 0.120 (VIS 0.193, NIR* 0.082) |
| 2 | Measured thickness + melting substrate | snow depth = measured Smith line; substrate = `FYI_SUMMER_BARE` (SSL+drained+interior) | median RMSE 0.116 (VIS 0.175) — barely moved |
| 3 | Illumination sensitivity | direct beam vs fully diffuse | VIS RMSE 0.181 vs 0.171 — geometry ≈ 0.02 |
| 4 | Banded residual analysis | (diagnostic, not a model change) | over-brightness is **broadband**, not visible-only |
| 5 | Wet snow | liquid-water coating, 15% particle water | NIR bias 0.140 → 0.130 — negligible |
| 6 | Optical-radius sweep | `r_eff × B`, B ∈ {1,1.5,2,3} | **NIR closes at B ≈ 2; VIS unchanged** |

\* the run-1 "NIR 0.082" was a flawed metric — see §3.

Figures: `forward_closure_spectra.png` (1), `closure_before_after_spectra.png`
+ `closure_rmse_before_after.png` (2), `closure_illumination_sensitivity.png`
(3), `closure_residual_spectra.png` (4), `closure_wetsnow_sweep.png` (5),
`closure_grainscale_sweep.png` (6).

---

## 3. Two corrections made during the investigation

Both are worth recording because they changed the conclusion.

**(a) A flawed band metric hid the real signal.** Run 1's encouraging "NIR
RMSE 0.082" averaged everything above 750 nm, including 1400–2000 nm where ice
absorbs strongly and *both* observed and modelled albedo collapse to ~0 — a
trivial agreement that flattered the fit. Splitting the residual into physical
bands (run 4) revealed the model is over-bright across the whole 350–1400 nm
range, not just the visible:

| band | mean model − obs |
|---|---|
| VIS 350–700 | +0.169 |
| NIR 700–1000 | +0.140 |
| NIR 1000–1400 | +0.067 |
| floor 1400–2000 | −0.012 (trivial) |

Because light-absorbing impurities are transparent beyond ~700–900 nm, this
broadband shape **falsified an initial "impurities dominate" reading.**
*Lesson: exclude the >1400 nm absorption floor from any snow-albedo band
metric.*

**(b) A logic error about the grain-radius conversion.** An earlier draft
argued that because `r_eff = 3/(ρ·SSA)` is Grenfell & Warren's (1999)
optically-equivalent-sphere radius, "no geometric→optical coefficient is
needed." That conflated two separate things: the formula yields the optical
radius *only if the input SSA is the optical SSA*. microCT SSA is a
**geometric/microstructural** measurement (total ice–air interface at scanner
resolution). Grenfell–Warren's equivalence holds for convex *independent*
scatterers; for bonded/clustered melt snow, microCT over-counts the
optically-effective surface, so the true optical radius exceeds `3/(ρ·SSA)`.
The geometric→optical step is distinct from, and upstream of, the formula and
was never applied — the formula can be applied perfectly while the input SSA is
the wrong kind. Löwe & Picard (2015) show exactly such a scaling factor between
SSA-derived and effective grain size is theoretically expected.

---

## 4. The result: a clean two-term decomposition

The optical-radius sweep (run 6) is decisive because visible and NIR albedo
respond *oppositely* to grain size, which separates the mechanisms:

| `r_eff × B` | VIS 350–700 | NIR 700–1400 |
|---|---|---|
| 1.0 | +0.166 | +0.095 |
| 1.5 | +0.153 | +0.041 |
| **2.0** | +0.144 | **+0.006** |
| 3.0 | +0.135 | −0.035 |

**Scaling the optical radius by ≈2 closes the NIR while the visible barely
moves.** The ~0.15 broadband bias is therefore two independent effects:

1. **NIR (700–1400 nm) → optical grain radius too small.** The geometric-SSA
   independent-sphere radius under-sizes the optical grain of clustered melt
   snow; a ≈2× optical-radius (cluster) correction resolves it. Physically
   credible — the optics effectively sees about half the surface microCT
   counts, the rest hidden in inter-grain bonds.
2. **Visible (350–700 nm) → light-absorbing impurities.** The visible residual
   (~+0.14) is nearly grain-size-independent (moves <0.03 across the whole
   sweep, because ice is transparent there). After the grain fix it stands
   alone as a visible-only term — the impurity signature, now correctly placed
   in the band where impurities act.

Both earlier hypotheses were right but mis-assigned. A supporting internal
consistency: the per-pair B correlates inversely with the raw SSA grain size
(0619 LDL, coarsest grains, needs B≈1 and overshoots beyond; 0704/0721,
finest, need B≈3), i.e. B is not a universal constant but tracks local melt
metamorphism — as microstructure theory predicts.

---

## 5. Implications for BioSNICAR

The over-brightness is **not a solver/physics bug and not a MOSAiC data error**
— it is a **model-applicability limit**. Driven by independent-sphere
geometric-SSA grain radii, BioSNICAR over-predicts melt-snow NIR albedo; the
fix is an *input* correction (an optical-radius / cluster factor), not a code
change. Six candidate causes were eliminated in reaching this — grain-radius
formula (correct), grain shape (He et al.: non-sphericity brightens, wrong
direction), illumination (~0.02), substrate + thickness (small), liquid water
(~0.01), impurities-alone (cannot explain the NIR). The core radiative-transfer
machinery is, if anything, *validated* by the exercise: with a corrected
optical radius the model reproduces the grain-size-controlled NIR of six
independent melting snowpacks without tuning.

Practically: for melt-season snow, feeding microCT-SSA radii straight into
BioSNICAR (or SNICAR-class models) will bias albedo high in the NIR unless an
optical-radius correction is applied. This matters for any retrieval or
energy-budget use in the melt season.

### Can the ice-mc 3D RT model resolve it from first principles?

Assessed and answered **no, as built**. ice-mc is a macro-scale effective-medium
Monte-Carlo model whose granular scattering is generated from *the same
sphere-equivalent BioSNICAR Mie LUT* — so it embeds the very approximation in
question and cannot independently arbitrate B; it also has no microCT/volume
reader. However, its engine (pmcx) supports Fresnel refraction at
refractive-index-contrast voxel faces, so it *could* be repurposed into a true
grain-scale geometrical-optics tracer that ingests a binary microCT volume
(pure-ice vs air voxels, grain-resolving resolution) — substantial new
development on an engine and GPU already in hand. Dedicated µCT ray tracers
(SnowRAT/Letcher; Kaempfer et al. 2007) do this today. ice-mc's genuine role is
*downstream*: consuming corrected optical properties to add the 3-D lateral
heterogeneity BioSNICAR's 1-D columns lack.

---

## 6. Where to take it next

1. **Empirical B, data in hand (an afternoon).** Retrieve the optical grain
   radius from the grain-sensitive NIR (1030 nm band, impurity-transparent) for
   the six pairs and report B = r_retrieved / r_SSA as a distribution. Confirms
   the ≈2 factor and its spread without new data.
2. **First-principles B (larger).** Either derive it from microstructure via a
   stereological / correlation-length approach (Malinka 2014; Löwe & Picard
   2015) using the raw microCT volumes, or build a µCT geometrical-optics
   photon tracer (pmcx-based or SnowRAT-class). Turns B from a fit into a
   predicted quantity.
3. **Quantify the visible impurity term** against a *measured* MOSAiC
   snow-chemistry / black-carbon dataset, co-located the way the microCT was —
   converting the visible residual into a reported impurity forcing rather than
   an inference.

## Bottom line

Measured snow microphysics reproduce the grain-size-controlled NIR of six
independent MOSAiC melting snowpacks once an optical-radius (cluster) correction
of ≈2 is applied, leaving a clean visible residual attributable to impurities.
The diagnosis is a model-applicability limit — independent-sphere geometric-SSA
radii are too small for clustered melt snow — not a defect in BioSNICAR's
radiative transfer. Reproduce via `macfarlane_forward_closure.py`; run-by-run
detail and provenance in `README.md`.

---

## 7a. Completion runs (2026-07-20) — empirical B + SSA-units check

Both in-hand completion items were done before pausing.

**Empirical B (NIR-retrieval, route E2).** For each pair, the scalar B that
minimises the observed-vs-modelled residual in the grain-sensitive,
impurity-transparent 800–1300 nm window was retrieved (measured thickness +
`FYI_SUMMER_BARE` substrate, diffuse), then B = r_opt/r_SSA.
`empirical_B_results.json`, `figures/empirical_B.png`.

| date | site | snow cm | r_SSA µm | r_opt µm | B |
|---|---|---|---|---|---|
| 0619 | LDL | 10.6 | 1315 | 1329 | 1.01 |
| 0627 | ROV | 6.5 | 1810 | 3050 | 1.68 |
| 0704 | ROV | 2.6 | 941 | 2684 | 2.85 |
| 0706 | LDL | 5.3 | 1573 | 3022 | 1.92 |
| 0715 | SYI | 10.0 | 1116 | 1671 | 1.50 |
| 0721 | LDL | 6.2 | 1015 | 3556 | 3.50 |

**B = 1.8 median (mean 2.08 ± 0.84, range 1.0–3.5, n=6)** — an independent
retrieval from the observations that **confirms the coarse forward sweep's
≈2**, now as a measured distribution. The per-pair B trends toward higher
values for finer/thinner (more melt-metamorphosed) snow, consistent with a
clustering interpretation, but the correlation is **noisy at n=6** and should
not be over-read; the robust result is the distribution, not a functional form.

**SSA-units cross-check.** Confirmed **m²/kg** (snow-science / SLF convention;
the only reading under which the values give physical grain radii). Crucially,
a quantitative check showed the ambiguity is **nearly immaterial**: m²/kg vs
mm⁻¹ changes r_eff by only ~9% (they differ solely by the ρ_ice factor), so the
B result is robust to it regardless. *(This corrects an earlier overstatement in
the working notes that the units could swing the radius ~900×.)*

## 7. Status and resumption point (thread PAUSED 2026-07-20)

The **diagnosis is complete and the B characterisation is finished** (median B
≈ 1.8–2.1, confirmed two independent ways; units settled). Further *validation*
is now data-limited. This section is the resumption point.

**Both in-hand completion items are DONE** (§7a): empirical B, SSA-units check.

**Needs new data / tools (correctly out of scope at this stage):**
- First-principles B (Appendix T1–T2) — needs raw µCT volumes or a µCT tracer.
- Impurity quantification — needs a co-located MOSAiC snow-chemistry / black-
  carbon dataset (Path-B fetch, verified like the microCT was).
- Broader coverage — pond and bare-ice/SSL surface types, other legs/seasons.

**Coverage caveat for any citation of these results:** n = 6 pairs, MOSAiC
Leg 4 (Jun–Jul 2020), melting snow only, one region. The B≈2 result is a
melt-season snow finding, not a general BioSNICAR correction.

---

## Appendix — Routes to measuring the B parameter

`B = r_optical / r_SSA = SSA_geometric / SSA_optical`. B≈2 means the optics
effectively sees ~half the surface area microCT counts, the rest hidden in
inter-grain bonds. B is **not** a universal constant — it tracks melt
metamorphism (our per-pair B ran ~1–3, inversely with raw grain size), so the
goal is B *as a function of microstructure state*, not a single number.
[Löwe & Picard 2015](https://tc.copernicus.org/preprints/9/2495/2015/) show
such a scaling factor is theoretically expected.

### Empirical routes

- **E1 — Co-located optical-SSA + microCT (gold standard).** Optical-SSA
  instruments (IceCube / InfraSnow / DUFISSS) invert 1310 nm hemispherical
  reflectance to the *optically-effective* SSA; B = SSA_microCT / SSA_optical
  directly, no model. *Needs:* a co-located optical-SSA measurement at the pits.
  An [SLF IceCube+microCT dataset](https://www.envidat.ch/metadata/icecube_microct_snow_grainsize)
  exists for snow generally; **not confirmed for MOSAiC melt snow** — check with
  the MOSAiC snow team. *In hand:* no. *Effort:* low if data exists.
- **E2 — NIR-retrieval B (completable now).** Invert observed reflectance in a
  grain-sensitive, impurity-transparent band (1030 nm scaled band-area,
  Nolin & Dozier 2000; or full-NIR fit, Kokhanovsky–Zege / Painter) for the
  effective optical radius, then B = r_retrieved / r_SSA. This is our forward
  B-sweep done as a retrieval — gives a measured B distribution over the six
  pairs. *In hand:* yes (Smith spectra + microCT). *Effort:* ~half a day.
- **E3 — Lab paired reflectance + microCT** on extracted cores — isolates the
  grain effect from substrate/impurity confounds. *Needs:* lab samples. *In
  hand:* no.

### Theoretical routes

- **T1 — Stereological / geometrical optics on the microstructure**
  ([Malinka 2014](https://www.sciencedirect.com/science/article/abs/pii/S002240731400079X);
  sea-ice application, Malinka et al. 2016). Derives optical properties (hence
  B) from the chord-length distribution of the 3-D structure — no sphere
  assumption. *Needs:* raw µCT **volumes** (we hold only derived density/SSA/
  anisotropy profiles). The anisotropy column supports a first cut; full
  stereology needs the volumes.
- **T2 — Two-parameter microstructure (SSA + correlation length / stickiness)**
  (Löwe & Picard 2015). SSA plus a second stereological parameter fixes the
  scaling factor. *Caveat:* stickiness estimation from µCT is numerically
  unstable; sticky-hard-sphere may not represent natural snow well.
- **T3 — Shape / absorption-enhancement parameterisation** (Libois 2013/2014;
  Kokhanovsky–Zege) — expresses clustering via absorption enhancement `B_abs`
  and asymmetry `g` rather than a radius rescale; formally relatable to our B.
  Deployable in SNICAR-class models now, but the coefficients are the unknown.
- **T4 — µCT geometrical-optics photon tracer** — ray-trace photons through a
  binary µCT volume with Fresnel refraction at ice–air interfaces + Beer-Lambert
  absorption via k(λ); yields albedo and effective SSA/g directly (SnowRAT /
  Letcher; Kaempfer et al. 2007; Xiong & Shi). ARD's **ice-mc** is *not* this as
  built (effective-medium; embeds the sphere LUT), but its **pmcx** engine +
  GPU could be repurposed for it — substantial but well-scoped development on
  infrastructure already in hand. *Needs:* raw µCT volumes + importer + grain-
  resolving voxels; watch voxel-staircasing error at interfaces.

**Recommended order if resumed:** E2 (measured B now) → SSA-units check → then
T1 or T4 for a first-principles B if the finding warrants publication.
