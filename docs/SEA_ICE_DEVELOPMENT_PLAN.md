# Sea Ice Retrieval — Development Plan & Roadmap

**Status:** planning · **Date:** 2026-06-14 · **Branch context:** `feature/sea-ice-mvp`

This plan develops the sea ice optical model and inversion (`biosnicar.sea_ice`)
under a **dual mandate**:

1. **A best-in-class standalone application** for retrieving sea-ice surface
   type and physical characteristics from optical reflectance — optimised for
   maximum performance in its own right, across **two first-class data
   modalities**: full **hyperspectral** inversion (field spectrometer, drone
   imager, hyperspectral satellite) and **satellite multispectral** inversion
   (Sentinel-2/3, Landsat, MODIS, PlanetScope, VIIRS).
2. **An ensemble-ready evidence stream** that later contributes calibrated,
   fusable likelihoods to a multi-sensor ice-classification system supplemented
   by SAR, meteorological reanalysis, passive microwave, ice-age/drift, etc.

It is grounded in the validated state and the limitations surfaced by the
2026-06 audit and the Smith/MOSAiC independent hold-out (see
[sea_ice_validation.md](sea_ice_validation.md) §6.4–6.5,
[SEA_ICE_RETRIEVAL.md](SEA_ICE_RETRIEVAL.md)).

---

## 1. North star: one architecture, two mandates, two modalities

The two mandates look like they pull in different directions (standalone wants
to use every scrap of information including met-data priors; the ensemble must
not double-count evidence it already holds). They are reconciled by a single
design choice:

> **Compute and expose the spectrum-only likelihood and the prior contribution
> separately, always.** Standalone mode consumes the full posterior (priors on);
> ensemble mode consumes the spectrum-only likelihood and lets the fusion layer
> own the met/SAR/age evidence. Same code, a toggle and a provenance vector.

The **methodological spine** that delivers this is **optimal estimation (OE) /
Bayesian retrieval** (Rodgers framework) per surface-type forward model, with
**Bayesian model selection** across types:

- **Per class:** OE retrieval → parameter posterior (mean + covariance),
  marginal likelihood, **averaging kernels** and **degrees of freedom for
  signal (DFS)**.
- **Across classes:** posterior probability over surface types from the marginal
  likelihoods (with optional class priors).

This one framework yields, for free, everything both mandates need:

| Need | OE product that supplies it |
|---|---|
| Calibrated uncertainty (ensemble) | posterior covariance |
| Spectral-vs-prior provenance (ensemble + honesty) | averaging kernel / gain matrix — literally "how much came from the data" |
| What is retrievable from *this* band set (modality) | DFS / information content |
| Standalone accuracy | MAP estimate using all available priors |
| Fusable output (ensemble) | per-class marginal likelihood with priors *off* |

The emulators make this cheap: MLP Jacobians are analytic/finite-difference in
microseconds, so OE + model selection runs at scene scale.

### Design principles
- **Probabilistic, not hard-label:** emit per-class posteriors and parameter posteriors; argmin is a reporting convenience, never the interface.
- **Information-content-aware:** retrieve only the parameters the observation's DFS supports; lean on priors for the rest — and *say which is which*.
- **Modality-first:** hyperspectral and multispectral are distinct regimes with distinct information content, error budgets, and validation — not one path with the other bolted on.
- **Physically grounded:** keep the forward model; priors are external evidence, not fitting freedom.
- **Regime-aware:** report per season/melt-state; never a single headline number.
- **Honest by construction:** provenance, quality flags, DFS, and known degeneracies are first-class outputs.
- **Reproducible:** versioned emulators/LUTs, deterministic rebuilds, committed result manifests.

---

## 2. The two modalities as a first-class axis

| | **Hyperspectral** | **Satellite multispectral** |
|---|---|---|
| Sources | ASD/field (350–2500 nm), drone imagers, EnMAP/PRISMA/EMIT (hyperspectral satellites) | Sentinel-2/3, Landsat 8/9, MODIS, PlanetScope, VIIRS |
| Bands | 100s contiguous, full SWIR, narrow features | 4–13 broad bands, often VIS–NIR only |
| Information content (DFS) | high — retrieve grain size, LWC, impurities, thickness | low — often only 1–3 DOF; classification + 1–2 params |
| Dominant error source | instrument noise, surface heterogeneity | **atmospheric correction**, band-set limits, mixed pixels |
| Degeneracy exposure | lower (narrow features separate parameters) | higher (broad bands + few of them) |
| Prior reliance | light | heavy — priors do real work when bands are few |
| Atmospheric handling | usually surface reflectance already | **TOA→surface is a hard dependency** |

**Implications baked into the plan:**
- Retrieval is **band-set-driven**: the same OE engine, but the number of
  retrieved parameters is set by DFS, not hardcoded — hyperspectral retrieves a
  rich state vector; a 4-band PlanetScope scene retrieves a class + one or two
  parameters and reports the rest as prior-dominated.
- The **satellite path needs an atmospheric-correction contract** (ingest L2A
  surface reflectance, or couple an atmospheric model) — without it, satellite
  "performance" is untrustworthy regardless of the optical model.
- **Per-modality validation and benchmarking** — a single accuracy number
  across modalities is meaningless.
- Hyperspectral satellites (EnMAP, EMIT, PRISMA; future SBG/CHIME) are the
  bridge case — global-ish coverage at high spectral resolution — and a
  priority deployment target because they get the most out of the physics.

---

## 3. Current state (honest baseline)

- **Validated (spring/cold regime, Arctic, VIS–NIR):** SHEBA spring snow 6/7, summer bare ice 16/16 (inferred labels); FYI_bare emulator held-out spectral R² 0.998.
- **Only independent ground-truth metric:** Morassutti pond depth (~22% within ±25% over the full 504-record run; reliable only >50 cm).
- **Independent hold-out (Smith/MOSAiC, full SWIR):** summer melting snow is optically degenerate with SSL/bare ice; fit RMS ~doubles out of distribution. Spring numbers do **not** generalise to the melt season.
- **Synthetic metrics are inverse-crime numbers** (same forward model for generation and inversion).
- **Untested:** non-Arctic surfaces; open water and young ice vs real spectra (synthetic-only); scene-scale performance; atmospheric/TOA coupling; BRDF/angular effects; confidence calibration; drone and hyperspectral-satellite modalities.
- **Deferred physics:** ~~linear liquidus~~ ✅ fixed (D1: liquidus now derived from Frankenstein & Garner 1967 brine volume, ~30 % cold-ice error removed); still pending: no liquid-water content in melting-surface forward models.

---

## 4. Development workstreams and priority tasks

Ordered within each stream by priority; **[gate]** marks a dependency for other streams.

### A. Inversion core — optimal estimation & probabilistic output  *(spine; highest priority)*
> **Status (2026-06-22):** A1, A2, A3 ✅ done; A4–A8 pending.
- **A1 [gate]** ✅ **DONE** — OE engine (`biosnicar/inverse/optimal_estimation.py`, `method="oe"`): parameter posterior (mean+cov), Jacobians, averaging kernels, **DFS**, Laplace evidence. Linear-Gaussian-exact, tested.
- **A2 [gate]** ✅ **DONE**, with a measured limitation — `retrieve_sea_ice(method="oe")` classifies by per-type posterior probability from the model evidence (`class_probabilities`). **2026-07-15 SHEBA SWIR experiment** (`oe_swir_classification_experiment.py`, validation doc §12): on real melt-season spectra the evidence ranking is unreliable (7/12 VIS-only, 2/12 VIS+SWIR vs the default masked method's 12/12 in both) and maximally overconfident (wrong class at p≈1.00). **Mode-dependent (2026-07-17 update):** the pathology is a many-correlated-bands artefact. **Spectral OE** is repaired by A7 `model_error=True` (VIS-only 58%→92%); **band-mode OE** (≤~9 satellite bands) never triggered it — the band-mode experiment (`oe_bandmode_classification_experiment.py`) shows OE summer 15/16 on Sentinel-2's 9 VIS-NIR bands, beating the default 13/16, at honest confidences. **The production (IceNav) path is therefore sound on this axis** — no urgent hybrid-classification change needed. Hybrid remains a candidate only for full-SWIR spectral classification. Remaining OE trust work: A6 calibration; the still-broken spectral VIS+SWIR case (needs the SWIR covariance term).
- **A3 [gate]** ✅ **DONE** — averaging kernel / DFS emitted per parameter (measurement-vs-prior split); plus `retrieve_sea_ice(use_priors=False)` for a **spectrum-only** retrieval (drops the `known_month` season priors, their per-emulator Vb translation, and the melt-season young-ice exclusion), and `flag_prior_influence=True` which runs a spectrum-only shadow pass and sets `prior_resolved` (True when the metadata prior changed the winning class) with `spectrum_only_surface_type` / `spectrum_only_class_probabilities` for transparency. Per-parameter provenance via `result.prior_dominated_parameters()`. (`biosnicar/sea_ice/retrieve.py`)
- **A4** Output data contract (serves both mandates): `{class: marginal_loglik}`, parameter posteriors, DFS, averaging-kernel summary, quality bitmask, provenance vector. *(result objects carry most of this; the formal contract/spec is not yet frozen.)*
- **A5** **Information-content-aware retrieval:** choose the retrieved sub-state from DFS per observation; report prior-dominated parameters as such.
- **A6** Confidence **calibration** against held-out data (reliability diagrams); posteriors must mean what they say.
- **A7** ◐ **step 1 done (2026-07-17)** — Per-band error model (instrument + **forward-model covariance**) replacing the band-mask rescaling heuristic; band masks become weights. *(blocks trustworthy OE uncertainties — `S_e` is instrument-noise-only today.)* **Field-residual library built**: `tests/validation_data/build_residual_library.py` collects post-fit residuals from all three spectral campaigns (188 spectra → 521 rows, winner + expected-type populations, regenerable artifact). First analysis confirms the mechanism behind the B6 overconfidence: VIS per-band σ median 0.018 but only **~2 effective independent DOF over 60 VIS bands** (94% of residual variance in 3 EOFs) — the current diagonal S_e over-counts evidence ~30×. Remaining: covariance model (low-rank EOF + diagonal floor, calibrate/hold-out split across campaigns), wire full-matrix S_e into `_run_oe`, re-run the OE SWIR experiment as the acceptance gate, A6-lite reliability check.
- **A8** Hierarchical/superclass reporting for degenerate clusters (`{melt bright granular}`, `{dark bare/thin ice}`) gated on ambiguity; fine classes retained for parameters.

### B. Modality support  *(co-priority with A — the standalone product is these two paths)*
**B-HS — Hyperspectral path**
- **B-HS1** Robust instrument→model-grid resampling for arbitrary hyperspectral inputs (SRF/FWHM aware; extends D3); per-instrument config (ASD, drone, EnMAP, EMIT, PRISMA).
- **B-HS2** Exploit full information: rich state-vector retrieval (grain size, LWC, impurities, thickness) where DFS supports it; narrowband feature use (liquid-water, grain-size bands).
- **B-HS3** Sub-pixel / linear mixing for heterogeneous drone & satellite-hyperspectral pixels.

**B-MS — Satellite multispectral path**
- **B-MS1 [gate]** **Atmospheric-correction contract:** ingest L2A surface reflectance with documented assumptions, or couple an atmospheric model for TOA input. Without this, satellite results are not credible.
- **B-MS2** Per-sensor band configurations and band-set-aware retrieval (DFS-limited); graceful degradation as bands drop.
- **B-MS3** Sun-glint / BRDF handling for open water and low-sun geometry (critical at high latitude).
- **B-MS4** Mixed-pixel handling at coarse resolution (MODIS/VIIRS 250 m–1 km).

### C. Metadata, priors & ensemble interface
- **C1 [gate]** Modular **metadata→prior adapter**: per-class log-priors + parameter priors from external inputs; toggleable per source so standalone uses everything and ensemble withholds what the fusion layer owns. `known_month` becomes one provider.
- **C2** **Surface/skin temperature** prior (TIR/reanalysis) → maps onto `sea_ice_temperature`; breaks the dark cluster (open water ≈ −1.8 °C vs cold bare ice) and melt-vs-frozen.
- **C3** **Ice-age/region** prior (NSIDC EASE-grid age, or geography) on the `spatial_coords` already in the batch path → breaks FYI_bare ↔ MYI_bare (spectrally unbreakable, RMS 4.7).
- **C4** Freezing-degree-day → Stefan-law thickness prior; freeze-up/polynya likelihood for young ice.
- **C5** Snow-presence prior (PMW / snow model / in-situ) — the only lever for snow ↔ SSL; coarse, lower priority.
- **C6** **Ensemble interface spec:** documented contract for emitting spectrum-only per-class likelihoods + parameter posteriors + provenance into the fusion layer; defines what this stream owns vs what the ensemble supplies (SAR, met) to avoid double-counting.

### D. Forward-model fidelity
- **D1** ✅ **DONE** — liquidus replaced: derived from the Frankenstein & Garner (1967) brine-volume relation by salt mass balance (the verifiable path; the Assur-fit Notz & Worster 2009 coefficients were paywalled). LUTs + emulators rebuilt. Removes the ~30 % cold-ice brine-salinity error. (`brine_volume.py`)
- **D2** **Liquid-water content** in the melting-surface (FYI_summer) forward model → closes the Smith SWIR misfit; makes melt-season grain/LWC retrieval meaningful.
- **D3** Independent forward-model validation vs the adding-doubling solver across **all regimes incl. cold ice** (the Cox & Weeks bug proved the suite couldn't detect a cold-regime error).
- **D4** BRDF / non-Lambertian surface and angular effects (couples to B-MS3).
- **D5** Spectral coverage to full SWIR fidelity for hyperspectral (verify brine/LUT physics across 1000–2500 nm).

### E. Empirical data collection — see §5.
### F. Validation framework — see §6.

### G. Standalone application & engineering
- **G1** Productised API + CLI + notebook examples; clear single-spectrum, batch, and scene entry points for both modalities.
- **G2** Scale benchmark at real scene/mosaic size (currently only 10-px tested); memory/timing.
- **G3** Vectorised inverse network behind `SeaIceSceneResult` for regional mosaics (engine swap, exports unchanged).
- **G4** Robustness: missing bands, noisy/atmospherically-imperfect input, out-of-range geometry, graceful failure with flags.
- **G5** Reproducibility: versioned emulator/LUT artifacts, deterministic rebuilds, result manifests; data-cube/STAC compatibility.
- **G6** Domain-expert review gate on melt-surface and young-ice physics before operational claims.

---

## 5. Empirical data collection plan

Binding constraint on credibility. Current evidence: ~23 SHEBA spectra +
Smith/MOSAiC snow, Arctic-only, one in-situ ground-truth metric.

**Targeted gaps (priority order):**
1. **Coincident spectral + structural ground truth** (spectra *with* measured snow depth, ice thickness, surface type) — validates *provenance*, which albedo alone cannot.
2. **Bare winter/cold ice spectra** (no snow, T < −15 °C) — `FYI_bare`/`MYI_bare` never validated against real spectra.
3. **Young ice / nilas / grease ice** (freeze-up campaigns) — currently zero; anchored only to Grenfell & Maykut broadband brackets.
4. **Open-water and lead spectra** — currently zero; open-water model is self-inverting.
5. **Non-Arctic (Antarctic) data** — all current data is Arctic.
6. **Satellite–in-situ matchups** in the real deployment modality (band mode, *with* atmospheric correction) — for B-MS.
7. **Hyperspectral-satellite scenes** (EnMAP/EMIT/PRISMA over sea ice) with coincident field data — for B-HS at the bridge case.
8. **Multi-angle / BRDF** measurements (D4, B-MS3).

**Mine existing archives before any field cost:** MOSAiC (multiple legs; ROV +
ASD + SUIT instruments — Arctic Data Center / PANGAEA), SHEBA, ICESCAPE, NSIDC,
AWI/PANGAEA Antarctic (SIPEX, ISPOL, AnZone), EnMAP/EMIT/PRISMA L2 archives, and
operational ice charts (NIC, AARI, DMI) for scene labels. Curate into one
versioned validation corpus with standardised metadata.

---

## 6. Validation framework (rigour, per modality and per regime)

- **V1 — Freeze an independent test corpus** never used for tuning (held-out MOSAiC leg + any Antarctic data); all headline numbers reported on it.
- **V2 — Perturbed-physics (model-mismatch) validation:** invert synthetic spectra generated with deliberately wrong physics → bounds the inverse-crime overstatement.
- **V3 — Per-modality benchmarking:** separate hyperspectral and multispectral skill; for multispectral, per-sensor and as a function of band set / DFS.
- **V4 — Per-regime validation:** cold/spring, melt-onset, peak-melt, freeze-up — reported separately.
- **V5 — Calibration validation:** reliability diagrams; posterior coverage tests.
- **V6 — Sensitivity analyses:** atmospheric-correction error (dominant for satellite), SZA beyond training, band degradation, noise level.
- **V7 — Ensemble-relevant metric:** does adding this stream measurably improve scene classification vs operational ice charts / expert labels, versus the ensemble without it.

---

## 7. Deployment milestones

- **D-A** Standalone hyperspectral app: rich retrieval + uncertainty + DFS, validated on field/drone data (mandate 1, hyperspectral).
- **D-B** Standalone satellite app: atmospheric-correction contract + per-sensor retrieval, validated on satellite–in-situ matchups (mandate 1, multispectral).
- **D-C** Probabilistic output contract frozen and documented (A1–A4).
- **D-D** Metadata-prior adapter live with provenance flagging (C1–C3); standalone uses priors, ensemble interface withholds them.
- **D-E** Scene/mosaic-scale engine (G2–G3).
- **D-F** Ensemble integration: emitting calibrated spectrum-only likelihoods into the fusion layer; V7 lift demonstrated (mandate 2).
- **D-G** Operational packaging + expert sign-off (G5–G6).

---

## 8. Phased roadmap with exit gates

Dependency-ordered; durations indicative (research, not fixed dates).

**Phase 0 — Probabilistic core & correctness** *(near term)*
A1–A4, A7, D1, D3, V1, V2.
**Gate:** OE-based per-class posteriors with provenance and DFS; ✅ liquidus fix shipped (D1); inverse-crime degradation bounded; independent test corpus frozen.

**Phase 1 — Both standalone modalities to first-class**
B-HS1–2, B-MS1–2, A5, A6, A8, G1, G4, V3, V5.
**Gate:** standalone hyperspectral *and* satellite paths validated end-to-end (incl. an atmospheric-correction contract for satellite); calibrated, DFS-aware retrieval; per-modality benchmarks published.

**Phase 2 — Priors, ensemble interface & empirical expansion**
C1–C4, C6, data collection (§5), V4, V6, D2.
**Gate:** honestly-attributed prior lift on the dark cluster / FY-MY split on real data; ensemble contract published; independent multi-campaign + ≥1 non-Arctic validation; melt-season SWIR fit improved.

**Phase 3 — Real-sensor fidelity & scale**
B-MS3–4, B-HS3, D4–D5, G2–G3.
**Gate:** BRDF/glint + mixed-pixel handling; scene/mosaic-scale benchmark; hyperspectral-satellite matchups.

**Phase 4 — Operational deployment (standalone + ensemble)**
D-F, D-G, G5–G6, V7.
**Gate:** standalone apps released; ensemble lift demonstrated; expert sign-off; reproducible deployment.

---

## 9. Risk register & kill/rescope criteria (honest)

| Risk | Likelihood | Mitigation / decision rule |
|---|---|---|
| Atmospheric correction dominates satellite error | medium-high | Quantify in V6; if it swamps the signal, satellite mandate rescopes to L2A-only / hyperspectral-satellite, and field/drone becomes the primary standalone use |
| Optical degeneracies irreducible even with priors | medium | Accept ensemble role; if SAR/thermal/age can't lift V7, value is parameter retrieval, scope classification claims accordingly |
| No coincident structural ground truth obtainable | medium | Provenance (snow vs ice) stays unvalidated → ship optical-state + parameters, not structural labels |
| Few-band satellite DFS too low for useful retrieval | medium | Report class + prior-dominated params honestly; position multispectral as classification-first, hyperspectral as retrieval-first |
| Cubic-liquidus/LWC rebuild shifts validated results | low-med | Phase-0 gate re-runs V1–V2 before accepting |
| Calibration infeasible with available data volume | medium | Report uncalibrated with explicit warning; fusion down-weights |

**Kill / rescope criteria:** if Phase 1 cannot get the satellite path past the
atmospheric-correction barrier **and** Phase 2 cannot obtain independent
validation, ship as a **hyperspectral field/drone retrieval tool** (where the
physics is strongest and best-validated) plus an ensemble likelihood emitter —
not an operational satellite classifier.

---

## 10. Immediate next actions (this branch)

Lowest-regret, highest-leverage, all serving **both** mandates:
1. ✅ **A1–A3 OE core + Bayesian classification + provenance** — DONE (`method="oe"`; `use_priors` / `flag_prior_influence` / `prior_resolved`). Remaining spine work: **A7** forward-model `S_e` term (the blocker for trustworthy OE uncertainties — needs field residuals).
2. ✅ **D1 liquidus fix** (F&G 1967-derived) — DONE; removed the ~30 % cold-ice brine-salinity error.
3. **B-MS1 atmospheric-correction contract** — without it the satellite mandate is not credible; cheapest to define early.
4. **V2 perturbed-physics validation** + **V1 freeze the MOSAiC corpus** — bound the inverse crime and stop tuning against test data, no new data needed.
5. **C1–C2 metadata adapter + temperature prior** — highest-leverage prior, and the modular toggle that lets the same code serve standalone (priors on) and ensemble (priors withheld).

These establish the probabilistic, DFS-aware, dual-modality, provenance-honest
core that both the standalone product and the ensemble depend on. The OE
engine and provenance toggle (A3) are in place; the next gating piece is the
`S_e` forward-model term (A7, needs field data).
