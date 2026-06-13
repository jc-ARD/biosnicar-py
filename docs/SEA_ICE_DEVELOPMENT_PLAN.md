# Sea Ice Retrieval — Development Plan & Roadmap

**Status:** planning · **Date:** 2026-06-13 · **Branch context:** `feature/sea-ice-mvp`

This plan develops the sea ice optical inversion (`biosnicar.sea_ice`) toward
real-world use as **one line of evidence in a multi-sensor ensemble** that
classifies sea-ice scenes and retrieves ice characteristics. It is grounded in
the current validated state and the limitations surfaced by the 2026-06 audit
and the Smith/MOSAiC independent hold-out (see
[sea_ice_validation.md](sea_ice_validation.md) §6.4–6.5,
[SEA_ICE_RETRIEVAL.md](SEA_ICE_RETRIEVAL.md)).

---

## 1. North star and what changes because of the ensemble framing

The module's job is **not** to be the final classifier. It is to convert an
observed reflectance (spectrum or satellite bands) into a *physically grounded,
calibrated probabilistic statement* — per-class likelihoods plus parameter
posteriors with uncertainty — that a downstream fusion layer combines with SAR,
thermal, passive microwave, ice-age/drift products, and operational ice charts.

Three consequences reorder the priorities versus a standalone classifier:

1. **Calibrated likelihoods beat headline accuracy.** A fusion layer needs
   outputs that mean what they say (a 0.7 should be right ~70% of the time).
   Today's `confidence = (2nd−best)/2nd` is an uncalibrated heuristic. Fixing
   this outranks adding classes or chasing accuracy points.
2. **Honest provenance is a hard requirement.** The ensemble must know whether
   a label came from the *spectrum* or from a *prior/metadata*, or it will
   double-count evidence (e.g. weight an ice-age prior twice if this module
   already folded it in). Outputs must separate spectral likelihood from
   prior contribution.
3. **Degeneracies are acceptable if reported as such.** The optical
   degeneracies we found (melting snow ↔ SSL; FYI_bare ↔ MYI_bare ↔ young ice ↔
   open water) don't need to be solved *within* this stream — they are exactly
   what complementary streams (SAR, thermal, ice age) resolve. This stream must
   report the ambiguity cleanly rather than force a confident pick.

### Design principles
- **Probabilistic, not hard-label:** emit posteriors, never just argmin.
- **Physically grounded:** keep the forward model; priors are external evidence, not curve-fitting freedom.
- **Regime-aware:** performance is season/melt-state dependent — validate and report per regime, never as a single headline number.
- **Honest by construction:** spectral vs prior-resolved provenance, quality flags, and known degeneracies are first-class outputs.
- **Reproducible:** versioned emulators/LUTs, deterministic rebuilds, committed result manifests.

---

## 2. Current state (honest baseline)

- **Validated (spring/cold regime, Arctic):** SHEBA spring snow 6/7, summer bare ice 16/16 (inferred labels); FYI_bare emulator held-out spectral R² 0.998.
- **The only independent ground-truth metric** is Morassutti pond depth (24% within ±25%).
- **Independent hold-out (Smith/MOSAiC):** summer melting snow is optically degenerate with SSL/bare ice (5% → FYI_snow); fit RMS roughly doubles out of distribution. Spring numbers do **not** generalise to the melt season.
- **Synthetic metrics (E1, demo) are inverse-crime numbers** — same forward model for generation and inversion.
- **Untested:** any non-Arctic surface; open water and young ice against real spectra (both currently synthetic-only); scene-scale performance (>10 px never benchmarked); atmospheric / TOA coupling; BRDF/angular effects; confidence calibration.
- **Known deferred physics:** linear liquidus (~31% brine-salinity error at −10 °C); no liquid-water content in melting-surface forward models.

---

## 3. Development workstreams and priority tasks

Ordered within each stream by priority; **[gate]** marks a task another stream depends on.

### A. Probabilistic inversion & output contract  *(highest priority — unblocks the ensemble)*
- **A1 [gate]** Replace argmin-χ² classification with proper Bayesian model selection → normalised **per-class posterior probabilities**. Define the output data contract the ensemble consumes: `{class: log_likelihood}`, parameter posteriors (mean+cov), quality bitmask, provenance vector.
- **A2 [gate]** **Provenance separation:** compute and emit the classification both with flat priors (spectrum-only) and with metadata priors, and flag when they disagree (`prior_resolved`). Never let a prior silently flip a label without saying so.
- **A3** Principled likelihood: replace the band-mask rescaling heuristic with a per-band noise model (instrument + forward-model error covariance). Band masks become down-weighting, not hard cuts.
- **A4** Confidence **calibration**: reliability diagrams; calibrate posteriors against held-out data so probabilities are meaningful to the fusion layer.
- **A5** Hierarchical / superclass reporting for degenerate clusters (`{melt bright granular}`, `{dark bare/thin ice}`), gated on `SPECTRALLY_AMBIGUOUS`; fine classes retained for parameter retrieval.
- **A6** End-to-end uncertainty propagation (obs noise + forward-model error → parameter posterior).

### B. Metadata → priors (the lever the spectrum lacks)
- **B1 [gate]** Generic **metadata→prior adapter**: produces per-class log-priors + parameter regularisation from external inputs; `known_month` becomes one provider. Clean interface so the ensemble can supply or withhold each stream.
- **B2** **Surface/skin temperature** prior (TIR or reanalysis) → maps onto existing `sea_ice_temperature`; breaks the dark cluster (open water ≈ −1.8 °C vs cold bare ice ≪ 0) and melt-vs-frozen. Highest leverage, lowest friction.
- **B3** **Ice-age / region** prior (NSIDC EASE-grid age, or geography) attached to the `spatial_coords` already in the batch path → breaks FYI_bare ↔ MYI_bare (spectrally unbreakable, RMS 4.7).
- **B4** Freezing-degree-day → Stefan-law thickness prior for young ice; freeze-up region/polynya likelihood.
- **B5** Snow-presence prior (passive microwave / snow model / in-situ) — the *only* lever for snow ↔ SSL; coarse and hardest, lower priority.

### C. Forward-model fidelity
- **C1** **Cubic liquidus** (Assur) replacing the linear form; rebuild LUTs + emulators. Removes the known ~31% brine-salinity error in the validated regime.
- **C2** **Liquid-water content** in the melting-surface (FYI_summer) forward model → closes the Smith SWIR misfit (p90 RMS 0.077) and makes melt-season grain/LWC retrieval meaningful (won't change classification — that's degenerate — but fixes the parameters).
- **C3** Independent forward-model validation against the adding-doubling solver across **all regimes incl. cold ice** (the Cox & Weeks bug proved the validation suite couldn't detect a cold-regime physics error).
- **C4** Atmospheric coupling: define and implement the TOA→surface contract (real satellite input is TOA; current model assumes surface reflectance). Either ingest an atmospheric correction or document the hard dependency.
- **C5** BRDF / angular effects (open-water sun glint, surface anisotropy) for real sensor geometry; multi-angle support.

### D. Empirical data collection — see §4 (the largest credibility gap).

### E. Validation framework — see §5.

### F. Deployment & engineering
- **F1** Scale benchmark at real scene/mosaic size (timing + memory); currently only 10-px tested.
- **F2** Vectorised inverse network behind the existing `SeaIceSceneResult` interface for regional scale (already designed-for; swap engine without changing exports).
- **F3** Ensemble integration API + data-cube/STAC compatibility; versioned, reproducible emulator/LUT artifacts with a result manifest.
- **F4** Domain-expert review gate on the melt-surface and young-ice physics before any operational claim.

---

## 4. Empirical data collection plan

This is the binding constraint on credibility. Current evidence: ~23 SHEBA
spectra + Smith/MOSAiC snow, Arctic-only, one in-situ ground-truth metric.

**Targeted gaps (priority order):**
1. **Coincident spectral + structural ground truth** — spectra *with* measured snow depth, ice thickness, and surface type. This is what lets us validate *provenance* (snow vs ice), not just optics — the thing albedo alone cannot give.
2. **Bare winter/cold ice spectra** (no snow, T < −15 °C) — `FYI_bare`/`MYI_bare` have *never* been validated against real spectra; the FY/MY split is unverified empirically.
3. **Young ice / nilas / grease ice spectra** — currently zero; the model is anchored only to Grenfell & Maykut (1977) broadband brackets. Freeze-up campaigns (autumn/early winter).
4. **Open-water spectra** — currently zero; the open-water model is fully self-inverting. Need real Arctic/Antarctic open-water and lead reflectance.
5. **Non-Arctic (Antarctic) data** — all current data is Arctic; generalisation is untested. Antarctic ice has different salinity/snow regimes.
6. **Satellite–in-situ matchups** — coincident Sentinel-2/Landsat/MODIS surface reflectance with field measurements, in the actual deployment modality (band mode, atmospheric correction included).
7. **Multi-angle / BRDF** measurements for glint and anisotropy (C5).

**Candidate archives to mine before any new field cost:** MOSAiC (multiple legs,
ROV + ASD + SUIT instruments; Arctic Data Center / PANGAEA), SHEBA, ICESCAPE,
NSIDC, AWI/PANGAEA Antarctic campaigns (SIPEX, ISPOL, AnZone), and operational
ice-chart archives (NIC, AARI, DMI) for scene-level labels. Curate into a single
versioned validation corpus with standardised metadata (date, lat/lon, SZA, sky,
surface type, structural measurements).

---

## 5. Validation milestones (rigour framework)

The goal is to convert today's optimistic, partly-circular numbers into
defensible, regime-stratified, ensemble-relevant skill estimates.

- **V1 — Freeze an independent test corpus** never used for tuning (start: a held-out MOSAiC leg + any Antarctic data). All headline numbers reported on it.
- **V2 — Perturbed-physics (model-mismatch) validation:** generate synthetic spectra with deliberately wrong physics, invert with the nominal model → bounds the inverse-crime overstatement; converts "R² 0.99 under perfect physics" into a real degradation estimate.
- **V3 — Per-regime validation:** cold/spring, melt-onset, peak-melt, freeze-up — reported separately, never as one headline (performance is demonstrably regime-dependent).
- **V4 — Calibration validation:** reliability diagrams; posteriors must be calibrated to be fusable (ties to A4).
- **V5 — Sensitivity analyses:** SZA beyond training bound, atmospheric-correction error, band-set degradation, noise level.
- **V6 — Ensemble-relevant metric:** does adding this stream measurably improve scene classification against operational ice charts / expert labels, versus the ensemble without it? This is the only metric that matters for the stated end use.

---

## 6. Deployment milestones

- **D1** Probabilistic output contract frozen and documented (depends A1–A2).
- **D2** Metadata-prior adapter live with provenance flagging (B1–B3).
- **D3** Scene-scale benchmark passed; vectorised engine for regional mosaics (F1–F2).
- **D4** TOA→surface pipeline or explicit, validated surface-reflectance contract (C4).
- **D5** Ensemble integration: this stream emitting calibrated likelihoods into the fusion layer; V6 lift demonstrated.
- **D6** Operational packaging (containers/cloud, reproducible artifacts) + domain-expert sign-off (F4).

---

## 7. Phased roadmap with exit gates

Phases are dependency-ordered; durations indicative (research, not fixed dates).

**Phase 0 — Correctness & honest probabilistic core** *(near term)*
Tasks: A1, A2, A3, C1, C3, V1, V2.
**Exit gate:** calibrated per-class posteriors with spectral/prior provenance; cubic liquidus shipped; inverse-crime degradation bounded; an independent test corpus frozen.

**Phase 1 — Metadata priors & ensemble interface**
Tasks: B1, B2, B3, A4, A5, V4, D1.
**Exit gate:** demonstrated, *honestly-attributed* lift on the dark cluster and FY/MY split from temperature + ice-age priors on real data; calibrated, fusable output contract published.

**Phase 2 — Empirical expansion & regime validation**
Tasks: D-stream (data collection §4), V3, V5, C2.
**Exit gate:** independent multi-campaign validation including ≥1 non-Arctic dataset; per-regime skill documented; melt-season SWIR fit improved; young-ice/open-water/bare-winter gaps at least partially filled with real spectra.

**Phase 3 — Real-sensor fidelity & scale**
Tasks: C4, C5, F1, F2.
**Exit gate:** TOA→surface pipeline validated on satellite–in-situ matchups; scene/mosaic-scale benchmark passed.

**Phase 4 — Operational ensemble deployment**
Tasks: D5, D6, F3, F4, V6.
**Exit gate:** measurable improvement in operational scene classification vs the ensemble-without-this-stream; expert sign-off; reproducible deployment.

---

## 8. Risk register & kill criteria (honest)

| Risk | Likelihood | Mitigation / decision rule |
|---|---|---|
| Optical degeneracies are irreducible even with priors | medium | Accept as ensemble role; if even SAR/thermal/age can't lift V6, the stream's value is parameter retrieval, not classification — scope accordingly |
| No coincident structural ground truth obtainable | medium | Provenance claims (snow vs ice) stay unvalidated → restrict claims to optical-state + parameters; do **not** ship structural labels |
| Atmospheric correction dominates error on real satellite data | medium-high | Quantify in C4/V5; if it swamps the spectral signal, reposition as a field/drone tool, not satellite |
| Cubic-liquidus/LWC rebuild shifts validated results unfavourably | low-medium | Phase-0 gate re-runs V1–V3 before accepting |
| Calibration impossible with available data volume | medium | Report uncalibrated with explicit warning; fusion layer down-weights accordingly |

**Kill / rescope criteria:** if Phase 1 shows priors cannot lift the dark
cluster on real data **and** Phase 2 cannot obtain independent validation, the
honest outcome is to ship this as a *parameter-retrieval and optical-state*
tool with explicit non-classification scope — not an operational classifier.

---

## 9. Immediate next actions (this branch)

The lowest-regret, highest-leverage starting set, all doable now:
1. **C1 cubic liquidus** + rebuild (removes a known substantive physics error).
2. **A1/A2 probabilistic output + provenance** (the ensemble's hard requirement).
3. **B2 temperature prior** via the metadata adapter (highest-leverage lever).
4. **V2 perturbed-physics validation** (bounds the inverse crime now, no new data).
5. **V1 freeze the held-out MOSAiC corpus** (stop tuning against test data).

These convert the sharpest open critiques from "acknowledged" to "addressed or
bounded" without waiting on field campaigns, and set up the probabilistic,
fusable interface the ensemble needs.
