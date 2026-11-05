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
place for the VIS.

Still deferred in the physics: **no liquid-water content in the melting-surface
forward models**, which is precisely the regime where the hold-out degrades.

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
