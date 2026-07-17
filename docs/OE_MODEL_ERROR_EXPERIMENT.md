# OE Classification and Forward-Model Error: the B6 → A7 Experiment Arc

**Dates:** 2026-07-15 → 2026-07-17
**Status:** complete through A7 step 2 (v1 covariance); SWIR extension and default-behaviour decision open.
**Scripts:** `tests/validation_data/oe_swir_classification_experiment.py`,
`tests/validation_data/build_residual_library.py`,
`scripts/build_model_error_covariance.py`
**Module:** `biosnicar/inverse/model_error.py`; artifact `data/model_error/field_se_v1.npz`

This document records, end to end, how an audit finding about the
optimal-estimation (OE) classifier became a measured failure, a diagnosed
mechanism, and a field-calibrated fix — including what is still broken.

---

## 1. The finding (audit item B6)

The 2026-07 repository audit noted that `retrieve_sea_ice(method="oe")`
classifies by Laplace model evidence computed over the **full observation**,
whereas the default method classifies with per-type **band masks** (bare-ice
candidates scored VIS-only). The masks exist because of an earlier headline
result: adding SWIR to summer bare-ice spectra collapses classification
(100% → 42%) — melting bare ice and coarse snow are near-identical in the
SWIR. Evidence comparison mathematically requires every candidate to score
the same data, so the masks cannot be applied to the OE path.

## 2. Experiment 1 — is OE classification actually degraded? (2026-07-15)

Design: 2×2 on the 12 SHEBA summer white-ice dates with paired ALBI SWIR
columns; `{default, oe}` × `{VIS 400–1000 nm, VIS+SWIR 400–2000 nm}`;
`known_month` priors on; expected class ∈ {FYI_bare, FYI_summer}.

| | VIS | VIS+SWIR |
|---|---|---|
| default (masks) | 12/12 | 12/12 |
| `method="oe"` | 7/12 (58%) | 2/12 (17%) |

**Worse than the hypothesis:** OE not only reproduced the SWIR collapse — it
was substantially worse than the default even VIS-only. Every failure
classified **FYI_snow at posterior probability ≈ 1.00**: confidently and
wrongly.

## 3. Could metadata priors fix it? (quantified, no)

A class prior can outvote the spectrum only by its odds ratio; an honest
August anti-snow prior is perhaps 3:1–10:1 (summer snow patches genuinely
exist — MOSAiC). The failures demanded prior odds of 19:1 and 49:1 in the two
mildest cases, and 10⁵–10²⁶⁰ (or literally infinite — the correct class at
float-zero probability) in the rest. **The likelihood was screaming; priors
whisper.** Metadata class priors (dev plan C1) remain worth building, but as
a complement to a fixed likelihood, not a substitute.

## 4. The mechanism, measured — the field-residual library (A7 step 1)

If the model were perfect and only the instrument (σ ≈ 0.02) noisy, post-fit
residuals against real spectra would be band-independent noise. They are not.
`build_residual_library.py` collects post-fit residuals (obs − pred, 480-band
grid) from all three spectral validation campaigns, fitted exactly as each
campaign's validation script does:

| Campaign | Spectra | Rows | Winner VIS RMS (median) |
|---|---|---|---|
| SHEBA 1998 | 23 | 40 | 0.0094 |
| Smith/MOSAiC 2020 | 44 | 108 | 0.0167 |
| Istomina/IceArc 2012 | 121 | 373 | 0.0172 |

(Winner medians reproduce the documented campaign values — collector fidelity
check.) Rows are stored for the winner fit and each expected-type fit
(`is_winner` / `is_expected`); the correct-model population is the A7 input.

**The key measurement:** on the clean correct-model rows, per-band VIS σ has
median 0.018 (max 0.041) — but the residuals are smooth, correlated curves:
**60 VIS bands carry only ~2.1 effective independent degrees of freedom**
(94% of variance in 3 EOFs). A diagonal S_e = 0.02 treats them as 60
independent measurements — over-counting spectral evidence ~30×. Divide the
observed 100+-nat evidence gaps by ~30 and they land exactly in the few-nat
regime where honest priors (and honest doubt) live. This is the overconfidence
mechanism, measured.

## 5. The fix — low-rank + diagonal model-error covariance (A7 step 2)

`biosnicar.inverse.model_error` implements

    S_e = diag(σ_instrument²) + B diag(λ) Bᵀ + diag(d)

with B = leading EOFs of the correct-model field residuals (uncentred second
moments — systematic model bias is part of the error budget), λ their
variances, d the per-band residual floor (min 1σ = 0.005).

**Calibration hygiene:**

* Fitted on SHEBA + Smith correct-model rows (46). **Istomina held out
  entirely.**
* SHEBA is VIS-only, so the EOF *correlation* structure is calibrated on
  Smith full-range rows; SHEBA informs the VIS diagonal.
* Threshold selection was evidence-based: widening the EOF domain into the
  SWIR (fewer full-coverage rows) *worsened* held-out χ² — v1 keeps a
  VIS-only EOF domain (60 bands, 2.8 effective DOF) with diagonal-only SWIR.
* No field residuals exist for young_ice / open_water; the model is a pooled
  estimate applied fleet-wide, flagged as such.

**Held-out coverage (Istomina, 84 correct-model rows, reduced χ²
mean/median):** instrument-only **35.1 / 32.1** → with model error
**5.1 / 4.0**. A ~7× honesty improvement; the remaining ~4× optimism is a
real regime effect (Istomina's late-melt surfaces fit worse than the
calibration campaigns) and is *reported, not tuned away* — tuning on the
hold-out would defeat its purpose.

Usage: `retrieve_sea_ice(..., method="oe", model_error=True)` (spectral mode
only; band-mode S_e is the atmospheric-correction budget, roadmap B-MS1).
Rebuild: `python scripts/build_model_error_covariance.py`.

## 6. Acceptance re-run (2026-07-17)

Same experiment, third variant `oe+Se`:

| | VIS | VIS+SWIR |
|---|---|---|
| default (masks) | 12/12 | 12/12 |
| oe (instrument-only) | 7/12 (58%) | 2/12 (17%) |
| **oe + model-error S_e** | **11/12 (92%)** | 4/12 (33%) |

* **VIS: repaired.** 58% → 92%, one miss (28 Aug — historically the most
  ambiguous date) at an *honest* confidence of 0.87 rather than a wall of
  1.00s. Posterior parameter uncertainties widen appropriately
  (e.g. pond-depth σ ×2 on the synthetic check) and DFS drops slightly —
  the information budget is now honest.
* **VIS+SWIR: still broken (33%), for exactly the predicted reason.** The
  v1 covariance has no SWIR correlation structure (diagonal only there), so
  SWIR evidence is still over-counted and snow still wins at p ≈ 1.00. The
  failure is confined to precisely the bands the model is not calibrated
  for — strong evidence the covariance explanation is the right one.

## 7. Standing guidance

| Task | Recommendation |
|---|---|
| Classification, any window | Default method (12/12; band masks) |
| Classification, VIS-only spectra, probabilistic output wanted | `method="oe", model_error=True` is now credible (92%); treat probabilities as approximate until A6 calibration |
| Classification, VIS+SWIR with OE | **Do not** — restrict with a 400–1000 nm `wavelength_mask` or use the default method |
| Parameter posteriors / DFS / averaging kernels | `method="oe", model_error=True` on a known or default-classified type |
| Satellite band mode | `model_error` refuses by design; atmospheric-correction error budget is separate (B-MS1) |

## 8. What's next

1. **SWIR extension of the covariance** — needs more full-range residual
   rows: mine further MOSAiC legs / Antarctic archives (dev plan §5), or a
   gap-tolerant EOF fit (EM/PPCA) that uses partially-covering rows without
   the full-coverage requirement that made wider domains worse in v1.
2. **A6 calibration** — reliability diagrams for the oe+Se probabilities on
   a held-out corpus; only then may "calibrated" language return.
3. **C1 metadata class priors** — now worth attaching: evidence gaps are in
   the few-nat regime where a 5:1 month/location prior has real leverage.
4. **Default-behaviour decision** — should `method="oe"` default to
   `model_error=True`? It widens uncertainties (honest) and changes
   `class_probabilities` for downstream consumers (IceNav) — coordinate
   before flipping.
5. **Hybrid classification** remains the strongest option where SWIR is
   present (default masked selection + OE posteriors on the winner).

## 9. Honest limitations of this arc

* n = 12 dates, one site (SHEBA), one season — the acceptance numbers are
  indicative, not definitive; the held-out χ² (Istomina, n=84) is the more
  robust statistic.
* The residual library's expected-type populations inherit each campaign's
  label quality (SHEBA season-inferred; Istomina coarse field notes).
* Post-fit residuals slightly understate model error (the fit absorbs some).
* The v1 covariance pools surface types; per-type stratification awaits more
  data per type.
