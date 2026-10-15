# Sea Ice Build — status

MVP complete 2026-06-12; **post-MVP audit + OE-robustness campaign complete
2026-07-17** (§ below). Suite: **778 tests** on `feature/sea-ice-mvp`.

---

## Post-MVP: three-lens audit, fixes, and OE robustness (2026-07)

A science / physics / code audit (six parallel reviewers) plus an
ecosystem recon (IceNav, SARSAR, auka) drove a round of fixes and a focused
optimal-estimation robustness effort. Full narratives:
`docs/OE_MODEL_ERROR_EXPERIMENT.md`, `docs/sea_ice_validation.md` §12,
and the audit findings are traceable to the commits below.

### Bugs fixed

| Commit | Content |
|---|---|
| `a52afa32` | core RT physics fixes brought onto this branch (float-`solzen` KeyError, coated-sphere `(-1)**n` + degenerate branches, ice density 917) |
| `10f537f6` | **B1 (critical)** `direct` defaults to 1 (omitting it silently classified every pixel `open_water`); **B4** emulator parameter-name validation; **B5** symmetric per-type prior penalty in classification; **B3** √2-corrected Hessian σ |
| `66325941` | **B7** batch: partial-coverage spectra accepted, `NO_RETRIEVAL` failure flag, OE/provenance fields carried into scene output |
| `558fbd23` | **B8** `verify()` refuses transform-built emulators; **B9** sweep spectral alignment after sort/filter; **B10** sweep impurity broadcast matches `run_model` |

### Documentation corrections (audit science findings)

| Commit | Content |
|---|---|
| `2b615255` | **S1** pond depth-accuracy vs classification split (94% was classification, depth ±25% ≈ 22%); **S2** "calibrated" OE language removed; OE band-mask caveat; stale 9/16 → 16/16 |
| `042c5b2f` | **M5** Vb-identifiability contradiction (bubble radius is the measured bare-ice quantity, Vb weakly identified); **M6** retired (T,S)-parameterisation examples corrected |

### OE classification robustness (B6 → A7 → C1 arc)

| Commit | Content |
|---|---|
| `0b19f6cc` | **B6** measured: full-spectrum OE classification unreliable (7/12 VIS, 2/12 SWIR) and overconfident (p≈1.00), vs default 12/12 — `oe_swir_classification_experiment.py` |
| `98ab3afe` | **A7 step 1** field-residual library (188 spectra → 521 rows); measured ~2 effective DOF over 60 VIS bands → diagonal S_e over-counts evidence ~30× |
| `43f6b03c` | **A7 step 2** field-calibrated forward-model error covariance (`model_error=True`); held-out reduced χ² 35→5; VIS OE 58%→92% |
| `b3d0742c` | **band-mode measurement**: IceNav's 9-band path is *not* affected by B6 (OE summer 15/16, honest confidences) — the production path is sound; guidance corrected to mode-dependent |
| `b730d99d` | **C1** metadata→class-prior adapter (`metadata_priors.py`): composable providers, `class_priors=` / `prior_sources=`, `known_month` as the `season` provider; priors bite in the honest regimes (band mode, `model_error`), lifted SHEBA summer band-mode S2 15→16/16, L8 13→15/16 |

### Standing next steps (data- or calibration-gated)

- **A6** reliability calibration — before OE probabilities / soft-prior magnitudes may be called "calibrated".
- **SWIR covariance extension** for `model_error` — needs more full-range field spectra (archive mining, or gap-tolerant EM-PCA).
- **C2–C4 concrete providers** (skin-temperature, ice-age, region) — each needs its external data source; the C1 caller channel accepts them today.
- **Default-behaviour decision**: whether `method="oe"` defaults to `model_error=True` — changes `class_probabilities` for IceNav, coordinate first.

---

## MVP build — COMPLETE (2026-06-12)

All twelve blocks of `docs/fable-sea-ice-build-guide.md` are implemented,
validated and committed on `feature/sea-ice-mvp`. Suite at completion: **721
passed, 0 failed**.

## Commits

| Commit | Content |
|---|---|
| `a72598f2` | B1 open water class (analytical Cox & Munk + Morel & Prieur; confidence 0.999 on acceptance test) |
| `f1181bf9` | C1 quality flag bitmask |
| `dcf81d87` | D1+D2 batch retrieval + SeaIceSceneResult (NetCDF/GeoTIFF/H3 exports, `[geo]` extras) |
| `65c1b72c` | D3 hyperspectral resampling utilities |
| `4db00668` | A3 LUT radius clamp + warning |
| `79fae251` | A1/A2/B2/C2/C3/E2 — tau_snow reparam, young ice (layer_type=6), band masks, SNR defaults, MCMC plumbing, FYI_bare 60k emulator promotion |
| `63352a4b` | A1/E1 validation tooling (audit + parameter retrieval suite) |
| `c3933092` | E3 documentation pass |

## Key outcomes & deviations from the guide (all documented in docs/)

- **A1**: "R² ≈ 0.70" was sklearn's training score in PCA-coefficient space.
  True held-out spectral R² was already 0.997; promoted the 60k-sample retrain
  (R² 0.9985, BBA MAE 0.0023). rho_DL/Vb not degenerate (mean |cos| 0.37).
  Cost: 2/120 borderline shallow-pond classifications in the SHEBA suite.
- **A2**: log-space alone did not fix bound-hitting (6/7 spring at bounds).
  `tau_snow` = depth/grain-radius (log-conditioned, grain bound 50 µm) fixed
  it: **0 % snow-parameter at-bounds**, classification unchanged, band-mode
  summer accuracy improved to 88 %/88 % (S2/L8). T bound kept at −5 °C —
  widening to −2 °C collapsed summer accuracy 16/16 → 9/16 (tested, reverted).
  Two May dates saturate T at −5 °C under the melt-season prior (informative).
- **B2**: spec's constant-ρ Beer-Lambert formula cannot reproduce G&M's
  albedo-vs-thickness growth; implemented as two-stream (Kubelka-Munk) slab
  over ocean, frazil scattering 3.0 m⁻¹ calibrated to G&M Table 3 (all
  checkpoints within ±0.03). Parameter is `ice_thickness_cm` (log-conditioned;
  metres derived) since log10(x+1) is a no-op on metre scale.
- **B1**: guide's blanket BBA ∈ [0.03, 0.10] cannot hold at SZA extremes
  (Fresnel physics); tests encode physical assertions instead.
- **C2**: masked classification chi-squared is rescaled to the full band count
  and includes the prior penalty, so costs stay comparable and `known_month`
  keeps steering classification.
- **E1**: structural parameters retrieve with R² 0.97–1.00; T/brine-volume/
  ocean-albedo honestly non-identifiable without priors (negative R²) — table
  in `tests/validation_data/parameter_retrieval_results.md`.

## Not committed (intentionally)

- `data/emulators/experiments/*.npz` and `tests/validation_data/residual_library.npz`
  (gitignored — regenerable via `scripts/experiments/fyi_bare_audit.py` and
  `tests/validation_data/build_residual_library.py`; the model-error artifact
  `data/model_error/field_se_v1.npz` *is* committed, as a retrieval input).

## Possible follow-ups (out of scope)

- WorldView-3 entry in `PLATFORM_SNR` when band uncertainties are available.
- Vectorised inverse network behind `SeaIceSceneResult` for >100k-pixel scenes.
- Young-ice validation against archived SHEBA October 1997 freeze-up spectra
  if locatable (Y-6 stretch goal).
- Shallow-pond (<10 cm) classification remains genuinely hard (spectral
  overlap with bare ice) — flagged via SPECTRALLY_AMBIGUOUS rather than fixed.
