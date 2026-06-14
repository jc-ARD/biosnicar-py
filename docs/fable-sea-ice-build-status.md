# Sea Ice Build — COMPLETE (2026-06-12)

All twelve blocks of `docs/fable-sea-ice-build-guide.md` are implemented,
validated and committed on `feature/sea-ice-mvp`. Final suite: **721 passed, 0
failed**.

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
  over ocean, frazil scattering 1.5 m⁻¹ calibrated to G&M Table 3 (all
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

- `docs/fable-sea-ice-build-guide.md` (user's planning doc) and this status file.
- `data/emulators/experiments/*.npz` (gitignored — regenerable via
  `scripts/experiments/fyi_bare_audit.py`; results JSON is committed).
- Pre-existing deletion of `.claude/scheduled_tasks.lock` left untouched.

## Possible follow-ups (out of scope)

- WorldView-3 entry in `PLATFORM_SNR` when band uncertainties are available.
- Vectorised inverse network behind `SeaIceSceneResult` for >100k-pixel scenes.
- Young-ice validation against archived SHEBA October 1997 freeze-up spectra
  if locatable (Y-6 stretch goal).
- Shallow-pond (<10 cm) classification remains genuinely hard (spectral
  overlap with bare ice) — flagged via SPECTRALLY_AMBIGUOUS rather than fixed.
