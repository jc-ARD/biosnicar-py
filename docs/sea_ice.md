# Sea Ice Extension — Primer

**Version**: v0.4 (forward model + young ice + open water + inversion)
**Module**: `biosnicar.sea_ice`

---

## How sea ice relates to the other layer types

BioSNICAR uses a `layer_type` integer per layer to select the optical-property model. The full set is:

| `layer_type` | Physical model | Fresnel surface |
|---|---|---|
| 0 | Granular snow or ice — discrete grains in air | No |
| 1 | Solid bubbly glacier ice + Fresnel air-ice surface | **Yes** |
| 2 | Solid bubbly glacier ice, no Fresnel correction | No |
| 3 | Granular water/ice sphere mixture (slush) | No |
| **4** | **Sea ice — brine and salt inclusions, air bubbles** | **Yes** |
| **5** | **Melt pond — liquid water layer (near-pure absorption)** | **Yes** (air–water, n ≈ 1.33) |
| **6** | **Young ice — semi-transparent thin slab over ocean (boundary condition, bypasses the τ/ω/g pipeline)** | **Yes** |

The smooth interface sits on top of the **first** layer of type 1, 4 or 5, and takes the refractive index of that layer — water for a melt pond, ice otherwise. `run_model(fresnel_layer=...)` overrides the rule: an integer puts the interface on top of that layer whatever its type, `None` removes it. The default placement is a modelling choice, not a fact: under a granular surface scattering layer (`[0, 4, 4]`) it puts a smooth interface *beneath* the grains.

Types 1 and 2 are the same optical model (bulk ice matrix, absorption from `k_ice`, scattering from air/water bubble LUTs) — the only difference is that type 1 triggers the Fresnel surface reflection in the solver and type 2 does not. Type 3 is a different physical picture altogether: discrete spheres of ice and water in air, combined by linear volume-fraction mixing, with no effective medium theory.

Sea ice (`layer_type=4`) is closest to type 1 in structure — solid bulk ice with inclusions and a Fresnel surface — but replaces the pure-ice absorption with an effective medium calculation that accounts for brine pockets. It adds three required per-layer inputs (`sea_ice_temperature`, `sea_ice_salinity`, `sea_ice_bubble_radius`) and draws from a separate pre-computed LUT. See [docs/METHODS.md](METHODS.md) for the full layer-type reference.

---

## What this extension does and doesn't do

### Does
- Computes spectral albedo (480 bands, 0.205–4.995 µm) for sea-ice columns with optional snow cover.
- Models brine through one phase model (Cox & Weeks / Leppäranta & Manninen) from −30 to 0 °C: its volume, its liquidus salinity, its absorption, and its scattering as pockets and tubes; plus the salts that precipitate from it in cold ice.
- Models melt ponds as liquid water layers (layer_type=5) with realistic water absorption.
- Accepts per-layer physical inputs: thickness, temperature, salinity, density, bubble radius, pond depth.
- Provides five pre-configured presets: three winter sea-ice types plus two melt pond depths.
- Works with the existing BioSNICAR `run_model()` API — no new classes required.
- Fully backward-compatible: all existing terrestrial-ice functionality unchanged.

### Does (added in v0.4)
- **Young ice** (`layer_type=6`): grease ice, nilas, grey/grey-white ice 0.5–30 cm — a two-stream thin slab over a reflecting ocean (see below).
- **Open water**: analytical Fresnel + subsurface albedo model (`biosnicar.sea_ice.open_water`).
- **Inverse retrieval and classification**: `retrieve_sea_ice()` classifies seven surface types and retrieves their parameters with uncertainties and quality flags — see [SEA_ICE_RETRIEVAL.md](SEA_ICE_RETRIEVAL.md).

### Doesn't (deferred to later versions)
- **Sea-ice algae** (v0.5): algal blooms in sea ice are not in the impurity database.
- **Salty snow** (v0.5): snow on sea ice is treated as fresh.
- **Vertical T/S profiles** (v0.5): each layer uses a single T and S value.
- **Antarctic-specific tuning** (v0.5): validation uses Arctic data only.
- **Pancake ice / ice rind** (young-ice spec out-of-scope): require lateral-scattering treatment.

---

## Physics in brief

### Why sea ice is different from glacier ice

Sea ice forms from seawater and retains dissolved salt as **brine inclusions**: pockets of liquid concentrated seawater that remain liquid even at temperatures well below 0 °C. These inclusions affect the ice's complex refractive index and therefore its spectral albedo. First-year ice (FYI) typically contains 4–12 psu of bulk salinity, while multiyear ice (MYI) has been desalinated by summer brine drainage and contains only 1–3 psu.

### Phase model: brine volume, gas volume, brine salinity, precipitated salts

The fraction of the ice that is liquid brine (**brine volume fraction ν_b**) depends on bulk salinity S (psu), temperature T (°C) and bulk density ρ_si (g/cm³):

```
ν_b   = S × ρ_si / F1(T)
ν_air = 1 − ρ_si/ρ_i + ρ_si × S × F2(T)/F1(T)        ρ_i = 0.917 − 1.403×10⁻⁴T

−22.9 ≤ T ≤ −2 °C   F1 = −4.732 − 22.45T − 0.6397T² − 0.01074T³          (Cox & Weeks 1983)
                    F2 = 0.08903 − 0.01763T − 5.330×10⁻⁴T² − 8.801×10⁻⁶T³
−30 ≤ T < −22.9 °C  F1 = 9899 + 1309T + 55.27T² + 0.7160T³               (Cox & Weeks 1983)
                    F2 = 8.547 + 1.089T + 4.518×10⁻²T² + 5.819×10⁻⁴T³
−2 < T ≤ 0 °C       F1 = −0.041221 − 18.407T + 0.58402T² + 0.21454T³     (Leppäranta & Manninen 1988)
                    F2 = 0.090312 − 0.016111T + 1.2291×10⁻⁴T² + 1.3603×10⁻⁴T³
```

At T = −10 °C and S = 8 psu, ν_b ≈ 0.045. The **melt season is covered**: at −1 °C ice of 5 psu is 24% brine, and as F1 → 0 the ice reaches its melting point (−0.27 °C at 5 psu; −1.86 °C at 34 psu, the freezing point of sea water) and is returned as fully melted. The model used to refuse anything above −2 °C. The Leppäranta & Manninen coefficients were checked against SMRT v1.7.3.

**Brine salinity is the same phase model, not a second one.** Brine sits at the liquidus, so its salinity depends on temperature alone; and it is tied to the brine volume by conservation of salt, `S·ρ_si = S_b·ρ_b·ν_b·(1 + k)`, with k the precipitated salt per unit of salt still dissolved. Since ν_b = S·ρ_si/F1, the Cox & Weeks F1 *is* the liquidus: `F1 = ρ_b·S_b·(1 + k)`, with `ρ_b = 1 + 0.0008·S_b`.

- **Above −8.2 °C** nothing has precipitated, and S_b is obtained from F1 by solving that quadratic. Brine volume and brine salinity then conserve salt identically (a test asserts it to 10⁻⁹).
- **−22.9 to −8.2 °C**: Assur's phase data, `S_b = 57.041 − 9.929T − 0.16204T² − 0.002396T³` (Cox & Weeks 1975).
- **Below −22.9 °C**: `S_b = 206.24 − 1.8907T − 0.060868T² − 0.0010247T³` (Cox & Weeks 1986), shifted by 0.7% to join.

That gives 37 psu at −2 °C, 87 at −5, 143 at −10, 228 at −22.9 and 234 at −30. It replaces a liquidus derived from the Frankenstein & Garner (1967) brine-volume relation, which disagreed with the Cox & Weeks volume it was used alongside by 1.7% at −2 °C rising to 13.9% at −22.9 °C.

**Precipitated salts.** Mirabilite (Na₂SO₄·10H₂O) forms below −8.2 °C and hydrohalite (NaCl·2H₂O) below −22.9 °C. Hydrohalite mass is the sodium chloride that leaves the brine as the cold-branch brine volume collapses, computed from the same fits (about 60% of the salt by −30 °C). Mirabilite mass is a saturating interpolation anchored to one literature value — 9×10⁻⁴ kg per kg of 4.7 ppt ice at −15 °C — and capped by the sulphate available; it is the weakest link in the phase model and matters least optically.

### Brine refractive index

**Real part.** In 400–700 nm the Quan & Fry (1995) formula is evaluated at the liquidus salinity and the actual temperature:

```
n(S_b, T, λ) = n₀ + (n₁ + n₂T + n₃T²)·S_b + n₄T²
             + (n₅ + n₆·S_b + n₇·T)/λ + n₈/λ² + n₉/λ³          (λ in nm)
```

plus a high-salinity term anchored to the measurements of Maykut & Light (1995), who found 1.341 at −2 °C and 1.397 at −32 °C at 589 nm. Quan & Fry alone reproduces the warm value exactly and is 0.0055 low at the cold end — it was fitted to S ≤ 40 ppt above 0 °C — and the extra term closes that. Outside 400–700 nm the tabulated water index is carried along with the formula's offset at the window edge.

The tabulated water index (Rowe et al. 2020) is no longer the baseline inside the window. Its real part is 0.003 low at 589 nm and 0.005 low at 700 nm against the accepted pure-water value, and because what matters optically is the 2–6% contrast between brine and ice, that error had removed about a tenth of it. The same file still supplies the water index for the melt-pond surface, where the effect is negligible (reflectance 2.02% against 2.05%).

The brine-in-ice relative index is now 1.024 at −2 °C and 1.065 at −30 °C, as in Light et al. (2004, Fig. 7a).

**Imaginary part.** NaCl has no absorption above 400 nm. Only a small UV ionic contribution (Cl⁻ band, zero by 0.4 µm) is added to the water absorption; in the NIR k_brine ≈ k_water.

### Effective medium — absorption

Ice and brine are mixed with the **symmetric Bruggeman rule**, which treats neither phase as the host and holds at any brine fraction:

```
(1 − f)(ε_ice − ε)/(ε_ice + 2ε) + f(ε_brine − ε)/(ε_brine + 2ε) = 0
```

It replaced Maxwell-Garnett with ice as host, which is only valid for dilute inclusions (f < ~0.3) and was being applied to melting ice beyond that. The numbers barely move — the two rules agree to 0.04% in the imaginary index across the whole range, because ice and brine differ by only ~5% in permittivity — so this is a matter of using the rule that is valid, not of changing the answer. The layer's absorption coefficient is `4π·k_eff/λ × (1 − ν_air)`.

### Scattering by brine inclusions and precipitated salts

Air bubbles are not the only scatterers. Brine sits in pockets and tubes with a refractive index 2–6% above the ice around them, and salts crystallise out of it in cold ice. These follow the structural-optical model of **Light, Maykut & Grenfell (2004)**, built from simultaneous microstructure and optical measurements on first-year ice between −33 and −1 °C (`biosnicar.sea_ice.inclusion_scattering`).

Every scatterer is reduced to *equivalent spheres* with the same total volume and surface area. For such a collection the cross-sectional area per unit volume is `ψ = A/4`, the equivalent radius is `3V/A`, and `σ = Q_sca·ψ`, with `Q_sca` and `g` from Mie theory for an optically soft sphere at the relative index of the scatterer in its surroundings.

- **Brine inclusions.** The reference microstructure is the measured −15 °C size distribution, `N(l) = 0.28·l⁻¹·⁹⁶` per mm³ per mm of inclusion length, with aspect ratio `10.3·l⁰·⁶⁷`; pockets (< 0.5 mm) are prolate ellipsoids and tubes are cylinders. More salt extends the distribution to longer tubes; a change of temperature scales every inclusion's volume with the bulk brine volume (pockets keep their shape, tubes keep their length); above −14 °C inclusions merge. The implementation reproduces the published figures: 24 inclusions per mm³, ψ = 28 and 82 m⁻¹ for pockets and tubes (published 30 and 80), and σ = 290 m⁻¹ at −15 °C for their 4.7 ppt sample (published 280).
- **Mirabilite and hydrohalite.** Cubes of 10 µm and 4.5 µm edge inside the brine, `ψ = 1.5·V/a`.

Brine inclusions scatter strongly but almost entirely forward (g ≈ 0.98–0.997), so what matters is the similarity parameter `s = σ(1 − g)`. For interior first-year ice at −15 °C and 4.7 ppt they contribute s ≈ 3 m⁻¹; 300 µm bubbles contribute about 7 m⁻¹ at 1% porosity and about 50 at the 7% of a drained layer. So in bubbly, drained ice this is a correction, and in dense interior ice it is comparable to the bubbles. Preset broadband albedos rise by 0.003–0.007. For Light et al.'s sample the layer now gives s = 10.5 m⁻¹ and g = 0.979, against their measured 8.7–9.3 and 0.978–0.982 (the gas in that comparison is ours, not theirs).

Known limits:

- **Cold ice is under-scattered.** Below −22.9 °C Light et al. match their observations with 4.5 µm hydrohalite crystals under `Q_sca = 2`. Crystals that small have a small phase shift and Mie theory gives them `Q_sca` nearer 1, so this model produces about a third of the hydrohalite scattering theirs does (s ≈ 3 against ≈ 9 m⁻¹ at −30 °C). The crystal size and refractive index (1.416 assumed; not well determined in the literature) are jointly uncertain. This regime is reached only under a snow cover in the shipped emulators.
- **One microstructure for all ice.** The distribution was measured on interior first-year ice of 4.7 and 15 ppt and is applied to everything, including multi-year ice. Above a brine fraction of 0.5 the geometry is held fixed, since the ice is then the inclusion.
- **Gas is not split** into bubbles inside brine inclusions, which grow on warming and escape, and bubbles in the ice matrix. The gas volume comes from the bulk density and all of it scatters as bubbles of the retrieved radius.

### Lower boundary

A sea-ice column sits on the ocean: when the bottom layer is sea ice the boundary reflectance is the ocean's subsurface irradiance reflectance (2–4% in the blue, black beyond the red). It used to be the default for glacier columns, a measured blue glacier-ice spectrum (16% at 450 nm). The effect is 0.002 in broadband albedo for 30 cm of ice and under 0.001 at 1.5 m.

### Air bubble scattering

Sea ice also contains air bubbles whose scattering dominates the NIR optical depth. The air volume fraction is the gas half of the same Cox & Weeks (1983) density balance that gives ν_b, evaluated at the layer's bulk density ρ_si (g/cm³):

```
ν_air = 1 − ρ_si/ρ_i + ρ_si × S × F2(T)/F1(T)
```

Because ν_b and ν_air come from one balance at one density, they are mutually consistent by construction. Earlier versions computed ν_b bubble-free and then took ν_air from a separate balance,
`ν_air = (917 − ρ_si + (ρ_brine − 917) × ν_b) / 917`, which used a Frankenstein & Garner-derived brine density — two phase models describing the same air, only one of which ν_b had assumed.

Scattering coefficients are looked up from the existing `bubbly_air.npz` LUT (pre-computed Mie theory for air bubbles in pure ice). *MVP approximation*: the effective-medium correction to bubble scattering (host RI changes slightly with brine content) is < 5% and is neglected.

---

### Young ice (layer_type=6) — thin-slab two-stream model

Ice thinner than ~15–20 cm violates the opaque-column assumption of
`layer_type=4`: radiation transmitted through the slab reflects off the ocean
(albedo 0.03–0.08) and returns. The young-ice model treats the column as a
single homogeneous slab and solves the diffuse two-flux (Kubelka–Munk)
equations analytically over the ocean boundary:

- **Absorption** `K(λ) = 2[(1−V_b)·κ_ice(λ) + V_b·κ_brine(λ)]` from pure-ice
  RI (Picard 2016) and liquidus brine (Cox & Weeks brine volume from bulk T, S
  — new ice is *not* desalinated, S = 10–35 psu).
- **Scattering** from frazil/congelation crystal boundaries: a constant
  S = 3.0 m⁻¹, calibrated so the BBA-vs-thickness curve matches
  Grenfell & Maykut (1977) Table 3 within ±0.03 (this coefficient is the
  least-constrained quantity in the model — G&M report extinction at only a
  few wavelengths).
- **Air–ice Fresnel** interface: external reflectance at the illumination
  geometry for the downwelling beam; the diffuse upwelling flux from the slab
  sees the interface from below, where total internal reflection gives an
  internal diffuse reflectance ≈0.45 (R_int = 1 − (1 − R_dif)/n²).

A pure Beer–Lambert slab with constant surface reflectance *cannot* reproduce
the observed albedo growth with thickness (it decays toward bare Fresnel as
the ocean dims) — internal backscattering must accumulate, which is what the
two-stream solution provides. Albedo increases monotonically with thickness,
darkens as the ice warms (more liquidus brine), and tracks the ocean albedo
when very thin. WMO stages map directly from thickness: grease/frazil (<1 cm),
dark/light nilas (1–10 cm), grey ice (10–15 cm), grey-white ice (15–30 cm).

### Open water — Fresnel + subsurface

Open water albedo is the sum of (1) specular Fresnel reflection at the rough
air–water interface — spectral water RI (Rowe 2020) averaged over the
Cox & Munk (1954) wind-slope distribution, so wind *darkens* water at high
solar zenith angle — and (2) diffuse subsurface upwelling from pure-seawater
molecular scattering, R(0⁻) = 0.33·b_b/(a+b_b) (Morel & Prieur 1977), which is
non-zero only in the blue/green. Typical BBA: 0.02 at SZA 20°, 0.06–0.07 at
60°, rising to ~0.3 for calm water at 80°. The model is fully analytical
(`OpenWaterModel`) and joins the retrieval fleet without training data.

## Known approximations and their consequences

### Brine parameterisation (all sea-ice layer types)

| Approximation | Effect | When it matters |
|---|---|---|
| Liquidus S_b(T) derived from F&G (1967) brine volume by salt balance (not a direct Assur fit) | Within a few psu of accepted values for −2 to −15 °C; saturates against the ~233 psu eutectic below ≈−18 °C, where inverting a volume fit is least reliable | Very cold ice (< −18 °C) |
| ~~CW83 gas-free brine volume + separate density-balance air fraction~~ | **Fixed 2026-11-05.** ν_b and ν_air now both come from CW83 at the layer bulk density. The overestimate was 1–9% across the shipped presets | — |
| ~~Absorption ignored air content~~ | **Fixed 2026-11-05.** Absorption was that of the ice+brine phase applied across the whole layer, so it did not fall as the layer became more porous (~82 m⁻¹ from 920 down to 500 kg/m³). Now scaled by (1 − ν_air) | — |
| ~~Bubble radius clamped outside 100–1000 µm~~ | **Fixed 2026-11-06.** The LUT bubble grid was `[100, 300, 500, 1000]` while the emulators sample 50–2000 µm, so τ and ω were bit-identical outside that window — and at r = 50 µm the clamped τ was wrong by 2×. Grid is now 11 log-spaced points over 50–2000 | — |
| Bubble scattering linear in ν_air | Independent-scatterer assumption from a single-bubble LUT. Bubbles are 100–1000 µm against 0.2–5 µm light, so inter-bubble clearance stays many wavelengths and the approximation holds well into high porosity | Untested above ~15% air |
| CW83 cold-branch polynomial clamped at −30 °C | No brine-volume change below −30 °C (polynomial validity limit) | Extreme winter T < −30 °C |

### Sea ice (layer_type=4)

| Approximation | Effect | When it matters |
|---|---|---|
| Brine real index anchored to two measured points (Maykut & Light 1995) | Shape between −2 and −32 °C assumed | Intermediate temperatures |
| Mirabilite mass interpolated from one literature value | Mirabilite contributes < 5% of the similarity parameter | −8 to −23 °C |
| Hydrohalite scattering under-predicted | About a third of the published model's below −22.9 °C | Very cold ice |
| One inclusion microstructure for all ice | Measured on interior first-year ice only | Multi-year ice, brine fraction > 0.5 |
| Air bubble scattering uses pure-ice LUT | < 5% error in scattering | All conditions |
| Snow layer treated as fresh water | Overestimates snow albedo for salty snow | Snow on FYI |
| Single T and S per layer | Ignores vertical gradients | Thick layers |

### Melt ponds (layer_type=5)

### The pond floor darkening problem

The most important finding from both prior literature and this model's validation is that **melt pond visible albedo is controlled primarily by the optical properties of the pond floor, not the depth of the water column** (Makshtas & Podgorny 1996, *Polar Research* 15(1), 43–52). This is physically intuitive: visible light is only weakly absorbed by water (k_water ≈ 3×10⁻⁹ at 500 nm), so it largely passes through to the ice below and reflects back. The water column dominates only in NIR (> 700 nm) where k_water is orders of magnitude larger.

The consequence: a model with a clean white ice floor (rho=895, bare FYI) overestimates observed visible pond albedo by +0.27–0.44 across all depth bins — the entire overestimate comes from the floor being too bright, not from errors in the water physics.

### How other models handle floor darkening

All major sea ice optical models deal with this through empirical calibration rather than first-principles microphysics:

- **Makshtas & Podgorny (1996)** prescribed pond floor albedo as a free parameter fitted to observations.
- **CCSM3 (Briegleb & Light 2007, NCAR/TN-472)** reduced ponded ice albedo by an arbitrary −0.075 per band, described in the report as "simply an educated guess, constrained somewhat by SHEBA measurements."
- **Delta-Eddington in CICE/CCSM4 (Briegleb & Light 2007; Holland et al. 2012)** infers Inherent Optical Properties (IOPs) from SHEBA-measured ponded ice albedo using the Light et al. (2004) structural-optical model. The SHEBA observations already include whatever biological and chemical darkening was present. The report states: "For bare and ponded sea ice, temperature dependent changes in the ice density, salinity, and brine volume are not considered. Instead, we make use of **well-observed SHEBA bare ice and ponded ice cases**, and infer IOPs from surface spectral albedo observations."
- **Light et al. (2008)** Monte Carlo model uses measured optical property profiles from SHEBA, implicitly including all absorbers present during measurement.

None of these approaches explicitly identify what is causing the dark floor — they encode it into calibrated parameters. BioSNICAR's approach of adding black carbon as an explicit absorber is more mechanistically transparent, even if BC is used as a proxy for a mixture of absorbers.

### Physical basis for floor darkening

Real Arctic melt pond floors are dark because of three co-occurring processes:

1. **Cryoconite accumulation**: Windblown mineral dust and black carbon particles deposit on sea ice throughout winter and spring. When the surface melts, these particles concentrate in the residual surface layer (melt-season concentration factor: ~10–20×). The particles are colonised by cyanobacteria and other microorganisms to form cryoconite granules, directly documented on Arctic sea ice by Takeuchi et al. (2009, *Polar Science* 3, 122–130) and earlier observations in Nansen (1897).

2. **Biological colonisation**: The ice surface layer below summer ponds hosts algal communities (predominantly *Chlamydomonas* and cyanobacteria) with chlorophyll-a absorption peaking at ~680 nm. This produces the sharp 600–700 nm drop seen in observations but not reproduced by a pure BC proxy.

3. **Melt-season concentration of atmospheric BC**: Light et al. (1998, JGR 103, 21817) measured light-absorbing particle concentrations of 100–2000 ng/g in Arctic sea ice surface layers, with the highest values in biologically-active melt layers. Marks & King (2013, *The Cryosphere* 7, 1213–1228) found median values of 100–500 ppb in surface sea ice, with upper-percentile values exceeding 1000 ppb in aged or biologically-active surface layers.

### Calibration against Morassutti (1995)

> **Stale as of 2026-11-05.** Every number in this subsection was produced
> before the air-fraction, absorption and LUT-density fixes, which move pond
> BBA by roughly +10% on the shipped presets. The *reasoning* — that the floor
> needs an explicit absorber, and that BC is a proxy for a mixture — still
> holds, but the fitted values (BC = 1000 ppb, rho = 895) are an optimum of the
> old forward model and the sweeps below should be re-run before either is
> quoted again. See the open-items table in `ROADMAP.md`.

Systematic calibration was performed against Morassutti (1995) melt pond spectral albedo (NSIDC G01169, doi:10.7265/N55Q4T1C), 504 records in 6 spectral bands (400–1000 nm).

**Clean-water model (no floor impurities, rho=895):**
VIS RMSE = 0.36, NIR RMSE = 0.07, BBA RMSE = 0.30

**BC in floor ice — BC sweep results (rho=895, T=−5°C, all depth bins):**

| BC (ppb) | VIS RMSE | NIR RMSE | BBA RMSE | Total |
|---|---|---|---|---|
| 0 | 0.362 | 0.071 | 0.296 | 0.729 |
| 400 | 0.129 | 0.044 | 0.134 | 0.307 |
| 800 | 0.068 | 0.031 | 0.076 | 0.175 |
| **1000** | **0.068** | **0.028** | **0.062** | **0.157** |
| 1200 | 0.076 | 0.026 | 0.055 | 0.157 |
| 1500 | 0.094 | 0.027 | 0.052 | 0.173 |

**Floor density sensitivity at BC=1000 ppb:**

| rho (kg/m³) | VIS RMSE | NIR RMSE | Sum |
|---|---|---|---|
| 870 | 0.085 | 0.044 | 0.229 |
| 880 | 0.073 | 0.037 | 0.195 |
| **895** | **0.068** | **0.028** | **0.157** |
| 910 | 0.101 | 0.032 | 0.188 |
| 920 | 0.157 | 0.053 | 0.294 |

**rho=895 is the optimum.** Higher densities (910, 920) worsen the fit because the near-zero air volume fraction at those densities makes the ice behave optically differently in a way that interacts poorly with the BC correction. The 895 kg/m³ value is also the physically correct density for summer FYI (see §Preset calibration notes in `presets.py`).

**Selected parameterisation: BC=1000 ppb, rho=895 kg/m³**

This gives VIS RMSE=0.068, NIR RMSE=0.028, BBA RMSE=0.062. NIR validates well against observations (6/6 depth bins within 0.10 tolerance). VIS is improved by 5× versus the clean-water model.

### What "1000 ppb BC" physically means

The 1000 ppb black carbon concentration must be interpreted as an **effective light-absorbing particle (LAP) concentration**, not a pure black carbon measurement. The actual pond floor darkening arises from a mixture:

- Black carbon: broadband absorber, dominates the overall level reduction
- Mineral dust (cryoconite): also broadband but weaker per ppb
- Chlorophyll-a (algae): selectively absorbs at 680 nm (explains residual 600–700 nm drop not reproduced by pure BC)
- Dissolved organic carbon: UV and blue absorption, minor in 400–1000 nm

A pure BC model matches the broadband and NIR well but slightly over-predicts relative to 600–700 nm (chlorophyll band). The 1000 ppb therefore represents the BC-equivalent optical depth of all these absorbers combined — consistent with the finding that SHEBA-calibrated models (Briegleb & Light 2007) implicitly encode similar total optical depth in their tuned IOPs without naming any specific absorber.

For comparison, the CCSM4/CICE pond parameterisation produces visible broadband pond albedos of ~0.30–0.40 for shallow-to-medium ponds (Holland et al. 2012, *J. Climate* 25, 1413–1430) — the same range our calibrated model achieves.

### Known limitations

| Approximation | Effect | When it matters |
|---|---|---|
| BC as proxy for mixed absorbers | Slightly wrong spectral shape at 600–700 nm (missing chlorophyll red edge) | Spectral analysis |
| Uniform BC throughout top 5cm ice layer | Overestimates darkening if LAPs are surface-concentrated | Very shallow ponds |
| No depth-varying LAP concentration | LAP loading fixed at 1000 ppb regardless of season or location | Regional application |
| Water–ice interface at the pond floor neglected | Normal-incidence reflectance of a 1.33 → 1.31 step is ~6×10⁻⁵ | Never |
| Flat pond surface | No wind roughening of the specular term | Large SZA, windy ponds |
| Pure liquid water at 0°C | Neglects pond water turbidity (DOM, suspended particles) | Turbid ponds |

The NIR prediction is robust — it depends only on water absorption physics and is nearly independent of floor properties for ponds deeper than ~5 cm.

### The pond surface is the refractive boundary (2026-11-12)

For a ponded column `[5, 4, 4]` the Fresnel interface used to land on the ice *under* the pond and be computed as air→ice, as if the water above were absent. Two consequences, both wrong:

- **No reflection off the water.** Wherever the pond is optically thick, the only light that can return is the specular reflection from its surface, so the old model gave a NIR albedo of exactly zero. Real water reflects 2% at normal incidence, 6% at SZA 60° and 13% at SZA 70°.
- **No light recycling under the water surface.** Upwelling diffuse light meets the water–air interface from below, where 47% of it is returned (`1 − (1 − R_a)/n²`). That traps visible light over the pond floor and raises the sensitivity of pond albedo to the floor's properties.

The interface is now air→water at the pond top, with water's own refractive index (Rowe et al. 2020) for the direct-beam Fresnel coefficients and the refracted angle, and exact diffuse reflectances from `biosnicar.classes.ice.diffuse_fresnel`. The water→ice step at the floor is neglected (table above). The total-internal-reflection override the solver applies to ice near 3 µm is skipped for water, whose real index never drops below 1.11.

Effect on broadband albedo is solar-geometry dependent rather than uniform: about −0.01 at SZA 50° (recycling dominates), about +0.06 at SZA 70° (specular dominates), and +0.01 to +0.02 under diffuse sky.

**A related inaccuracy in the inherited ice tables, not changed here.** `fl_reflection_diffuse.npz` (from SNICAR-ADv4) agrees with `diffuse_fresnel` to 2×10⁻⁷ for diffuse light from above, but its from-below values sit ~0.02 under the `n²` relation (0.427 against 0.451 at 0.7 µm; CICE uses 0.455 for n = 1.31). Applying the exact value to ice lowers broadband albedo by 0.003–0.005 on the sea-ice presets and on glacier ice. It is left alone because the glacier-ice benchmarks are pinned to those tables — a decision for the forward model as a whole, not for the sea-ice extension.

---

## YAML configuration

Add `LAYER_TYPE: 4` to enable sea ice. The required additional fields are:
`SEA_ICE_TEMPERATURE`, `SEA_ICE_SALINITY`, and `SEA_ICE_BUBBLE_RADIUS` (all per-layer lists).

```yaml
ICE:
  DZ: [0.05, 1.45]          # layer thicknesses (m)
  LAYER_TYPE: [4, 4]        # 4 = sea ice
  RHO: [920, 915]           # bulk density (kg/m³)
  SEA_ICE_TEMPERATURE: [-25, -10]   # temperature (°C)
  SEA_ICE_SALINITY:    [12, 8]      # bulk salinity (psu)
  SEA_ICE_BUBBLE_RADIUS: [100, 200] # air bubble radius (µm)
  RF: 2                     # Picard 2016 refractive index (recommended)
  # Other fields (RDS, SHP, etc.) are required by the parser but unused for layer_type=4
  RDS: [500, 500]
  SHP: [0, 0]
  CDOM: [0, 0]
  WATER_COATING: [0, 0]
  LWC: [0, 0]
  LWC_PCT_BBL: 0
  HEX_SIDE: [10000, 10000]
  HEX_LENGTH: [10000, 10000]
  SHP_FCTR: [0, 0]
  AR: [0, 0]
```

*Note*: Use the adding-doubling solver (the default) — sea ice requires Fresnel interfaces at the air-ice boundary, which the Toon solver does not handle correctly.

---

## Python API

Sea ice uses the same `run_model()` entry point as terrestrial ice.
Both return an `Outputs` object with identical attribute names.

### Quick start with a preset

```python
from biosnicar import run_model, FYI_WINTER_BARE

# By preset dict
outputs = run_model(preset=FYI_WINTER_BARE, solzen=60)

# Or by preset name
outputs = run_model(preset="FYI_WINTER_BARE", solzen=60)

print(f"Broadband albedo:  {outputs.BBA:.3f}")       # or outputs.broadband
print(f"Visible albedo:    {outputs.BBAVIS:.3f}")     # or outputs.visible
print(f"NIR albedo:        {outputs.BBANIR:.3f}")     # or outputs.nir

import matplotlib.pyplot as plt
plt.plot(outputs.wavelengths, outputs.albedo)         # or outputs.spectrum
plt.xlabel("Wavelength (µm)")
plt.ylabel("Spectral albedo")
plt.title("FYI Winter Bare — SZA 60°")
plt.xlim(0.3, 2.5)
plt.ylim(0, 1)
plt.show()
```

### Custom column — flat kwargs

Sea ice uses the same flat keyword-argument style as terrestrial ice.
Simply add `layer_type=4` and three sea-ice-specific lists:

```python
from biosnicar import run_model

# Snow on sea ice — identical call style to snow on glacier ice
outputs = run_model(
    solzen=60, direct=1, incoming=2,
    layer_type=[0, 4, 4],              # snow + two sea ice layers
    dz=[0.10, 0.05, 1.50],
    rds=[200, 500, 500],               # snow grain; 500 is unused dummy for sea ice
    rho=[300, 895, 895],
    sea_ice_salinity=[None, 10, 6],    # None for snow layer
    sea_ice_temperature=[None, -20, -8],
    sea_ice_bubble_radius=[None, 100, 200],
    black_carbon=500,                  # impurities on any/all layers
)
print(outputs.BBA)
```

The illumination parameters match `run_model()` exactly:
- `solzen` — solar zenith angle in degrees (1–89)
- `direct` — 1 = direct beam (clear), 0 = diffuse (cloudy/overcast)
- `incoming` — irradiance spectrum index (0–6): 0=mid-lat winter, 1=mid-lat summer, 2=sub-Arctic winter, 3=sub-Arctic summer, 4=summit, 5=high mountain, 6=tropical

### `Outputs` attributes (same as terrestrial ice)

| Attribute | Alias | Description |
|---|---|---|
| `.BBA` | `.broadband` | Flux-weighted broadband albedo |
| `.BBAVIS` | `.visible` | Flux-weighted visible (0.205–0.75 µm) albedo |
| `.BBANIR` | `.nir` | Flux-weighted NIR (0.75–4.995 µm) albedo |
| `.albedo` | `.spectrum` | Spectral albedo array, shape (480,) |
| `.wavelengths` | | Wavelength grid in µm, shape (480,) |
| `.flx_slr` | | Solar flux spectrum, shape (480,) |
| `.to_platform(name)` | | Satellite/GCM band convolution |
| `.plot()` | | Spectral albedo plot |

---

## The three presets

Presets are plain dicts of `run_model()` kwargs. Use them as a starting
point and layer overrides on top:

```python
from biosnicar import run_model, FYI_WINTER_BARE, FYI_WINTER_SNOW, MYI_WINTER_BARE

# Merge preset with additional overrides
outputs = run_model(preset=FYI_WINTER_SNOW, solzen=75, black_carbon=2000)
```

### `FYI_WINTER_BARE`
Bare first-year ice in winter. Two sea-ice layers (T=−25/−20°C, S=12/8 psu, ρ=895 kg/m³). Typical BBA: 0.50–0.60.

### `FYI_WINTER_SNOW`
First-year ice with 15 cm of snow (ρ=250 kg/m³, grain=200 µm). Calibrated against SHEBA April 1998 data (RMSE ≈ 0.025–0.038, 7/7 pass). Typical BBA: 0.75–0.85.

### `MYI_WINTER_BARE`
Bare multiyear ice in winter. Two layers (T=−20/−8°C, S=1/3 psu, ρ=860/870 kg/m³, large bubbles 500/700 µm). Typical BBA: 0.49–0.52.

### `FYI_POND_SHALLOW`
Shallow melt pond (10 cm) on summer FYI. Floor ice: rho=895, T=−5°C, BC=1000 ppb (calibrated LAP loading). BBA ≈ 0.38–0.42, NIR < 0.15. Calibrated to Morassutti (1995).

### `FYI_POND_DEEP`
Deep melt pond (40 cm) on summer FYI. Same floor parameterisation as FYI_POND_SHALLOW. BBA ≈ 0.28–0.32, NIR < 0.04.

---

## Melt ponds (layer_type=5)

Melt ponds form when surface snow and ice melt in summer and the meltwater collects in topographic depressions. They can reduce surface albedo dramatically — a 20 cm deep pond has BBA ≈ 0.35, vs ~0.78 for snow-covered FYI.

### Physics

A melt pond layer is modelled as **liquid water with near-zero scattering**:

```
absorption coefficient:  α(λ) = 4π k_water(λ) / λ    [m⁻¹]
scattering coefficient:  β(λ) ≈ 0.003 × (0.55/λ_µm)⁴ [m⁻¹]  (Rayleigh, for stability)
single-scattering albedo: ω ≈ β / (α + β) → 0 in NIR, ~0.01 in visible
```

Water optical constants k(λ) come from Rowe et al. (2020) at 0 °C — appropriate for melt pond water (near 0 °C in summer). Key properties:
- **Visible (400–700 nm)**: water is nearly transparent. Most light passes through and reflects from the ice below. Pond VIS ≈ ice_VIS × exp(−2α × depth).
- **NIR (700–1000 nm)**: strong O-H absorption. A 5 cm pond reduces NIR by ~50%; a 20 cm pond reduces it by ~90%.
- **SWIR (> 1000 nm)**: essentially opaque even at < 1 cm depth.

### Usage

```python
from biosnicar import run_model

# Calibrated pond presets (black_carbon=1000 ppb in pond floor ice)
outputs = run_model(preset="FYI_POND_SHALLOW", solzen=60)   # 10 cm pond
outputs = run_model(preset="FYI_POND_DEEP",    solzen=60)   # 40 cm pond

# Custom pond — adjust BC concentration as needed for your site conditions
# BC=0: clean-water model (NIR good, VIS ~0.4 too high)
# BC=1000 (default): calibrated to Morassutti 1995 summer Arctic ponds
# BC range: 200–1500 ppb encompasses most observed Arctic summer conditions
outputs = run_model(
    solzen=60,
    layer_type=[5, 4, 4],               # pond on top of FYI
    dz=[0.20, 0.05, 1.45],              # 20 cm pond
    rds=[500, 500, 500],
    rho=[1000, 895, 895],               # 1000 kg/m³ liquid; 895 corrected FYI
    sea_ice_salinity=[None, 8, 6],
    sea_ice_temperature=[None, -5, -5], # summer FYI
    sea_ice_bubble_radius=[None, 100, 200],
    black_carbon=[0, 1000, 0],          # 1000 ppb LAPs in floor ice; 0 in pond
)
print(f"BBA={outputs.BBA:.3f}  VIS={outputs.BBAVIS:.3f}  NIR={outputs.BBANIR:.3f}")
```

The `black_carbon` parameter here represents the total **effective light-absorbing particle (LAP) loading** of the pond floor ice — a mixture of BC, cryoconite, mineral dust, and biological pigments expressed as a BC-equivalent concentration. The value 1000 ppb is calibrated to Morassutti (1995) summer Arctic pond observations and consistent with the range 100–2000 ppb reported for biologically active Arctic sea ice surface layers (Light et al. 1998; Marks & King 2013).

### Expected albedo vs pond depth (calibrated model, BC=1000 ppb)

| Depth | BBA | VIS (400–700 nm) | NIR (700–1000 nm) | Obs BBA (Morassutti) |
|---|---|---|---|---|
| 0 (bare FYI) | ~0.51 | ~0.73 | ~0.27 | — |
| ~2.5 cm | ~0.47 | ~0.69 | ~0.21 | ~0.43 |
| ~7.5 cm | ~0.43 | ~0.64 | ~0.14 | ~0.24 |
| ~15 cm | ~0.39 | ~0.60 | ~0.09 | ~0.21 |
| ~25 cm | ~0.36 | ~0.56 | ~0.06 | ~0.18 |
| ~40 cm | ~0.32 | ~0.50 | ~0.03 | ~0.20 |

*SZA=60°, rho=895, T=−5°C, BC=1000 ppb. Obs BBA from Morassutti (1995) depth-bin means.*

BBA is still somewhat overestimated for shallow ponds because the chlorophyll-a red-edge absorption (~680 nm) is not captured by BC alone, and some ponds have even darker bottoms than the calibrated mean. NIR validation is robust (RMSE=0.028).

### Validation summary against Morassutti (1995)

| Depth bin | Obs NIR | Model NIR | NIR Δ | Obs VIS | Model VIS | VIS Δ |
|---|---|---|---|---|---|---|
| 0–5 cm | 0.292 | 0.332 | +0.040 | 0.574 | 0.652 | +0.078 |
| 5–10 cm | 0.102 | 0.198 | +0.097 | 0.375 | 0.586 | +0.211 |
| 10–20 cm | 0.066 | 0.114 | +0.048 | 0.354 | 0.534 | +0.180 |
| 20–30 cm | 0.031 | 0.059 | +0.028 | 0.331 | 0.485 | +0.154 |
| 30–50 cm | 0.035 | 0.032 | −0.002 | 0.355 | 0.445 | +0.090 |
| > 50 cm | 0.026 | 0.012 | −0.013 | 0.342 | 0.385 | +0.043 |

NIR pass rate (|Δ| ≤ 0.10): **6/6** bins. VIS is improved by 5× relative to the clean-water model (mean RMSE 0.068 vs 0.362) but a residual ~+0.10–0.20 overestimate remains, reflecting the chlorophyll signal and site-to-site variability in pond bottom properties not captured by a single BC concentration. See `tests/validation_data/morassutti1995/` for the full validation script and figures.

---

## Validation summary

### Sea ice (winter, snow-covered and bare)

Validated against Grenfell & Light (2007) SHEBA spectral albedo, doi:10.5065/D6765CQ1.
Spring snow (April–May 1998, ~76°N). See `docs/sea_ice_validation.md` for full results.

| Condition | Observed BBA (400–1000 nm) | Model BBA | Spectral RMSE | Pass? |
|---|---|---|---|---|
| FYI snow-covered (Apr) | 0.912–0.935 | 0.934–0.941 | 0.023–0.031 | ✓ 7/7 |
| FYI bare (Aug–Sep, summer ice) | 0.618–0.809 | ~0.56 | ~0.19 | n/a — season mismatch |
| MYI bare (inferred) | 0.50–0.65 | ~0.49 | — | Qualitative |

### Melt ponds

Validated against Morassutti (1995) Canadian Arctic melt pond data (NSIDC G01169, doi:10.7265/N55Q4T1C). 504 records, summer 1994, Barrow Strait, Nunavut, 6 bands (400–1000 nm). See `tests/validation_data/morassutti1995/validate_morassutti1995.py`.

| Depth bin | Observed NIR | Model NIR | Observed BBA | Model BBA | Notes |
|---|---|---|---|---|---|
| 5–10 cm | 0.095 | ~0.10 | 0.229 | ~0.41 | NIR ✓, VIS high (dark bottoms) |
| 10–20 cm | 0.058 | ~0.05 | 0.205 | ~0.36 | NIR ✓, VIS high |
| 20–30 cm | 0.032 | ~0.02 | 0.182 | ~0.33 | NIR ✓, VIS high |

NIR validates well. BBA overestimated by ~0.15–0.18 because the model assumes clear water with white sea-ice bottom; real ponds have dark bottoms from algae and sediment (see *Known approximations*).

---

## References

- Cox, G. F. N. & Weeks, W. F. (1983). Equations for determining the gas and brine volumes in sea-ice samples. *J. Glaciology*, 29(102), 306–316.
- Grenfell, T. C. & Light, B. (2007). SHEBA Spectral Albedo. UCAR/NCAR EOL. doi:10.5065/D6765CQ1
- Light, B. et al. (2004). Two-dimensional Monte Carlo model of radiative transfer in sea ice. *J. Geophys. Res.*, 109, C03028.
- Maxwell Garnett, J. C. (1904). Colours in metal glasses and in metallic films. *Phil. Trans. R. Soc. Lond. A*, 203, 385–420.
- Morassutti, M. (1995). Sea Ice Melt Pond Data from the Canadian Arctic. NSIDC G01169. doi:10.7265/N55Q4T1C
- Pegau, W. S., Gray, D. & Zaneveld, J. R. V. (1997). Absorption and attenuation of visible and near-infrared light in water. *Limnol. Oceanogr.*, 42(3), 443–452.
- Perovich, D. K. et al. (2002). Seasonal evolution of the albedo of multiyear Arctic sea ice. *J. Geophys. Res.*, 107(C10), 8044.
- Picard, G. et al. (2016). Refinement of the ice absorption spectrum. *J. Glaciology*.
- Quan, X. & Fry, E. S. (1995). Empirical equation for the refractive index of seawater. *Appl. Optics*, 34, 3477.
- Rowe, P. M. et al. (2020). Refractive index of liquid water at 0 °C. *J. Geophys. Res.*, 125, e2019JD031822.
- Sihvola, A. (1999). *Electromagnetic Mixing Formulas and Applications*. IEE.
- Timco, G. W. & Frederking, R. M. W. (1996). A review of sea ice density. *Cold Reg. Sci. Tech.*, 24(1), 1–6.

---

## Cross-references to other BioSNICAR documentation

- [METHODS.md](METHODS.md) — emulator architecture, inversion framework, band convolution.
- [SUBSURFACE.md](SUBSURFACE.md) — subsurface light field calculation.
- [BANDS.md](BANDS.md) — satellite band convolution (Sentinel-2, MODIS, etc.).
- [PLOTTING.md](PLOTTING.md) — spectral plotting utilities.
