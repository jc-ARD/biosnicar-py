# Data Directory

All data required by biosnicar-py. The 480-band spectral grid runs from 0.205 to 4.995 um in 0.01 um steps.

```python
import numpy as np
wavelengths = np.arange(0.205, 4.999, 0.01)  # shape (480,)
```

---

## Directory structure

```
data/
├── OP_data/480band/         Core optical property data
│   ├── lap.npz              Light-absorbing particle (impurity) optical properties
│   ├── fsds.npz             Downwelling spectral solar flux (illumination)
│   ├── rfidx_ice.npz        Ice refractive indices
│   ├── fl_reflection_diffuse.npz  Fresnel diffuse reflectances
│   ├── luts/                 Grain/bubble scattering look-up tables
│   └── r_sfc/               Surface reflectance spectra (CSV)
├── OP_data/                  Additional optical constants (CSV)
├── band_srfs/                Satellite/GCM spectral response functions (CSV)
├── emulators/                Saved neural-network emulators (.npz)
├── pigments/                 Individual pigment mass absorption cross-sections (CSV)
└── additional_data/          Field spectroscopy and pigment reference data (CSV)
```

---

## `OP_data/480band/lap.npz` — Impurity optical properties

The main archive of light-absorbing particle (LAP) optical properties. Contains 40 impurity types, each with 2-5 spectral arrays.

### Key naming convention

Keys follow the pattern `{file_stem}__{property}`, where `file_stem` identifies the impurity and `property` is one of:

| Property          | Description                                             | Units         |
| ----------------- | ------------------------------------------------------- | ------------- |
| `ext_cff_mss`     | Mass extinction coefficient                             | m2 kg-1       |
| `ext_xsc`         | Extinction cross-section (per cell, for algae)          | m2 cell-1     |
| `ext_cff_mss_ncl` | Mass extinction of core only (sulfate-coated particles) | m2 kg-1       |
| `ss_alb`          | Single-scattering albedo                                | dimensionless |
| `asm_prm`         | Asymmetry parameter                                     | dimensionless |

All arrays are shape `(480,)` float64 (one value per wavelength band).

### Special key

- `_names` — array of all 40 file stems (shape `(40,)`, dtype string). Use this to discover available impurities programmatically:

```python
d = np.load("data/OP_data/480band/lap.npz", allow_pickle=False)
print(list(d["_names"]))
# Access black carbon mass extinction:
mac_bc = d["bc_ChCB_rn40_dns1270__ext_cff_mss"]
```

### Available impurities

**Black carbon**
| File stem                      | Description                                                          |
| ------------------------------ | -------------------------------------------------------------------- |
| `bc_ChCB_rn40_dns1270`         | Black carbon (Chang & Charalampopoulos 1990, r=40nm, rho=1270 kg/m3) |
| `brC_Kirch_BCsd`               | Brown carbon (Kirchstetter et al.)                                   |
| `brC_Kirch_BCsd_slfcot`        | Brown carbon, sulfate-coated                                         |
| `mie_sot_ChC90_dns_1317`       | Mie soot (Chang & Charalampopoulos 1990, rho=1317 kg/m3)             |
| `miecot_slfsot_ChC90_dns_1317` | Mie soot, sulfate-coated                                             |

**Mineral dust — Balkanski (global)**
| File stem                                      | Description                                             |
| ---------------------------------------------- | ------------------------------------------------------- |
| `dust_balkanski_central_size1` through `size5` | Balkanski et al., 5 size bins, central refractive index |

**Mineral dust — Greenland**
| File stem                                      | Description                                  |
| ---------------------------------------------- | -------------------------------------------- |
| `dust_greenland_central_size1` through `size5` | Greenland local dust, 5 size bins            |
| `dust_greenland_Cook_CENTRAL_20190911`         | Cook et al. 2019 (central estimate)          |
| `dust_greenland_Cook_HIGH_20190911`            | Cook et al. 2019 (high estimate)             |
| `dust_greenland_Cook_LOW_20190911`             | Cook et al. 2019 (low estimate)              |
| `dust_greenland_Cook_CENTRAL_20190911_OLD`     | Cook et al. 2019 (old, 470-band, do not use) |

**Mineral dust — Skiles (Colorado)**
| File stem                           | Description                |
| ----------------------------------- | -------------------------- |
| `dust_skiles_size1` through `size5` | Skiles et al., 5 size bins |

**Mineral dust — Mars**
| File stem                         | Description                        |
| --------------------------------- | ---------------------------------- |
| `dust_mars_size1` through `size5` | Martian dust analogue, 5 size bins |

**Volcanic ash**
| File stem                                     | Description                        |
| --------------------------------------------- | ---------------------------------- |
| `volc_ash_eyja_central_size1` through `size5` | Eyjafjallajokull 2010, 5 size bins |
| `volc_ash_mtsthelens_20081011`                | Mt St Helens                       |

**Algae**
| File stem                                        | Description                                        |
| ------------------------------------------------ | -------------------------------------------------- |
| `ice_algae_empirical_Chevrollier2023`            | Glacier algae (empirical, Chevrollier et al. 2023) |
| `snow_algae_empirical_Chevrollier2023`           | Snow algae (empirical, Chevrollier et al. 2023)    |
| `Cook2020_glacier_algae_4_40`                    | Glacier algae (Cook et al. 2020, 4-40 um)          |
| `Glacier_Algae_IS`                               | Glacier algae (in situ)                            |
| `snw_alg_r025um_chla020_chlb025_cara150_carb140` | Snow algae (bio-optical model, r=25um)             |

### Linking to `inputs.yaml`

The `FILE` field in `inputs.yaml` IMPURITIES section must match a file stem from `_names`, with `.npz` appended:

```yaml
IMPURITIES:
  black_carbon:
    FILE: "bc_ChCB_rn40_dns1270.npz"   # matches stem bc_ChCB_rn40_dns1270
    UNIT: 0                              # 0 = mass-based (ppb), 1 = cell-based (cells/mL)
    ...
```

The model strips the `.npz` extension and uses the stem to look up `{stem}__ext_cff_mss` (or `ext_xsc` for UNIT=1 algae) from `lap.npz`.

---

## `OP_data/480band/fsds.npz` — Solar spectral irradiance

Downwelling spectral flux for 7 standard atmospheres, each with clear-sky and cloudy variants, and clear-sky variants for every integer SZA from 0-89 degrees. 647 arrays total, each shape `(480,)`.

### Key naming convention

`swnb_480bnd_{atm}_{sky}` or `swnb_480bnd_{atm}_{sky}_SZA{nn}`

| Atmosphere code | Description                       |
| --------------- | --------------------------------- |
| `mlw`           | Mid-latitude winter               |
| `mls`           | Mid-latitude summer               |
| `saw`           | Sub-arctic winter                 |
| `sas`           | Sub-arctic summer                 |
| `smm`           | Summit, Greenland                 |
| `hmn`           | High mountain                     |
| `trp`           | Tropical                          |
| `toa`           | Top of atmosphere (no atmosphere) |

| Sky code      | Description                                  |
| ------------- | -------------------------------------------- |
| `clr`         | Clear sky (default SZA from YAML)            |
| `cld`         | Cloudy (diffuse)                             |
| `clr_SZA{nn}` | Clear sky at solar zenith angle `nn` degrees |

```python
d = np.load("data/OP_data/480band/fsds.npz", allow_pickle=False)
# Clear-sky mid-latitude summer at SZA=50
flux = d["swnb_480bnd_mls_clr_SZA50"]
```

### Special key

- `_names` — array of all 647 key names

---

## `OP_data/480band/rfidx_ice.npz` — Ice refractive indices

Complex refractive index of ice from three sources plus CO2 ice. Shape `(480,)` each.

| Key         | Description                          |
| ----------- | ------------------------------------ |
| `re_Wrn84`  | Real part, Warren 1984               |
| `im_Wrn84`  | Imaginary part, Warren 1984          |
| `re_Wrn08`  | Real part, Warren & Brandt 2008      |
| `im_Wrn08`  | Imaginary part, Warren & Brandt 2008 |
| `re_Pic16`  | Real part, Picard et al. 2016        |
| `im_Pic16`  | Imaginary part, Picard et al. 2016   |
| `re_co2ice` | Real part, CO2 ice                   |
| `im_co2ice` | Imaginary part, CO2 ice              |
| `wvl`       | Wavelength grid (um)                 |

---

## `OP_data/480band/fl_reflection_diffuse.npz` — Fresnel reflectances

Diffuse Fresnel reflectances for the ice-air interface. Shape `(480,)` each.

| Key                  | Description                                           |
| -------------------- | ----------------------------------------------------- |
| `R_dif_fa_ice_Wrn84` | Diffuse reflectance, forward (air->ice), Warren 1984  |
| `R_dif_fb_ice_Wrn84` | Diffuse reflectance, backward (ice->air), Warren 1984 |
| `R_dif_fa_ice_Wrn08` | Forward, Warren & Brandt 2008                         |
| `R_dif_fb_ice_Wrn08` | Backward, Warren & Brandt 2008                        |
| `R_dif_fa_ice_Pic16` | Forward, Picard et al. 2016                           |
| `R_dif_fb_ice_Pic16` | Backward, Picard et al. 2016                          |

---

## `OP_data/480band/luts/` — Scattering look-up tables

Pre-computed Mie/geometric optics results for ice grains, bubbles, and water droplets. Each `.npz` contains 2D arrays indexed by particle size (rows) and wavelength (columns).

### Ice spheres

Files: `ice_sphere_Pic16.npz`, `ice_sphere_BH83_Wrn08.npz`, `ice_sphere_BH83_Wrn84.npz`, `ice_sphere_BH83_Pic16.npz`

| Key           | Shape      | Description                   |
| ------------- | ---------- | ----------------------------- |
| `radii`       | `(N,)`     | Grain radii (um), int32       |
| `ext_cff_mss` | `(N, 480)` | Mass extinction coefficient   |
| `ext_cff_vlm` | `(N, 480)` | Volume extinction coefficient |
| `sca_cff_vlm` | `(N, 480)` | Volume scattering coefficient |
| `ss_alb`      | `(N, 480)` | Single-scattering albedo      |
| `asm_prm`     | `(N, 480)` | Asymmetry parameter           |

N = 1471 radii for `ice_sphere_BH83_*`, varies for others. `BH83` = Bohren & Huffman 1983 absorption corrections.

### Bubbly ice (air-in-ice)

Files: `bubbly_air.npz`, `bubbly_air_BH83.npz`

| Key           | Shape        | Description                   |
| ------------- | ------------ | ----------------------------- |
| `radii`       | `(549,)`     | Bubble radii (um)             |
| `sca_cff_vlm` | `(549, 480)` | Volume scattering coefficient |
| `asm_prm`     | `(549, 480)` | Asymmetry parameter           |

No extinction/absorption keys — air bubbles are purely scattering.

### Bubbly ice (water-in-ice)

File: `bubbly_water.npz` — same keys as bubbly_air, for liquid water inclusions.

### Hexagonal columns

Files: `hex_Wrn08.npz`, `hex_Wrn84.npz`, `hex_Pic16.npz`

| Key           | Shape        | Description                       |
| ------------- | ------------ | --------------------------------- |
| `sides`       | `(261,)`     | Hexagonal column side length (um) |
| `lengths`     | `(261,)`     | Column length (um)                |
| `ext_cff_mss` | `(261, 480)` | Mass extinction coefficient       |
| `ss_alb`      | `(261, 480)` | Single-scattering albedo          |
| `asm_prm`     | `(261, 480)` | Asymmetry parameter               |

### Water spheres

File: `water_sphere.npz`

| Key           | Shape         | Description                   |
| ------------- | ------------- | ----------------------------- |
| `radii`       | `(1646,)`     | Droplet radii (um)            |
| `ext_cff_vlm` | `(1646, 480)` | Volume extinction coefficient |
| `sca_cff_vlm` | `(1646, 480)` | Volume scattering coefficient |
| `asm_prm`     | `(1646, 480)` | Asymmetry parameter           |

---

## `emulators/` — Saved emulators

Neural-network emulators stored as `.npz` with weights, PCA decomposition, and metadata.

### Keys in an emulator `.npz`

| Key                        | Shape      | Description                     |
| -------------------------- | ---------- | ------------------------------- |
| `weights_0` .. `weights_N` | varies     | Neural network weight matrices  |
| `biases_0` .. `biases_N`   | varies     | Neural network bias vectors     |
| `pca_components`           | `(K, 480)` | PCA component vectors           |
| `pca_mean`                 | `(480,)`   | PCA mean spectrum               |
| `input_min`                | `(P,)`     | Input normalisation minimum     |
| `input_max`                | `(P,)`     | Input normalisation maximum     |
| `flx_slr`                  | `(480,)`   | Solar flux used during training |
| `metadata`                 | scalar     | JSON string with build info     |

### Metadata JSON fields

```python
import json, numpy as np
d = np.load("data/emulators/glacier_ice_4param.npz", allow_pickle=False)
meta = json.loads(str(d["metadata"]))
# Keys: param_names, bounds, n_samples, n_pca_components,
#        training_r2, build_timestamp, fixed_overrides, solver
```

### Loading an emulator

```python
from biosnicar.emulator import Emulator
emu = Emulator.load("data/emulators/glacier_ice_4param.npz")
albedo = emu.predict(rds=1000, rho=600, black_carbon=5000, glacier_algae=50000)
```

---

## `band_srfs/` — Spectral response functions

CSV files with the spectral response function for each satellite/GCM platform. First column is `wavelength_um` (matching the 480-band grid), remaining columns are per-band weights.

| File                 | Platform        | Bands             |
| -------------------- | --------------- | ----------------- |
| `sentinel2_msi.csv`  | Sentinel-2 MSI  | B1-B12 (13 bands) |
| `landsat8_oli.csv`   | Landsat 8 OLI   | B1-B7             |
| `sentinel3_olci.csv` | Sentinel-3 OLCI | Oa01-Oa21         |

GCM platforms (CESM, MAR, HadCM3) use wavelength-interval averaging and are defined in code (`biosnicar/bands/`), not as SRF files.

---

## `pigments/` — Pigment absorption spectra

Individual pigment mass absorption cross-sections at 1 nm resolution (4799-4800 rows covering 200-5000 nm). Single-column CSV files (no header), units m2 mg-1.

Used by the bio-optical sub-model to build cell-level absorption from pigment composition.

| File                            | Pigment                                  |
| ------------------------------- | ---------------------------------------- |
| `chl-a.csv`                     | Chlorophyll a                            |
| `chl-b.csv`                     | Chlorophyll b                            |
| `total_astaxanthin.csv`         | Total astaxanthin                        |
| `trans_astaxanthin.csv`         | Trans-astaxanthin                        |
| `trans_astaxanthin_ester.csv`   | Trans-astaxanthin ester                  |
| `cis_astaxanthin_monoester.csv` | Cis-astaxanthin monoester                |
| `cis_astaxanthin_diester.csv`   | Cis-astaxanthin diester                  |
| `alloxanthin.csv`               | Alloxanthin                              |
| `antheraxanthin.csv`            | Antheraxanthin                           |
| `lutein.csv`                    | Lutein                                   |
| `neoxanthin.csv`                | Neoxanthin                               |
| `violaxanthin.csv`              | Violaxanthin                             |
| `zeaxanthin.csv`                | Zeaxanthin                               |
| `Photop_carotenoids.csv`        | Photoprotective carotenoids              |
| `Photos_carotenoids.csv`        | Photosynthetic carotenoids               |
| `pheophytin.csv`                | Pheophytin                               |
| `ppg.csv`                       | Purpurogallin (phenolic pigment)         |
| `pckg_GA.csv`                   | Pigment packaging factor — glacier algae |
| `pckg_SA.csv`                   | Pigment packaging factor — snow algae    |

---

## `OP_data/` — Additional optical constants (CSV)

| File                                                | Description                                |
| --------------------------------------------------- | ------------------------------------------ |
| `wavelengths.csv`                                   | 480-band wavelength grid                   |
| `ice_optical_constants.csv`                         | Ice optical constants (n, k)               |
| `ice_n.csv`, `ice_k.csv`                            | Real and imaginary refractive index of ice |
| `water_RI.csv`                                      | Liquid water refractive index              |
| `Refractive_Index_Ice_Warren_1984.csv`              | Warren 1984 ice RI                         |
| `Refractive_Index_Liquid_Water_Segelstein_1981.csv` | Segelstein 1981 water RI                   |
| `k_ice_480.csv`                                     | Imaginary RI of ice, 480 bands             |
| `k_cdom_240_750.csv`                                | CDOM absorption coefficient                |

## `OP_data/480band/r_sfc/` — Surface reflectance spectra

| File                              | Description                                       |
| --------------------------------- | ------------------------------------------------- |
| `blue_ice_spectrum_s10290721.csv` | Blue ice reflectance (default underlying surface) |
| `rain_polished_ice_spectrum.csv`  | Rain-polished ice reflectance                     |

---

## `additional_data/` — Field and lab reference data

Empirical measurements from field campaigns and laboratory studies. Used for validation and the bio-optical sub-model.

| File                                   | Description                           |
| -------------------------------------- | ------------------------------------- |
| `Albedo_master.csv`                    | Field albedo spectra collection       |
| `HCRF_master_16171819.csv`             | HCRF spectra from 2016-2019 campaigns |
| `ARF_master.csv`                       | Anisotropy reflectance factor         |
| `Spectra_Metadata.csv`                 | Metadata for field spectra            |
| `Chlorophyll_a_m2mg.csv`               | Chl-a mass absorption cross-section   |
| `PhenolicPigment_m2mg.csv`             | Phenolic pigment MAC                  |
| `Photoprotective_carotenoids_m2mg.csv` | Photoprotective carotenoid MAC        |
| `Photosynthetic_carotenoids_m2mg.csv`  | Photosynthetic carotenoid MAC         |
| `InVivoPhenolData.csv`                 | In vivo phenol measurements           |
| `phenol_MAC.csv`                       | Phenol mass absorption coefficient    |
| `phenol_mac_correction.csv`            | Phenol MAC correction factors         |
| `phenol_mac_packaging_corrected.csv`   | Phenol MAC with packaging correction  |
| `pigmentMAC_200nm.csv`                 | Pigment MACs starting at 200 nm       |
| `pigmentMAC_250nm.csv`                 | Pigment MACs starting at 250 nm       |

---

## How to add a new impurity type

1. Obtain or compute the optical properties (MAC/extinction, SSA, asymmetry parameter) at the 480-band wavelength grid.

2. Choose a descriptive file stem (e.g. `my_dust_type1`). Create arrays named:
   - `my_dust_type1__ext_cff_mss` — mass extinction (m2/kg) for mass-based impurities (UNIT=0)
   - `my_dust_type1__ext_xsc` — extinction cross-section (m2/cell) for cell-based impurities (UNIT=1)
   - `my_dust_type1__ss_alb` — single-scattering albedo
   - `my_dust_type1__asm_prm` — asymmetry parameter

3. Add the new arrays to `lap.npz` and update the `_names` array:

   ```python
   import numpy as np

   # Load existing archive
   old = dict(np.load("data/OP_data/480band/lap.npz", allow_pickle=False))

   # Add new impurity
   old["my_dust_type1__ext_cff_mss"] = my_mac_array      # shape (480,)
   old["my_dust_type1__ss_alb"] = my_ssa_array
   old["my_dust_type1__asm_prm"] = my_asm_array

   # Update _names
   old_names = list(old["_names"])
   old_names.append("my_dust_type1")
   old["_names"] = np.array(old_names)

   # Save
   np.savez_compressed("data/OP_data/480band/lap.npz", **old)
   ```

4. Register it in `biosnicar/inputs.yaml`:

   ```yaml
   IMPURITIES:
     my_dust_type1:
       FILE: "my_dust_type1.npz"
       COATED: False
       UNIT: 0           # 0 = mass-based (ppb), 1 = cell-count (cells/mL)
       CONC: [0, 0]
   ```

5. Use it in the model:

   ```python
   from biosnicar import run_model
   outputs = run_model(my_dust_type1=5000)
   ```

## How t index into .npz files
                                                                                                                                                                                                                
# Load the archive (lazy — arrays aren't read until accessed)                                                                                                                                                    

```py
lap = np.load("Data/OP_data/480band/lap.npz")
```

# List all keys

```py
print(lap.files)
# → ['bc_ChCB_rn40_dns1270__ss_alb', 'bc_ChCB_rn40_dns1270__ext_cff_mss', ..., '_names']
```

Keys follow the pattern:  {stem}__{variable}

Variables are: ss_alb, asm_prm, ext_cff_mss (or ext_xsc for per-cell units)

e.g. Get mass absorption coefficient for a specific impurity

```py
mac = lap["bc_ChCB_rn40_dns1270__ext_cff_mss"]   # shape (480,)
```

Get all three optical properties for an impurity

```py
stem = "snow_algae_empirical_Chevrollier2023"
ss_alb  = lap[f"{stem}__ss_alb"]      # single-scattering albedo
asm_prm = lap[f"{stem}__asm_prm"]     # asymmetry parameter
ext     = lap[f"{stem}__ext_xsc"]     # extinction cross-section (per-cell)
```

List all impurity stems

```py
names = lap["_names"]
# → array(['bc_ChCB_rn40_dns1270', 'snow_algae_empirical_Chevrollier2023', ...])

# The LUT files (in Data/OP_data/480band/luts/) work similarly but are
# indexed by radius. For example:

lut = np.load("Data/OP_data/480band/luts/ice_sphere_Pic16.npz")
radii = lut["radii"]            # shape (N,) — sorted integer radii in µm
ss_alb = lut["ss_alb"]          # shape (N, 480) — one row per radius

# To get data for radius 1000 µm:
idx = np.searchsorted(radii, 1000)
ss_alb_1000 = lut["ss_alb"][idx]  # shape (480,)
```


### Key points:
- np.load() returns a NpzFile object — index it like a dict with data["key"]
- .files lists all available keys
- Arrays are loaded lazily (only read from disk when you access a key)
- For lap.npz: keys are {stem}__{variable}, plus _names listing all stems
- For the LUT files: 2D arrays indexed by a radii (or sides/lengths) coordinate array
