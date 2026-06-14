# BioSNICAR

A radiative transfer model for predicting the spectral albedo of snow and glacier ice, with integrated neural-network emulator, inverse retrieval framework and native support for common satellite and GCM platforms.

### **[Interactive classroom — biosnicar-classroom.netlify.app](https://biosnicar-classroom.netlify.app/)**

Learn the fundamentals of radiative transfer in snow and ice, explore real BioSNICAR spectra with interactive sliders, and try your hand at parameter retrieval — all in the browser, no installation required.

### **[Full documentation at biosnicar.vercel.app](https://biosnicar.vercel.app)**

<img src="./example-output.jpg" width=500>

BioSNICAR computes 480-band spectral albedo (0.2–5.0 µm) from the physical properties of ice (grain/bubble size, density, layer structure) and the concentrations of light-absorbing particles (black carbon, mineral dust, snow and glacier algae). Two radiative transfer solvers are available: the Toon et al. (1989) matrix method and a vectorised adding-doubling solver with Fresnel-reflecting layers (Briegleb & Light 2007; Dang et al. 2019; Whicker et al. 2022). A coupled bio-optical model enables calculation of algal optical properties from pigment inventories (Cook et al. 2017, 2020; Chevrollier et al. 2023).

**Key capabilities:**

- **Forward model** — multi-layer ice/snow with arbitrary impurity profiles, liquid water, grain shape corrections
- **Sea ice** — first-year/multiyear Arctic sea ice (Maxwell-Garnett brine effective medium), melt ponds, semi-transparent young ice, and open water (`biosnicar.sea_ice`)
- **Sea ice retrieval** — classify seven surface types and retrieve their physical properties from spectra or satellite bands, with quality flags, WMO/SIGRID-3 codes, and batch scene processing ([docs/SEA_ICE_RETRIEVAL.md](docs/SEA_ICE_RETRIEVAL.md))
- **Parameter sweeps** — Cartesian product over any input parameter, returns a pandas DataFrame
- **Platform bands** — map spectral albedo onto 8 satellite/GCM platforms (Sentinel-2/3, Landsat 8, MODIS, CESM, MAR, HadCM3) with spectral indices
- **Neural-network emulator** — ~50,000× faster than the full model, enabling optimisation and MCMC
- **Spectral inversion** — retrieve ice properties from observed albedo (spectral or satellite bands) with uncertainty estimates
- **Subsurface light** — PAR depth profiles, spectral heating rates at layer interfaces
- **Built-in plotting** — spectral albedo, PAR profiles, retrieval diagnostics, sensitivity analysis
- **Web app** — browser-based GUI at [bit.ly/bio-snicar](https://bit.ly/bio-snicar)

## Installation

Requires Python ≥ 3.10. Clone and install in a fresh environment:

```bash
git clone https://github.com/jmcook1186/biosnicar-py.git
cd biosnicar-py
pip install -r requirements.txt
pip install -e .
```

## Quick start

```python
from biosnicar import run_model

# Run with defaults
outputs = run_model()
print(outputs.BBA)           # broadband albedo
print(outputs.albedo[:5])    # first 5 spectral bands

# Override parameters
outputs = run_model(solzen=50, rds=1000, glacier_algae=10000)
outputs.plot(show=True)
```

`run_model()` is the single entry point for the forward model. It accepts keyword overrides, runs the full pipeline (setup → optical properties → impurity mixing → radiative transfer), and returns an `Outputs` object with spectral albedo, broadband albedos, absorbed fluxes, and subsurface light fields.

**Supported overrides:** `solzen`, `direct`, `incoming`, `rds`, `rho`, `dz`, `lwc`, `layer_type`, `shp`, `grain_ar`, `cdom`, `water`, `hex_side`, `hex_length`, `shp_fctr`, and impurity names (`black_carbon`, `snow_algae`, `glacier_algae`). Scalar ice parameters are broadcast to all layers; scalar impurity concentrations are applied to the first layer only. If a list override changes the number of layers, all per-layer attributes resize automatically.

Parameters not exposed as overrides (refractive index variant, surface reflectance file, RT approximation type) can be changed in `biosnicar/inputs.yaml` or via `input_file=`.


## Parameter sweeps

Run the model over the Cartesian product of any input parameters:

```python
from biosnicar.drivers.sweep import parameter_sweep

df = parameter_sweep(
    params={"solzen": [30, 40, 50, 60, 70], "rds": [100, 500, 1000, 2000]},
)
df.pivot_table(values="BBA", index="solzen", columns="rds").plot()
df.plot_sensitivity(show=True)
```


## Platform band convolution

Map spectral albedo onto satellite or GCM bands. Chains onto `run_model()` and `parameter_sweep()`:

```python
# Single run → Sentinel-2 bands + spectral indices
s2 = run_model(solzen=50, rds=1000).to_platform("sentinel2")
print(s2.B3, s2.NDSI)

# Sweep → band columns appended to DataFrame
df = parameter_sweep(
    params={"rds": [500, 1000], "solzen": [50, 60]},
).to_platform("sentinel2")
```

| Platform | Key | Method | Bands |
|----------|-----|--------|------:|
| Sentinel-2 | `sentinel2` | SRF convolution | 13 |
| Sentinel-3 | `sentinel3` | SRF convolution | 21 |
| Landsat 8 | `landsat8` | SRF convolution | 7 |
| MODIS | `modis` | Interval average | 7 + 3 broadband |
| CESM 2-band | `cesm2band` | Interval average | 2 |
| CESM RRTMG | `cesmrrtmg` | Interval average | 14 |
| MAR | `mar` | Interval average | 4 |
| HadCM3 | `hadcm3` | Interval average | 6 |

See [docs/BANDS.md](docs/BANDS.md) for band definitions, spectral indices, and data provenance.


## Emulator

A PCA + MLP neural network surrogate trained on forward model outputs. Predicts 480-band spectral albedo in ~microseconds (~50,000× speedup), making optimisation and MCMC practical.

A pre-built default emulator ships with the repo:

```python
from biosnicar.emulator import Emulator
from biosnicar import run_emulator

emu = Emulator.load("data/emulators/glacier_ice_8_param_default.npz")
outputs = run_emulator(emu, rds=1000, rho=600, solzen=50, direct=1,
                       black_carbon=0, dust=0, snow_algae=0, glacier_algae=50000)
print(outputs.BBA)
outputs.to_platform("sentinel2")  # chaining works identically to run_model()
```

Build a custom emulator for different parameter ranges or ice configurations:

```python
emu = Emulator.build(
    params={"rds": (100, 5000), "rho": (100, 917),
            "black_carbon": (0, 100000), "glacier_algae": (0, 500000)},
    n_samples=5000, layer_type=1, solzen=50, direct=1,
)
emu.save("my_emulator.npz")
```

See [docs/EMULATOR.md](docs/EMULATOR.md) and [examples/04–06](examples/).


## Inversion

Retrieve ice properties from observed albedo using emulator-powered optimisation. The recommended approach retrieves **specific surface area (SSA)** — the quantity the spectral inversion actually constrains — rather than bubble radius and density individually:

```python
from biosnicar.inverse import retrieve

result = retrieve(
    observed=measured_albedo,
    parameters=["ssa", "glacier_algae"],
    emulator=emu,
    fixed_params={"direct": 1, "solzen": 50, "black_carbon": 0,
                  "dust": 0, "snow_algae": 0},
)
print(result.summary())
result.plot(show=True)
```

Works with satellite observations too:

```python
import numpy as np

result = retrieve(
    observed=np.array([0.82, 0.78, 0.75, 0.45, 0.03]),
    parameters=["ssa", "glacier_algae"],
    emulator=emu,
    platform="sentinel2",
    observed_band_names=["B2", "B3", "B4", "B8", "B11"],
    obs_uncertainty=np.array([0.02, 0.02, 0.02, 0.03, 0.05]),
    fixed_params={"direct": 1, "solzen": 50, "black_carbon": 0,
                  "dust": 0, "snow_algae": 0},
)
```

Four optimisation methods: `L-BFGS-B` (fast default), `Nelder-Mead`, `differential_evolution` (global), `mcmc` (full posterior). See [docs/INVERSION.md](docs/INVERSION.md) and [examples/07–09](examples/).


## Subsurface light

The adding-doubling solver resolves spectral fluxes at every layer interface, enabling subsurface light diagnostics:

```python
outputs = run_model(solzen=50, rds=1000)

# PAR at depth
print(outputs.par(0.1))                    # at 10 cm
print(outputs.par([0.0, 0.1, 0.5, 1.0]))  # array of depths

# Spectral fluxes at arbitrary depth
fluxes = outputs.subsurface_flux(0.05)     # dict with F_up, F_dwn, F_net

# Spectral heating rate per layer (K/hr per band)
heat = outputs.spectral_heating_rate()

# Plot PAR depth profile
outputs.plot_subsurface(show=True)
```

See [docs/SUBSURFACE.md](docs/SUBSURFACE.md) and [examples/11_subsurface_light.py](examples/11_subsurface_light.py).


## Sea ice

The `biosnicar.sea_ice` module extends BioSNICAR to Arctic sea ice with a **forward model** (predict albedo from ice properties) and an **inversion** (retrieve ice type and properties from observed albedo). Brine inclusions use Cox & Weeks (1983) brine volume + Maxwell-Garnett effective medium; melt ponds, semi-transparent young ice, and open water are modelled too. It runs through the standard `run_model()` API — no special classes needed.

```python
from biosnicar import run_model

# Forward model via a preset (FYI_WINTER_BARE / _SNOW, MYI_WINTER_BARE, FYI_SUMMER_BARE, FYI_POND_*)
out = run_model(preset="FYI_WINTER_BARE", solzen=60)
print(out.BBA, out.albedo[:5])          # broadband + 480-band spectrum

# Or specify a column directly with flat kwargs (layer_type 4 = sea ice, 5 = pond, 6 = young ice)
out = run_model(
    layer_type=[0, 4, 4], dz=[0.10, 0.05, 1.45],
    rds=[200, None, None], rho=[300, 895, 895],
    sea_ice_salinity=[None, 12, 8], sea_ice_temperature=[None, -20, -8],
    sea_ice_bubble_radius=[None, 100, 200], solzen=60,
)
```

Use the adding-doubling solver (the default); the Toon solver does not handle the Fresnel air-ice interface. The deprecated `SeaIceColumn`/`SeaIceLayer` classes in `biosnicar.sea_ice.api` still work but are superseded by `run_model()`.

**Surface types & scope:** seven optical surface classes are supported end-to-end — bare/snow-covered/summer FYI, bare MYI, melt ponds, young ice (grease/nilas/grey, `layer_type=6`), and open water. *Deferred:* sea-ice algae, salty snow, vertical T/S profiles, Antarctic-specific tuning.

See [docs/sea_ice.md](docs/sea_ice.md) (physics primer), [docs/SEA_ICE_RETRIEVAL.md](docs/SEA_ICE_RETRIEVAL.md) (end-to-end retrieval guide), and [examples/13_sea_ice.py](examples/13_sea_ice.py).

### Satellite scene retrieval (batch)

`retrieve_sea_ice()` classifies an observed albedo spectrum (or satellite band set)
into one of seven surface types — FYI bare/snow/summer/pond, MYI bare, young ice,
open water — and retrieves physical parameters with quality flags.
`retrieve_sea_ice_batch()` scales this to whole scenes with spatial output
(requires `pip install biosnicar[geo]`):

```python
import numpy as np
from biosnicar.sea_ice import retrieve_sea_ice_batch

# image: (H, W, 4) Sentinel-2 band albedos (B2, B3, B4, B8)
scene = retrieve_sea_ice_batch(
    image,
    platform="sentinel2",
    observed_band_names=["B2", "B3", "B4", "B8"],
    solzen=62, direct=1, known_month=6,
    spatial_coords=latlon,             # (H, W, 2) lat/lon → enables H3 export
    crs="EPSG:32633", transform=affine,  # georeference → enables GeoTIFF export
)
print(scene.summary())                   # per-type pixel counts, flagged pixels
scene.to_geotiff("classified.tif")       # surface_type_code, confidence, cost, flags
scene.to_h3_geojson("classified.geojson", resolution=9)
ds = scene.to_xarray()                   # full per-pixel parameters + uncertainties
```

Field/drone spectra on instrument-native wavelength grids resample via
`biosnicar.sea_ice.spectral_utils.resample_to_model_grid()`.
See [docs/INVERSION.md](docs/INVERSION.md#sea-ice-inversion) and
[docs/SEA_ICE_EMULATOR.md](docs/SEA_ICE_EMULATOR.md).


## Built-in plotting

All result objects have `.plot()` methods that return `(fig, axes)` for further customisation:

```python
# Spectral albedo (with optional platform band overlay)
outputs.plot(show=True)
outputs.plot(platform="sentinel2", save="albedo.png")

# Compare multiple runs
clean = run_model(rds=1000)
dirty = run_model(rds=1000, black_carbon=100)
clean.plot(dirty, labels=["Clean", "100 ppb BC"])

# Subsurface PAR (normalised + absolute)
outputs.plot_subsurface(irradiance=800, show=True)

# Inversion diagnostics
result.plot(true_values={"ssa": 0.67}, save="retrieval.png")

# Sweep sensitivity (auto-selects line/heatmap/multi-line)
df.plot_sensitivity(y="BBA", show=True)
```

Requires `matplotlib` (optional dependency). See [docs/PLOTTING.md](docs/PLOTTING.md).


## Working with impurities

Impurity concentrations use descriptive names matching entries in `biosnicar/inputs.yaml`:

| Name | Unit | Description |
|------|------|-------------|
| `black_carbon` | ppb | Black carbon |
| `snow_algae` | cells/mL | Snow algae |
| `glacier_algae` | cells/mL | Glacier algae |

These names work consistently across `run_model()`, `parameter_sweep()`, emulator building, and inversions.

**Adding a new impurity type:** add an optical property `.npz` file to the LAP archive and a corresponding entry under `IMPURITIES` in `inputs.yaml` — the YAML key becomes the parameter name everywhere.


## Web app

A browser-based GUI is available at [bit.ly/bio-snicar](https://bit.ly/bio-snicar), or run locally:

```bash
./start_app.sh
```

<img src="./app.png" width=500>


## Configuration

The default configuration is in `biosnicar/inputs.yaml`. A guide to choosing physically realistic inputs is available at [biosnicar.vercel.app](https://biosnicar.vercel.app).


## Documentation

| Topic | Location |
|-------|----------|
| Full user guide | [biosnicar.vercel.app](https://biosnicar.vercel.app) |
| **Sea ice extension** | [docs/sea_ice.md](docs/sea_ice.md) |
| **Sea ice retrieval (end-to-end)** | [docs/SEA_ICE_RETRIEVAL.md](docs/SEA_ICE_RETRIEVAL.md) |
| Band convolution | [docs/BANDS.md](docs/BANDS.md) |
| Emulator | [docs/EMULATOR.md](docs/EMULATOR.md) |
| Inversion | [docs/INVERSION.md](docs/INVERSION.md) |
| Subsurface light | [docs/SUBSURFACE.md](docs/SUBSURFACE.md) |
| Plotting | [docs/PLOTTING.md](docs/PLOTTING.md) |
| RT methods | [docs/METHODS.md](docs/METHODS.md) |
| Worked examples | [examples/](examples/) |
| Interactive classroom | [biosnicar-classroom.netlify.app](https://biosnicar-classroom.netlify.app/) |
| Tutorial notebooks | [notebooks/](notebooks/) |


## Repository history note (March 2026)

The git history was rewritten in March 2026 to remove large binary files, reducing the download from ~1 GB to ~170 MB. If you cloned before this date, delete your local clone and re-clone. No source code or data was lost.


## Contributing

Issues and pull requests are welcome. PRs trigger CI tests via GitHub Actions; PRs that pass will be reviewed.


## License

MIT. Please cite the following if you use this code:

Cook, J. et al. (2020): Glacier algae accelerate melt rates on the western Greenland Ice Sheet, *The Cryosphere*, doi:10.5194/tc-14-309-2020

Flanner, M. et al. (2007): Present-day climate forcing and response from black carbon in snow, *J. Geophys. Res.*, 112, D11202, doi:10.1029/2006JD008003

If using the adding-doubling solver, please also cite Dang et al. (2019) and Whicker et al. (2022). The aspherical grain corrections come from He et al. (2016).


## References

<details>
<summary>Full reference list</summary>

Balkanski, Y., Schulz, M., Claquin, T., & Guibert, S. (2007). Reevaluation of Mineral aerosol radiative forcings suggests a better agreement with satellite and AERONET data. *Atmospheric Chemistry and Physics*, 7(1), 81-95.

Bidigare, R. R., Ondrusek, M. E., Morrow, J. H., & Kiefer, D. A. (1990). In-vivo absorption properties of algal pigments. In *Ocean Optics X* (Vol. 1302, pp. 290-302).

Bohren, C. F., & Huffman, D. R. (1983). *Absorption and scattering of light by small particles*. John Wiley & Sons.

Briegleb, B. P., and B. Light (2007). A Delta-Eddington multiple scattering parameterization for solar radiation in the sea ice component of the Community Climate System Model. *NCAR technical note*.

Chevrollier, L-A., et al. (2023). Light absorption and albedo reduction by pigmented microalgae on snow and ice. *Journal of Glaciology*, 69(274), 333-341.

Clementson, L. A., & Wojtasiewicz, B. (2019). Dataset on the absorption characteristics of extracted phytoplankton pigments. *Data in Brief*, 24, 103875.

Cook, J. M., et al. (2017). Quantifying bioalbedo: A new physically-based model and critique of empirical methods for characterizing biological influence on ice and snow albedo. *The Cryosphere*, doi:10.5194/tc-2017-73.

Cook, J. M., et al. (2020). Glacier algae accelerate melt rates on the western Greenland Ice Sheet. *The Cryosphere*, doi:10.5194/tc-14-309-2020.

Dang, C., Zender, C., Flanner, M. (2019). Intercomparison and improvement of two-stream shortwave radiative transfer schemes in Earth system models for a unified treatment of cryospheric surfaces. *The Cryosphere*, 13, 2325-2343.

Flanner, M., et al. (2007). Present-day climate forcing and response from black carbon in snow. *J. Geophys. Res.*, 112, D11202.

Flanner, M., et al. (2009). Springtime warming and reduced snow cover from carbonaceous particles. *Atmospheric Chemistry and Physics*, 9, 2481-2497.

Flanner, M. G., Gardner, A. S., Eckhardt, S., Stohl, A., & Perket, J. (2014). Aerosol radiative forcing from the 2010 Eyjafjallajökull volcanic eruptions. *J. Geophys. Res. Atmos.*, 119(15), 9481-9491.

Flanner, M. G., et al. (2021). SNICAR-ADv3: a community tool for modeling spectral snow albedo. *Geosci. Model Dev.*, 14, 7673-7704.

Halbach, L., et al. (2022). Pigment signatures of algal communities and their implications for glacier surface darkening. *Scientific Reports*, 12(1), 17643.

He, C., et al. (2017). Impact of snow grain shape and black carbon–snow internal mixing on snow optical properties: Parameterizations for climate models. *J. Climate*, 30(24), 10019-10036.

He, C., et al. (2018). Impact of grain shape and multiple black carbon internal mixing on snow albedo: Parameterization and radiative effect analysis. *J. Geophys. Res. Atmos.*, 123(2), 1253-1268.

Kirchstetter, T. W., Novakov, T., & Hobbs, P. V. (2004). Evidence that the spectral dependence of light absorption by aerosols is affected by organic carbon. *J. Geophys. Res. Atmos.*, 109(D21).

Lee, E., & Pilon, L. (2013). Absorption and scattering by long and randomly oriented linear chains of spheres. *JOSA A*, 30(9), 1892-1900.

Picard, G., Libois, Q., & Arnaud, L. (2016). Refinement of the ice absorption spectrum in the visible using radiance profile measurements in Antarctic snow. *The Cryosphere*, 10(6), 2655-2672.

Polashenski, C., et al. (2015). Neither dust nor black carbon causing apparent albedo decline in Greenland's dry snow zone. *Geophys. Res. Lett.*, 42, 9319-9327.

Skiles, S. M., Painter, T., & Okin, G. S. (2017). A method to retrieve the spectral complex refractive index and single scattering optical properties of dust deposited in mountain snow. *J. Glaciology*, 63(237), 133-147.

Toon, O. B., McKay, C. P., Ackerman, T. P., & Santhanam, K. (1989). Rapid calculation of radiative heating rates and photodissociation rates in inhomogeneous multiple scattering atmospheres. *J. Geophys. Res.*, 94(D13), 16287-16301.

van Diedenhoven, B., et al. (2014). A flexible parameterization for shortwave optical properties of ice crystals. *J. Atmos. Sci.*, 71, 1763-1782.

Warren, S. G. (1984). Optical constants of ice from the ultraviolet to the microwave. *Applied Optics*, 23(8), 1206-1225.

Warren, S. G., & Brandt, R. E. (2008). Optical constants of ice from the ultraviolet to the microwave: A revised compilation. *J. Geophys. Res. Atmos.*, 113(D14).

Whicker, C. A., et al. (2022). SNICAR-ADv4: a physically based radiative transfer model to represent the spectral albedo of glacier ice. *The Cryosphere*, 16(4), 1197-1220.

</details>
