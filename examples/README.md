# BioSNICAR Examples

Self-contained example scripts covering the full BioSNICAR workflow. Each script prints results to the console and optionally produces figures with `--plot`.

| #   | Script                                              | Description                                                                             |
| --- | --------------------------------------------------- | --------------------------------------------------------------------------------------- |
| 01  | [basic_forward_model.py](01_basic_forward_model.py) | `run_model()` with defaults, overrides, multi-layer config, output access               |
| 02  | [parameter_sweeps.py](02_parameter_sweeps.py)       | `parameter_sweep()` for 2-param, 3-param, impurity, and spectral sweeps                 |
| 03  | [platform_bands.py](03_platform_bands.py)           | `.to_platform()` chaining on runs and sweeps, GCM bands, multi-platform                 |
| 04  | [emulator_build.py](04_emulator_build.py)           | Building custom emulators, inspecting properties, accuracy validation                   |
| 05  | [emulator_predict.py](05_emulator_predict.py)       | Load default emulator, `predict()`, `predict_batch()`, `run_emulator()`, speed test     |
| 06  | [emulator_save_load.py](06_emulator_save_load.py)   | Save/load roundtrip, metadata inspection, `.npz` portability                            |
| 07  | [inversion_spectral.py](07_inversion_spectral.py)   | SSA spectral retrieval with `retrieve()`, `fixed_params`, `wavelength_mask`, regularization |
| 08  | [inversion_satellite.py](08_inversion_satellite.py) | SSA band-mode retrieval with Sentinel-2, Landsat 8, MODIS, `obs_uncertainty`                |
| 09  | [inversion_methods.py](09_inversion_methods.py)     | L-BFGS-B, Nelder-Mead, differential evolution, MCMC comparison (SSA mode)                   |
| 10  | [end_to_end_workflow.py](10_end_to_end_workflow.py) | Full pipeline: load emulator, synthetic S2 observation, SSA retrieval, validate              |
| 11  | [subsurface_light.py](11_subsurface_light.py)       | Subsurface fluxes, PAR depth profiles, spectral heating rates                                |
| 12  | [planetscope.py](12_planetscope.py)                  | PlanetScope band convolution                                                                 |
| 13  | [sea_ice.py](13_sea_ice.py)                          | Sea ice forward model: presets, custom columns, snow cover, melt ponds                       |
| 14  | [sea_ice_emulator.py](14_sea_ice_emulator.py)        | Sea ice emulator training and prediction                                                     |
| 15  | [sea_ice_inversion.py](15_sea_ice_inversion.py)      | `retrieve_sea_ice()`: classification, band mode, uncertainty, multi-scenario                 |
| 16  | [sea_ice_scene_classification.py](16_sea_ice_scene_classification.py) | Young ice, open water, quality flags, batch scenes with NetCDF/H3 export   |

## Running

From the repository root:

```
python examples/01_basic_forward_model.py
python examples/05_emulator_predict.py --plot
python examples/09_inversion_methods.py --mcmc --plot
```

Examples 05-10 require the pre-built default emulator at `data/emulators/glacier_ice_8_param_default.npz`. To rebuild it:

```
python scripts/build_default_emulator.py
```

## Dependencies

- Examples 01-03 need only the base BioSNICAR dependencies
- Examples 04 and 06 require `scikit-learn` (for `Emulator.build()`)
- Example 09 with `--mcmc` requires `emcee`
- `--plot` requires `matplotlib`
