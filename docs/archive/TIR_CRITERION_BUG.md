# Bug: False Total Internal Reflection Outside the Reststrahlen Band

**File:** `biosnicar/rt_solvers/adding_doubling_solver.py`  
**Function:** `calc_correction_fresnel_layer`  
**Affects:** solid-ice configurations (`layer_type=1`) at near-grazing solar angles

---

## The bug

At SZA=89°, 70 wavelength bands between 4.145–4.835 µm were incorrectly
overridden to Fresnel reflectance = 1.0.  These bands lie outside the
Reststrahlen anomalous-dispersion region of ice (n_re≈1.35, n_im≈0.016–0.031).
The physical Fresnel reflectance at near-grazing incidence is ~0.90.

## Root cause

The TIR detection uses:

```python
critical_angle = np.arcsin(ref_indx)   # ref_indx = n_re + i·k, complex
tir = beam_angle >= critical_angle[:nbr_wvl].real
```

For any `k ≠ 0`, `arcsin(n_re + ik).real` is slightly less than π/2. At
SZA=89°, `beam_angle ≈ π/2 − 0.017 rad`, so even the tiny k of normal ice
outside the Reststrahlen (~0.016) is enough to trigger the override.

## The fix

Gate TIR detection to the Reststrahlen anomalous-dispersion window (2.5–4.0 µm).
Outside this window, ice has a normal refractive index and TIR cannot occur.

```python
# Before:
tir = beam_angle >= critical_angle[:nbr_wvl].real

# After:
wvl = 0.205 + np.arange(nbr_wvl) * 0.01
in_reststrahlen = (wvl >= 2.5) & (wvl <= 4.0)
tir = in_reststrahlen & (beam_angle >= critical_angle[:nbr_wvl].real)
```

This preserves the full B&L/Whicker-intended TIR behaviour across the entire
Reststrahlen band (including the outer bands where n_re > 1 but n_im is large),
while definitively excluding the false trigger at 4.1–4.9 µm.

## Validation

```bash
uv run pytest tests/test_snicar.py -k "tir" -v
```
