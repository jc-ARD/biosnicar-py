#!/usr/bin/python
"""Calculates optical properties of ice/snow column from inputs.

These functions take the user-defined inputs for the physical
properties of the snow and ice and any impurities and calculate
the single scattering albedo, asymmetry parameter and optical
thickness which are then passed to one or other of our radiative
transfer solvers (Toon solver or adding-doubling solver).

"""


import os

import numpy as np
import pandas as pd
from scipy.interpolate import pchip, RegularGridInterpolator

import biosnicar.optical_properties.mie_coated_water_spheres as wcs
from biosnicar.optical_properties.op_lookup import get_hex_lut, get_lut
import biosnicar as _biosnicar

_SEA_ICE_LUT_PATH = str(
    _biosnicar.DATA_DIR / "OP_data" / "480band" / "luts" / "sea_ice.npz"
)

# Module-level cache for the sea-ice LUT interpolators
_sea_ice_lut_interp = None


def _load_sea_ice_lut():
    """Lazy-load and cache 4-D RegularGridInterpolators for the sea-ice LUT."""
    global _sea_ice_lut_interp
    if _sea_ice_lut_interp is not None:
        return _sea_ice_lut_interp

    data = np.load(_SEA_ICE_LUT_PATH)
    T_grid = data["T_grid"]          # shape (7,)  decreasing (−2 to −30)
    S_grid = data["S_grid"]          # shape (6,)  increasing
    rho_grid = data["rho_grid"]      # shape (4,)  increasing
    bbl_grid = data["bubble_radius_grid"]  # shape (4,) increasing
    tau = data["tau_per_m"]          # (7, 6, 4, 4, 480)
    ssa = data["ssa"]
    asm = data["asm"]

    # RegularGridInterpolator needs strictly increasing axes; flip T.
    T_asc = T_grid[::-1]
    tau_asc = tau[::-1]
    ssa_asc = ssa[::-1]
    asm_asc = asm[::-1]

    def _make_interp(arr):
        return RegularGridInterpolator(
            (T_asc, S_grid, rho_grid, bbl_grid),
            arr,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    _sea_ice_lut_interp = (
        _make_interp(tau_asc),
        _make_interp(ssa_asc),
        _make_interp(asm_asc),
    )
    return _sea_ice_lut_interp


def _interpolate_sea_ice_lut(salinity_psu, temperature_C, density_kg_m3, bubble_radius_um):
    """Interpolate sea-ice LUT to get (tau_per_m, ssa, g) at 480 bands.

    Returns arrays of shape (480,) for tau_per_m, ssa, g.
    """
    interp_tau, interp_ssa, interp_asm = _load_sea_ice_lut()

    # Load LUT to get grid bounds for clamping
    data = np.load(_SEA_ICE_LUT_PATH)
    T_min = data["T_grid"].min()
    T_max = data["T_grid"].max()
    S_min = data["S_grid"].min()
    S_max = data["S_grid"].max()
    rho_min = data["rho_grid"].min()
    rho_max = data["rho_grid"].max()
    bbl_min = data["bubble_radius_grid"].min()
    bbl_max = data["bubble_radius_grid"].max()

    T_c = float(np.clip(temperature_C, T_min, T_max))
    S_c = float(np.clip(salinity_psu, S_min, S_max))
    rho_c = float(np.clip(density_kg_m3, rho_min, rho_max))
    bbl_c = float(np.clip(bubble_radius_um, bbl_min, bbl_max))

    # The interpolators were built over the 4-D grid (T, S, rho, bbl) with the
    # 480-band axis as multi-dimensional output values, so a single (1, 4)
    # query point returns the full (480,) spectrum (RegularGridInterpolator
    # supports vector-valued outputs).
    pt = np.array([[T_c, S_c, rho_c, bbl_c]])
    tau_per_m = interp_tau(pt)[0]   # shape (480,)
    ssa_vals = interp_ssa(pt)[0]
    g_vals = interp_asm(pt)[0]

    tau_per_m = np.maximum(tau_per_m, 0.0)
    ssa_vals = np.clip(ssa_vals, 1e-8, 1.0 - 1e-8)
    g_vals = np.clip(g_vals, -0.9999, 0.9999)

    return tau_per_m, ssa_vals, g_vals


def _ri_name(ice):
    """Extract refractive index name (e.g. 'Pic16') from ice.op_dir."""
    return ice.op_dir.split("/")[0].replace("ice_", "")


def _sphere_lut_path(model_config, ice):
    """Resolve path to sphere ice LUT .npz file."""
    ri = _ri_name(ice)
    if "BH83" in model_config.sphere_ice_path:
        name = f"ice_sphere_BH83_{ri}.npz"
    else:
        name = f"ice_sphere_{ri}.npz"
    return os.path.join(model_config.lut_dir, name)


def _hex_lut_path(model_config, ice):
    """Resolve path to hex column LUT .npz file."""
    ri = _ri_name(ice)
    return os.path.join(model_config.lut_dir, f"hex_{ri}.npz")


def _bubbly_air_lut_path(model_config):
    """Resolve path to bubbly air LUT .npz file."""
    if "BH83" in model_config.bubbly_ice_path:
        name = "bubbly_air_BH83.npz"
    else:
        name = "bubbly_air.npz"
    return os.path.join(model_config.lut_dir, name)


def _bubbly_water_lut_path(model_config):
    """Resolve path to bubbly water LUT .npz file."""
    return os.path.join(model_config.lut_dir, "bubbly_water.npz")


def _water_sphere_lut_path(model_config):
    """Resolve path to water sphere LUT .npz file."""
    return os.path.join(model_config.lut_dir, "water_sphere.npz")


def get_layer_OPs(ice, model_config):
    """Calculates optical properties (tauy, ssa, g) of ice column.

    Takes configuration from ice and model_config and uses the data
    to calculate the optical properties of the ice column. There are
    separate routes for layers with granular ice and solid ice.
    Function calls are made to add liquid water coatings or adjust
    the optical properties for aspherical grains where toggled.

    Args:
        ice: instance of Ice class
        model_config: instance of ModelConfig class

    Returns:
        ssa_snw: single scatterign albedo of each layer
        g_snw: asymmetry parameter of each layer
        mac_snw: mass absorption coefficient of each layer

    """

    ssa_snw = np.empty([ice.nbr_lyr, model_config.nbr_wvl])
    mac_snw = np.empty([ice.nbr_lyr, model_config.nbr_wvl])
    g_snw = np.empty([ice.nbr_lyr, model_config.nbr_wvl])
    abs_cff_mss_ice = np.empty(model_config.nbr_wvl)

    # calculations of ice OPs in each layer
    for i in np.arange(0, ice.nbr_lyr, 1):

        if ice.layer_type[i] == 0:  # granular layer

            if ice.shp[i] == 4:  # large hex prisms (geometric optics)
                lut = get_hex_lut(_hex_lut_path(model_config, ice))
                ssa_snw[i, :] = lut.get(ice.hex_side[i], ice.hex_length[i], "ss_alb")
                mac_snw[i, :] = lut.get(ice.hex_side[i], ice.hex_length[i], "ext_cff_mss")
                g_snw[i, :] = lut.get(ice.hex_side[i], ice.hex_length[i], "asm_prm")

            elif ice.shp[i] < 4:
                sphere_lut = get_lut(_sphere_lut_path(model_config, ice))
                radius = ice.rds[i]

                # if liquid water coatings are applied
                if ice.water[i] > ice.rds[i]:
                    ext_cff_mss_ice = sphere_lut.get(radius, "ext_cff_mss")
                    ssa_snw[i, :], g_snw[i,:], mac_snw[i,:] = add_water_coating(
                        ice, model_config, ssa_snw[i,:], g_snw[i,:], mac_snw[i,:], i,
                        ext_cff_mss_ice
                    )

                else:
                    ssa_snw[i, :] = sphere_lut.get(radius, "ss_alb")
                    mac_snw[i, :] = sphere_lut.get(radius, "ext_cff_mss")
                    g_snw[i, :] = sphere_lut.get(radius, "asm_prm")

                    # Correct g for aspherical particles - He et al.(2017)
                    # Applies only when ice.shp!=0
                    # g_snw asymmetry factor parameterization coefficients
                    # (6 bands) from Table 3 & Eqs. 6-7 in He et al. (2017)
                    # assume same values for 4-5 um band, which leads
                    # to very small biases (<3%)

                    if (ice.shp[i] > 0) & (ice.shp[i] < 4):
                        g_snw = correct_for_asphericity(ice, g_snw, ssa_snw, i, model_config)

        # solid ice layer with air/water inclusions

        elif (ice.layer_type[i] == 1) or (ice.layer_type[i] == 2):

            if ice.cdom[i]:
                cdom = pd.read_csv(
                    model_config.dir_base + "data/OP_data/k_cdom_240_750.csv"
                )
                cdom_ref_idx_im = np.array(cdom).flatten()
                # rescale to SNICAR resolution
                cdom_ref_idx_im_rescaled = cdom_ref_idx_im[::10]
                ice.ref_idx_im[3:54] = np.fmax(
                    ice.ref_idx_im[3:54], cdom_ref_idx_im_rescaled
                )

            # neglecting air mass:
            vlm_frac_ice = (ice.rho[i] - ice.lwc[i] * 1000) / 917
            vlm_frac_air = 1 - ice.lwc[i] - vlm_frac_ice

            # air bbl ssps
            bubbly_air = get_lut(_bubbly_air_lut_path(model_config))
            radius = ice.rds[i]
            sca_cff_vlm_air_bbl = bubbly_air.get(radius, "sca_cff_vlm")
            g_air_bbl = bubbly_air.get(radius, "asm_prm")

            if ice.lwc[i] == 0:

                abs_cff_mss_ice[:] = ((4 * np.pi * ice.ref_idx_im) / (model_config.wavelengths * 1e-6)) / 917
                mac_snw[i, :] = (
                    (sca_cff_vlm_air_bbl * vlm_frac_air) / ice.rho[i]
                ) + abs_cff_mss_ice

                ssa_snw[i, :] = (
                    (sca_cff_vlm_air_bbl * vlm_frac_air) / ice.rho[i]
                ) / mac_snw[i, :]

                g_snw[i, :] = g_air_bbl

            elif ice.lwc[i] != 0:

                # water bubbles ssps
                bubbly_water = get_lut(_bubbly_water_lut_path(model_config))
                sca_cff_vlm_water = bubbly_water.get(radius, "sca_cff_vlm")
                ext_cff_vlm_water = bubbly_water.get(radius, "ext_cff_vlm")
                g_water = bubbly_water.get(radius, "asm_prm")

                vlm_frac_lw_in_ice = ice.lwc[i] * (1 - ice.lwc_pct_bbl)
                vlm_frac_lw_in_bbl = ice.lwc[i] * ice.lwc_pct_bbl

                # neglecting air absorption:
                abs_cff_mss_ice[:] = (vlm_frac_ice * 917 / ice.rho[i]) * (
                    4 * np.pi * ice.ref_idx_im / (model_config.wavelengths * 1e-6)
                ) / 917 + (vlm_frac_lw_in_ice * 1000 / ice.rho[i]) * (
                    4 * np.pi * ice.ref_idx_im_water / (model_config.wavelengths * 1e-6)
                ) / 1000

                # scattering-coefficient weighted asymmetry parameter
                sca_air = sca_cff_vlm_air_bbl * vlm_frac_air
                sca_water = sca_cff_vlm_water * vlm_frac_lw_in_bbl
                g_snw[i, :] = (g_air_bbl * sca_air + g_water * sca_water) / (
                    sca_air + sca_water
                )

                # volume weighted extinction coefficient
                mac_snw[i, :] = (
                    (sca_cff_vlm_air_bbl * vlm_frac_air) / ice.rho[i]
                    + (ext_cff_vlm_water * vlm_frac_lw_in_bbl) / ice.rho[i]
                ) + abs_cff_mss_ice

                # volume weighted scattering coefficient
                ssa_snw[i, :] = (
                    (sca_cff_vlm_air_bbl * vlm_frac_air) / ice.rho[i]
                    + (sca_cff_vlm_water * vlm_frac_lw_in_bbl) / ice.rho[i]
                ) / mac_snw[i, :]


        # granular layer with mixed ice and water spheres
        elif ice.layer_type[i] == 3:

            # neglecting air mass:
            vlm_frac_ice = (ice.rho[i] - ice.lwc[i] * 1000) / 917
            vlm_frac_air = 1 - ice.lwc[i] - vlm_frac_ice

            radius = ice.rds[i]
            sphere_lut = get_lut(_sphere_lut_path(model_config, ice))
            water_lut = get_lut(_water_sphere_lut_path(model_config))

            g = (
                water_lut.get(radius, "asm_prm") * ice.lwc[i]
                + sphere_lut.get(radius, "asm_prm") * vlm_frac_ice
            ) / (ice.lwc[i] + vlm_frac_ice)

            g_snw[i, :] = g

            ext_cff_mss = (
                water_lut.get(radius, "ext_cff_vlm") * ice.lwc[i]
                + sphere_lut.get(radius, "ext_cff_vlm") * vlm_frac_ice
            ) / (ice.rho[i])

            mac_snw[i, :] = ext_cff_mss

            ssa = (
                (
                    water_lut.get(radius, "sca_cff_vlm") * ice.lwc[i]
                    + sphere_lut.get(radius, "sca_cff_vlm") * vlm_frac_ice
                )
                / ice.rho[i]
                / mac_snw[i, :]
            )

            ssa_snw[i, :] = ssa

        # sea ice layer (brine inclusions, Maxwell-Garnett effective medium)
        elif ice.layer_type[i] == 4:
            S = getattr(ice, "sea_ice_salinity", [None])[i]
            T = getattr(ice, "sea_ice_temperature", [None])[i]
            bbl = getattr(ice, "sea_ice_bubble_radius", [None])[i]

            if S is None or T is None or bbl is None:
                raise ValueError(
                    f"Layer {i} has layer_type=4 (sea ice) but sea_ice_salinity, "
                    "sea_ice_temperature, or sea_ice_bubble_radius is None. "
                    "Set these fields on the Ice object or via the YAML config."
                )

            tau_per_m, ssa_si, g_si = _interpolate_sea_ice_lut(
                salinity_psu=S,
                temperature_C=T,
                density_kg_m3=ice.rho[i],
                bubble_radius_um=bbl,
            )
            # Convert per-metre optical depth to mass extinction coefficient:
            # mac = tau_per_m / rho  [m²/kg], so that mix_in_impurities gives
            # tau_layer = L_snw * mac = rho * dz * (tau_per_m / rho) = tau_per_m * dz
            mac_snw[i, :] = tau_per_m / ice.rho[i]
            ssa_snw[i, :] = ssa_si
            g_snw[i, :] = g_si

        # melt pond (liquid water layer — near-pure absorption, minimal scattering)
        elif ice.layer_type[i] == 5:
            # Absorption from liquid water imaginary RI (Rowe et al. 2020, 0°C).
            # Melt pond water is near 0°C so the 273K reference is appropriate.
            k_water = ice.ref_idx_im_water  # shape (480,), already loaded
            lam_m   = model_config.wavelengths * 1e-6   # µm → m
            abs_coeff = 4.0 * np.pi * k_water / lam_m  # m⁻¹

            # Rayleigh-like scattering from molecular fluctuations in liquid water.
            # β_w ≈ 0.003 × (0.55/λ_µm)⁴ m⁻¹ — very small but avoids ssa=0 singularities.
            lam_um  = model_config.wavelengths
            beta_w  = 0.003 * (0.55 / lam_um) ** 4   # m⁻¹

            ext_coeff = abs_coeff + beta_w
            rho_pond  = ice.rho[i]   # should be ~1000 kg/m³ for liquid water

            mac_snw[i, :] = ext_coeff / rho_pond
            ssa_snw[i, :] = np.clip(beta_w / ext_coeff, 1e-8, 1.0 - 1e-8)
            g_snw[i, :]   = 0.0    # isotropic molecular scattering

    return ssa_snw, g_snw, mac_snw


# Frazil/congelation crystal scattering coefficient for young ice (m⁻¹).
# Poorly constrained in the literature — Grenfell & Maykut (1977) report
# bulk extinction at a few wavelengths only.  Calibrated so the thin-slab
# BBA-vs-thickness curve matches their Table 3 brackets (dark nilas ≈
# 0.08–0.12, light nilas ≈ 0.10–0.18, grey ice ≈ 0.15–0.22) within ±0.03;
# grease ice (~1 cm) sits at ≈0.08.  Recalibrated to 3.0 after the
# internal-reflectance fix (the upward escape through the ice-air interface
# is (1-R_int)≈0.55, not (1-R_ext)≈0.94).
_YOUNG_ICE_SCAT = 3.0


def _compute_young_ice_ops(thickness_m, temperature_C, salinity_psu,
                           ocean_albedo, solzen=60, direct=1):
    """Spectral albedo of a thin young-ice slab over ocean (layer_type=6).

    Two-stream (Kubelka-Munk) slab solution: absorption from pure ice
    (Picard 2016 RI) and liquidus brine (Cox & Weeks brine volume), constant
    frazil scattering coefficient, ocean reflectance as the lower boundary,
    Fresnel reflection at the air-ice interface.  This is the Grenfell &
    Maykut (1977) thin-ice model; note that a pure Beer-Lambert slab with
    constant surface reflectance cannot reproduce their albedo-vs-thickness
    data (albedo must *grow* with thickness as internal backscattering
    accumulates), hence the two-stream form.

    Returns a 480-band albedo array — used as a boundary condition,
    bypassing the τ/ω/g pipeline.
    """
    from biosnicar.sea_ice.brine_optics import compute_brine_rfidx
    from biosnicar.sea_ice.brine_volume import compute_brine_volume
    from biosnicar.sea_ice.open_water import _fresnel_unpolarized
    from biosnicar.sea_ice.sea_ice_optics import _load_ice_ri

    d = float(thickness_m)
    if d <= 0:
        raise ValueError(f"ice_thickness must be > 0, got {d}")
    T = float(np.clip(temperature_C, -44.0, -2.0))
    S = float(salinity_psu)
    r_ocean = float(ocean_albedo)

    wvl_um = np.arange(0.205, 4.999, 0.01)
    lam_m = wvl_um * 1e-6

    m_ice = _load_ice_ri()
    vb = float(np.clip(compute_brine_volume(S, T), 0.0, 0.7))
    k_brine = compute_brine_rfidx(T).imag

    kappa_abs = 4.0 * np.pi * (
        (1.0 - vb) * m_ice.imag + vb * k_brine
    ) / lam_m                                   # m⁻¹

    # Kubelka-Munk slab over a reflecting boundary (diffuse two-flux):
    # K = 2×absorption, S = backscatter coefficient.
    K = 2.0 * kappa_abs
    Ssc = _YOUNG_ICE_SCAT
    a_km = 1.0 + K / Ssc
    b_km = np.sqrt(np.maximum(a_km**2 - 1.0, 1e-12))
    coth = 1.0 / np.tanh(np.minimum(b_km * Ssc * d, 50.0))
    r_slab = (1.0 - r_ocean * (a_km - b_km * coth)) / (a_km + b_km * coth - r_ocean)
    r_slab = np.clip(r_slab, 0.0, 1.0)

    # Air-ice Fresnel interface.  Downwelling entry uses the external
    # reflectance at the illumination geometry; the diffuse upwelling flux
    # from the slab sees the interface from BELOW, where total internal
    # reflection beyond the critical angle makes the internal diffuse
    # reflectance ~0.45 (radiance invariance: R_int = 1 - (1 - R_dif)/n²),
    # not the external beam value (~0.06).
    th = np.radians(np.arange(0.0, 90.0, 1.0))
    w = 2.0 * np.cos(th) * np.sin(th)
    r_dif = (w.reshape(-1, 1)
             * _fresnel_unpolarized(np.cos(th), m_ice)).sum(axis=0) / w.sum()
    r_int = 1.0 - (1.0 - r_dif) / m_ice.real**2
    if int(direct):
        cos_i = np.array([np.cos(np.radians(float(solzen)))])
        r_f = _fresnel_unpolarized(cos_i, m_ice)[0]
    else:
        r_f = r_dif

    albedo = r_f + (1.0 - r_f) * (1.0 - r_int) * r_slab / (1.0 - r_int * r_slab)
    return np.clip(albedo, 0.0, 1.0)


def add_water_coating(ice, model_config, ssa_snw, g_snw, mac_snw, i, ext_cff_mss_ice):

    """Recalculates layer optical properties where grains are coated in liquid water.

    Feature originally added by Niklas Bohn. Where value of water exceeds value of rds
    for a given layer it is interpreted as having a liquid water film. in this case
    Mie calculations for a coated sphere are executed with the outer coating havign radius
    water - rds.

    Args:
        ice: instance of Ice class
        model_config: instance of ModelConfig class
        ssa_snw: single scattering albedo of each layer
        g_snw: asymmetry parameter for each layer
        mac_snw: mass absorption coefficient of each layer
        i: layer counter
        ext_cff_mss_ice: pre-loaded extinction coefficient from sphere LUT

    Returns:
        ssa_snw: updated single scattering albedo for each layer
        g_snw: updated asymmetry parameter for each layer
        mac_snw: updated mass absorption coefficient for each layer

    Raises:
        ValueError if ice.shp!= 0 (i.e. grains not spherical)

    """

    if ice.shp[i] != 0:
        raise ValueError("Water coating can only be applied to spheres")

    res = wcs.miecoated_driver(
        rice=ice.rds[i],
        rwater=ice.water[i],
        fn_ice=model_config.dir_base+model_config.fn_ice,
        rf_ice=ice.rf,
        fn_water=model_config.dir_base+model_config.fn_water,
        wvl=model_config.wavelengths,
    )


    ssa_snw = res["ssa"]
    g_snw = res["asymmetry"]
    mac_snw = ext_cff_mss_ice

    return ssa_snw, g_snw, mac_snw


def correct_for_asphericity(ice, g_snw, ssa_snw, i, model_config):
    """Adjusts asymmetry parameter for aspherical grains.

    Implements work from Fu et al. 2007 and He et al. 2017.
    Asymmetry parameter is adjusted to account for asphericity
    for the defined shape of each layer.

    Ice grain shape can be
    0 = sphere,
    1 = spheroid,
    2 = hexagonal plate,
    3 = koch snowflake,
    4 = hexagonal prisms

    Args:
        ice: instance of Ice class
        g_snw: asymmetry parameter for each layer
        i: layer counter

    Returns:
        g_snw: updated asymmetry parameter for layer
    """

    g_wvl = np.array([0.25, 0.70, 1.41, 1.90, 2.50, 3.50, 4.00, 5.00])

    g_wvl_center = np.array(g_wvl[1:8]) / 2 + np.array(g_wvl[0:7]) / 2
    g_b0 = np.array(
        [
            9.76029e-01,
            9.67798e-01,
            1.00111e00,
            1.00224e00,
            9.64295e-01,
            9.97475e-01,
            9.97475e-01,
        ]
    )

    g_b1 = np.array(
        [
            5.21042e-01,
            4.96181e-01,
            1.83711e-01,
            1.37082e-01,
            5.50598e-02,
            8.48743e-02,
            8.48743e-02,
        ]
    )

    g_b2 = np.array(
        [
            -2.66792e-04,
            1.14088e-03,
            2.37011e-04,
            -2.35905e-04,
            8.40449e-04,
            -4.71484e-04,
            -4.71484e-04,
        ]
    )

    # Tables 1 & 2 and Eqs. 3.1-3.4 from Fu, 2007
    g_f07_c2 = np.array(
        [
            1.349959e-1,
            1.115697e-1,
            9.853958e-2,
            5.557793e-2,
            -1.233493e-1,
            0.0,
            0.0,
        ]
    )
    g_f07_c1 = np.array(
        [
            -3.987320e-1,
            -3.723287e-1,
            -3.924784e-1,
            -3.259404e-1,
            4.429054e-2,
            -1.726586e-1,
            -1.726586e-1,
        ]
    )
    g_f07_c0 = np.array(
        [
            7.938904e-1,
            8.030084e-1,
            8.513932e-1,
            8.692241e-1,
            7.085850e-1,
            6.412701e-1,
            6.412701e-1,
        ]
    )
    g_f07_p2 = np.array(
        [
            3.165543e-3,
            2.014810e-3,
            1.780838e-3,
            6.987734e-4,
            -1.882932e-2,
            -2.277872e-2,
            -2.277872e-2,
        ]
    )
    g_f07_p1 = np.array(
        [
            1.140557e-1,
            1.143152e-1,
            1.143814e-1,
            1.071238e-1,
            1.353873e-1,
            1.914431e-1,
            1.914431e-1,
        ]
    )
    g_f07_p0 = np.array(
        [
            5.292852e-1,
            5.425909e-1,
            5.601598e-1,
            6.023407e-1,
            6.473899e-1,
            4.634944e-1,
            4.634944e-1,
        ]
    )

    fs_hex = 0.788  # shape factor for hex plate

    # eff grain diameter
    diam_ice = 2.0 * ice.rds[i] / 0.544

    if ice.shp_fctr[i] == 0:
        # default shape factor for koch snowflake;
        # He et al. (2017), Table 1
        fs_koch = 0.712

    else:

        fs_koch = ice.shp_fctr[i]

    if ice.grain_ar[i] == 0:
        # default aspect ratio for koch
        # snowflake; He et al. (2017), Table 1
        ar_tmp = 2.5

    else:

        ar_tmp = ice.grain_ar[i]

    # Eq.7, He et al. (2017)
    g_snw_cg_tmp = g_b0 * (fs_koch / fs_hex) ** g_b1 * diam_ice**g_b2

    # Eqn. 3.3 in Fu (2007)
    gg_snw_f07_tmp = (
        g_f07_p0 + g_f07_p1 * np.log(ar_tmp) + g_f07_p2 * (np.log(ar_tmp)) ** 2
    )

    # 1 = spheroid, He et al. (2017)
    if ice.shp[i] == 1:

        # effective snow grain diameter
        diam_ice = 2.0 * ice.rds[i]

        # default shape factor for spheroid;
        # He et al. (2017), Table 1
        if ice.shp_fctr[i] == 0:

            fs_sphd = 0.929

        else:
            # if shp_factor not 0,
            # then use user-defined value
            fs_sphd = ice.shp_fctr[i]

        if ice.grain_ar[i] == 0:
            # default aspect ratio for spheroid;
            # He et al. (2017), Table 1
            ar_tmp = 0.5

        else:

            ar_tmp = ice.grain_ar[i]

        # Eq.7, He et al. (2017)
        g_snw_cg_tmp = g_b0 * (fs_sphd / fs_hex) ** g_b1 * diam_ice**g_b2

        # Eqn. 3.1 in Fu (2007)
        gg_snw_F07_tmp = g_f07_c0 + g_f07_c1 * ar_tmp + g_f07_c2 * ar_tmp**2

    # 3=hexagonal plate,
    # He et al. 2017 parameterization
    if ice.shp[i] == 2:

        # effective snow grain diameter
        diam_ice = 2.0 * ice.rds[i]

        if ice.shp_fctr[i] == 0:
            # default shape factor for
            # hexagonal plates;
            # He et al. (2017), Table 1
            fs_hex0 = 0.788

        else:

            fs_hex0 = ice.shp_fctr[i]

        if ice.grain_ar[i] == 0:
            # default aspect ratio
            # for hexagonal plate;
            # He et al. (2017), Table 1
            ar_tmp = 2.5

        else:

            ar_tmp = ice.grain_ar[i]

        # Eq.7, He et al. (2017)
        g_snw_cg_tmp = g_b0 * (fs_hex0 / fs_hex) ** g_b1 * diam_ice**g_b2

        # Eqn. 3.3 in Fu (2007)
        gg_snw_F07_tmp = (
            g_f07_p0 + g_f07_p1 * np.log(ar_tmp) + g_f07_p2 * (np.log(ar_tmp)) ** 2
        )

    # 4=koch snowflake,
    # He et al. (2017)
    #  parameterization
    if ice.shp[i] == 3:

        # effective snow grain diameter
        diam_ice = 2.0 * ice.rds[i] / 0.544

        if ice.shp_fctr[i] == 0:
            # default shape factor
            # for koch snowflake;
            # He et al. (2017), Table 1
            fs_koch = 0.712

        else:

            fs_koch = ice.shp_fctr[i]

        # default aspect ratio for
        # koch snowflake; He et al. (2017), Table 1
        if ice.grain_ar[i] == 0:

            ar_tmp = 2.5

        else:

            ar_tmp = ice.grain_ar[i]

        # Eq.7, He et al. (2017)
        g_snw_cg_tmp = g_b0 * (fs_koch / fs_hex) ** g_b1 * diam_ice**g_b2

        # Eqn. 3.3 in Fu (2007)
        gg_snw_F07_tmp = (
            g_f07_p0 + g_f07_p1 * np.log(ar_tmp) + g_f07_p2 * (np.log(ar_tmp)) ** 2
        )

    # 6 wavelength bands for g_snw to be
    # interpolated into 480-bands of SNICAR
    # shape-preserving piecewise interpolation
    # into 480-bands
    g_Cg_intp = pchip(g_wvl_center, g_snw_cg_tmp)(model_config.wavelengths)
    gg_f07_intp = pchip(g_wvl_center, gg_snw_F07_tmp)(model_config.wavelengths)
    g_snw_F07 = (
        gg_f07_intp + (1.0 - gg_f07_intp) / ssa_snw[i, :] / 2
    )  # Eq.2.2 in Fu (2007)
    # Eq.6, He et al. (2017)
    g_snw[i, :] = g_snw_F07 * g_Cg_intp
    g_snw[i, 381:480] = g_snw[i, 380]
    # assume same values for 4-5 um band,
    # with v small biases (<3%)

    g_snw[g_snw <= 0] = 0.00001
    g_snw[g_snw > 0.99] = 0.99  # avoid unreasonable
    # values (so far only occur in large-size spheroid cases)

    return g_snw


def mix_in_impurities(ssa_snw, g_snw, mac_snw, ice, impurities, model_config):
    """Updates optical properties for the presence of light absorbing particles.

    Takes the optical properties of the clean ice column and adjusts them for
    the presence of light absorbing particles in the ice. Each impurity is an
    instance of the Impurity class whose attributes include the path to the
    specific optical properties for that impurity. Its concentration is generally
    provided in ppb, but concentration of algae can also be given in cells/mL.


    Args:
        ssa_snw: single scattering albedo of eahc layer
        g_snw: asymmetry parameter for each layer
        mac_snw: mass absorption coefficient of each layer
        ice: instance of Ice class
        impurities: array containing instances of Impurity class
        model_config: instance of ModelConfig class

    Returns:
        tau: updated optical thickness
        ssa: updated single scattering albedo
        g: updated asymmetry parameter
        L_snw: mass of ice ine ach layer

    """

    ssa_aer = np.zeros([len(impurities), model_config.nbr_wvl])
    mac_aer = np.zeros([len(impurities), model_config.nbr_wvl])
    g_aer = np.zeros([len(impurities), model_config.nbr_wvl])
    mss_aer = np.zeros([ice.nbr_lyr, len(impurities)])
    g_sum = np.zeros([ice.nbr_lyr, model_config.nbr_wvl])
    ssa_sum = np.zeros([ice.nbr_lyr, len(impurities), model_config.nbr_wvl])
    tau = np.zeros([ice.nbr_lyr, model_config.nbr_wvl])
    ssa = np.zeros([ice.nbr_lyr, model_config.nbr_wvl])
    g = np.zeros([ice.nbr_lyr, model_config.nbr_wvl])
    L_aer = np.zeros([ice.nbr_lyr, len(impurities)])
    tau_aer = np.zeros([ice.nbr_lyr, len(impurities), model_config.nbr_wvl])
    tau_sum = np.zeros([ice.nbr_lyr, model_config.nbr_wvl])
    ssa_sum = np.zeros([ice.nbr_lyr, model_config.nbr_wvl])
    L_snw = np.zeros(ice.nbr_lyr)
    tau_snw = np.zeros([ice.nbr_lyr, model_config.nbr_wvl])

    for i, impurity in enumerate(impurities):

        g_aer[i, :] = impurity.g
        ssa_aer[i, :] = impurity.ssa

        if impurity.unit == 1:

            mss_aer[0 : ice.nbr_lyr, i] = (
                np.array(impurity.conc) / 917 * 10**6
            ) 

        else:
            mss_aer[0 : ice.nbr_lyr, i] = (
                np.array(impurity.conc) * 1e-9
            ) 

    # for each layer, the layer mass (L) is density * layer thickness
    # for each layer the optical ice.depth is
    # the layer mass * the mass extinction coefficient
    # first for the ice in each layer

    for i in range(ice.nbr_lyr):

        L_snw[i] = ice.rho[i] * ice.dz[i]

        for j, impurity in enumerate(impurities):

            mac_aer[j, :] = impurity.mac

            # kg ice m-2 * cells kg-1 ice = cells m-2
            L_aer[i, j] = L_snw[i] * mss_aer[i, j]
            # cells m-2 * m2 cells-1

            tau_aer[i, j, :] = L_aer[i, j] * mac_aer[j, :]
            tau_sum[i, :] = tau_sum[i, :] + tau_aer[i, j, :]
            ssa_sum[i, :] = ssa_sum[i, :] + (tau_aer[i, j, :] * ssa_aer[j, :])
            g_sum[i, :] = g_sum[i, :] + (tau_aer[i, j, :] * ssa_aer[j, :] * g_aer[j, :])

            # ice mass = snow mass - impurity mass (generally tiny correction)
            # if aer == algae and L_aer is in cells m-2, should be converted
            # to m-2 kg-1 : 1 cell = 1ng = 10**(-12) kg

            if impurity.unit == 1:

                L_snw[i] = L_snw[i] - L_aer[i, j] * 10 ** (-12)

            else:
                L_snw[i] = L_snw[i] - L_aer[i, j]

        tau_snw[i, :] = L_snw[i] * mac_snw[i, :]

        # finally, for each layer calculate the effective ssa, tau and g
        # for the snow+LAP
        tau[i, :] = tau_sum[i, :] + tau_snw[i, :]
        nonzero = tau[i, :] > 0
        ssa[i, nonzero] = (1 / tau[i, nonzero]) * (
            ssa_sum[i, nonzero] + (ssa_snw[i, nonzero] * tau_snw[i, nonzero])
        )
        scat = tau[i, :] * ssa[i, :]
        nonzero_scat = scat > 0
        g[i, nonzero_scat] = (1 / scat[nonzero_scat]) * (
            g_sum[i, nonzero_scat]
            + (g_snw[i, nonzero_scat] * ssa_snw[i, nonzero_scat] * tau_snw[i, nonzero_scat])
        )

    # just in case any unrealistic values arise (none detected so far)
    ssa[ssa <= 0] = 0.00000001
    ssa[ssa >= 1] = 0.99999999
    g[g <= 0] = 0.00001
    g[g > 0.99] = 0.99

    return tau, ssa, g, L_snw


if __name__ == "__main__":
    pass
