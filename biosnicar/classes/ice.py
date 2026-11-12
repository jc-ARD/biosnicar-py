import os
import numpy as np
import pandas as pd
import biosnicar
from biosnicar.utils.load_inputs import load_inputs

_BASE = str(biosnicar.PROJECT_ROOT)


class Ice:
    """Snow or ice column physical properties.

    Instances of Ice contain all the physical properties of each vertical layer of the
    snow or ice column and the underlying surface.

    Attributes:
        dz: array containing thickness of each layer in m
        layer_type: array containing type (0 = grains, 1 = solid ice) in each layer
        cdom: array containing Boolean (1/0) toggling presence of cdom in each layer
        rho: array containing density of each layer in kg/m3
        sfc: array with reflectance of underlying surface per wavelength
        rf: refractive index to use, 0, 1 or 2 (see docs for definition)
        shp: grain shape per layer where layer_type==1
        rds: grain radius (layer_type==0) or bubble radius (layer_type==0) in each layer
        water: radius of grain+water coating in each layer where layer_type==0
        hex_side: length of each side of hexagonal face for grain_shp==4
        hex_length: column length for hexagonal face for grain_shp==4
        shp_fctr: ratio of nonspherical eff radii to equal vol sphere, in each layer
        ar: aspect ratio of grains in each layer where layer_type==0
        nbr_lyr: number of vertical layers
    """

    def __init__(self, input_file):
        # use config to calculate refractive indices
        inputs = load_inputs(input_file)

        self.dz = inputs["ICE"]["DZ"]
        self.layer_type = inputs["ICE"]["LAYER_TYPE"]
        self.cdom = inputs["ICE"]["CDOM"]
        self.rho = inputs["ICE"]["RHO"]
        _sfc_path = _BASE + "/" + inputs["PATHS"]["SFC"]
        with open(_sfc_path, "rt") as _fh:
            self.sfc = np.genfromtxt(_fh, delimiter="csv")
        self.rf = inputs["ICE"]["RF"]
        self.shp = inputs["ICE"]["SHP"]
        self.rds = inputs["ICE"]["RDS"]
        self.water = inputs["ICE"]["WATER_COATING"]
        self.hex_side = inputs["ICE"]["HEX_SIDE"]
        self.hex_length = inputs["ICE"]["HEX_LENGTH"]
        self.shp_fctr = inputs["ICE"]["SHP_FCTR"]
        self.grain_ar = inputs["ICE"]["AR"]
        self.nbr_lyr = len(self.dz)
        self.lwc = inputs["ICE"]["LWC"]
        self.lwc_pct_bbl = inputs["ICE"]["LWC_PCT_BBL"]
        _water_ri = pd.read_csv(
            os.path.join(_BASE, inputs["PATHS"]["RI_ICE"],
                         "refractive_index_water_273K_Rowe2020.csv")
        )
        self.ref_idx_im_water = _water_ri.k.values
        # Real part, needed for the air-water Fresnel interface at the top of
        # a melt pond (layer_type=5).
        self.ref_idx_re_water = _water_ri.n.values

        # Sea-ice per-layer parameters (layer_type=2).
        # Defaults to None for each layer when not specified in config.
        n = len(self.dz)
        ice_sec = inputs.get("ICE", {})
        self.sea_ice_salinity = ice_sec.get("SEA_ICE_SALINITY", [None] * n)
        self.sea_ice_temperature = ice_sec.get("SEA_ICE_TEMPERATURE", [None] * n)
        self.sea_ice_bubble_radius = ice_sec.get("SEA_ICE_BUBBLE_RADIUS", [None] * n)

        self.calculate_refractive_index(input_file)

    def calculate_refractive_index(self, input_file):
        """Calculates ice refractive index from initialized class attributes.

        Takes self.rf and config from inpouts.yaml and uses them to calculate
        new attributes related to the ice refractive index.

        Args:
            self

        Returns:
            ref_idx_im: imaginary part of refractive index
            ref_idx_re: real part of refractive index
            fl_r_dif_a: precomputed diffuse reflectance "perpendicular polarized)
            fl_r_dif_b: precomputed diffuse reflectance "parallel polarized)
            op_dir: directory containing optical properties

        Raises:
            ValueError if rf out of range
        """
        if self.rf < 0 or self.rf > 2:
            raise ValueError("Ice ref index type out of range - between 0 and 2 only")

        inputs = load_inputs(input_file)

        ri_ice_dir = os.path.join(_BASE, inputs["PATHS"]["RI_ICE"])

        refidx_file = np.load(os.path.join(ri_ice_dir, "rfidx_ice.npz"))
        fresnel_diffuse_file = np.load(
            os.path.join(ri_ice_dir, "fl_reflection_diffuse.npz")
        )

        rf = inputs["ICE"]["RF"]
        op_dir_stub = inputs["PATHS"]["OP_DIR_STUBS"][rf]
        ref_idx_name = op_dir_stub[4:9]

        self.ref_idx_re = refidx_file[str("re_" + ref_idx_name)]
        self.ref_idx_im = refidx_file[str("im_" + ref_idx_name)].copy()
        self.fl_r_dif_a = fresnel_diffuse_file[str("R_dif_fa_ice_" + ref_idx_name)]
        self.fl_r_dif_b = fresnel_diffuse_file[str("R_dif_fb_ice_" + ref_idx_name)]
        self.op_dir = op_dir_stub
        self.fl_r_dif_a_water, self.fl_r_dif_b_water = diffuse_fresnel(
            self.ref_idx_re_water
        )


def diffuse_fresnel(n, n_gauss=256):
    """Diffuse Fresnel reflectances of a smooth interface between air and a
    medium of real refractive index *n*.

    Returns ``(R_dif_a, R_dif_b)``: the reflectance for isotropic diffuse
    flux arriving from above (air side) and from below (medium side).

    ``R_dif_a`` is the flux-weighted hemispheric integral of the unpolarised
    Fresnel reflectance, ``2 * integral R(mu) mu dmu`` (Briegleb & Light 2007,
    Eq. 25), by Gauss-Legendre quadrature.

    ``R_dif_b`` follows from it exactly rather than by a second quadrature:
    radiance conservation across a refracting interface (the n^2 law) gives

        1 - R_dif_b = (1 - R_dif_a) / n^2

    which is what makes the two directions consistent with each other —
    total internal reflection beyond the critical angle is contained in it.
    For n = 1.31 this gives 0.063 / 0.455, the constants CICE uses.

    Used for the air-water interface at the top of a melt pond.  The ice
    interface still reads its precomputed tables (``fl_reflection_diffuse.npz``,
    inherited from SNICAR-ADv4): their from-above values agree with this
    function to 2e-7, but their from-below values sit ~0.02 *under* the n^2
    relation.  That is a property of the inherited ice tables and is left
    alone here because the glacier-ice benchmarks are pinned to them.
    """
    n = np.asarray(n, dtype=float)
    out_a = np.full(n.shape, np.nan)
    ok = n > 1.0
    x, w = np.polynomial.legendre.leggauss(n_gauss)
    mu = (0.5 * (x + 1.0))[:, None]
    w = (0.5 * w)[:, None]
    nn = n[ok][None, :]
    mut = np.sqrt(1.0 - (1.0 - mu ** 2) / nn ** 2)
    r_s = ((mu - nn * mut) / (mu + nn * mut)) ** 2
    r_p = ((nn * mu - mut) / (nn * mu + mut)) ** 2
    out_a[ok] = (0.5 * (r_s + r_p) * mu * w).sum(axis=0) / (mu * w).sum()
    out_b = 1.0 - (1.0 - out_a) / n ** 2
    return out_a, out_b
