"""Neural network emulator for the BioSNICAR forward model.

Build phase uses scikit-learn for MLP training and PCA fitting.
Predict phase uses pure numpy — no sklearn dependency at inference.

The builder is general-purpose: any ``run_model()`` keyword can be a
free parameter or a fixed override.  The default configuration targets
solid glacier ice, but custom emulators can be built for snow, different
impurity sets, or any other model configuration.
"""

import json
import warnings
from collections import OrderedDict
from datetime import datetime

import numpy as np


def _latin_hypercube(n_samples, n_dims, seed=None):
    """Generate Latin hypercube samples in [0, 1]^d.

    Each dimension is stratified into *n_samples* equal intervals with
    exactly one sample per stratum, then columns are independently
    shuffled.  This gives uniform marginals and good space-filling
    without requiring ``scipy.stats.qmc`` (added in scipy 1.7).

    Parameters
    ----------
    n_samples : int
        Number of sample points.
    n_dims : int
        Number of dimensions.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_samples, n_dims)
        Samples in [0, 1]^d.
    """
    rng = np.random.RandomState(seed)
    samples = np.empty((n_samples, n_dims))
    for j in range(n_dims):
        # One sample per stratum, uniformly within the stratum
        strata = np.arange(n_samples)
        rng.shuffle(strata)
        samples[:, j] = (strata + rng.uniform(size=n_samples)) / n_samples
    return samples


# Binary parameters are snapped to {0, 1} during LHS.
_BINARY_PARAMS = {"direct"}

# Parameters that require integer rounding before passing to run_model.
# The optical-property lookup tables use integer keys (e.g. rds on a
# 5/10/500 grid), so continuous LHS values must be snapped.
_INTEGER_PARAMS = {"rds", "solzen"}

# Valid solar zenith angles — irradiance data (fsds.npz) only contains
# integer SZA keys from 0 to 89.  Validation enforces [1, 89].
_SOLZEN_RANGE = (1, 89)

# Impurity parameters sampled in log10(x+1) space during LHS to ensure
# adequate coverage of low concentrations (including near-zero / clean ice).
# Without this, uniform sampling in e.g. [0, 500000] puts < 0.2% of
# training points below 1000, and the emulator fails for clean spectra.
_LOG_SAMPLE_PARAMS = {
    "black_carbon", "snow_algae", "glacier_algae", "dust",
    # Sea ice: bubble radius spans ~1.3 orders of magnitude; log-space sampling
    # gives better coverage of the small-bubble (high-scattering) regime.
    "sea_ice_bubble_radius",
}


def _snap_rds(value):
    """Snap a continuous rds value to the nearest lookup-table radius.

    The bubbly-ice LUT grid is: step 5 for [10, 100), step 10 for
    [100, 5000], step 500 for (5000, 25000].
    """
    v = float(value)
    if v < 100:
        return int(round(v / 5) * 5)
    elif v <= 5000:
        return int(round(v / 10) * 10)
    else:
        return int(round(v / 500) * 500)


class VerificationResult:
    """Results from :meth:`Emulator.verify`.

    Attributes
    ----------
    n_points : int
        Number of benchmark parameter sets tested.
    mae_per_point : np.ndarray of shape (N,)
        Mean absolute error (over 480 bands) for each test point.
    max_err_per_point : np.ndarray of shape (N,)
        Maximum absolute error (over 480 bands) for each test point.
    bba_err_per_point : np.ndarray of shape (N,)
        Absolute broadband-albedo error for each test point.
    mae : float
        Overall mean absolute spectral error across all points and bands.
    max_err : float
        Worst-case absolute error across all points and bands.
    mae_bba : float
        Mean absolute broadband-albedo error.
    max_bba_err : float
        Maximum absolute broadband-albedo error.
    r2 : float
        Coefficient of determination (R²) over all predicted vs reference
        albedo values.
    benchmark_params : list of dict
        The parameter sets that were tested.
    emulator_albedos : np.ndarray of shape (N, 480)
        Emulator predictions for each benchmark point.
    reference_albedos : np.ndarray of shape (N, 480)
        Forward-model reference for each benchmark point (clamped to [0, 1]).
    unphysical_indices : list of int
        Indices of benchmark points where the forward model produced
        unphysical albedo values (outside [0, 1]).  These are excluded
        from aggregate error statistics.
    physical_mask : np.ndarray of shape (N,)
        Boolean mask — True for points with physical forward-model output.
    n_physical : int
        Number of benchmark points with physical forward-model output
        (used for aggregate statistics).
    """

    def __init__(
        self,
        benchmark_params,
        emulator_albedos,
        reference_albedos,
        flx_slr,
        unphysical_indices=None,
        physical_mask=None,
    ):
        self.benchmark_params = benchmark_params
        self.emulator_albedos = emulator_albedos
        self.reference_albedos = reference_albedos
        self.n_points = len(benchmark_params)
        self.unphysical_indices = unphysical_indices or []

        if physical_mask is None:
            physical_mask = np.ones(self.n_points, dtype=bool)
        self.physical_mask = physical_mask
        self.n_physical = int(np.sum(physical_mask))

        # Per-point errors (computed for all points)
        diff = np.abs(emulator_albedos - reference_albedos)
        self.mae_per_point = np.mean(diff, axis=1)
        self.max_err_per_point = np.max(diff, axis=1)

        bba_emu = np.sum(flx_slr * emulator_albedos, axis=1) / np.sum(flx_slr)
        bba_ref = np.sum(flx_slr * reference_albedos, axis=1) / np.sum(flx_slr)
        self.bba_err_per_point = np.abs(bba_emu - bba_ref)

        # Aggregate errors (only over physical points)
        if self.n_physical > 0:
            phys_diff = diff[physical_mask]
            self.mae = float(np.mean(phys_diff))
            self.max_err = float(np.max(phys_diff))
            self.mae_bba = float(np.mean(self.bba_err_per_point[physical_mask]))
            self.max_bba_err = float(np.max(self.bba_err_per_point[physical_mask]))

            ref_flat = reference_albedos[physical_mask].ravel()
            emu_flat = emulator_albedos[physical_mask].ravel()
            ss_res = np.sum((ref_flat - emu_flat) ** 2)
            ss_tot = np.sum((ref_flat - np.mean(ref_flat)) ** 2)
            self.r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 1.0
        else:
            self.mae = self.max_err = float("nan")
            self.mae_bba = self.max_bba_err = float("nan")
            self.r2 = float("nan")

    def summary(self):
        """Return a human-readable summary string."""
        header = f"Emulator verification ({self.n_points} benchmark points"
        if self.unphysical_indices:
            header += f", {len(self.unphysical_indices)} excluded as unphysical"
        header += ")"
        lines = [
            header,
            f"  Spectral MAE:      {self.mae:.6f}",
            f"  Spectral max err:  {self.max_err:.6f}",
            f"  BBA MAE:           {self.mae_bba:.6f}",
            f"  BBA max err:       {self.max_bba_err:.6f}",
            f"  R²:                {self.r2:.8f}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"VerificationResult(n={self.n_points}, mae={self.mae:.6f}, "
            f"r2={self.r2:.6f})"
        )


class Emulator:
    """Fast neural-network surrogate for the BioSNICAR forward model.

    Build with :meth:`build` (requires scikit-learn), then predict with
    :meth:`predict` (pure numpy, ~microseconds per call).  Save/load
    via :meth:`save` / :meth:`load` using a portable ``.npz`` format
    that does not require scikit-learn.

    Examples
    --------
    >>> emu = Emulator.build(
    ...     params={"rds": (100, 5000), "black_carbon": (0, 50000)},
    ...     n_samples=500, solzen=50, progress=False,
    ... )
    >>> albedo = emu.predict(rds=800, black_carbon=5000)
    >>> emu.save("my_emulator.npz")
    >>> emu2 = Emulator.load("my_emulator.npz")
    """

    def __init__(self):
        # Populated by build() or load()
        self._weights = []  # list of 2-D weight matrices
        self._biases = []  # list of 1-D bias vectors
        self._pca_components = None  # (n_pca, 480)
        self._pca_mean = None  # (480,)
        self._input_min = None  # (n_params,)
        self._input_max = None  # (n_params,)
        self._flx_slr = None  # (480,)
        self._param_names = []  # ordered list of param names
        self._bounds = OrderedDict()  # {name: (lo, hi)}
        self._n_pca_components = 0
        self._training_score = None
        self._metadata = {}

    # ── Properties ────────────────────────────────────────────────────

    @property
    def param_names(self):
        """Ordered list of parameter names."""
        return list(self._param_names)

    @property
    def bounds(self):
        """Parameter bounds ``{name: (min, max)}``."""
        return dict(self._bounds)

    @property
    def n_pca_components(self):
        """Number of PCA components retained."""
        return self._n_pca_components

    @property
    def training_score(self):
        """R² score on held-out validation set (from build)."""
        return self._training_score

    @property
    def flx_slr(self):
        """480-element solar flux spectrum stored at build time."""
        return self._flx_slr

    # ── Build ─────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        params,
        n_samples=10000,
        solver="adding-doubling",
        input_file="default",
        progress=True,
        seed=42,
        transform_fn=None,
        hidden_layer_sizes=(128, 128, 64),
        **fixed_overrides,
    ):
        """Build an emulator by training an MLP on forward-model outputs.

        Parameters
        ----------
        params : dict
            ``{param_name: (min_value, max_value)}`` for each free
            parameter.  Any keyword accepted by ``run_model()`` can be
            used (e.g. ``rds``, ``rho``, ``solzen``, ``direct``,
            ``black_carbon``, ``glacier_algae``, etc.).
        n_samples : int
            Number of Latin hypercube training samples.  More samples
            improve accuracy but increase build time (~50 ms per sample).
        solver : str
            ``"adding-doubling"`` (default) or ``"toon"``.
        input_file : str
            Path to YAML config, or ``"default"`` for the built-in
            ``inputs.yaml``.
        progress : bool
            Show a tqdm progress bar during forward-model runs.
        seed : int
            Random seed for Latin hypercube sampling and MLP training.
        transform_fn : callable or None
            Optional mapping ``(sampled_params: dict) -> run_model kwargs``.
            When provided it replaces the default behaviour of passing sampled
            parameters directly to ``run_model()``, allowing sea-ice and other
            multi-layer configurations where parameters must be broadcast to
            per-layer lists or mapped to structural kwargs (e.g. ``pond_depth``
            → ``dz=[pond_depth, 0.05, 1.40]``).  The callable receives a dict
            of ``{param_name: scalar}`` for all sampled parameters and must
            return a dict of valid ``run_model()`` keyword arguments.  When
            ``transform_fn`` is set, ``**fixed_overrides`` are ignored — the
            transform is responsible for the complete column specification.
        hidden_layer_sizes : tuple of int
            MLP hidden layer architecture.  Default ``(128, 128, 64)`` works
            well for most surfaces.  For bare sea ice (layer_type=4) the
            brine optics create a high-dimensional spectral manifold (~27 PCA
            components vs ~6 for snow); use ``(256, 256, 128, 64)`` or larger
            for substantially better accuracy on those surface types.
        **fixed_overrides
            Fixed parameters passed to every ``run_model()`` call.
            For glacier ice, ``layer_type=1`` is typical.
            Ignored when *transform_fn* is supplied.

        Returns
        -------
        Emulator
            Trained emulator ready for :meth:`predict` calls.
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.neural_network import MLPRegressor
        except ImportError:
            raise ImportError(
                "scikit-learn is required to build an emulator.  "
                "Install it with:  pip install scikit-learn>=1.0\n"
                "Once built, the emulator can be loaded and used "
                "without scikit-learn."
            )

        from biosnicar.drivers.run_model import run_model

        param_names = list(params.keys())
        bounds_ordered = OrderedDict(
            (name, (float(lo), float(hi))) for name, (lo, hi) in params.items()
        )
        n_dims = len(param_names)

        # --- 1. Latin hypercube sampling ---
        lhs_unit = _latin_hypercube(n_samples, n_dims, seed=seed)

        # Scale to parameter bounds.  Impurity parameters are sampled in
        # log10(x+1) space so that low concentrations (including clean ice)
        # are represented as well as high concentrations.
        lo_arr = np.array([bounds_ordered[n][0] for n in param_names])
        hi_arr = np.array([bounds_ordered[n][1] for n in param_names])
        lhs_scaled = lo_arr + lhs_unit * (hi_arr - lo_arr)

        for j, name in enumerate(param_names):
            if name in _LOG_SAMPLE_PARAMS:
                log_lo = np.log10(lo_arr[j] + 1)
                log_hi = np.log10(hi_arr[j] + 1)
                lhs_scaled[:, j] = (
                    10 ** (log_lo + lhs_unit[:, j] * (log_hi - log_lo)) - 1
                )

        # Snap binary parameters to {0, 1}
        for j, name in enumerate(param_names):
            if name in _BINARY_PARAMS:
                lhs_scaled[:, j] = np.round(lhs_scaled[:, j]).astype(float)

        # --- 2. Run forward model for each sample ---
        albedos = []
        flx_slr_ref = None

        iterator = range(n_samples)
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Building emulator")
            except ImportError:
                pass

        for i in iterator:
            overrides = dict(fixed_overrides)
            for j, name in enumerate(param_names):
                val = lhs_scaled[i, j]
                # Snap to valid lookup-table values
                if name == "rds":
                    val = _snap_rds(val)
                elif name in _BINARY_PARAMS:
                    val = int(val)
                elif name == "solzen":
                    val = int(np.clip(round(val), *_SOLZEN_RANGE))
                elif name in _INTEGER_PARAMS:
                    val = int(round(val))
                overrides[name] = val

            if transform_fn is not None:
                run_kwargs = transform_fn(dict(overrides))
            else:
                run_kwargs = overrides
            outputs = run_model(input_file=input_file, solver=solver, **run_kwargs)
            albedos.append(np.array(outputs.albedo, dtype=np.float64))
            if flx_slr_ref is None:
                flx_slr_ref = np.array(outputs.flx_slr, dtype=np.float64)

        albedo_matrix = np.array(albedos)  # (n_samples, 480)

        # Drop spectra with unphysical values.  The forward model
        # occasionally produces negative albedos at specific pathological
        # parameter combinations (numerical instability in the RT solver).
        # Including them would distort the PCA basis and MLP training.
        physical_mask = np.all((albedo_matrix >= 0) & (albedo_matrix <= 1.01), axis=1)
        n_unphysical = int(np.sum(~physical_mask))
        if n_unphysical:
            warnings.warn(
                f"{n_unphysical} of {n_samples} training spectra had "
                f"unphysical albedo values (outside [0, 1]) and were "
                f"excluded from training.",
                stacklevel=2,
            )
            albedo_matrix = albedo_matrix[physical_mask]
            lhs_scaled = lhs_scaled[physical_mask]

        # --- 3. PCA compression ---
        pca = PCA(n_components=0.999)  # retain 99.9% variance
        pca_coeffs = pca.fit_transform(albedo_matrix)
        n_pca = pca.n_components_

        # --- 4. Min-max scale inputs to [0, 1] ---
        input_min = lo_arr.copy()
        input_max = hi_arr.copy()
        X_scaled = (lhs_scaled - input_min) / (input_max - input_min + 1e-30)

        # --- 5. Train MLP ---
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            max_iter=2000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )
        mlp.fit(X_scaled, pca_coeffs)

        # Validation score (R² on held-out validation set)
        r2_score = mlp.score(X_scaled, pca_coeffs)

        # --- 6. Extract weights and assemble emulator ---
        emu = cls()
        emu._param_names = list(param_names)
        emu._bounds = bounds_ordered
        emu._input_min = input_min
        emu._input_max = input_max
        emu._pca_components = pca.components_.copy()  # (n_pca, 480)
        emu._pca_mean = pca.mean_.copy()  # (480,)
        emu._n_pca_components = n_pca
        emu._flx_slr = flx_slr_ref
        emu._training_score = float(r2_score)

        # Extract MLP weights/biases as numpy arrays
        emu._weights = [w.copy() for w in mlp.coefs_]
        emu._biases = [b.copy() for b in mlp.intercepts_]

        emu._metadata = {
            "param_names": list(param_names),
            "bounds": {n: list(v) for n, v in bounds_ordered.items()},
            "n_samples": int(n_samples),
            "n_pca_components": int(n_pca),
            "training_r2": float(r2_score),
            "build_timestamp": datetime.utcnow().isoformat(),
            "fixed_overrides": {k: _jsonable(v) for k, v in fixed_overrides.items()},
            "solver": solver,
            "transform_fn": getattr(transform_fn, "__name__", None),
            "hidden_layer_sizes": list(hidden_layer_sizes),
        }

        return emu

    # ── Predict ───────────────────────────────────────────────────────

    def predict(self, **params):
        """Predict 480-band spectral albedo (~microseconds).

        Parameters
        ----------
        **params
            Keyword arguments for each emulator parameter
            (e.g. ``rds=800, black_carbon=5000``).

        Returns
        -------
        np.ndarray of shape (480,)
            Predicted spectral albedo, clipped to [0, 1].
        """
        x = self._params_to_scaled(params)
        return self._forward_pass(x).ravel()

    def predict_batch(self, points):
        """Predict for N parameter combinations.

        Parameters
        ----------
        points : np.ndarray of shape (N, n_params)
            Each row is one parameter set in the order of
            :attr:`param_names`.

        Returns
        -------
        np.ndarray of shape (N, 480)
            Predicted spectral albedo per row.
        """
        points = np.atleast_2d(points)
        # Scale
        x = (points - self._input_min) / (self._input_max - self._input_min + 1e-30)
        x = np.clip(x, 0.0, 1.0)
        return self._forward_pass(x)

    def _params_to_scaled(self, params_dict):
        """Convert keyword params to a scaled (0-1) 1-D array."""
        missing = set(self._param_names) - set(params_dict.keys())
        if missing:
            raise ValueError(f"Missing parameters: {missing}")
        x = np.array([float(params_dict[n]) for n in self._param_names])
        # Warn only for violations larger than 0.1 % of the parameter range.
        # Small violations (< 1e-3 * range) are numerical artefacts from the
        # finite-difference gradient steps used by the optimiser and Hessian
        # computation — they do not affect predictions because the scaled value
        # is clipped to [0, 1] below.  Larger violations indicate genuine
        # extrapolation and still produce a warning.
        for i, name in enumerate(self._param_names):
            lo, hi = self._bounds[name]
            tol = 1e-3 * (hi - lo)
            if x[i] < lo - tol or x[i] > hi + tol:
                warnings.warn(
                    f"Parameter {name}={x[i]} is outside training bounds "
                    f"[{lo}, {hi}].  Predictions may be unreliable.",
                    stacklevel=3,
                )
        x = (x - self._input_min) / (self._input_max - self._input_min + 1e-30)
        x = np.clip(x, 0.0, 1.0)
        return x.reshape(1, -1)

    def _forward_pass(self, x):
        """Pure-numpy MLP forward pass + PCA reconstruction.

        Parameters
        ----------
        x : np.ndarray of shape (N, n_params)
            Scaled inputs in [0, 1].

        Returns
        -------
        np.ndarray of shape (N, 480)
        """
        for i, (W, b) in enumerate(zip(self._weights, self._biases)):
            x = x @ W + b
            if i < len(self._weights) - 1:
                x = np.maximum(x, 0)  # ReLU on all but last layer
        # x is now PCA coefficients of shape (N, n_pca)
        albedo = x @ self._pca_components + self._pca_mean
        return np.clip(albedo, 0.0, 1.0)

    # ── Verify ──────────────────────────────────────────────────────────

    def verify(
        self,
        benchmark_params=None,
        n_points=20,
        seed=123,
        input_file="default",
        progress=True,
    ):
        """Measure emulator accuracy against the full forward model.

        Runs both the emulator and ``run_model()`` for a suite of benchmark
        parameter sets and computes error statistics.

        Parameters
        ----------
        benchmark_params : list of dict, optional
            Explicit parameter sets to test.  Each dict must contain all
            emulator parameters (e.g. ``{"rds": 500, "black_carbon": 0}``).
            If *None*, generates *n_points* stratified random points
            spanning the training bounds.
        n_points : int
            Number of benchmark points to generate when *benchmark_params*
            is not provided.  Ignored if *benchmark_params* is given.
        seed : int
            Random seed for benchmark generation.
        input_file : str
            YAML config path for ``run_model()``, or ``"default"``.
        progress : bool
            Show a progress bar for the forward-model calls.

        Returns
        -------
        VerificationResult
            Object with per-point and aggregate error metrics, plus
            a ``.summary()`` method for pretty-printing.
        """
        from biosnicar.drivers.run_model import run_model

        # --- Build benchmark parameter sets ---
        if benchmark_params is None:
            benchmark_params = self._generate_benchmark_params(n_points, seed)

        n = len(benchmark_params)

        # Get fixed overrides from build metadata
        fixed = dict(self._metadata.get("fixed_overrides", {}))
        solver = self._metadata.get("solver", "adding-doubling")

        # --- Run emulator and forward model ---
        emu_albedos = np.empty((n, 480))
        ref_albedos = np.empty((n, 480))

        iterator = range(n)
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Verifying emulator")
            except ImportError:
                pass

        unphysical_indices = []

        for i in iterator:
            params = benchmark_params[i]

            # Emulator prediction
            emu_albedos[i] = self.predict(**params)

            # Forward model reference
            overrides = dict(fixed)
            overrides.update(params)
            outputs = run_model(input_file=input_file, solver=solver, **overrides)
            raw = np.array(outputs.albedo, dtype=np.float64)

            if np.any(raw < 0) or np.any(raw > 1.01):
                unphysical_indices.append(i)
            ref_albedos[i] = raw

        # Exclude unphysical points from error statistics
        physical_mask = np.ones(n, dtype=bool)
        if unphysical_indices:
            physical_mask[unphysical_indices] = False
            warnings.warn(
                f"Forward model produced unphysical albedo (outside [0, 1]) "
                f"for {len(unphysical_indices)} of {n} benchmark points.  "
                f"These have been excluded from error statistics.",
                stacklevel=2,
            )

        return VerificationResult(
            benchmark_params=benchmark_params,
            emulator_albedos=emu_albedos,
            reference_albedos=ref_albedos,
            flx_slr=self._flx_slr,
            unphysical_indices=unphysical_indices,
            physical_mask=physical_mask,
        )

    def _generate_benchmark_params(self, n_points, seed):
        """Generate stratified benchmark parameter sets within bounds."""
        n_dims = len(self._param_names)
        lhs = _latin_hypercube(n_points, n_dims, seed=seed)

        lo = np.array([self._bounds[n][0] for n in self._param_names])
        hi = np.array([self._bounds[n][1] for n in self._param_names])
        scaled = lo + lhs * (hi - lo)

        # Log-space sampling for impurities (matches training distribution)
        for j, name in enumerate(self._param_names):
            if name in _LOG_SAMPLE_PARAMS:
                log_lo = np.log10(lo[j] + 1)
                log_hi = np.log10(hi[j] + 1)
                scaled[:, j] = 10 ** (log_lo + lhs[:, j] * (log_hi - log_lo)) - 1

        params_list = []
        for i in range(n_points):
            d = {}
            for j, name in enumerate(self._param_names):
                val = scaled[i, j]
                if name == "rds":
                    val = _snap_rds(val)
                elif name in _BINARY_PARAMS:
                    val = int(round(val))
                elif name == "solzen":
                    val = int(np.clip(round(val), *_SOLZEN_RANGE))
                elif name in _INTEGER_PARAMS:
                    val = int(round(val))
                d[name] = val
            params_list.append(d)

        return params_list

    # ── Save / Load ───────────────────────────────────────────────────

    def save(self, path):
        """Save emulator to a ``.npz`` file.

        The file contains only numpy arrays and a JSON metadata string
        — no pickle, no sklearn objects.  This ensures portability
        across Python and sklearn versions.

        Parameters
        ----------
        path : str or Path
            Output file path (should end in ``.npz``).
        """
        arrays = {}
        for i, (w, b) in enumerate(zip(self._weights, self._biases)):
            arrays["weights_{}".format(i)] = w
            arrays["biases_{}".format(i)] = b
        arrays["pca_components"] = self._pca_components
        arrays["pca_mean"] = self._pca_mean
        arrays["input_min"] = self._input_min
        arrays["input_max"] = self._input_max
        arrays["flx_slr"] = self._flx_slr
        arrays["metadata"] = np.array(json.dumps(self._metadata))
        np.savez_compressed(str(path), **arrays)

    @classmethod
    def load(cls, path):
        """Load emulator from a ``.npz`` file.  No sklearn required.

        Parameters
        ----------
        path : str or Path
            Path to a ``.npz`` file produced by :meth:`save`.

        Returns
        -------
        Emulator
        """
        data = np.load(str(path), allow_pickle=False)
        meta = json.loads(str(data["metadata"]))

        emu = cls()
        emu._param_names = meta["param_names"]
        emu._bounds = OrderedDict(
            (n, tuple(meta["bounds"][n])) for n in meta["param_names"]
        )
        emu._n_pca_components = meta["n_pca_components"]
        emu._training_score = meta.get("training_r2")
        emu._metadata = meta

        emu._pca_components = data["pca_components"]
        emu._pca_mean = data["pca_mean"]
        emu._input_min = data["input_min"]
        emu._input_max = data["input_max"]
        emu._flx_slr = data["flx_slr"]

        # Load weight/bias pairs
        i = 0
        while "weights_{}".format(i) in data:
            emu._weights.append(data["weights_{}".format(i)])
            emu._biases.append(data["biases_{}".format(i)])
            i += 1

        return emu

    def __repr__(self):
        return "Emulator(params={}, n_pca={}, training_r2={})".format(
            self._param_names,
            self._n_pca_components,
            self._training_score,
        )


def _jsonable(v):
    """Convert a value to a JSON-serialisable type."""
    if isinstance(v, (np.integer, np.int_)):
        return int(v)
    if isinstance(v, (np.floating, np.float64)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v
