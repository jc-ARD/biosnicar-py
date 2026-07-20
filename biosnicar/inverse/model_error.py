"""Forward-model error covariance for OE retrievals (roadmap A7).

The OE measurement covariance ``S_e`` was instrument-noise-only (diagonal,
default 0.02), which treats every band as an independent piece of evidence.
The field-residual library (``tests/validation_data/build_residual_library.py``)
shows real post-fit residuals are smooth, strongly correlated curves: ~2
effective independent degrees of freedom over 60 VIS bands. Ignoring that
correlation over-counts spectral evidence ~30x and is the measured mechanism
behind the overconfident (p~1.00) OE misclassifications
(``docs/sea_ice_validation.md`` §12).

This module provides a low-rank + diagonal model of the forward-model error:

    S_model = B diag(lam) B^T + diag(d)

* ``B`` — leading EOFs of the correct-model field residuals (uncentred
  second moments, so systematic model bias is folded into the error budget
  rather than pretended away),
* ``lam`` — their variances,
* ``d`` — per-band residual variance not captured by the EOFs (and the full
  per-band variance outside the EOF domain).

At retrieval time ``S_e = diag(sigma_instrument^2) + S_model[mask, mask]``.

Calibration honesty (v1):

* Fitted on the SHEBA + Smith **correct-model** rows (``is_expected &
  is_winner``); **Istomina is held out** for coverage validation — never
  calibrate S_e and quote validation numbers from the same campaign.
* SHEBA spectra are VIS-only, so the EOF *correlation* structure is
  necessarily calibrated on Smith full-range rows alone; SHEBA still
  informs the VIS diagonal. Stated, not hidden.
* No field residuals exist for young_ice or open_water: this model is
  calibrated on snow/bare/melting-ice surfaces and is applied fleet-wide as
  the best available estimate — a pooling assumption, flagged here.
* The residuals bundle footprint heterogeneity, geometry and calibration
  effects with model error; that bundle is what a real retrieval faces.

Build the shipped artifact with ``scripts/build_model_error_covariance.py``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

import biosnicar

MODEL_ERROR_VERSION = 1
DEFAULT_ARTIFACT = str(
    Path(biosnicar.DATA_DIR) / "model_error" / "field_se_v1.npz"
)

# Minimum per-band model-error floor, 1-sigma 0.005. Well below the observed
# correct-model residual median (~0.018, A7 step 1) so it never inflates the
# real error, but non-zero so a sparsely-sampled or EOF-emptied band can never
# be claimed better-constrained than any plausible field measurement (which
# would re-introduce the over-counting this whole term exists to prevent).
_FLOOR_SIGMA = 0.005


@dataclass
class ModelErrorCovariance:
    """Low-rank + diagonal forward-model error covariance on the 480 grid."""

    eof_basis: np.ndarray      # (k, 480), zero outside the EOF domain
    eof_var: np.ndarray        # (k,)
    diag_var: np.ndarray       # (480,) residual diagonal variance
    domain: np.ndarray         # (480,) bool — bands with calibrated covariance
    meta: dict

    def dense(self, mask):
        """S_model restricted to the masked bands, as a dense (m, m) array."""
        mask = np.asarray(mask, dtype=bool)
        B = self.eof_basis[:, mask]
        S = (B.T * self.eof_var) @ B
        S[np.diag_indices_from(S)] += self.diag_var[mask]
        return S

    def effective_dof(self, mask):
        """Participation-ratio effective DOF of S_model over the masked bands."""
        ev = np.linalg.eigvalsh(self.dense(mask))
        ev = np.clip(ev, 0, None)
        return float(ev.sum() ** 2 / (ev**2).sum())

    def save(self, path):
        np.savez_compressed(
            path,
            version=np.array(MODEL_ERROR_VERSION),
            eof_basis=self.eof_basis.astype(np.float32),
            eof_var=self.eof_var.astype(np.float64),
            diag_var=self.diag_var.astype(np.float64),
            domain=self.domain.astype(bool),
            meta=np.array(_meta_to_json(self.meta)),
        )

    @classmethod
    def load(cls, path=None):
        import json
        path = str(path or DEFAULT_ARTIFACT)
        cached = _LOAD_CACHE.get(path)
        if cached is not None:
            return cached
        data = np.load(path, allow_pickle=False)
        model = cls(
            eof_basis=np.asarray(data["eof_basis"], dtype=float),
            eof_var=np.asarray(data["eof_var"], dtype=float),
            diag_var=np.asarray(data["diag_var"], dtype=float),
            domain=np.asarray(data["domain"], dtype=bool),
            meta=json.loads(str(data["meta"])),
        )
        _LOAD_CACHE[path] = model
        return model


_LOAD_CACHE: dict = {}


def _meta_to_json(meta):
    import json
    return json.dumps(meta)


def fit_model_error(residuals, masks, n_eofs=3, eof_min_rows=30,
                    diag_min_rows=8, domain_max_nm=None, meta=None):
    """Fit the low-rank + diagonal model from residual rows.

    Parameters
    ----------
    residuals : (n, 480) array
        Post-fit residuals, NaN outside each row's mask.
    masks : (n, 480) bool array
        Valid-band masks per row.
    n_eofs : int
        Number of EOFs for the correlated term. Default 3 because the field
        residuals concentrate ~94% of their variance in the first 3 modes
        (A7 step-1 analysis, docs/OE_MODEL_ERROR_EXPERIMENT.md §4); more modes
        fit calibration noise, fewer under-model the correlation. It is also
        the effective rank the correlation collapses to (~2-3 DOF over 60 VIS
        bands), so 3 spans the real structure without overfitting.
    eof_min_rows : int
        A band enters the EOF domain only if at least this many rows cover it,
        and only rows covering the full domain enter the SVD. Default 30 — a
        floor for a stable low-rank covariance estimate (n_rows must exceed a
        small multiple of n_eofs, and 30 is what the calibration set actually
        supplies for the VIS domain). Widening the domain into the SWIR pushed
        n_rows below this and empirically worsened held-out chi2, which is why
        v1 is VIS-only.
    diag_min_rows : int
        Minimum rows for a per-band variance estimate; bands below this get the
        median calibrated variance instead. Default 8 — a variance from fewer
        than ~8 samples is too noisy to trust, and the median fallback is
        deliberately conservative (better too wide than falsely precise).
    """
    residuals = np.asarray(residuals, dtype=float)
    masks = np.asarray(masks, dtype=bool)
    n, nb = residuals.shape

    # Per-band diagonal variance (uncentred: bias is part of the error budget)
    counts = masks.sum(axis=0)
    diag_var = np.full(nb, np.nan)
    for b in range(nb):
        if counts[b] >= diag_min_rows:
            v = residuals[masks[:, b], b]
            diag_var[b] = float(np.mean(v**2))
    fallback = float(np.nanmedian(diag_var))
    calibrated = np.isfinite(diag_var)
    diag_var[~calibrated] = fallback

    # EOF domain: densely covered bands; EOF rows: full coverage of the domain
    domain = counts >= eof_min_rows
    if domain_max_nm is not None:
        # Explicitly cap the correlated-term domain (e.g. to the VIS). The SWIR
        # error has different correlation structure and, with only a handful of
        # full-range calibration rows, a single VIS+SWIR EOF fit either
        # regresses the held-out chi2 or collapses to near-diagonal — so the
        # SWIR extension awaits a gap-tolerant (EM-PCA) fit, not just a wider
        # domain. Bands above the cap keep their per-band diagonal variance.
        wl_nm = 205.0 + 10.0 * np.arange(nb)
        domain &= wl_nm <= domain_max_nm
    rows_full = masks[:, domain].all(axis=1)
    eof_basis = np.zeros((n_eofs, nb))
    eof_var = np.zeros(n_eofs)
    if domain.any() and rows_full.sum() >= eof_min_rows:
        R = residuals[rows_full][:, domain]
        # Uncentred SVD: eigenvectors of the second-moment matrix R^T R / n
        _, s, vt = np.linalg.svd(R, full_matrices=False)
        k = min(n_eofs, vt.shape[0])
        eof_basis[:k, domain] = vt[:k]
        eof_var[:k] = (s[:k] ** 2) / R.shape[0]
        # Remove the EOF-captured variance from the diagonal inside the domain
        captured = ((eof_basis[:, domain].T ** 2) * eof_var).sum(axis=1)
        diag_var[domain] = np.maximum(diag_var[domain] - captured,
                                      _FLOOR_SIGMA**2)
    diag_var = np.maximum(diag_var, _FLOOR_SIGMA**2)

    m = dict(meta or {})
    m.update(n_rows=int(n), n_eof_rows=int(rows_full.sum()),
             n_eofs=int(n_eofs), eof_domain_bands=int(domain.sum()),
             version=MODEL_ERROR_VERSION)
    return ModelErrorCovariance(eof_basis=eof_basis, eof_var=eof_var,
                                diag_var=diag_var, domain=domain, meta=m)
