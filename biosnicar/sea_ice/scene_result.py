"""Spatial scene output for batch sea ice retrievals.

:class:`SeaIceSceneResult` wraps an :mod:`xarray` Dataset holding per-pixel
classification and parameter fields, and provides export to pandas, NetCDF,
GeoTIFF (via rioxarray) and H3-aggregated GeoJSON.

The xarray Dataset is the single internal representation; every export method
operates on it.  This keeps the retrieval engine swappable: a future
vectorised inverse network only needs to produce the same Dataset.

Variable naming: retrieved parameters are stored as ``param_<name>`` and
their 1-sigma uncertainties as ``unc_<name>`` (xarray/NetCDF variable names
cannot contain ``.``).

Requires the ``geo`` optional dependencies::

    pip install biosnicar[geo]
"""

import json
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

# Integer codes for raster export of the categorical surface type.
SURFACE_TYPE_CODES = {
    "FYI_bare":   0,
    "FYI_snow":   1,
    "FYI_summer": 2,
    "MYI_bare":   3,
    "FYI_pond":   4,
    "open_water": 5,
    "young_ice":  6,
}
NO_DATA_CODE = 255

_CODE_TO_TYPE = {v: k for k, v in SURFACE_TYPE_CODES.items()}

_DEFAULT_RASTER_VARS = ["surface_type_code", "confidence", "cost", "quality_flags"]


def _require(module, feature):
    try:
        return __import__(module)
    except ImportError:
        raise ImportError(
            f"{feature} requires the '{module}' package.  "
            f"Install the geo extras with:  pip install biosnicar[geo]"
        )


@dataclass
class SeaIceSceneResult:
    """Batch retrieval result for a spatial scene.

    Core object is an xarray Dataset (access via :meth:`to_xarray`).
    Pixels that could not be retrieved (masked input or failed optimisation)
    carry ``surface_type_code == 255`` and NaN parameter fields.
    """

    _ds: Any                    # xr.Dataset — access via to_xarray()
    crs: Optional[str] = None   # EPSG string, e.g. "EPSG:4326"
    transform: Optional[Any] = None  # rasterio.Affine or None

    # ── Construction ────────────────────────────────────────────────────

    @classmethod
    def from_records(cls, records: List[Optional[dict]], shape=None,
                     latlon=None, crs=None, transform=None):
        """Assemble from per-pixel record dicts (None = no-data pixel).

        Each record holds ``surface_type``, ``confidence``, ``cost``,
        ``quality_flags``, ``parameters`` and ``uncertainty`` dicts.
        *shape* (H, W) reshapes the flat pixel list to a 2-D grid;
        *latlon* is an (N, 2) array of (lat, lon) per pixel.
        """
        xr = _require("xarray", "SeaIceSceneResult")

        n = len(records)
        param_names = sorted({
            p for r in records if r is not None for p in r["parameters"]
        })

        stype = np.full(n, "no_data", dtype="U16")
        code = np.full(n, NO_DATA_CODE, dtype=np.uint8)
        conf = np.full(n, np.nan, dtype=np.float32)
        cost = np.full(n, np.nan, dtype=np.float32)
        qflags = np.zeros(n, dtype=np.uint8)
        params = {p: np.full(n, np.nan, dtype=np.float32) for p in param_names}
        uncs = {p: np.full(n, np.nan, dtype=np.float32) for p in param_names}

        for i, r in enumerate(records):
            if r is None:
                continue
            stype[i] = r["surface_type"]
            code[i] = SURFACE_TYPE_CODES.get(r["surface_type"], NO_DATA_CODE)
            conf[i] = r["confidence"]
            cost[i] = r["cost"]
            qflags[i] = r["quality_flags"]
            for p, v in r["parameters"].items():
                params[p][i] = v
            for p, v in r.get("uncertainty", {}).items():
                if p in uncs and np.isfinite(v):
                    uncs[p][i] = v

        if shape is not None:
            dims = ("y", "x")
            def _r(a):
                return a.reshape(shape)
        else:
            dims = ("pixel",)
            def _r(a):
                return a

        data_vars = {
            "surface_type": (dims, _r(stype)),
            "surface_type_code": (dims, _r(code)),
            "confidence": (dims, _r(conf)),
            "cost": (dims, _r(cost)),
            "quality_flags": (dims, _r(qflags)),
        }
        for p in param_names:
            data_vars[f"param_{p}"] = (dims, _r(params[p]))
            data_vars[f"unc_{p}"] = (dims, _r(uncs[p]))

        if latlon is not None:
            latlon = np.asarray(latlon, dtype=np.float32).reshape(n, 2)
            data_vars["lat"] = (dims, _r(latlon[:, 0].copy()))
            data_vars["lon"] = (dims, _r(latlon[:, 1].copy()))

        ds = xr.Dataset(
            data_vars,
            attrs={
                "surface_type_codes": json.dumps(SURFACE_TYPE_CODES),
                "no_data_code": NO_DATA_CODE,
                "crs": crs or "",
            },
        )
        return cls(_ds=ds, crs=crs, transform=transform)

    # ── Accessors ───────────────────────────────────────────────────────

    def to_xarray(self):
        """Return the underlying xarray Dataset."""
        return self._ds

    def to_dataframe(self):
        """Flatten to a pandas DataFrame (one row per pixel)."""
        return self._ds.to_dataframe().reset_index()

    # ── Exports ─────────────────────────────────────────────────────────

    def to_netcdf(self, path: str):
        """Write the full Dataset to NetCDF4 (xarray native)."""
        self._ds.to_netcdf(path)

    def to_geotiff(self, path: str, variables=None, compress="lzw"):
        """Write selected variables to a multi-band GeoTIFF (float32).

        Requires rioxarray, a 2-D scene, and the CRS + affine transform
        supplied to :func:`retrieve_sea_ice_batch`.
        """
        _require("rioxarray", "to_geotiff")
        import xarray as xr

        if self.crs is None or self.transform is None:
            raise ValueError(
                "to_geotiff requires `crs` and `transform` — pass them to "
                "retrieve_sea_ice_batch()."
            )
        if "y" not in self._ds.dims:
            raise ValueError("to_geotiff requires a 2-D (H, W) scene.")

        variables = variables or [
            v for v in _DEFAULT_RASTER_VARS if v in self._ds
        ]
        bad = [v for v in variables if self._ds[v].dtype.kind not in "uif"]
        if bad:
            raise ValueError(f"Non-numeric variables cannot be rasterised: {bad}")

        stack = xr.concat(
            [self._ds[v].astype(np.float32) for v in variables],
            dim="band",
        )
        stack = stack.assign_coords(band=("band", np.arange(1, len(variables) + 1)))
        stack.attrs["long_name"] = variables
        stack = stack.rio.write_crs(self.crs)
        stack = stack.rio.write_transform(self.transform)
        stack.rio.to_raster(path, compress=compress)

    def to_h3_geojson(self, path: str, resolution: int = 8,
                      variables=None, aggregation="mode"):
        """Aggregate pixels to H3 cells and write a GeoJSON FeatureCollection.

        Categorical variables aggregate by mode, continuous by mean
        (*aggregation* sets the default for ambiguous integer fields).
        Requires per-pixel ``lat``/``lon`` (pass ``spatial_coords`` to
        :func:`retrieve_sea_ice_batch`).  At resolution 8 cells are ~0.7 km²,
        at 9 ~0.1 km²; for Sentinel-2 10 m pixels use 9-10.
        """
        h3 = _require("h3", "to_h3_geojson")

        if "lat" not in self._ds or "lon" not in self._ds:
            raise ValueError(
                "to_h3_geojson requires lat/lon — pass `spatial_coords` to "
                "retrieve_sea_ice_batch()."
            )

        variables = variables or ["surface_type", "confidence"]
        lat = self._ds["lat"].values.ravel()
        lon = self._ds["lon"].values.ravel()
        valid = self._ds["surface_type_code"].values.ravel() != NO_DATA_CODE
        valid &= np.isfinite(lat) & np.isfinite(lon)

        # h3 v4 renamed the v3 API
        to_cell = getattr(h3, "latlng_to_cell", None) or h3.geo_to_h3
        to_boundary = getattr(h3, "cell_to_boundary", None) or h3.h3_to_geo_boundary

        cells = {}
        idx_valid = np.flatnonzero(valid)
        for i in idx_valid:
            cell = to_cell(float(lat[i]), float(lon[i]), resolution)
            cells.setdefault(cell, []).append(i)

        var_data = {v: self._ds[v].values.ravel() for v in variables}

        features = []
        for cell, idxs in cells.items():
            props = {"h3_index": cell, "n_pixels": len(idxs)}
            for v in variables:
                vals = var_data[v][idxs]
                if var_data[v].dtype.kind in "US O":
                    props[v] = Counter(vals).most_common(1)[0][0]
                elif var_data[v].dtype.kind in "ui" and aggregation == "mode":
                    props[v] = int(Counter(vals.tolist()).most_common(1)[0][0])
                else:
                    finite = vals[np.isfinite(vals.astype(float))]
                    props[v] = float(np.mean(finite)) if len(finite) else None
            boundary = to_boundary(cell)  # ((lat, lng), ...)
            ring = [[float(lng), float(lat_)] for lat_, lng in boundary]
            ring.append(ring[0])
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": props,
            })

        with open(path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)

    # ── Convenience ─────────────────────────────────────────────────────

    def summary(self) -> str:
        code = self._ds["surface_type_code"].values.ravel()
        n = code.size
        n_valid = int((code != NO_DATA_CODE).sum())
        lines = [
            f"SeaIceSceneResult: {n} pixels ({n_valid} retrieved, "
            f"{n - n_valid} no-data)",
        ]
        for name, c in SURFACE_TYPE_CODES.items():
            count = int((code == c).sum())
            if count:
                lines.append(f"  {name:12s} {count:8d}  ({count / n_valid:.1%})")
        flagged = int((self._ds["quality_flags"].values.ravel() != 0).sum())
        lines.append(f"  quality-flagged pixels: {flagged}")
        return "\n".join(lines)

    def __repr__(self):
        shape = dict(self._ds.sizes)
        return f"SeaIceSceneResult(dims={shape}, crs={self.crs!r})"
