"""Tests for batch retrieval (D1) and SeaIceSceneResult exports (D2)."""

import json
from pathlib import Path

import numpy as np
import pytest

from biosnicar.sea_ice.emulator_configs import (
    SEA_ICE_EMULATOR_CONFIGS,
    load_sea_ice_emulators,
    trained_emulator_names,
)
from biosnicar.sea_ice.open_water import OpenWaterModel
from biosnicar.sea_ice.retrieve import retrieve_sea_ice_batch
from biosnicar.sea_ice.scene_result import (
    NO_DATA_CODE,
    SURFACE_TYPE_CODES,
    SeaIceSceneResult,
)

xr = pytest.importorskip("xarray")

BUILT = all(
    Path(SEA_ICE_EMULATOR_CONFIGS[n]["emulator_file"]).exists()
    for n in trained_emulator_names()
)


def _records():
    return [
        {
            "surface_type": "open_water",
            "confidence": 0.99,
            "cost": 0.001,
            "quality_flags": 0x20,
            "parameters": {"wind_speed_ms": 5.0},
            "uncertainty": {"wind_speed_ms": 0.5},
        },
        None,  # no-data pixel
        {
            "surface_type": "FYI_pond",
            "confidence": 0.6,
            "cost": 0.01,
            "quality_flags": 0,
            "parameters": {"pond_depth": 0.2, "sea_ice_temperature": -4.0},
            "uncertainty": {"pond_depth": 0.05, "sea_ice_temperature": 1.0},
        },
        {
            "surface_type": "FYI_pond",
            "confidence": 0.7,
            "cost": 0.02,
            "quality_flags": 0x04,
            "parameters": {"pond_depth": 0.3, "sea_ice_temperature": -3.0},
            "uncertainty": {"pond_depth": 0.05, "sea_ice_temperature": 1.0},
        },
    ]


class TestFromRecords:
    def test_flat_scene(self):
        scene = SeaIceSceneResult.from_records(_records())
        ds = scene.to_xarray()
        assert ds.sizes == {"pixel": 4}
        assert ds["surface_type_code"].values.tolist() == [
            SURFACE_TYPE_CODES["open_water"], NO_DATA_CODE,
            SURFACE_TYPE_CODES["FYI_pond"], SURFACE_TYPE_CODES["FYI_pond"],
        ]
        assert np.isnan(ds["confidence"].values[1])
        assert np.isnan(ds["param_pond_depth"].values[0])
        assert ds["param_pond_depth"].values[2] == pytest.approx(0.2)
        assert ds["quality_flags"].dtype == np.uint8

    def test_2d_scene(self):
        scene = SeaIceSceneResult.from_records(_records(), shape=(2, 2))
        ds = scene.to_xarray()
        assert ds.sizes == {"y": 2, "x": 2}
        assert ds["surface_type"].values[0, 0] == "open_water"

    def test_dataframe(self):
        df = SeaIceSceneResult.from_records(_records()).to_dataframe()
        assert len(df) == 4
        assert "param_pond_depth" in df.columns

    def test_summary(self):
        s = SeaIceSceneResult.from_records(_records()).summary()
        assert "no-data" in s and "FYI_pond" in s


class TestExports:
    def test_netcdf_roundtrip(self, tmp_path):
        scene = SeaIceSceneResult.from_records(_records(), shape=(2, 2))
        path = tmp_path / "scene.nc"
        scene.to_netcdf(str(path))
        back = xr.open_dataset(path)
        assert back["surface_type_code"].shape == (2, 2)
        assert json.loads(back.attrs["surface_type_codes"]) == SURFACE_TYPE_CODES
        back.close()

    def test_geotiff_roundtrip(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        pytest.importorskip("rioxarray")
        from rasterio.transform import from_origin

        transform = from_origin(500000, 8_000_000, 10, 10)
        scene = SeaIceSceneResult.from_records(
            _records(), shape=(2, 2), crs="EPSG:32633", transform=transform,
        )
        path = tmp_path / "scene.tif"
        scene.to_geotiff(str(path))
        with rasterio.open(path) as src:
            assert src.count == 4  # code, confidence, cost, quality_flags
            assert src.width == 2 and src.height == 2
            band1 = src.read(1)
            assert band1[0, 0] == SURFACE_TYPE_CODES["open_water"]
            assert band1[0, 1] == NO_DATA_CODE

    def test_geotiff_requires_georef(self):
        scene = SeaIceSceneResult.from_records(_records(), shape=(2, 2))
        with pytest.raises(ValueError, match="crs"):
            scene.to_geotiff("/tmp/never.tif")

    def test_h3_geojson(self, tmp_path):
        pytest.importorskip("h3")
        latlon = np.array([
            [78.001, 15.001], [78.001, 15.002],
            [78.002, 15.001], [78.002, 15.002],
        ])
        scene = SeaIceSceneResult.from_records(
            _records(), shape=(2, 2), latlon=latlon,
        )
        path = tmp_path / "scene.geojson"
        scene.to_h3_geojson(str(path), resolution=6)
        gj = json.loads(path.read_text())
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) >= 1
        f0 = gj["features"][0]
        assert f0["geometry"]["type"] == "Polygon"
        assert "surface_type" in f0["properties"]
        # 3 valid pixels in one ~36 km2 res-6 cell: mode is FYI_pond
        total = sum(f["properties"]["n_pixels"] for f in gj["features"])
        assert total == 3  # no-data pixel excluded

    def test_h3_requires_latlon(self):
        scene = SeaIceSceneResult.from_records(_records())
        with pytest.raises(ValueError, match="lat/lon"):
            scene.to_h3_geojson("/tmp/never.geojson")


@pytest.mark.skipif(not BUILT, reason="pre-built sea ice emulators not found")
class TestBatchRetrieval:
    @pytest.fixture(scope="class")
    def scene(self):
        from biosnicar.drivers.run_model import run_model

        ow = OpenWaterModel()
        fleet = load_sea_ice_emulators(["FYI_pond"])
        fleet["open_water"] = ow
        rng = np.random.default_rng(11)
        water = ow.predict(solzen=60, wind_speed_ms=4.0)
        # forward model, not the emulator, generates the pond observation
        pond = np.asarray(run_model(
            **SEA_ICE_EMULATOR_CONFIGS["FYI_pond"]["transform_fn"](dict(
                pond_depth=0.25, sea_ice_temperature=-4, black_carbon=100,
                solzen=60, direct=1,
            ))
        ).albedo)
        pixels = []
        for i in range(10):
            base = water if i < 5 else pond
            pixels.append(np.clip(base + rng.normal(0, 0.002, 480), 0, 1))
        pixels[3] = np.full(480, np.nan)  # masked pixel
        obs = np.array(pixels).reshape(2, 5, 480)
        coords = np.dstack([
            np.full((2, 5), 78.0) + 0.001 * np.arange(5),
            np.full((2, 5), 15.0) + 0.001 * np.arange(2)[:, None],
        ])
        return retrieve_sea_ice_batch(
            obs, n_jobs=2, chunksize=4,
            spatial_coords=coords, crs="EPSG:4326",
            emulators=fleet, solzen=60, direct=1,
        )

    def test_shapes_and_types(self, scene):
        ds = scene.to_xarray()
        assert ds.sizes == {"y": 2, "x": 5}
        code = ds["surface_type_code"].values
        flat_types = ds["surface_type"].values.ravel()
        assert flat_types[0] == "open_water"
        assert flat_types[9] == "FYI_pond"
        assert code.ravel()[3] == NO_DATA_CODE

    def test_parameters_recovered(self, scene):
        ds = scene.to_xarray()
        wind = ds["param_wind_speed_ms"].values.ravel()
        assert abs(np.nanmean(wind[:3]) - 4.0) < 2.0
        depth = ds["param_pond_depth"].values.ravel()
        assert abs(np.nanmean(depth[5:]) - 0.25) < 0.05

    def test_quality_flags_populated(self, scene):
        ds = scene.to_xarray()
        flags = ds["quality_flags"].values.ravel()
        assert flags[0] & 0x20  # open_water informational bit
