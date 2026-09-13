"""Optional HF environment: synthetic hierarchy and independent native comparisons."""

import asyncio
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

zarr = pytest.importorskip("zarr", minversion="3.3")
NativeDataset = pytest.importorskip("spaces.native_explorer.reader").NativeDataset


@pytest.fixture
def dataset(tmp_path):
    group = zarr.open_group(str(tmp_path / "fields.zarr"), mode="w")
    levels = []
    for level in range(2):
        width = 2**-level
        dx = 2 * width / 8
        axis = -width + (np.arange(8) + 0.5) * dx
        xyz = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
        values = np.concatenate((xyz + level * 10, xyz + level * 10 + 100), axis=-1)
        group.create_array(
            f"level_{level}",
            data=np.stack((np.zeros_like(values), values)),
            chunks=(1, 4, 4, 4, 3),
            shards=(1, 8, 8, 8, 6),
        )
        levels.append(
            {
                "level": level,
                "shape": [8, 8, 8, 6],
                "origin": [-width] * 3,
                "spacing": [dx] * 3,
            }
        )
    manifest = {
        "schema_version": 1,
        "kind": "native-fixed-level-voxels",
        "validated": True,
        "levels": levels,
        "times": [0, 0.9],
        "complete_history": True,
        "source_frames": 2,
        "color_max": {"velocity": 20, "force": 200},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return NativeDataset(folder=tmp_path)


@pytest.mark.parametrize(
    "plane,axes", [("yz", (1, 2, 0)), ("xz", (0, 2, 1)), ("xy", (0, 1, 2))]
)
@pytest.mark.parametrize("coordinate", [-1, -0.5, -0.01, 0, 0.49, 0.5, 1])
@pytest.mark.parametrize("quantity,offset", [("velocity", 0), ("force", 100)])
def test_planes_match_independently_enumerated_cells(
    dataset, plane, axes, coordinate, quantity, offset
):
    h, v, f = axes
    axis = -1 + (np.arange(64) + 0.5) / 32
    xyz = np.empty((64, 64, 3))
    xyz[..., h], xyz[..., v] = np.meshgrid(axis, axis, indexing="ij")
    xyz[..., f] = min(coordinate, np.nextafter(2.0, 0.0) - 1.0)
    level = ((xyz >= -0.5) & (xyz < 0.5)).all(axis=-1).astype(int)
    origin = -(2.0**-level)
    dx = 2.0**-level / 4
    cell = np.floor((xyz - origin[..., None]) / dx[..., None])
    expected = (
        origin[..., None]
        + (cell + 0.5) * dx[..., None]
        + 10 * level[..., None]
        + offset
    )
    actual, actual_level = dataset.sample_plane(1, plane, coordinate, quantity, 64)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_level, level)


def test_rest_and_all_display_sizes(dataset):
    for n in (64, 128, 256, 512, 1024):
        actual, _ = dataset.sample_plane(0, "xy", 0, n=n)
        assert actual.shape == (n, n, 3) and not np.any(actual)


@pytest.mark.parametrize("frame", [-1, 2, 0.5, True, float("nan"), float("inf"), "1"])
def test_invalid_frames(dataset, frame):
    with pytest.raises(ValueError):
        dataset.sample_plane(frame, "xy", 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"plane": "zz"},
        {"coordinate": 1.001},
        {"coordinate": float("nan")},
        {"quantity": "pressure"},
        {"n": 192},
    ],
)
def test_invalid_controls(dataset, kwargs):
    with pytest.raises(ValueError):
        dataset.sample_plane(**{"frame": 1, "plane": "xy", "coordinate": 0} | kwargs)


def test_remote_requires_immutable_commit():
    with pytest.raises(ValueError, match="pinned"):
        NativeDataset(repo_id="example/data", revision="main")


def test_ui_renders_all_planes_with_actual_time(dataset):
    pytest.importorskip("gradio")
    from spaces.native_explorer.app import make_app, render_state

    *images, status = render_state(dataset, 1, 0, 0, 0, "velocity", 128)
    assert len(images) == 3 and all(image.shape == (528, 528, 3) for image in images)
    assert "t = 0.90000000" in status and "Frame 2/2" in status
    config = make_app(dataset).config
    labels = {c.get("props", {}).get("label") for c in config["components"]}
    assert {"x · YZ slice", "y · XZ slice", "z · XY slice", "t · saved frame"} <= labels
    assert any(d.get("cancels") for d in config["dependencies"])


def test_playback_yields_every_state_and_can_close(dataset):
    pytest.importorskip("gradio")
    from spaces.native_explorer.app import make_app

    app = make_app(dataset)
    animate = next(
        f.fn for f in app.fns.values() if f.fn and f.fn.__name__ == "animate"
    )

    async def verify():
        events = [event async for event in animate(0, 0, 0, 0, "velocity", 128)]
        assert [event[0] for event in events] == [0, 1]
        assert "t = 0.00000000" in events[0][-1]
        assert "t = 0.90000000" in events[1][-1]
        generator = animate(0, 0, 0, 0, "force", 128)
        assert (await anext(generator))[0] == 0
        await generator.aclose()
        with pytest.raises(StopAsyncIteration):
            await anext(generator)

    asyncio.run(verify())


def test_native_archive_matches_independent_cpp_slice_export(tmp_path):
    required = [
        os.getenv(name)
        for name in ("NS_TEST_DATASET", "NS_TEST_NATIVE", "NS_TEST_SLICE_EXPORTER")
    ]
    if not all(required):
        pytest.skip("supply the three NS_TEST_* archive paths for native integration")
    data_path, source, exporter = map(Path, required)
    data = NativeDataset(folder=data_path)
    for i, frame in enumerate(data.manifest["frames"]):
        output = tmp_path / f"slices-{i}.bin"
        subprocess.run(
            [
                str(exporter),
                f"plot={source / frame['source_frame']}",
                f"output={output}",
                "display_n=256",
            ],
            check=True,
            capture_output=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OMP_THREAD_LIMIT": "1"},
            timeout=60,
        )
        native = np.fromfile(output, dtype="<f8").reshape(3, 256, 256, 6)
        for p, plane in enumerate(("xz", "xy", "yz")):
            for quantity, part in (("velocity", slice(0, 3)), ("force", slice(3, 6))):
                actual, _ = data.sample_plane(i, plane, 0, quantity)
                np.testing.assert_array_equal(actual, native[p, ..., part])
