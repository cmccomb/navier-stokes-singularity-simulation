"""The 3D detail presentation preserves native ownership and honest motion."""

import json
from argparse import Namespace

import numpy as np
import pytest

pytest.importorskip("zarr")
pv = pytest.importorskip("pyvista")

from scripts.render_native_detail import (
    CORE_HALF,
    SEEDS,
    DetailScene,
    assemble,
    core_streamlines,
    native_core,
    overview_volume,
    quarter_cut,
    sha,
)


def core_field(n=32):
    dx = 2 * CORE_HALF / n
    x = -CORE_HALF + (np.arange(n) + 0.5) * dx
    values = np.zeros((n, n, n, 6))
    level = {
        "origin": [-CORE_HALF] * 3,
        "spacing": [dx] * 3,
        "shape": list(values.shape),
    }
    return values, level, np.meshgrid(x, x, x, indexing="ij")


def test_native_core_centers_and_component_order():
    values, level, (x, y, z) = core_field()
    values[..., :3] = np.stack((x, y, z), axis=-1)
    values[..., 3] = 99
    grid = native_core(values, level)
    np.testing.assert_allclose(grid.points, grid["velocity"])
    np.testing.assert_allclose(grid["log_force"], 2)
    assert grid.n_points == 32**3
    assert np.abs(grid.points).max() < CORE_HALF


def test_exact_rest_has_no_tracer_paths():
    values, level, _ = core_field()
    assert core_streamlines(native_core(values, level)).n_points == 0


def test_uniform_vector_gives_straight_upward_streamlines():
    values, level, _ = core_field()
    values[..., 2] = 1
    lines = core_streamlines(native_core(values, level))
    assert lines.n_cells == 2 * len(SEEDS)
    np.testing.assert_allclose(lines["axial_fraction"], 1)
    for i in range(lines.n_cells):
        points = lines.get_cell(i).points
        np.testing.assert_allclose(
            points[:, :2],
            np.broadcast_to(points[0, :2], points[:, :2].shape),
            atol=1e-7,
        )


def test_rotation_preserves_radius_with_halved_spatial_step():
    values, level, (x, y, _) = core_field()
    values[..., 0], values[..., 1] = -y, x
    grid = native_core(values, level)
    for step in (0.5, 0.25):
        lines = core_streamlines(grid, max_step=step)
        assert lines.n_cells == 2 * len(SEEDS)
        np.testing.assert_allclose(lines["axial_fraction"], 0, atol=1e-12)
        for i in range(lines.n_cells):
            points = lines.get_cell(i).points
            radius = np.linalg.norm(points[:, :2], axis=1)
            np.testing.assert_allclose(radius, radius[0], atol=2e-6)


def test_quarter_cut_removes_front_quadrant_without_inventing_caps():
    mesh = pv.Sphere(radius=0.5, theta_resolution=32, phi_resolution=32)
    cut = quarter_cut(mesh)
    centers = cut.cell_centers().points
    assert not np.any((centers[:, 0] > 1e-6) & (centers[:, 1] < -1e-6))
    assert cut.n_open_edges > 0
    assert cut.n_points > 0


def test_scene_clear_refreshes_framebuffer():
    scene = DetailScene(320, 320, 0.12, CORE_HALF)
    try:
        blank = np.asarray(scene.image()).copy()
        scene.surface(pv.Sphere(radius=0.05), "#ff0000", 1.0)
        red = np.asarray(scene.image()).copy()
        assert np.abs(red.astype(float) - blank).mean() > 1
        scene.clear()
        cleared = np.asarray(scene.image())
        np.testing.assert_allclose(cleared, blank, atol=1)
    finally:
        scene.close()


def test_volume_transfer_function_is_visible_and_cleared():
    values, level, _ = core_field()
    values[..., 3] = 9999
    grid = native_core(values, level)
    scene = DetailScene(320, 320, 0.12, CORE_HALF)
    try:
        blank = np.asarray(scene.image()).copy()
        actor = scene.volume(grid)
        assert actor.prop.GetScalarOpacity().GetValue(4) == pytest.approx(
            0.22, abs=0.01
        )
        rendered = np.asarray(scene.image()).copy()
        assert np.abs(rendered.astype(float) - blank).mean() > 1
        scene.clear()
        np.testing.assert_allclose(np.asarray(scene.image()), blank, atol=1)
    finally:
        scene.close()


def test_volume_preserves_display_coordinates_and_magnitude():
    axes = [np.linspace(-0.5, 0.5, 5)] * 3
    scalar = np.arange(125).reshape(5, 5, 5) / 20
    grid = overview_volume(axes, scalar)
    expected_points = np.column_stack(
        [v.ravel(order="F") for v in np.meshgrid(*axes, indexing="ij")]
    )
    np.testing.assert_array_equal(grid.points, expected_points)
    np.testing.assert_allclose(
        10 ** grid["log_magnitude"] - 1, scalar.ravel(order="F"), atol=1e-6
    )
    with pytest.raises(ValueError):
        overview_volume(axes, -np.ones((5, 5, 5)))


def test_cutaway_is_cropped_volume_not_contour_geometry():
    axes = [np.linspace(-0.06, 0.06, 6)] * 3
    grid = overview_volume(axes, np.full((6, 6, 6), 4.0))
    original = grid["log_magnitude"].copy()
    scene = DetailScene(320, 320, 0.12, CORE_HALF)
    try:
        blank = np.asarray(scene.image()).copy()
        actor = scene.volume(grid, "flow", cutaway=True)
        assert actor.mapper.GetCropping() == 1
        assert actor.mapper.GetCroppingRegionFlags() == ((1 << 27) - 1) ^ (1 << 13)
        assert actor.mapper.GetCroppingRegionPlanes() == pytest.approx(
            (
                0,
                0.06,
                -0.06,
                0,
                -0.06,
                0.06,
            )
        )
        assert np.abs(np.asarray(scene.image()).astype(float) - blank).mean() > 1
        np.testing.assert_array_equal(grid["log_magnitude"], original)
    finally:
        scene.close()


@pytest.mark.parametrize(
    "damage", [None, "missing", "duplicate", "camera", "image", "native"]
)
def test_worker_assembly_is_complete_and_source_bound(tmp_path, damage):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    evidence = {
        "validated": True,
        "complete_history": True,
        "source_record_sha256": "source",
        "exporter_sha256": "exporter",
        "records": [{"time": i / 10, "level_sha256": [str(i)]} for i in range(2)],
    }
    (prepared / "manifest.json").write_text(json.dumps(evidence))
    parts = []
    for i in range(2):
        folder = tmp_path / str(i)
        folder.mkdir()
        images = {}
        for q in ("flow", "force"):
            (folder / q).mkdir()
            name = f"{q}/frame-{i:04d}.png"
            (folder / name).write_bytes(b"verified image bytes")
            images[q] = {"name": name, "sha256": sha(folder / name)}
        item = {
            "index": i,
            "time": i / 10,
            "native_level_sha256": [str(i)],
            "images": images,
        }
        part = {
            "validated": True,
            "complete_history": False,
            "source_record_sha256": "source",
            "exporter_sha256": "exporter",
            "native_frame_count": 2,
            "camera": [1, 2, 3],
            "records": [item],
            "indices": [i],
            "frame_times": [i / 10],
            "media": {},
        }
        if i == 1:
            if damage == "camera":
                part["camera"] = [3, 2, 1]
            if damage == "native":
                item["native_level_sha256"] = ["wrong"]
            if damage == "image":
                (folder / images["flow"]["name"]).write_bytes(b"changed")
        (folder / "manifest.json").write_text(json.dumps(part))
        parts.append(folder)
    if damage == "missing":
        parts.pop()
    elif damage == "duplicate":
        parts.append(parts[0])
    args = Namespace(prepared=prepared, parts=parts, output=tmp_path / "assembled")
    if damage:
        with pytest.raises(ValueError):
            assemble(args)
        assert not args.output.exists()
    else:
        assemble(args)
        result = json.loads((args.output / "manifest.json").read_text())
        assert result["complete_history"] and result["indices"] == [0, 1]
        assert result["frame_times"] == [0, 0.1]
