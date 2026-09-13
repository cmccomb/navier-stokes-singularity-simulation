"""Scientific sampling, view refresh and movie coverage regressions."""

import hashlib
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("zarr")

from scripts.render_native_3d import (
    Scene,
    audited_fields,
    composite_axes,
    contours,
    durations,
    encode_frames,
    fit_camera_scales,
    interpolate_scalar,
    sample_speed,
)


def level(n, origin, dx):
    return {"origin": [origin] * 3, "spacing": [dx] * 3, "shape": [n, n, n, 6]}


def test_axes_keep_native_core_spacing_and_monotonicity():
    levels = [level(64, -1 / 2**i, 2 / 64 / 2**i) for i in range(5)]
    axes = composite_axes(levels)
    for axis in axes:
        assert len(axis) == 192
        assert np.all(np.diff(axis) > 0)
        np.testing.assert_allclose(axis, -axis[::-1])
        core = axis[np.abs(axis) < 0.0625]
        assert len(core) == 64
        np.testing.assert_array_equal(np.diff(core), np.full(63, 2 / 1024))


def test_trilinear_reconstruction_exact_on_affine_field():
    x, y, z = np.indices((4, 5, 6))
    native = 3 * x + 2 * y + z
    coords = [np.array([0.25, 1.5, 2.8]), np.array([0.6, 2.8]), np.array([0.9, 4.3])]
    sampled = interpolate_scalar(native, coords)
    expected = (
        3 * coords[0][:, None, None]
        + 2 * coords[1][None, :, None]
        + coords[2][None, None, :]
    )
    np.testing.assert_allclose(sampled, expected)
    assert sampled.min() >= native.min()
    assert sampled.max() <= native.max()


def test_finest_level_owns_core_and_constant_rest_is_exact():
    levels = [level(4, -1, 0.5), level(4, -0.5, 0.25)]
    axes = composite_axes(levels)
    outer, inner = np.zeros((4, 4, 4, 6)), np.zeros((4, 4, 4, 6))
    assert not sample_speed([outer, inner], levels, axes).any()
    outer[..., 0], inner[..., 0] = 2, 7
    result = sample_speed([outer, inner], levels, axes)
    x, y, z = np.meshgrid(*axes, indexing="ij")
    inside = (abs(x) < 0.5) & (abs(y) < 0.5) & (abs(z) < 0.5)
    assert np.all(result[inside] == 7)
    assert np.all(result[~inside] == 2)


def test_source_hash_rejects_missing_zarr_zero_fill():
    original = np.ones((2, 2, 2, 6), dtype="<f8")
    expected = hashlib.sha256(original.tobytes()).hexdigest()
    data = SimpleNamespace(
        arrays=[np.zeros((1, 2, 2, 2, 6))],
        manifest={"frames": [{"level_sha256": [expected]}]},
    )
    with pytest.raises(ValueError, match="Native field hash"):
        audited_fields(data, 0)
    data.arrays = [original[None]]
    np.testing.assert_array_equal(audited_fields(data, 0)[0], original)


def test_rest_produces_no_surfaces():
    pytest.importorskip("pyvista")
    axes = [np.linspace(-1, 1, 8)] * 3
    assert all(m.n_points == 0 for m in contours(axes, np.zeros((8, 8, 8))))


def test_contours_preserve_nonuniform_xyz_coordinates():
    pytest.importorskip("pyvista")
    axes = [
        np.array([-1, -0.6, -0.1, 0.04, 0.3, 1.0]),
        np.linspace(-1, 1, 7),
        np.linspace(-1, 1, 9),
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    meshes = contours(axes, x + 2 * y + 3 * z)
    for value, mesh in zip((0.15, 1.5, 4.0), meshes):
        assert mesh.n_points
        np.testing.assert_allclose(
            mesh.points @ np.array([1, 2, 3]), value, rtol=0, atol=1e-6
        )


def test_camera_refresh_and_core_visibility():
    pv = pytest.importorskip("pyvista")
    scene = Scene(800, 600)
    try:
        mesh = pv.Sphere(radius=0.06, center=(0.09, 0.03, 0.08))
        mesh.scale((1, 1, 2), inplace=True)
        scene.set_surfaces([pv.PolyData(), mesh, pv.PolyData()])
        side, top = np.asarray(scene.image("side")), np.asarray(scene.image("top"))
        assert np.mean(np.abs(side.astype(float) - top)) > 0.1
        scene.set_surfaces([mesh, pv.PolyData(), pv.PolyData()])
        scene.image("core")
        assert not scene.actors[0][0].GetVisibility()
        scene.image("isometric")
        assert scene.actors[0][0].GetVisibility()
    finally:
        scene.close()


def test_playback_exact_holds():
    assert len(durations(280)) == 280
    assert sum(durations(280)) == 57100
    assert durations(3) == [1000, 200, 500]
    with pytest.raises(ValueError):
        durations(1)


def test_preflight_fits_early_pulses_without_per_frame_zoom():
    pv = pytest.importorskip("pyvista")
    # A broad early low-speed shell needs a wide overview, but must not
    # determine the scale of the high-speed core view where it is hidden.
    broad = pv.Sphere(radius=0.9)
    core = pv.Sphere(radius=0.08)
    empty = pv.PolyData()
    scales = {"isometric": 1.05, "side": 1.05, "top": 1.05, "core": 0.36}
    fit_camera_scales([broad, core, empty], scales)
    assert scales["isometric"] > 1.05
    assert scales["isometric"] == scales["side"] == scales["top"]
    assert scales["core"] == 0.36
    frozen = scales.copy()
    fit_camera_scales([core, core, empty], scales)
    assert scales == frozen


def test_encoder_retains_identical_rest_states(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    folder = tmp_path / "frames"
    folder.mkdir()
    for i, color in enumerate(["black", "black", "cyan"]):
        Image.new("RGB", (160, 120), color).save(folder / f"frame-{i:04d}.png")
    record = encode_frames(folder, tmp_path / "test", [0, 0.55, 0.995])
    assert record["saved_states"] == 3
    assert record["duration_seconds"] == 1.7
    with Image.open(tmp_path / "test.gif") as gif:
        gif.seek(2)
        assert gif.convert("RGB").getpixel((80, 60)) == (0, 255, 255)


def test_gif_reserves_white_type_despite_blue_dominance(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    folder = tmp_path / "frames"
    folder.mkdir()
    for i in range(3):
        im = Image.new("RGB", (160, 120), (20, 70 + i * 20, 110 + i * 20))
        im.putpixel((80, 60), (255, 255, 255))
        im.putpixel((81, 60), (233, 241, 245))
        im.save(folder / f"frame-{i:04d}.png")
    encode_frames(folder, tmp_path / "test", [0, 0.5, 0.995])
    with Image.open(tmp_path / "test.gif") as gif:
        for i in range(3):
            gif.seek(i)
            assert gif.convert("RGB").getpixel((80, 60)) == (255, 255, 255)
            assert gif.convert("RGB").getpixel((81, 60)) == (233, 241, 245)
