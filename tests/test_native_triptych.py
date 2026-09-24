"""View order, native ownership and full-history triptych contracts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr")
from scripts.render_native_3d import composite_axes, sample_speed
from scripts.render_native_triptych import ORDER, center_plane


def test_xy_xz_are_actual_center_slices_in_order():
    assert ORDER == ("xy", "xz", "isometric")
    x, y, z = np.indices((4, 4, 4))
    values = np.repeat((x + 10 * y + 100 * z)[..., None], 6, axis=-1)
    levels = [{"origin": [-1] * 3, "spacing": [0.5] * 3, "shape": [4, 4, 4, 6]}]
    np.testing.assert_array_equal(
        center_plane([values], levels, "xy", 4), values[:, :, 2]
    )
    np.testing.assert_array_equal(
        center_plane([values], levels, "xz", 4), values[:, 2, :]
    )


def test_force_is_selected_from_force_components():
    levels = [{"origin": [-1] * 3, "spacing": [0.5] * 3, "shape": [4, 4, 4, 6]}]
    values = np.zeros((4, 4, 4, 6))
    values[..., 0] = 7
    values[..., 3:6] = (3, 4, 0)
    axes = composite_axes(levels)
    assert np.all(sample_speed([values], levels, axes, "force") == 5)
    assert np.all(sample_speed([values], levels, axes) == 7)
    with pytest.raises(ValueError, match="Unknown vector"):
        sample_speed([values], levels, axes, "unknown")


def test_fine_cells_replace_covered_coarse_cells_in_both_planes():
    levels = [
        {"origin": [-h] * 3, "spacing": [h / 2] * 3, "shape": [4, 4, 4, 6]}
        for h in (1, 0.5)
    ]
    values = [np.full((4, 4, 4, 6), v) for v in (1, 2)]
    for plane in ORDER[:2]:
        result = center_plane(values, levels, plane, 8)
        assert np.all(result[2:6, 2:6] == 2)
        assert np.all(result[0] == 1)


def test_published_triptychs_cover_exact_best_history():
    site = Path(__file__).resolve().parents[1] / "site"
    path = site / "data/three-view.json"
    if not path.exists():
        pytest.skip("Media has not yet been generated")
    record = json.loads(path.read_text())
    best = json.loads((site / "data/best.json").read_text())
    assert record["view_order"] == list(ORDER)
    assert record["complete_history"] and record["validated"]
    assert record["frame_times"] == best["frame_times"]
    assert record["saved_frames"] == best["saved_frames"]
    assert record["source_record_sha256"] == best["source_record_sha256"]
    for name in ("flow", "force"):
        assert record["media"][name]["saved_states"] == best["saved_frames"]
        from PIL import Image

        assets = record["media"][name]["files"]
        for asset in assets.values():
            with (site / asset["path"]).open("rb") as stream:
                assert (
                    hashlib.file_digest(stream, "sha256").hexdigest() == asset["sha256"]
                )
        with Image.open(site / assets["gif"]["path"]) as gif:
            assert gif.n_frames == 280 and gif.size == (1600, 640)
            for i, hold in enumerate(best["playback"]["source_frame_duration_ms"]):
                gif.seek(i)
                assert gif.info["duration"] == hold


def test_colorbar_label_does_not_overlap_playback_caption():
    from scripts.render_native_triptych import Triptych

    figure = Triptych("velocity", 7.4, (0.15, 1.5, 4))
    try:
        figure.fig.canvas.draw()
        renderer = figure.fig.canvas.get_renderer()
        footer = figure.fig.texts[-1].get_window_extent(renderer)
        colorbar = figure.fig.axes[-1]
        for label in [colorbar.xaxis.label, *colorbar.get_xticklabels()]:
            assert not footer.overlaps(label.get_window_extent(renderer))
    finally:
        figure.close()


def test_grid_has_three_ordered_views_and_active_edges_only():
    from scripts.render_grid_views import hierarchy, plane_edges, render, render_panel

    site = Path(__file__).resolve().parents[1] / "site"
    record = json.loads((site / "data/best.json").read_text())
    # The legacy rectangular renderer remains available for rectangular inputs.
    record = {**record, "stored_cells": 10485760, "active_cells": 9437184}
    levels = hierarchy(record)
    assert levels[-1]["spacing"] == 2 / 2048
    for level, a, b in plane_edges(levels):
        if level + 1 < len(levels):
            middle = (np.array(a) + b) / 2
            assert not np.all(abs(middle) < levels[level + 1]["half"])
    svg = render(record)
    assert svg.index("x–y slice") < svg.index("x–z slice") < svg.index("Isometric ·")
    for view in ("xy", "isometric"):
        assert "<svg" in render_panel(record, view)


def test_current_mesh_panels_render_the_published_irregular_geometry():
    from scripts.render_ragged_mesh import render_panel

    site = Path(__file__).resolve().parents[1] / "site"
    mesh = json.loads((site / "data/native-mesh.json").read_text())
    for view in ("xy", "isometric"):
        assert render_panel(mesh, view) == (site / f"media/mesh-{view}.svg").read_text()
