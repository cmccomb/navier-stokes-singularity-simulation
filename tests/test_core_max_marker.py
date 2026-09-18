"""Velocity-only maximum markers use actual native cells, not visual peaks."""

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("zarr")
pytest.importorskip("pyvista")
from scripts.mark_native_core_max import annotate, core_maximum, project


def field():
    return np.zeros((8, 8, 8, 6)), {"origin": [-0.0625] * 3, "spacing": [0.015625] * 3}


def test_maximum_uses_velocity_not_force_and_native_centers():
    values, level = field()
    values[2, 3, 5, :3] = (3, 4, 0)
    values[6, 1, 2, 3:] = 1000000
    result = core_maximum(values, level)
    assert result["value"] == 5
    assert result["cell_index"] == [2, 3, 5]
    np.testing.assert_allclose(
        result["xyz"],
        np.array(level["origin"]) + (np.array([2, 3, 5]) + 0.5) * level["spacing"],
    )


def test_rest_has_no_unique_marker_and_changes_no_pixels():
    values, level = field()
    maximum = core_maximum(values, level)
    assert maximum["xyz"] is None and maximum["cell_index"] is None
    image = Image.new("RGB", (2400, 1440), "#07111f")
    np.testing.assert_array_equal(annotate(image, maximum), image)


def test_exact_ties_are_deterministic_without_tracking():
    values, level = field()
    values[2, 3, 5, 1] = 5
    values[4, 1, 2, 2] = 5
    assert core_maximum(values, level)["cell_index"] == [2, 3, 5]
    values[4, 1, 2, 2] += 1e-10
    assert core_maximum(values, level)["cell_index"] == [4, 1, 2]


def test_projection_is_core_panel_center_and_marker_stays_in_core_panel():
    np.testing.assert_allclose(project([0, 0, 0]), [1880, 735])
    image = Image.new("RGB", (2400, 1440), "#07111f")
    for point in ([0, 0, 0], [0.062, -0.062, 0.062], [-0.062, 0.062, -0.062]):
        marked = np.asarray(annotate(image, {"xyz": point, "value": 7.415}))
        changed = np.any(marked != np.asarray(image), axis=-1)
        y, x = np.where(changed)
        assert len(x) > 20
        assert x.min() >= 1400 and x.max() < 2400
        assert y.min() > 230 and y.max() < 1275
