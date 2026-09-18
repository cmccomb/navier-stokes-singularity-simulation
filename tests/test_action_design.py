import numpy as np
import pytest

from scripts.design_action_refinement import (
    angular_cube_fraction,
    bands_from_masks,
    geometric_cells,
    runs,
    selected_bins,
)


def test_runs_keep_disjoint_annuli():
    assert runs([False, True, True, False, True]) == [(1, 3), (4, 5)]


def test_selection_requires_activity_and_variation_in_same_quantity():
    f = {
        name: np.zeros((2, 2))
        for name in ("speed", "force", "velocity_gradient", "force_gradient")
    }
    f["native_spacing"] = np.ones((2, 2)) / 8
    f["speed"][0] = [1, 0.01]
    f["velocity_gradient"][0] = [1, 100]
    assert selected_bins(f, 0.1, 0.1).tolist() == [[True, False], [False, False]]


def test_band_buffers_include_ancestors_without_symmetrizing_z():
    masks = [np.zeros((2, 2), dtype=bool) for _ in range(2)]
    masks[1][1, 0] = True
    r, z = np.array([0, 0.1, 0.2]), np.array([-0.3, -0.2, -0.1])
    bands = bands_from_masks(masks, r, z, 128)
    assert {b[0] for b in bands} == {0, 1}
    for _, lo, hi, bottom, top in bands:
        assert lo < 0.1 and hi > 0.2 and bottom < -0.3 and -0.2 < top < 0


def test_exact_baseline_and_revolved_cylinder_volume():
    assert geometric_cells([], 128, [0.5, 0.25, 0.125, 0.0625], 128) == [128**3] * 5
    assert angular_cube_fraction(np.array([0, 0.25, 0.5, 1]), 0.5).tolist() == [
        1,
        1,
        1,
        0,
    ]
    bands = [(0, 0.75, 0.875, -0.25, 0.25)]
    cells = geometric_cells(bands, 128, [0.5], 1024)
    added = np.pi * (0.875**2 - 0.75**2) * 0.5 / (1 / 128) ** 3
    assert cells[1] - 128**3 == pytest.approx(added, rel=1e-12)
