"""Mesh-design summaries retain 3D peaks and composite native ownership."""

import numpy as np
import pytest

pytest.importorskip("zarr")

from scripts.audit_native_action import design_edges, frame_maps, gradient_norm


def field(n, half, velocity):
    dx = 2 * half / n
    axis = -half + (np.arange(n) + 0.5) * dx
    xyz = np.meshgrid(axis, axis, axis, indexing="ij")
    values = np.zeros((n, n, n, 6))
    values[..., :3] = velocity(*xyz)
    return values, {
        "origin": [-half] * 3,
        "spacing": [dx] * 3,
        "shape": list(values.shape),
    }


def test_gradient_recovers_all_cartesian_components():
    values, level = field(
        16, 1, lambda x, y, z: np.stack((2 * x + 3 * y, 4 * z, 5 * x), axis=-1)
    )
    np.testing.assert_allclose(
        gradient_norm(values[..., :3], level["spacing"]), np.sqrt(54), atol=1e-12
    )


def test_revolved_map_uses_maximum_not_azimuthal_average():
    values, level = field(16, 1, lambda x, y, z: np.zeros((*x.shape, 3)))
    values[10, 8, 8, 0] = 7
    values[8, 10, 8, 0] = 3
    maps, sums, volume, _ = frame_maps([values], [level], *design_edges(0.125))
    assert maps["speed"].max() == 7
    assert sums["speed"].sum() == pytest.approx((49 + 9) * 0.125**3)
    assert volume.sum() == 8


def test_composite_integrals_ignore_covered_coarse_cells():
    coarse, a = field(16, 1, lambda x, y, z: np.ones((*x.shape, 3)))
    fine, b = field(16, 0.5, lambda x, y, z: np.full((*x.shape, 3), 2))
    maps, sums, volume, _ = frame_maps([coarse, fine], [a, b], *design_edges(0.0625))
    assert volume.sum() == 8
    assert sums["speed"].sum() == pytest.approx(7 * 3 + 1 * 12)
    assert maps["speed"].max() == pytest.approx(np.sqrt(12))


def test_exact_rest_has_zero_activity():
    values, level = field(8, 1, lambda x, y, z: np.zeros((*x.shape, 3)))
    maps, sums, volume, _ = frame_maps([values], [level], *design_edges(0.25))
    assert volume.sum() == 8
    assert all(not a.any() for a in [*maps.values(), *sums.values()])
