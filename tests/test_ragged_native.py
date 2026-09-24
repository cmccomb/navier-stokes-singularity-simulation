"""Ragged display ownership must preserve holes, seams and native diagnostics."""

import numpy as np
import pytest

pytest.importorskip("zarr")
from scripts import ragged_native as rn


def block(level, low, shape):
    dx = 0.5 / 2**level
    return {"level": level, "index_lo": low, "shape": [*shape, 6],
            "spacing": [dx] * 3, "origin": (-1 + np.array(low) * dx).tolist()}


def geometry():
    return [block(0, [0, 0, 0], [2, 4, 4]),
            block(0, [2, 0, 0], [2, 4, 4]),
            block(1, [0, 0, 0], [2, 2, 2]),
            block(1, [2, 2, 2], [2, 4, 4]),
            block(1, [4, 2, 2], [2, 4, 4])]


def test_holes_keep_coarse_ownership_and_native_centers_keep_native_values():
    blocks = geometry()
    fields = [np.zeros(b["shape"]) for b in blocks]
    for f, b in zip(fields, blocks, strict=True):
        f[..., 0] = 7 if b["level"] == 0 else 3
    levels = rn.bounding_levels(blocks)
    axes = [np.array([-0.875, -0.625, -0.375, -0.125, 0.125, 0.375, 0.625])] * 3
    result = rn.sample_speed(fields, blocks, levels, axes)
    assert result[0, 0, 0] == 3  # Detached outer refined box.
    assert result[3, 3, 3] == 3  # A native center in the joined core boxes.
    assert result[0, 4, 4] == 7  # Hole inside the refined bounding rectangle.
    assert result[-1, -1, -1] == 7
    assert np.isfinite(result).all()


def test_same_level_box_boundary_does_not_clamp_the_reconstruction():
    blocks = geometry()
    fields = []
    for b in blocks:
        xyz = np.meshgrid(*[o + (np.arange(n) + 0.5) * d for o, n, d in zip(
            b["origin"], b["shape"][:3], b["spacing"], strict=True
        )], indexing="ij")
        values = np.zeros(b["shape"])
        values[..., 0] = 2 + 0.2 * xyz[0] + 0.1 * xyz[1]
        fields.append(values)
    axes = [np.array([-0.02, 0, 0.02]), np.array([-0.125, 0.125]), np.array([0.125])]
    result = rn.sample_speed(fields, blocks, rn.bounding_levels(blocks), axes)
    expected = 2 + 0.2 * axes[0][:, None, None] + 0.1 * axes[1][None, :, None]
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-14)


def test_active_native_diagnostics_exclude_covered_cells():
    blocks = geometry()
    fields = [np.zeros(b["shape"]) for b in blocks]
    masks = rn.diagnostic_masks(blocks)
    for f, mask in zip(fields, masks, strict=True):
        f[..., 0] = np.where(mask, 2, 100)
    assert rn.diagnostics(fields, blocks, masks) == {
        "volume": 8, "energy": 16, "peak_speed": 2}
    with pytest.raises(ValueError, match="nonzero rest"):
        rn.diagnostics(fields, blocks, masks, rest=True)


def test_irregular_sections_preserve_coarse_holes_and_axis_orientation():
    from scripts.render_ragged_mesh import plane_edges

    # A detached fine cube intersects x=0 but not y=0 or z=0.
    blocks = [block(0, [0, 0, 0], [4, 4, 4]),
              block(1, [4, 0, 0], [2, 4, 4])]
    masks = rn.diagnostic_masks(blocks)
    sections = [plane_edges(blocks, masks, fixed) for fixed in range(3)]
    assert any(edge[0] == 1 for edge in sections[0])
    assert all(edge[0] == 0 for plane in sections[1:] for edge in plane)
    # The fine cube's interior removes the coarse face there.
    assert not any(level == 0 and a == [0, -0.5, -1] and b == [0, -0.5, 1]
                   for level, *points in sections[0]
                   for a, b in [(points[:3], points[3:])])
    for fixed, edges in enumerate(sections):
        for _level, *points in edges:
            assert points[fixed] == points[fixed + 3] == 0
