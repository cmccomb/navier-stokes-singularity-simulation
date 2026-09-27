"""Analytic vector fields exercise the native derivative and masking rules."""

import numpy as np
import pytest

from scripts.paper_precursor_diagnostics import active_mask, interior_derivatives


def coordinates(n=8, spacing=0.25):
    axis = -1 + (np.arange(n) + 0.5) * spacing
    return np.meshgrid(axis, axis, axis, indexing="ij")


def test_solid_rotation_has_constant_curl_and_zero_divergence():
    x, y, z = coordinates()
    velocity = np.stack((-y, x, np.zeros_like(z)), axis=-1)
    result = interior_derivatives(velocity, np.ones((8, 8, 8), dtype=bool), [0.25] * 3)
    assert result["cells"] == 6**3
    assert result["peak_vorticity"] == pytest.approx(2)
    assert result["vorticity_squared_sum"] == pytest.approx(4 * 6**3)
    assert result["divergence_linf"] == pytest.approx(0)


def test_expansion_has_constant_divergence_and_excludes_interface_neighbors():
    x, y, z = coordinates()
    velocity = np.stack((x, y, z), axis=-1)
    active = np.ones((8, 8, 8), dtype=bool)
    active[3:5, 3:5, 3:5] = False
    result = interior_derivatives(velocity, active, [0.25] * 3)
    assert result["cells"] < 6**3
    assert result["cells"] > 0
    assert result["peak_vorticity"] == pytest.approx(0)
    assert result["divergence_linf"] == pytest.approx(3)
    assert result["divergence_squared_sum"] == pytest.approx(9 * result["cells"])


def test_nested_cube_mask_preserves_full_domain_volume():
    levels = [
        {"shape": [8, 8, 8, 6], "origin": [-1.0] * 3, "spacing": [0.25] * 3},
        {"shape": [8, 8, 8, 6], "origin": [-0.5] * 3, "spacing": [0.125] * 3},
    ]
    coarse, fine = (active_mask(levels, i) for i in range(2))
    assert coarse.sum() == 8**3 - 4**3
    assert fine.sum() == 8**3
    assert coarse.sum() * 0.25**3 + fine.sum() * 0.125**3 == pytest.approx(8)
