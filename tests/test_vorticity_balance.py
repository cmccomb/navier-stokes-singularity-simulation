"""Validate pressure-free balance against a semi-discrete decaying shear."""

import numpy as np

from scripts.vorticity_balance import balance, eroded_active


def test_diffusing_shear_has_small_five_point_residual():
    n, h, viscosity = 16, 0.1, 0.02
    wave = 2 * np.pi / (n * h)
    eigenvalue = 4 * np.sin(wave * h / 2) ** 2 / h**2
    x = (np.arange(n) + 0.5) * h
    times = [-0.021, -0.01, 0, 0.011, 0.022]
    fields = []
    for time in times:
        field = np.zeros((n, n, n, 6))
        field[..., 1] = np.exp(-viscosity * eigenvalue * time) * np.sin(wave * x)[:, None, None]
        fields.append(field)
    result = balance(fields, times, h, viscosity)
    assert np.max(np.abs(result["advection_curl"])) == 0
    assert np.sqrt(np.mean(result["residual_five"] ** 2)) < 1e-9
    assert np.sqrt(np.mean(result["residual_three"] ** 2)) > 1e-7


def test_active_stencil_excludes_patch_edge_and_finer_overlay():
    active = np.ones((9, 9, 9), bool)
    active[4, 4, 4] = False
    valid = eroded_active(active)
    assert valid.shape == (5, 5, 5)
    assert not valid[2, 2, 2]
    assert not valid[0, 2, 2]
    assert valid[0, 0, 0]
