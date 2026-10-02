"""Verify native block precursor measurements on simple vector fields."""

import json

import numpy as np
import pytest

from scripts.outer_precursor_diagnostics import measure_raw

BLOCK = {"level": 0, "shape": [8, 8, 8, 6], "spacing": [0.25] * 3, "index_lo": [0, 0, 0], "origin": [-1] * 3, "offset_bytes": 0}


def test_constant_velocity_and_force_cover_full_active_domain(tmp_path):
    field = np.zeros((8, 8, 8, 6), dtype="<f8")
    field[..., 0] = 1
    field[..., 4] = 2
    raw = tmp_path / "constant.bin"
    raw.write_bytes(field.tobytes())
    result = measure_raw(raw, [BLOCK], {"peak_speed": 1, "energy": 4}, 0.25)
    assert result["kinetic_energy"] == pytest.approx(4)
    assert result["force_l2"] == pytest.approx(2)
    assert result["half_peak_support_volume"] == pytest.approx(8)
    assert result["peak_vorticity"] == pytest.approx(0)
    assert result["divergence_rms"] == pytest.approx(0)
    assert result["derivative_interior_volume"] == pytest.approx(6**3 * 0.25**3)
    assert result["finest_core_half_peak_equivalent_radius"] is None
    assert json.loads(json.dumps(result))["peak_in_finest_core"] is False


def test_solid_rotation_has_vorticity_two_and_zero_divergence(tmp_path):
    field = np.zeros((8, 8, 8, 6), dtype="<f8")
    x = -1 + (np.arange(8) + 0.5) * 0.25
    field[..., 0] = -x[None, :, None]
    field[..., 1] = x[:, None, None]
    speed_sq = np.square(field[..., :3]).sum(axis=-1)
    raw = tmp_path / "rotation.bin"
    raw.write_bytes(field.tobytes())
    result = measure_raw(raw, [BLOCK], {"peak_speed": float(np.sqrt(speed_sq.max())), "energy": float(speed_sq.sum() * 0.25**3 / 2)}, 0.25)
    assert result["peak_vorticity"] == pytest.approx(2)
    assert result["divergence_rms"] == pytest.approx(0)
