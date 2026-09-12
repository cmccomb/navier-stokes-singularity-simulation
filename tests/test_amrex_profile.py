from dataclasses import replace

import numpy as np
import pytest

from navier_stokes_sim.amrex_profile import export_profile
from navier_stokes_sim.config import SimulationConfig
from navier_stokes_sim.profile import (
    _paper_vector_potentials,
    cell_centers,
    discrete_divergence,
    target_velocity,
)


def test_localized_revision_removes_only_the_missing_core_cutoff():
    old = SimulationConfig(resolution=64, frames=2, t_end=0.8)
    new = replace(old, paper_profile_revision="appendix-b-axis-v2-localized")
    new.validate()
    x, y, z = cell_centers(new)
    radius = np.hypot(x, y)
    outside = (radius > new.localization_outer) | (np.abs(z) > new.localization_outer)
    a_old, _, _ = _paper_vector_potentials(0.7, old)
    a_new, _, _ = _paper_vector_potentials(0.7, new)
    assert np.max(np.abs(a_old[2][outside])) > 1e-7
    assert np.max(np.abs(np.stack(a_new, axis=-1)[outside])) == 0
    u_old, u_new = target_velocity(0.7, old), target_velocity(0.7, new)
    inner = (radius < new.localization_inner - 2 * new.dx) & (
        np.abs(z) < new.localization_inner - 2 * new.dx
    )
    np.testing.assert_array_equal(u_old[inner], u_new[inner])
    assert np.max(np.abs(discrete_divergence(u_new, new.dx))) < 1e-11


def test_profile_export_is_versioned_and_does_not_overwrite(tmp_path):
    cfg = SimulationConfig(
        resolution=16, frames=2, paper_profile_revision="appendix-b-axis-v2-localized"
    )
    path = tmp_path / "profile.tbl"
    record = export_profile(cfg, path)
    assert record["parameters"]["fully_localized"] == 1
    assert set(record["tables"]) == {"radial_u", "radial_phi", "axis_phi", "exterior"}
    assert path.read_text().startswith("NS_PAPER_TABLES_V1\n")
    with pytest.raises(FileExistsError):
        export_profile(cfg, path)
    with pytest.raises(ValueError, match="cubic"):
        export_profile(
            replace(cfg, profile_interpolation="linear"), tmp_path / "bad.tbl"
        )
