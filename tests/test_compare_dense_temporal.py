"""Validate the local paired timestep gate and source identity checks."""

import pytest

from scripts.compare_dense_temporal import center_gates, compatible


def receipt(factor):
    return {
        "status": "completed",
        "validated": True,
        "parent_record_sha256": "parent",
        "source_binary_sha256": "binary",
        "source_profile_sha256": "profile",
        "checkpoint_step": 250,
        "checkpoint_time": 0.6,
        "checkpoint_files": {"Header": {"sha256": "checkpoint"}},
        "centers": [0.615, 0.622, 0.629],
        "spacing": 0.0005,
        "planned_times": [0.615, 0.622, 0.629],
        "native_frames": [{"time": t} for t in (0.615, 0.622, 0.629)],
        "time_factor": factor,
        "max_dt": 0.00025 * factor,
        "integration_phase_step": 0.0375 * factor,
        "force_definition_unchanged": True,
    }


def test_same_checkpoint_and_halved_integration_controls_required():
    baseline, half = receipt(1), receipt(0.5)
    compatible(baseline, half)
    half["native_frames"].pop()
    with pytest.raises(ValueError, match="frame count"):
        compatible(baseline, half)


def test_local_gate_uses_quarter_of_measured_spatial_gap():
    times = [0.615, 0.622, 0.629]
    temporal = [{"time": t, "relative_velocity_l2": gap} for t, gap in zip(times, (0.003, 0.004, 0.006), strict=True)]
    spatial = {"checkpoints": [{"time": t, "screen": "pass", "comparison": {"frames": [{"velocity": {"relative_l2_difference": 0.02}}]}} for t in times]}
    gates = center_gates(temporal, times, spatial)
    assert [gate["passed"] for gate in gates] == [True, True, False]
    assert all(gate["quarter_spatial_limit"] == pytest.approx(0.005) for gate in gates)
