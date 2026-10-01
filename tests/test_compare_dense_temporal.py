"""Validate the local paired timestep gate and source identity checks."""

import pytest

from scripts.compare_dense_temporal import (
    LEGACY_BASELINE_RUNNER_SHA256,
    center_gates,
    compatible,
    temporal_frame,
)


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


def test_half_to_quarter_controls_and_exact_checkpoint_parent_are_audited():
    half, quarter = receipt(0.5), receipt(0.25)
    quarter["parent_record_sha256"] = "later-parent-snapshot"
    with pytest.raises(ValueError, match="parent_record"):
        compatible(half, quarter, baseline_factor=0.5)
    audit = compatible(
        half,
        quarter,
        baseline_factor=0.5,
        allow_equivalent_checkpoint_parent=True,
    )
    assert audit["checkpoint_parent_identity"] == "exact_checkpoint_state"
    quarter["checkpoint_files"]["Header"]["sha256"] = "different-state"
    with pytest.raises(ValueError, match="checkpoint_files"):
        compatible(
            half,
            quarter,
            baseline_factor=0.5,
            allow_equivalent_checkpoint_parent=True,
        )


def test_pinned_legacy_baseline_controls_are_derived_from_parent():
    baseline, half = receipt(1), receipt(0.5)
    for key in (
        "time_factor",
        "max_dt",
        "integration_phase_step",
        "force_definition_unchanged",
    ):
        baseline.pop(key)
    baseline["runner_sha256"] = LEGACY_BASELINE_RUNNER_SHA256
    baseline["command"] = ["solver", "ns.max_dt=0.00025", "ns.epsilon_tau_ratio=0"]
    parent = {
        "binary_sha256": "binary",
        "profile_manifest": {
            "sha256": "profile",
            "parameters": {"forcing_phase_step": 0.0375},
        },
        "parameters": {"max_dt": 0.00025, "integration_phase_step": 0},
    }
    audit = compatible(baseline, half, parent)
    assert audit["baseline"]["inferred_from_pinned_legacy_runner"] is True
    assert audit["baseline"]["integration_phase_step"] == 0.0375
    baseline["runner_sha256"] = "unknown"
    with pytest.raises(ValueError, match="unrecognized"):
        compatible(baseline, half, parent)


def test_native_checker_velocity_only_row_preserves_force_readback():
    row = {
        "compared": True,
        "report_difference": True,
        "reference_resolution_ratio": 1,
        "time": 0.615,
        "force_linf_error": 0,
        "velocity_l2_difference": 0.001,
        "reference_velocity_l2": 0.5,
    }
    baseline, half = (
        {"step": 300, "force_linf_error": 0},
        {"step": 350, "force_linf_error": 0},
    )
    result = temporal_frame(row, baseline, half, 0.615)
    assert result["relative_velocity_l2"] == pytest.approx(0.002)
    assert "relative_force_l2" not in result
    with pytest.raises(ValueError, match="force readback"):
        temporal_frame(row, baseline, {**half, "force_linf_error": 1}, 0.615)


def test_local_gate_uses_quarter_of_measured_spatial_gap():
    times = [0.615, 0.622, 0.629]
    temporal = [
        {"time": t, "relative_velocity_l2": gap}
        for t, gap in zip(times, (0.003, 0.004, 0.006), strict=True)
    ]
    spatial = {
        "checkpoints": [
            {
                "time": t,
                "screen": "pass",
                "comparison": {
                    "frames": [{"velocity": {"relative_l2_difference": 0.02}}]
                },
            }
            for t in times
        ]
    }
    gates = center_gates(temporal, times, spatial)
    assert [gate["passed"] for gate in gates] == [True, True, False]
    assert all(gate["quarter_spatial_limit"] == pytest.approx(0.005) for gate in gates)
