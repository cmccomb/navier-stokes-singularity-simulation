"""Check that a failed checkpoint breaks the prospective spatial sequence."""

import pytest

from scripts.assess_outer_band_protocol import assess


def test_three_consecutive_passes_only_after_two_failures():
    protocol = {
        "kind": "prospective-outer-band-spatial-checkpoints",
        "source_binary_sha256": "binary",
        "source_profile_sha256": "profile",
        "mesh_pair_base_n": [64, 128],
        "checkpoints": [{"frame_index": i, "time": float(i)} for i in range(6)],
        "spatial_screen": {"velocity_relative_l2_max": 0.02, "force_relative_l2_max": 0.05},
    }
    reports = []
    for i, velocity in enumerate((0.021, 0.0201, 0.019, 0.018, 0.017)):
        reports.append((str(i), {
            "kind": "mixed-geometry-native-sensitivity",
            "validated": True,
            "source_binary_sha256": "binary",
            "source_profile_sha256": "profile",
            "mesh_pair_base_n": [64, 128],
            "source_run_state": [{}, {}],
            "frames": [{"time": float(i), "velocity": {"relative_l2_difference": velocity}, "force": {"relative_l2_difference": 0.04}}],
        }))
    result = assess(protocol, reports)
    assert [row["screen"] for row in result["checkpoints"]] == ["fail", "fail", "pass", "pass", "pass", "pending"]
    assert result["candidate_sequences"] == [{"first_frame_index": 2, "last_frame_index": 4, "count": 3}]
    assert not result["all_checkpoints_assessed"]


def test_rejects_unrelated_binary():
    protocol = {
        "kind": "prospective-outer-band-spatial-checkpoints",
        "source_binary_sha256": "expected",
        "source_profile_sha256": "profile",
        "mesh_pair_base_n": [64, 128],
        "checkpoints": [{"frame_index": 1, "time": 0.6}],
        "spatial_screen": {"velocity_relative_l2_max": 0.02, "force_relative_l2_max": 0.05},
    }
    report = {"kind": "mixed-geometry-native-sensitivity", "validated": True, "source_binary_sha256": "other"}
    with pytest.raises(ValueError, match="source_binary_sha256"):
        assess(protocol, [("digest", report)])
