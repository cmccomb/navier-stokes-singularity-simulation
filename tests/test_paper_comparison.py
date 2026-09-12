from copy import deepcopy

import pytest

from scripts.paper_comparison import compatible, frame_headers, matched_frames


def test_header_discovery_ignores_existing_audit_logs(tmp_path):
    plot = tmp_path / "plt00000"
    plot.mkdir()
    (plot / "Header").write_text(
        "HyperCLaw-V1.1\n6\nvelx\nvely\nvelz\nforcing_x\nforcing_y\nforcing_z\n3\n0\n3\n"
    )
    (tmp_path / "plt00000-read.log").write_text("previous verification")
    assert frame_headers(tmp_path) == [{"path": "plt00000", "time": 0}]


def test_missing_events_still_fail_and_duplicates_are_preserved():
    frames = [{"path": "plt1", "time": 0.6 - 3e-15}, {"path": "plt2", "time": 0.6}]
    groups = matched_frames(frames, [0.6])
    assert groups == [[frames[1], frames[0]]]
    assert len(frames) == 2
    with pytest.raises(ValueError, match="missing requested"):
        matched_frames(frames, [0.625])


def test_comparison_rejects_uncontrolled_changes():
    a = {
        "adapter_sha256": "a",
        "binary_sha256": "b",
        "inputs_sha256": "c",
        "parameters": {
            "base_n": 32,
            "max_dt": 0.00025,
            "end": 0.85,
            "box": 32,
            "widths": [0.5],
        },
        "profile_manifest": {
            "parameters": {"forcing_phase_step": 0.0375, "viscosity": 0.01}
        },
    }
    b = deepcopy(a)
    b["parameters"]["base_n"] = 64
    b["parameters"]["end"] = 0.985
    compatible(a, b, "spatial")
    b["profile_manifest"]["parameters"]["viscosity"] = 0.02
    with pytest.raises(ValueError, match="uncontrolled"):
        compatible(a, b, "spatial")
    b = deepcopy(a)
    b["parameters"]["max_dt"] /= 2
    b["profile_manifest"]["parameters"]["forcing_phase_step"] /= 2
    compatible(a, b, "temporal")
    b["adapter_sha256"] = "d"
    with pytest.raises(ValueError, match="adapter"):
        compatible(a, b, "temporal")
