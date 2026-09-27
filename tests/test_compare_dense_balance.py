"""Fieldwise time-branch comparison catches residual and force differences."""

import json

import numpy as np
import pytest

from scripts import compare_dense_balance as subject


def branch(tmp_path, name, coefficient, force_coefficient=0):
    root = tmp_path / name
    root.mkdir()
    times = [0.2 + i * 0.1 for i in (-2, -1, 0, 1, 2)]
    block = {
        "level": 0,
        "index_lo": [0, 0, 0],
        "shape": [8, 8, 8, 6],
        "spacing": [0.25] * 3,
        "origin": [-1] * 3,
        "offset_bytes": 0,
    }
    probe = {
        "status": "completed",
        "validated": True,
        "parent_record_sha256": "same",
        "source_binary_sha256": "binary",
        "source_profile_sha256": "profile",
        "checkpoint_step": 250,
        "checkpoint_time": 0.1,
        "checkpoint_files": {},
        "centers": [0.2],
        "spacing": 0.1,
        "planned_times": times,
        "native_frames": [{}] * 5,
        "time_factor": 1 if name == "baseline" else 0.5,
        "max_dt": 0.01 if name == "baseline" else 0.005,
        "integration_phase_step": 0.02 if name == "baseline" else 0.01,
        "force_definition_unchanged": True,
    }
    paths, manifests = [], []
    x = -1 + (np.arange(8) + 0.5) * 0.25
    for index, time in enumerate(times):
        folder = root / str(index)
        folder.mkdir()
        field = np.zeros((8, 8, 8, 6), dtype="<f8")
        field[..., 1] = coefficient * time * x[:, None, None]
        field[..., 4] = force_coefficient * x[:, None, None]
        field.tofile(folder / "state.bin")
        (folder / "manifest.json").write_text(json.dumps({"index": index}))
        (folder / "probe-snapshot.json").write_text(json.dumps(probe))
        paths.append(folder)
        manifests.append(
            {
                "frames": [
                    {"path": "state.bin", "export": {"time": time, "blocks": [block]}}
                ],
                "parameters": {"widths": [0.5]},
                "profile_sha256": "profile",
            }
        )
    run = {
        "binary_sha256": "binary",
        "profile_manifest": {"parameters": {"viscosity": 0}},
    }
    return paths, manifests, [run] * 5, probe


def test_fieldwise_residual_gap_and_force_identity(tmp_path, monkeypatch):
    baseline = branch(tmp_path, "baseline", 1.0)
    half = branch(tmp_path, "half", 1.1)
    monkeypatch.setattr(
        subject,
        "load_branch",
        lambda paths: baseline[1:] if paths[0] == baseline[0][0] else half[1:],
    )
    result = subject.measure_pair(baseline[0], half[0])
    core = result["regions"]["finest_core_interior"]
    assert core["volume"] == pytest.approx(1)
    assert core["rms"]["residual_five"]["difference"] == pytest.approx(0.1)
    assert core["rms"]["force_curl"]["difference"] == 0
    assert result["force_field_identity"]["difference_rms"] == 0
    changed_force = branch(tmp_path, "force-shift", 1.1, 0.2)
    monkeypatch.setattr(
        subject,
        "load_branch",
        lambda paths: baseline[1:] if paths[0] == baseline[0][0] else changed_force[1:],
    )
    with pytest.raises(ValueError, match="fixed force fields differ"):
        subject.measure_pair(baseline[0], changed_force[0])
