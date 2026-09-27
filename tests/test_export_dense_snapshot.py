"""Dense native exports must retain their checkpoint continuation lineage."""

import json

import numpy as np
import pytest

from scripts.compare_mixed_native import load_snapshot
from scripts.export_dense_snapshot import diagnostic_rows, selected_frames
from scripts.export_mesh_snapshot import sha


def test_selected_frames_requires_completed_five_point_stencil():
    center = 0.62
    times = [center + i * 0.0005 for i in (-2, -1, 0, 1, 2)]
    probe = {"status": "completed", "validated": True, "planned_times": times,
             "native_frames": [{"step": i + 1, "time": t} for i, t in enumerate(times)],
             "centers": [center], "spacing": 0.0005}
    assert [f["step"] for f in selected_frames(probe, [center])] == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="completed native audit"):
        selected_frames({**probe, "validated": False}, [center])
    with pytest.raises(ValueError, match="invalid dense frame selection"):
        selected_frames({**probe, "native_frames": probe["native_frames"][:-1]}, [center])


def test_dense_diagnostic_steps_are_unique():
    text = 'NS_INCFLO_RESULT {"step": 1, "time": 0.1}\n'
    assert diagnostic_rows(text)[1]["time"] == 0.1
    with pytest.raises(ValueError, match="duplicate"):
        diagnostic_rows(text * 2)


def test_snapshot_loader_rejects_tampered_dense_receipt(tmp_path):
    raw = tmp_path / "plt00001.bin"
    np.zeros((8, 8, 8, 6), dtype="<f8").tofile(raw)
    run_file = tmp_path / "run-snapshot.json"
    run_file.write_text(json.dumps({"binary_sha256": "binary", "profile_manifest": {"sha256": "profile"}}))
    audit = {"step": 1, "time": 0.1}
    probe_file = tmp_path / "probe-snapshot.json"
    probe_file.write_text(json.dumps({"status": "completed", "validated": True,
                                      "parent_record_sha256": sha(run_file), "source_binary_sha256": "binary",
                                      "source_profile_sha256": "profile", "native_frames": [audit]}))
    block = {"level": 0, "index_lo": [0, 0, 0], "shape": [8, 8, 8, 6],
             "spacing": [0.25] * 3, "origin": [-1] * 3, "offset_bytes": 0}
    frame = {"step": 1, "path": raw.name, "sha256": sha(raw), "diagnostic": {"volume": 8, "energy": 0, "peak_speed": 0},
             "native_audit": audit, "export": {"time": 0.1, "blocks": [block]}}
    manifest = {"complete": True, "source_record_sha256": sha(run_file),
                "source_probe_sha256": sha(probe_file), "profile_sha256": "profile", "frames": [frame]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    load_snapshot(tmp_path)
    frame["native_audit"] = {"step": 1, "time": 0.2}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="native frame audit"):
        load_snapshot(tmp_path)
