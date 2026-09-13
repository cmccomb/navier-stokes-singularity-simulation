import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import render_native_site as renderer


def test_preparation_creates_completion_record_parent(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.json").write_text("{}")
    (source / "run.log").write_text("test fixture")
    exporter = tmp_path / "exporter"
    exporter.write_text("test exporter")
    record = {
        "status": "completed",
        "validated": True,
        "updated_at": "2026-09-13T04:55:26+00:00",
        "wall_seconds": 10,
        "adapter_sha256": "test-adapter",
        "binary_sha256": "test-binary",
        "parameters": {"base_n": 64, "widths": [0.5, 0.25, 0.125, 0.0625], "end": 0.6},
        "profile_manifest": {"parameters": {"paper_time_cutoff_start": 0.55}},
        "planned_frames": [0, 0.6],
        "native_frames": [
            {"path": "plt00000", "time": 0, "peak_speed": 0},
            {"path": "plt00001", "time": 0.6, "peak_speed": 1},
        ],
    }
    final = {"stored_cells": 1310720, "active_cells": 1179648}
    monkeypatch.setattr(renderer, "snapshot", lambda _: (record, [final]))

    def export(command, **kwargs):
        path = Path(
            next(s.removeprefix("output=") for s in command if s.startswith("output="))
        )
        active = path.stem == "plt00001"
        data = np.zeros((4, 256, 256, 6), dtype="<f8")
        if active:
            data[..., 0] = 1
            data[..., 3] = 2
        data.tofile(path)
        row = {
            "peak_xy": True,
            "time": 0.6 if active else 0,
            "peak_speed": 1 if active else 0,
            "peak_position": [0, 0, 0.125 if active else 0],
        }
        return SimpleNamespace(stdout="NS_SLICE_RESULT " + json.dumps(row), stderr="")

    def render(frames, field, stem, resolution, **kwargs):
        assert len(frames) == 2 and frames[-1]["time"] == 0.6
        assert frames[-1]["plane_coordinates"][1] == (0.125 if frames.moving else 0)
        stem.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("gif", "mp4", "png"):
            stem.with_suffix("." + ext).write_bytes(b"fixture")
        return {"color_max": kwargs["color_max"]}

    monkeypatch.setattr(renderer.subprocess, "run", export)
    monkeypatch.setattr(renderer, "render_pair", render)
    output = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_native_site",
            "--run",
            str(source),
            "--executable",
            str(exporter),
            "--output",
            str(output),
        ],
    )
    renderer.main()
    result = json.loads((output / "site/data/best.json").read_text())
    assert result["completed_at"] == record["updated_at"]
    assert result["saved_frames"] == 2 and len(result["media"]) == 4
    assert not result["vector_arrows"] and not result["production_accuracy_certified"]
