import pytest

from scripts.render_native_gifs import prefix_frames


def fixture():
    record = {
        "parameters": {"end": 0.9, "frame_dt": 0.25},
        "profile_manifest": {"parameters": {"t_star": 1}},
        "status": "running",
        "validated": False,
    }
    history = [{"time": t} for t in (0, 0.25, 0.5 - 2e-15, 0.5, 0.6)]
    headers = [
        {"path": f"plt{i:05d}", "time": r["time"]} for i, r in enumerate(history[:-1])
    ]
    return record, history, headers


def test_preview_preserves_duplicates_and_original_status():
    record, history, headers = fixture()
    selected, expected = prefix_frames(record, history, headers, 0.5)
    assert selected == headers and expected == [0, 0.25, 0.5]
    assert record["status"] == "running" and record["validated"] is False


def test_preview_rejects_missing_zero_and_middle_events():
    record, history, headers = fixture()
    for missing in (headers[1:], [headers[0], *headers[2:]]):
        with pytest.raises(ValueError, match="missing requested"):
            prefix_frames(record, history, missing, 0.5)


@pytest.mark.parametrize("through", [float("nan"), 0.8, 0.45])
def test_preview_rejects_unavailable_or_unscheduled_endpoint(through):
    with pytest.raises(ValueError):
        prefix_frames(*fixture(), through)


def test_preview_rejects_header_history_mismatch():
    record, history, headers = fixture()
    headers[1]["path"] = "plt00000"
    with pytest.raises(ValueError, match="diagnostic step"):
        prefix_frames(record, history, headers, 0.5)


def test_preview_honors_explicit_nonuniform_schedule():
    record, history, headers = fixture()
    record["planned_frames"] = [0, 0.25, 0.5, 0.55, 0.7, 0.9]
    selected, expected = prefix_frames(record, history, headers, 0.5)
    assert len(selected) == 4 and expected == [0, 0.25, 0.5]


def test_complete_render_has_no_arrows_and_keeps_every_frame(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    from matplotlib.axes import Axes
    from PIL import Image

    from scripts import render_native_gifs as renderer

    source = tmp_path / "source"
    source.mkdir()
    (source / "run.json").write_text("{}")
    exporter = tmp_path / "exporter"
    exporter.write_text("synthetic test exporter")
    output = tmp_path / "render"
    record = {
        "status": "completed",
        "validated": True,
        "adapter_sha256": "test-adapter",
        "binary_sha256": "test-binary",
        "checker_sha256": "test-checker",
        "parameters": {"base_n": 64, "widths": [0.5, 0.25, 0.125, 0.0625]},
        "profile_manifest": {"parameters": {"paper_time_cutoff_start": 0.55}},
        "native_frames": [
            {"path": "plt00000", "time": 0, "peak_speed": 0},
            {"path": "plt00001", "time": 0.6, "peak_speed": 1},
        ],
    }
    monkeypatch.setattr(renderer, "snapshot", lambda _: (record, []))

    def export(command, **kwargs):
        destination = Path(
            next(s.removeprefix("output=") for s in command if s.startswith("output="))
        )
        data = np.zeros((3, 256, 256, 6), dtype="<f8")
        if destination.stem == "plt00001":
            data[..., 0] = 1
            data[..., 3] = 2
        data.tofile(destination)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def forbidden_arrows(*args, **kwargs):
        pytest.fail("The magnitude renderer must not draw vector arrows")

    monkeypatch.setattr(renderer.subprocess, "run", export)
    monkeypatch.setattr(Axes, "quiver", forbidden_arrows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_native_gifs",
            "--run",
            str(source),
            "--executable",
            str(exporter),
            "--output",
            str(output),
        ],
    )
    renderer.main()
    manifest = json.loads((output / "render.json").read_text())
    assert manifest["vector_arrows"] is False
    assert manifest["render_scope"] == "completed sequence"
    assert [f["time"] for f in manifest["frames"]] == [0, 0.6]
    assert len(manifest["gifs"]) == 4
    for name, properties in manifest["gifs"].items():
        assert properties["color_max"] == (1 if name.startswith("velocity") else 2)
        with Image.open(output / name) as gif:
            assert gif.n_frames == properties["frames"] == 2
