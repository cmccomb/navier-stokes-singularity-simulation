"""Final native export selection must remain receipt-bound."""

import json

import pytest

from scripts.export_final_continuation_snapshot import final_frame


def receipt(tmp_path):
    source = tmp_path / "continuation"
    plot = source / "extension/plt00042"
    plot.mkdir(parents=True)
    row = {"step": 42, "time": 0.9975}
    frame = {
        "step": 42,
        "time": 0.9975,
        "path": str(plot),
        "force_linf_error": 0,
    }
    value = {
        "status": "completed",
        "validated": True,
        "end": 0.9975,
        "stages": {
            "extension": {
                "status": "completed",
                "validated": True,
                "history": [row],
                "native_frames": [frame],
            }
        },
    }
    (source / "continuation.json").write_text(json.dumps(value))
    return source


def test_selects_only_audited_extension_endpoint(tmp_path):
    source = receipt(tmp_path)
    _, row, frame = final_frame(source)
    assert row["step"] == 42 and frame["time"] == 0.9975


@pytest.mark.parametrize("change", ["running", "time", "force", "path"])
def test_rejects_incomplete_or_changed_endpoint(tmp_path, change):
    source = receipt(tmp_path)
    path = source / "continuation.json"
    value = json.loads(path.read_text())
    frame = value["stages"]["extension"]["native_frames"][0]
    if change == "running":
        value["status"] = "running"
    elif change == "time":
        frame["time"] = 0.9974
    elif change == "force":
        frame["force_linf_error"] = 1e-6
    else:
        outside = tmp_path / "outside/plt00042"
        outside.mkdir(parents=True)
        frame["path"] = str(outside)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        final_frame(source)
