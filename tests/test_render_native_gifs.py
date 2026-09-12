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
