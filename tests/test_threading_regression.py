from copy import deepcopy

import pytest

from scripts.threading_regression import compare_records


def records():
    single = {
        "status": "completed",
        "validated": True,
        "execution": {"threads": 1, "force_threads": 1},
        "binary_sha256": "binary",
        "adapter_sha256": "source",
        "parameters": {"end": 0.56},
        "profile_manifest": {"sha256": "table"},
        "history": [
            {"step": 0, "time": 0, "dt": 0},
            {"step": 1, "time": 0.56, "dt": 0.56},
        ],
        "native_frames": [
            {"time": 0, "levels": 2, "stored_cells": 8000},
            {"time": 0.56, "levels": 2, "stored_cells": 8000},
        ],
    }
    threaded = deepcopy(single)
    threaded["execution"] = {"threads": 2, "force_threads": 2}
    return single, threaded


def test_threading_comparison_is_like_for_like():
    single, threaded = records()
    assert len(compare_records(single, threaded)) == 2
    for key, value in [
        ("status", "running"),
        ("validated", False),
        ("binary_sha256", "another binary"),
        ("adapter_sha256", "another source"),
        ("parameters", {"end": 0.6}),
        ("profile_manifest", {"sha256": "another force"}),
        ("native_frames", []),
        ("execution", {"threads": 1, "force_threads": 1}),
    ]:
        changed = deepcopy(threaded)
        changed[key] = value
        with pytest.raises(ValueError):
            compare_records(single, changed)
    for section in ("history", "native_frames"):
        changed = deepcopy(threaded)
        changed[section][-1]["time"] += 0.001
        with pytest.raises(ValueError):
            compare_records(single, changed)
