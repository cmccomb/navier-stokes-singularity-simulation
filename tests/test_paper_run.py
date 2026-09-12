import json
from copy import deepcopy

import pytest

from scripts.paper_run import MARKER, thread_environment, validate_history


def fixture():
    record = {
        "adapter_sha256": "a" * 64,
        "parameters": {"end": 0.5502, "max_dt": 0.00025, "widths": [0.5]},
        "profile_manifest": {
            "parameters": {
                "paper_time_cutoff_start": 0.55,
                "forcing_log_rate_bound": 42.69,
                "forcing_phase_step": 0.0375,
                "t_star": 1,
            }
        },
    }
    base = {
        "adapter_sha256": "a" * 64,
        "case": "paper",
        "levels": 2,
        "volume": 8,
        "energy": 0,
        "peak_speed": 0,
        "l2_error": 0,
        "linf_error": 0,
        "peak_rss_mib": 100,
    }
    times = [0, 0.5, 0.55, 0.5502]
    rows = [
        {**base, "step": i, "time": t, "dt": t - times[i - 1] if i else 0}
        for i, t in enumerate(times)
    ]
    return record, rows


def text(rows):
    return "\n".join(MARKER + json.dumps(row) for row in rows)


def test_paper_history_checks_from_rest_and_phase_clock():
    record, rows = fixture()
    assert validate_history(text(rows), record) == rows
    for index, key, value in [
        (0, "time", 0.1),
        (0, "peak_speed", 1e-30),
        (1, "energy", 1e-30),
        (1, "dt", 0.6),
        (2, "time", 0.5501),
        (2, "step", 4),
        (3, "adapter_sha256", "b" * 64),
        (3, "l2_error", float("nan")),
        (3, "levels", 3),
        (3, "volume", 7.99),
        (3, "case", "shear"),
    ]:
        altered = deepcopy(rows)
        altered[index][key] = value
        with pytest.raises(ValueError):
            validate_history(text(altered), record)
    changed = deepcopy(record)
    changed["profile_manifest"]["parameters"]["forcing_phase_step"] = 0.001
    with pytest.raises(ValueError, match="phase step"):
        validate_history(text(rows), changed)
    changed = deepcopy(record)
    changed["parameters"]["max_dt"] = 0.0001
    with pytest.raises(ValueError, match="timestep ceiling"):
        validate_history(text(rows), changed)
    with pytest.raises(ValueError, match="endpoint"):
        validate_history(text(rows[:-1]), record)


def test_explicit_child_thread_limits():
    assert thread_environment(2, 1) == {
        "OMP_NUM_THREADS": "2",
        "OMP_THREAD_LIMIT": "2",
        "OMP_DYNAMIC": "FALSE",
        "OMP_MAX_ACTIVE_LEVELS": "1",
    }
    for solver, force in [(0, 1), (-1, 1), (2, 0), (1, 2)]:
        with pytest.raises(ValueError, match="force_threads"):
            thread_environment(solver, force)


def test_runtime_threading_must_match_provenance():
    record, rows = fixture()
    record["execution"] = {"threads": 2, "force_threads": 1}
    for row in rows:
        row.update(openmp_max_threads=2, force_threads_limit=1)
    assert validate_history(text(rows), record) == rows
    for key in ("openmp_max_threads", "force_threads_limit"):
        altered = deepcopy(rows)
        altered[1][key] = 4
        with pytest.raises(ValueError, match="threading"):
            validate_history(text(altered), record)
