import json
from copy import deepcopy
from itertools import pairwise

import pytest

from scripts.paper_run import (
    MARKER,
    output_times,
    resource_check,
    thread_environment,
    validate_history,
)


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


def test_prescribed_phase_frames_include_rest_and_endpoint():
    import math

    clock = fixture()[0]["profile_manifest"]["parameters"]
    times = output_times(0.995, 0.025, 0.75, clock)
    assert times[0] == 0 and times[-1] == 0.995 and 0.55 in times
    assert len(times) < 300
    for a, b in pairwise(times):
        assert 1e-11 < b - a <= 0.025 + 1e-14
        if a >= 0.55:
            assert (
                clock["forcing_log_rate_bound"] * math.log((1 - a) / (1 - b))
                <= 0.75 + 1e-11
            )
    legacy = output_times(0.62, 0.025, 0, clock)
    assert len(legacy) == 26 and legacy[-1] == 0.62
    for phase in [-1, 1e-20]:
        with pytest.raises(ValueError):
            output_times(0.62, 0.025, phase, clock)


def test_resource_ceilings_are_scoped_and_stop_without_cleanup(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "scripts.paper_run.shutil.disk_usage",
        lambda _: SimpleNamespace(free=30 * 1024**3),
    )
    monkeypatch.setattr(
        "scripts.paper_run.subprocess.run",
        lambda *a, **kw: SimpleNamespace(stdout="2048\n", returncode=0),
    )
    resource_check(tmp_path, 123, 3, 20)
    with pytest.raises(RuntimeError, match="RSS"):
        resource_check(tmp_path, 123, 1, 20)
    with pytest.raises(RuntimeError, match="disk reserve"):
        resource_check(tmp_path, 123, 3, 40)


def test_activation_roundoff_still_enforces_active_step_caps():
    record, rows = fixture()
    rows[2]["time"] -= 1e-16
    rows[2]["dt"] = rows[2]["time"] - rows[1]["time"]
    rows[3]["dt"] = rows[3]["time"] - rows[2]["time"]
    validate_history(text(rows), record)
    changed = deepcopy(rows)
    changed[-1]["dt"] = 0.025
    changed[-1]["time"] = rows[2]["time"] + 0.025
    with pytest.raises(ValueError, match="phase step|timestep ceiling"):
        validate_history(text(changed), record)
