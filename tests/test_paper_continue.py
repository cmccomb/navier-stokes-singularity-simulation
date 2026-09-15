import copy
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.paper_continue import (
    checkpoint_clock,
    command,
    continuation_times,
    temporal_gate,
    tree,
    validate_segment,
)


def fixture():
    record = {
        "command": [
            "/parent/ns_incflo",
            "/parent/inputs.paper",
            "stop_time=0.995",
            "ns.table_file=/parent/profile.tbl",
            "ns.max_dt=0.00025",
            "amr.n_cell=128 128 128",
            "ns.epsilon_tau_ratio=0",
        ],
        "execution": {"threads": 4, "force_threads": 4},
        "profile_manifest": {
            "parameters": {
                "t_star": 1.0,
                "paper_time_cutoff_start": 0.55,
                "forcing_log_rate_bound": 42.693450360193765,
                "forcing_phase_step": 0.0375,
            }
        },
    }
    initial = {
        "step": 5510,
        "time": 0.995,
        "dt": 1e-6,
        "adapter_sha256": "a" * 64,
        "case": "paper",
        "levels": 5,
        "stored_cells": 10485760,
        "active_cells": 9437184,
        "volume": 8,
        "peak_speed": 7.4,
        "energy": 0.04,
        "l2_error": 0.002,
        "linf_error": 0.1,
        "peak_rss_mib": 9700,
        "openmp_max_threads": 4,
        "force_threads_limit": 4,
    }
    return record, initial


def test_append_only_phase_clock_and_limits():
    record, initial = fixture()
    clock = record["profile_manifest"]["parameters"]
    times = continuation_times(initial["time"], 0.9975, 0.025, 0.75, clock)
    assert times[0] == 0.995 and times[-1] == 0.9975 and len(times) == 41
    for a, b in zip(times, times[1:]):
        assert 0 < b - a <= 0.025
        assert (
            clock["forcing_log_rate_bound"] * math.log((1 - a) / (1 - b)) < 0.75 + 1e-10
        )
    for start, end, phase in [
        (0.995, 1, 0.75),
        (0.995, 0.99, 0.75),
        (0.995, 0.9975, 0),
        (0.995, 0.9975, float("nan")),
        (0.995, 0.9975, 1e-30),
    ]:
        with pytest.raises(ValueError):
            continuation_times(start, end, 0.025, phase, clock)


def test_command_changes_only_integration_not_force_or_mesh():
    record, _ = fixture()
    record["command"].extend(["amr.plot_per_exact=0.025", "amr.plot_per_exact=-1"])
    before = copy.deepcopy(record)
    argv = command(
        record,
        Path("/new/bundle"),
        Path("/new/chk05510"),
        0.9951,
        [0.995, 0.9951],
        2e-6,
    )
    params = dict(s.split("=", 1) for s in argv[2:])
    assert params["amr.n_cell"] == "128 128 128"
    assert params["ns.epsilon_tau_ratio"] == "0"
    assert float(params["ns.max_dt"]) == 2e-6
    assert params["ns.table_file"] == "/new/bundle/profile.tbl"
    assert not any("phase_step" in s for s in argv)
    assert record == before
    assert len(params) == len(argv) - 2
    argv = command(
        record, Path("/b"), Path("/chk"), 0.995, [0.995], 0.00025, readback=True
    )
    params = dict(s.split("=", 1) for s in argv[2:])
    assert params["max_step"] == "0" and params["amr.check_int"] == "-1"
    assert params["amr.plotfile_on_restart"] == "0"
    record["command"].append("ns.max_dt=0.1")
    with pytest.raises(ValueError, match="duplicate"):
        command(record, Path("/b"), Path("/chk"), 0.995, [0.995], 0.00025)


def test_segment_checks_parent_boundary_and_finer_ceiling():
    record, initial = fixture()
    row = {
        **initial,
        "step": initial["step"] + 1,
        "dt": 2e-6,
        "time": initial["time"] + 2e-6,
    }
    text = "NS_INCFLO_RESULT " + json.dumps(row)
    assert validate_segment(text, record, initial, row["time"], 2e-6) == [row]
    for key, value in [
        ("step", 1),
        ("dt", 1e-5),
        ("levels", 6),
        ("stored_cells", 1),
        ("active_cells", 1),
        ("volume", 7),
        ("peak_speed", float("nan")),
        ("adapter_sha256", "b"),
        ("force_threads_limit", 8),
        ("case", "shear"),
    ]:
        changed = {**row, key: value}
        with pytest.raises(ValueError):
            validate_segment(
                "NS_INCFLO_RESULT " + json.dumps(changed),
                record,
                initial,
                row["time"],
                2e-6,
            )
    with pytest.raises(ValueError, match="ceiling"):
        validate_segment(text, record, initial, row["time"], 1e-6)
    with pytest.raises(ValueError, match="endpoint"):
        validate_segment(text, record, initial, 0.9975, 2e-6)
    with pytest.raises(ValueError, match="no diagnostic"):
        validate_segment("", record, initial, 0.9975, 2e-6)


def test_temporal_gate_is_same_mesh_and_explicit_sensitivity():
    row = {
        "compared": True,
        "report_difference": True,
        "reference_resolution_ratio": 1,
        "velocity_l2_difference": 1e-5,
        "reference_velocity_l2": 0.1,
        "velocity_linf_difference": 0.001,
        "peak_speed": 7.4,
    }
    assert temporal_gate(row, 0.001, 0.01)["passed"]
    assert not temporal_gate({**row, "velocity_l2_difference": 0.001}, 0.001, 0.01)[
        "passed"
    ]
    assert not temporal_gate({**row, "velocity_linf_difference": 0.1}, 0.001, 0.01)[
        "passed"
    ]
    with pytest.raises(ValueError, match="mesh"):
        temporal_gate({**row, "reference_resolution_ratio": 2}, 0.001, 0.01)
    with pytest.raises(ValueError, match="invalid"):
        temporal_gate({**row, "velocity_l2_difference": float("nan")}, 0.001, 0.01)
    with pytest.raises(ValueError, match="normalization"):
        temporal_gate({**row, "reference_velocity_l2": 0}, 0.001, 0.01)


@pytest.mark.parametrize("status", ["running", "failed", "stopped"])
def test_unready_parent_never_launches_or_changes_parent(tmp_path, status):
    parent = tmp_path / "parent"
    parent.mkdir()
    original = json.dumps({"status": status, "validated": False})
    (parent / "run.json").write_text(original)
    output = tmp_path / "new"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.paper_continue",
            "--parent",
            str(parent),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    receipt = json.loads((output / "continuation.json").read_text())
    assert receipt["status"] == "failed" and receipt["stages"] == {}
    assert (parent / "run.json").read_text() == original
    assert (output / "runner/scripts/paper_continue.py").is_file()


def test_checkpoint_clock_and_symlink_rejection(tmp_path):
    p = tmp_path / "chk05510"
    p.mkdir()
    (p / "Header").write_text("Checkpoint version: 1\n4\n5510\n0.995\n")
    for i in range(4):
        (p / f"field{i}").write_bytes(b"checkpoint")
    assert checkpoint_clock(p) == (5510, 0.995, 5)
    assert len(tree(p)) == 5
    (p / "link").symlink_to(p / "Header")
    with pytest.raises(ValueError, match="non-regular"):
        tree(p)
